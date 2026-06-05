"""
Standalone Knowledge Graph Traversal Engine.

Takes predicted_classes from predictions.csv one-by-one, traverses the KG
per class per UID (no shared state between classes), and writes individual
triplet JSONs per UID into output/uid_traversals/.

Key Design -

1. ACYCLIC  — Adjacency index is SUBJECT-ONLY (forward edges only).
              visited set prevents any node from being expanded twice,
              which breaks any forward cycle even if one exists in the raw KG.

2. ISOLATED — Each (uid, class) pair runs its own independent DFS with its
              own visited set. One class cannot "infect" another class's graph.

3. ORDERED  — For each UID the classes are processed in the original CSV order.
              Within each class, triplets are ranked by clinical relation priority.

4. NO DUPLICATES — Triplets are deduplicated within each class traversal AND
                   across the merged final triplet list for the UID.

5. LEAF-NODE TRAVERSAL — No fixed depth limit. Each seed entity is expanded
              forward through the KG until leaf nodes (entities with no outgoing
              edges) are reached. ALL triplets along every root-to-leaf path are
              collected, giving full coverage of each reachable sub-graph.

Adjacency Format
----------------
  adj["entity"] = [(relation, object), (relation, object), ...]
  Subjects are indexed in lowercase for case-insensitive lookup.
  Objects are NEVER keys → traversal is strictly forward (subject → object).

Output layout
    <uid>_traversal.json   ← one file per UID

Each file contains:
  uid               : str
  gt_labels         : str  (ground truth, for evaluation)
  predicted_classes : [str, ...]
  kg_classes        : [str, ...]  (after conflict resolution)
  conflict_resolution : str
  is_uncertain      : bool
  per_class_traversal : {
      "<ClassName>": {
          "seeds_used"    : [str, ...]   seeds found in the KG
          "seeds_missing" : [str, ...]   seeds not found in KG
          "triplets"      : [triplet, ...]
          "triplet_count" : int
          "leaf_paths"    : [str, ...]   human-readable root→leaf path per path
      }
  }
  merged_triplets : [triplet, ...]   deduplicated union, ranked
  merged_count    : int
  leaf_traversal  : true
  acyclic         : true

"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RELATION_PRIORITY: Dict[str, int] = {
    "strongly_suggests": 6,
    "confirmed_by":      5,
    "suggests":          4,
    "is_finding_of":     3,
    "is_symptom_of":     3,
    "requires":          2,
    "located_in":        2,
    "weakly_suggests":   1,
    "absence_weakens":   1,
    "contradicts":       4
}
HUB_ENTITIES: Set[str] = {
    "chest radiograph", "lung fields", "pulmonary vasculature",
    "cardiothoracic ratio", "thoracic cavity", "thoracic spine",
    "pleural space", "pericardial space", "lung", "heart",
    "mediastinum", "chest", "thorax", "lungs", "chest x-ray",
    "radiograph", "disease", "vasculature",
}

# Noisy / data-leakage subjects to drop from output
KG_NOISE_SUBJECTS: Set[str] = {
    "dataset cases", "dataset", "study cases", "cases",
    "training cases", "test cases", "sample cases",
}

NORMAL_TRIPLETS: List[dict] = [
    {"s": "normal", "p": "is_finding_of", "o": "clear lung fields"},
    {"s": "normal", "p": "is_finding_of", "o": "normal cardiac size"},
    {"s": "normal", "p": "is_finding_of", "o": "normal mediastinal contours"},
    {"s": "normal", "p": "is_finding_of", "o": "sharp and well-defined hemidiaphragms"},
    {"s": "normal", "p": "is_finding_of", "o": "midline trachea"},
    {"s": "normal", "p": "is_finding_of", "o": "no acute fracture"},
    {"s": "normal", "p": "is_finding_of", "o": "no pleural effusion"},
    {"s": "normal", "p": "is_finding_of", "o": "no pneumothorax"},
    {"s": "normal", "p": "is_finding_of", "o": "no consolidation"},
    {"s": "normal", "p": "contradicts", "o": "Cardiomegaly"},
    {"s": "normal", "p": "contradicts", "o": "Pleural Effusion"},
    {"s": "normal", "p": "contradicts", "o": "Pneumothorax"},
    {"s": "normal", "p": "contradicts", "o": "Consolidation"},
    {"s": "normal", "p": "contradicts", "o": "Edema"},
    {"s": "normal", "p": "contradicts", "o": "Mass"},
    {"s": "normal", "p": "contradicts", "o": "Emphysema"},
    {"s": "clear lung fields",           "p": "contradicts",   "o": "acute cardiopulmonary disease"},
    {"s": "clear lung fields",           "p": "contradicts",   "o": "pneumonia"},
    {"s": "clear lung fields",           "p": "contradicts",   "o": "congestive heart failure"},
    {"s": "clear lung fields",           "p": "contradicts",   "o": "pneumothorax"},
    {"s": "clear lung fields",           "p": "is_finding_of", "o": "well-defined pulmonary vasculature"},
    {"s": "clear lung fields",           "p": "is_finding_of", "o": "normal cardiothoracic ratio"},
    {"s": "normal cardiac silhouette",   "p": "contradicts",   "o": "cardiomegaly"},
    {"s": "normal cardiac silhouette",   "p": "contradicts",   "o": "pericardial effusion"},
    {"s": "normal cardiac silhouette",   "p": "is_finding_of", "o": "normal cardiothoracic ratio"},
    {"s": "normal cardiothoracic ratio", "p": "contradicts",   "o": "cardiomegaly"},
    {"s": "normal cardiothoracic ratio", "p": "contradicts",   "o": "congestive heart failure"},
    {"s": "normal mediastinal contours", "p": "contradicts",   "o": "mediastinal widening"},
    {"s": "normal mediastinal contours", "p": "contradicts",   "o": "aortic aneurysm"},
    {"s": "normal examination",          "p": "contradicts",   "o": "acute cardiopulmonary disease"},
    {"s": "normal examination",          "p": "contradicts",   "o": "active pulmonary disease"},
    {"s": "normal examination",          "p": "is_finding_of", "o": "clear lung fields"},
    {"s": "normal examination",          "p": "is_finding_of", "o": "normal cardiac silhouette"},
    {"s": "Normal chest radiograph",     "p": "contradicts",   "o": "cardiomegaly"},
    {"s": "Normal chest radiograph",     "p": "contradicts",   "o": "calcinosis"},
    {"s": "Normal chest radiograph",     "p": "contradicts",   "o": "pulmonary edema"},
    {"s": "Normal chest radiograph",     "p": "contradicts",   "o": "pleural effusion"},
    {"s": "Normal chest radiograph",     "p": "contradicts",   "o": "pneumothorax"},
    {"s": "Normal chest radiograph",     "p": "contradicts",   "o": "lobar consolidation"},
    {"s": "Normal chest radiograph",     "p": "absence_weakens", "o": "acute cardiopulmonary disease"},
    {"s": "Normal chest radiograph",     "p": "weakly_suggests", "o": "normal cardiopulmonary status"},
    {"s": "Normal chest radiograph",     "p": "confirmed_by",  "o": "radiographic interpretation"},
    {"s": "Normal chest radiograph",     "p": "requires",      "o": "adequate inspiratory effort"},
    {"s": "Normal cardiac shadow",       "p": "contradicts",   "o": "cardiomegaly"},
    {"s": "Normal cardiac shadow",       "p": "weakly_suggests", "o": "normal cardiac size"}
 ]

CLASS_TO_SEEDS: Dict[str, List[str]] = {
    "Normal": [
        "normal" ,"clear lung fields", "Normal chest radiograph",
        "normal cardiac silhouette", "normal cardiothoracic ratio",
        "normal examination", "Normal cardiac shadow",
    ],
    "Degenerative Change": [
        "degenerative changes", "degenerative spondylosis",
        "osteophyte formation", "osteophyte",
        "age-related wear and tear", "severe degenerative disease",
    ],
    "Lesion": [
        "parenchymal lesion", "pulmonary nodule", "pulmonary mass",
        "focal opacity", "solitary pulmonary nodule",
    ],
    "Hyperinflation": [
        "hyperinflation", "emphysema", "barrel-chest appearance",
        "hyperlucency", "flattened hemidiaphragms", "hyperlucent lung fields",
    ],
    "Calcified Granuloma": [
        "calcified granuloma", "granuloma", "calcification",
        "popcorn calcification", "target calcification",
        "calcification of costal cartilages",
    ],
    "Cardiomegaly": [
        "cardiomegaly", "enlarged cardiac silhouette",
        "cardiac shadow enlargement", "increased cardiothoracic ratio",
        "chronic congestive heart failure", "left ventricular enlargement",
    ],
    "Volume Loss": [
        "volume loss", "atelectasis", "linear atelectasis",
        "mediastinal shift", "diaphragm",
    ],
    "Calcinosis": [
        "calcinosis", "calcinosis cutis", "soft tissue calcification",
        "cutaneous calcified nodules", "vascular calcifications",
    ],
    "Airspace Disease": [
        "airspace disease", "patchy bilateral airspace opacities",
        "alveolar filling process", "consolidation",
        "focal consolidation", "pulmonary consolidation", "air bronchograms",
    ],
    "Fibrosis": [
        "fibrosis", "pulmonary fibrosis", "reticular opacities",
        "honeycombing pattern", "traction bronchiectasis",
        "architectural distortion",
    ],
    "Increased Lung Markings": [
        "diffusely increased bronchovascular markings",
        "increased lung markings", "bronchovascular markings",
        "peribronchial cuffing", "peribronchial thickening", "vascular congestion",
    ],
    "Pleural Effusion": [
        "pleural effusion", "small effusions", "large effusions",
        "blunting of the costophrenic angle", "meniscus sign",
        "free-flowing pleural effusion",
    ],
    "Emphysema": [
        "emphysema", "bullous disease", "bullae",
        "parenchymal destruction", "subcutaneous emphysema",
        "chronic obstructive pulmonary disease",
    ],
    "Nodule": [
        "pulmonary nodule", "solitary pulmonary nodule", "nodule",
        "centrilobular nodules", "pulmonary mass",
    ],
    "Edema": [
        "pulmonary edema", "interstitial edema", "kerley b lines",
        "bat-wing pattern", "perihilar infiltrates",
        "vascular cephalization", "cardiogenic pulmonary edema",
    ],
    "Scoliosis": [
        "scoliosis", "vertebral rotation", "thoracic spine",
    ],
    "Fractures": [
        "fractures", "rib fracture", "vertebral compression fracture",
        "acute fracture", "cortical disruption", "callus formation",
    ],
    "Hernia": [
        "hernia", "hiatal hernia", "diaphragmatic hernia",
        "morgagni hernia", "bochdalek hernia", "diaphragm",
    ],
    "Pleural Thickening": [
        "pleural thickening", "pleural plaques", "pleuritis",
        "chronic pleuritis", "tuberculosis",
    ],
    "Osteophyte": [
        "osteophyte", "osteophyte formation", "bony spurs",
        "degenerative spondylosis", "disc space narrowing", "endplate sclerosis",
    ],
    "Interstitial Lung Disease": [
        "interstitial lung disease", "reticulonodular opacities",
        "ground-glass opacification", "peribronchovascular thickening",
        "irregular septal lines", "interlobular septal thickening",
    ],
    "Consolidation": [
        "consolidation", "focal consolidation", "pulmonary consolidation",
        "air bronchograms", "silhouette sign", "homogeneous opacification",
    ],
    "Cardiac Shadow (abnormal)": [
        "abnormal cardiac silhouette", "cardiac silhouette",
        "cardiac shadow enlargement", "pericardial effusion",
        "right ventricular enlargement", "left heart border",
    ],
    "Thickening": [
        "diffuse bronchial wall thickening", "thickening of bronchial walls",
        "bronchial walls", "peribronchial thickening",
    ],
    "Kyphosis": [
        "kyphosis", "vertebral wedging",
        "anterior vertebral height", "thoracic spine",
    ],
    "Pneumothorax": [
        "pneumothorax", "tension pneumothorax", "visceral pleural line",
        "collapsed lung", "mediastinal shift", "lucency in the pleural space",
    ],
    "Mass": [
        "pulmonary mass", "mass", "hilar lymphadenopathy",
        "bronchogenic carcinoma", "malignancy",
        "right cardiophrenic angle mass",
    ],
    "Pulmonary Artery Enlargement": [
        "pulmonary artery enlargement",
        "dilated central pulmonary arteries",
        "prominent main pulmonary artery segment",
        "pulmonary artery segment",
        "caliber discrepancy between hilar and peripheral vessels",
    ],
    "Pulmonary Fibrosis": [
        "pulmonary fibrosis", "reticular opacities", "honeycombing",
        "honeycombing pattern", "traction bronchiectasis", "lower lobe",
    ],
    "Effusion": [
        "pleural effusion", "pericardial effusion", "fluid collection",
        "small effusions", "large effusions",
    ],
    "Bronchiectasis": [
        "bronchiectasis", "dilated bronchi", "cystic bronchiectasis",
        "tram-track sign", "parallel linear opacities", "thickened walls",
    ],
    "Bullous Disease": [
        "bullous disease", "bullae", "emphysema", "parenchymal destruction",
    ],
    "Rib Fracture": [
        "rib fracture", "fractures", "acute fracture",
        "cortical disruption", "soft tissue swelling",
    ],
    "Subcutaneous Emphysema": [
        "subcutaneous emphysema", "emphysema",
    ],
}


class KnowledgeGraph:
    """
    In-memory directed KG loaded from a JSONL file.

    Adjacency format:
        adj["entity_lower"] = [(relation, object), ...]

    Objects are NEVER used as keys → traversal is strictly forward
    (subject → object), enforcing a DAG traversal order.

    Cycle prevention proof:
      Suppose a forward cycle exists: A→B→C→A
      Step 1: expand A  → visit B  (A added to visited)
      Step 2: expand B  → visit C  (B added to visited)
      Step 3: expand C  → would visit A, but A is in visited → SKIPPED
      Result: the cycle is broken without any depth cap.
    """

    def __init__(self, path: Path, adj_path: Optional[Path] = None) -> None:
        self.path = path
        self.adj_path = adj_path
        # adj[subject_lower] = [(relation, object_original_case), ...]
        self._adj: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        # Store original-case subject for reconstructing full triplets
        self._subject_orig: Dict[str, str] = {}
        self._total: int = 0
        # Prefer pre-built adjacency list if available, else parse JSONL
        if adj_path is not None and adj_path.exists():
            self._load_from_adj(adj_path)
        else:
            self._load()
            # Persist the newly built adj list for future runs
            _adj_out = adj_path if adj_path is not None else path.parent / "graph_adj.json"
            self.save_adj(_adj_out)

    # ------------------------------------------------------------------
    def _load(self) -> None:
        seen: Set[Tuple[str, str, str]] = set()
        with open(self.path, "r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    log.warning("KG line %d: invalid JSON — skipped", lineno)
                    continue

                # Normalise whitespace; keep original casing for output
                s = raw["s"].strip()
                p = raw["p"].strip()
                o = raw["o"].strip()

                key: Tuple[str, str, str] = (s.lower(), p.lower(), o.lower())
                if key in seen:
                    continue
                seen.add(key)

                sl = s.lower()
                # Adjacency: subject_lower → (relation, object_orig)
                self._adj[sl].append((p, o))
                # Remember original-case subject for triplet reconstruction
                if sl not in self._subject_orig:
                    self._subject_orig[sl] = s
                self._total += 1

        log.info(
            "KG loaded: %d unique triplets | %d unique subjects | file=%s",
            self._total, len(self._adj), self.path.name,
        )

    def _load_from_adj(self, adj_path: Path) -> None:
        """
        Fast-load the adjacency list from a pre-built graph_adj.json.

        Expected format (written by step2_extract_relations.py or save_adj):
            {
                "subject_lower": [["relation", "object"], ...],
                ...
            }
        """
        with open(adj_path, "r", encoding="utf-8") as fh:
            raw: Dict[str, list] = json.load(fh)

        for subject_lower, edges in raw.items():
            for edge in edges:
                relation, obj = edge[0], edge[1]
                self._adj[subject_lower].append((relation, obj))
                self._total += 1
            if subject_lower not in self._subject_orig:
                # Best-effort original-case recovery (capitalise first letter)
                self._subject_orig[subject_lower] = subject_lower

        log.info(
            "KG adj loaded (fast-path): %d unique triplets | %d unique subjects | file=%s",
            self._total, len(self._adj), adj_path.name,
        )

    def save_adj(self, adj_path: Path) -> None:
        """
        Persist the current in-memory adjacency list to *adj_path* as JSON.

        Format written:
            {
                "subject_lower": [["relation", "object"], ...],
                ...
            }
        """
        adj_path.parent.mkdir(parents=True, exist_ok=True)
        serialisable = {
            subject: [[rel, obj] for rel, obj in edges]
            for subject, edges in self._adj.items()
        }
        with open(adj_path, "w", encoding="utf-8") as fh:
            json.dump(serialisable, fh, indent=2, ensure_ascii=False)
        log.info(
            "KG adj saved: %d subjects → %s",
            len(serialisable), adj_path,
        )

    def outgoing(self, subject: str) -> List[Tuple[str, str]]:
        """Return [(relation, object), ...] for subject (forward edges only)."""
        return self._adj.get(subject.strip().lower(), [])

    def has_subject(self, subject: str) -> bool:
        """True if the KG contains at least one triplet with this subject."""
        return subject.strip().lower() in self._adj

    def subject_orig(self, subject_lower: str) -> str:
        """Return original-case subject string, falling back to the input."""
        return self._subject_orig.get(subject_lower, subject_lower)

    def subject_count(self) -> int:
        return len(self._adj)

    @property
    def adjacency_list(self) -> Dict[str, List[Tuple[str, str]]]:
        """
        Expose the full adjacency list in the format:
            { "entity": [(relation, object), ...], ... }
        """
        return dict(self._adj)

    def __len__(self) -> int:
        return self._total


def _dfs_to_leaves(
    kg: KnowledgeGraph,
    seeds: List[str],
) -> Tuple[List[dict], List[str], List[str], List[List[str]]]:
    """
    Full leaf-node DFS from each seed entity.

    Strategy
    --------
    For each seed, perform a depth-first traversal following forward edges
    (subject → object) until nodes with NO outgoing edges (leaf nodes) are
    reached.  Every triplet encountered along every root-to-leaf path is
    collected.

    A global visited set prevents any entity from being expanded more than
    once, which:
      (a) breaks forward cycles in the raw KG, and
      (b) avoids exponential blow-up on diamond-shaped sub-graphs.

    HUB_ENTITIES are collected as triplet objects but are NOT expanded
    (their outgoing sub-graphs are too broad and would pull in the whole KG).

    Returns
    -------
    triplets      : all triplets along all root-to-leaf paths (deduped, ranked)
    seeds_used    : seeds that had at least one outgoing edge in the KG
    seeds_missing : seeds with no outgoing edge in the KG
    leaf_paths    : list of edge-string lists — one list per complete path
    """
    # Global visited set — each entity is expanded at most once per class
    visited: Set[str] = set()
    seen_keys: Set[Tuple[str, str, str]] = set()
    triplets: List[dict] = []

    seeds_used: List[str] = []
    seeds_missing: List[str] = []
    all_leaf_paths: List[List[str]] = []

    # Seed classification and initial stack seeding
    stack: List[Tuple[str, List[str]]] = []   # (entity_lower, path_so_far)
    for seed in seeds:
        sl = seed.strip().lower()
        if not sl:
            continue
        if kg.has_subject(sl):
            seeds_used.append(seed)
        else:
            seeds_missing.append(seed)
        if sl not in visited:
            visited.add(sl)
            stack.append((sl, []))

    # Iterative DFS — avoids Python recursion limits on deep graphs
    while stack:
        current, path_so_far = stack.pop()
        edges = kg.outgoing(current)          # [(relation, object), ...]

        if not edges:
            # ── LEAF NODE ── record the complete root→leaf path
            if path_so_far:
                all_leaf_paths.append(path_so_far)
            continue

        current_orig = kg.subject_orig(current)
        expanded_any = False

        for relation, obj in edges:
            # Always collect this triplet (every edge on the way to a leaf)
            tk: Tuple[str, str, str] = (current, relation.lower(), obj.lower())
            if tk not in seen_keys:
                seen_keys.add(tk)
                triplets.append({"s": current_orig, "p": relation, "o": obj})

            obj_lower = obj.strip().lower()

            # Hub guard: collect the edge but do NOT expand hub entities
            if obj_lower in HUB_ENTITIES:
                continue

            if obj_lower in visited:
                continue   # cycle-breaker / already-expanded guard

            visited.add(obj_lower)
            edge_str = f"{current_orig}  --{relation}-->  {obj}"
            stack.append((obj_lower, path_so_far + [edge_str]))
            expanded_any = True
        if not expanded_any and path_so_far:
            all_leaf_paths.append(path_so_far)

    # Rank by relation priority (most clinically relevant first)
    triplets.sort(
        key=lambda t: RELATION_PRIORITY.get(t["p"], 0),
        reverse=True,
    )
    return triplets, seeds_used, seeds_missing, all_leaf_paths


def _filter_noise(triplets: List[dict]) -> List[dict]:
    """Drop triplets with blank s/o fields or known noisy subjects."""
    return [
        t for t in triplets
        if t.get("s", "").strip()
        and t.get("o", "").strip()
        and t["s"].strip().lower() not in KG_NOISE_SUBJECTS
    ]


def _build_leaf_path_strings(leaf_paths: List[List[str]]) -> List[str]:
    """
    Convert raw leaf_paths (list-of-lists of edge strings) into a flat list
    of complete root→leaf path strings for human-readable output.

    Example output element:
        "cardiomegaly  --strongly_suggests-->  enlarged cardiac silhouette  |  \
enlarged cardiac silhouette  --has_finding-->  left ventricular hypertrophy"
    """
    return ["  |  ".join(path) for path in leaf_paths if path]

def traverse_class(
    kg: KnowledgeGraph,
    class_name: str,
    top_k: int,
) -> dict:
    """
    Run one independent leaf-node DFS for a single predicted class.

    Traverses all paths from the class seed entities forward through the KG
    until leaf nodes (no outgoing edges) are reached.  All triplets on every
    root-to-leaf path are collected.

    Never shares visited state with any other class — full isolation.

    Returns a dict with:
      seeds_used    : seeds that had outgoing edges in the KG
      seeds_missing : seeds with no outgoing edges
      triplets      : all collected triplets (deduped, priority-ranked, capped at top_k)
      triplet_count : len(triplets)
      leaf_paths    : human-readable complete root→leaf path strings
    """
    if class_name == "Normal":
        trips = list(NORMAL_TRIPLETS)
        capped = trips[:top_k]
        leaf_paths = [f"{t['s']}  --{t['p']}-->  {t['o']}" for t in capped]
        return {
            "seeds_used":    ["normal examination", "clear lung fields",
                              "normal cardiac silhouette", "normal cardiothoracic ratio",
                              "Normal chest radiograph", "Normal cardiac shadow"],
            "seeds_missing": [],
            "triplets":      capped,
            "triplet_count": len(capped),
            "leaf_paths":    leaf_paths,
        }

    seeds = CLASS_TO_SEEDS.get(class_name)
    if not seeds:
        log.warning("  No seeds defined for class '%s' — skipping", class_name)
        return {
            "seeds_used":    [],
            "seeds_missing": [],
            "triplets":      [],
            "triplet_count": 0,
            "leaf_paths":    [],
        }

    raw_triplets, seeds_used, seeds_missing, leaf_paths = _dfs_to_leaves(kg, seeds)
    clean = _filter_noise(raw_triplets)[:top_k]

    return {
        "seeds_used":    seeds_used,
        "seeds_missing": seeds_missing,
        "triplets":      clean,
        "triplet_count": len(clean),
        "leaf_paths":    _build_leaf_path_strings(leaf_paths),
    }



def traverse_uid(
    kg: KnowledgeGraph,
    uid: str,
    kg_classes: List[str],
    predicted_classes: List[str],
    gt_labels: str,
    conflict_resolution: str,
    is_uncertain: bool,
    uncertainty_reason: str,
    top_k: int,
) -> dict:
    """
    Traverse the KG for each class of this UID, one class at a time.

    Each class gets its own isolated leaf-node DFS (separate visited set) so
    that the traversal for 'Lesion' is never contaminated by visited nodes
    from 'Cardiomegaly', etc.

    The final merged_triplets is the deduplicated union ranked globally.
    """
    per_class_traversal: Dict[str, dict] = {}

    # Process each class independently, in order
    for cls in kg_classes:
        log.debug("  uid=%-8s  class=%s", uid, cls)
        per_class_traversal[cls] = traverse_class(kg, cls, top_k)

    # ---- Build deduplicated merged list across all classes ----
    seen_keys: Set[Tuple[str, str, str]] = set()
    merged: List[dict] = []
    for cls in kg_classes:
        for t in per_class_traversal[cls]["triplets"]:
            tk = (t["s"].lower(), t["p"].lower(), t["o"].lower())
            if tk not in seen_keys:
                seen_keys.add(tk)
                merged.append(t)

    # Global rank
    merged.sort(key=lambda t: RELATION_PRIORITY.get(t["p"], 0), reverse=True)

    return {
        "uid":                  uid,
        "gt_labels":            gt_labels,
        "predicted_classes":    predicted_classes,
        "kg_classes":           kg_classes,
        "conflict_resolution":  conflict_resolution,
        "is_uncertain":         is_uncertain,
        "uncertainty_reason":   uncertainty_reason,
        "per_class_traversal":  per_class_traversal,
        "merged_triplets":      merged,
        "merged_count":         len(merged),
        "top_k_per_class":      top_k,
        "leaf_traversal":       True,
        "acyclic":              True,
    }



def load_predictions(csv_path: Path) -> List[dict]:
    """
    Parse predictions.csv produced by step_A.

    Reads (per row):
      uid, gt_labels, predicted_classes, resolved_classes, kg_classes,
      conflict_resolution, is_uncertain, uncertainty_reason
    """
    records: List[dict] = []

    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            uid = row.get("uid", "").strip()
            if not uid:
                continue

            # predicted_classes: pipe-separated raw predictions (may include 'normal')
            raw_predicted = row.get("predicted_classes", "").strip()
            predicted_classes = [
                c.strip() for c in raw_predicted.split("|") if c.strip()
            ]

            # kg_classes: post-conflict-resolution classes used for KG traversal
            raw_kg = row.get("kg_classes", "").strip()
            kg_classes = [c.strip() for c in raw_kg.split("|") if c.strip()]

            records.append({
                "uid":                 uid,
                "gt_labels":           row.get("gt_labels", "").strip(),
                "predicted_classes":   predicted_classes,
                "kg_classes":          kg_classes,
                "conflict_resolution": row.get("conflict_resolution", "no_conflict").strip(),
                "is_uncertain":        row.get("is_uncertain", "False").strip().lower() == "true",
                "uncertainty_reason":  row.get("uncertainty_reason", "").strip(),
            })

    log.info("Loaded %d UID records from %s", len(records), csv_path.name)
    return records

def _print_summary(
    results: List[dict],
    elapsed_total: float,
    output_dir: Path,
) -> None:
    total_uids = len(results)
    if total_uids == 0:
        log.warning("No results to summarise.")
        return

    total_classes = sum(len(r["kg_classes"]) for r in results)
    total_merged  = sum(r["merged_count"] for r in results)

    class_counts: Dict[str, int] = defaultdict(int)
    for r in results:
        for cls in r["kg_classes"]:
            class_counts[cls] += 1

    top_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:8]

    log.info(
        "\n",
        total_uids,
        total_classes,
        total_classes / total_uids,
        total_merged  / total_uids,
        elapsed_total,
        str(output_dir),
    )
    log.info("Top predicted classes:")
    for cls, cnt in top_classes:
        log.info("  %-35s %d UIDs", cls, cnt)

def run_traversal(
    predictions_path: Path,
    kg_path:          Path,
    output_dir:       Path,
    top_k:            int = 20,
) -> None:
    """
    Main entry point.

    1. Load the KG (once, shared across all UIDs).
    2. Load prediction records from CSV.
    3. For each UID, process each kg_class with an independent leaf-node DFS.
    4. Write one JSON per UID to output_dir.
    5. Write a consolidated summary JSONL.
    """
    # ---- Validate inputs ----
    if not predictions_path.exists():
        log.error("predictions CSV not found: %s", predictions_path)
        sys.exit(1)
    if not kg_path.exists():
        log.error("KG JSONL not found: %s", kg_path)
        sys.exit(1)

    # ---- Load KG (shared, read-only) ----
    # Use pre-built adjacency list (graph_adj.json) if available next to
    # the JSONL file — avoids re-parsing JSONL on every run.
    adj_path = kg_path.parent / "graph_adj.json"
    kg = KnowledgeGraph(kg_path, adj_path=adj_path)

    # ---- Load prediction records ----
    records = load_predictions(predictions_path)
    if not records:
        log.error("No prediction records found in %s", predictions_path)
        sys.exit(1)

    # ---- Prepare output directory ----
    output_dir.mkdir(parents=True, exist_ok=True)
    consolidated_path = output_dir / "all_uid_traversals.jsonl"

    results: List[dict] = []
    t_start = time.time()

    log.info(
        "Starting KG traversal (leaf-node DFS) | UIDs=%d | top_k=%d",
        len(records), top_k,
    )

    with open(consolidated_path, "w", encoding="utf-8") as cons_fh:

        for rec in tqdm(records, desc="Traversing UIDs", unit="uid"):
            uid        = rec["uid"]
            kg_classes = rec["kg_classes"]

            if not kg_classes:
                log.warning("uid=%-8s  SKIP — no kg_classes", uid)
                continue
            result = traverse_uid(
                kg                  = kg,
                uid                 = uid,
                kg_classes          = kg_classes,
                predicted_classes   = rec["predicted_classes"],
                gt_labels           = rec["gt_labels"],
                conflict_resolution = rec["conflict_resolution"],
                is_uncertain        = rec["is_uncertain"],
                uncertainty_reason  = rec["uncertainty_reason"],
                top_k               = top_k,
            )

            cls_summary = " | ".join(
                f"{cls}({result['per_class_traversal'][cls]['triplet_count']})"
                for cls in kg_classes
            )
            log.info(
                "uid=%-8s  classes=[%s]  merged=%d",
                uid, cls_summary, result["merged_count"],
            )

            results.append(result)

            # ---- Per-UID JSON ----
            uid_json_path = output_dir / f"{uid}_traversal.json"
            with open(uid_json_path, "w", encoding="utf-8") as uf:
                json.dump(result, uf, indent=2, ensure_ascii=False)

            # ---- Consolidated JSONL (compact, without path_traces for size) ----
            compact = {
                k: v for k, v in result.items()
                if k != "per_class_traversal"  # full detail in per-UID JSON
            }
            # Add per-class triplet counts to compact summary
            compact["per_class_counts"] = {
                cls: result["per_class_traversal"][cls]["triplet_count"]
                for cls in kg_classes
            }
            cons_fh.write(json.dumps(compact, ensure_ascii=False) + "\n")

    elapsed = time.time() - t_start
    _print_summary(results, elapsed, output_dir)
    log.info("Per-UID JSONs written to : %s", output_dir)
    log.info("Consolidated JSONL       : %s", consolidated_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "KG Traversal: traverse knowledge graph per predicted class "
            "per UID using leaf-node DFS (acyclic, no fixed depth cap)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--predictions",
        default="/Users/bkishor/Desktop/kg_new/BiomedCLIP/Output/predictions.csv",
        help="Path to predictions.csv from step_A",
    )
    p.add_argument(
        "--kg_path",
        default="/Users/bkishor/Desktop/kg_new/Knowlege-Graph/output/candidate_triples.jsonl",
        help="Path to candidate_triples.jsonl (KG edge file)",
    )
    p.add_argument(
        "--output_dir",
        default="/Users/bkishor/Desktop/kg_new/Knowlege-Graph/output/per_uid_traversals",
        help="Directory where per-UID JSON files will be written",
    )
    p.add_argument(
        "--top_k",
        type=int,
        default=30,
        help="Maximum triplets to keep per class after leaf-node DFS",
    )
    return p.parse_args()

def main() -> None:
    args = parse_args()
    run_traversal(
        predictions_path = Path(args.predictions),
        kg_path          = Path(args.kg_path),
        output_dir       = Path(args.output_dir),
        top_k            = args.top_k,
    )


if __name__ == "__main__":
    main()
