"""

we are using llama-3.3-70b-versatile , groq model 
test_pipeline.py — End-to-End Chest X-Ray Analysis Pipeline
=============================================================
Given 1 image (frontal) or 2 images (frontal + lateral), this script:

  1. CLASSIFICATION  — Runs the fine-tuned BiomedCLIP multi-label classifier
                       to predict disease classes with dual-view fusion,
                       conflict resolution, and uncertainty flagging.

  2. KG TRIPLETS     — For every predicted class, traverses the Knowledge
                       Graph (leaf-node DFS) and returns ranked triplets
                       describing findings, relations, and clinical context.

  3. RETRIEVAL       — Encodes the query image with the fine-tuned CLIP
                       retriever and returns the top-K most similar
                       radiology reports from the training database
                       (hybrid text + image blended scoring).

Usage
-----
  # Single frontal image
  python test_pipeline.py --frontal /path/to/frontal.png

  # Frontal + lateral (dual-view fusion)
  python test_pipeline.py --frontal /path/to/frontal.png --lateral /path/to/lateral.png

  # With options
  python test_pipeline.py --frontal img.png --top_k 5 --threshold 0.5 --device auto

"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")

try:
    from openai import AzureOpenAI as _AzureOpenAIClient
    _azure_api_key   = os.getenv("AZURE_OPENAI_API_KEY", "")
    _azure_endpoint  = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    _azure_api_ver   = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
    _azure_deploy    = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME_5_1", "gpt-4o-mini")
    _azure_client    = _AzureOpenAIClient(
        api_key=_azure_api_key,
        azure_endpoint=_azure_endpoint,
        api_version=_azure_api_ver,
    ) if _azure_api_key and _azure_endpoint else None
except ImportError:
    _azure_client = None
    _azure_deploy = "gpt-4o-mini"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

warnings.filterwarnings("ignore")

try:
    import open_clip
except ImportError:
    sys.exit("[ERROR] open_clip not installed. Run: pip install open-clip-torch")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent

# Classifier
CLASSIFIER_CKPT = BASE_DIR / "BiomedCLIP" / "epoch_search_e100_fold2_new.pth"

# Knowledge Graph
KG_JSONL   = BASE_DIR / "Knowlege-Graph" / "output" / "candidate_triples.jsonl"
KG_ADJ     = BASE_DIR / "Knowlege-Graph" / "output" / "graph_adj.json"

# Retriever
RETRIEVER_CKPT   = BASE_DIR / "Retriever" / "checkpoints" / "best_indiana_clip.pt"
DB_TEXT_VECTORS   = BASE_DIR / "Retriever" / "database" / "indiana_database_vectors.npy"
DB_IMG_VECTORS    = BASE_DIR / "Retriever" / "database" / "indiana_database_img_vectors.npy"
DB_METADATA       = BASE_DIR / "Retriever" / "database" / "indiana_database_metadata.json"
BLEND_WEIGHTS     = BASE_DIR / "Retriever" / "database" / "blend_weights.json"

BIOMED_CLIP_NAME = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

CLASS_LABELS: List[str] = [
    "normal", "degenerative disease", "lesion", "hypoinflation",
    "calcified granuloma", "cardiomegaly", "cicatrix", "calcinosis",
    "airspace disease", "fibrosis", "lung/hyperdistention", "pleural effusion",
    "emphysema", "nodule", "edema", "scoliosis", "fractures", "hernia",
    "pleural thickening", "osteophyte", "interstitial lung disease",
    "consolidation", "cardiac shadow", "thickening", "kyphosis", "pneumothorax",
    "mass", "pulmonary artery enlargement", "pulmonary fibrosis", "effusion",
    "bronchiectasis", "bullous disease", "rib fracture", "subcutaneous emphysema",
    "bronchiolitis",
]

_LABEL_SYNONYM_OVERRIDES: Dict[str, str] = {
    "degenerative disease": "Degenerative Change",
    "cicatrix":             "Fibrosis",
    "hypoinflation":        "Volume Loss",
    "lung/hyperdistention": "Hyperinflation",
    "cardiac shadow":       "Cardiac Shadow (abnormal)",
}

NORMAL_DOMINANCE_THRESHOLD = 0.70
NORMAL_OVERRIDE_DISEASE    = 0.42
UNCERTAIN_UPPER            = 0.73
UNCERTAIN_DISEASE_MIN      = 0.32


def _build_label_normalize_map(label_columns: List[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for cls in label_columns:
        cls_lower = cls.strip().lower()
        if cls_lower in _LABEL_SYNONYM_OVERRIDES:
            mapping[cls_lower] = _LABEL_SYNONYM_OVERRIDES[cls_lower]
            continue
        kg_name = cls.title().replace("/", " ").strip()
        if "/" in cls:
            kg_name = " ".join(w.capitalize() for w in cls.replace("/", " ").split())
        mapping[cls_lower] = kg_name
    return mapping


LABEL_NORMALIZE_MAP: Dict[str, str] = _build_label_normalize_map(CLASS_LABELS)


class BiomedCLIPClassifier(nn.Module):
    """Multi-label classifier head on top of BiomedCLIP visual encoder."""

    def __init__(self, num_classes: int, unfreeze_last_n_blocks: int = 4):
        super().__init__()
        self.model, self.preprocess_train, self.preprocess_val = \
            open_clip.create_model_and_transforms(BIOMED_CLIP_NAME)

        for param in self.model.parameters():
            param.requires_grad = False

        if unfreeze_last_n_blocks > 0 and hasattr(self.model, "visual"):
            blocks = []
            if hasattr(self.model.visual, "transformer"):
                blocks = self.model.visual.transformer.resblocks
            elif hasattr(self.model.visual, "blocks"):
                blocks = self.model.visual.blocks
            for block in blocks[-unfreeze_last_n_blocks:]:
                for p in block.parameters():
                    p.requires_grad = True
            if hasattr(self.model.visual, "ln_post"):
                for p in self.model.visual.ln_post.parameters():
                    p.requires_grad = True

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            embed_dim = self.model.encode_image(dummy).shape[-1]

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.model.encode_image(x)
        features = F.normalize(features, dim=-1)
        return self.classifier(features)


def _load_classifier(device: torch.device) -> Tuple[BiomedCLIPClassifier, List[str]]:
    """Load the fine-tuned multi-label classifier from checkpoint."""
    if not CLASSIFIER_CKPT.exists():
        sys.exit(f"[ERROR] Classifier checkpoint not found: {CLASSIFIER_CKPT}")

    log.info("Loading classifier checkpoint: %s", CLASSIFIER_CKPT.name)
    ckpt = torch.load(CLASSIFIER_CKPT, map_location=device, weights_only=False)

    label_columns: List[str] = ckpt.get("label_columns", CLASS_LABELS)
    model = BiomedCLIPClassifier(
        num_classes=len(label_columns),
        unfreeze_last_n_blocks=4,
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device).eval()

    # Rebuild label→KG name mapping for these exact columns
    global LABEL_NORMALIZE_MAP
    LABEL_NORMALIZE_MAP = _build_label_normalize_map(label_columns)

    log.info("  Classes=%d | Epoch=%s | Val AUC=%.4f",
             len(label_columns), ckpt.get("epoch", "?"), ckpt.get("val_auc", 0.0))
    return model, label_columns


@torch.no_grad()
def _infer_single_image(
    image_path: Path,
    model: BiomedCLIPClassifier,
    device: torch.device,
    label_columns: List[str],
) -> Optional[Dict[str, float]]:
    """Run classifier on a single image → {class: sigmoid_prob}."""
    if not image_path.exists():
        log.warning("Image not found: %s", image_path)
        return None
    try:
        image  = Image.open(image_path).convert("RGB")
        tensor = model.preprocess_val(image).unsqueeze(0).to(device)
        logits = model(tensor)
        probs  = torch.sigmoid(logits).squeeze(0).cpu().numpy()
        return {label_columns[i]: float(probs[i]) for i in range(len(label_columns))}
    except Exception as exc:
        log.error("Cannot infer %s: %s", image_path.name, exc)
        return None


def _fuse_probs(
    frontal_prob: Optional[Dict[str, float]],
    lateral_prob: Optional[Dict[str, float]],
    label_columns: List[str],
    frontal_weight: float = 0.65,
) -> Tuple[Dict[str, float], str]:
    """Weighted-average fusion of frontal and lateral probabilities."""
    lateral_weight = 1.0 - frontal_weight

    if frontal_prob is not None and lateral_prob is not None:
        fused = {
            c: round(frontal_weight * frontal_prob[c] + lateral_weight * lateral_prob[c], 6)
            for c in label_columns
        }
        note = f"weighted_avg(f={frontal_weight:.2f},l={lateral_weight:.2f})"
    elif frontal_prob is not None:
        fused = frontal_prob
        note  = "frontal_only"
    elif lateral_prob is not None:
        fused = lateral_prob
        note  = "lateral_only"
    else:
        fused = {c: 0.0 for c in label_columns}
        note  = "no_image"
    return fused, note


def _resolve_conflict(
    predicted_classes: List[str],
    prob_dict: Dict[str, float],
) -> Tuple[List[str], str]:
    """Normal vs Disease conflict resolution."""
    has_normal   = "normal" in predicted_classes
    disease_list = [c for c in predicted_classes if c != "normal"]
    if not has_normal or not disease_list:
        return predicted_classes, "no_conflict"
    normal_prob  = prob_dict.get("normal", 0.0)
    max_dis_prob = max((prob_dict.get(c, 0.0) for c in disease_list), default=0.0)
    if normal_prob >= NORMAL_DOMINANCE_THRESHOLD and max_dis_prob < NORMAL_OVERRIDE_DISEASE:
        return ["normal"], "normal_dominates"
    return disease_list, "disease_dominates"


def _compute_uncertainty(
    resolved_classes: List[str],
    prob_dict: Dict[str, float],
) -> Tuple[bool, str]:
    if resolved_classes != ["normal"]:
        return False, ""
    normal_prob = prob_dict.get("normal", 1.0)
    if normal_prob >= UNCERTAIN_UPPER:
        return False, ""
    lurking = [
        (c, p) for c, p in prob_dict.items()
        if c != "normal" and p >= UNCERTAIN_DISEASE_MIN
    ]
    if not lurking:
        return False, ""
    top3 = sorted(lurking, key=lambda x: x[1], reverse=True)[:3]
    reason = "normal confidence is %s with elevated scores for: %s" % (
        _confidence_label(normal_prob),
        ", ".join("%s (%s confidence)" % (c, _confidence_label(p)) for c, p in top3),
    )
    return True, reason


def run_classification(
    frontal_path: Optional[Path],
    lateral_path: Optional[Path],
    device: torch.device,
    threshold: float = 0.5,
    frontal_weight: float = 0.65,
) -> dict:
    """
    Stage 1: Classify image(s) → predicted classes, KG class names, probabilities.
    """
    classifier, label_columns = _load_classifier(device)
    thresholds = {cls: threshold for cls in label_columns}

    # Infer each view
    frontal_prob = _infer_single_image(frontal_path, classifier, device, label_columns) \
                   if frontal_path else None
    lateral_prob = _infer_single_image(lateral_path, classifier, device, label_columns) \
                   if lateral_path else None

    # Fuse
    fused_prob, fusion_note = _fuse_probs(
        frontal_prob, lateral_prob, label_columns, frontal_weight
    )

    # Threshold → multi-label
    predicted_classes: List[str] = sorted(
        [cls for cls, prob in fused_prob.items() if prob >= thresholds[cls]],
        key=lambda c: fused_prob[c],
        reverse=True,
    )

    # Conflict resolution
    resolved_classes, conflict = _resolve_conflict(predicted_classes, fused_prob)

    # KG name mapping
    kg_classes: List[str] = []
    for cls in resolved_classes:
        kg_key = LABEL_NORMALIZE_MAP.get(cls.lower())
        if kg_key and kg_key not in kg_classes:
            kg_classes.append(kg_key)

    # Uncertainty
    is_uncertain, uncertainty_reason = _compute_uncertainty(resolved_classes, fused_prob)

    # Top-5 probabilities for display
    top5_probs = sorted(fused_prob.items(), key=lambda x: x[1], reverse=True)[:5]

    return {
        "predicted_classes":    predicted_classes,
        "resolved_classes":     resolved_classes,
        "kg_classes":           kg_classes,
        "conflict_resolution":  conflict,
        "is_uncertain":         is_uncertain,
        "uncertainty_reason":   uncertainty_reason,
        "fusion_mode":          fusion_note,
        "probabilities":        fused_prob,
        "top5_probs":           top5_probs,
    }







RELATION_PRIORITY: Dict[str, int] = {
    "strongly_suggests": 6,  "confirmed_by": 5,  "suggests": 4,
    "is_finding_of": 3,      "is_symptom_of": 3, "requires": 2,
    "located_in": 2,         "weakly_suggests": 1, "absence_weakens": 1,
    "contradicts": 4,
}

HUB_ENTITIES: Set[str] = {
    "chest radiograph", "lung fields", "pulmonary vasculature",
    "cardiothoracic ratio", "thoracic cavity", "thoracic spine",
    "pleural space", "pericardial space", "lung", "heart",
    "mediastinum", "chest", "thorax", "lungs", "chest x-ray",
    "radiograph", "disease", "vasculature",
}

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
    {"s": "clear lung fields",           "p": "contradicts",     "o": "acute cardiopulmonary disease"},
    {"s": "clear lung fields",           "p": "contradicts",     "o": "pneumonia"},
    {"s": "clear lung fields",           "p": "contradicts",     "o": "congestive heart failure"},
    {"s": "clear lung fields",           "p": "contradicts",     "o": "pneumothorax"},
    {"s": "clear lung fields",           "p": "is_finding_of",   "o": "well-defined pulmonary vasculature"},
    {"s": "clear lung fields",           "p": "is_finding_of",   "o": "normal cardiothoracic ratio"},
    {"s": "normal cardiac silhouette",   "p": "contradicts",     "o": "cardiomegaly"},
    {"s": "normal cardiac silhouette",   "p": "contradicts",     "o": "pericardial effusion"},
    {"s": "normal cardiac silhouette",   "p": "is_finding_of",   "o": "normal cardiothoracic ratio"},
    {"s": "normal cardiothoracic ratio", "p": "contradicts",     "o": "cardiomegaly"},
    {"s": "normal cardiothoracic ratio", "p": "contradicts",     "o": "congestive heart failure"},
    {"s": "normal mediastinal contours", "p": "contradicts",     "o": "mediastinal widening"},
    {"s": "normal mediastinal contours", "p": "contradicts",     "o": "aortic aneurysm"},
    {"s": "normal examination",          "p": "contradicts",     "o": "acute cardiopulmonary disease"},
    {"s": "normal examination",          "p": "contradicts",     "o": "active pulmonary disease"},
    {"s": "normal examination",          "p": "is_finding_of",   "o": "clear lung fields"},
    {"s": "normal examination",          "p": "is_finding_of",   "o": "normal cardiac silhouette"},
    {"s": "Normal chest radiograph",     "p": "contradicts",     "o": "cardiomegaly"},
    {"s": "Normal chest radiograph",     "p": "contradicts",     "o": "calcinosis"},
    {"s": "Normal chest radiograph",     "p": "contradicts",     "o": "pulmonary edema"},
    {"s": "Normal chest radiograph",     "p": "contradicts",     "o": "pleural effusion"},
    {"s": "Normal chest radiograph",     "p": "contradicts",     "o": "pneumothorax"},
    {"s": "Normal chest radiograph",     "p": "contradicts",     "o": "lobar consolidation"},
    {"s": "Normal chest radiograph",     "p": "absence_weakens", "o": "acute cardiopulmonary disease"},
    {"s": "Normal chest radiograph",     "p": "weakly_suggests", "o": "normal cardiopulmonary status"},
    {"s": "Normal chest radiograph",     "p": "confirmed_by",    "o": "radiographic interpretation"},
    {"s": "Normal chest radiograph",     "p": "requires",        "o": "adequate inspiratory effort"},
    {"s": "Normal cardiac shadow",       "p": "contradicts",     "o": "cardiomegaly"},
    {"s": "Normal cardiac shadow",       "p": "weakly_suggests", "o": "normal cardiac size"},
]

CLASS_TO_SEEDS: Dict[str, List[str]] = {
    "Normal": [
        "normal", "clear lung fields", "Normal chest radiograph",
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
    "Bronchiolitis": [
        "bronchiolitis", "diffuse bilateral peribronchiolar nodular opacities",
        "centrilobular nodules", "hypersensitivity reactions",
        "chronic productive cough", "recurrent respiratory infections",
    ],
}


class KnowledgeGraph:
    """In-memory directed KG with forward-only adjacency (subject → object)."""

    def __init__(self, jsonl_path: Path, adj_path: Optional[Path] = None) -> None:
        self._adj: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        self._subject_orig: Dict[str, str] = {}
        self._total: int = 0

        if adj_path is not None and adj_path.exists():
            self._load_from_adj(adj_path)
        elif jsonl_path.exists():
            self._load_jsonl(jsonl_path)
        else:
            log.warning("KG files not found — triplet stage will be empty")

    def _load_jsonl(self, path: Path) -> None:
        seen: Set[Tuple[str, str, str]] = set()
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    continue
                s, p, o = raw["s"].strip(), raw["p"].strip(), raw["o"].strip()
                key = (s.lower(), p.lower(), o.lower())
                if key in seen:
                    continue
                seen.add(key)
                sl = s.lower()
                self._adj[sl].append((p, o))
                if sl not in self._subject_orig:
                    self._subject_orig[sl] = s
                self._total += 1
        log.info("KG loaded (JSONL): %d triplets, %d subjects", self._total, len(self._adj))

    def _load_from_adj(self, adj_path: Path) -> None:
        with open(adj_path, "r", encoding="utf-8") as fh:
            raw: Dict[str, list] = json.load(fh)
        for subject_lower, edges in raw.items():
            for edge in edges:
                self._adj[subject_lower].append((edge[0], edge[1]))
                self._total += 1
            if subject_lower not in self._subject_orig:
                self._subject_orig[subject_lower] = subject_lower
        log.info("KG loaded (adj JSON): %d triplets, %d subjects", self._total, len(self._adj))

    def outgoing(self, subject: str) -> List[Tuple[str, str]]:
        return self._adj.get(subject.strip().lower(), [])

    def has_subject(self, subject: str) -> bool:
        return subject.strip().lower() in self._adj

    def subject_orig(self, subject_lower: str) -> str:
        return self._subject_orig.get(subject_lower, subject_lower)


def _dfs_to_leaves(
    kg: KnowledgeGraph,
    seeds: List[str],
) -> Tuple[List[dict], List[str], List[str]]:
    """Iterative DFS from seeds to leaf nodes; returns (triplets, used, missing)."""
    visited: Set[str] = set()
    seen_keys: Set[Tuple[str, str, str]] = set()
    triplets: List[dict] = []
    seeds_used: List[str] = []
    seeds_missing: List[str] = []

    stack: List[str] = []
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
            stack.append(sl)

    while stack:
        current = stack.pop()
        edges = kg.outgoing(current)
        if not edges:
            continue
        current_orig = kg.subject_orig(current)
        for relation, obj in edges:
            tk = (current, relation.lower(), obj.lower())
            if tk not in seen_keys:
                seen_keys.add(tk)
                triplets.append({"s": current_orig, "p": relation, "o": obj})
            obj_lower = obj.strip().lower()
            if obj_lower in HUB_ENTITIES or obj_lower in visited:
                continue
            visited.add(obj_lower)
            stack.append(obj_lower)

    triplets.sort(key=lambda t: RELATION_PRIORITY.get(t["p"], 0), reverse=True)
    return triplets, seeds_used, seeds_missing


def _filter_noise(triplets: List[dict]) -> List[dict]:
    return [
        t for t in triplets
        if t.get("s", "").strip()
        and t.get("o", "").strip()
        and t["s"].strip().lower() not in KG_NOISE_SUBJECTS
    ]


def run_kg_traversal(
    kg_classes: List[str],
    top_k_per_class: int = 20,
) -> dict:
    """
    Stage 2: For each predicted KG class, traverse the KG and collect triplets.
    """
    kg = KnowledgeGraph(KG_JSONL, adj_path=KG_ADJ)

    per_class: Dict[str, dict] = {}
    for cls in kg_classes:
        if cls == "Normal":
            capped = NORMAL_TRIPLETS[:top_k_per_class]
            per_class[cls] = {
                "seeds_used":    ["normal examination", "clear lung fields",
                                  "normal cardiac silhouette"],
                "seeds_missing": [],
                "triplets":      capped,
                "triplet_count": len(capped),
            }
            continue

        seeds = CLASS_TO_SEEDS.get(cls)
        if not seeds:
            per_class[cls] = {
                "seeds_used": [], "seeds_missing": [],
                "triplets": [], "triplet_count": 0,
            }
            continue

        raw, seeds_used, seeds_missing = _dfs_to_leaves(kg, seeds)
        clean = _filter_noise(raw)[:top_k_per_class]
        per_class[cls] = {
            "seeds_used":    seeds_used,
            "seeds_missing": seeds_missing,
            "triplets":      clean,
            "triplet_count": len(clean),
        }

    # Merge across classes (deduplicated)
    seen: Set[Tuple[str, str, str]] = set()
    merged: List[dict] = []
    for cls in kg_classes:
        for t in per_class[cls]["triplets"]:
            tk = (t["s"].lower(), t["p"].lower(), t["o"].lower())
            if tk not in seen:
                seen.add(tk)
                merged.append(t)
    merged.sort(key=lambda t: RELATION_PRIORITY.get(t["p"], 0), reverse=True)

    return {
        "per_class_traversal": per_class,
        "merged_triplets":     merged,
        "merged_count":        len(merged),
    }


# ══════════════════════════════════════════════════════════════════════
#  STAGE 3 — RETRIEVAL (from Retriever module, hybrid blended)
# ══════════════════════════════════════════════════════════════════════

def _load_retriever(device: torch.device):
    """Load fine-tuned CLIP model, preprocess, and the hybrid database."""
    if not RETRIEVER_CKPT.exists():
        sys.exit(f"[ERROR] Retriever checkpoint not found: {RETRIEVER_CKPT}")

    log.info("Loading retriever checkpoint: %s", RETRIEVER_CKPT.name)
    model, _, preprocess = open_clip.create_model_and_transforms(BIOMED_CLIP_NAME)
    ckpt = torch.load(RETRIEVER_CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    # Text DB
    text_vectors = np.load(DB_TEXT_VECTORS)
    db_text = torch.from_numpy(text_vectors).to(device)

    # Image DB (optional, for hybrid blending)
    db_img = None
    if DB_IMG_VECTORS.exists():
        img_vectors = np.load(DB_IMG_VECTORS)
        db_img = torch.from_numpy(img_vectors).to(device)

    # Metadata
    with open(DB_METADATA) as f:
        metadata = json.load(f)

    # Blend weights
    text_weight, image_weight = 0.6, 0.4
    if BLEND_WEIGHTS.exists():
        with open(BLEND_WEIGHTS) as f:
            bw = json.load(f)
        text_weight  = bw.get("text", 0.6)
        image_weight = bw.get("image", 0.4)

    log.info("  DB: %d reports | blend: text=%.2f, image=%.2f",
             len(metadata), text_weight, image_weight)
    return model, preprocess, db_text, db_img, metadata, text_weight, image_weight


@torch.no_grad()
def run_retrieval(
    frontal_path: Optional[Path],
    lateral_path: Optional[Path],
    device: torch.device,
    top_k: int = 5,
) -> List[dict]:
    """
    Stage 3: Encode query image(s) and retrieve top-K reports (hybrid blend).

    If both frontal and lateral are provided, their image embeddings are
    averaged (same approach as training DB construction and evaluation).
    """
    model, preprocess, db_text, db_img, metadata, text_w, img_w = _load_retriever(device)

    # Encode all available images and average
    image_features_list = []
    for img_path in [frontal_path, lateral_path]:
        if img_path is None or not img_path.exists():
            continue
        image = Image.open(img_path).convert("RGB")
        tensor = preprocess(image).unsqueeze(0).to(device)
        feat = model.encode_image(tensor)
        feat = F.normalize(feat, dim=-1)
        image_features_list.append(feat)

    if not image_features_list:
        log.error("No valid images for retrieval")
        return []

    # Average across views
    avg_feat = torch.mean(torch.stack(image_features_list), dim=0)
    avg_feat = F.normalize(avg_feat, dim=-1)

    # Hybrid blended similarity
    text_sim = (avg_feat @ db_text.T).squeeze(0)

    if db_img is not None and img_w > 0:
        img_sim = (avg_feat @ db_img.T).squeeze(0)
        similarities = text_w * text_sim + img_w * img_sim
    else:
        similarities = text_sim

    top_k_vals, top_k_indices = torch.topk(similarities, min(top_k, len(metadata)))

    results = []
    for rank, (score, idx) in enumerate(
        zip(top_k_vals.cpu().numpy(), top_k_indices.cpu().numpy()), 1
    ):
        entry = metadata[idx]
        results.append({
            "rank":       rank,
            "uid":        entry["uid"],
            "caption":    entry["caption"],
            "similarity": round(float(score), 4),
        })
    return results


# ══════════════════════════════════════════════════════════════════════
#  STAGE 4 — CLINICAL SUMMARY GENERATION
# ══════════════════════════════════════════════════════════════════════

def _confidence_label(prob: float) -> str:
    """Map a probability to a human-readable confidence word."""
    if prob >= 0.85:
        return "high"
    if prob >= 0.65:
        return "moderate"
    if prob >= 0.50:
        return "low"
    return "borderline"


def _extract_kg_context_per_class(
    per_class_traversal: Dict[str, dict],
) -> Dict[str, dict]:
    """
    For each class, bucket its triplets by relation type into
    human-readable lists: findings, suggestions, confirmations,
    locations, contradictions, requirements.
    """
    context: Dict[str, dict] = {}
    for cls_name, trav in per_class_traversal.items():
        buckets: Dict[str, List[str]] = {
            "findings":       [],
            "suggestions":    [],
            "confirmations":  [],
            "locations":      [],
            "contradictions": [],
            "requirements":   [],
        }
        for t in trav["triplets"]:
            rel = t["p"].lower()
            obj = t["o"]
            if rel in ("is_finding_of", "is_symptom_of"):
                buckets["findings"].append(obj)
            elif rel in ("strongly_suggests", "suggests", "weakly_suggests"):
                buckets["suggestions"].append(obj)
            elif rel == "confirmed_by":
                buckets["confirmations"].append(obj)
            elif rel == "located_in":
                buckets["locations"].append(obj)
            elif rel in ("contradicts", "absence_weakens"):
                buckets["contradictions"].append(obj)
            elif rel == "requires":
                buckets["requirements"].append(obj)
        # Deduplicate while preserving order
        for k in buckets:
            seen = set()
            deduped = []
            for v in buckets[k]:
                vl = v.lower()
                if vl not in seen:
                    seen.add(vl)
                    deduped.append(v)
            buckets[k] = deduped
        context[cls_name] = buckets
    return context


def _extract_retrieval_evidence(retrieved: List[dict]) -> dict:
    """
    Analyse the top-K retrieved reports: extract common keywords that
    appear across multiple reports to highlight retrieval-backed evidence.
    """
    # Clinical keywords to track
    CLINICAL_TERMS = [
        "cardiomegaly", "pleural effusion", "pneumothorax", "consolidation",
        "edema", "emphysema", "fibrosis", "nodule", "mass", "atelectasis",
        "fracture", "scoliosis", "kyphosis", "hernia", "calcification",
        "granuloma", "osteophyte", "thickening", "bronchiectasis",
        "opacity", "infiltrate", "congestion", "effusion",
        "normal", "clear", "unremarkable", "no acute",
        "degenerative", "hyperexpanded", "hyperinflat",
        "pneumonia", "interstitial", "pulmonary artery",
    ]
    term_counts: Dict[str, int] = {t: 0 for t in CLINICAL_TERMS}
    n = len(retrieved)
    for r in retrieved:
        caption_lower = r["caption"].lower()
        for term in CLINICAL_TERMS:
            if term in caption_lower:
                term_counts[term] += 1

    # Keep terms found in >= 2 reports (or >= 40% of reports)
    threshold = max(2, int(n * 0.4))
    common = {t: c for t, c in term_counts.items() if c >= threshold}
    common_sorted = sorted(common.items(), key=lambda x: x[1], reverse=True)

    return {
        "total_reports": n,
        "common_findings": common_sorted,
        "avg_similarity": round(np.mean([r["similarity"] for r in retrieved]), 4) if retrieved else 0.0,
    }


def _build_llm_prompt(result: dict) -> str:
    """Build a detailed prompt for the Groq LLM from pipeline data."""
    cls = result["classification"]
    kg  = result["kg_triplets"]
    ret = result["retrieved_reports"]

    resolved   = cls["resolved_classes"]
    probs      = cls["probabilities"]
    kg_classes = cls["kg_classes"]

    lines = []
    lines.append("You are an expert radiologist AI assistant. Below is the structured analysis "
                 "of a chest X-ray produced by an automated pipeline. Your task is to produce "
                 "a comprehensive, medically accurate, natural-language clinical summary that a "
                 "clinician or patient can read and that enables answering ANY follow-up question "
                 "about this chest X-ray.")
    lines.append("")
    lines.append("=== STAGE 1: CLASSIFICATION RESULTS ===")
    lines.append(f"Detected conditions: {', '.join(resolved) if resolved else 'None above threshold'}")
    lines.append(f"Fusion mode: {cls['fusion_mode']}")
    lines.append(f"Conflict resolution: {cls['conflict_resolution']}")
    lines.append(f"Uncertain flag: {cls['is_uncertain']}")
    if cls["is_uncertain"]:
        lines.append(f"Uncertainty reason: {cls['uncertainty_reason']}")
    lines.append("")
    lines.append("Class probabilities (top 10):")
    top10 = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:10]
    for name, prob in top10:
        lines.append(f"  {name}: {prob:.4f}")

    lines.append("")
    lines.append("=== STAGE 2: KNOWLEDGE GRAPH TRIPLETS ===")
    for cls_name in kg_classes:
        trav = kg["per_class_traversal"].get(cls_name, {})
        triplets = trav.get("triplets", [])[:15]
        if triplets:
            lines.append(f"\n[{cls_name}]")
            for t in triplets:
                lines.append(f"  ({t['s']}) --[{t['p']}]--> ({t['o']})")

    lines.append("")
    lines.append("=== STAGE 3: RETRIEVED SIMILAR REPORTS (TOP-K) ===")
    for i, r in enumerate(ret, 1):
        lines.append(f"\nReport {i} (similarity={r.get('similarity', 0):.4f}):")
        lines.append(f"  {r.get('caption', 'N/A')[:400]}")

    lines.append("")
    lines.append("=== YOUR TASK ===")
    lines.append("Using ALL the above information, write a detailed clinical summary with these sections:")
    lines.append("")
    lines.append("1. FINDINGS: A thorough paragraph describing what is observed in this chest X-ray, "
                 "including all detected conditions with their confidence levels.")
    lines.append("2. CLINICAL CONTEXT: For each detected condition, describe what it means clinically, "
                 "what findings support it, and what conditions it rules out (use the KG triplets).")
    lines.append("3. EVIDENCE FROM SIMILAR CASES: Summarize what similar retrieved cases show and how "
                 "they corroborate or add context to the findings.")
    lines.append("4. CONFIDENCE & UNCERTAINTY: State overall confidence in the findings and note any "
                 "uncertainty or caveats.")
    lines.append("5. KEY Q&A: Answer these specific questions in bullet form:")
    lines.append("   - What conditions are detected?")
    lines.append("   - What is the most likely primary diagnosis?")
    lines.append("   - What are the supporting radiological findings?")
    lines.append("   - What conditions are ruled out?")
    lines.append("   - Is this a normal chest X-ray?")
    lines.append("   - What follow-up actions or tests might be recommended?")
    lines.append("   - What are the clinical implications for the patient?")
    lines.append("")
    lines.append("IMPORTANT: Only describe conditions as PRESENT if they appear in the CLASSIFICATION RESULTS above.")
    lines.append("Do NOT hallucinate or invent conditions not supported by the classifier output.")
    lines.append("Be thorough, medically precise, and ensure the summary is comprehensive enough "
                 "to answer any question a clinician might ask about this chest X-ray.")

    return "\n".join(lines)


def _strip_markdown(text: str) -> str:
    """Remove markdown formatting characters (* # `) from LLM output."""
    import re
    # Remove bold/italic markers (**text**, *text*, __text__, _text_)
    text = re.sub(r'\*{1,3}(.*?)\*{1,3}', r'\1', text)
    text = re.sub(r'_{1,3}(.*?)_{1,3}', r'\1', text)
    # Remove leading # heading markers
    text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)
    # Replace bullet * or - at line start with a dash
    text = re.sub(r'^\s*[\*\-]\s+', '  - ', text, flags=re.MULTILINE)
    # Remove any remaining stray asterisks or backticks
    text = re.sub(r'[`\*]', '', text)
    return text


def _call_azure_llm(prompt: str) -> Optional[str]:
    """Call Azure OpenAI with the given prompt. Returns generated text or None on failure."""
    if _azure_client is None:
        log.warning("Azure OpenAI client not available (missing package or credentials). Falling back to rule-based summary.")
        return None
    try:
        log.info("Calling Azure OpenAI (%s) for clinical summary generation...", _azure_deploy)
        response = _azure_client.chat.completions.create(
            model=_azure_deploy,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert radiologist AI assistant specializing in chest X-ray analysis. "
                        "You produce precise, structured radiology reports using standard clinical language. "
                        "You ALWAYS follow the DETECTED / NOT DETECTED lists exactly — never contradict them. "
                        "You write 'no [condition]' for every condition in the NOT DETECTED list."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=2048,
        )
        text = response.choices[0].message.content
        text = _strip_markdown(text)
        log.info("Azure OpenAI summary generated (%d chars).", len(text))
        return text
    except Exception as exc:
        log.warning("Azure OpenAI call failed: %s. Falling back to rule-based summary.", exc)
        return None


def generate_summary(result: dict) -> dict:
    """
    Stage 4: Combine classification, KG triplets, and retrieval into
    a structured clinical summary that makes any question about the
    image directly answerable.

    Uses Azure OpenAI for natural language generation.
    Falls back to rule-based generation if Azure OpenAI is unavailable.

    Returns a dict with:
      - patient_findings    : natural language paragraph (or full LLM narrative)
      - clinical_context    : per-class KG-derived bullet points
      - retrieval_evidence  : what similar cases show
      - confidence          : overall assessment
      - key_facts           : flat Q&A-ready dict
      - llm_narrative       : full LLM-generated summary (if available)
      - summary_source      : "groq_llm" or "rule_based"
    """
    cls = result["classification"]
    kg  = result["kg_triplets"]
    ret = result["retrieved_reports"]

    resolved   = cls["resolved_classes"]
    probs      = cls["probabilities"]
    kg_classes = cls["kg_classes"]

    # ── 1. Patient Findings Summary (paragraph) ──
    if not resolved:
        findings_paragraph = (
            "No significant abnormalities were detected above the classification "
            "threshold. The chest radiograph appears unremarkable based on the "
            "model's assessment."
        )
    elif resolved == ["normal"]:
        findings_paragraph = (
            "The chest radiograph appears NORMAL (%s confidence). "
            "No significant cardiopulmonary abnormalities are detected. "
            "The lungs are clear, the cardiac silhouette is within normal limits, "
            "and there is no pleural effusion, pneumothorax, or focal consolidation."
            % (_confidence_label(probs.get("normal", 0)))
        )
    else:
        parts = []
        for c in resolved:
            p = probs.get(c, 0.0)
            conf = _confidence_label(p)
            parts.append(f"{c.upper()} ({conf} confidence)")
        findings_paragraph = (
            "This chest radiograph shows the following condition(s): "
            + "; ".join(parts) + ". "
        )
        if cls["conflict_resolution"] == "disease_dominates":
            findings_paragraph += (
                "Normal was also predicted but was overridden because disease "
                "probabilities were stronger (disease-dominant conflict resolution). "
            )
        if cls["is_uncertain"]:
            findings_paragraph += (
                "NOTE: The result carries some uncertainty — " + cls["uncertainty_reason"] + " "
            )

    # ── 2. Clinical Context from KG ──
    kg_context = _extract_kg_context_per_class(kg["per_class_traversal"])

    clinical_context: Dict[str, dict] = {}
    for cls_name, buckets in kg_context.items():
        summary_parts = []
        if buckets["findings"]:
            summary_parts.append(
                "Associated findings: " + ", ".join(buckets["findings"][:6])
            )
        if buckets["suggestions"]:
            summary_parts.append(
                "Suggests: " + ", ".join(buckets["suggestions"][:4])
            )
        if buckets["confirmations"]:
            summary_parts.append(
                "Can be confirmed by: " + ", ".join(buckets["confirmations"][:3])
            )
        if buckets["locations"]:
            summary_parts.append(
                "Located in: " + ", ".join(buckets["locations"][:3])
            )
        if buckets["contradictions"]:
            summary_parts.append(
                "Contradicts/rules out: " + ", ".join(buckets["contradictions"][:5])
            )
        if buckets["requirements"]:
            summary_parts.append(
                "Requires: " + ", ".join(buckets["requirements"][:3])
            )
        clinical_context[cls_name] = {
            "summary_bullets": summary_parts,
            "raw_buckets":     buckets,
        }

    # ── 3. Retrieval Evidence ──
    retrieval_evidence = _extract_retrieval_evidence(ret)

    retrieval_summary_parts = []
    if retrieval_evidence["common_findings"]:
        for term, count in retrieval_evidence["common_findings"]:
            retrieval_summary_parts.append(
                f"{count}/{retrieval_evidence['total_reports']} retrieved reports mention '{term}'"
            )
    retrieval_narrative = (
        f"Among the top {retrieval_evidence['total_reports']} most similar cases "
        f"(avg similarity: {retrieval_evidence['avg_similarity']:.4f}): "
    )
    if retrieval_summary_parts:
        retrieval_narrative += "; ".join(retrieval_summary_parts) + "."
    else:
        retrieval_narrative += "no single clinical term was dominant across reports."

    # ── 4. Confidence Assessment ──
    if not resolved:
        overall_confidence = "indeterminate"
        confidence_note = "No class exceeded the threshold."
    else:
        max_prob = max(probs.get(c, 0) for c in resolved)
        overall_confidence = _confidence_label(max_prob)
        n_views = 0
        if "frontal" in cls["fusion_mode"].lower():
            n_views += 1
        if "lateral" in cls["fusion_mode"].lower():
            n_views += 1
        if n_views == 0:
            n_views = 1  # at least one view was used
        confidence_note = (
            f"Overall confidence: {overall_confidence} "
            f"(strongest prediction: {max_prob:.3f}). "
            f"Based on {n_views} view(s) with {cls['fusion_mode']} fusion. "
        )
        if cls["conflict_resolution"] != "no_conflict":
            confidence_note += f"Conflict resolution applied: {cls['conflict_resolution']}. "
        if cls["is_uncertain"]:
            confidence_note += "FLAGGED AS UNCERTAIN. "

    # ── 5. Key Facts (Q&A-ready) ──
    conditions_ruled_out = []
    for cls_name, buckets in kg_context.items():
        conditions_ruled_out.extend(buckets["contradictions"])
    # Deduplicate
    seen_ruled = set()
    ruled_out_unique = []
    for c in conditions_ruled_out:
        cl = c.lower()
        if cl not in seen_ruled:
            seen_ruled.add(cl)
            ruled_out_unique.append(c)

    most_likely = resolved[0] if resolved else "none"
    most_likely_prob = probs.get(most_likely, 0.0) if resolved else 0.0

    supporting_findings = []
    for cls_name in kg_classes:
        if cls_name in kg_context and kg_context[cls_name]["findings"]:
            supporting_findings.extend(kg_context[cls_name]["findings"][:3])

    key_facts = {
        "what_conditions_detected":        resolved if resolved else ["none"],
        "most_likely_diagnosis":           most_likely,
        "most_likely_probability":         round(most_likely_prob, 4),
        "all_kg_classes":                  kg_classes,
        "supporting_findings_from_kg":     supporting_findings[:8],
        "conditions_ruled_out":            ruled_out_unique[:10],
        "similar_cases_avg_similarity":    retrieval_evidence["avg_similarity"],
        "common_terms_in_similar_cases":   [
            t for t, _ in retrieval_evidence["common_findings"]
        ],
        "is_normal":                       resolved == ["normal"],
        "is_uncertain":                    cls["is_uncertain"],
        "number_of_conditions":            len(resolved),
        "conflict_resolution":             cls["conflict_resolution"],
        "fusion_mode":                     cls["fusion_mode"],
    }

    # ── LLM Summary Generation ──
    llm_narrative: Optional[str] = None
    summary_source = "rule_based"
    try:
        llm_prompt = _build_llm_prompt(result)
        llm_narrative = _call_azure_llm(llm_prompt)
        if llm_narrative:
            summary_source = "azure_openai"
    except Exception as exc:
        log.warning("LLM summary generation failed: %s", exc)

    return {
        "patient_findings":    llm_narrative if llm_narrative else findings_paragraph,
        "clinical_context":    clinical_context,
        "retrieval_evidence":  retrieval_narrative,
        "confidence":          {
            "level": overall_confidence,
            "detail": confidence_note,
        },
        "key_facts":           key_facts,
        "llm_narrative":       llm_narrative,
        "summary_source":      summary_source,
    }


# ══════════════════════════════════════════════════════════════════════
#  STAGE 5 — PRECISE SUMMARY (for end-to-end accuracy evaluation)
# ══════════════════════════════════════════════════════════════════════

def _build_precise_summary_prompt(result: dict) -> str:
    """
    Build a concise prompt that instructs the LLM to produce a precise
    radiology-style findings + impression, matching the format used in
    ground-truth reports (for future semantic accuracy evaluation).
    """
    cls     = result["classification"]
    summary = result.get("summary", {})
    kg      = result["kg_triplets"]
    ret     = result["retrieved_reports"]

    resolved   = cls["resolved_classes"]
    probs      = cls["probabilities"]
    kg_classes = cls["kg_classes"]

    # Confidence threshold: include all resolved conditions above 0.50
    HIGH_CONF_THRESHOLD = 0.58
    high_conf = [c for c in resolved if probs.get(c, 0) >= HIGH_CONF_THRESHOLD]
    low_conf  = [c for c in resolved if probs.get(c, 0) <  HIGH_CONF_THRESHOLD]

    # ── Build EXPLICIT NEGATIVE list ──────────────────────────────────
    # Conditions that were NOT detected → must be written as "no X" in FINDINGS
    # to match ground-truth report style.
    STANDARD_NEGATABLE: List[tuple] = [
        ("pneumothorax",   "pneumothorax"),
        ("pleural effusion","pleural effusion"),
        ("effusion",        "pleural effusion"),
        ("consolidation",   "consolidation"),
        ("edema",           "pulmonary edema"),
        ("cardiomegaly",    "cardiomegaly"),
        ("mass",            "mass"),
        ("nodule",          "discrete nodule"),
    ]
    detected_lower = {c.strip().lower() for c in resolved}
    seen_neg: set = set()
    explicit_negatives: List[str] = []
    for raw_cls, display_name in STANDARD_NEGATABLE:
        if raw_cls not in detected_lower and display_name not in seen_neg:
            seen_neg.add(display_name)
            explicit_negatives.append(display_name)

    # ── Collect KG-derived supporting facts ──────────────────────────
    kg_facts: List[str] = []
    kg_locations: List[str] = []
    for cls_name in kg_classes:
        trav = kg["per_class_traversal"].get(cls_name, {})
        for t in trav.get("triplets", [])[:10]:
            if t["p"] in ("strongly_suggests", "is_finding_of", "confirmed_by",
                          "suggests", "is_symptom_of"):
                obj = t["o"].strip()
                if obj and obj not in kg_facts:
                    kg_facts.append(obj)
            elif t["p"] == "located_in":
                loc = t["o"].strip()
                if loc and loc not in kg_locations:
                    kg_locations.append(loc)

    # ── Condition strings for prompt ─────────────────────────────────
    if high_conf:
        conditions_str = ", ".join(
            f"{c} ({_confidence_label(probs.get(c, 0))} confidence)" for c in high_conf
        )
    elif resolved:
        conditions_str = ", ".join(f"{c} (low confidence)" for c in resolved)
    else:
        conditions_str = "NONE — write a normal/unremarkable chest X-ray report"

    # ── Top retrieved reports (full text, up to 500 chars each) ──────
    retrieved_texts = [
        r.get("caption", "").strip() for r in ret[:3] if r.get("caption", "").strip()
    ]

    # ── Stage 4 summary excerpt ───────────────────────────────────────
    stage4_context = ""
    if summary.get("llm_narrative"):
        stage4_context = summary["llm_narrative"][:500].strip()

    lines = [
        "You are a radiologist writing an Indiana University chest X-ray report.",
        "Produce ONLY two short sections, exactly labeled, with no other output:",
        "",
        "FINDINGS: <1-3 sentences: describe what is observed on the chest X-ray,",
        "           including heart size, lung fields, pleural spaces, mediastinum,",
        "           bones, and any specific abnormalities>",
        "IMPRESSION: <1 sentence: the primary radiological conclusion>",
        "",
        "══ CRITICAL RULES (violations cause dangerous clinical errors) ══",
        "1. POSITIVE RULE: Every condition listed under 'DETECTED CONDITIONS' MUST be",
        "   described as PRESENT in FINDINGS (e.g., 'cardiomegaly is present').",
        "2. NEGATIVE RULE: Every condition listed under 'NOT DETECTED' MUST appear as",
        "   'no [condition]' in FINDINGS (e.g., 'no pneumothorax', 'no pleural effusion').",
        "3. CONSISTENCY: Do NOT invent or assert any condition that is not in either list.",
        "4. LATERALITY: Only specify left/right if KG facts explicitly state a location.",
        "   Do NOT guess laterality.",
        "5. FORMAT: Plain clinical language only. No bullets, asterisks, bold, markdown.",
        "   No headers other than 'FINDINGS:' and 'IMPRESSION:'.",
        "6. STYLE: Match the sentence structure and tone of the REFERENCE REPORTS below.",
        "",
        "═══ DETECTED CONDITIONS (assert as PRESENT) ═══",
        conditions_str,
    ]

    if low_conf:
        lines.append(f"Low-confidence (mention cautiously, only if KG supports): "
                     + ", ".join(low_conf))

    if cls.get("uncertainty_reason"):
        lines.append(f"Uncertainty note: {cls['uncertainty_reason']}")

    lines.append("")
    lines.append("═══ NOT DETECTED (write 'no X' for each in FINDINGS) ═══")
    if explicit_negatives:
        lines.append(", ".join(explicit_negatives))
    else:
        lines.append("None (all standard conditions are detected)")

    if kg_facts:
        lines.append("")
        lines.append("═══ KG-DERIVED RADIOLOGICAL EVIDENCE (use as supporting details) ═══")
        lines.append("; ".join(kg_facts[:12]))

    if kg_locations:
        lines.append(f"Anatomical locations: {', '.join(kg_locations[:5])}")

    if retrieved_texts:
        lines.append("")
        lines.append("═══ REFERENCE REPORTS — MATCH THIS STYLE EXACTLY ═══")
        for i, txt in enumerate(retrieved_texts, 1):
            lines.append(f"Reference {i}: {txt[:500]}")

    if stage4_context:
        lines.append("")
        lines.append("═══ ADDITIONAL CLINICAL CONTEXT ═══")
        lines.append(stage4_context)

    lines.append("")
    lines.append("Write ONLY the FINDINGS and IMPRESSION lines. No other text.")

    return "\n".join(lines)


def _check_llm_consistency(
    findings_text: str,
    impression_text: str,
    resolved: List[str],
    probs: Dict[str, float],
) -> bool:
    """
    Quick consistency check: verify LLM didn't negate a detected condition
    or assert a clearly absent condition.
    Returns True if output appears consistent, False if a critical mismatch found.
    """
    import re
    combined = (findings_text + " " + impression_text).lower()

    # Negation cue patterns (window: any of these followed by a condition)
    NEG_PATTERNS = [
        r'\bno\s+{cond}\b',
        r'\bwithout\s+{cond}\b',
        r'\bno\s+evidence\s+of\s+{cond}\b',
        r'\babsent\s+{cond}\b',
        r'\bnegative\s+for\s+{cond}\b',
    ]
    # Surface forms for key conditions
    SURFACE: Dict[str, List[str]] = {
        "cardiomegaly":   ["cardiomegaly", "enlarged heart", "enlarged cardiac"],
        "pleural effusion": ["pleural effusion", "pleural fluid"],
        "effusion":       ["effusion"],
        "pneumothorax":   ["pneumothorax"],
        "consolidation":  ["consolidation"],
        "edema":          ["pulmonary edema", "edema"],
        "mass":           ["mass"],
        "nodule":         ["nodule"],
    }
    detected_lower = {c.strip().lower() for c in resolved}
    for raw_cls, surfaces in SURFACE.items():
        if raw_cls in detected_lower:
            # This condition WAS detected — LLM should NOT negate it
            for surface in surfaces:
                for pat in NEG_PATTERNS:
                    if re.search(pat.format(cond=re.escape(surface)), combined):
                        log.warning(
                            "Consistency check: LLM negated detected condition '%s' "
                            "(surface='%s'). Will use rule-based fallback.", raw_cls, surface
                        )
                        return False
    return True


def run_precise_summary(result: dict) -> dict:
    """
    Stage 5: Generate a precise radiology-style findings + impression summary
    using Groq LLM.
    Returns:
        {
            "findings"  : concise radiological findings sentence(s),
            "impression": one-line clinical conclusion,
            "full_text" : combined "FINDINGS: ... IMPRESSION: ..." string,
            "source"    : "groq_llm" or "rule_based",
        }
    """
    log.info("Generating Stage 5 precise summary...")

    # ── Try LLM ──
    llm_text: Optional[str] = None
    try:
        prompt = _build_precise_summary_prompt(result)
        llm_text = _call_azure_llm(prompt)
    except Exception as exc:
        log.warning("Stage 5 LLM call failed: %s", exc)

    if llm_text:
        llm_text = _strip_markdown(llm_text).strip()
        # Parse FINDINGS / IMPRESSION from LLM response
        findings_text = ""
        impression_text = ""
        for line in llm_text.splitlines():
            line = line.strip()
            if line.upper().startswith("FINDINGS:"):
                findings_text = line[len("FINDINGS:"):].strip()
            elif line.upper().startswith("IMPRESSION:"):
                impression_text = line[len("IMPRESSION:"):].strip()
        # Fallback: if parsing fails, use full LLM text as findings
        if not findings_text and not impression_text:
            findings_text = llm_text
            impression_text = ""

        # ── Consistency check: reject if LLM negated a detected condition ──
        cls_r  = result["classification"]
        passed = _check_llm_consistency(
            findings_text, impression_text,
            cls_r["resolved_classes"], cls_r["probabilities"],
        )
        if passed:
            full_text = f"FINDINGS: {findings_text}\nIMPRESSION: {impression_text}"
            return {
                "findings":   findings_text,
                "impression": impression_text,
                "full_text":  full_text,
                "source":     "azure_openai",
            }
        else:
            log.warning("Stage 5: LLM output failed consistency check; using rule-based fallback.")
            llm_text = None  # fall through to rule-based

    # ── Rule-based fallback ──
    cls        = result["classification"]
    resolved   = cls["resolved_classes"]
    probs      = cls["probabilities"]
    kf         = result.get("summary", {}).get("key_facts", {})
    kg         = result["kg_triplets"]

    # Build explicit negatives list (same logic as LLM path)
    STANDARD_NEGATABLE_FB: List[tuple] = [
        ("pneumothorax",    "pneumothorax"),
        ("pleural effusion","pleural effusion"),
        ("effusion",        "pleural effusion"),
        ("consolidation",   "consolidation"),
        ("edema",           "pulmonary edema"),
        ("cardiomegaly",    "cardiomegaly"),
        ("mass",            "mass"),
        ("nodule",          "discrete nodule"),
    ]
    detected_lower_fb = {c.strip().lower() for c in resolved}
    seen_neg_fb: set = set()
    neg_phrases_fb: List[str] = []
    for raw_cls_fb, display_name_fb in STANDARD_NEGATABLE_FB:
        if raw_cls_fb not in detected_lower_fb and display_name_fb not in seen_neg_fb:
            seen_neg_fb.add(display_name_fb)
            neg_phrases_fb.append(f"no {display_name_fb}")
    neg_str_fb = (". ".join(neg_phrases_fb) + ".") if neg_phrases_fb else ""

    if not resolved or resolved == []:
        findings_text   = (
            "The lungs are clear. The cardiac silhouette and mediastinum are within normal limits. "
            + (neg_str_fb if neg_str_fb else "No pleural effusion or pneumothorax identified.")
        )
        impression_text = "No acute cardiopulmonary findings."
    elif resolved == ["normal"]:
        findings_text   = (
            "The lungs are clear. The cardiac silhouette and mediastinum are within normal limits. "
            + (neg_str_fb if neg_str_fb else "No pleural effusion, pneumothorax, or focal consolidation identified.")
        )
        impression_text = "No acute cardiopulmonary findings."
    else:
        HIGH_CONF_THRESHOLD = 0.58
        high_conf_fb = [c for c in resolved if probs.get(c, 0) >= HIGH_CONF_THRESHOLD]
        low_conf_fb  = [c for c in resolved if probs.get(c, 0) <  HIGH_CONF_THRESHOLD]

        # Collect KG-derived findings (no raw class names or probabilities)
        kg_facts_fb: List[str] = []
        for cls_name in cls["kg_classes"]:
            trav = kg["per_class_traversal"].get(cls_name, {})
            for t in trav.get("triplets", [])[:5]:
                if t["p"] in ("is_finding_of", "strongly_suggests", "confirmed_by",
                              "suggests", "is_symptom_of"):
                    obj = t["o"].strip()
                    if obj and obj not in kg_facts_fb:
                        kg_facts_fb.append(obj)

        if not high_conf_fb and low_conf_fb:
            # All predictions are low-confidence → write normal with explicit negatives
            findings_text   = (
                "The chest radiograph is within normal limits. "
                + (neg_str_fb if neg_str_fb else "No definite acute cardiopulmonary abnormality is identified.")
                + " Correlation with clinical findings is recommended."
            )
            impression_text = "No acute cardiopulmonary findings. Clinical correlation recommended."
        elif high_conf_fb:
            # Use clean class titles — no prob= values
            cond_names = [c.title() for c in high_conf_fb]
            if kg_facts_fb:
                facts_str = "; ".join(kg_facts_fb[:4])
                findings_text = (
                    f"The chest radiograph demonstrates {', '.join(cond_names)}. "
                    f"{facts_str.capitalize()}. "
                    + (neg_str_fb if neg_str_fb else "")
                ).strip()
            else:
                findings_text = (
                    f"The chest radiograph demonstrates {', '.join(cond_names)}. "
                    + (neg_str_fb if neg_str_fb else "")
                ).strip()
            impression_text = f"{', '.join(cond_names[:3])}."
        else:
            findings_text   = (
                "No significant radiological abnormality identified. "
                + (neg_str_fb if neg_str_fb else "")
            ).strip()
            impression_text = "Unremarkable chest radiograph."

    full_text = f"FINDINGS: {findings_text}\nIMPRESSION: {impression_text}"
    return {
        "findings":   findings_text,
        "impression": impression_text,
        "full_text":  full_text,
        "source":     "rule_based",
    }


def run_pipeline(
    frontal_path: Optional[Path],
    lateral_path: Optional[Path],
    device: torch.device,
    threshold: float = 0.5,
    frontal_weight: float = 0.65,
    top_k: int = 5,
    top_k_triplets: int = 20,
) -> dict:
    """Run the complete 3-stage pipeline and return a unified result dict."""
    t_start = time.time()

    # ── Stage 1: Observation Classification ──
    log.info("=" * 60)
    log.info("STAGE 1: OBSERVATION CLASSIFICATION")
    log.info("=" * 60)
    cls_result = run_classification(
        frontal_path, lateral_path, device, threshold, frontal_weight
    )

    # ── Stage 2: KG Triplets ──
    log.info("=" * 60)
    log.info("STAGE 2: KNOWLEDGE GRAPH TRAVERSAL")
    log.info("=" * 60)
    kg_result = run_kg_traversal(
        cls_result["kg_classes"], top_k_per_class=top_k_triplets
    )

    # ── Stage 3: Retrieval ──
    log.info("=" * 60)
    log.info("STAGE 3: REPORT RETRIEVAL")
    log.info("=" * 60)
    retrieved = run_retrieval(frontal_path, lateral_path, device, top_k)

    # ── Stage 4: Summary Generation ──
    log.info("=" * 60)
    log.info("STAGE 4: CLINICAL SUMMARY GENERATION")
    log.info("=" * 60)
    intermediate = {
        "classification":    cls_result,
        "kg_triplets":       kg_result,
        "retrieved_reports": retrieved,
    }
    summary = generate_summary(intermediate)

    # ── Stage 5: Precise Summary ──
    log.info("=" * 60)
    log.info("STAGE 5: PRECISE SUMMARY GENERATION")
    log.info("=" * 60)
    intermediate["summary"] = summary
    precise = run_precise_summary(intermediate)

    elapsed = round(time.time() - t_start, 2)

    return {
        "classification":    cls_result,
        "kg_triplets":       kg_result,
        "retrieved_reports": retrieved,
        "summary":           summary,
        "precise_summary":   precise,
        "pipeline_time_s":   elapsed,
    }

def _build_report_text(
    result: dict,
    frontal_path: Optional[Path],
    lateral_path: Optional[Path],
) -> str:
    """Build a human-readable text report string."""
    from datetime import datetime

    cls = result["classification"]
    kg  = result["kg_triplets"]
    ret = result["retrieved_reports"]

    lines: List[str] = []
    w = lines.append

    w("=" * 70)
    w("       CHEST X-RAY ANALYSIS — END-TO-END PIPELINE RESULTS")
    w("=" * 70)
    w(f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ── Input ──
    w("")
    w("INPUT IMAGES")
    w(f"   Frontal : {frontal_path or 'Not provided'}")
    w(f"   Lateral : {lateral_path or 'Not provided'}")
    w(f"   Fusion  : {cls['fusion_mode']}")

    # ── Stage 1 ──
    w("")
    w("-" * 70)
    w("STAGE 1: PREDICTED OBSERVATIONS")
    w("-" * 70)
    if cls["resolved_classes"]:
        for i, c in enumerate(cls["resolved_classes"], 1):
            prob = cls["probabilities"].get(c, 0.0)
            kg_name = LABEL_NORMALIZE_MAP.get(c.lower(), c)
            w(f"   {i}. {c:<30s}  (prob={prob:.4f})  ->  KG: {kg_name}")
    else:
        w("   No classes predicted above threshold.")

    if cls["conflict_resolution"] != "no_conflict":
        w(f"")
        w(f"   Conflict resolution: {cls['conflict_resolution']}")
    if cls["is_uncertain"]:
        w(f"   Uncertain: {cls['uncertainty_reason']}")

    w("")
    w("   Top-5 class probabilities:")
    for name, prob in cls["top5_probs"]:
        bar = "#" * int(prob * 30)
        w(f"     {name:<30s} {prob:.4f}  {bar}")

    # ── Stage 2 ──
    w("")
    w("-" * 70)
    w("STAGE 2: KNOWLEDGE GRAPH TRIPLETS")
    w("-" * 70)
    for cls_name, traversal in kg["per_class_traversal"].items():
        n = traversal["triplet_count"]
        w(f"")
        w(f"   [{cls_name}]  ({n} triplets, seeds found: {len(traversal['seeds_used'])})")
        w(f"   Seeds used   : {', '.join(traversal['seeds_used']) or 'none'}")
        w(f"   Seeds missing: {', '.join(traversal['seeds_missing']) or 'none'}")
        w("")
        for idx, t in enumerate(traversal["triplets"], 1):
            w(f"      {idx:>3d}. {t['s']}  --{t['p']}-->  {t['o']}")

    w("")
    w(f"   Merged total: {kg['merged_count']} unique triplets across all classes")

    # ── Stage 3 ──
    w("")
    w("-" * 70)
    w(f"STAGE 3: TOP-{len(ret)} RETRIEVED REPORTS")
    w("-" * 70)
    for r in ret:
        w(f"")
        w(f"   RANK {r['rank']}  |  Score: {r['similarity']:.4f}  |  UID: {r['uid']}")
        w(f"   Report: {r['caption']}")

    # ── Stage 4 ──
    summary = result.get("summary", {})
    if summary:
        w("")
        w("-" * 70)
        w("STAGE 4: CLINICAL SUMMARY")
        w("-" * 70)
        w("")

        src = summary.get('summary_source', 'rule_based')
        narrative = summary.get('llm_narrative') or summary.get('patient_findings', '')

        if src == 'groq_llm' and narrative:
            # Pretty-print the LLM narrative, indented
            for line in narrative.splitlines():
                w(f"   {line}")
        else:
            # Rule-based fallback — structured but human-readable
            w(f"   {summary.get('patient_findings', '')}")
            w("")
            conf = summary.get("confidence", {})
            w(f"   Confidence: {conf.get('level', 'N/A').upper()} — {conf.get('detail', '')}")
            w("")
            kf = summary.get("key_facts", {})
            w(f"   Most likely diagnosis : {kf.get('most_likely_diagnosis', 'none')} (prob={kf.get('most_likely_probability', 0):.4f})")
            findings_list = kf.get('supporting_findings_from_kg', [])
            w(f"   Supporting findings   : {', '.join(findings_list) if findings_list else 'N/A'}")
            ruled = kf.get('conditions_ruled_out', [])
            w(f"   Conditions ruled out  : {', '.join(ruled[:6]) if ruled else 'None'}")
            w(f"   Normal X-ray?         : {'Yes' if kf.get('is_normal') else 'No'}")
            w(f"   Uncertain?            : {'Yes' if kf.get('is_uncertain') else 'No'}")

    w("")
    w("=" * 70)
    w(f"  Total pipeline time: {result['pipeline_time_s']}s")
    w("=" * 70)

    # ── Stage 5: Precise Summary ──
    precise = result.get("precise_summary", {})
    if precise:
        w("")
        w("=" * 70)
        w("STAGE 5: PRECISE SUMMARY")
        w("=" * 70)
        w(f"   Source  : {precise.get('source', 'N/A').upper()}")
        w("")
        w(f"   FINDINGS: {precise.get('findings', 'N/A')}")
        w("")
        w(f"   IMPRESSION: {precise.get('impression', 'N/A')}")
        w("")

    return "\n".join(lines)


def display_results(result: dict, frontal_path: Optional[Path], lateral_path: Optional[Path]) -> None:
    """Pretty-print the full pipeline output to terminal."""

    cls = result["classification"]
    kg  = result["kg_triplets"]
    ret = result["retrieved_reports"]

    print("\n")
    print("=" * 70)
    print("       CHEST X-RAY ANALYSIS — END-TO-END PIPELINE RESULTS")
    print("=" * 70)

    # Input info
    print("\n INPUT IMAGES")
    print(f"   Frontal : {frontal_path or 'Not provided'}")
    print(f"   Lateral : {lateral_path or 'Not provided'}")
    print(f"   Fusion  : {cls['fusion_mode']}")

    # Classification
    print("\n" + "─" * 70)
    print(" STAGE 1: PREDICTED OBSERVATIONS")
    print("─" * 70)
    if cls["resolved_classes"]:
        for i, c in enumerate(cls["resolved_classes"], 1):
            prob = cls["probabilities"].get(c, 0.0)
            kg_name = LABEL_NORMALIZE_MAP.get(c.lower(), c)
            print(f"   {i}. {c:<30s}  (prob={prob:.4f})  →  KG: {kg_name}")
    else:
        print("   No classes predicted above threshold.")

    if cls["conflict_resolution"] != "no_conflict":
        print(f"\n   ⚖️  Conflict resolution: {cls['conflict_resolution']}")
    if cls["is_uncertain"]:
        print(f"    Uncertain: {cls['uncertainty_reason']}")

    print(f"\n   Top-5 class probabilities:")
    for name, prob in cls["top5_probs"]:
        bar = "█" * int(prob * 30)
        print(f"     {name:<30s} {prob:.4f}  {bar}")

    # KG Triplets
    print("\n" + "─" * 70)
    print("STAGE 2: KNOWLEDGE GRAPH TRIPLETS")
    print("─" * 70)
    for cls_name, traversal in kg["per_class_traversal"].items():
        n = traversal["triplet_count"]
        print(f"\n    {cls_name}  ({n} triplets, seeds found: {len(traversal['seeds_used'])})")
        for t in traversal["triplets"][:8]:
            print(f"      {t['s']}  ──{t['p']}──▶  {t['o']}")
        if n > 8:
            print(f"      ... and {n - 8} more")

    print(f"\n   Merged total: {kg['merged_count']} unique triplets across all classes")

    # Retrieved reports
    print("\n" + "─" * 70)
    print(f" STAGE 3: TOP-{len(ret)} RETRIEVED REPORTS")
    print("─" * 70)
    for r in ret:
        print(f"\n   RANK {r['rank']}  |  Score: {r['similarity']:.4f}  |  UID: {r['uid']}")
        caption = r["caption"]
        # Wrap long captions
        if len(caption) > 100:
            caption = caption[:100] + "..."
        print(f"   Report: {caption}")

    # Stage 4: Summary
    summary = result.get("summary", {})
    if summary:
        print("\n" + "─" * 70)
        print("STAGE 4: CLINICAL SUMMARY")
        print("─" * 70)

        src = summary.get('summary_source', 'rule_based')
        narrative = summary.get('llm_narrative') or summary.get('patient_findings', '')

        if src == 'groq_llm' and narrative:
            print(f"\n   [Generated by Groq LLM — llama-3.3-70b-versatile]\n")
            # Print each line of the LLM narrative with indentation
            for line in narrative.splitlines():
                print(f"   {line}")
        else:
            print("\n   [Rule-based summary (Groq LLM unavailable)]\n")
            print(f"   {summary.get('patient_findings', '')}")
            print()
            conf = summary.get("confidence", {})
            print(f"   Confidence : {conf.get('level', 'N/A').upper()} — {conf.get('detail', '')}")
            print()
            kf = summary.get("key_facts", {})
            print(f"   Most likely diagnosis : {kf.get('most_likely_diagnosis', 'none')} (prob={kf.get('most_likely_probability', 0):.4f})")
            findings_list = kf.get('supporting_findings_from_kg', [])
            print(f"   Supporting findings   : {', '.join(findings_list) if findings_list else 'N/A'}")
            ruled = kf.get('conditions_ruled_out', [])
            print(f"   Conditions ruled out  : {', '.join(ruled[:6]) if ruled else 'None'}")
            print(f"   Normal X-ray?         : {'Yes' if kf.get('is_normal') else 'No'}")
            print(f"   Uncertain?            : {'Yes' if kf.get('is_uncertain') else 'No'}")

    print("\n" + "=" * 70)
    print(f" Total pipeline time: {result['pipeline_time_s']}s")
    print("=" * 70)

    # Stage 5: Precise Summary
    precise = result.get("precise_summary", {})
    if precise:
        print("\n" + "=" * 70)
        print("STAGE 5: PRECISE SUMMARY  (for accuracy evaluation)")
        print("=" * 70)
        print(f"   Source  : {precise.get('source', 'N/A').upper()}\n")
        print(f"   FINDINGS: {precise.get('findings', 'N/A')}")
        print(f"\n   IMPRESSION: {precise.get('impression', 'N/A')}")
        print()

    print()


def save_results(
    result: dict,
    frontal_path: Optional[Path],
    lateral_path: Optional[Path],
    output_dir: Path,
) -> Tuple[Path, Path]:
    """
    Save pipeline results to output_dir:
      - pipeline_result.txt   (human-readable full report)
      - pipeline_result.json  (machine-readable, all data)
    Returns (txt_path, json_path).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Text report ──
    txt_path = output_dir / "pipeline_result.txt"
    report_text = _build_report_text(result, frontal_path, lateral_path)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    # ── JSON (full data) ──
    json_path = output_dir / "pipeline_result.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)

    return txt_path, json_path


# ══════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "End-to-End Chest X-Ray Analysis Pipeline.\n"
            "Given frontal (required) and optional lateral image:\n"
            "  1. Predicts disease classes (BiomedCLIP classifier)\n"
            "  2. Retrieves KG triplets per class (leaf-node DFS)\n"
            "  3. Retrieves top-K similar reports (hybrid CLIP retriever)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--frontal", required=True, type=str,
                   help="Path to frontal chest X-ray image (required)")
    p.add_argument("--lateral", default=None, type=str,
                   help="Path to lateral chest X-ray image (optional)")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Classification threshold (default: 0.5)")
    p.add_argument("--frontal_weight", type=float, default=0.65,
                   help="Frontal weight for dual-view fusion (default: 0.65)")
    p.add_argument("--top_k", type=int, default=5,
                   help="Number of reports to retrieve (default: 5)")
    p.add_argument("--top_k_triplets", type=int, default=20,
                   help="Max triplets per class from KG (default: 20)")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"],
                   help="Compute device (default: auto)")
    p.add_argument("--output_dir", default=None, type=str,
                   help="Directory to save results (default: output/pipeline_results/)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    log.info("Device: %s", device)

    frontal_path = Path(args.frontal)
    lateral_path = Path(args.lateral) if args.lateral else None

    if not frontal_path.exists():
        sys.exit(f"[ERROR] Frontal image not found: {frontal_path}")
    if lateral_path and not lateral_path.exists():
        sys.exit(f"[ERROR] Lateral image not found: {lateral_path}")

    # Run full pipeline
    result = run_pipeline(
        frontal_path=frontal_path,
        lateral_path=lateral_path,
        device=device,
        threshold=args.threshold,
        frontal_weight=args.frontal_weight,
        top_k=args.top_k,
        top_k_triplets=args.top_k_triplets,
    )

    # Display in terminal
    display_results(result, frontal_path, lateral_path)

    # Auto-save results to files
    output_dir = Path(args.output_dir) if args.output_dir else BASE_DIR / "output" / "pipeline_results"
    txt_path, json_path = save_results(result, frontal_path, lateral_path, output_dir)

    log.info("Results saved:")
    log.info("   Text report : %s", txt_path)
    log.info("   JSON data   : %s", json_path)
    print(f"\n Results saved to:\n   {txt_path}\n   {json_path}\n")


if __name__ == "__main__":
    main()
