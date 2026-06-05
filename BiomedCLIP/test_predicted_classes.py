"""
step_A_classifier.py  (DUAL-VIEW REVISION)
===========================================
Stage 1: Load test images + CSV metadata, run BiomedCLIP multi-label
classifier on BOTH Frontal and Lateral views, fuse probabilities per UID,
and save all class probability scores + predicted classes.

Dual-View Feature Fusion
------------------------
For every UID the Indiana dataset provides two projections:
  • Frontal (PA / AP)  — primary view
  • Lateral            — complementary depth / posterior view

The model (BiomedCLIPClassifier) is now trained MULTI-VIEW: it encodes both
views with the BiomedCLIP backbone, L2-normalises each feature vector, and
fuses them by MEAN before the shared classifier head produces the logits.
Inference mirrors training exactly — there is NO probability-averaging
post-processing anymore (the old --fusion_mode / --frontal_weight options are
gone); fusion happens inside the model at the feature level.

If ONLY one view exists for a UID (missing Frontal or Lateral), that view's
feature is used directly — the missing view is masked out, no fusion penalty.

For inspection we still report per-view probabilities: what the model would
predict from each single view alone (the other view masked out).

Multi-label behaviour
---------------------
This is a MULTI-LABEL problem — a patient (UID) can have zero, one, or
many classes simultaneously (e.g. Cardiomegaly + Pleural Effusion + Edema).
The model produces an independent sigmoid probability in [0,1] per class.

What we save (one row per UID — not per image)
----------------------------------------------
  uid
  frontal_image, lateral_image    : filenames used (or "" if missing)
  prob_<class> x 35               : fused sigmoid probability for each class
  frontal_prob_<class> x 35       : per-view probabilities (raw)
  lateral_prob_<class> x 35       : per-view probabilities (raw)
  predicted_classes               : classes with fused prob >= threshold
  resolved_classes                : after Normal/Disease conflict resolution
  kg_classes                      : KG-normalised names for step_B
  conflict_resolution, is_uncertain, uncertainty_reason
  num_predicted, fusion_mode, inference_time_s

DAG compliance note
-------------------
The kg_classes column in the output is the set of seed nodes for the
Knowledge Graph traversal in step_B.  The KG itself must be a DAG (no
directed cycles).  By feeding ONLY the fused, resolved predictions as
seeds, we guarantee that the traversal starts from valid leaf nodes in
the DAG and can only proceed toward ancestor/consequence nodes.

Usage
-----
  python step_A_classifier.py \\
      --test_folder   testing/test1 \\
      --projections   indiana_projections.csv \\
      --checkpoint    epoch_search_e100_fold2_new.pth \\
      --threshold     0.5 \\
      --output_dir    output/predictions \\
      --device        auto

  Revision: multi-view feature fusion (matches multi-view training)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

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
logger = logging.getLogger(__name__)


# ===========================================================================
# Constants — must match training exactly
# ===========================================================================
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

# ---------------------------------------------------------------------------
# Class label -> KG name mapping
# ---------------------------------------------------------------------------
_LABEL_SYNONYM_OVERRIDES: Dict[str, str] = {
    "degenerative disease": "Degenerative Change",
    "cicatrix":             "Fibrosis",
    "hypoinflation":        "Volume Loss",
    "lung/hyperdistention": "Hyperinflation",
    "cardiac shadow":       "Cardiac Shadow (abnormal)",
}


def _build_label_normalize_map(label_columns: List[str]) -> Dict[str, str]:
    STOP_WORDS = {"of", "the", "a", "an", "and", "or", "in", "on",
                  "at", "to", "for", "with", "by"}
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

# Normal/Disease conflict resolution thresholds
NORMAL_DOMINANCE_THRESHOLD = 0.70
NORMAL_OVERRIDE_DISEASE    = 0.42
UNCERTAIN_UPPER            = 0.73
UNCERTAIN_DISEASE_MIN      = 0.32

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


# ===========================================================================
# Model (must exactly match training architecture)
# ===========================================================================
class BiomedCLIPClassifier(nn.Module):
    def __init__(self, num_classes: int, unfreeze_last_n_blocks: int = 4):
        super().__init__()
        model_name = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        self.model, self.preprocess_train, self.preprocess_val = \
            open_clip.create_model_and_transforms(model_name)

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

    def forward(
        self,
        frontal: torch.Tensor,
        lateral: torch.Tensor,
        frontal_mask: torch.Tensor,
        lateral_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Multi-view forward: encode both views, fuse features by masked mean.

        frontal_mask / lateral_mask are (B,) float tensors (1=present, 0=missing),
        so a UID with a single available view contributes only that view.
        Mirrors BiomedCLIPClassifier.forward in train_jaccard_new.py.
        """
        feat_f = F.normalize(self.model.encode_image(frontal), dim=-1)
        feat_l = F.normalize(self.model.encode_image(lateral), dim=-1)

        mf = frontal_mask.unsqueeze(1)
        ml = lateral_mask.unsqueeze(1)
        fused = feat_f * mf + feat_l * ml
        denom = (mf + ml).clamp(min=1.0)
        fused = fused / denom

        return self.classifier(fused)

    @torch.no_grad()
    def encode(self, image: torch.Tensor) -> torch.Tensor:
        """L2-normalised image features for one batch of images (1 view)."""
        return F.normalize(self.model.encode_image(image), dim=-1)

    @torch.no_grad()
    def classify_features(self, features: torch.Tensor) -> torch.Tensor:
        """Logits from already-fused feature vector(s)."""
        return self.classifier(features)


def load_model(
    checkpoint_path: Path,
    device: torch.device,
    unfreeze_last_n_blocks: int = 4,
) -> Tuple[BiomedCLIPClassifier, List[str]]:
    if not checkpoint_path.exists():
        raise FileNotFoundError("Checkpoint not found: %s" % checkpoint_path)

    logger.info("Loading checkpoint: %s", checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    label_columns: List[str] = ckpt.get("label_columns", CLASS_LABELS)
    logger.info(
        "  Classes=%d | Epoch=%s | Fold=%s | Val AUC=%.4f",
        len(label_columns),
        ckpt.get("epoch", "N/A"),
        ckpt.get("fold", "N/A"),
        ckpt.get("val_auc", 0.0),
    )

    model = BiomedCLIPClassifier(
        num_classes=len(label_columns),
        unfreeze_last_n_blocks=unfreeze_last_n_blocks,
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device).eval()
    logger.info("  Parameters: %s", "{:,}".format(sum(p.numel() for p in model.parameters())))

    global LABEL_NORMALIZE_MAP
    LABEL_NORMALIZE_MAP = _build_label_normalize_map(label_columns)
    logger.info("  Label->KG mapping rebuilt for %d classes", len(label_columns))

    return model, label_columns


def load_thresholds(
    threshold_csv: Optional[Path],
    label_columns: List[str],
    fallback: float = 0.5,
) -> Dict[str, float]:
    thresholds = {cls: fallback for cls in label_columns}
    if threshold_csv is None or not threshold_csv.exists():
        logger.info("Using uniform threshold = %.3f for all %d classes", fallback, len(label_columns))
        return thresholds
    try:
        with open(threshold_csv, newline="") as fh:
            reader = csv.DictReader(fh)
            loaded = 0
            for row in reader:
                cls = row.get("Class", "").strip()
                thr = row.get("threshold", "").strip()
                if cls in thresholds and thr:
                    thresholds[cls] = float(thr)
                    loaded += 1
        logger.info(
            "Loaded per-class thresholds from %s (%d/%d classes set)",
            threshold_csv, loaded, len(label_columns),
        )
    except Exception as e:
        logger.warning("Could not parse threshold CSV: %s — using fallback %.3f", e, fallback)
    return thresholds


def load_csv_index(
    projections_csv: Path,
) -> Dict[str, dict]:
    """
    Build a UID-keyed index containing BOTH Frontal and Lateral image paths.

    Returns
    -------
    dict: uid -> {
        "uid":           str,
        "frontal_files": list[str]  (filenames, usually 1, occasionally 2),
        "lateral_files": list[str]  (filenames, usually 1),
    }

    Why UID-keyed instead of filename-keyed (old behaviour)
    --------------------------------------------------------
    Old code only indexed Frontal files by stem, so Lateral images in the
    test folder were silently ignored.  The new code groups BOTH views under
    their shared UID, enabling dual-view fusion before thresholding.

    Edge cases handled
    ------------------
    • UID with ONLY Frontal (no Lateral found)  -> lateral_files = []
    • UID with ONLY Lateral (no Frontal found)  -> frontal_files = []
    • UID with MULTIPLE Frontal images          -> all listed; all inferred
    """
    # Build uid -> {frontal_files, lateral_files}
    uid_to_files: Dict[str, dict] = defaultdict(lambda: {"frontal_files": [], "lateral_files": []})
    with open(projections_csv, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            uid        = row.get("uid", "").strip()
            filename   = row.get("filename", "").strip()
            projection = row.get("projection", "").strip().lower()
            if not uid or not filename:
                continue
            if projection == "frontal":
                uid_to_files[uid]["frontal_files"].append(filename)
            elif projection == "lateral":
                uid_to_files[uid]["lateral_files"].append(filename)

    # Build index
    index: Dict[str, dict] = {}
    n_frontal_only = n_lateral_only = n_both = 0
    for uid, files in uid_to_files.items():
        entry = {
            "uid":           uid,
            "frontal_files": files["frontal_files"],
            "lateral_files": files["lateral_files"],
        }
        index[uid] = entry
        has_f = bool(files["frontal_files"])
        has_l = bool(files["lateral_files"])
        if has_f and has_l:
            n_both += 1
        elif has_f:
            n_frontal_only += 1
        elif has_l:
            n_lateral_only += 1

    logger.info(
        "CSV index: %d UIDs | both-views=%d | frontal-only=%d | lateral-only=%d",
        len(index), n_both, n_frontal_only, n_lateral_only,
    )
    return index


# ===========================================================================
# Feature-level helpers (low-level)
# ===========================================================================
@torch.no_grad()
def _encode_view(
    image_path: Optional[Path],
    model: BiomedCLIPClassifier,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """
    Encode ONE image into an L2-normalised feature vector of shape (1, D).
    Returns None if the path is missing or the image cannot be opened.
    """
    if image_path is None or not image_path.exists():
        return None
    try:
        image  = Image.open(image_path).convert("RGB")
        tensor = model.preprocess_val(image).unsqueeze(0).to(device)
        return model.encode(tensor)
    except Exception as exc:
        logger.debug("  Cannot encode %s: %s", image_path.name, exc)
        return None


@torch.no_grad()
def _classify_features(
    features: Optional[torch.Tensor],
    model: BiomedCLIPClassifier,
    label_columns: List[str],
) -> Dict[str, float]:
    """Run the classifier head on a fused feature vector -> {class: sigmoid_prob}."""
    if features is None:
        return _zeros_prob_dict(label_columns)
    logits = model.classify_features(features)
    probs  = torch.sigmoid(logits).squeeze(0).cpu().numpy()
    return {label_columns[i]: float(probs[i]) for i in range(len(label_columns))}


def _zeros_prob_dict(label_columns: List[str]) -> Dict[str, float]:
    return {c: 0.0 for c in label_columns}


def resolve_conflict(
    predicted_classes: List[str],
    prob_dict: Dict[str, float],
) -> Tuple[List[str], str]:
    has_normal   = "normal" in predicted_classes
    disease_list = [c for c in predicted_classes if c != "normal"]

    if not has_normal or not disease_list:
        return predicted_classes, "no_conflict"

    normal_prob  = prob_dict.get("normal", 0.0)
    max_dis_prob = max((prob_dict.get(c, 0.0) for c in disease_list), default=0.0)

    if normal_prob >= NORMAL_DOMINANCE_THRESHOLD and max_dis_prob < NORMAL_OVERRIDE_DISEASE:
        return ["normal"], "normal_dominates"
    else:
        return disease_list, "disease_dominates"


def compute_uncertainty(
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
    top3   = sorted(lurking, key=lambda x: x[1], reverse=True)[:3]
    reason = "normal_prob=%.3f (<%.2f) with elevated disease scores: %s" % (
        normal_prob, UNCERTAIN_UPPER,
        ", ".join("%s=%.3f" % (c, p) for c, p in top3),
    )
    return True, reason


# ===========================================================================
# Per-UID inference (dual-view)
# ===========================================================================
@torch.no_grad()
def predict_uid(
    uid: str,
    frontal_path: Optional[Path],
    lateral_path: Optional[Path],
    model: BiomedCLIPClassifier,
    thresholds: Dict[str, float],
    device: torch.device,
    label_columns: List[str],
) -> dict:
    """
    Full multi-view multi-label inference for one UID (feature-level fusion).

    Steps
    -----
    1. Encode the Frontal view  -> L2-normalised feature (or None if missing)
    2. Encode the Lateral view  -> L2-normalised feature (or None if missing)
    3. Fuse features by MEAN (masked) — exactly as in multi-view training
    4. Classifier head on the fused feature -> fused sigmoid probs
    5. Threshold -> predicted_classes (multi-label)
    6. Conflict resolution (Normal vs Disease) + KG name mapping

    Per-view probabilities (frontal_probs / lateral_probs) are computed by
    running the classifier on each single-view feature alone — i.e. what the
    model would predict from that view by itself — purely for inspection.

    Returns
    -------
    Full result dict including per-view probs, fused probs, and all metadata.
    """
    t0 = time.time()

    # ---- Steps 1-2: Encode each available view ----
    feat_f = _encode_view(frontal_path, model, device)
    feat_l = _encode_view(lateral_path, model, device)

    frontal_used = frontal_path.name if (frontal_path is not None and feat_f is not None) else ""
    lateral_used = lateral_path.name if (lateral_path is not None and feat_l is not None) else ""

    # ---- Step 3: Feature-level fusion (masked mean — matches training) ----
    if feat_f is not None and feat_l is not None:
        fused_feat  = (feat_f + feat_l) / 2.0
        fusion_note = "feature_fusion(frontal+lateral)"
    elif feat_f is not None:
        fused_feat  = feat_f
        fusion_note = "frontal_only"
    elif feat_l is not None:
        fused_feat  = feat_l
        fusion_note = "lateral_only"
    else:
        fused_feat  = None
        fusion_note = "no_image"
        logger.warning("  UID %s has NO valid images — returning zero probabilities", uid)

    # ---- Step 4: Classify fused + per-view features ----
    fused_prob   = _classify_features(fused_feat, model, label_columns)
    frontal_prob = _classify_features(feat_f, model, label_columns) if feat_f is not None else None
    lateral_prob = _classify_features(feat_l, model, label_columns) if feat_l is not None else None

    # ---- Step 5: Threshold -> multi-label predicted_classes ----
    predicted_classes: List[str] = sorted(
        [cls for cls, prob in fused_prob.items() if prob >= thresholds.get(cls, 0.5)],
        key=lambda c: fused_prob[c],
        reverse=True,
    )

    # ---- Step 6: Conflict resolution ----
    resolved_classes, conflict = resolve_conflict(predicted_classes, fused_prob)

    # ---- Step 7: KG name mapping ----
    kg_classes: List[str] = []
    for cls in resolved_classes:
        kg_key = LABEL_NORMALIZE_MAP.get(cls.lower())
        if kg_key and kg_key not in kg_classes:
            kg_classes.append(kg_key)

    # ---- Uncertainty ----
    is_uncertain, uncertainty_reason = compute_uncertainty(resolved_classes, fused_prob)

    return {
        "uid":               uid,
        # View information
        "frontal_image":     frontal_used,
        "lateral_image":     lateral_used,
        "fusion_mode":       fusion_note,
        # Per-view raw probabilities (for inspection / debugging)
        "frontal_probs":     frontal_prob if frontal_prob is not None else _zeros_prob_dict(label_columns),
        "lateral_probs":     lateral_prob if lateral_prob is not None else _zeros_prob_dict(label_columns),
        # Fused probabilities (primary output)
        "probabilities":     fused_prob,
        # Multi-label predictions
        "predicted_classes": predicted_classes,   # above threshold, sorted by confidence
        "resolved_classes":  resolved_classes,    # after conflict resolution
        "kg_classes":        kg_classes,           # KG-normalised names for step_B DAG traversal
        "num_predicted":     len(resolved_classes),
        # Metadata
        "conflict_resolution":  conflict,
        "is_uncertain":         is_uncertain,
        "uncertainty_reason":   uncertainty_reason,
        "inference_time_s":     round(time.time() - t0, 4),
    }


# ===========================================================================
# Batch runner  (UID-level, dual-view)
# ===========================================================================
def run_classification(
    test_folder:        Path,
    projections_csv:    Path,
    checkpoint:         Path,
    output_dir:         Path,
    threshold_csv:      Optional[Path],
    fallback_threshold: float,
    device_str:         str,
    unfreeze_layers:    int,
) -> None:

    # Device
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device_str == "auto" else torch.device(device_str)
    )
    logger.info("Device: %s", device)
    logger.info("Fusion: feature-level (mean of L2-normalised frontal+lateral features)")

    # Load CSV metadata (UID-keyed, both views)
    csv_index = load_csv_index(projections_csv)

    # Build stem->path lookup from test folder
    # (covers ALL images — Frontal and Lateral alike)
    stem_to_path: Dict[str, Path] = {}
    for p in test_folder.rglob("*"):
        if p.suffix.lower() in IMAGE_EXTENSIONS:
            stem_to_path[p.stem] = p

    if not stem_to_path:
        logger.error("No images found in: %s", test_folder)
        return
    logger.info("Found %d image file(s) in test folder", len(stem_to_path))

    # Model + thresholds
    model, label_columns = load_model(checkpoint, device, unfreeze_layers)
    thresholds = load_thresholds(threshold_csv, label_columns, fallback_threshold)

    # Output setup
    output_dir.mkdir(parents=True, exist_ok=True)
    json_dir  = output_dir / "per_uid_json"
    json_dir.mkdir(exist_ok=True)
    csv_out   = output_dir / "predictions.csv"

    # CSV columns
    prob_cols          = ["prob_%s"          % c for c in label_columns]
    frontal_prob_cols  = ["frontal_prob_%s"  % c for c in label_columns]
    lateral_prob_cols  = ["lateral_prob_%s"  % c for c in label_columns]
    csv_fields = (
        ["uid", "frontal_image", "lateral_image", "fusion_mode"]
        + prob_cols
        + frontal_prob_cols
        + lateral_prob_cols
        + ["predicted_classes", "resolved_classes", "kg_classes",
           "num_predicted", "conflict_resolution",
           "is_uncertain", "uncertainty_reason", "inference_time_s"]
    )

    all_results = []
    errors      = []
    n_no_image  = 0

    with open(csv_out, "w", newline="", encoding="utf-8") as csv_fh:
        writer = csv.DictWriter(csv_fh, fieldnames=csv_fields)
        writer.writeheader()

        for uid, meta in tqdm(csv_index.items(), desc="Classifying UIDs"):
            # Resolve to a SINGLE frontal + single lateral path (first available),
            # matching multi-view training which uses the first view of each type.
            frontal_path: Optional[Path] = None
            for fname in meta["frontal_files"]:
                stem = Path(fname).stem
                if stem in stem_to_path:
                    frontal_path = stem_to_path[stem]
                    break

            lateral_path: Optional[Path] = None
            for fname in meta["lateral_files"]:
                stem = Path(fname).stem
                if stem in stem_to_path:
                    lateral_path = stem_to_path[stem]
                    break

            if frontal_path is None and lateral_path is None:
                # UID is in the CSV but no image file exists in test_folder — skip
                n_no_image += 1
                logger.debug("  UID %s — no images in test folder, skipping", uid)
                continue

            try:
                result = predict_uid(
                    uid, frontal_path, lateral_path,
                    model, thresholds, device, label_columns,
                )
            except Exception as exc:
                logger.error("Error on UID %s: %s", uid, exc)
                errors.append({"uid": uid, "error": str(exc)})
                continue

            # Per-UID JSON
            json_result = {k: v for k, v in result.items()
                           if k not in ("frontal_probs", "lateral_probs", "probabilities")}
            json_result["probabilities"]    = result["probabilities"]
            json_result["frontal_probs"]    = result["frontal_probs"]
            json_result["lateral_probs"]    = result["lateral_probs"]
            with open(json_dir / ("uid_%s_pred.json" % uid), "w", encoding="utf-8") as jf:
                json.dump(json_result, jf, indent=2)

            # CSV row
            fused_p   = result["probabilities"]
            frontal_p = result["frontal_probs"]
            lateral_p = result["lateral_probs"]
            row = {
                "uid":            uid,
                "frontal_image":  result["frontal_image"],
                "lateral_image":  result["lateral_image"],
                "fusion_mode":    result["fusion_mode"],
            }
            for cls in label_columns:
                row["prob_%s"          % cls] = round(fused_p.get(cls, 0.0),   5)
                row["frontal_prob_%s"  % cls] = round(frontal_p.get(cls, 0.0), 5)
                row["lateral_prob_%s"  % cls] = round(lateral_p.get(cls, 0.0), 5)

            row["predicted_classes"]   = "|".join(result["predicted_classes"])
            row["resolved_classes"]    = "|".join(result["resolved_classes"])
            row["kg_classes"]          = "|".join(result["kg_classes"])
            row["num_predicted"]       = result["num_predicted"]
            row["conflict_resolution"] = result["conflict_resolution"]
            row["is_uncertain"]        = result["is_uncertain"]
            row["uncertainty_reason"]  = result["uncertainty_reason"]
            row["inference_time_s"]    = result["inference_time_s"]
            writer.writerow(row)

            all_results.append(result)

            logger.debug(
                "UID %-6s  [F=%d L=%d]  predicted=%s",
                uid,
                int(frontal_path is not None), int(lateral_path is not None),
                result["resolved_classes"],
            )

    # --- Summary ---
    n_ok  = len(all_results)
    n_err = len(errors)
    if n_ok:
        avg_classes  = sum(r["num_predicted"] for r in all_results) / n_ok
        n_conflict   = sum(1 for r in all_results if r["conflict_resolution"] != "no_conflict")
        n_uncertain  = sum(1 for r in all_results if r["is_uncertain"])
        n_no_pred    = sum(1 for r in all_results if r["num_predicted"] == 0)
        n_dual       = sum(1 for r in all_results if "lateral" in r["fusion_mode"] and "frontal" in r["fusion_mode"])
        n_single     = n_ok - n_dual
        logger.info(
            "\n=== Classification Complete ===\n"
            "  UIDs processed    : %d  |  Errors: %d  |  Skipped (no image): %d\n"
            "  Dual-view fused   : %d\n"
            "  Single-view only  : %d\n"
            "  Avg classes/UID   : %.2f\n"
            "  No class found    : %d UIDs\n"
            "  Conflicts resolved: %d\n"
            "  Uncertain flagged : %d\n"
            "  Predictions CSV   : %s\n"
            "  Per-UID JSON      : %s",
            n_ok, n_err, n_no_image,
            n_dual, n_single,
            avg_classes, n_no_pred,
            n_conflict, n_uncertain,
            csv_out, json_dir,
        )

    if errors:
        err_path = output_dir / "errors.json"
        with open(err_path, "w") as ef:
            json.dump(errors, ef, indent=2)
        logger.warning("Error details -> %s", err_path)


# ===========================================================================
# CLI
# ===========================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Step A — Dual-view multi-label chest X-ray classifier.\n"
            "Processes BOTH Frontal and Lateral images per UID, fuses probabilities,\n"
            "and outputs per-UID multi-label predictions for KG DAG traversal."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--test_folder",      default="/Users/bkishor/Desktop/kg_new/testing",
                   help="Folder containing test images (Frontal + Lateral)")
    p.add_argument("--projections",      default="/Users/bkishor/Desktop/kg_new/testing/indiana_projections.csv",
                   help="CSV: uid, filename, projection  (frontal/lateral)")
    p.add_argument("--checkpoint",       default="/Users/bkishor/Desktop/kg_new/BiomedCLIP/epoch_search_e100_fold2_new.pth",
                   help="Fine-tuned BiomedCLIP checkpoint (.pth)")
    p.add_argument("--threshold_csv",    default=None,
                   help="Optional CSV with per-class thresholds (columns: Class, threshold)")
    p.add_argument("--threshold",        type=float, default=0.5,
                   help="Uniform threshold when --threshold_csv is not provided")
    p.add_argument("--output_dir",       default="/Users/bkishor/Desktop/kg_new/BiomedCLIP/Output",
                   help="Where to save predictions.csv and per_uid_json/")
    p.add_argument("--device",           default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--unfreeze_layers",  type=int, default=4)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_classification(
        test_folder        = Path(args.test_folder),
        projections_csv    = Path(args.projections),
        checkpoint         = Path(args.checkpoint),
        output_dir         = Path(args.output_dir),
        threshold_csv      = Path(args.threshold_csv) if args.threshold_csv else None,
        fallback_threshold = args.threshold,
        device_str         = args.device,
        unfreeze_layers    = args.unfreeze_layers,
    )


if __name__ == "__main__":
    main()
