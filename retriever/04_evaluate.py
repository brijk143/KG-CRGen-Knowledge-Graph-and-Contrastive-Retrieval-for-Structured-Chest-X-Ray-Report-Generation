"""
04_evaluate.py
===========
Evaluates the trained CLIP retriever on BOTH validation and test sets.
For each image in the split, retrieves top-K reports from the training
database and computes retrieval metrics.

Metrics computed:
    - Recall@K (K=1, 3, 5, 10): does a relevant report appear in top-K?
    (Since exact UID won't be in training, we use LABEL OVERLAP as ground truth)
    - Accuracy@K (alias of Recall@K for retrieval hit-rate reporting)
  - Precision@K: among top-K retrieved, how many share labels with query?
    - F1@K from Precision@K and Recall@K
  - Mean Reciprocal Rank (MRR)
  - Mean similarity score of top-1

Note on "correct match":
  Since val/test UIDs are NOT in the training database, there's no exact match.
  We define a "relevant" retrieved report as one that shares ≥1 label with the query.
  IMPORTANT: We use the 'Problems' column (MeSH terms) for ALL reports, because
  the 'label' column uses a different vocabulary and causes massive mismatch.
"""

import json
import csv
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
from PIL import Image
import open_clip
import pandas as pd
from tqdm import tqdm

from torchvision import transforms as T

# ─── Paths ───────────────────────────────────────────────────────────
BASE_DIR = Path("/Users/bkishor/Desktop/kg_new")
CKPT_DIR = BASE_DIR / "Retriever" / "checkpoints"
DB_DIR = BASE_DIR / "Retriever" / "database"
SPLITS_DIR = BASE_DIR / "Retriever" / "splits"
EVAL_DIR = BASE_DIR / "Retriever" / "evaluation"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
TOP_K_VALUES = (1, 3, 5, 10)

# ─── Label extraction helpers ────────────────────────────────────────

# Labels that are radiology DB metadata tags, not semantic disease labels.
# Retrieval can never match these, so they unfairly penalise F1.
JUNK_LABELS = {"no indexing", "no finding", "not reported", "normal examination"}


def parse_labels(label_str: str) -> set:
    """Parse semicolon/comma-separated label string into a set of labels.
    
    Filters out junk DB-metadata tags that are not meaningful for retrieval
    (e.g. 'no indexing' which always scores 0 because nothing in the DB
    carries it as a semantic disease label).
    """
    if not label_str or pd.isna(label_str) or label_str.strip() == "":
        return set()
    # Handle both separators
    labels = set()
    for sep in [";", ","]:
        if sep in str(label_str):
            labels.update(l.strip().lower() for l in str(label_str).split(sep) if l.strip())
            return labels - JUNK_LABELS
    raw = str(label_str).strip().lower()
    return set() if raw in JUNK_LABELS else {raw}


def build_uid_labels():
    """Build UID → set of labels mapping for both training and data reports.
    
    Uses the 'Problems' column (MeSH terms) for ALL reports to ensure
    consistent vocabulary between train and val/test sets.
    The 'label' column in data reports uses a different clinical vocabulary
    which causes ~16% of val/test UIDs to be unmatchable.
    """
    uid_labels = {}

    # Training reports: use 'Problems' column
    train_reports = pd.read_csv(BASE_DIR / "training" / "training_indiana_reports.csv")
    for _, row in train_reports.iterrows():
        uid = int(row["uid"])
        uid_labels[uid] = parse_labels(row.get("Problems", ""))

    # Data reports: ALSO use 'Problems' column (NOT 'label') for consistent vocabulary
    data_reports = pd.read_csv(BASE_DIR / "testing" / "indiana_reports.csv")
    for _, row in data_reports.iterrows():
        uid = int(row["uid"])
        uid_labels[uid] = parse_labels(row.get("Problems", ""))

    return uid_labels


# ══════════════════════════════════════════════════════════════════════
# EVALUATION ENGINE
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_split(split_name: str, model, preprocess, db_vectors, db_metadata,
                   uid_labels: dict, device, top_k_values=TOP_K_VALUES,
                   db_img_vectors=None, text_weight: float = 1.0, image_weight: float = 0.0):
    """
    Evaluate retrieval performance on a split.
    
    Supports HYBRID retrieval: blends text DB similarity and image DB similarity.
      score = text_weight * text_sim + image_weight * img_sim
    If db_img_vectors is None, falls back to text-only (text_weight=1.0).

    For each unique UID in the split:
      1. Encode its image(s) with the visual encoder (with TTA)
      2. Retrieve top-K from training database using blended score
      3. Check if any retrieved report shares labels with query
    """
    # Load split data
    with open(SPLITS_DIR / f"{split_name}.json") as f:
        split_data = json.load(f)

    print(f"\n{'='*60}")
    print(f"EVALUATING: {split_name.upper()} SET")
    print(f"{'='*60}")
    print(f"Total image-caption pairs: {len(split_data)}")

    # Group by UID (evaluate per UID, not per image)
    uid_images = defaultdict(list)
    for entry in split_data:
        uid_images[entry["uid"]].append(entry)

    print(f"Unique UIDs: {len(uid_images)}")

    max_k = max(top_k_values)

    # TTA transform: horizontal flip (chest X-rays are symmetric)
    tta_hflip = T.functional.hflip

    # Per-UID results
    all_results = []
    skipped_no_labels = 0

    for uid, entries in tqdm(uid_images.items(), desc=f"Eval {split_name}"):
        query_labels = uid_labels.get(uid, set())

        # Encode all images for this UID with TTA (original + hflip per image),
        # then average across views (frontal + lateral).
        # TTA: average original+flip FIRST per image, then average across images.
        # This treats each view equally regardless of how many are present.
        image_features_list = []
        for entry in entries:
            image = Image.open(entry["image_path"])
            if image.mode != "RGB":
                image = image.convert("RGB")
            # Original
            img_tensor = preprocess(image).unsqueeze(0).to(device)
            feat = model.encode_image(img_tensor)
            feat = F.normalize(feat, dim=-1)
            # TTA: horizontal flip
            img_flipped = tta_hflip(image)
            img_tensor_f = preprocess(img_flipped).unsqueeze(0).to(device)
            feat_f = model.encode_image(img_tensor_f)
            feat_f = F.normalize(feat_f, dim=-1)
            # Average TTA for this view before adding to view list
            view_feat = F.normalize((feat + feat_f) / 2.0, dim=-1)
            image_features_list.append(view_feat)
        
        # Average across all views (frontal + lateral)
        avg_features = torch.mean(torch.stack(image_features_list), dim=0)
        avg_features = F.normalize(avg_features, dim=-1)

        # Retrieve top-K using hybrid score: text_sim + image_sim
        # text_sim: image query vs text DB (cross-modal, main signal)
        text_similarities = (avg_features @ db_vectors.T).squeeze(0)

        if db_img_vectors is not None and image_weight > 0:
            # image_sim: image query vs image DB (same-modal, visual similarity)
            img_similarities = (avg_features @ db_img_vectors.T).squeeze(0)
            similarities = text_weight * text_similarities + image_weight * img_similarities
        else:
            similarities = text_similarities

        top_k_vals, top_k_indices = torch.topk(similarities, min(max_k, len(db_metadata)))

        retrieved = []
        for score, idx in zip(top_k_vals.cpu().numpy(), top_k_indices.cpu().numpy()):
            r_uid = db_metadata[idx]["uid"]
            r_caption = db_metadata[idx].get("caption", "")
            r_labels = uid_labels.get(r_uid, set())
            is_relevant = bool(query_labels & r_labels) if query_labels and r_labels else False
            retrieved.append({
                "uid": r_uid,
                "score": float(score),
                "caption": r_caption,
                "labels": list(r_labels),
                "is_relevant": is_relevant,
            })

        # Compute metrics for this UID
        uid_result = {
            "uid": uid,
            "query_labels": list(query_labels),
            "query_caption": entries[0]["caption"][:100],
            "num_images": len(entries),
            "top1_score": retrieved[0]["score"] if retrieved else 0,
            "top1_uid": retrieved[0]["uid"] if retrieved else None,
        }

        # Recall@K: is there at least one relevant result in top-K?
        for k in top_k_values:
            top_k_retrieved = retrieved[:k]
            uid_result[f"recall@{k}"] = int(any(r["is_relevant"] for r in top_k_retrieved))

        # Precision@K: what fraction of top-K are relevant?
        for k in top_k_values:
            top_k_retrieved = retrieved[:k]
            if top_k_retrieved:
                uid_result[f"precision@{k}"] = sum(
                    r["is_relevant"] for r in top_k_retrieved) / len(top_k_retrieved)
            else:
                uid_result[f"precision@{k}"] = 0.0

        # MRR: rank of first relevant result
        mrr = 0.0
        for rank, r in enumerate(retrieved, 1):
            if r["is_relevant"]:
                mrr = 1.0 / rank
                break
        uid_result["mrr"] = mrr

        # Store top-5 retrieved for analysis
        uid_result["top5_retrieved"] = retrieved[:5]

        all_results.append(uid_result)

    # ─── Aggregate metrics ────────────────────────────────────
    # Only include UIDs that have labels — empty-label UIDs always score 0
    # and would unfairly drag down F1/MRR
    scored_results = [r for r in all_results if r.get("query_labels")]
    skipped_no_labels = len(all_results) - len(scored_results)
    if skipped_no_labels:
        print(f"  Skipping {skipped_no_labels} UIDs with no labels from metric aggregation")

    n = len(scored_results) if scored_results else len(all_results)
    results_for_metrics = scored_results if scored_results else all_results
    metrics = {
        "split": split_name,
        "num_uids": len(all_results),
        "num_scored_uids": n,
        "num_pairs": len(split_data),
    }

    for k in top_k_values:
        metrics[f"recall@{k}"]    = sum(r[f"recall@{k}"]    for r in results_for_metrics) / n
        metrics[f"precision@{k}"] = sum(r[f"precision@{k}"] for r in results_for_metrics) / n
        metrics[f"accuracy@{k}"]  = metrics[f"recall@{k}"]
        precision_k = metrics[f"precision@{k}"]
        recall_k    = metrics[f"recall@{k}"]
        metrics[f"f1@{k}"] = (2 * precision_k * recall_k / (precision_k + recall_k)) if (precision_k + recall_k) > 0 else 0.0

    metrics["mrr"]            = sum(r["mrr"]         for r in results_for_metrics) / n
    metrics["avg_top1_score"] = sum(r["top1_score"]  for r in results_for_metrics) / n

    print(f"\n{'─'*40}")
    print(f"RESULTS: {split_name.upper()}")
    print(f"{'─'*40}")
    print(f"  Scored UIDs: {n} / {len(all_results)} (excluded {len(all_results)-n} with no labels)")
    for k in top_k_values:
        print(f"  Accuracy@{k}:  {metrics[f'accuracy@{k}']:.4f}  "
              f"({sum(r[f'recall@{k}'] for r in results_for_metrics)}/{n})")
    print(f"  MRR:         {metrics['mrr']:.4f}")
    for k in top_k_values:
        print(f"  Precision@{k}: {metrics[f'precision@{k}']:.4f}")
    for k in top_k_values:
        print(f"  F1@{k}:       {metrics[f'f1@{k}']:.4f}")
    print(f"  Avg top-1 similarity: {metrics['avg_top1_score']:.4f}")

    # ─── Save detailed results ────────────────────────────────
    results_path = EVAL_DIR / f"{split_name}_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "metrics": metrics,
            "per_uid_results": all_results,
        }, f, indent=2)
    print(f"\n✅ Detailed results saved: {results_path}")

    return metrics, all_results


def main():
    # ─── Device ──────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # ─── Load model + checkpoint ─────────────────────────────
    print(f"Loading model: {MODEL_NAME}")
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME)
    ckpt = torch.load(CKPT_DIR / "best_indiana_clip.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"Checkpoint: epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}")

    # ─── Load database ───────────────────────────────────────
    text_vectors = np.load(DB_DIR / "indiana_database_vectors.npy")
    with open(DB_DIR / "indiana_database_metadata.json") as f:
        metadata = json.load(f)
    db_vectors = torch.from_numpy(text_vectors).to(device)
    print(f"Text DB: {len(metadata)} reports, shape {text_vectors.shape}")

    # Load image DB if available (built by new 03_build_database.py)
    img_db_path = DB_DIR / "indiana_database_img_vectors.npy"
    blend_path  = DB_DIR / "blend_weights.json"
    db_img_vectors = None
    text_weight, image_weight = 1.0, 0.0
    if img_db_path.exists():
        img_vectors = np.load(img_db_path)
        db_img_vectors = torch.from_numpy(img_vectors).to(device)
        print(f"Image DB: shape {img_vectors.shape}  ✅ Hybrid retrieval enabled")
        if blend_path.exists():
            import json as _json
            w = _json.load(open(blend_path))
            text_weight  = w.get("text",  0.6)
            image_weight = w.get("image", 0.4)
        else:
            text_weight, image_weight = 0.6, 0.4
        print(f"Blend weights → text: {text_weight}, image: {image_weight}")
    else:
        print("Image DB not found — using text-only retrieval")

    # ─── Build label mappings ─────────────────────────────────
    uid_labels = build_uid_labels()
    print(f"UID-label mappings: {len(uid_labels)}")

    # ─── Evaluate BOTH splits ─────────────────────────────────
    val_metrics, val_results = evaluate_split(
        "val", model, preprocess, db_vectors, metadata, uid_labels, device,
        db_img_vectors=db_img_vectors, text_weight=text_weight, image_weight=image_weight)
    
    test_metrics, test_results = evaluate_split(
        "test", model, preprocess, db_vectors, metadata, uid_labels, device,
        db_img_vectors=db_img_vectors, text_weight=text_weight, image_weight=image_weight)

    # ─── Save comparison summary ──────────────────────────────
    summary_path = EVAL_DIR / "comparison_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "validation", "test"])
        metric_keys = ["num_uids", "num_pairs"]
        for k in TOP_K_VALUES:
            metric_keys.extend([f"accuracy@{k}", f"precision@{k}", f"f1@{k}"])
        metric_keys.extend(["mrr", "avg_top1_score"])
        for key in metric_keys:
            writer.writerow([key, f"{val_metrics.get(key, 'N/A')}", 
                                  f"{test_metrics.get(key, 'N/A')}"])
    print(f"\n✅ Comparison summary saved: {summary_path}")

    # ─── Final comparison table ───────────────────────────────
    print(f"\n{'='*60}")
    print("VALIDATION vs TEST COMPARISON")
    print(f"{'='*60}")
    print(f"{'Metric':<20} {'Validation':>12} {'Test':>12}")
    print(f"{'─'*44}")
    comparison_keys = []
    for k in TOP_K_VALUES:
        comparison_keys.extend([f"accuracy@{k}", f"precision@{k}", f"f1@{k}"])
    comparison_keys.extend(["mrr", "avg_top1_score"])
    for key in comparison_keys:
        v = val_metrics.get(key, 0)
        t = test_metrics.get(key, 0)
        print(f"{key:<20} {v:>12.4f} {t:>12.4f}")


if __name__ == "__main__":
    main()
