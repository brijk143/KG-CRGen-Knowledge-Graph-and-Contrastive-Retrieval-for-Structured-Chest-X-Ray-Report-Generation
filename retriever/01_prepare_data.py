"""
01_prepare_data.py
===============
Reads training and data CSV files, pairs each image with its combined
(findings + impression) caption, verifies image existence, and writes
train.json, val.json, test.json.

Val/Test split: data/indiana_reports.csv UIDs sorted → first 50% = val, last 50% = test.
Multi-image UIDs: each image becomes its own (image, caption) pair.
Missing text: skip UIDs where BOTH findings AND impression are NaN/empty.
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path("/Users/bkishor/Desktop/kg_new")
TRAIN_REPORTS = BASE_DIR / "training" / "training_indiana_reports.csv"
TRAIN_PROJECTIONS = BASE_DIR / "training" / "training_indiana_projections.csv"
TRAIN_IMAGE_DIR = BASE_DIR / "training" / "files"

DATA_REPORTS = BASE_DIR / "testing" / "indiana_reports.csv"
DATA_PROJECTIONS = BASE_DIR / "testing" / "indiana_projections.csv"
DATA_IMAGE_DIR = BASE_DIR / "testing" / "test1"

OUTPUT_DIR = BASE_DIR / "Retriever" / "splits"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MAX_CAPTION_CHARS = 400   # ~60-70 tokens after tokenization


def build_caption(row: pd.Series) -> str:
    """
    Build caption: IMPRESSION first (most clinically important), then FINDINGS.
    Truncates to MAX_CAPTION_CHARS so critical terms aren't cut off by the
    tokenizer's 77-token hard limit.
    """
    findings   = str(row.get("findings",   "")).strip() if pd.notna(row.get("findings"))   else ""
    impression = str(row.get("impression", "")).strip() if pd.notna(row.get("impression")) else ""

    # Impression first — CLIP truncates from the right, so put key info early
    if impression and findings:
        caption = impression + " " + findings
    elif impression:
        caption = impression
    else:
        caption = findings

    # Truncate at word boundary to avoid mid-word cuts
    if len(caption) > MAX_CAPTION_CHARS:
        caption = caption[:MAX_CAPTION_CHARS].rsplit(" ", 1)[0]

    return caption.strip()


def build_pairs(reports_df: pd.DataFrame, proj_df: pd.DataFrame,
                image_dir: Path, split_name: str) -> list[dict]:
    """
    For each UID:
      1. Build caption from findings + impression
      2. Skip if caption is empty
      3. For each image in projections, verify file exists
      4. Create an entry: {uid, image_path, caption, projection, split}
    """
    pairs = []
    skipped_no_text = 0
    skipped_no_image = 0

    for _, row in reports_df.iterrows():
        uid = int(row["uid"])
        caption = build_caption(row)

        if not caption:
            skipped_no_text += 1
            continue

        # Get all images for this UID
        uid_images = proj_df[proj_df["uid"] == uid]

        if uid_images.empty:
            skipped_no_image += 1
            continue

        for _, img_row in uid_images.iterrows():
            filename = img_row["filename"]
            projection = img_row["projection"]
            image_path = image_dir / filename

            if not image_path.exists():
                skipped_no_image += 1
                continue

            pairs.append({
                "uid": uid,
                "image_path": str(image_path),
                "caption": caption,
                "projection": projection,
                "split": split_name,
            })

    print(f"  [{split_name}] Built {len(pairs)} image-caption pairs")
    print(f"  [{split_name}] Skipped {skipped_no_text} UIDs (no text), "
          f"{skipped_no_image} images (not found or no projection)")
    return pairs


def main():
    print("=" * 60)
    print("STEP 1: PREPARE DATA SPLITS")
    print("=" * 60)

    # ─── Load CSVs ───────────────────────────────────────────
    train_reports = pd.read_csv(TRAIN_REPORTS)
    train_proj = pd.read_csv(TRAIN_PROJECTIONS)
    data_reports = pd.read_csv(DATA_REPORTS)
    data_proj = pd.read_csv(DATA_PROJECTIONS)

    print(f"\nTraining reports: {len(train_reports)} UIDs")
    print(f"Data reports:     {len(data_reports)} UIDs")

    # ─── Stratified split: balance normal vs abnormal in val & test ────
    # Sorting by UID was arbitrary and could create unbalanced splits.
    # We stratify by whether the UID is 'normal' or 'abnormal' so both
    # val and test have the same label distribution as the full dataset.
    def is_normal(uid: int) -> bool:
        rows = data_reports[data_reports["uid"] == uid]
        if rows.empty:
            return True
        prob = str(rows.iloc[0].get("Problems", "")).strip().lower()
        return prob in ("", "normal")

    data_uids = sorted(data_reports["uid"].unique())
    normal_uids   = [u for u in data_uids if     is_normal(u)]
    abnormal_uids = [u for u in data_uids if not is_normal(u)]

    # Alternate normal/abnormal into val/test buckets for balance
    val_uids  = set(normal_uids[0::2] + abnormal_uids[0::2])   # every other
    test_uids = set(normal_uids[1::2] + abnormal_uids[1::2])

    val_reports  = data_reports[data_reports["uid"].isin(val_uids)].copy()
    test_reports = data_reports[data_reports["uid"].isin(test_uids)].copy()

    print(f"\nStratified split:")
    print(f"  Normal UIDs   : {len(normal_uids)} → val={len([u for u in normal_uids if u in val_uids])}, test={len([u for u in normal_uids if u in test_uids])}")
    print(f"  Abnormal UIDs : {len(abnormal_uids)} → val={len([u for u in abnormal_uids if u in val_uids])}, test={len([u for u in abnormal_uids if u in test_uids])}")
    print(f"  Total Val UIDs: {len(val_uids)}  |  Total Test UIDs: {len(test_uids)}")

    # ─── Build pairs ─────────────────────────────────────────
    print("\n--- Building training pairs ---")
    train_pairs = build_pairs(train_reports, train_proj, TRAIN_IMAGE_DIR, "train")

    print("\n--- Building validation pairs ---")
    val_pairs = build_pairs(val_reports, data_proj, DATA_IMAGE_DIR, "val")

    print("\n--- Building test pairs ---")
    test_pairs = build_pairs(test_reports, data_proj, DATA_IMAGE_DIR, "test")

    # ─── Write JSON files ────────────────────────────────────
    for name, pairs in [("train", train_pairs), ("val", val_pairs), ("test", test_pairs)]:
        out_path = OUTPUT_DIR / f"{name}.json"
        with open(out_path, "w") as f:
            json.dump(pairs, f, indent=2)
        print(f"\n✅ Saved {out_path}  ({len(pairs)} pairs)")

    # ─── Summary stats ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, pairs in [("train", train_pairs), ("val", val_pairs), ("test", test_pairs)]:
        uids = set(p["uid"] for p in pairs)
        frontal = sum(1 for p in pairs if p["projection"] == "Frontal")
        lateral = sum(1 for p in pairs if p["projection"] == "Lateral")
        avg_caption_len = np.mean([len(p["caption"]) for p in pairs]) if pairs else 0
        print(f"  {name:5s}: {len(pairs):5d} pairs | {len(uids):4d} UIDs | "
              f"Frontal: {frontal} | Lateral: {lateral} | "
              f"Avg caption len: {avg_caption_len:.0f} chars")


if __name__ == "__main__":
    main()
