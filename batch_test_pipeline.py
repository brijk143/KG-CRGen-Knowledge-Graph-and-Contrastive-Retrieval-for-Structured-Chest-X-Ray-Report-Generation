"""
batch_test_pipeline.py — Batch runner for the End-to-End Chest X-Ray Pipeline
===============================================================================
Processes all UIDs found in a test image folder, using a projections CSV
to determine which images are Frontal vs Lateral for each UID.

For each UID:
  - If both Frontal + Lateral exist  → runs dual-view pipeline
  - If only Frontal exists           → runs single-view pipeline
  - If only Lateral (no frontal)     → runs single-view pipeline with lateral as input 

Results per UID are saved under:
  output/batch_results/
    uid_<UID>/
      pipeline_result.txt
      pipeline_result.json

A summary file (batch_summary.json) is also written at the root of
output/batch_results/.

Usage
-----
 run this file using this command- 
  python batch_test_pipeline.py \
    --image_dir testing/test1 \
    --projections_csv testing/indiana_projections.csv \
    --no_resume


  python batch_test_pipeline.py \\
      --image_dir testing/test1 \\
      --projections_csv testing/indiana_projections.csv

  # With options
  python batch_test_pipeline.py \\
      --image_dir testing/test1 \\
      --projections_csv testing/indiana_projections.csv \\
      --output_dir output/batch_results \\
      --top_k 5 --threshold 0.5 --device auto
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from test_pipeline import (  # noqa: E402
    run_pipeline,
    display_results,
    save_results,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def load_projection_map(csv_path: Path, image_dir: Path) -> Dict[int, Dict[str, Optional[Path]]]:
    """
    Reads the projections CSV and cross-references with actual files in
    image_dir.  Returns a dict keyed by uid:
      {
        uid: {
          "frontal": Path | None,   # first Frontal image found on disk
          "lateral": Path | None,   # first Lateral image found on disk
        },
        ...
      }
    If a UID has multiple Frontal images, the first one found is used
    (consistent with the CSV row order).
    """
    import csv

    uid_map: Dict[int, Dict[str, Optional[Path]]] = {}

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = int(row["uid"])
            filename = row["filename"].strip()
            projection = row["projection"].strip().lower()  # "frontal" / "lateral"

            img_path = image_dir / filename
            if not img_path.exists():
                continue  # image not present in the folder — skip silently

            if uid not in uid_map:
                uid_map[uid] = {"frontal": None, "lateral": None}

            if projection == "frontal" and uid_map[uid]["frontal"] is None:
                uid_map[uid]["frontal"] = img_path
            elif projection == "lateral" and uid_map[uid]["lateral"] is None:
                uid_map[uid]["lateral"] = img_path

    return uid_map


def run_batch(
    image_dir: Path,
    projections_csv: Path,
    output_dir: Path,
    device: torch.device,
    threshold: float = 0.5,
    frontal_weight: float = 0.65,
    top_k: int = 5,
    top_k_triplets: int = 20,
    resume: bool = True,
) -> None:
    """
    Iterates over every UID that has at least one image on disk and runs
    the full pipeline.  Results are stored per UID.
    """
    log.info("Loading projection map from: %s", projections_csv)
    uid_map = load_projection_map(projections_csv, image_dir)

    if not uid_map:
        log.error("No matching images found in %s. Check paths.", image_dir)
        sys.exit(1)

    uids = sorted(uid_map.keys())
    log.info("Found %d UIDs with at least one image on disk.", len(uids))

    output_dir.mkdir(parents=True, exist_ok=True)

    summary: List[dict] = []
    skipped: List[int] = []

    total = len(uids)
    for idx, uid in enumerate(uids, 1):
        frontal_path: Optional[Path] = uid_map[uid]["frontal"]
        lateral_path: Optional[Path] = uid_map[uid]["lateral"]

        # Determine view mode and resolve which image is primary
        if frontal_path is not None and lateral_path is not None:
            view_mode = "dual_view"
            primary_path   = frontal_path
            secondary_path = lateral_path
        elif frontal_path is not None:
            view_mode = "frontal_only"
            primary_path   = frontal_path
            secondary_path = None
        elif lateral_path is not None:
            # No frontal — use lateral as the sole input
            view_mode = "lateral_only"
            primary_path   = lateral_path
            secondary_path = None
        else:
            log.warning("[%d/%d] UID %d — no images found at all, skipping.", idx, total, uid)
            skipped.append(uid)
            continue

        uid_out_dir = output_dir / f"uid_{uid}"

        # Resume: skip if already processed
        if resume and (uid_out_dir / "pipeline_result.json").exists():
            log.info("[%d/%d] UID %d — already processed, skipping (use --no_resume to redo).", idx, total, uid)
            try:
                with open(uid_out_dir / "pipeline_result.json", encoding="utf-8") as f:
                    prev = json.load(f)
                summary.append({
                    "uid": uid,
                    "frontal": str(frontal_path) if frontal_path else None,
                    "lateral": str(lateral_path) if lateral_path else None,
                    "mode": view_mode,
                    "status": "skipped_resume",
                    "predicted_classes": prev.get("classification", {}).get("final_classes", []),
                })
            except Exception:
                pass
            continue
        log.info(
            "[%d/%d] UID %d — mode: %s | frontal: %s | lateral: %s",
            idx, total, uid, view_mode,
            frontal_path.name if frontal_path else "—",
            lateral_path.name if lateral_path else "—",
        )

        # For lateral_only, pass the lateral as frontal_path so the pipeline
        # processes it as a single image (pipeline accepts Optional[Path])
        pipeline_frontal = primary_path
        pipeline_lateral = secondary_path if view_mode == "dual_view" else None

        t0 = time.time()
        try:
            result = run_pipeline(
                frontal_path=pipeline_frontal,
                lateral_path=pipeline_lateral,
                device=device,
                threshold=threshold,
                frontal_weight=frontal_weight,
                top_k=top_k,
                top_k_triplets=top_k_triplets,
            )

            txt_path, json_path = save_results(result, frontal_path, lateral_path, uid_out_dir)
            elapsed = time.time() - t0

            predicted = result.get("classification", {}).get("final_classes", [])
            log.info(
                "   ✓ Done in %.1fs | classes: %s",
                elapsed,
                ", ".join(predicted) if predicted else "none",
            )

            summary.append({
                "uid": uid,
                "frontal": str(frontal_path) if frontal_path else None,
                "lateral": str(lateral_path) if lateral_path else None,
                "mode": view_mode,
                "status": "success",
                "elapsed_sec": round(elapsed, 2),
                "predicted_classes": predicted,
                "output_dir": str(uid_out_dir),
            })

        except Exception as exc:
            elapsed = time.time() - t0
            log.error("   ✗ UID %d FAILED after %.1fs: %s", uid, elapsed, exc, exc_info=True)
            summary.append({
                "uid": uid,
                "frontal": str(frontal_path) if frontal_path else None,
                "lateral": str(lateral_path) if lateral_path else None,
                "mode": view_mode,
                "status": "error",
                "error": str(exc),
                "elapsed_sec": round(elapsed, 2),
            })

        # Write running summary after each UID so progress is not lost
        _write_summary(summary, skipped, output_dir)

    # Final summary
    _write_summary(summary, skipped, output_dir)
    _print_final_stats(summary, skipped, output_dir)


def _write_summary(summary: List[dict], skipped: List[int], output_dir: Path) -> None:
    summary_path = output_dir / "batch_summary.json"
    data = {
        "total_processed": len(summary),
        "skipped_no_frontal": skipped,
        "results": summary,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def _print_final_stats(summary: List[dict], skipped: List[int], output_dir: Path) -> None:
    success = [s for s in summary if s["status"] == "success"]
    errors  = [s for s in summary if s["status"] == "error"]
    resumed = [s for s in summary if s["status"] == "skipped_resume"]

    print("\n" + "═" * 60)
    print("  BATCH COMPLETE")
    print("═" * 60)
    print(f"  Total UIDs processed : {len(summary)}")
    print(f"  ✓ Success            : {len(success)}")
    print(f"  ↩ Resumed (skipped)  : {len(resumed)}")
    print(f"  ✗ Errors             : {len(errors)}")
    print(f"  ⚠ No images skipped  : {len(skipped)}")
    print(f"\n  Results saved to     : {output_dir}")
    print(f"  Summary JSON         : {output_dir / 'batch_summary.json'}")
    print("═" * 60 + "\n")

    if errors:
        print("  Failed UIDs:")
        for e in errors:
            print(f"    UID {e['uid']}: {e.get('error', 'unknown error')}")
        print()


# ══════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch-run the chest X-ray pipeline over a folder of images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--image_dir",
        default="testing/test1",
        type=str,
        help="Folder containing test images (default: testing/test1)",
    )
    p.add_argument(
        "--projections_csv",
        default="testing/indiana_projections.csv",
        type=str,
        help="CSV with uid,filename,projection columns (default: testing/indiana_projections.csv)",
    )
    p.add_argument(
        "--output_dir",
        default="output/batch_results",
        type=str,
        help="Root directory for per-UID results (default: output/batch_results)",
    )
    p.add_argument("--threshold",      type=float, default=0.5)
    p.add_argument("--frontal_weight", type=float, default=0.65)
    p.add_argument("--top_k",          type=int,   default=5)
    p.add_argument("--top_k_triplets", type=int,   default=20)
    p.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Compute device (default: auto)",
    )
    p.add_argument(
        "--no_resume",
        action="store_true",
        help="Re-process UIDs even if output already exists",
    )
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

    image_dir       = BASE_DIR / args.image_dir
    projections_csv = BASE_DIR / args.projections_csv
    output_dir      = BASE_DIR / args.output_dir

    if not image_dir.exists():
        sys.exit(f"[ERROR] Image directory not found: {image_dir}")
    if not projections_csv.exists():
        sys.exit(f"[ERROR] Projections CSV not found: {projections_csv}")

    run_batch(
        image_dir=image_dir,
        projections_csv=projections_csv,
        output_dir=output_dir,
        device=device,
        threshold=args.threshold,
        frontal_weight=args.frontal_weight,
        top_k=args.top_k,
        top_k_triplets=args.top_k_triplets,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
