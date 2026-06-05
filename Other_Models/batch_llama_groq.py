"""
Batch VLM Baseline — Claude Sonnet 4.5 via OpenRouter
======================================================
Runs claude_openrouter.py logic on all UIDs found in a projections CSV,
sending frontal (+ lateral if available) images to Claude Vision.

Results saved under:
  Other_Models/vlm_results/
    uid_<UID>/
      result.json      — raw + parsed response
      result.txt       — human-readable printout

  Other_Models/vlm_results/batch_summary.json  — all UIDs summary

Usage:
  python Other_Models/batch_claude_baseline.py
  python Other_Models/batch_claude_baseline.py --image_dir training2/images --projections_csv training2/projections.csv
  python Other_Models/batch_claude_baseline.py --limit 10          # first 10 UIDs only
  python Other_Models/batch_claude_baseline.py --no_resume         # re-run all UIDs
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from groq import RateLimitError

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from Other_Models.llama import run_vlm_baseline   # noqa: E402
DEFAULT_IMAGE_DIR  = BASE_DIR / "testing2" / "images"
DEFAULT_PROJ_CSV   = BASE_DIR / "testing2" / "projections.csv"
DEFAULT_OUTPUT_DIR = BASE_DIR / "Other_Models" / "llama_groq"

API_DELAY = 3.0

def load_projections(csv_path: Path) -> dict[str, dict[str, Path]]:
    """
    Returns { uid_str: {"frontal": Path, "lateral": Path|None} }
    """
    import csv
    mapping: dict[str, dict] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid  = str(row["uid"]).strip()
            fname = row["filename"].strip()
            proj  = row["projection"].strip().lower()
            if uid not in mapping:
                mapping[uid] = {"frontal": None, "lateral": None}
            if "frontal" in proj:
                mapping[uid]["frontal"] = fname
            elif "lateral" in proj:
                mapping[uid]["lateral"] = fname
    return mapping


def format_txt(uid: str, result: dict, frontal: Path, lateral: Path | None) -> str:
    lines = [
        "=" * 70,
        f"  VLM BASELINE — UID {uid}",
        f"  Model  : {result.get('model', 'claude')}",
        f"  Input  : {frontal.name}" + (f" + {lateral.name}" if lateral else ""),
        "=" * 70,
        result["raw_response"],
        "=" * 70,
        "",
        "PARSED OUTPUT",
        "-" * 70,
        f"FINDINGS:\n  {result['precise_summary']['findings']}",
        "",
        f"IMPRESSION:\n  {result['precise_summary']['impression']}",
        "=" * 70,
    ]
    return "\n".join(lines)



def main() -> None:
    parser = argparse.ArgumentParser(description="Batch Claude Vision baseline on all UIDs.")
    parser.add_argument("--image_dir",       default=str(DEFAULT_IMAGE_DIR),  type=Path)
    parser.add_argument("--projections_csv", default=str(DEFAULT_PROJ_CSV),   type=Path)
    parser.add_argument("--output_dir",      default=str(DEFAULT_OUTPUT_DIR), type=Path)
    parser.add_argument("--limit",           default=None, type=int,
                        help="Process only first N UIDs (for testing).")
    parser.add_argument("--no_resume",       action="store_true",
                        help="Re-process UIDs even if result already exists.")
    parser.add_argument("--delay",           default=API_DELAY, type=float,
                        help=f"Seconds between API calls (default {API_DELAY}).")
    args = parser.parse_args()

    image_dir  = Path(args.image_dir)
    proj_csv   = Path(args.projections_csv)
    output_dir = Path(args.output_dir)

    if not image_dir.exists():
        sys.exit(f"[ERROR] Image directory not found: {image_dir}")
    if not proj_csv.exists():
        sys.exit(f"[ERROR] Projections CSV not found: {proj_csv}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load projections
    projections = load_projections(proj_csv)
    uid_list = sorted(projections.keys(), key=lambda x: int(x) if x.isdigit() else x)

    if args.limit:
        uid_list = uid_list[: args.limit]

    total   = len(uid_list)
    success = 0
    skipped = 0
    failed  = 0
    summary: list[dict] = []

    print(f"\n{'='*60}")
    print(f"  Batch Claude Vision Baseline")
    print(f"  UIDs to process : {total}")
    print(f"  Image dir       : {image_dir}")
    print(f"  Output dir      : {output_dir}")
    print(f"  Resume mode     : {not args.no_resume}")
    print(f"{'='*60}\n")

    for idx, uid in enumerate(uid_list, 1):
        uid_out = output_dir / f"uid_{uid}"
        result_json = uid_out / "result.json"

        # Skip if already done
        if not args.no_resume and result_json.exists():
            print(f"[{idx}/{total}] UID {uid:>6} — SKIP (already done)")
            skipped += 1
            summary.append({"uid": uid, "status": "skipped"})
            continue

        views = projections[uid]
        frontal_fname = views.get("frontal")
        lateral_fname = views.get("lateral")

        # Resolve frontal
        if frontal_fname:
            frontal_path = image_dir / frontal_fname
        else:
            # Try any image for this UID as fallback
            candidates = list(image_dir.glob(f"{uid}_*.png")) + list(image_dir.glob(f"{uid}_*.jpg"))
            if not candidates:
                print(f"[{idx}/{total}] UID {uid:>6} — SKIP (no image found)")
                failed += 1
                summary.append({"uid": uid, "status": "no_image"})
                continue
            frontal_path = candidates[0]

        if not frontal_path.exists():
            print(f"[{idx}/{total}] UID {uid:>6} — SKIP (frontal missing: {frontal_path.name})")
            failed += 1
            summary.append({"uid": uid, "status": "frontal_missing", "file": str(frontal_path)})
            continue

        lateral_path = None
        if lateral_fname:
            lp = image_dir / lateral_fname
            if lp.exists():
                lateral_path = lp

        view_tag = "frontal+lateral" if lateral_path else "frontal_only"
        print(f"[{idx}/{total}] UID {uid:>6} — {view_tag} … ", end="", flush=True)

        max_retries = 5
        for attempt in range(1, max_retries + 1):
            try:
                result = run_vlm_baseline(frontal_path, lateral_path)
                result["uid"] = uid
                result["view"] = view_tag

                uid_out.mkdir(parents=True, exist_ok=True)

                # Save JSON
                with open(result_json, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2)

                # Save TXT
                txt = format_txt(uid, result, frontal_path, lateral_path)
                with open(uid_out / "result.txt", "w", encoding="utf-8") as f:
                    f.write(txt)

                print("OK")
                success += 1
                summary.append({
                    "uid": uid,
                    "status": "success",
                    "view": view_tag,
                    "impression": result["precise_summary"].get("impression", ""),
                })
                break  # success — exit retry loop

            except RateLimitError as e:
                err_str = str(e)
                # Parse suggested wait time from error message, e.g. "16m19.9s"
                m = re.search(r"try again in (\d+)m([\d.]+)s", err_str)
                if m:
                    wait_sec = int(m.group(1)) * 60 + float(m.group(2)) + 5
                else:
                    wait_sec = 60 * attempt  # fallback: 1 min, 2 min, ...
                print(f"RATE LIMITED (attempt {attempt}/{max_retries}) — waiting {wait_sec:.0f}s …")
                time.sleep(wait_sec)
                if attempt == max_retries:
                    print(f"FAILED after {max_retries} retries — {e}")
                    failed += 1
                    summary.append({"uid": uid, "status": "failed", "error": err_str})

            except Exception as e:
                print(f"FAILED — {e}")
                failed += 1
                summary.append({"uid": uid, "status": "failed", "error": str(e)})
                break  # non-rate-limit error, don't retry

        # Rate-limit delay
        if idx < total:
            time.sleep(args.delay)

    summary_path = output_dir / "batch_summary.json"
    batch_meta = {
        "total": total,
        "success": success,
        "skipped": skipped,
        "failed": failed,
        "results": summary,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(batch_meta, f, indent=2)

    print(f"  DONE  — {success} success | {skipped} skipped | {failed} failed")
    print(f"  Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
