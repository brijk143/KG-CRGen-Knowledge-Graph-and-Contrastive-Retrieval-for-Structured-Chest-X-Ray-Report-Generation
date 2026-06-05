"""
Batch VLM Baseline — Claude Sonnet 4.5 via OpenRouter
======================================================
Processes all UIDs from testing2/projections.csv, sends frontal (+lateral)
chest X-ray images to Claude Sonnet 4.5 via OpenRouter, and saves results.

Results saved under:
  other_models/claude_openrouter/
    uid_<UID>/
      result.json      — raw + parsed response
      result.txt       — human-readable printout
  other_models/claude_openrouter/batch_summary.json

Usage:
  python other_models/batch_claude_openrouter.py
  python other_models/batch_claude_openrouter.py --limit 5
  python other_models/batch_claude_openrouter.py --no_resume

Caffeinate mode:
  caffeinate -i python other_models/batch_claude_openrouter.py
"""
from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

MODEL          = "anthropic/claude-sonnet-4-5"
DEFAULT_IMAGE_DIR  = BASE_DIR / "testing2" / "images"
DEFAULT_PROJ_CSV   = BASE_DIR / "testing2" / "projections.csv"
DEFAULT_OUTPUT_DIR = BASE_DIR / "other_models" / "claude_openrouter"
API_DELAY = 3.0

QUESTIONS_PROMPT = """\
You are an expert radiologist conducting a research study on automated \
chest X-ray analysis. You have been given one or two chest X-ray images \
(frontal view, or frontal + lateral).

Carefully examine the image(s) and provide a detailed radiological analysis \
by answering each question below:

KEY Q&A:
1. What conditions are detected?
2. What is the most likely primary diagnosis?
3. What are the supporting radiological findings?
4. What conditions are ruled out?
5. Is this a normal chest X-ray?
6. What follow-up actions or tests might be recommended?
7. What are the clinical implications for the patient?

After answering all questions, provide a structured clinical report in \
EXACTLY this format on its own lines:

FINDINGS: <one detailed paragraph describing all radiological observations>
IMPRESSION: <one concise sentence with the primary diagnosis>
"""



def to_base64_jpeg(path: Path) -> str:
    img = Image.open(path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def image_part(path: Path) -> dict:
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{to_base64_jpeg(path)}"},
    }

def run_claude(frontal_path: Path, lateral_path: Path | None, client: OpenAI) -> dict:
    content: list[dict] = [{"type": "text", "text": QUESTIONS_PROMPT}]
    content.append(image_part(frontal_path))
    if lateral_path:
        content.append({"type": "text", "text": "This is the lateral view:"})
        content.append(image_part(lateral_path))

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": content}],
        max_tokens=800,
        temperature=0.0,
    )
    raw = response.choices[0].message.content

    findings_m   = re.search(r"FINDINGS:\s*(.+?)(?=IMPRESSION:|$)", raw, re.DOTALL | re.IGNORECASE)
    impression_m = re.search(r"IMPRESSION:\s*(.+?)$",               raw, re.DOTALL | re.IGNORECASE)

    return {
        "model": MODEL,
        "raw_response": raw,
        "precise_summary": {
            "findings":   findings_m.group(1).strip()   if findings_m   else raw,
            "impression": impression_m.group(1).strip() if impression_m else "",
        },
    }


def load_projections(csv_path: Path) -> dict[str, dict]:
    mapping: dict[str, dict] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            uid   = str(row["uid"]).strip()
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
        f"  VLM BASELINE (Claude Sonnet 4.5) — UID {uid}",
        f"  Model  : {result.get('model', MODEL)}",
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
    parser = argparse.ArgumentParser(description="Batch Claude Sonnet 4.5 Vision baseline.")
    parser.add_argument("--image_dir",       default=str(DEFAULT_IMAGE_DIR),  type=Path)
    parser.add_argument("--projections_csv", default=str(DEFAULT_PROJ_CSV),   type=Path)
    parser.add_argument("--output_dir",      default=str(DEFAULT_OUTPUT_DIR), type=Path)
    parser.add_argument("--limit",    default=None, type=int,
                        help="Process only first N UIDs (for testing).")
    parser.add_argument("--no_resume", action="store_true",
                        help="Re-process UIDs even if result already exists.")
    parser.add_argument("--delay",    default=API_DELAY, type=float)
    args = parser.parse_args()

    image_dir  = Path(args.image_dir)
    proj_csv   = Path(args.projections_csv)
    output_dir = Path(args.output_dir)

    if not image_dir.exists():
        sys.exit(f"[ERROR] Image directory not found: {image_dir}")
    if not proj_csv.exists():
        sys.exit(f"[ERROR] Projections CSV not found: {proj_csv}")

    api_key = os.getenv("openrouter_api_key")
    if not api_key:
        sys.exit("[ERROR] openrouter_api_key not found in .env")

    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "https://github.com/research/xray-pipeline",
            "X-Title": "Chest X-Ray Claude Baseline",
        },
    )

    output_dir.mkdir(parents=True, exist_ok=True)

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
    print(f"  Batch Claude Sonnet 4.5 (OpenRouter) Baseline")
    print(f"  Model           : {MODEL}")
    print(f"  UIDs to process : {total}")
    print(f"  Image dir       : {image_dir}")
    print(f"  Output dir      : {output_dir}")
    print(f"  Resume mode     : {not args.no_resume}")
    print(f"{'='*60}\n")

    for idx, uid in enumerate(uid_list, 1):
        uid_out     = output_dir / f"uid_{uid}"
        result_json = uid_out / "result.json"

        # Resume — skip already done
        if not args.no_resume and result_json.exists():
            print(f"[{idx}/{total}] UID {uid:>6} — SKIP (already done)")
            skipped += 1
            summary.append({"uid": uid, "status": "skipped"})
            continue

        views         = projections[uid]
        frontal_fname = views.get("frontal")
        lateral_fname = views.get("lateral")

        # Resolve frontal path
        if frontal_fname:
            frontal_path = image_dir / frontal_fname
        else:
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
            summary.append({"uid": uid, "status": "frontal_missing"})
            continue

        lateral_path = None
        if lateral_fname:
            lp = image_dir / lateral_fname
            if lp.exists():
                lateral_path = lp

        view_tag = "frontal+lateral" if lateral_path else "frontal_only"
        print(f"[{idx}/{total}] UID {uid:>6} — {view_tag} … ", end="", flush=True)

        # Retry loop for 429 / credit errors
        max_retries = 5
        for attempt in range(1, max_retries + 1):
            try:
                result      = run_claude(frontal_path, lateral_path, client)
                result["uid"]  = uid
                result["view"] = view_tag

                uid_out.mkdir(parents=True, exist_ok=True)
                with open(result_json, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2)
                with open(uid_out / "result.txt", "w", encoding="utf-8") as f:
                    f.write(format_txt(uid, result, frontal_path, lateral_path))

                print("OK")
                success += 1
                summary.append({
                    "uid": uid, "status": "success", "view": view_tag,
                    "impression": result["precise_summary"].get("impression", ""),
                })
                break  # done

            except Exception as e:
                err_str = str(e)
                # 429 rate-limit: parse wait time and sleep
                if "429" in err_str or "rate" in err_str.lower():
                    m = re.search(r"try again in (\d+)m([\d.]+)s", err_str)
                    wait = int(m.group(1)) * 60 + float(m.group(2)) + 5 if m else 60 * attempt
                    print(f"RATE LIMITED (attempt {attempt}/{max_retries}) — waiting {wait:.0f}s …")
                    time.sleep(wait)
                    if attempt == max_retries:
                        print(f"FAILED after {max_retries} retries.")
                        failed += 1
                        summary.append({"uid": uid, "status": "failed", "error": err_str})
                # 402 insufficient credits
                elif "402" in err_str:
                    print(f"FAILED — Insufficient credits. Top up at https://openrouter.ai/settings/credits")
                    failed += 1
                    summary.append({"uid": uid, "status": "no_credits", "error": err_str})
                    break
                else:
                    print(f"FAILED — {e}")
                    failed += 1
                    summary.append({"uid": uid, "status": "failed", "error": err_str})
                    break

        # Rate-limit delay between calls
        if idx < total:
            time.sleep(args.delay)
    summary_path = output_dir / "batch_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"total": total, "success": success,
                   "skipped": skipped, "failed": failed,
                   "results": summary}, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  DONE  — {success} success | {skipped} skipped | {failed} failed")
    print(f"  Results : {output_dir}")
    print(f"  Summary : {summary_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
