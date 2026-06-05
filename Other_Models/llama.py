"""
VLM Baseline: Send chest X-ray image to Llama-4 Maverick (via Groq — FREE)
and answer the same KEY Q&A questions as the main pipeline.

Usage:
  python Other_Models/claude_openrouter.py --frontal path/to/frontal.png
  python Other_Models/claude_openrouter.py --frontal path/to/frontal.png --lateral path/to/lateral.png
  python Other_Models/claude_openrouter.py --frontal path/to/frontal.png --output output/vlm_baseline/result.json
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR.parent / ".env")

MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

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
    """Open any image, convert to JPEG, return base64 string."""
    img = Image.open(path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def image_part(path: Path) -> dict:
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{to_base64_jpeg(path)}"
        },
    }



def run_vlm_baseline(frontal_path: Path, lateral_path: Path | None = None) -> dict:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        sys.exit("[ERROR] GROQ_API_KEY not found in .env")

    client = Groq(api_key=api_key)

    content: list[dict] = [{"type": "text", "text": QUESTIONS_PROMPT}]
    content.append(image_part(frontal_path))
    if lateral_path:
        content.append({"type": "text", "text": "This is the lateral view:"})
        content.append(image_part(lateral_path))

    print(f"\n  Model  : {MODEL}  (via Groq — free)")
    print(f"  Input  : {frontal_path.name}" +
          (f" + {lateral_path.name}" if lateral_path else ""))
    print("  Analysing image(s) …\n")

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": content}],
        max_tokens=1024,
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="VLM Baseline — Claude Sonnet 4.5 Vision via OpenRouter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--frontal", required=True, type=Path,
                   help="Path to the frontal chest X-ray image.")
    p.add_argument("--lateral", default=None, type=Path,
                   help="(Optional) Path to the lateral chest X-ray image.")
    p.add_argument("--output", default=None, type=Path,
                   help="(Optional) Path to save JSON output.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.frontal.exists():
        sys.exit(f"[ERROR] Frontal image not found: {args.frontal}")
    if args.lateral and not args.lateral.exists():
        sys.exit(f"[ERROR] Lateral image not found: {args.lateral}")

    result = run_vlm_baseline(args.frontal, args.lateral)

    # ── Print to terminal ─────────────────────────────────────────────
    print("=" * 70)
    print(f"  VLM BASELINE ({MODEL}) — CHEST X-RAY ANALYSIS")
    print("=" * 70)
    print(result["raw_response"])
    print("=" * 70)
    print("\nPARSED OUTPUT")
    print("-" * 70)
    ps = result["precise_summary"]
    print(f"FINDINGS:\n  {ps['findings']}\n")
    print(f"IMPRESSION:\n  {ps['impression']}")
    print("=" * 70)

    # ── Save JSON if requested ────────────────────────────────────────
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"\n  Saved to: {args.output}")


if __name__ == "__main__":
    main()