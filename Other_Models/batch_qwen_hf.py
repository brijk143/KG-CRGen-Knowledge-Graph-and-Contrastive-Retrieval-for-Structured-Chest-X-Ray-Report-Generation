"""
Batch VLM Baseline — Qwen2.5-VL-7B via Hugging Face
=====================================================
Processes all UIDs from testing2/projections.csv, sends frontal (+lateral)
chest X-ray images to Qwen2.5-VL-7B (local HF inference), and saves results.

Results saved under:
  other_models/qwen_hf/
    uid_<UID>/
      result.json      — raw + parsed response
      result.txt       — human-readable printout
  other_models/qwen_hf/batch_summary.json

Requires:
  pip install transformers torch pillow

Usage:
  python other_models/batch_qwen_hf.py
  python other_models/batch_qwen_hf.py --limit 5
  python other_models/batch_qwen_hf.py --no_resume

Caffeinate mode:
  caffeinate -i python other_models/batch_qwen_hf.py
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
from dotenv import load_dotenv
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

import os
from huggingface_hub import login as hf_login

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")
_hf_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
if _hf_token:
    hf_login(token=_hf_token, add_to_git_credential=False)

MODEL_ID       = "Qwen/Qwen2-VL-2B-Instruct"
DEFAULT_IMAGE_DIR  = BASE_DIR / "testing2" / "images"
DEFAULT_PROJ_CSV   = BASE_DIR / "testing2" / "projections.csv"
DEFAULT_OUTPUT_DIR = BASE_DIR / "other_models" / "qwen_hf"

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
print(f"[INFO] Using device: {DEVICE}")

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


# ── Model loader ──────────────────────────────────────────────────────

def load_model_and_processor():
    """Load Qwen2.5-VL-7B model and processor (called only once)."""
    print(f"[INFO] Loading model: {MODEL_ID}  (device={DEVICE})")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    if DEVICE == "cuda":
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
    elif DEVICE == "mps":
        # Apple Silicon — use float32, float16 causes buffer overflows on MPS
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        ).to("mps")
    else:
        # CPU fallback
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
    model.eval()
    print(f"[INFO] Model loaded successfully.")
    return model, processor


def run_qwen(
    frontal_path: Path,
    lateral_path: Path | None,
    model: Qwen2VLForConditionalGeneration,
    processor: AutoProcessor,
) -> dict:
    """Send images to Qwen and get analysis."""
    MAX_SIZE = 560  

    def load_and_resize(path: Path) -> Image.Image:
        img = Image.open(path).convert("RGB")
        w, h = img.size
        scale = MAX_SIZE / max(w, h)
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        return img

    frontal = load_and_resize(frontal_path)
    lateral = load_and_resize(lateral_path) if lateral_path else None
    content = [{"type": "text", "text": QUESTIONS_PROMPT}]
    content.append({"type": "image", "image": frontal})
    if lateral:
        content.append({"type": "text", "text": "This is the lateral view:"})
        content.append({"type": "image", "image": lateral})

    messages = [{"role": "user", "content": content}]

    # Process and generate
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # Generate
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=512)

    generated_ids = [
        out[len(inp):]
        for inp, out in zip(inputs["input_ids"], output_ids)
    ]
    raw = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Parse FINDINGS / IMPRESSION
    import re
    findings_m   = re.search(r"FINDINGS:\s*(.+?)(?=IMPRESSION:|$)", raw, re.DOTALL | re.IGNORECASE)
    impression_m = re.search(r"IMPRESSION:\s*(.+?)$",               raw, re.DOTALL | re.IGNORECASE)

    return {
        "model": MODEL_ID,
        "raw_response": raw,
        "precise_summary": {
            "findings":   findings_m.group(1).strip()   if findings_m   else raw,
            "impression": impression_m.group(1).strip() if impression_m else "",
        },
    }


def load_projections(csv_path: Path) -> dict[str, dict]:
    """Load projections CSV, return {uid: {frontal: fname, lateral: fname}}."""
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
        f"  VLM BASELINE (Qwen2.5-VL-7B) — UID {uid}",
        f"  Model  : {result.get('model', MODEL_ID)}",
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
    parser = argparse.ArgumentParser(description="Batch Qwen2.5-VL-7B Vision baseline.")
    parser.add_argument("--image_dir",       default=str(DEFAULT_IMAGE_DIR),  type=Path)
    parser.add_argument("--projections_csv", default=str(DEFAULT_PROJ_CSV),   type=Path)
    parser.add_argument("--output_dir",      default=str(DEFAULT_OUTPUT_DIR), type=Path)
    parser.add_argument("--limit",    default=100, type=int,
                        help="Process only first N UIDs (default: 100).")
    parser.add_argument("--no_resume", action="store_true",
                        help="Re-process UIDs even if result already exists.")
    args = parser.parse_args()

    image_dir  = Path(args.image_dir)
    proj_csv   = Path(args.projections_csv)
    output_dir = Path(args.output_dir)

    if not image_dir.exists():
        sys.exit(f"[ERROR] Image directory not found: {image_dir}")
    if not proj_csv.exists():
        sys.exit(f"[ERROR] Projections CSV not found: {proj_csv}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model once
    print("[INFO] Initializing model...")
    model, processor = load_model_and_processor()
    print("[INFO] Model ready.\n")

    projections = load_projections(proj_csv)
    uid_list = sorted(projections.keys(), key=lambda x: int(x) if x.isdigit() else x)
    if args.limit:
        uid_list = uid_list[: args.limit]

    total   = len(uid_list)
    success = 0
    skipped = 0
    failed  = 0
    summary: list[dict] = []

    print(f"{'='*60}")
    print(f"  Batch Qwen2.5-VL-7B (HuggingFace) Baseline")
    print(f"  Model           : {MODEL_ID}")
    print(f"  UIDs to process : {total}")
    print(f"  Image dir       : {image_dir}")
    print(f"  Output dir      : {output_dir}")
    print(f"  Device          : {DEVICE}")
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

        try:
            result      = run_qwen(frontal_path, lateral_path, model, processor)
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

        except Exception as e:
            print(f"FAILED — {e}")
            failed += 1
            summary.append({"uid": uid, "status": "failed", "error": str(e)})

    # ── Batch summary ─────────────────────────────────────────────────
    summary_path = output_dir / "batch_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "total": total, "success": success,
            "skipped": skipped, "failed": failed,
            "model": MODEL_ID,
            "device": DEVICE,
            "results": summary
        }, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  DONE  — {success} success | {skipped} skipped | {failed} failed")
    print(f"  Results : {output_dir}")
    print(f"  Summary : {summary_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
