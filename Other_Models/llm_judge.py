"""
LLM-as-Judge — Compare 4 VLM approaches against ground truth
=============================================================
Uses Azure OpenAI GPT-4o-mini to semantically evaluate predictions
from Claude, LLaMA, Qwen, and KG (our approach) against ground truth
reports from testing2/reports.csv.

Usage:
  python other_models/llm_judge.py
  python other_models/llm_judge.py --limit 30
  caffeinate -i python other_models/llm_judge.py
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from openai import AzureOpenAI

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# ── Directories ───────────────────────────────────────────────────────
CLAUDE_DIR  = BASE_DIR / "Other_Models" / "claude_openrouter"
LLAMA_DIR   = BASE_DIR / "Other_Models" / "llama_groq"
QWEN_DIR    = BASE_DIR / "Other_Models" / "qwen_hf"
KG_DIR      = BASE_DIR / "output" / "batch_results_shuffle"
GT_CSV      = BASE_DIR / "testing2" / "reports.csv"
OUTPUT_DIR  = BASE_DIR / "other_models" / "judge_results"

MODEL_DIRS = {
    "claude_openrouter": CLAUDE_DIR,
    "llama_groq":        LLAMA_DIR,
    "qwen_hf":           QWEN_DIR,
    "our_approach":      KG_DIR,
}

JUDGE_PROMPT = """\
You are an expert radiologist acting as a judge in a research study. Your task \
is to evaluate how well a MODEL PREDICTION matches the GROUND TRUTH radiology report.

## GROUND TRUTH
**Findings:** {gt_findings}
**Impression:** {gt_impression}

## MODEL PREDICTION
{prediction}

## EVALUATION CRITERIA
Score each dimension from 1 (poor) to 5 (excellent):

1. **Condition Detection (1-5):** Did the model correctly identify the conditions/pathologies present (or correctly identify a normal study)?
2. **Primary Diagnosis Accuracy (1-5):** Does the model's primary diagnosis match the ground truth impression?
3. **Finding Completeness (1-5):** How completely does the model capture the radiological findings described in the ground truth?
4. **False Positive Rate (1-5):** 5 = no false positives; 1 = many fabricated findings not in ground truth.
5. **Clinical Relevance (1-5):** Are the model's recommendations and clinical implications appropriate given the ground truth?

## OUTPUT FORMAT (strict JSON only, no extra text)
{{
  "condition_detection": <int 1-5>,
  "primary_diagnosis": <int 1-5>,
  "finding_completeness": <int 1-5>,
  "false_positive_rate": <int 1-5>,
  "clinical_relevance": <int 1-5>,
  "overall_score": <float, average of above 5 scores>,
  "brief_justification": "<1-2 sentences>"
}}
"""


def load_ground_truth(csv_path: Path) -> dict[str, dict]:
    """Load reports.csv → {uid: {findings, impression}}."""
    gt = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames
        # Try common column name variations
        find_col = next((c for c in cols if c.lower().strip() in ("findings", "finding")), None)
        imp_col  = next((c for c in cols if c.lower().strip() in ("impression", "impressions")), None)
        uid_col  = next((c for c in cols if c.lower().strip() in ("uid",)), None)
        if not uid_col:
            sys.exit(f"[ERROR] No 'uid' column in {csv_path}. Columns: {cols}")
        for row in reader:
            uid = str(row[uid_col]).strip()
            gt[uid] = {
                "findings":   row.get(find_col, "").strip() if find_col else "",
                "impression": row.get(imp_col, "").strip()  if imp_col  else "",
            }
    return gt


def read_result_txt(uid_dir: Path) -> str | None:
    """Read result.txt or pipeline_result.txt from a uid folder."""
    for fname in ("result.txt", "pipeline_result.txt"):
        txt_path = uid_dir / fname
        if txt_path.exists():
            return txt_path.read_text(encoding="utf-8")
    json_path = uid_dir / "result.json"
    if json_path.exists():
        data = json.loads(json_path.read_text(encoding="utf-8"))
        return data.get("raw_response", json.dumps(data.get("precise_summary", {})))
    return None


def get_available_uids(model_dirs: dict[str, Path]) -> dict[str, set[str]]:
    """Find UIDs available per model."""
    available = {}
    for name, d in model_dirs.items():
        if not d.exists():
            available[name] = set()
            continue
        uids = set()
        for sub in d.iterdir():
            if sub.is_dir() and sub.name.startswith("uid_"):
                uid = sub.name[4:]
                if (sub / "result.txt").exists() or (sub / "result.json").exists() or (sub / "pipeline_result.txt").exists():
                    uids.add(uid)
        available[name] = uids
    return available


def judge_single(
    client: AzureOpenAI,
    deployment: str,
    gt: dict,
    prediction: str,
) -> dict:
    """Call Azure OpenAI to judge one prediction."""
    prompt = JUDGE_PROMPT.format(
        gt_findings=gt["findings"] or "(not provided)",
        gt_impression=gt["impression"] or "(not provided)",
        prediction=prediction[:3000],  # truncate if huge
    )
    resp = client.chat.completions.create(
        model=deployment,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.0,
    )
    raw = resp.choices[0].message.content.strip()
    # Extract JSON
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        return json.loads(m.group())
    return {"error": "parse_failed", "raw": raw}


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-as-Judge for 4 VLM approaches.")
    parser.add_argument("--limit", default=None, type=int, help="Max UIDs per model.")
    parser.add_argument("--delay", default=1.0, type=float)
    parser.add_argument("--no_resume", action="store_true")
    args = parser.parse_args()

    # Azure OpenAI setup
    api_key    = os.getenv("AZURE_OPENAI_API_KEY")
    api_ver    = os.getenv("AZURE_OPENAI_API_VERSION")
    endpoint   = os.getenv("AZURE_OPENAI_ENDPOINT")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME_5_1", "gpt-4o-mini")

    if not all([api_key, endpoint]):
        sys.exit("[ERROR] Azure OpenAI env vars not set (AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT)")

    client = AzureOpenAI(
        api_key=api_key,
        api_version=api_ver or "2024-02-15-preview",
        azure_endpoint=endpoint,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load ground truth
    gt_data = load_ground_truth(GT_CSV)
    print(f"[INFO] Ground truth loaded: {len(gt_data)} UIDs")

    # Find available UIDs per model
    available = get_available_uids(MODEL_DIRS)
    for name, uids in available.items():
        print(f"[INFO] {name:>20}: {len(uids)} UIDs available")

    # ── Per-model evaluation (all available UIDs per model) ─────────
    # Build per-model UID lists (intersected with ground truth only)
    per_model_uids: dict[str, list[str]] = {}
    for name, uids in available.items():
        valid = sorted(uids & set(gt_data.keys()), key=lambda x: int(x) if x.isdigit() else x)
        if args.limit:
            valid = valid[:args.limit]
        per_model_uids[name] = valid
        print(f"[INFO] {name:>20}: {len(valid)} UIDs to evaluate (with GT)")

    # Common UIDs (for head-to-head comparison table)
    common_set = set(gt_data.keys())
    for uids in available.values():
        common_set &= uids
    common_uids = sorted(common_set, key=lambda x: int(x) if x.isdigit() else x)
    if args.limit:
        common_uids = common_uids[:args.limit]
    print(f"[INFO] Common UIDs (all 4 models): {len(common_uids)}")

    # Count total API calls needed
    total_calls = sum(len(v) for v in per_model_uids.values())
    print(f"\n{'='*70}")
    print(f"  LLM-as-Judge Evaluation (per-model + common)")
    print(f"  Judge Model  : Azure OpenAI / {deployment}")
    print(f"  Total judgements: ~{total_calls}")
    print(f"{'='*70}\n")

    # Per-model scores
    all_scores: dict[str, list[dict]] = {name: [] for name in MODEL_DIRS}
    detailed_results: list[dict] = []
    call_count = 0

    for model_name, model_dir in MODEL_DIRS.items():
        uid_list = per_model_uids[model_name]
        n = len(uid_list)
        print(f"\n── {model_name} ({n} UIDs) ──")

        for idx, uid in enumerate(uid_list, 1):
            uid_dir = model_dir / f"uid_{uid}"
            result_cache = OUTPUT_DIR / f"uid_{uid}_{model_name}.json"

            # Resume
            if not args.no_resume and result_cache.exists():
                scores = json.loads(result_cache.read_text())
                all_scores[model_name].append(scores)
                detailed_results.append({"uid": uid, "model": model_name, "scores": scores})
                print(f"  [{idx}/{n}] UID {uid:>6}: {scores.get('overall_score', '?'):.2f} (cached)")
                continue

            gt = gt_data[uid]
            prediction = read_result_txt(uid_dir)
            if not prediction:
                print(f"  [{idx}/{n}] UID {uid:>6}: SKIP (no result)")
                continue

            try:
                scores = judge_single(client, deployment, gt, prediction)
                scores["uid"] = uid
                scores["model"] = model_name

                with open(result_cache, "w") as f:
                    json.dump(scores, f, indent=2)

                all_scores[model_name].append(scores)
                detailed_results.append({"uid": uid, "model": model_name, "scores": scores})
                ov = scores.get("overall_score", "?")
                print(f"  [{idx}/{n}] UID {uid:>6}: {ov:.2f}" if isinstance(ov, (int, float)) else f"  [{idx}/{n}] UID {uid:>6}: {ov}")
                call_count += 1
                time.sleep(args.delay)

            except Exception as e:
                print(f"  [{idx}/{n}] UID {uid:>6}: ERROR — {e}")
                time.sleep(args.delay)

    # ── Aggregate scores ──────────────────────────────────────────────
    DIMS = ["condition_detection", "primary_diagnosis", "finding_completeness",
            "false_positive_rate", "clinical_relevance", "overall_score"]

    def compute_avgs(scores_list: list[dict]) -> dict[str, float]:
        avgs = {}
        for dim in DIMS:
            vals = [s[dim] for s in scores_list if isinstance(s.get(dim), (int, float))]
            avgs[dim] = sum(vals) / len(vals) if vals else 0.0
        return avgs

    def print_table(title: str, leaderboard: dict[str, dict], counts: dict[str, int]):
        print(f"\n{'='*95}")
        print(f"  {title}")
        print(f"{'='*95}")
        print(f"{'Model':>22} | {'CondDet':>7} | {'PrimDx':>7} | {'FindComp':>8} | {'FPRate':>7} | {'ClinRel':>7} | {'OVERALL':>7} | {'N':>3}")
        print("-" * 95)
        for model_name in MODEL_DIRS:
            if model_name not in leaderboard:
                print(f"{model_name:>22} | {'N/A':>7} | {'N/A':>7} | {'N/A':>8} | {'N/A':>7} | {'N/A':>7} | {'N/A':>7} | {0:>3}")
                continue
            a = leaderboard[model_name]
            n = counts[model_name]
            print(f"{model_name:>22} | {a['condition_detection']:>7.2f} | {a['primary_diagnosis']:>7.2f} | "
                  f"{a['finding_completeness']:>8.2f} | {a['false_positive_rate']:>7.2f} | "
                  f"{a['clinical_relevance']:>7.2f} | {a['overall_score']:>7.2f} | {n:>3}")
        if leaderboard:
            ranked = sorted(leaderboard.items(), key=lambda x: x[1]["overall_score"], reverse=True)
            print(f"\n  RANKING:")
            for rank, (name, avgs) in enumerate(ranked, 1):
                print(f"    #{rank}  {name:<22}  Overall: {avgs['overall_score']:.2f}")

    # ── TABLE 1: Per-model (all available UIDs) ───────────────────────
    lb_all = {}
    counts_all = {}
    for model_name in MODEL_DIRS:
        sl = all_scores[model_name]
        if sl:
            lb_all[model_name] = compute_avgs(sl)
            counts_all[model_name] = len(sl)
    print_table("PER-MODEL RESULTS — All Available UIDs (1-5 scale)", lb_all, counts_all)

    # ── TABLE 2: Common UIDs only (head-to-head) ─────────────────────
    lb_common = {}
    counts_common = {}
    for model_name in MODEL_DIRS:
        common_scores = [s for s in all_scores[model_name] if s.get("uid") in common_set]
        if common_scores:
            lb_common[model_name] = compute_avgs(common_scores)
            counts_common[model_name] = len(common_scores)
    print_table(f"HEAD-TO-HEAD — Common UIDs Only ({len(common_uids)} UIDs, 1-5 scale)", lb_common, counts_common)

    # Save summary
    summary = {
        "per_model": {
            "leaderboard": {k: {d: round(v, 3) for d, v in avgs.items()} for k, avgs in lb_all.items()},
            "counts": counts_all,
            "ranking": [{"rank": i+1, "model": name, "overall": round(avgs["overall_score"], 3)}
                         for i, (name, avgs) in enumerate(sorted(lb_all.items(),
                         key=lambda x: x[1]["overall_score"], reverse=True))],
        },
        "head_to_head": {
            "num_common_uids": len(common_uids),
            "leaderboard": {k: {d: round(v, 3) for d, v in avgs.items()} for k, avgs in lb_common.items()},
            "counts": counts_common,
            "ranking": [{"rank": i+1, "model": name, "overall": round(avgs["overall_score"], 3)}
                         for i, (name, avgs) in enumerate(sorted(lb_common.items(),
                         key=lambda x: x[1]["overall_score"], reverse=True))],
        },
        "detailed": detailed_results,
    }
    summary_path = OUTPUT_DIR / "judge_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved: {summary_path}")

    # ── Save CSV results ──────────────────────────────────────────────
    # 1. Per-UID detailed CSV
    detail_csv = OUTPUT_DIR / "judge_detailed.csv"
    with open(detail_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["uid", "model", "condition_detection", "primary_diagnosis",
                         "finding_completeness", "false_positive_rate", "clinical_relevance",
                         "overall_score", "brief_justification"])
        for entry in detailed_results:
            s = entry["scores"]
            if "error" in s:
                continue
            writer.writerow([
                entry["uid"], entry["model"],
                s.get("condition_detection", ""),
                s.get("primary_diagnosis", ""),
                s.get("finding_completeness", ""),
                s.get("false_positive_rate", ""),
                s.get("clinical_relevance", ""),
                s.get("overall_score", ""),
                s.get("brief_justification", ""),
            ])
    print(f"  Detailed CSV : {detail_csv}")

    # 2. Leaderboard summary CSV
    leader_csv = OUTPUT_DIR / "judge_leaderboard.csv"
    with open(leader_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["table", "model", "n_uids", "condition_detection", "primary_diagnosis",
                         "finding_completeness", "false_positive_rate", "clinical_relevance",
                         "overall_score"])
        for model_name, avgs in lb_all.items():
            writer.writerow(["per_model", model_name, counts_all.get(model_name, 0)] +
                            [round(avgs[d], 3) for d in DIMS])
        for model_name, avgs in lb_common.items():
            writer.writerow(["head_to_head", model_name, counts_common.get(model_name, 0)] +
                            [round(avgs[d], 3) for d in DIMS])
    print(f"  Leaderboard  : {leader_csv}")

    # ── Visualizations ────────────────────────────────────────────────
    PLOT_DIR = OUTPUT_DIR / "plots"
    PLOT_DIR.mkdir(exist_ok=True)
    DIM_LABELS = ["Cond.\nDetection", "Primary\nDiagnosis", "Finding\nComplete.", "False Pos.\nRate", "Clinical\nRelevance", "Overall"]

    colors = {"claude_openrouter": "#E74C3C", "llama_groq": "#3498DB",
              "qwen_hf": "#2ECC71", "our_approach": "#9B59B6"}

    # ── Plot 1: Grouped bar chart (per-model, all UIDs) ───────────────
    if lb_all:
        fig, ax = plt.subplots(figsize=(12, 6))
        models = [m for m in MODEL_DIRS if m in lb_all]
        x = np.arange(len(DIMS))
        width = 0.18
        for i, model in enumerate(models):
            vals = [lb_all[model][d] for d in DIMS]
            bars = ax.bar(x + i * width, vals, width, label=f"{model} (n={counts_all[model]})",
                          color=colors.get(model, "#888"), edgecolor="white", linewidth=0.5)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=7, fontweight="bold")
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(DIM_LABELS, fontsize=9)
        ax.set_ylim(0, 5.5)
        ax.set_ylabel("Score (1-5)", fontsize=11)
        ax.set_title("LLM-as-Judge: Per-Model Evaluation (All Available UIDs)", fontsize=13, fontweight="bold")
        ax.legend(loc="upper right", fontsize=8)
        ax.axhline(y=5, color="gray", linestyle="--", alpha=0.3)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        fig.savefig(PLOT_DIR / "per_model_bar.png", dpi=150)
        plt.close(fig)
        print(f"  Plot saved   : {PLOT_DIR / 'per_model_bar.png'}")

    # ── Plot 2: Radar / spider chart (head-to-head, common UIDs) ──────
    if lb_common:
        dims_no_overall = DIMS[:-1]
        labels_no_overall = DIM_LABELS[:-1]
        models = [m for m in MODEL_DIRS if m in lb_common]
        angles = np.linspace(0, 2 * np.pi, len(dims_no_overall), endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        for model in models:
            vals = [lb_common[model][d] for d in dims_no_overall]
            vals += vals[:1]
            ax.plot(angles, vals, "o-", label=f"{model} (n={counts_common[model]})",
                    color=colors.get(model, "#888"), linewidth=2, markersize=6)
            ax.fill(angles, vals, alpha=0.08, color=colors.get(model, "#888"))
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels_no_overall, fontsize=9)
        ax.set_ylim(0, 5.5)
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.set_title(f"Head-to-Head Comparison ({len(common_uids)} Common UIDs)", fontsize=13,
                     fontweight="bold", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
        plt.tight_layout()
        fig.savefig(PLOT_DIR / "head_to_head_radar.png", dpi=150)
        plt.close(fig)
        print(f"  Plot saved   : {PLOT_DIR / 'head_to_head_radar.png'}")

    # ── Plot 3: Overall score comparison (horizontal bar) ─────────────
    if lb_all:
        fig, ax = plt.subplots(figsize=(8, 4))
        models = sorted(lb_all.keys(), key=lambda m: lb_all[m]["overall_score"])
        scores = [lb_all[m]["overall_score"] for m in models]
        clrs = [colors.get(m, "#888") for m in models]
        bars = ax.barh(models, scores, color=clrs, edgecolor="white", height=0.5)
        for bar, s in zip(bars, scores):
            ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                    f"{s:.2f}", va="center", fontsize=11, fontweight="bold")
        ax.set_xlim(0, 5.5)
        ax.set_xlabel("Overall Score (1-5)", fontsize=11)
        ax.set_title("Overall Score Ranking", fontsize=13, fontweight="bold")
        ax.axvline(x=5, color="gray", linestyle="--", alpha=0.3)
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        fig.savefig(PLOT_DIR / "overall_ranking.png", dpi=150)
        plt.close(fig)
        print(f"  Plot saved   : {PLOT_DIR / 'overall_ranking.png'}")

    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()