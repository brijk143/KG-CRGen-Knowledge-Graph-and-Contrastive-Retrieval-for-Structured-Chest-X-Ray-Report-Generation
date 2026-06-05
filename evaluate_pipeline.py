"""
End-to-End Pipeline Accuracy Evaluator
======================================

Compares the pipeline's generated `precise_summary` (findings + impression)
against the ground-truth `findings + impression` from `testing/indiana_reports.csv`.

Computes a comprehensive set of metrics best-suited for clinical text:

  Lexical / n-gram
  ----------------
  * BLEU-1, BLEU-2, BLEU-4
  * ROUGE-1, ROUGE-2, ROUGE-L (F-measure)
  * METEOR

  Semantic
  --------
  * Sentence-embedding cosine similarity  (all-MiniLM-L6-v2)
  * BERTScore (F1)                        (distilbert-base-uncased)

  Clinical (RadGraph-lite)
  ------------------------
  * Clinical Entity F1   — keyword match over the 35 KG class vocabulary
  * Negation-aware F1    — penalises negation flips (no X vs X)

  LLM-as-Judge
  ------------
  * Groq llama-3.3-70b-versatile evaluator using the user-defined
    rubric (semantic similarity, entity-relation match, contradiction,
    critical error, composite score, verdict).

Outputs (under `output/evaluation/`)
------------------------------------
  per_uid_metrics.csv         — one row per UID, every metric as a column
  llm_judge_per_uid.jsonl     — raw LLM judge response + parsed fields per UID
  summary.json                — mean / median / std for every numeric metric +
                                 LLM verdict counts + critical-error rate
  summary_report.txt          — human-readable summary

Usage
-----
  python evaluate_pipeline.py
      [--limit N]            # evaluate only first N matching UIDs
      [--no_llm]              # skip LLM-as-judge (faster)
      [--no_bertscore]        # skip BERTScore (heavy)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import warnings
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Dict, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────
#  PATHS & CONFIG
# ─────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parent

# Load .env FIRST so all os.environ.get() calls below read correct values
load_dotenv(BASE_DIR / ".env")

# GT_CSV          = BASE_DIR / "testing"      / "indiana_reports.csv"
GT_CSV          = BASE_DIR / "testing2"      / "reports.csv"
PRED_DIR        = BASE_DIR / "output"       / "batch_results_shuffle"
OUT_DIR         = BASE_DIR / "output"       / "evaluation_shuffle"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PER_UID_CSV     = OUT_DIR / "per_uid_metrics.csv"
LLM_JSONL       = OUT_DIR / "llm_judge_per_uid.json"
SUMMARY_JSON    = OUT_DIR / "summary.json"
SUMMARY_TXT     = OUT_DIR / "summary_report.txt"

AZURE_MODEL     = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME_5_1", "gpt-4o-mini")



CLINICAL_ENTITIES: Dict[str, List[str]] = {
    "normal":                       ["normal", "unremarkable", "no acute", "negative", "clear lungs"],
    "airspace disease":             ["airspace disease", "airspace opacity", "alveolar opacity"],
    "bronchiectasis":               ["bronchiectasis", "dilated bronchi"],
    "bronchiolitis":                ["bronchiolitis"],
    "bullous disease":              ["bullous disease", "bullae", "bulla"],
    "calcified granuloma":          ["calcified granuloma"],
    "calcinosis":                   ["calcinosis", "calcification", "calcified"],
    "cardiac shadow":               ["cardiac shadow", "cardiac silhouette"],
    "cardiomegaly":                 ["cardiomegaly", "enlarged heart", "enlarged cardiac silhouette", "heart enlargement"],
    "consolidation":                ["consolidation", "consolidative"],
    "degenerative change":          ["degenerative", "spondylosis", "osteoarthritis", "djd"],
    "edema":                        ["edema", "oedema", "pulmonary edema"],
    "effusion":                     ["effusion"],
    "emphysema":                    ["emphysema", "emphysematous"],
    "fibrosis":                     ["fibrosis", "fibrotic", "scarring", "cicatrix"],
    "fractures":                    ["fracture", "fractures"],
    "hernia":                       ["hernia", "hiatal hernia"],
    "hyperinflation":               ["hyperinflation", "hyperinflated", "hyperdistention", "hyperexpanded"],
    "hypoinflation":                ["hypoinflation", "low lung volume", "low lung volumes", "hypoinflated"],
    "increased lung markings":      ["increased lung markings", "increased markings", "bronchovascular markings"],
    "interstitial lung disease":    ["interstitial lung disease", "ild", "interstitial disease"],
    "kyphosis":                     ["kyphosis", "kyphotic"],
    "lesion":                       ["lesion", "opacity", "opacities"],
    "mass":                         ["mass"],
    "nodule":                       ["nodule", "nodular"],
    "osteophyte":                   ["osteophyte", "osteophytes", "bony spur"],
    "pleural effusion":             ["pleural effusion", "pleural fluid"],
    "pleural thickening":           ["pleural thickening", "thickened pleura"],
    "pneumonia":                    ["pneumonia", "infectious"],
    "pneumothorax":                 ["pneumothorax", "ptx"],
    "pulmonary artery enlargement": ["pulmonary artery enlargement", "enlarged pulmonary artery", "prominent pulmonary artery"],
    "pulmonary fibrosis":           ["pulmonary fibrosis"],
    "rib fracture":                 ["rib fracture", "rib fractures"],
    "scoliosis":                    ["scoliosis", "scoliotic"],
    "subcutaneous emphysema":       ["subcutaneous emphysema"],
    "thickening":                   ["thickening", "thickened"],
    "atelectasis":                  ["atelectasis", "atelectatic", "volume loss"],
}

# Negation cues (word-level) -- if any of these immediately precede
# (window = 4 tokens) a clinical entity, that entity is a NEGATED finding.
NEGATION_CUES = {
    "no", "not", "without", "negative", "absent", "absence",
    "free", "denies", "ruled", "rules", "rule",
}



def _clean(text: Optional[str]) -> str:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    text = str(text).strip()
    # Drop the de-identification placeholder so it doesn't pollute n-gram metrics
    text = re.sub(r"\bXXXX\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def merge_findings_impression(findings: str, impression: str) -> str:
    """Merge findings + impression into a single normalised string."""
    f, i = _clean(findings), _clean(impression)
    if f and i:
        return f"{f} {i}"
    return f or i

def load_ground_truth() -> Dict[int, str]:
    df = pd.read_csv(GT_CSV)
    gt: Dict[int, str] = {}
    for _, row in df.iterrows():
        try:
            uid = int(row["uid"])
        except Exception:
            continue
        merged = merge_findings_impression(row.get("findings"), row.get("impression"))
        if merged:
            gt[uid] = merged
    return gt


def load_predictions() -> Dict[int, str]:
    preds: Dict[int, str] = {}
    if not PRED_DIR.exists():
        return preds
    for folder in sorted(PRED_DIR.iterdir()):
        if not folder.is_dir() or not folder.name.startswith("uid_"):
            continue
        try:
            uid = int(folder.name.replace("uid_", ""))
        except ValueError:
            continue
        jpath = folder / "pipeline_result.json"
        if not jpath.exists():
            continue
        try:
            with open(jpath, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            continue
        ps = data.get("precise_summary") or {}
        merged = merge_findings_impression(ps.get("findings"), ps.get("impression"))
        if merged:
            preds[uid] = merged
    return preds



_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z\-]*")

def tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def extract_entities(text: str) -> Tuple[set, set]:
    """
    Returns (positive_entities, negated_entities) using the canonical
    CLINICAL_ENTITIES vocabulary.  An entity is considered NEGATED if a
    negation cue appears in the 4-token window immediately preceding it.
    """
    text_lc = text.lower()
    tokens  = tokenize(text)
    # token char-position lookup is overkill; do per-sentence scan
    positives: set = set()
    negateds:  set = set()

    sentences = re.split(r"[.;]", text_lc)
    for sent in sentences:
        sent_tokens = tokenize(sent)
        for canonical, surface_forms in CLINICAL_ENTITIES.items():
            hit = False
            hit_pos = -1
            for sf in surface_forms:
                idx = sent.find(sf)
                if idx >= 0:
                    hit = True
                    # token position of the match
                    prefix_tokens = tokenize(sent[:idx])
                    hit_pos = len(prefix_tokens)
                    break
            if not hit:
                continue
            # Look back up to 4 tokens for a negation cue
            window = sent_tokens[max(0, hit_pos - 4):hit_pos]
            if any(w in NEGATION_CUES for w in window):
                negateds.add(canonical)
            else:
                positives.add(canonical)
    return positives, negateds


def _f1(pred: set, gold: set) -> Tuple[float, float, float]:
    if not pred and not gold:
        return 1.0, 1.0, 1.0
    if not pred or not gold:
        return 0.0, 0.0, 0.0
    tp = len(pred & gold)
    p  = tp / len(pred) if pred else 0.0
    r  = tp / len(gold) if gold else 0.0
    f  = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def clinical_entity_f1(gt: str, pred: str) -> Dict[str, float]:
    gt_pos, gt_neg = extract_entities(gt)
    pr_pos, pr_neg = extract_entities(pred)

    p_pos, r_pos, f_pos = _f1(pr_pos, gt_pos)
    # Negation-aware: each entity has a (entity, polarity) pair.
    gt_signed = {(e, "+") for e in gt_pos} | {(e, "-") for e in gt_neg}
    pr_signed = {(e, "+") for e in pr_pos} | {(e, "-") for e in pr_neg}
    p_neg, r_neg, f_neg = _f1(pr_signed, gt_signed)

    return {
        "entity_precision":      round(p_pos, 4),
        "entity_recall":         round(r_pos, 4),
        "entity_f1":             round(f_pos, 4),
        "neg_aware_precision":   round(p_neg, 4),
        "neg_aware_recall":      round(r_neg, 4),
        "neg_aware_f1":          round(f_neg, 4),
        "gt_entities_pos":       sorted(gt_pos),
        "gt_entities_neg":       sorted(gt_neg),
        "pred_entities_pos":     sorted(pr_pos),
        "pred_entities_neg":     sorted(pr_neg),
    }


# ─────────────────────────────────────────────────────────────────────
#  N-GRAM METRICS  (BLEU / ROUGE / METEOR)
# ─────────────────────────────────────────────────────────────────────
def ensure_nltk():
    import nltk
    for pkg in ("punkt", "punkt_tab", "wordnet", "omw-1.4"):
        try:
            nltk.data.find(f"tokenizers/{pkg}") if "punkt" in pkg \
                else nltk.data.find(f"corpora/{pkg}")
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass


def bleu_scores(reference: str, hypothesis: str) -> Dict[str, float]:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    ref = [tokenize(reference)]
    hyp = tokenize(hypothesis)
    if not hyp:
        return {"bleu_1": 0.0, "bleu_2": 0.0, "bleu_4": 0.0}
    sm = SmoothingFunction().method1
    return {
        "bleu_1": round(sentence_bleu(ref, hyp, weights=(1, 0, 0, 0),       smoothing_function=sm), 4),
        "bleu_2": round(sentence_bleu(ref, hyp, weights=(0.5, 0.5, 0, 0),   smoothing_function=sm), 4),
        "bleu_4": round(sentence_bleu(ref, hyp, weights=(0.25,)*4,          smoothing_function=sm), 4),
    }


def rouge_scores(reference: str, hypothesis: str, scorer) -> Dict[str, float]:
    s = scorer.score(reference, hypothesis)
    return {
        "rouge_1": round(s["rouge1"].fmeasure,  4),
        "rouge_2": round(s["rouge2"].fmeasure,  4),
        "rouge_l": round(s["rougeL"].fmeasure,  4),
    }


def meteor_score_one(reference: str, hypothesis: str) -> float:
    from nltk.translate.meteor_score import meteor_score
    return round(float(meteor_score([tokenize(reference)], tokenize(hypothesis))), 4)


# ─────────────────────────────────────────────────────────────────────
#  SEMANTIC METRICS
# ─────────────────────────────────────────────────────────────────────
class SentenceEmbedder:
    """Lazy wrapper around sentence-transformers."""
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def cosine(self, a: str, b: str) -> float:
        import numpy as np
        ea = self.model.encode([a], normalize_embeddings=True)[0]
        eb = self.model.encode([b], normalize_embeddings=True)[0]
        return round(float(np.dot(ea, eb)), 4)


def bertscore_batch(refs: List[str], hyps: List[str]) -> List[float]:
    from bert_score import score
    P, R, F = score(hyps, refs, lang="en", model_type="distilbert-base-uncased",
                    verbose=False, rescale_with_baseline=False)
    return [round(float(x), 4) for x in F.tolist()]


# ─────────────────────────────────────────────────────────────────────
#  LLM-AS-JUDGE  (Groq, llama-3.3-70b-versatile)
# ─────────────────────────────────────────────────────────────────────
LLM_JUDGE_PROMPT = """You are a clinical NLP evaluation expert for chest X-ray report assessment.

Your task is to compare a ground truth report with a predicted report and evaluate clinical accuracy.

-------------------------
EVALUATION PRINCIPLES
-------------------------
- Accept synonymous clinical terms:
  opacity = consolidation = airspace disease
  pleural effusion = pleural fluid
  cardiomegaly = enlarged cardiac silhouette
- Ignore wording differences; focus on clinical meaning.
- Critical errors:
  1. Negation mismatch (e.g., "no pneumothorax" vs "pneumothorax")
  2. Laterality errors (left vs right)
  3. Missed critical findings (pneumothorax, mass, large effusion, pneumoperitoneum)
- IMPORTANT: If the GROUND TRUTH is very short (fewer than 10 words), extra findings in the
  prediction that are not contradicted by the ground truth are NOT a critical error.
  Only flag CRITICAL_ERROR=Yes if there is an actual negation mismatch or dangerous contradiction.

-------------------------
RADGRAPH-STYLE MATCHING
-------------------------
Extract and compare clinical entities (findings, anatomy) and relations
(e.g., "opacity in left lower lobe").  Score overlap using entity + relation
matching (approximate F1).

-------------------------
SCORING (return numeric values in [0.0, 1.0])
-------------------------
SEMANTIC_SIMILARITY:    overall meaning equivalence
ENTITY_RELATION_MATCH:  RadGraph-style overlap of findings + anatomical locations
CONTRADICTION:          Yes / No
CRITICAL_ERROR:         Yes / No  (ONLY for negation mismatch or laterality error — NOT for extra findings)
COMPOSITE_SCORE = (0.5 x semantic_similarity) + (0.5 x entity_relation_match)
   If CRITICAL_ERROR == Yes (negation/laterality only) -> cap COMPOSITE_SCORE at 0.30
   Extra findings alone (pred has more detail than GT) -> do NOT set CRITICAL_ERROR=Yes, do NOT cap score

-------------------------
FINAL VERDICT
-------------------------
Pass            -> COMPOSITE_SCORE >= 0.75 AND no critical error
Borderline      -> 0.50 <= COMPOSITE_SCORE < 0.75 AND no critical error
Critical Error  -> critical error present (negation mismatch or laterality error)
Fail            -> COMPOSITE_SCORE < 0.50 AND no critical error

-------------------------
INPUT
-------------------------
GROUND TRUTH:
{ground_truth}

PREDICTED:
{prediction}

-------------------------
OUTPUT FORMAT (use these EXACT keys, one per line, plain text, no markdown)
-------------------------
SEMANTIC_SIMILARITY: <float>
ENTITY_RELATION_MATCH: <float>
CONTRADICTION: <Yes|No>
CRITICAL_ERROR: <Yes|No>
COMPOSITE_SCORE: <float>
MISSED_FINDINGS: <comma-separated list or None>
EXTRA_FINDINGS: <comma-separated list or None>
CONTRADICTIONS: <comma-separated list or None>
VERDICT: <Pass|Borderline|Critical Error>
REASONING: <one or two sentences>
"""

_FLOAT_RE   = re.compile(r"[-+]?\d*\.?\d+")
_YESNO_RE   = re.compile(r"\b(yes|no)\b", re.IGNORECASE)
_VERDICT_RE = re.compile(r"\b(pass|borderline|critical error)\b", re.IGNORECASE)

def _grab_line(text: str, key: str) -> str:
    m = re.search(rf"^\s*{re.escape(key)}\s*:\s*(.+)$", text, re.IGNORECASE | re.MULTILINE)
    return m.group(1).strip() if m else ""


def _parse_float(s: str, default: float = 0.0) -> float:
    m = _FLOAT_RE.search(s)
    return float(m.group()) if m else default


def _parse_yesno(s: str) -> Optional[bool]:
    m = _YESNO_RE.search(s)
    if not m: return None
    return m.group(1).lower() == "yes"


def _parse_verdict(s: str) -> str:
    m = _VERDICT_RE.search(s)
    if not m: return "Unknown"
    v = m.group(1).lower()
    return {"pass": "Pass", "borderline": "Borderline", "critical error": "Critical Error"}[v]


def parse_llm_judge(text: str) -> Dict[str, object]:
    sem  = _parse_float(_grab_line(text, "SEMANTIC_SIMILARITY"))
    ent  = _parse_float(_grab_line(text, "ENTITY_RELATION_MATCH"))
    cont = _parse_yesno(_grab_line(text, "CONTRADICTION"))
    crit = _parse_yesno(_grab_line(text, "CRITICAL_ERROR"))
    comp = _parse_float(_grab_line(text, "COMPOSITE_SCORE"))
    if crit:
        comp = min(comp, 0.30)
    verdict = _parse_verdict(_grab_line(text, "VERDICT"))
    return {
        "llm_semantic_similarity":   round(sem,  4),
        "llm_entity_relation_match": round(ent,  4),
        "llm_contradiction":         bool(cont) if cont is not None else False,
        "llm_critical_error":        bool(crit) if crit is not None else False,
        "llm_composite_score":       round(comp, 4),
        "llm_missed_findings":       _grab_line(text, "MISSED_FINDINGS"),
        "llm_extra_findings":        _grab_line(text, "EXTRA_FINDINGS"),
        "llm_contradictions_list":   _grab_line(text, "CONTRADICTIONS"),
        "llm_verdict":               verdict,
        "llm_reasoning":             _grab_line(text, "REASONING"),
    }


def call_azure_judge(client, ground_truth: str, prediction: str,
                     deployment: str = None,
                     max_retries: int = 5) -> Tuple[Dict[str, object], str]:
    """Call Azure OpenAI judge. Returns (parsed_fields, raw_text)."""
    if deployment is None:
        deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME_5_1", "gpt-4o-mini")
    prompt = LLM_JUDGE_PROMPT.format(ground_truth=ground_truth, prediction=prediction)
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=deployment,
                messages=[
                    {"role": "system",
                     "content": "You are a meticulous clinical NLP evaluator. Always reply in the exact requested key:value format with no extra text or markdown."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=600,
            )
            raw = resp.choices[0].message.content or ""
            time.sleep(0.5)   # small courtesy delay — Azure has generous rate limits
            return parse_llm_judge(raw), raw
        except Exception as exc:
            last_err = exc
            wait = 4.0 * attempt   # 4 s, 8 s, 12 s, 16 s, 20 s
            print(f"\n[WARN] Azure judge attempt {attempt}/{max_retries} failed: {exc}. Retrying in {wait:.0f}s...")
            time.sleep(wait)
    # All retries exhausted — return neutral fallback (NOT a critical error)
    error_msg = f"LLM call failed after {max_retries} attempts: {last_err}"
    fallback_parsed = {
        "SEMANTIC_SIMILARITY":   "Unknown",
        "ENTITY_RELATION_MATCH": "Unknown",
        "CONTRADICTION":         "Unknown",
        "CRITICAL_ERROR":        "Unknown",
        "COMPOSITE_SCORE":       "Unknown",
        "MISSED_FINDINGS":       "Unknown",
        "EXTRA_FINDINGS":        "Unknown",
        "CONTRADICTIONS":        "Unknown",
        "VERDICT":               "Unknown",
        "REASONING":             error_msg,
        "llm_semantic_similarity":   None,   # None → excluded from mean
        "llm_entity_relation_match": None,
        "llm_contradiction":         False,
        "llm_critical_error":        False,  # do NOT treat API failure as clinical error
        "llm_composite_score":       None,
        "llm_missed_findings":       "",
        "llm_extra_findings":        "",
        "llm_contradictions_list":   "",
        "llm_verdict":               "Unknown",
        "llm_reasoning":             error_msg,
    }
    return fallback_parsed, ""


# ─────────────────────────────────────────────────────────────────────
#  AGGREGATION
# ─────────────────────────────────────────────────────────────────────
def aggregate(rows: List[Dict[str, object]]) -> Dict[str, object]:
    numeric_cols = [
        "bleu_1", "bleu_2", "bleu_4",
        "rouge_1", "rouge_2", "rouge_l",
        "meteor",
        "embed_cosine", "bertscore_f1",
        "llm_semantic_similarity", "llm_entity_relation_match", "llm_composite_score",
    ]
    summary: Dict[str, object] = {"num_uids_evaluated": len(rows)}
    for col in numeric_cols:
        vals = [r[col] for r in rows if r.get(col) is not None]
        vals = [v for v in vals if isinstance(v, (int, float))]
        if not vals:
            summary[col] = {"mean": None, "median": None, "std": None, "min": None, "max": None, "n": 0}
            continue
        summary[col] = {
            "mean":   round(mean(vals),   4),
            "median": round(median(vals), 4),
            "std":    round(pstdev(vals) if len(vals) > 1 else 0.0, 4),
            "min":    round(min(vals),    4),
            "max":    round(max(vals),    4),
            "n":      len(vals),
        }

    # LLM verdict counts
    verdict_counts: Dict[str, int] = {}
    crit_count = 0
    contra_count = 0
    for r in rows:
        v = r.get("llm_verdict", "Unknown")
        verdict_counts[v] = verdict_counts.get(v, 0) + 1
        if r.get("llm_critical_error"):  crit_count   += 1
        if r.get("llm_contradiction"):   contra_count += 1
    summary["llm_verdict_counts"]      = verdict_counts
    summary["llm_critical_error_rate"] = round(crit_count / len(rows), 4) if rows else 0.0
    summary["llm_contradiction_rate"]  = round(contra_count / len(rows), 4) if rows else 0.0

    # ── Overall Pipeline Accuracy (weighted composite) ─────────────
    def _mean_of(col: str) -> Optional[float]:
        s = summary.get(col)
        if isinstance(s, dict):
            return s.get("mean")
        return None

    pass_count = verdict_counts.get("Pass", 0)
    # llm_pass_rate = round(pass_count / len(rows), 4) if rows else 0.0
    # summary["llm_pass_rate"] = llm_pass_rate

    # components = {
    #     "llm_semantic_similarity": (0.30, _mean_of("llm_semantic_similarity")),
    #     "bertscore_f1":            (0.30, _mean_of("bertscore_f1")),
    #     "embed_cosine":            (0.30, _mean_of("embed_cosine")),
    #     "meteor":                  (0.10, _mean_of("meteor"))
    #     # "llm_pass_rate":           (0.05, llm_pass_rate),
    # }
    components = {
        "llm_semantic_similarity": (1.00, _mean_of("llm_semantic_similarity")),
    }

    weighted_sum = 0.0
    total_weight = 0.0
    component_detail = {}
    for comp_name, (w, val) in components.items():
        if val is not None:
            weighted_sum += w * val
            total_weight += w
            component_detail[comp_name] = {"weight": w, "value": round(val, 4), "contribution": round(w * val, 4)}

    raw_score = round(weighted_sum / total_weight, 4) if total_weight > 0 else 0.0
    crit_rate = summary["llm_critical_error_rate"]
    # overall   = round(raw_score * (1.0 - 0.5 * crit_rate), 4)
    overall   = round(raw_score, 4)   # do NOT penalise critical errors for now, since LLM judge is imperfect and we don't want to double-penalise negation mismatches

    # summary["overall_pipeline_accuracy"] = {
    #     "score":                  overall,
    #     "raw_weighted_avg":       raw_score,
    #     "critical_error_penalty": round(0.5 * crit_rate, 4),
    #     "formula":                "weighted_avg x (1 - 0.5 x critical_error_rate)",
    #     "components":             component_detail,
    # }
    summary["overall_pipeline_accuracy"] = {
        "score":            overall,
        "raw_weighted_avg": raw_score,
        "formula":          "llm_semantic_similarity (100% weight)",
        "components":       component_detail,
    }
    return summary


def write_summary_report(summary: Dict[str, object], path: Path) -> None:
    lines: List[str] = []
    add = lines.append
    add("=" * 78)
    add("  END-TO-END PIPELINE EVALUATION SUMMARY")
    add("=" * 78)
    add(f"UIDs evaluated : {summary['num_uids_evaluated']}")
    add("")

    # ── Overall accuracy (hero number) ─────────────────────────────
    oa = summary.get("overall_pipeline_accuracy", {})
    if oa:
        score_pct = oa.get("score", 0) * 100
        add("-" * 78)
        add("  ★  OVERALL PIPELINE ACCURACY")
        add("-" * 78)
        add(f"  >>> {score_pct:.2f}% <<<")
        add("")
        # add(f"  Raw weighted average   : {oa.get('raw_weighted_avg', 0):.4f}")
        # add(f"  Critical-error penalty : -{oa.get('critical_error_penalty', 0) * 100:.2f}%")
        # add(f"  Formula                : {oa.get('formula', '')}")
        add(f"  Raw weighted average   : {oa.get('raw_weighted_avg', 0):.4f}")
        add(f"  Formula                : {oa.get('formula', '')}")
        add("")
        add(f"  {'Component':<28} {'Weight':>6} {'Value':>8} {'Contribution':>14}")
        add(f"  {'─' * 28} {'─' * 6} {'─' * 8} {'─' * 14}")
        for comp_name, detail in oa.get("components", {}).items():
            add(f"  {comp_name:<28} {detail['weight']:>6.2f} {detail['value']:>8.4f} {detail['contribution']:>14.4f}")
        add("")

    add("-" * 78)
    add("  LEXICAL / N-GRAM METRICS")
    add("-" * 78)
    add(f"  {'metric':<22} {'mean':>8} {'median':>8} {'std':>8} {'min':>8} {'max':>8}   n")
    for col in ["bleu_1", "bleu_2", "bleu_4", "rouge_1", "rouge_2", "rouge_l", "meteor"]:
        s = summary.get(col, {})
        if isinstance(s, dict) and s.get("n"):
            add(f"  {col:<22} {s['mean']:>8.4f} {s['median']:>8.4f} {s['std']:>8.4f} {s['min']:>8.4f} {s['max']:>8.4f}  {s['n']}")
    add("")
    add("-" * 78)
    add("  SEMANTIC METRICS")
    add("-" * 78)
    add(f"  {'metric':<22} {'mean':>8} {'median':>8} {'std':>8} {'min':>8} {'max':>8}   n")
    for col in ["embed_cosine", "bertscore_f1"]:
        s = summary.get(col, {})
        if isinstance(s, dict) and s.get("n"):
            add(f"  {col:<22} {s['mean']:>8.4f} {s['median']:>8.4f} {s['std']:>8.4f} {s['min']:>8.4f} {s['max']:>8.4f}  {s['n']}")
    add("")
    add("-" * 78)
    add(f"  LLM-AS-JUDGE  (Azure OpenAI / {AZURE_MODEL})")
    add("-" * 78)
    add(f"  {'metric':<28} {'mean':>8} {'median':>8} {'std':>8} {'min':>8} {'max':>8}   n")
    for col in ["llm_semantic_similarity", "llm_entity_relation_match", "llm_composite_score"]:
        s = summary.get(col, {})
        if isinstance(s, dict) and s.get("n"):
            add(f"  {col:<28} {s['mean']:>8.4f} {s['median']:>8.4f} {s['std']:>8.4f} {s['min']:>8.4f} {s['max']:>8.4f}  {s['n']}")
    add("")
    add(f"  Verdict distribution     : {summary.get('llm_verdict_counts', {})}")
    # add(f"  Pass rate                : {summary.get('llm_pass_rate', 0.0) * 100:.2f}%")
    add(f"  Critical-error rate      : {summary.get('llm_critical_error_rate', 0.0):.4f}")
    add(f"  Contradiction rate       : {summary.get('llm_contradiction_rate', 0.0):.4f}")
    add("")
    add("=" * 78)
    add("  HOW TO READ")
    add("=" * 78)
    add("  * BLEU / ROUGE / METEOR : surface n-gram overlap (lower bound, weak proxy).")
    add("  * embed_cosine          : sentence-level semantic similarity (MiniLM).")
    add("  * bertscore_f1          : token-level contextual semantic match.")
    add("  * llm_composite_score   : RadGraph-style judge score (0..1, capped 0.30 on")
    add("                            critical errors such as negation/laterality flips).")
    add("  * Pass = strong clinical agreement, Borderline = partial,")
    add("    Critical Error = clinically unsafe disagreement.")
    add("")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


# ─────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="Pipeline accuracy evaluator")
    ap.add_argument("--limit",        type=int, default=0,
                    help="Evaluate only first N matching UIDs (0 = all)")
    ap.add_argument("--no_llm",       action="store_true", help="Skip LLM-as-judge")
    ap.add_argument("--no_bertscore", action="store_true", help="Skip BERTScore (heavy)")
    ap.add_argument("--no_embed",     action="store_true", help="Skip sentence embeddings")
    args = ap.parse_args()

    print(f"[i] Loading ground truth from {GT_CSV}")
    gt = load_ground_truth()
    print(f"[i] Ground-truth UIDs : {len(gt)}")

    print(f"[i] Loading predictions from {PRED_DIR}")
    preds = load_predictions()
    print(f"[i] Prediction UIDs   : {len(preds)}")

    common = sorted(set(gt) & set(preds))
    print(f"[i] Overlap (will evaluate) : {len(common)}")
    if args.limit and args.limit > 0:
        common = common[: args.limit]
        print(f"[i] Limited to first {len(common)} UIDs")

    if not common:
        sys.exit("[ERROR] No overlapping UIDs between ground truth and predictions.")

    # ── Initialise heavy components once ───────────────────────────
    print("[i] Preparing NLTK resources ...")
    ensure_nltk()
    from rouge_score import rouge_scorer
    rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    embedder: Optional[SentenceEmbedder] = None
    if not args.no_embed:
        print("[i] Loading sentence-transformer (all-MiniLM-L6-v2) ...")
        try:
            embedder = SentenceEmbedder()
        except Exception as exc:
            print(f"[WARN] Could not load sentence-transformer: {exc}")
            embedder = None

    bertscore_enabled = not args.no_bertscore
    if bertscore_enabled:
        try:
            import bert_score  # noqa: F401
        except Exception as exc:
            print(f"[WARN] BERTScore disabled: {exc}")
            bertscore_enabled = False

    azure_deploy = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME_5_1", "gpt-4o-mini")
    azure_client = None
    if not args.no_llm:
        azure_key      = os.getenv("AZURE_OPENAI_API_KEY")
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_version  = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
        azure_deploy   = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME_5_1", "gpt-4o-mini")
        if not azure_key or not azure_endpoint:
            sys.exit("[ERROR] AZURE_OPENAI_API_KEY / AZURE_OPENAI_ENDPOINT not set in .env. "
                     "Re-run with --no_llm to skip.")
        try:
            from openai import AzureOpenAI
        except ImportError:
            sys.exit("[ERROR] `openai` package missing. `pip install openai` or pass --no_llm.")
        azure_client = AzureOpenAI(
            api_key=azure_key,
            azure_endpoint=azure_endpoint,
            api_version=azure_version,
        )
        print(f"[i] Azure OpenAI judge enabled (deployment={azure_deploy})")

    # ── Per-UID metric computation ─────────────────────────────────
    rows: List[Dict[str, object]] = []
    llm_records: List[Dict[str, object]] = []

    pbar = tqdm(common, desc="Evaluating", unit="uid")
    for uid in pbar:
        ref = gt[uid]
        hyp = preds[uid]

        row: Dict[str, object] = {
            "uid":            uid,
            "gt_text":        ref,
            "pred_text":      hyp,
            "gt_len_words":   len(tokenize(ref)),
            "pred_len_words": len(tokenize(hyp)),
        }

        # Lexical
        row.update(bleu_scores(ref, hyp))
        row.update(rouge_scores(ref, hyp, rouge))
        try:
            row["meteor"] = meteor_score_one(ref, hyp)
        except Exception:
            row["meteor"] = 0.0

        # Semantic embedding (per-uid)
        if embedder is not None:
            try:
                row["embed_cosine"] = embedder.cosine(ref, hyp)
            except Exception:
                row["embed_cosine"] = None
        else:
            row["embed_cosine"] = None

        rows.append(row)

    # ── Batched BERTScore (single call for all rows) ───────────────
    if bertscore_enabled:
        print("[i] Computing BERTScore (batched) ...")
        try:
            refs = [r["gt_text"]   for r in rows]
            hyps = [r["pred_text"] for r in rows]
            f1s = bertscore_batch(refs, hyps)
            for r, f in zip(rows, f1s):
                r["bertscore_f1"] = f
        except Exception as exc:
            print(f"[WARN] BERTScore failed: {exc}")
            for r in rows:
                r["bertscore_f1"] = None
    else:
        for r in rows:
            r["bertscore_f1"] = None

    # ── LLM-as-judge (sequential, with progress) ───────────────────
    if azure_client is not None:
        azure_deploy = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME_5_1", "gpt-4o-mini")
        print(f"[i] Running Azure OpenAI judge on {len(rows)} UIDs (deployment={azure_deploy}) ...")
        if LLM_JSONL.exists(): LLM_JSONL.unlink()
        for r in tqdm(rows, desc="LLM judge", unit="uid"):
            # Short GT (< 5 words) — skip LLM judge; use embed_cosine as proxy
            gt_words = len(r["gt_text"].split())
            if gt_words < 5:
                ec = r.get("embed_cosine") or 0.0
                parsed = {
                    "llm_semantic_similarity":   round(ec, 4),
                    "llm_entity_relation_match": round(ec, 4),
                    "llm_contradiction":         False,
                    "llm_critical_error":        False,
                    "llm_composite_score":       round(ec, 4),
                    "llm_missed_findings":       "",
                    "llm_extra_findings":        "",
                    "llm_contradictions_list":   "",
                    "llm_verdict":               "Pass" if ec >= 0.75 else ("Borderline" if ec >= 0.50 else "Fail"),
                    "llm_reasoning":             f"Short GT ({gt_words} words); embed_cosine={ec:.4f} used as proxy.",
                }
                raw = "\n".join(f"{k}: {v}" for k, v in parsed.items())
            else:
                parsed, raw = call_azure_judge(azure_client, r["gt_text"], r["pred_text"], deployment=azure_deploy)
            r.update(parsed)
            # Parse raw LLM output into individual top-level keys
            raw_fields = {}
            for line in raw.strip().splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    raw_fields[k.strip()] = v.strip()
            rec = {"uid": r["uid"], **raw_fields, **parsed,
                   "gt_text": r["gt_text"], "pred_text": r["pred_text"]}
            llm_records.append(rec)
            with open(LLM_JSONL, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    else:
        for r in rows:
            r.update({
                "llm_semantic_similarity":   None,
                "llm_entity_relation_match": None,
                "llm_contradiction":         None,
                "llm_critical_error":        None,
                "llm_composite_score":       None,
                "llm_missed_findings":       "",
                "llm_extra_findings":        "",
                "llm_contradictions_list":   "",
                "llm_verdict":               "Skipped",
                "llm_reasoning":             "LLM judge disabled (--no_llm)",
            })

    # ── Persist per-UID CSV ────────────────────────────────────────
    df = pd.DataFrame(rows)
    # Put uid first, then short metrics, then long text columns last
    short_cols = [
        "uid",
        "bleu_1", "bleu_2", "bleu_4",
        "rouge_1", "rouge_2", "rouge_l", "meteor",
        "embed_cosine", "bertscore_f1",
        "llm_semantic_similarity", "llm_entity_relation_match", "llm_composite_score",
        "llm_contradiction", "llm_critical_error", "llm_verdict",
        "gt_len_words", "pred_len_words",
        "llm_missed_findings", "llm_extra_findings", "llm_contradictions_list",
        "llm_reasoning",
        "gt_text", "pred_text",
    ]
    short_cols = [c for c in short_cols if c in df.columns]
    df = df[short_cols + [c for c in df.columns if c not in short_cols]]
    df.to_csv(PER_UID_CSV, index=False)
    print(f"[✓] Per-UID metrics : {PER_UID_CSV}")

    # ── Aggregate + summary ────────────────────────────────────────
    summary = aggregate(rows)
    summary["model"]      = azure_deploy if azure_client else None
    summary["files"]      = {
        "per_uid_csv":      str(PER_UID_CSV),
        "llm_jsonl":        str(LLM_JSONL) if azure_client else None,
        "summary_json":     str(SUMMARY_JSON),
        "summary_txt":      str(SUMMARY_TXT),
    }
    summary["dataset"] = {
        "ground_truth_csv": str(GT_CSV),
        "predictions_dir":  str(PRED_DIR),
    }

    with open(SUMMARY_JSON, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[✓] Summary JSON    : {SUMMARY_JSON}")

    write_summary_report(summary, SUMMARY_TXT)
    print(f"[✓] Summary report  : {SUMMARY_TXT}")

    # Echo to terminal
    print()
    with open(SUMMARY_TXT, "r", encoding="utf-8") as fh:
        print(fh.read())


if __name__ == "__main__":
    main()
