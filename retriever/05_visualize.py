"""
05_visualize.py
============
Generates all visualization plots for the CLIP retriever pipeline:

  1. Training & Validation loss curves (from training_log.csv)
  2. Retrieval examples: query image → top-5 retrieved reports (val & test)
  3. t-SNE of image & text embeddings (colored by label)
  4. Similarity score distributions (val vs test)
  5. Per-label recall heatmap

All plots saved to Retriever/plots/
"""

import json
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from collections import Counter, defaultdict

import torch
import torch.nn.functional as F
from PIL import Image
import open_clip
import pandas as pd
from sklearn.manifold import TSNE
from tqdm import tqdm

# ─── Paths ───────────────────────────────────────────────────────────
BASE_DIR = Path("/Users/bkishor/Desktop/kg_new")
LOG_DIR = BASE_DIR / "Retriever" / "logs"
CKPT_DIR = BASE_DIR / "Retriever" / "checkpoints"
DB_DIR = BASE_DIR / "Retriever" / "database"
SPLITS_DIR = BASE_DIR / "Retriever" / "splits"
EVAL_DIR = BASE_DIR / "Retriever" / "evaluation"
PLOT_DIR = BASE_DIR / "Retriever" / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

def plot_loss_curves():
    """Plot training and validation loss curves from training_log.csv."""
    log_path = LOG_DIR / "training_log.csv"
    if not log_path.exists():
        print("⚠️  training_log.csv not found, skipping loss curves")
        return

    df = pd.read_csv(log_path)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss curves
    ax1 = axes[0]
    ax1.plot(df["epoch"], df["train_loss"].astype(float), "b-", label="Train Loss", linewidth=2)
    ax1.plot(df["epoch"], df["val_loss"].astype(float), "r-", label="Val Loss", linewidth=2)
    
    # Mark best epoch
    best_row = df[df["is_best"] == True]
    if not best_row.empty:
        best_epoch = best_row.iloc[-1]["epoch"]
        best_val = float(best_row.iloc[-1]["val_loss"])
        ax1.axvline(x=best_epoch, color="green", linestyle="--", alpha=0.7, label=f"Best (epoch {best_epoch})")
        ax1.scatter([best_epoch], [best_val], color="green", s=100, zorder=5)
    
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("Training & Validation Loss", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Learning rate
    ax2 = axes[1]
    ax2.plot(df["epoch"], df["lr"].astype(float), "purple", linewidth=2)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Learning Rate", fontsize=12)
    ax2.set_title("Learning Rate Schedule", fontsize=14, fontweight="bold")
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3)

    # Logit scale (temperature)
    ax3 = axes[2]
    ax3.plot(df["epoch"], df["logit_scale"].astype(float), "orange", linewidth=2)
    ax3.set_xlabel("Epoch", fontsize=12)
    ax3.set_ylabel("Logit Scale (τ)", fontsize=12)
    ax3.set_title("Temperature Parameter", fontsize=14, fontweight="bold")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = PLOT_DIR / "01_loss_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════
# PLOT 2: RETRIEVAL EXAMPLES
# ══════════════════════════════════════════════════════════════════════
def plot_retrieval_examples(split_name: str, num_examples: int = 4):
    """Show query images with their top-5 retrieved report captions."""
    results_path = EVAL_DIR / f"{split_name}_results.json"
    if not results_path.exists():
        print(f"⚠️  {split_name}_results.json not found, skipping retrieval examples")
        return

    with open(results_path) as f:
        data = json.load(f)
    
    per_uid = data["per_uid_results"]
    
    # Pick examples: 2 with good recall, 2 with poor recall
    good = [r for r in per_uid if r.get("recall@5", 0) == 1]
    bad = [r for r in per_uid if r.get("recall@5", 0) == 0]
    
    examples = []
    if good:
        examples.extend(good[:min(2, len(good))])
    if bad:
        examples.extend(bad[:min(2, len(bad))])
    # Fill remaining with random
    while len(examples) < num_examples and len(per_uid) > len(examples):
        candidate = per_uid[len(examples)]
        if candidate not in examples:
            examples.append(candidate)

    if not examples:
        print(f"⚠️  No results to plot for {split_name}")
        return

    # Load split to get image paths
    with open(SPLITS_DIR / f"{split_name}.json") as f:
        split_data = json.load(f)
    uid_to_images = defaultdict(list)
    for entry in split_data:
        uid_to_images[entry["uid"]].append(entry["image_path"])

    fig, axes = plt.subplots(len(examples), 2, figsize=(16, 5 * len(examples)),
                              gridspec_kw={"width_ratios": [1, 2]})
    if len(examples) == 1:
        axes = [axes]

    for i, result in enumerate(examples):
        uid = result["uid"]
        
        # Show query image
        img_paths = uid_to_images.get(uid, [])
        if img_paths:
            img = Image.open(img_paths[0])
            axes[i][0].imshow(img, cmap="gray")
        
        # Show query caption (ground truth) below image title
        q_caption = result.get("query_caption", "")
        q_labels_str = ", ".join(result["query_labels"][:3]) if result["query_labels"] else "N/A"
        axes[i][0].set_title(f"Query UID: {uid} | Labels: {q_labels_str}\n"
                              f"GT: {q_caption[:80]}...",
                              fontsize=9, fontweight="bold")
        axes[i][0].axis("off")

        # Show top-5 retrieved captions + labels from database
        top5 = result.get("top5_retrieved", [])
        query_label_set = set(l.lower() for l in result.get("query_labels", []))
        text_lines = []
        for rank, r in enumerate(top5, 1):
            match_marker = "[HIT]" if r["is_relevant"] else "[MISS]"
            score = r["score"]
            r_uid = r["uid"]
            # Show the actual retrieved caption (truncated)
            caption = r.get("caption", "(no caption)")
            caption_short = caption[:110] + "..." if len(caption) > 110 else caption
            # Show labels with matching ones highlighted
            r_labels = r.get("labels", [])
            if r_labels:
                label_parts = []
                for lb in r_labels[:5]:
                    if lb.lower() in query_label_set:
                        label_parts.append(f"*{lb}*")  # highlight matching label
                    else:
                        label_parts.append(lb)
                labels_str = ", ".join(label_parts)
            else:
                labels_str = "N/A"
            text_lines.append(
                f"Rank {rank} {match_marker}  Score: {score:.3f}  (UID {r_uid})\n"
                f"    Labels: [{labels_str}]\n"
                f"    {caption_short}"
            )
        
        text_content = "\n\n".join(text_lines)
        axes[i][1].text(0.02, 0.95, text_content, transform=axes[i][1].transAxes,
                         fontsize=8, verticalalignment="top", fontfamily="monospace",
                         bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
        recall_1 = result.get("recall@1", 0)
        recall_5 = result.get("recall@5", 0)
        axes[i][1].set_title(f"Top-5 Retrieved Captions from Database | R@1={recall_1} R@5={recall_5}",
                              fontsize=10, fontweight="bold")
        axes[i][1].axis("off")

    plt.suptitle(f"Retrieval Examples: {split_name.upper()} Set", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    out_path = PLOT_DIR / f"02_retrieval_examples_{split_name}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════
# PLOT 3: t-SNE EMBEDDING VISUALIZATION
# ══════════════════════════════════════════════════════════════════════
def plot_tsne(max_samples: int = 300):
    """
    Encode a subset of images and texts, project to 2D with t-SNE,
    color by label. Shows how well image-text alignment works.
    """
    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Load model
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME)
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    
    ckpt_path = CKPT_DIR / "best_indiana_clip.pt"
    if not ckpt_path.exists():
        print("⚠️  Checkpoint not found, skipping t-SNE")
        return
    
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model = model.to(device)
    model.eval()

    # Load val data (smaller, faster)
    with open(SPLITS_DIR / "val.json") as f:
        val_data = json.load(f)

    # Subsample
    if len(val_data) > max_samples:
        indices = np.random.choice(len(val_data), max_samples, replace=False)
        val_data = [val_data[i] for i in indices]

    # Load data reports for labels (use Problems column for consistency)
    data_reports = pd.read_csv(BASE_DIR / "testing" / "indiana_reports.csv")
    train_reports = pd.read_csv(BASE_DIR / "training" / "training_indiana_reports.csv")
    uid_to_label = {}
    for _, row in list(data_reports.iterrows()) + list(train_reports.iterrows()):
        label = str(row.get("Problems", "unknown"))
        if pd.isna(row.get("Problems")):
            label = "unknown"
        # Take the first label for coloring
        primary = label.split(";")[0].strip()
        uid_to_label[int(row["uid"])] = primary

    # Encode images and texts
    image_features_list = []
    text_features_list = []
    labels = []

    with torch.no_grad():
        for entry in tqdm(val_data, desc="Encoding for t-SNE"):
            # Image
            img = Image.open(entry["image_path"])
            if img.mode != "RGB":
                img = img.convert("RGB")
            img_tensor = preprocess(img).unsqueeze(0).to(device)
            img_feat = model.encode_image(img_tensor)
            img_feat = F.normalize(img_feat, dim=-1)
            image_features_list.append(img_feat.cpu().numpy())

            # Text
            tokens = tokenizer([entry["caption"]]).to(device)
            txt_feat = model.encode_text(tokens)
            txt_feat = F.normalize(txt_feat, dim=-1)
            text_features_list.append(txt_feat.cpu().numpy())

            labels.append(uid_to_label.get(entry["uid"], "unknown"))

    image_features = np.vstack(image_features_list)  # (N, 1024)
    text_features = np.vstack(text_features_list)     # (N, 1024)

    # Combine for joint t-SNE
    combined = np.vstack([image_features, text_features])  # (2N, 1024)
    n_samples = len(image_features)

    print(f"Running t-SNE on {combined.shape[0]} points...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    embeddings_2d = tsne.fit_transform(combined)

    img_2d = embeddings_2d[:n_samples]
    txt_2d = embeddings_2d[n_samples:]

    # Get top-5 most common labels for coloring
    label_counts = Counter(labels)
    top_labels = [l for l, _ in label_counts.most_common(6)]
    
    cmap = plt.cm.get_cmap("tab10")
    label_colors = {label: cmap(i) for i, label in enumerate(top_labels)}
    label_colors["other"] = (0.7, 0.7, 0.7, 0.5)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Plot 1: Images and texts colored by label
    ax = axes[0]
    for i in range(n_samples):
        label = labels[i] if labels[i] in top_labels else "other"
        color = label_colors[label]
        ax.scatter(img_2d[i, 0], img_2d[i, 1], c=[color], marker="o", s=30, alpha=0.6)
        ax.scatter(txt_2d[i, 0], txt_2d[i, 1], c=[color], marker="x", s=30, alpha=0.6)
    
    # Legend
    for label in top_labels:
        ax.scatter([], [], c=[label_colors[label]], marker="o", label=f"{label} (img)")
    ax.scatter([], [], c="gray", marker="x", label="text (×)")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("t-SNE: Image (●) & Text (×) Embeddings", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.2)

    # Plot 2: Show image-text pairs connected by lines
    ax2 = axes[1]
    for i in range(min(50, n_samples)):  # Only draw lines for 50 pairs
        label = labels[i] if labels[i] in top_labels else "other"
        color = label_colors[label]
        ax2.plot([img_2d[i, 0], txt_2d[i, 0]], [img_2d[i, 1], txt_2d[i, 1]],
                 c=color, alpha=0.3, linewidth=0.8)
        ax2.scatter(img_2d[i, 0], img_2d[i, 1], c=[color], marker="o", s=30, alpha=0.6)
        ax2.scatter(txt_2d[i, 0], txt_2d[i, 1], c=[color], marker="x", s=30, alpha=0.6)
    
    ax2.set_title("Image-Text Pairs Connected\n(shorter lines = better alignment)", 
                   fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    out_path = PLOT_DIR / "03_tsne_embeddings.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════
# PLOT 4: SIMILARITY SCORE DISTRIBUTIONS
# ══════════════════════════════════════════════════════════════════════
def plot_similarity_distributions():
    """Compare top-1 similarity score distributions between val and test."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for i, split_name in enumerate(["val", "test"]):
        results_path = EVAL_DIR / f"{split_name}_results.json"
        if not results_path.exists():
            print(f"⚠️  {split_name}_results.json not found")
            continue

        with open(results_path) as f:
            data = json.load(f)

        scores = [r["top1_score"] for r in data["per_uid_results"]]
        
        ax = axes[i]
        ax.hist(scores, bins=30, color="steelblue" if split_name == "val" else "coral",
                alpha=0.8, edgecolor="white")
        ax.axvline(np.mean(scores), color="red", linestyle="--", linewidth=2,
                    label=f"Mean: {np.mean(scores):.3f}")
        ax.axvline(np.median(scores), color="green", linestyle="--", linewidth=2,
                    label=f"Median: {np.median(scores):.3f}")
        ax.set_xlabel("Top-1 Similarity Score", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title(f"{split_name.upper()} Set: Top-1 Similarity Distribution",
                      fontsize=13, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = PLOT_DIR / "04_similarity_distributions.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════
# PLOT 5: PER-LABEL RECALL ANALYSIS
# ══════════════════════════════════════════════════════════════════════
def plot_per_label_recall():
    """Bar chart comparing Recall@5 per label category for val and test."""
    fig, ax = plt.subplots(figsize=(14, 6))

    label_recalls = {}

    for split_name in ["val", "test"]:
        results_path = EVAL_DIR / f"{split_name}_results.json"
        if not results_path.exists():
            continue

        with open(results_path) as f:
            data = json.load(f)

        # Aggregate recall@5 per primary label
        per_label = defaultdict(list)
        for r in data["per_uid_results"]:
            q_labels = r.get("query_labels", [])
            primary = q_labels[0] if q_labels else "unknown"
            per_label[primary].append(r.get("recall@5", 0))

        label_recalls[split_name] = {
            label: np.mean(recalls) for label, recalls in per_label.items()
            if len(recalls) >= 3  # Only labels with ≥3 samples
        }

    # Get common labels
    all_labels = sorted(set(label_recalls.get("val", {}).keys()) | 
                        set(label_recalls.get("test", {}).keys()))

    if not all_labels:
        print("⚠️  No label data for per-label recall plot")
        return

    x = np.arange(len(all_labels))
    width = 0.35

    val_recalls = [label_recalls.get("val", {}).get(l, 0) for l in all_labels]
    test_recalls = [label_recalls.get("test", {}).get(l, 0) for l in all_labels]

    ax.bar(x - width/2, val_recalls, width, label="Validation", color="steelblue", alpha=0.8)
    ax.bar(x + width/2, test_recalls, width, label="Test", color="coral", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Recall@5", fontsize=12)
    ax.set_title("Recall@5 by Label Category (Val vs Test)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    out_path = PLOT_DIR / "05_per_label_recall.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════
# PLOT 6: METRICS COMPARISON BAR CHART
# ══════════════════════════════════════════════════════════════════════
def plot_metrics_comparison():
    """Side-by-side grouped bar chart of all metrics (accuracy, precision, F1 @K, MRR, avg sim) for val vs test."""
    summary_path = EVAL_DIR / "comparison_summary.csv"
    if not summary_path.exists():
        print("⚠️  comparison_summary.csv not found")
        return

    df = pd.read_csv(summary_path)

    # Ordered metric names to display
    metric_names = [
        "accuracy@1", "accuracy@3", "accuracy@5", "accuracy@10",
        "precision@1", "precision@3", "precision@5", "precision@10",
        "f1@1", "f1@3", "f1@5", "f1@10",
        "mrr", "avg_top1_score",
    ]

    df_metrics = df[df["metric"].isin(metric_names)].copy()
    # Preserve the desired order
    df_metrics["metric"] = pd.Categorical(df_metrics["metric"], categories=metric_names, ordered=True)
    df_metrics = df_metrics.sort_values("metric")
    df_metrics["validation"] = df_metrics["validation"].astype(float)
    df_metrics["test"] = df_metrics["test"].astype(float)

    # ── Grouped bar chart ──
    fig, ax = plt.subplots(figsize=(16, 6))
    x = np.arange(len(df_metrics))
    width = 0.35

    bars_v = ax.bar(x - width / 2, df_metrics["validation"], width,
                    label="Validation", color="steelblue", alpha=0.85)
    bars_t = ax.bar(x + width / 2, df_metrics["test"], width,
                    label="Test", color="coral", alpha=0.85)

    # Value labels on bars
    for j, (v, t) in enumerate(zip(df_metrics["validation"], df_metrics["test"])):
        ax.text(j - width / 2, v + 0.012, f"{v:.3f}", ha="center", fontsize=7, fontweight="bold")
        ax.text(j + width / 2, t + 0.012, f"{t:.3f}", ha="center", fontsize=7, fontweight="bold")

    # Vertical separators between metric families
    for sep in [4, 8, 12]:
        if sep < len(df_metrics):
            ax.axvline(x=sep - 0.5, color="grey", linewidth=0.7, linestyle="--", alpha=0.5)

    # Family labels across the top
    family_labels = [("Accuracy @K", 0, 3), ("Precision @K", 4, 7),
                     ("F1 @K", 8, 11), ("Other", 12, 13)]
    for label, s, e in family_labels:
        if e < len(df_metrics):
            mid = (s + e) / 2
            ax.text(mid, ax.get_ylim()[1] * 0.97, label,
                    ha="center", fontsize=9, fontstyle="italic", color="dimgrey")

    ax.set_xticks(x)
    ax.set_xticklabels(df_metrics["metric"], rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Validation vs Test — Full Retrieval Metrics (Accuracy / Precision / F1 @K)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, max(df_metrics["validation"].max(), df_metrics["test"].max()) * 1.15)

    plt.tight_layout()
    out_path = PLOT_DIR / "06_metrics_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)

    print("\n📊 Plot 1: Loss Curves")
    plot_loss_curves()

    print("\n📊 Plot 2: Retrieval Examples (Validation)")
    plot_retrieval_examples("val")

    print("\n📊 Plot 3: Retrieval Examples (Test)")
    plot_retrieval_examples("test")

    print("\n📊 Plot 4: t-SNE Embeddings")
    plot_tsne()

    print("\n📊 Plot 5: Similarity Distributions")
    plot_similarity_distributions()

    print("\n📊 Plot 6: Per-Label Recall")
    plot_per_label_recall()

    print("\n📊 Plot 7: Metrics Comparison")
    plot_metrics_comparison()

    print(f"\n{'='*60}")
    print(f"ALL PLOTS SAVED TO: {PLOT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
