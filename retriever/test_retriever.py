"""
inference.py
============
Given a new chest X-ray image (never seen during training), retrieves the
top-K most similar reports from the training database.

Usage:
  python inference.py --image /path/to/xray.png --top_k 5
  python inference.py --uid 3500  # looks up image from test split
"""

import json
import argparse
import numpy as np
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
import open_clip
BASE_DIR = Path("/Users/bkishor/Desktop/kg_new")
CKPT_DIR = BASE_DIR / "Retriever" / "checkpoints"
DB_DIR = BASE_DIR / "Retriever" / "database"
SPLITS_DIR = BASE_DIR / "Retriever" / "splits"

MODEL_NAME = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


def load_model_and_database(device):
    """Load fine-tuned CLIP model and text embedding database."""
    # Load model
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME)
    ckpt = torch.load(CKPT_DIR / "best_indiana_clip.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Load database
    vectors = np.load(DB_DIR / "indiana_database_vectors.npy")
    with open(DB_DIR / "indiana_database_metadata.json") as f:
        metadata = json.load(f)

    # Convert to tensor for fast similarity computation
    db_vectors = torch.from_numpy(vectors).to(device)

    return model, preprocess, db_vectors, metadata


@torch.no_grad()
def retrieve(model, preprocess, image_path, db_vectors, metadata, device, top_k=5):
    """
    Encode a query image and retrieve top-K similar reports.
    
    Returns list of dicts with uid, caption, similarity_score, rank.
    """
    # Load and preprocess image
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    # Encode image
    image_features = model.encode_image(image_tensor)
    image_features = F.normalize(image_features, dim=-1)

    # Compute similarities: (1, 1024) @ (1024, N) → (1, N)
    similarities = (image_features @ db_vectors.T).squeeze(0)  # (N,)
    
    # Get top-K
    top_k_vals, top_k_indices = torch.topk(similarities, min(top_k, len(metadata)))

    results = []
    for rank, (score, idx) in enumerate(zip(top_k_vals.cpu().numpy(), 
                                             top_k_indices.cpu().numpy()), 1):
        entry = metadata[idx]
        results.append({
            "rank": rank,
            "uid": entry["uid"],
            "caption": entry["caption"],
            "similarity": float(score),
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Retrieve similar reports for a chest X-ray")
    parser.add_argument("--image", type=str, help="Path to chest X-ray image")
    parser.add_argument("--uid", type=int, help="UID from test split to use as query")
    parser.add_argument("--top_k", type=int, default=5, help="Number of results to return")
    args = parser.parse_args()

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")
    print("Loading model and database...")
    model, preprocess, db_vectors, metadata = load_model_and_database(device)
    print(f"Database: {len(metadata)} reports")

    # Determine query image
    if args.image:
        image_path = args.image
    elif args.uid:
        # Look up from test split
        with open(SPLITS_DIR / "test.json") as f:
            test_data = json.load(f)
        matches = [e for e in test_data if e["uid"] == args.uid]
        if not matches:
            print(f"UID {args.uid} not found in test split!")
            return
        image_path = matches[0]["image_path"]  # Use first image for this UID
        print(f"\nQuery UID: {args.uid}")
        print(f"Ground truth caption: {matches[0]['caption'][:100]}...")
    else:
        print("Please provide --image or --uid")
        return
    print(f"Query image: {image_path}\n")
    results = retrieve(model, preprocess, image_path, db_vectors, metadata, device, args.top_k)
    print(f"TOP-{args.top_k} RETRIEVED REPORTS")
    print(f"{'='*70}")
    for r in results:
        print(f"\n  RANK {r['rank']} | Score: {r['similarity']:.4f} | UID: {r['uid']}")
        print(f"  Caption: {r['caption']}...")


if __name__ == "__main__":
    main()
