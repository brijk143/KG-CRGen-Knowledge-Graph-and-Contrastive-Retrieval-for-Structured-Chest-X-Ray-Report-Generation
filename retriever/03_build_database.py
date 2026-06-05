"""
03_build_database.py
=================
After training, loads the best checkpoint and builds a HYBRID database
containing BOTH text and image embeddings per UID.

Hybrid retrieval strategy:
  - Text DB : encode training report captions → text embeddings
  - Image DB: encode training images (frontal+lateral averaged) → image embeddings
  At query time, 04_evaluate.py blends:
    score = TEXT_WEIGHT * text_similarity + IMAGE_WEIGHT * image_similarity
  Image-to-image similarity captures visual appearance directly;
  text similarity captures semantic/diagnostic labels.

Output:
  - indiana_database_vectors.npy       → (N, D) text embeddings  (text DB)
  - indiana_database_img_vectors.npy   → (N, D) image embeddings (image DB)
  - indiana_database_metadata.json     → [{uid, caption}, ...] aligned with both
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
from PIL import Image
import open_clip
from tqdm import tqdm

BASE_DIR = Path("/Users/bkishor/Desktop/kg_new")
SPLITS_DIR = BASE_DIR / "Retriever" / "splits"
CKPT_DIR = BASE_DIR / "Retriever" / "checkpoints"
DB_DIR = BASE_DIR / "Retriever" / "database"
DB_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"  # Must match 02_train.py

# Blend weight for text vs image similarity at retrieval time.
# TEXT_DB_WEIGHT + IMAGE_DB_WEIGHT = 1.0
TEXT_DB_WEIGHT  = 0.6
IMAGE_DB_WEIGHT = 0.4


def main():
    # ─── Device ──────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # ─── Load model + checkpoint ─────────────────────────────
    print(f"\nLoading model: {MODEL_NAME}")
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME)
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)

    ckpt_path = CKPT_DIR / "best_indiana_clip.pt"
    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"Checkpoint from epoch {checkpoint['epoch']}, "
          f"val_loss={checkpoint['val_loss']:.4f}")

    # ─── Load training data ─────────────────────────────────
    with open(SPLITS_DIR / "train.json") as f:
        train_data = json.load(f)

    # Group images by UID; keep first-seen caption
    uid_caption = {}
    uid_images = defaultdict(list)
    for entry in train_data:
        uid = entry["uid"]
        if uid not in uid_caption:
            uid_caption[uid] = entry["caption"]
        uid_images[uid].append(entry["image_path"])

    uids = sorted(uid_caption.keys())
    captions = [uid_caption[uid] for uid in uids]
    print(f"\nUnique UIDs in training set: {len(uids)}")

    # ─── Encode all captions (text DB) ────────────────────────
    print("\nEncoding captions (text DB)...")
    text_vectors = []
    batch_size = 64

    with torch.no_grad():
        for i in tqdm(range(0, len(captions), batch_size)):
            batch_captions = captions[i : i + batch_size]
            tokens = tokenizer(batch_captions).to(device)
            feat = model.encode_text(tokens)
            feat = F.normalize(feat, dim=-1)
            text_vectors.append(feat.cpu().numpy())

    text_db = np.vstack(text_vectors).astype(np.float32)
    print(f"Text DB vectors shape: {text_db.shape}")

    # ─── Encode all training images (image DB) ────────────────
    # For each UID: average all available image embeddings (frontal + lateral)
    print("\nEncoding training images (image DB)...")
    img_vectors = []

    with torch.no_grad():
        for uid in tqdm(uids):
            img_paths = uid_images[uid]
            view_feats = []
            for img_path in img_paths:
                try:
                    img = Image.open(img_path).convert("RGB")
                    tensor = preprocess(img).unsqueeze(0).to(device)
                    feat = model.encode_image(tensor)
                    feat = F.normalize(feat, dim=-1)
                    view_feats.append(feat)
                except Exception:
                    continue
            if view_feats:
                avg_feat = F.normalize(torch.mean(torch.stack(view_feats), dim=0), dim=-1)
                img_vectors.append(avg_feat.cpu().numpy())
            else:
                # Fallback: zero vector (will score poorly but won't crash)
                dim = text_db.shape[1]
                img_vectors.append(np.zeros((1, dim), dtype=np.float32))

    img_db = np.vstack(img_vectors).astype(np.float32)
    print(f"Image DB vectors shape: {img_db.shape}")

    # ─── Save ────────────────────────────────────────────────
    text_path = DB_DIR / "indiana_database_vectors.npy"
    np.save(text_path, text_db)
    print(f"✅ Saved text vectors:  {text_path}")

    img_path = DB_DIR / "indiana_database_img_vectors.npy"
    np.save(img_path, img_db)
    print(f"✅ Saved image vectors: {img_path}")

    metadata = [{"uid": uid, "caption": uid_caption[uid]} for uid in uids]
    metadata_path = DB_DIR / "indiana_database_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Saved metadata:      {metadata_path}")

    # Save blend weights alongside DB so 04_evaluate.py can read them
    weights_path = DB_DIR / "blend_weights.json"
    with open(weights_path, "w") as f:
        json.dump({"text": TEXT_DB_WEIGHT, "image": IMAGE_DB_WEIGHT}, f)
    print(f"✅ Saved blend weights: {weights_path}  (text={TEXT_DB_WEIGHT}, image={IMAGE_DB_WEIGHT})")

    # ─── Quick sanity check ──────────────────────────────────
    print(f"\n--- Sanity Check ---")
    print(f"Text  vector norms: min={np.linalg.norm(text_db, axis=1).min():.4f}  "
          f"max={np.linalg.norm(text_db, axis=1).max():.4f}")
    print(f"Image vector norms: min={np.linalg.norm(img_db,  axis=1).min():.4f}  "
          f"max={np.linalg.norm(img_db,  axis=1).max():.4f}")
    print(f"\nText DB  mean pairwise cos sim: "
          f"{(text_db[:200] @ text_db[:200].T - np.eye(200)).mean():.4f}")
    print(f"Image DB mean pairwise cos sim: "
          f"{(img_db[:200] @ img_db[:200].T - np.eye(200)).mean():.4f}")


if __name__ == "__main__":
    main()
