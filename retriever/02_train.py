"""
02_train.py
========
Fine-tunes BiomedCLIP (ViT-B/16, medical domain) on Indiana chest X-ray
data using contrastive learning (symmetric cross-entropy).

Key improvements over RN50 baseline:
  - BiomedCLIP ViT-B/16 pre-trained on 15M biomedical image-text pairs
  - Unfreezes last 4 transformer blocks of the ViT visual encoder
  - Larger effective batch (gradient accumulation) for better contrastive signal
  - Early stopping patience=5 to prevent overfitting past best val_loss
  - Higher LR (1e-5) for trainable ViT blocks

Anti-overfitting measures:
  - Only last 4 ViT blocks + projection + text encoder are trainable
  - Data augmentation: random resized crop, horizontal flip, color jitter
  - Gradient clipping + weight decay (0.01)
  - Cosine LR schedule with warmup
"""

import os
import json
import time
import csv
import math
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import open_clip
from torchvision import transforms as T
from tqdm import tqdm
BASE_DIR = Path("/Users/bkishor/Desktop/kg_new")
SPLITS_DIR = BASE_DIR / "Retriever" / "splits"
OUTPUT_DIR = BASE_DIR / "Retriever" / "checkpoints"
LOG_DIR = BASE_DIR / "Retriever" / "logs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_NAME = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

# Gradient accumulation: effective batch = batch_size * GRAD_ACCUM_STEPS
# Larger effective batch = more negatives per positive = better contrastive signal
# 4 steps × 64 MPS batch = 256 effective batch
GRAD_ACCUM_STEPS = 4

# Unfreeze last N ViT blocks.  6 blocks gives more capacity than 4 while
# still keeping early feature extraction frozen from biomedical pre-training.
UNFREEZE_LAST_N_BLOCKS = 6
class ChestXrayDataset(Dataset):
    """
    Loads image-caption pairs from a JSON split file.
    Each item returns (preprocessed_image_tensor, tokenized_text_tensor).
    
    Grayscale images are converted to RGB by repeating across 3 channels.
    Uses a resize cache (256×256 JPEG) to avoid repeatedly loading 2048×2048 images.
    """
    CACHE_DIR = BASE_DIR / "Retriever" / ".image_cache"

    def __init__(self, json_path: str, preprocess, tokenizer, 
                 use_cache: bool = True, augment: bool = False):
        with open(json_path) as f:
            self.data = json.load(f)
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.use_cache = use_cache
        
        # Data augmentation for training
        # BiomedCLIP uses standard ImageNet normalization (same as ViT-B/16)
        if augment:
            self.augment_transform = T.Compose([
                T.RandomResizedCrop(224, scale=(0.75, 1.0)),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.15, contrast=0.15),
                T.RandomAffine(degrees=5, translate=(0.05, 0.05)),  # slight rotation/shift for X-rays
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
            ])
        else:
            self.augment_transform = None

        if use_cache:
            self.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _get_cached_path(self, image_path: str) -> Path:
        """Generate cache file path from original image path."""
        # Use filename as cache key (filenames are unique)
        name = Path(image_path).stem + ".jpg"
        return self.CACHE_DIR / name

    def _load_image(self, image_path: str) -> Image.Image:
        """Load image with optional disk cache of resized version."""
        if self.use_cache:
            cache_path = self._get_cached_path(image_path)
            if cache_path.exists():
                # Load from cache (already 256×256 RGB JPEG)
                return Image.open(cache_path).convert("RGB")
            else:
                # Load original, resize, save to cache
                img = Image.open(image_path)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                # Resize to 256×256 (slightly larger than 224 for augmentation headroom)
                img_resized = img.resize((256, 256), Image.LANCZOS)
                img_resized.save(cache_path, "JPEG", quality=95)
                return img_resized
        else:
            img = Image.open(image_path)
            if img.mode != "RGB":
                img = img.convert("RGB")
            return img

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        
        # Load image (from cache if available)
        image = self._load_image(entry["image_path"])
        
        # Use augmentation for training, standard preprocess otherwise
        if self.augment_transform is not None:
            image = self.augment_transform(image)
        else:
            image = self.preprocess(image)
        
        # Tokenize caption (auto-truncates to 77 tokens)
        text = self.tokenizer([entry["caption"]])[0]
        
        return image, text

def clip_contrastive_loss(image_features, text_features, logit_scale, label_smoothing: float = 0.1):
    """
    Symmetric InfoNCE / CLIP contrastive loss with label smoothing.
    
    image_features: (B, D) L2-normalized
    text_features:  (B, D) L2-normalized
    logit_scale:    learnable temperature (scalar)
    label_smoothing: smoothing factor (0.1 = 10% mass on negatives)
    
    For batch size B, the similarity matrix is B×B.
    Diagonal entries are positives (image_i matches text_i).
    Loss = average of image→text CE + text→image CE.
    
    Label smoothing prevents the model from becoming over-confident,
    which prevents the embedding space from collapsing to near-uniform
    high cosine similarities.
    """
    # Similarity matrix: (B, B)
    logits_per_image = logit_scale * image_features @ text_features.T
    logits_per_text = logits_per_image.T

    # Labels: image_i should match text_i → diagonal
    batch_size = image_features.shape[0]
    labels = torch.arange(batch_size, device=image_features.device)

    # Symmetric cross-entropy with label smoothing
    loss_i2t = F.cross_entropy(logits_per_image, labels, label_smoothing=label_smoothing)
    loss_t2i = F.cross_entropy(logits_per_text, labels, label_smoothing=label_smoothing)

    return (loss_i2t + loss_t2i) / 2.0
class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Cosine decay with linear warmup."""
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-7, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            scale = self.last_epoch / max(1, self.warmup_steps)
        else:
            # Cosine decay
            progress = (self.last_epoch - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            scale = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [max(base_lr * scale, self.min_lr) for base_lr in self.base_lrs]


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Compute average contrastive loss on a dataset (no gradients)."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for images, texts in dataloader:
        images = images.to(device)
        texts = texts.to(device)

        image_features = model.encode_image(images)
        text_features = model.encode_text(texts)

        # L2 normalize
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        logit_scale = model.logit_scale.exp()
        loss = clip_contrastive_loss(image_features, text_features, logit_scale)
        total_loss += loss.item()
        num_batches += 1

    model.train()
    return total_loss / max(1, num_batches)


def train(args):
    # ─── Device selection ─────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = "CUDA GPU"
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        device_name = "Apple MPS GPU"
    else:
        device = torch.device("cpu")
        device_name = "CPU"

    print(f"\n{'='*60}")
    print(f"DEVICE: {device_name}")
    print(f"{'='*60}")

    # Adjust hyperparameters based on device
    # Larger batches give more negatives per positive → better contrastive signal
    if device.type == "cuda":
        batch_size = args.batch_size or 128
        num_epochs = args.epochs or 60
        num_workers = 4
    elif device.type == "mps":
        batch_size = args.batch_size or 64
        num_epochs = args.epochs or 40
        num_workers = 0  # MPS works best with 0 workers
    else:
        batch_size = args.batch_size or 32
        num_epochs = args.epochs or 30
        num_workers = 0

    grad_accum_steps = GRAD_ACCUM_STEPS  # effective batch = batch_size * grad_accum_steps
    lr = args.lr or 5e-6               # lower LR for 6-block ViT fine-tuning (prevents overfit)
    weight_decay = args.weight_decay or 0.01
    patience = args.patience or 5      # stop early — val_loss diverges fast

    print(f"Grad accumulation steps: {grad_accum_steps}  (effective batch: {batch_size * grad_accum_steps})")

    print(f"Batch size: {batch_size}")
    print(f"Epochs: {num_epochs}")
    print(f"Learning rate: {lr}")
    print(f"Weight decay: {weight_decay}")
    print(f"Early stopping patience: {patience}")

    # ─── Load model ───────────────────────────────────────────
    print(f"\nLoading model: {MODEL_NAME}")
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME)
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    model = model.to(device)
    model.train()

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Total params: {total_params:.1f}M | Trainable (before freeze): {trainable_params:.1f}M")

    # ─── Selective unfreezing for BiomedCLIP ViT ──────────────
    # Strategy: freeze early ViT blocks (low-level features already good from
    # biomedical pre-training), unfreeze last 6 blocks to adapt to X-ray retrieval.
    # Text encoder is fully trainable to align with image features.
    #
    # ViT-B/16 has 12 transformer blocks (0-11). We unfreeze blocks 6-11.

    # First freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze text encoder entirely (aligns text with fine-tuned image features)
    if hasattr(model, "text"):
        for param in model.text.parameters():
            param.requires_grad = True
    for name, param in model.named_parameters():
        if name.startswith("text") or "token_embedding" in name or "positional_embedding" in name:
            param.requires_grad = True

    # Unfreeze last N ViT transformer blocks in visual encoder
    # BiomedCLIP uses TimmModel wrapper: visual.trunk (VisionTransformer) → .blocks
    visual = model.visual
    vit_blocks = None

    # BiomedCLIP / TimmModel layout: visual.trunk.blocks
    if hasattr(visual, "trunk") and hasattr(visual.trunk, "blocks"):
        vit_blocks = visual.trunk.blocks
    # Standard open_clip ViT layout: visual.transformer.resblocks
    elif hasattr(visual, "transformer") and hasattr(visual.transformer, "resblocks"):
        vit_blocks = visual.transformer.resblocks
    # Fallback
    elif hasattr(visual, "blocks"):
        vit_blocks = visual.blocks

    if vit_blocks is not None:
        num_blocks = len(vit_blocks)
        for i, block in enumerate(vit_blocks):
            if i >= num_blocks - UNFREEZE_LAST_N_BLOCKS:
                for param in block.parameters():
                    param.requires_grad = True
        print(f"  ViT blocks: {num_blocks} total | unfreezing last {UNFREEZE_LAST_N_BLOCKS} (blocks {num_blocks-UNFREEZE_LAST_N_BLOCKS}-{num_blocks-1})")
    else:
        print("  WARNING: Could not find ViT transformer blocks — unfreezing all visual")
        for param in visual.parameters():
            param.requires_grad = True

    # Always unfreeze final visual norm + projection head
    # BiomedCLIP: norm is visual.trunk.norm, projection is visual.head
    for attr in ["ln_post", "proj", "ln_pre", "norm"]:
        for container in [visual, getattr(visual, "trunk", None)]:
            if container is not None and hasattr(container, attr):
                m = getattr(container, attr)
                if isinstance(m, torch.nn.Module):
                    for param in m.parameters():
                        param.requires_grad = True
                elif isinstance(m, torch.nn.Parameter):
                    m.requires_grad = True

    # BiomedCLIP: visual.head is the final projection Sequential — always unfreeze
    if hasattr(visual, "head"):
        for param in visual.head.parameters():
            param.requires_grad = True

    # Unfreeze logit_scale
    model.logit_scale.requires_grad = True

    frozen_count   = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Frozen: {frozen_count/1e6:.1f}M params | Trainable: {trainable_count/1e6:.1f}M params")

    # ─── Load datasets ───────────────────────────────────────
    print("\nLoading datasets...")
    train_dataset = ChestXrayDataset(SPLITS_DIR / "train.json", preprocess, tokenizer, augment=True)
    val_dataset = ChestXrayDataset(SPLITS_DIR / "val.json", preprocess, tokenizer)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
        drop_last=True  # Important: last incomplete batch can cause issues with contrastive loss
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
        drop_last=False
    )

    print(f"Train: {len(train_dataset)} pairs → {len(train_loader)} batches (grad_accum={grad_accum_steps}, effective_batch={batch_size * grad_accum_steps})")
    print(f"Val:   {len(val_dataset)} pairs → {len(val_loader)} batches")

    # ─── Optimizer & Scheduler ────────────────────────────────
    # Separate param groups: ViT visual blocks get lower lr, text + proj get full lr
    trainable_params_list = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    visual_block_params = [p for n, p in trainable_params_list
                           if "visual" in n and "transformer" in n and "logit_scale" not in n]
    other_params        = [p for n, p in trainable_params_list
                           if not ("visual" in n and "transformer" in n) and "logit_scale" not in n]
    params = [
        {"params": visual_block_params, "lr": lr * 0.5, "weight_decay": weight_decay},  # ViT blocks: half LR
        {"params": other_params,        "lr": lr,       "weight_decay": weight_decay},  # text + proj: full LR
        {"params": [model.logit_scale], "lr": lr * 5,  "weight_decay": 0.0},           # logit_scale: fast
    ]
    optimizer = torch.optim.AdamW(params)

    total_steps = num_epochs * len(train_loader)
    warmup_steps = min(len(train_loader) * 2, total_steps // 10)  # ~2 epochs or 10%
    scheduler = CosineWarmupScheduler(optimizer, warmup_steps, total_steps)

    # ─── Training log ─────────────────────────────────────────
    log_path = LOG_DIR / "training_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "lr", "logit_scale",
                         "epoch_time_sec", "is_best"])

    # ─── Training loop ────────────────────────────────────────
    best_val_loss = float("inf")
    patience_counter = 0
    best_epoch = -1

    print(f"\n{'='*60}")
    print("TRAINING STARTED")
    print(f"{'='*60}\n")

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        model.train()
        total_train_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d}/{num_epochs}",
                     leave=True, ncols=100)

        optimizer.zero_grad()  # zero once before accumulation loop
        for step_idx, (images, texts) in enumerate(pbar):
            images = images.to(device)
            texts = texts.to(device)

            # Forward pass
            image_features = model.encode_image(images)
            text_features = model.encode_text(texts)

            # L2 normalize
            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)

            # Loss (scale by accum steps so gradients average correctly)
            logit_scale = model.logit_scale.exp()
            # Clamp logit_scale to prevent instability
            model.logit_scale.data = torch.clamp(model.logit_scale.data, max=math.log(100.0))

            loss = clip_contrastive_loss(image_features, text_features, logit_scale)
            loss_scaled = loss / grad_accum_steps
            loss_scaled.backward()

            # Update weights every grad_accum_steps mini-batches
            if (step_idx + 1) % grad_accum_steps == 0 or (step_idx + 1) == len(train_loader):
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_train_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                "τ": f"{logit_scale.item():.2f}"
            })

        avg_train_loss = total_train_loss / max(1, num_batches)

        # ─── Validation ───────────────────────────────────────
        val_loss = evaluate(model, val_loader, device)
        epoch_time = time.time() - epoch_start

        # ─── Check for best ───────────────────────────────────
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0

            # Save best checkpoint
            ckpt_path = OUTPUT_DIR / "best_indiana_clip.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "train_loss": avg_train_loss,
                "logit_scale": model.logit_scale.item(),
                "model_name": MODEL_NAME,
                "grad_accum_steps": grad_accum_steps,
                "unfreeze_last_n_blocks": UNFREEZE_LAST_N_BLOCKS,
            }, ckpt_path)
            print(f" Best model saved! (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1

        # ─── Log ──────────────────────────────────────────────
        current_lr = scheduler.get_last_lr()[0]
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{avg_train_loss:.6f}", f"{val_loss:.6f}",
                             f"{current_lr:.2e}", f"{logit_scale.item():.4f}",
                             f"{epoch_time:.1f}", is_best])

        print(f"  Epoch {epoch:3d} | Train: {avg_train_loss:.4f} | Val: {val_loss:.4f} | "
              f"LR: {current_lr:.2e} | τ: {logit_scale.item():.2f} | "
              f"Time: {epoch_time:.0f}s | "
              f"{'★ BEST' if is_best else f'patience {patience_counter}/{patience}'}")

        # ─── Early stopping ───────────────────────────────────
        if patience_counter >= patience:
            print(f"\n Early stopping at epoch {epoch}! "
                  f"Best val_loss={best_val_loss:.4f} at epoch {best_epoch}")
            break

    # ─── Final summary ────────────────────────────────────────
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoint: {OUTPUT_DIR / 'best_indiana_clip.pt'}")
    print(f"Log: {log_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune OpenCLIP on Indiana CXR")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size (auto-selected if not set)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of epochs (auto-selected if not set)")
    parser.add_argument("--lr", type=float, default=5e-6,
                        help="Learning rate (default: 5e-6 for BiomedCLIP ViT 6-block fine-tuning)")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay (default: 0.01)")
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience (default: 5 — stops fast to avoid overfitting)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
