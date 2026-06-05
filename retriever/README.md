# Retriever Module - Medical Image-Report Retrieval System

## Overview

The **Retriever Module** implements a **BiomedCLIP-based medical image-text retrieval system** for chest radiography. It trains a fine-tuned CLIP model on paired chest X-ray images and radiology reports, enabling semantic retrieval of similar patient reports based on query images.

### Core Purpose
- Fine-tune BiomedCLIP (vision-language model) on medical image-text pairs
- Build searchable database of patient reports with image/text embeddings
- Retrieve semantically similar reports for new chest X-ray images
- Evaluate retrieval performance using medical label overlap
- Generate comprehensive visualizations of model behavior

### Key Innovation
Uses **hybrid retrieval** combining both text and image embeddings to capture:
- **Text similarity**: Diagnostic/semantic labels from radiology reports
- **Image similarity**: Visual appearance and radiological patterns

---

## Architecture Overview

### Model: BiomedCLIP-PubMedBERT
```
BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
├── Vision Encoder: ViT-B/16 (vision transformer)
│   ├── Pre-trained on 15M biomedical images
│   ├── Input: 224×224 RGB images
│   ├── Output: 768-dim embeddings
│   └── Unfrozen: Last 6 transformer blocks (for fine-tuning)
│
├── Text Encoder: PubMedBERT-256
│   ├── Pre-trained on PubMed abstracts
│   ├── Input: 77-token sequences (auto-truncated)
│   └── Output: 768-dim embeddings
│
└── Contrastive Loss: Symmetric InfoNCE
    ├── image-to-text matching
    └── text-to-image matching
```

### Training Strategy
- **Frozen components**: Early ViT blocks + text encoder (keep biomedical pre-training)
- **Fine-tuned components**: Last 6 ViT blocks + projection layers
- **Loss**: Symmetric cross-entropy with label smoothing (0.1)
- **Optimization**: Cosine decay with linear warmup
- **Regularization**: Gradient accumulation, clipping, weight decay, data augmentation

---

## 5-Step Pipeline

### **Step 1: Data Preparation**
**Script:** `01_prepare_data.py`

Processes raw Indiana chest X-ray dataset and creates train/val/test splits.

**Input:**
- `training/training_indiana_reports.csv` - Training patient reports (UID, findings, impression)
- `training/training_indiana_projections.csv` - Training image metadata (UID, filename, projection)
- `training/files/` - Training chest X-ray images
- `testing/indiana_reports.csv` - Validation/test patient reports
- `testing/indiana_projections.csv` - Validation/test image metadata
- `testing/test1/` - Validation/test chest X-ray images

**Process:**
1. Combines findings + impression into single caption (impression first for CLIP truncation)
2. Truncates caption to 400 characters (~60-70 tokens, within CLIP's 77-token limit)
3. Pairs each UID with its images (frontal, lateral, or multiple views)
4. Splits val/test UIDs 50-50 after sorting
5. Filters out UIDs with empty findings AND impression
6. Creates train/val/test JSON files with full image paths and captions

**Output:**
- `splits/train.json` - Training pairs (~2.5 MB)
  ```json
  [
    {
      "uid": 3220,
      "image_path": "/path/to/image.png",
      "caption": "Normal chest radiograph. No acute findings.",
      "projection": "frontal",
      "split": "train"
    },
    ...
  ]
  ```
- `splits/val.json` - Validation pairs (~309 KB)
- `splits/test.json` - Test pairs (~318 KB)

**Key Design:**
- **Multi-image per UID**: Each image becomes separate training example with shared caption
- **Caption priority**: Impression (clinical summary) placed first before findings (detailed observations)
- **Length limits**: 400-char caption + 77-token CLIP hard limit prevents truncation of critical info

---

### **Step 2: Model Training**
**Script:** `02_train.py`

Fine-tunes BiomedCLIP on Indiana dataset using contrastive learning.

**Architecture:**
```python
Model: BiomedCLIP (ViT-B/16 + PubMedBERT)
├── Vision: 768-dim embeddings from unfrozen ViT blocks
├── Text: 768-dim embeddings from frozen PubMedBERT
└── Projection: Linear layers to shared 256-dim embedding space
```

**Training Configuration:**
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Epochs** | 30+ | Until early stopping (patience=5) |
| **Batch Size** | 64 | Per device (MPS/CUDA/CPU) |
| **Grad Accumulation** | 4 steps | Effective batch = 256 for better contrastive signal |
| **Learning Rate** | 1e-5 | Small LR for fine-tuning pre-trained model |
| **Warmup Steps** | 500 | Linear warmup before cosine decay |
| **Optimizer** | AdamW | Weight decay = 0.01 |
| **Loss Function** | Symmetric InfoNCE | image→text + text→image CE / 2 |
| **Label Smoothing** | 0.1 | Prevents embedding collapse |

**Data Augmentation (Training only):**
```python
RandomResizedCrop(224, scale=(0.75, 1.0))    # Crop + zoom
RandomHorizontalFlip(p=0.5)                  # Mirror for symmetric findings
ColorJitter(brightness=0.15, contrast=0.15)  # Brightness/contrast variation
RandomAffine(degrees=5, translate=(0.05, 0.05))  # Slight rotation/shift
```

**Contrastive Loss Details:**
```
For batch size B:
  - Create B×B similarity matrix between image & text embeddings
  - Diagonal = positive pairs (image_i with text_i)
  - Off-diagonal = negatives
  - Loss = symmetric cross-entropy with label smoothing
  - Result: Image & text embeddings in same space, positives closer
```

**Performance Optimization:**
- **Image Cache**: 256×256 JPEG cached on disk (avoid re-loading 2048×2048 originals)
- **Gradient Accumulation**: Larger effective batch without memory overhead
- **Mixed Precision**: Float32 (BiomedCLIP requirement, not AMP)
- **Early Stopping**: Stop if val_loss doesn't improve for 5 epochs

**Output:**
- `checkpoints/best_indiana_clip.pt` - Best model checkpoint
  ```python
  {
    "model_state_dict": {...},
    "optimizer_state_dict": {...},
    "epoch": 25,
    "val_loss": 3.756233,
    "logit_scale": 85.1708
  }
  ```
- `logs/training_log.csv` - Per-epoch metrics

**Training Example Output:**
```
Epoch 1/30 | Train Loss: 7.26 | Val Loss: 6.71 | LR: 3.13e-07
Epoch 2/30 | Train Loss: 6.02 | Val Loss: 5.36 | LR: 6.25e-07  ✓ Best
...
Epoch 25/30 | Train Loss: 3.62 | Val Loss: 3.76 | LR: 2.42e-06  ✓ Best
Epoch 26/30 | Train Loss: 3.60 | Val Loss: 3.74 | LR: 2.41e-06
...Early stopped at epoch 29 (patience=5)
```

---

### **Step 3: Build Database**
**Script:** `03_build_database.py`

Encodes all training images and captions into searchable embeddings.

**Input:**
- `checkpoints/best_indiana_clip.pt` - Fine-tuned model
- `splits/train.json` - Training split

**Process:**
1. Loads best checkpoint into BiomedCLIP model
2. **For each training UID:**
   - If multiple images (frontal + lateral), average embeddings
   - Encode via vision encoder → 768-dim vector
3. **For each caption:**
   - Tokenize and encode via text encoder → 768-dim vector
4. L2-normalize all embeddings (for cosine similarity)
5. Save as dense numpy arrays indexed by UID

**Output:**
- `database/indiana_database_vectors.npy` - Text embeddings (N, 768)
- `database/indiana_database_img_vectors.npy` - Image embeddings (N, 768)
- `database/indiana_database_metadata.json` - UID/caption metadata
  ```json
  [
    {"uid": 3220, "caption": "Normal chest radiograph..."},
    {"uid": 3221, "caption": "Consolidation in right lower lobe..."},
    ...
  ]
  ```
- `database/blend_weights.json` - Retrieval blend weights
  ```json
  {"text_weight": 0.6, "image_weight": 0.4}
  ```

**Key Features:**
- **Multi-image averaging**: If UID has frontal+lateral, embeddings averaged (single vector per UID)
- **L2 normalization**: Enable fast cosine similarity via dot product
- **Hybrid DB**: Both text and image embeddings for blended retrieval

---

### **Step 4: Evaluation**
**Script:** `04_evaluate.py`

Evaluates retrieval performance on validation and test splits.

**Metrics Computed:**

| Metric | Formula | Meaning |
|--------|---------|---------|
| **Recall@K** | % of queries where relevant report in top-K | Hit rate |
| **Precision@K** | % of top-K retrieved that are relevant | Precision of retrieval |
| **F1@K** | 2 × (P×R)/(P+R) | Harmonic mean of P and R |
| **MRR** | Average 1/rank of first relevant | Rewards earlier hits |
| **Top-1 Similarity** | Mean similarity score of rank-1 result | Confidence |

**Ground Truth Definition:**
Since val/test UIDs are NOT in training database, exact match impossible. Instead:
- **Relevant report**: Shares ≥1 medical label with query
- **Medical labels**: MeSH terms from "Problems" column (standardized vocabulary)
- **Label filtering**: Removes DB metadata tags (e.g., "no finding", "not reported")

**Retrieval Process:**
```python
For each query UID in val/test:
  1. Encode query image → image embedding
  2. Compute similarities: query_emb @ database_vectors.T
  3. Blend with text similarity (if hybrid): 
     score = 0.6 × text_sim + 0.4 × image_sim
  4. Get top-K results by score
  5. Check if any top-K share medical labels with query
```

**Output:**
- `evaluation/val_results.json` - Per-query retrieval results (~1.1 MB)
  ```json
  {
    "uid": 3220,
    "gt_labels": ["Normal", "No findings"],
    "retrieved_at_1": {
      "rank": 1,
      "uid": 4015,
      "caption": "Normal chest radiograph...",
      "similarity": 0.89,
      "label_overlap": ["Normal"],
      "is_relevant": true
    },
    "retrieved_at_3": [...],
    "recall_1": 1,
    "recall_3": 1,
    "recall_5": 1,
    "recall_10": 1,
    "mrr": 1.0
  }
  ```
- `evaluation/test_results.json` - Same structure for test split
- Metrics aggregation: Recall@K, Precision@K, F1@K, MRR per split

**Interpretation:**
- High Recall@1 = Model confidently finds similar patients
- High F1@K = Balanced precision/recall trade-off
- MRR > 0.8 = First-hit typically among top-2

---

### **Step 5: Visualization**
**Script:** `05_visualize.py`

Generates comprehensive plots and analysis visualizations.

**Plots Generated:**

1. **Loss Curves** (`01_loss_curves.png`)
   - Training loss (blue) vs validation loss (red) across epochs
   - Shows convergence behavior and early stopping trigger

2. **Retrieval Examples** (`02_retrieval_examples_val.png`, `02_retrieval_examples_test.png`)
   - Query image (top) → Top-5 retrieved reports (thumbnails + captions)
   - Similarity scores displayed
   - Color-coded: Green = relevant (label overlap), Red = irrelevant

3. **t-SNE Embeddings** (`03_tsne_embeddings.png`)
   - 2D visualization of 768-dim embeddings
   - Colored by medical label (disease type)
   - Shows cluster structure: related conditions group together

4. **Similarity Distributions** (`04_similarity_distributions.png`)
   - Histogram of top-1 similarity scores (val vs test)
   - Shows model confidence distribution
   - Test should be similar to val (no distribution shift)

5. **Per-Label Recall Heatmap** (`05_per_label_recall.heatmap.png`)
   - Rows = disease labels, Columns = K values (1, 3, 5, 10)
   - Cells = Recall@K for that label
   - Identifies which conditions are retrieved well vs poorly

6. **Metrics Comparison** (`06_metrics_comparison.png`)
   - Bar charts: Recall@K, Precision@K, F1@K
   - Val vs Test side-by-side
   - Overall trends visible

**Saved to:**
- `plots/` - All visualization PNG files

---

## Data Files

### Input Data Structure

```
training/
├── training_indiana_reports.csv       # UID, findings, impression, ...
├── training_indiana_projections.csv   # UID, filename, projection, ...
└── files/                              # Chest X-ray JPEG/PNG images

testing/
├── indiana_reports.csv                # UID, findings, impression, ...
├── indiana_projections.csv            # UID, filename, projection, ...
└── test1/                              # Chest X-ray images
```

### Report CSV Format
```
uid,findings,impression,Problems,...
3220,"Clear lung fields...","No acute pathology...","Normal|No findings"
3221,"Consolidation in RLL...","Pneumonia","Pneumonia|Airspace disease"
```

### Generated Outputs

**After Step 1 (Data Prep):**
```
Retriever/splits/
├── train.json    (2.5 MB)  # ~8,000 image-caption pairs
├── val.json      (309 KB)  # ~1,000 pairs
└── test.json     (318 KB)  # ~1,000 pairs
```

**After Step 2 (Training):**
```
Retriever/
├── checkpoints/best_indiana_clip.pt  # Model weights
├── logs/training_log.csv             # 30+ epoch metrics
└── .image_cache/                     # 256×256 cached images
```

**After Step 3 (Database):**
```
Retriever/database/
├── indiana_database_vectors.npy      # (N, 768) text embeddings
├── indiana_database_img_vectors.npy  # (N, 768) image embeddings
├── indiana_database_metadata.json    # UID/caption pairs
└── blend_weights.json                # Retrieval weights
```

**After Step 4 (Evaluation):**
```
Retriever/evaluation/
├── val_results.json    # Per-query retrieval metrics
└── test_results.json   # Per-query retrieval metrics

Retriever/output/
├── val_results.json    # Aggregated val metrics
├── test_results.json   # Aggregated test metrics
└── comparison_summary.csv  # Summary across models
```

**After Step 5 (Visualization):**
```
Retriever/plots/
├── 01_loss_curves.png
├── 02_retrieval_examples_val.png
├── 02_retrieval_examples_test.png
├── 03_tsne_embeddings.png
├── 04_similarity_distributions.png
├── 05_per_label_recall.png
└── 06_metrics_comparison.png
```

---

## Usage Guide

### Running the Full Pipeline

**Option 1: Complete pipeline (Step 1→5)**
```bash
python run_clip_pipeline.py
```

**Option 2: Start from specific step**
```bash
python run_clip_pipeline.py --step 3  # Start from DB building
```

**Option 3: Run only one step**
```bash
python run_clip_pipeline.py --only 2  # Run only training
```

**Option 4: Override training hyperparameters**
```bash
python run_clip_pipeline.py --epochs 50 --batch_size 32 --lr 2e-5 --patience 10
```

### Individual Script Usage

#### Step 1: Prepare Data
```bash
python 01_prepare_data.py
# Output: splits/train.json, splits/val.json, splits/test.json
```

#### Step 2: Train Model
```bash
python 02_train.py --epochs 30 --batch_size 64 --lr 1e-5 --patience 5
# Output: checkpoints/best_indiana_clip.pt, logs/training_log.csv
```

#### Step 3: Build Database
```bash
python 03_build_database.py
# Output: database/indiana_database_vectors.npy, metadata.json, etc.
```

#### Step 4: Evaluate
```bash
python 04_evaluate.py
# Output: evaluation/{val,test}_results.json
```

#### Step 5: Visualize
```bash
python 05_visualize.py
# Output: plots/*.png
```

### Inference on New Image

**Script:** `test_retriever.py`

```bash
# Retrieve for a specific test UID
python test_retriever.py --uid 3500 --top_k 5

# Retrieve for a custom image
python test_retriever.py --image /path/to/xray.png --top_k 5
```

**Output:**
```
Query: /path/to/xray.png
Top-5 Similar Reports:
  1. UID 4015 | Similarity: 0.89 | "Normal chest radiograph. No findings."
  2. UID 4102 | Similarity: 0.87 | "Slightly hyperlucent lung fields..."
  3. UID 4200 | Similarity: 0.85 | "Normal cardiothoracic ratio..."
  4. UID 4301 | Similarity: 0.83 | "No acute findings..."
  5. UID 4450 | Similarity: 0.80 | "Unremarkable exam..."
```

---

## Technical Details

### Model Architecture

**BiomedCLIP Components:**
```
Vision Tower (Frozen except last 6 blocks):
  Input: (B, 3, 224, 224) RGB image
  ├── Patch embedding + positional encoding
  ├── 12 ViT transformer blocks (each ~64M params)
  ├── Last 6 unfrozen for fine-tuning
  └── Output: (B, 768) normalized embeddings

Text Tower (Frozen):
  Input: (B, 77) token IDs
  ├── Token embedding + position encoding
  ├── PubMedBERT transformer blocks
  └── Output: (B, 768) normalized embeddings

Projection Head:
  ├── image_proj: Linear(768 → 256)
  ├── text_proj: Linear(768 → 256)
  └── logit_scale: Learnable temperature parameter
```

### Contrastive Learning

**Symmetric InfoNCE Loss:**
```python
# Given batch of (image, text) pairs
image_emb = model.encode_image(images)  # (B, 768)
text_emb = model.encode_text(texts)     # (B, 768)

# Normalize to unit norm
image_emb = normalize(image_emb)
text_emb = normalize(text_emb)

# Compute similarity matrix
sim = image_emb @ text_emb.T  # (B, B)
logits = logit_scale * sim    # Scale by learnable temperature

# Symmetric cross-entropy
loss_i2t = CE(logits, labels)      # image→text
loss_t2i = CE(logits.T, labels)    # text→image
loss = (loss_i2t + loss_t2i) / 2
```

### Hybrid Retrieval

**Query Time Blending:**
```python
# For test image query:
query_img_emb = model.encode_image(query)
query_txt_emb = model.encode_text(query_caption)

# Compute both similarities
img_sim = query_img_emb @ database_img_emb.T  # (1, N)
txt_sim = query_txt_emb @ database_txt_emb.T  # (1, N)

# Blend with learned weights
blended_sim = 0.6 * txt_sim + 0.4 * img_sim

# Get top-K
top_k = argtop_k(blended_sim, k=5)
```

---

## Performance Characteristics

### Training Time
- **Single epoch**: ~40-60 minutes (on MPS/GPU)
- **Full training (30 epochs)**: ~20-30 hours
- **Early stopping**: Typically stops around epoch 25-28

### Model Size
- **Checkpoint**: ~500 MB (BiomedCLIP ViT-B/16)
- **Database vectors**: ~300 MB (8000 UIDs × 768 dims × 4 bytes)

### Retrieval Speed
- **Per-query inference**: <100ms (on GPU), ~1-2s (on CPU)
- **Batch retrieval (1000 queries)**: <1 minute (on GPU)

### Memory Requirements
- **Training**: ~8-16 GB (with gradient accumulation)
- **Inference**: ~2-4 GB (model + database in memory)

---

## Key Design Decisions

### 1. BiomedCLIP Over Generic CLIP
- Pre-trained on 15M biomedical images (vs ImageNet)
- PubMedBERT text encoder (medical domain)
- Captures radiological patterns better than generic vision models

### 2. Hybrid Retrieval (Text + Image)
- Text similarity captures diagnostic labels and semantic meaning
- Image similarity captures visual appearance and radiological patterns
- Blend (0.6 text + 0.4 image) leverages both modalities

### 3. Partial Fine-tuning
- Freeze early ViT blocks (preserve biomedical pre-training)
- Unfreeze last 6 blocks + projections (adapt to Indiana domain)
- Prevents catastrophic forgetting of medical knowledge

### 4. Label-based Ground Truth
- No exact UID match possible (val/test disjoint from train)
- Use medical label overlap as proxy for relevance
- More realistic for clinical application (finding similar cases)

### 5. Image Caching
- Cache 256×256 JPEG on disk (avoid slow disk I/O)
- Significantly speeds up training (8-16× faster than no cache)
- 95% quality preserves radiological details

---

## Integration Points

This module integrates with:
1. **BiomedCLIP** - Vision-language model for chest X-rays
2. **Knowledge Graph** - For semantic enrichment of retrieved reports
3. **Other Models** - For comparative evaluation (Llama, Claude, Qwen)
4. **Evaluation Pipeline** - For comparative metrics across models

---

## Troubleshooting

### Training Not Converging
- Check learning rate (default 1e-5 may be too high/low)
- Verify data splits loaded correctly (train.json exists?)
- Check for NaN/inf in loss: may indicate scale issue with logit_scale

### Retrieval Quality Poor
- Verify database built from best checkpoint
- Check blend weights (text vs image contribution)
- Ensure medical labels properly parsed from "Problems" column

### Out of Memory
- Reduce batch_size (--batch_size 32)
- Reduce grad_accum_steps in code (lower effective batch)
- Disable image cache (use_cache=False in dataset)

### Slow Training
- Enable image cache (.image_cache/ directory must be writable)
- Use GPU/MPS instead of CPU (100× faster)
- Reduce number of epochs for prototyping

---

## File Structure

```
Retriever/
├── README.md                          # This file
├── 01_prepare_data.py                 # Data preparation script
├── 02_train.py                        # Training script
├── 03_build_database.py               # Database building script
├── 04_evaluate.py                     # Evaluation script
├── 05_visualize.py                    # Visualization script
├── test_retriever.py                  # Inference/testing script
├── run_clip_pipeline.py               # Master pipeline orchestrator
│
├── checkpoints/
│   └── best_indiana_clip.pt           # Best fine-tuned model
│
├── database/
│   ├── indiana_database_vectors.npy   # Text embeddings (N, 768)
│   ├── indiana_database_img_vectors.npy # Image embeddings (N, 768)
│   ├── indiana_database_metadata.json # UID/caption metadata
│   └── blend_weights.json             # Retrieval blend weights
│
├── splits/
│   ├── train.json                     # Training split (~2.5 MB)
│   ├── val.json                       # Validation split (~309 KB)
│   └── test.json                      # Test split (~318 KB)
│
├── logs/
│   └── training_log.csv               # Per-epoch training metrics
│
├── evaluation/
│   ├── val_results.json               # Per-query val metrics
│   └── test_results.json              # Per-query test metrics
│
├── output/
│   ├── val_results.json               # Aggregated val results
│   ├── test_results.json              # Aggregated test results
│   └── comparison_summary.csv         # Cross-model comparison
│
├── plots/
│   ├── 01_loss_curves.png
│   ├── 02_retrieval_examples_val.png
│   ├── 02_retrieval_examples_test.png
│   ├── 03_tsne_embeddings.png
│   ├── 04_similarity_distributions.png
│   ├── 05_per_label_recall.png
│   └── 06_metrics_comparison.png
│
├── .image_cache/                      # Cached 256×256 images (temporary)
└── training_output_v3.log             # Detailed training log
```

---

## Dependencies

```
torch                 # Deep learning framework
open-clip             # OpenCLIP models
torchvision           # Image preprocessing
PIL                   # Image loading
numpy                 # Numerical arrays
pandas                # Data manipulation
scikit-learn          # t-SNE visualization
matplotlib            # Plotting
tqdm                  # Progress bars
```

Install via:
```bash
pip install -r requirements.txt
```

---

## Medical Context

### Domain: Chest Radiography
This retriever focuses on **Indiana chest X-ray dataset**:
- **Data**: Patient chest radiographs with paired diagnostic reports
- **Task**: Image-to-report retrieval (find similar patients)
- **Features**: Normal anatomy, pathologies (pneumonia, CHF, etc.)

### Clinical Application
- **Literature search**: Find similar historical cases for case review
- **Clinical decision support**: Retrieve comparable diagnostic findings
- **Knowledge base**: Build repository of annotated radiographs

---

## Future Enhancements

- [ ] Multi-modal fusion (combine with KG for enhanced reasoning)
- [ ] Hard negative mining (focus on confusing similar cases)
- [ ] Explainable retrieval (visualize which regions drive similarity)
- [ ] Real-time inference API (REST endpoint for production)
- [ ] Cross-modality retrieval (text query → find similar images)
- [ ] Interactive fine-tuning (user feedback improves model)
- [ ] Temporal analysis (retrieve from similar timeframes)

---

## References

- **Paper**: Learning Transferable Visual Models From Natural Language Supervision (CLIP)
- **BiomedCLIP**: Domain-adapted CLIP for biomedical images
- **Indiana Dataset**: https://www.kaggle.com/raddar/chest-xrays-indiana-university
- **OpenCLIP**: https://github.com/mlfoundations/open_clip

