# Knowledge Graph (KG) Module

## Overview

The **Knowledge Graph (KG) Module** is a comprehensive system for extracting, constructing, and traversing medical knowledge graphs from radiology text data. It uses LLM-based techniques to identify medical entities (findings, diseases, symptoms, anatomy) and relationships between them, enabling structured reasoning about clinical diagnoses.

### Key Purpose
- Extract seed entities from medical texts
- Build knowledge graph triplets (subject-relation-object)
- Traverse the knowledge graph to infer clinical relationships
- Support predictive diagnosis systems with structured knowledge

---

## Architecture & Pipeline

### **3-Step Pipeline**

#### **Step 1: Seed Entity Extraction**
**File:** `step1_extract_entities.py`

Extracts clinically meaningful entities from radiology text descriptions using Hugging Face LLM.

**Input:**
- `All_Classes_Description.txt` - Medical text descriptions of chest radiograph findings

**Process:**
- Splits large text into manageable chunks (max 10,000 characters)
- Samples up to 500 chunks for processing
- Uses `meta-llama/Meta-Llama-3-8B-Instruct` model
- Prompts LLM to extract explicit entities (no inference/invention)

**Output:**
- `output/seed_entities.txt` - List of extracted entities (one per line)
  - Examples: "Consolidation", "Pulmonary edema", "Pleural effusion", "Cardiomegaly"

**Entity Types Targeted:**
- **Finding Nodes**: Observable radiological patterns (Air bronchograms, Meniscus sign)
- **Disease Nodes**: Clinical conditions (Pneumonia, Tuberculosis, COPD)
- **Symptom Nodes**: Patient-reported evidence (Cough, Fever, Dyspnea)
- **Anatomy Nodes**: Location context (Right lower lobe, Pleural space)
- **Normal Nodes**: Explicit absence of pathology (Clear lung fields)

---

#### **Step 2: Relation Extraction & Graph Construction**
**File:** `step2_extract_relations.py`

Extracts medical relationships between entities to build the knowledge graph.

**Input:**
- `seed_entities.txt` - Extracted entities from Step 1
- `All_Classes_Description.txt` - Source medical text

**Process:**
- Loads abstract descriptions and chunks them
- For each seed entity, queries the text for relationships
- Uses `meta-llama/Meta-Llama-3-8B-Instruct` model
- Extracts triplets in (subject, relation, object) format
- Normalizes relations to predefined schema
- Builds adjacency index (subject-only, acyclic)

**Output:**
- `output/candidate_triples.jsonl` - Triplets in JSONL format
  - Example: `{"s": "consolidation", "p": "located_in", "o": "lower lobes"}`
- `output/graph_adj.json` - Adjacency index for graph traversal
  - Format: `{"entity": [["relation", "object"], ...]}`

**Supported Relations:**
| Relation | Direction | Meaning |
|----------|-----------|---------|
| `strongly_suggests` | Finding → Disease | High confidence diagnosis indicator |
| `suggests` | Finding → Disease | Probable diagnosis indicator |
| `weakly_suggests` | Finding → Disease | Weak diagnosis indicator |
| `contradicts` | Finding/Normal → Disease | Contradicts diagnosis |
| `requires` | Disease → Finding | Disease requires specific finding |
| `is_finding_of` | Finding → Disease | Clinical manifestation |
| `is_symptom_of` | Symptom → Disease | Symptom of disease |
| `located_in` | Finding → Anatomy | Anatomical location |
| `absence_weakens` | Missing Finding → Disease | Missing finding weakens diagnosis |
| `confirmed_by` | Disease → Test | Diagnostic test confirms |

---

#### **Step 3: Graph Traversal & Knowledge Inference**
**File:** `test_graph_traversal.py`

Traverses the knowledge graph to infer clinical relationships for predicted diagnoses.

**Input:**
- `graph_adj.json` - Adjacency index
- `candidate_triples.jsonl` - Full triplet list
- Predictions CSV with `uid` and `predicted_classes` columns

**Process:**
- Performs **acyclic leaf-node DFS traversal** from seed entities
- Tracks visited nodes to prevent cycles
- Extracts all triplets along root-to-leaf paths
- Per UID: processes each predicted class independently
- Ranks triplets by clinical relation priority
- Deduplicates triplets within and across classes

**Traversal Algorithm Properties:**
1. **ACYCLIC** - Subject-only adjacency + visited set prevents cycles
2. **ISOLATED** - Each (UID, class) pair has independent DFS state
3. **ORDERED** - Classes processed in CSV order; triplets ranked by priority
4. **NO DUPLICATES** - Deduplication within and across merged triplets
5. **LEAF-NODE TRAVERSAL** - All root-to-leaf paths explored until leaf nodes

**Output:**
- `output/all_uid_traversals.jsonl` - Consolidated traversals for all UIDs
- `output/per_uid_traversals/{uid}_traversal.json` - Per-UID traversal results

**Output JSON Structure:**
```json
{
  "uid": "3220",
  "gt_labels": "Normal",
  "predicted_classes": ["Normal", "Cardiomegaly"],
  "kg_classes": ["Cardiomegaly"],
  "conflict_resolution": "Strategy applied if conflicts found",
  "is_uncertain": false,
  "per_class_traversal": {
    "Cardiomegaly": {
      "seeds_used": ["Cardiomegaly", "Cardiothoracic ratio"],
      "seeds_missing": [],
      "triplets": [
        ["Cardiomegaly", "strongly_suggests", "Congestive heart failure"],
        ["Cardiomegaly", "located_in", "Left heart"]
      ],
      "triplet_count": 25,
      "leaf_paths": ["Cardiomegaly → Congestive heart failure"]
    }
  },
  "merged_triplets": [...],
  "merged_count": 45,
  "leaf_traversal": true,
  "acyclic": true
}
```

---

## Data Files

### **Input Files**

#### `All_Classes_Description.txt` (69 lines)
Medical text descriptions of chest radiograph findings. Contains detailed clinical descriptions for:
- Normal findings
- Consolidation
- Pleural effusion
- Cardiomegaly
- Pulmonary edema
- Pneumothorax
- Emphysema
- Interstitial lung disease
- Pulmonary nodules
- And more...

---

### **Output Files**

#### `output/seed_entities.txt` (907 lines)
Raw list of extracted medical entities. Examples:
```
Chest radiograph
Lung fields
Pulmonary vasculature
Cardiothoracic ratio
Cardiomegaly
Consolidation
...
```

#### `output/candidate_triples.jsonl` (1980 lines)
JSONL format triplets with schema enforcement:
```jsonl
{"s": "consolidation", "p": "located_in", "o": "lower lobes"}
{"s": "consolidation", "p": "suggests", "o": "infectious pneumonia"}
{"s": "cardiomegaly", "p": "strongly_suggests", "o": "congestive heart failure"}
{"s": "pleural effusion", "p": "requires", "o": "investigation for malignancy"}
```

#### `output/graph_adj.json`
Adjacency index for fast graph traversal:
```json
{
  "consolidation": [
    ["located_in", "lower lobes"],
    ["suggests", "infectious pneumonia"],
    ["suggests", "aspiration"]
  ],
  "cardiomegaly": [
    ["strongly_suggests", "congestive heart failure"],
    ["strongly_suggests", "pulmonary vascular redistribution"]
  ]
}
```

#### `output/all_uid_traversals.jsonl`
Consolidated traversal results per UID (one JSON object per line)

#### `output/per_uid_traversals/{uid}_traversal.json`
Individual traversal files per UID with complete triplet paths and leaf traversals

---

## Prompt Templates

### `prompts/prompts_step1.txt`
Instructions for entity extraction. Key rules:
- Extract ONLY entities explicitly in text
- Entities must be nouns/noun-phrases
- Include explicit normal/absence phrases as-is
- No sentence-level extractions
- No adjectives standalone
- No entity repetition

### `prompts/prompts_step2.txt`
Instructions for relationship extraction. Key rules:
- Extract ONLY explicit relationships
- Use exact phrases from text
- Subject from query concept or exact text match
- Relations must be from predefined set
- Output format: `(subject, relation, object)`

---

## Usage Guide

### **Running the Full Pipeline**

#### 1. Extract Seed Entities
```bash
python step1_extract_entities.py
```
**Requires:**
- `.env` file with `HUGGINGFACEHUB_ACCESS_TOKEN`
- `All_Classes_Description.txt`
- `prompts/prompts_step1.txt`

**Produces:** `output/seed_entities.txt`

#### 2. Extract Relations & Build Graph
```bash
python step2_extract_relations.py
```
**Requires:**
- `output/seed_entities.txt` (from Step 1)
- `All_Classes_Description.txt`
- `prompts/prompts_step2.txt`

**Produces:**
- `output/candidate_triples.jsonl`
- `output/graph_adj.json`

#### 3. Traverse Graph for Predictions
```bash
python test_graph_traversal.py \
  --predictions predictions.csv \
  --output_dir output
```
**Requires:**
- `output/graph_adj.json` (from Step 2)
- `output/candidate_triples.jsonl` (from Step 2)
- Predictions CSV with columns: `uid`, `predicted_classes`

**Produces:**
- `output/all_uid_traversals.jsonl`
- `output/per_uid_traversals/{uid}_traversal.json` (for each UID)

---

## Technical Details

### LLM Configuration
- **Model**: `meta-llama/Meta-Llama-3-8B-Instruct`
- **Temperature**: 0.1 (low randomness, deterministic)
- **Max Tokens**: 512 (Step 1), variable (Step 2)
- **Framework**: LangChain with Hugging Face Hub

### Graph Properties
- **Acyclic**: Subject-only adjacency prevents forward cycles
- **Forward-only**: Edges point from subjects to objects
- **Deduplicated**: Triplets deduplicated within and across classes
- **Ranked**: Relations prioritized by clinical significance

### Computational Features
- **Chunking**: Text split into ~10,000 character chunks
- **Sampling**: Up to 500 chunks sampled for efficiency
- **Normalization**: Relation names normalized to predefined schema
- **Hub Filtering**: High-degree hub entities filtered during traversal

---

## Key Design Decisions

### 1. **Acyclic Traversal**
The graph uses a **subject-only adjacency index** where only outgoing edges from subjects are stored. Objects never become keys, ensuring that:
- Traversal is strictly forward (subject → object)
- Even if cycles exist in raw data, visited sets prevent re-expansion
- Search is naturally acyclic

### 2. **Per-UID Isolation**
Each (UID, class) pair runs its own independent DFS with:
- Independent visited set
- Independent triplet collection
- Zero cross-contamination between UIDs or classes

### 3. **Leaf-Node Traversal**
Rather than fixed depth limits:
- Traversal continues until leaf nodes (no outgoing edges)
- ALL root-to-leaf paths are explored
- Complete reachable subgraph coverage

### 4. **No Entity Inference**
Step 1 and Step 2 extract ONLY entities and relationships:
- Explicitly present in text
- No LLM-inferred or generated entities
- Preserves clinical accuracy and traceability

---

## Medical Context

### Domains Covered
This knowledge graph focuses on **chest radiography findings** including:
- **Findings**: Consolidation, effusion, edema, nodules, masses
- **Diseases**: Pneumonia, CHF, COPD, TB, fibrosis, cancer
- **Anatomy**: Lungs, lobes, pleura, mediastinum, heart, diaphragm
- **Tests**: X-ray signs, diagnostic markers
- **Symptoms**: Cough, dyspnea, fever, chest pain

### Clinical Relationships
The relation types capture:
- **Diagnostic Evidence**: Findings that suggest/strongly suggest disease
- **Contradictory Evidence**: Findings that contradict disease
- **Anatomical Context**: Location of findings
- **Diagnostic Tests**: Tests that confirm disease
- **Disease Requirements**: What a disease requires to be present

---

## Integration Points

This module integrates with:
1. **BiomedCLIP** - For image-based medical classification
2. **Other Models** - For comparative inference (Claude, Llama, Qwen)
3. **Retriever** - For knowledge-based document retrieval
4. **Evaluation Pipeline** - For accuracy metrics and judge evaluation

---

## File Structure

```
Knowlege-Graph/
├── README.md                          # This file
├── All_Classes_Description.txt        # Input: Medical text descriptions
├── step1_extract_entities.py          # Entity extraction (LLM-based)
├── step2_extract_relations.py         # Relation extraction & KG construction
├── test_graph_traversal.py            # Graph traversal & inference
├── prompts/
│   ├── prompts_step1.txt              # Entity extraction prompts
│   └── prompts_step2.txt              # Relation extraction prompts
└── output/
    ├── seed_entities.txt              # Extracted entities
    ├── candidate_triples.jsonl        # Extracted triplets
    ├── graph_adj.json                 # Adjacency index
    ├── all_uid_traversals.jsonl       # All UID traversals
    └── per_uid_traversals/            # Individual UID traversals
        ├── 3220_traversal.json
        ├── 3221_traversal.json
        └── ...
```

---

## Dependencies

- **LangChain**: LLM orchestration and prompting
- **Hugging Face Hub**: Model access and API
- **tqdm**: Progress bars
- **Python 3.8+**: Core language

Install via:
```bash
pip install -r requirements.txt
```

---

## Troubleshooting

### Entity Extraction Fails
- Verify `HUGGINGFACEHUB_ACCESS_TOKEN` in `.env`
- Ensure `All_Classes_Description.txt` is readable
- Check model availability on Hugging Face Hub

### No Triplets Generated
- Verify `seed_entities.txt` contains entities
- Check entity name matching (case-sensitive in index)
- Ensure relations match predefined schema

### Graph Traversal Empty
- Verify `graph_adj.json` is properly constructed
- Check predictions CSV format (uid, predicted_classes)
- Ensure class names match entity names in graph

---

## Performance Characteristics

- **Entity Extraction**: ~5-10 minutes (500 chunks, 8B model)
- **Relation Extraction**: ~10-20 minutes (depends on entity count)
- **Graph Traversal**: ~1-2 seconds per UID (cached adjacency)
- **Storage**: ~50-100 MB for full triplet set

---

## Future Enhancements

- [ ] Bidirectional graph traversal
- [ ] Entity disambiguation and linking
- [ ] Relation confidence scoring
- [ ] Multi-hop inference chains
- [ ] Graph visualization tools
- [ ] Real-time inference API

---

## References

- **Medical Domain**: Chest radiography findings and clinical reasoning
- **Graph Structure**: Acyclic, subject-only indexed knowledge graphs
- **LLM Framework**: LangChain + Hugging Face Hub integration
- **Inference**: Depth-first search with leaf-node traversal

