import os
import sys
import json
import logging
from collections import Counter
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from tqdm import tqdm
import re
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
if not hf_token:
    sys.exit(
        "ERROR: Please set the HUGGINGFACEHUB_ACCESS_TOKEN environment variable "
        "(or add it to a .env file)."
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)
OUTPUT_BASE_DIR = "output"
INPUT_CONCEPTS_PATH  = os.path.join(OUTPUT_BASE_DIR, "seed_entities.txt")
INPUT_ABSTRACTS_PATH = "All_Classes_Description.txt"
OUTPUT_FILE          = os.path.join(OUTPUT_BASE_DIR, "candidate_triples.jsonl")
OUTPUT_ADJ_FILE      = os.path.join(OUTPUT_BASE_DIR, "graph_adj.json")
PROMPT_PATH          = os.path.join("prompts", "prompts_step2.txt")
MODEL_NAME           = "meta-llama/Meta-Llama-3-8B-Instruct"
MAX_INPUT_CHAR = 10000

RELATION_DEFS = {
    "strongly_suggests": {"label": "strongly_suggests", "direction": "Finding -> Disease"},
    "suggests": {"label": "suggests", "direction": "Finding -> Disease"},
    "weakly_suggests": {"label": "weakly_suggests", "direction": "Finding -> Disease"},
    "contradicts": {"label": "contradicts", "direction": "Finding/Normal -> Disease"},
    "requires": {"label": "requires", "direction": "Disease -> Finding"},
    "is_finding_of": {"label": "is_finding_of", "direction": "Finding -> Disease"},
    "is_symptom_of": {"label": "is_symptom_of", "direction": "Symptom -> Disease"},
    "located_in": {"label": "located_in", "direction": "Finding -> Anatomy"},
    "absence_weakens": {"label": "absence_weakens", "direction": "Missing Finding -> Disease"},
    "confirmed_by": {"label": "confirmed_by", "direction": "Disease -> Test"}
}

def load_abstracts():
    if not os.path.exists(INPUT_ABSTRACTS_PATH):
        logging.error(f"Abstracts file not found: {INPUT_ABSTRACTS_PATH}")
        return None

    with open(INPUT_ABSTRACTS_PATH, 'r', encoding='utf-8') as f:
        content = f.read().strip()

    if not content:
        logging.error("Abstracts file is empty")
        return None

    logging.info(f"Loaded {len(content)} characters from abstracts file")
    return content


def chunk_content(content, max_chars):
    chunks = []
    words = content.split()
    current_chunk = []
    current_length = 0

    for word in words:
        word_length = len(word) + 1
        if current_length + word_length > max_chars and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = word_length
        else:
            current_chunk.append(word)
            current_length += word_length

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks


def normalize_relation(rel):
    """Normalize relation casing and enforce predefined schema."""
    if not rel or not isinstance(rel, str):
        return None
   
    rel_normalized = rel.strip().lower().replace(" ", "_").replace("-", "_")
    
    # Direct match in RELATION_DEFS
    if rel_normalized in RELATION_DEFS:
        return rel_normalized
    
    mapping = {
        "strongly_suggests": "strongly_suggests",
        "strongly_suggest": "strongly_suggests",
        "suggest": "suggests",
        "suggests": "suggests",
        "weakly_suggest": "weakly_suggests",
        "weakly_suggests": "weakly_suggests",
        "contradict": "contradicts",
        "contradicts": "contradicts",
        "require": "requires",
        "requires": "requires",
        "required": "requires",
        "is_finding_of": "is_finding_of",
        "isfindingof": "is_finding_of",
        "is_symptom_of": "is_symptom_of",
        "issymptomof": "is_symptom_of",
        "located_in": "located_in",
        "locatedin": "located_in",
        "absence_weaken": "absence_weakens",
        "absence_weakens": "absence_weakens",
        "confirmed_by": "confirmed_by",
        "confirmedby": "confirmed_by"
    }
    
    result = mapping.get(rel_normalized, None)
    if result:
        return result
    
    logging.debug(f"Could not normalize relation: '{rel}' (normalized: '{rel_normalized}')")
    return None


def validate_triplet(triplet, existing_triplets, graph):
    """
    Validate triplet with 5 checks:
    1. Clinical plausibility (relation exists in RELATION_DEFS)
    2. Entity verification (non-empty entities)
    3. Acyclicity check (no cycles in graph)
    4. Uniqueness (not already in output)
    5. Bidirectional consistency (no inverse conflicts)
    """
    s = triplet.get('s', '').strip().lower()
    p = triplet.get('p')
    o = triplet.get('o', '').strip().lower()
    
    # Check 1: Clinical Plausibility - relation must be valid
    if not p or p not in RELATION_DEFS:
        return False, "Invalid relation"
    
    # Check 2: Entity Verification - entities must not be empty
    if not s or not o or len(s) < 2 or len(o) < 2:
        return False, "Invalid entities (empty or too short)"
    
    # Check 3: Acyclicity - detect cycles
    # Build path from s to o using existing graph
    if would_create_cycle(s, o, graph):
        return False, "Would create cycle"
    
    # Check 4: Uniqueness - check if triplet already exists
    triplet_key = (s, p, o)
    if triplet_key in existing_triplets:
        return False, "Duplicate triplet"
    
    # Check 5: Bidirectional Consistency - check for inverse conflicts
    inverse_triplet_key = (o, p, s)
    if inverse_triplet_key in existing_triplets:
        return False, "Bidirectional conflict with existing triplet"
    
    return True, "Valid"


def would_create_cycle(source, target, graph):
    """Check if adding edge source->target would create a cycle."""
    if source == target:
        return True
    
    # BFS from target to see if we can reach source
    visited = set()
    queue = [target]
    
    while queue:
        node = queue.pop(0)
        if node == source:
            return True
        if node in visited:
            continue
        visited.add(node)
        
        if node in graph:
            queue.extend(graph[node])
    
    return False

def parse_triplet_output(text):
    """
    Parse LLM output into structured triplets (s, p, o).
    Handles multiple formats: JSON, (subject, relation, object), and plain text.
    """
    triplets = []

    if not text or text.strip().lower() == "none":
        return triplets

    # Remove markdown code blocks
    text = text.replace("```", "").replace("```python", "").replace("```markdown", "").strip()

    # 1. Try JSON parsing first
    try:
        json_data = json.loads(text)
        if isinstance(json_data, list):
            for item in json_data:
                if isinstance(item, dict) and 's' in item and 'p' in item and 'o' in item:
                    rel = normalize_relation(item['p'])
                    if rel:
                        triplets.append({
                            's': item['s'].strip(),
                            'p': rel,
                            'o': item['o'].strip()
                        })
            if triplets:
                return triplets
    except (json.JSONDecodeError, ValueError):
        pass

    # 2. Regex parsing for (subject, relation, object) format
    pattern = r'\(\s*([^,]+?)\s*,\s*([^,]+?)\s*,\s*([^)]+?)\s*\)'
    matches = re.findall(pattern, text)

    for match in matches:
        subject = match[0].strip()
        relation = match[1].strip()
        obj = match[2].strip()
        
        # Skip if any part is empty or contains invalid characters
        if not subject or not relation or not obj:
            continue
        if len(subject) < 2 or len(obj) < 2:
            continue
        
        rel = normalize_relation(relation)
        if rel:  # Only add if relation normalizes to valid type
            triplets.append({
                's': subject,
                'p': rel,
                'o': obj
            })

    return triplets


def extract_candidate_triples():
    logging.info("Starting Step 2: Candidate Triple Extraction (HuggingFace)")

    model = ChatHuggingFace(
        llm=HuggingFaceEndpoint(
            repo_id=MODEL_NAME,
            temperature=0.2,  # Lower temperature for more consistent output
            max_new_tokens=256,  # Reduced for focused triplet extraction
            huggingfacehub_api_token=hf_token,
            timeout=60
        )
    )

    if not os.path.exists(INPUT_CONCEPTS_PATH):
        logging.error(f"Input concepts file not found: {INPUT_CONCEPTS_PATH}")
        return

    with open(INPUT_CONCEPTS_PATH, 'r', encoding='utf-8') as f:
        query_concepts = [line.strip().strip('"') for line in f if line.strip()]

    all_content = load_abstracts()
    if not all_content:
        logging.error("No content loaded")
        return
        
    if not os.path.exists(PROMPT_PATH):
        logging.error(f"Prompt file not found: {PROMPT_PATH}")
        return

    prompt_txt = open(PROMPT_PATH, 'r', encoding='utf-8').read()
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a medical knowledge graph builder. Extract ONLY explicit medical relationships. Output ONLY triplets in format: (subject, relation, object). One per line. No explanations."),
        ("user", prompt_txt)
    ])

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    # Data structures for tracking
    existing_triplets = set()  # For deduplication
    graph = {}  # For cycle detection
    extracted_counts = Counter()
    triplet_counter = 0
    skipped_counter = 0
     
    with open(OUTPUT_FILE, 'w', encoding='utf-8', buffering=1) as outp:
        for query_concept in tqdm(query_concepts, desc="Processing query concepts"):
            content_chunks = chunk_content(all_content, MAX_INPUT_CHAR)
            
            for chunk_idx, content_chunk in enumerate(content_chunks):
                relation_defs_text = ", ".join(RELATION_DEFS.keys())

                try:
                    invocation = prompt_template.invoke({
                        "query_concept": query_concept,            
                        "content": content_chunk,                  
                        "relation_definitions": relation_defs_text
                    })

                    response = model.invoke(invocation)
                    text = getattr(response, "content", "").strip()
                    
                except Exception as e:
                    logging.debug(f"Error calling model for '{query_concept}', chunk {chunk_idx}: {e}")
                    continue

                if not text or text.lower() == "none":
                    continue

                # Parse triplets
                parsed_triplets = parse_triplet_output(text)
                
                if not parsed_triplets:
                    logging.debug(f"No triplets parsed for '{query_concept}', chunk {chunk_idx}")
                    continue

                # Validate and write each triplet
                for triplet in parsed_triplets:
                    is_valid, reason = validate_triplet(triplet, existing_triplets, graph)
                    
                    if not is_valid:
                        skipped_counter += 1
                        logging.debug(f"Skipped invalid triplet: {reason} - {triplet}")
                        continue
                    
                    # Add to graph for cycle detection
                    s_lower = triplet['s'].strip().lower()
                    o_lower = triplet['o'].strip().lower()
                    if s_lower not in graph:
                        graph[s_lower] = []
                    graph[s_lower].append(o_lower)
                    
                    # Add to deduplication set
                    triplet_key = (s_lower, triplet['p'], o_lower)
                    existing_triplets.add(triplet_key)
                    
                    # Write to file
                    outp.write(json.dumps(triplet) + "\n")
                    outp.flush()
                    triplet_counter += 1
                    extracted_counts[triplet['p']] += 1
                    
                    # Log with counter
                    logging.info(f"[{triplet_counter}] ✓ {triplet['s']} --{triplet['p']}--> {triplet['o']}")

    # ---- Persist adjacency list ----
    adj_output_path = os.path.join(OUTPUT_BASE_DIR, "graph_adj.json")
    # Convert to serialisable format: {subject: [[relation, object], ...], ...}
    adj_serialisable = {s: [[r, o] for r, o in edges] for s, edges in graph.items()}
    with open(adj_output_path, "w", encoding="utf-8") as adj_fh:
        json.dump(adj_serialisable, adj_fh, indent=2, ensure_ascii=False)
    logging.info(f"Adjacency list saved → {adj_output_path}  ({len(adj_serialisable)} subjects)")

    # Final summary
    logging.info("")
    logging.info("=" * 70)
    logging.info(f"✓ EXTRACTION COMPLETE")
    logging.info(f"Total extracted triples: {triplet_counter}")
    logging.info(f"Total skipped: {skipped_counter}")
    logging.info(f"Output file: {OUTPUT_FILE}")
    logging.info(f"Adjacency list file: {adj_output_path}")
    logging.info(f"")
    logging.info(f"Triplets by relation type:")
    for rel, count in sorted(extracted_counts.items(), key=lambda x: x[1], reverse=True):
        logging.info(f"  {rel}: {count}")
    logging.info("=" * 70)


if __name__ == "__main__":
    extract_candidate_triples()


