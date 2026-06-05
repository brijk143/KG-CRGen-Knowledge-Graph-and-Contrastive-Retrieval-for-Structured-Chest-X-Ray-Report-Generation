import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import json
import random
import logging
from dotenv import load_dotenv
from tqdm import tqdm
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)
hf_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
if not hf_token:
    raise RuntimeError(
        f"HUGGINGFACEHUB_ACCESS_TOKEN not found in {env_path}"
    )

print(" Hugging Face token loaded")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)
def extract_seed_entities(
    model_name,
    raw_text_file,
    output_file,
    prompt_path,
    max_input_char,
    num_samples
):
    logging.info("Starting Step 1: Seed Entity Extraction (Hugging Face LLM)")
    model = ChatHuggingFace(
        llm=HuggingFaceEndpoint(
            repo_id=model_name,
            temperature=0.1,
            max_new_tokens=512,
            huggingfacehub_api_token=hf_token
        )
    )
    try:
        with open(raw_text_file, "r", encoding="utf-8") as rf:
            full_text = rf.read()
    except IOError as e:
        logging.error(f"Could not read text file {raw_text_file}: {e}")
        return

    if not full_text.strip():
        logging.error("Text file is empty")
        return

    text_chunks = []
    if "\n\n" in full_text:
        text_chunks = [
            chunk.strip()
            for chunk in full_text.split("\n\n")
            if chunk.strip()
        ]

    elif len(full_text) > max_input_char * 2:
        sentences = full_text.split(".")
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk + sentence) < max_input_char:
                current_chunk += sentence + "."
            else:
                if current_chunk:
                    text_chunks.append(current_chunk.strip())
                current_chunk = sentence + "."

        if current_chunk:
            text_chunks.append(current_chunk.strip())

    else:
        text_chunks = [full_text]

    logging.info(f"Split text into {len(text_chunks)} chunks")

   
    if len(text_chunks) > num_samples:
        samples = random.sample(text_chunks, num_samples)
        logging.info(f"Sampled {num_samples} chunks from {len(text_chunks)} total")
    else:
        samples = text_chunks
        logging.info(f"Using all {len(text_chunks)} chunks")

    
    try:
        with open(prompt_path, "r", encoding="utf-8") as pf:
            prompt_txt = pf.read()
    except IOError as e:
        logging.error(f"Could not read prompt file {prompt_path}: {e}")
        return

    candidate_concepts = []
    successful_extractions = 0

    for i, text_chunk in enumerate(
        tqdm(samples, desc="Extracting seed entities")
    ):
        try:
            truncated_text = text_chunk[:max_input_char]

            full_prompt = f"{prompt_txt}\n\nContent:\n{truncated_text}"

            response = model.invoke(full_prompt)

            response_text = (
                response.content
                if hasattr(response, "content")
                else str(response)
            )

            candidate_concepts.append(response_text.strip())
            successful_extractions += 1

        except Exception as e:
            logging.warning(f"Error processing chunk {i + 1}: {e}")
            continue

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as wf:
        for concepts in candidate_concepts:
            wf.write(concepts + "\n\n")

    logging.info(
        f"Extracted entities from "
        f"{successful_extractions}/{len(samples)} chunks"
    )
    logging.info(f"Results saved to {output_file}")


def main():

    RAW_TEXT_FILE = "All_Classes_Description.txt"
    OUTPUT_DIR = "output"
    OUTPUT_FILE_NAME = "seed_entities.txt"
    OUTPUT_FILE = os.path.join(OUTPUT_DIR, OUTPUT_FILE_NAME)
    PROMPT_PATH = "prompts/prompts_step1.txt"
    MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

    MAX_INPUT_CHAR = 10000
    NUM_SAMPLES = 500

    if not os.path.isfile(RAW_TEXT_FILE):
        sys.exit(f"ERROR: Text file {RAW_TEXT_FILE} does not exist")

    if not os.path.isfile(PROMPT_PATH):
        sys.exit(f"ERROR: Prompt file {PROMPT_PATH} does not exist")

    extract_seed_entities(
        model_name=MODEL_NAME,
        raw_text_file=RAW_TEXT_FILE,
        output_file=OUTPUT_FILE,
        prompt_path=PROMPT_PATH,
        max_input_char=MAX_INPUT_CHAR,
        num_samples=NUM_SAMPLES
    )


if __name__ == "__main__":
    main()