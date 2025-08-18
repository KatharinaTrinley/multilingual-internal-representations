from huggingface_hub import login
from tqdm import tqdm
import json
import datasets
from datasets import Dataset
import argparse
import torch
import os
import sys
import random
import logging
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

login(token='YOUR_TOKEN') # toshiki's token

random.seed(42)
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("neural_machine_translation")

def main():
    data_split = "train"
    parser = argparse.ArgumentParser(description="inference on nmt")
    parser.add_argument('--model_name', type=str, default="CohereForAI/aya-23-8B", 
                        help='HuggingFace model name or path')
    parser.add_argument('--language_pair', type=str, default="en-de",
                        help='Language pair to use from google wmt24pp')
    parser.add_argument('--output_dir', type=str, default="outputs",
                        help='output directory')
    parser.add_argument('--fp16', action='store_true', 
                        help='Use half precision (fp16)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for processing')
    parser.add_argument('--max_samples', type=int, default=100,
                        help='Maximum number of samples to process')
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum sequence length')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    logger.info("Successfully loaded tokenizer")
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    logger.info("Successfully loaded model")
    if args.model_name == "meta-llama/Llama-3.1-8B":
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("since it is llama3, using eos token as the pad token!")

    model.to(device)
    model.eval()

    if args.fp16 and device.type == "cuda":
        logger.info("Using half precision (fp16)")
        model.half()

    logger.info(f"Loading wmt24 dataset for {args.language_pair}")
    src_lang, tgt_lang = args.language_pair.split('-')
    try:
        is_reverse_direction = False
        if src_lang == "en":
            dataset_key = f"en-{tgt_lang}" 
        else:
            dataset_key = f"en-{src_lang}"
            is_reverse_direction = True

        logger.info(f"Using dataset key: {dataset_key} for language pair: {args.language_pair}")
        if len(dataset_key) == 8:
            dataset = datasets.load_dataset("google/wmt24pp", dataset_key, trust_remote_code=True)
        else:
            dataset = datasets.load_dataset("json", data_files="./codemixing_dataset/wmt24pp_" + dataset_key + ".json")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        sys.exit(1)

    split_data = dataset[data_split]
    translations = []
    for example in tqdm(split_data):
        if is_reverse_direction:
            # If we're doing xx-en (reverse direction), swap source and target
            src_text = example.get("target", "")  # This is the target lang in the dataset
            tgt_text = example.get("source", "")    # This is English
        else:
            # For en-xx (forward direction)
            src_text = example.get("source", "")     # This is English
            tgt_text = example.get("target", "")  # This is the target lang
        
        if not src_text:
            continue
    
    translations = []
    D = {}
    n = 0
    for example in tqdm(split_data):
        if is_reverse_direction:
            # If we're doing xx-en (reverse direction), swap source and target
            src_text = example.get("target", "")  # Source becomes target
            tgt_text = example.get("source", "")  # Target becomes source
        else:
            # For en-xx (forward direction)
            src_text = example.get("source", "")  
            tgt_text = example.get("target", "")
        
        if not src_text:
            continue
        
        # Few shot prompt code
        r1 = n
        while n == r1:
            r1 = random.randint(0, len(split_data)-1)
        r2 = r1
        while (r2 == r1) or (r2 == n):
            r2 = random.randint(0, len(split_data)-1)
        n += 1
        
        if is_reverse_direction:
            src_ex1 = split_data[r1].get("target", "")
            tgt_ex1 = split_data[r1].get("source", "")
            src_ex2 = split_data[r2].get("target", "")
            tgt_ex2 = split_data[r2].get("source", "")
        else:
            src_ex1 = split_data[r1].get("source", "")
            tgt_ex1 = split_data[r1].get("target", "")
            src_ex2 = split_data[r2].get("source", "")
            tgt_ex2 = split_data[r2].get("target", "")
        
        task_text = f"""Translate the following text into English like the given 2 examples.
    [Example 1]
    source: {src_ex1}
    translation: {tgt_ex1}

    [Example 2]
    source: {src_ex2}
    translation: {tgt_ex2}

    [Task]
    source: {src_text}
    translation: """
        translations.append(task_text)
        D[task_text] = tgt_text

    if args.max_samples > 0 and len(translations) > args.max_samples:
        translations = translations[:args.max_samples]
        logger.info(f"Limited to {args.max_samples} samples")

    logger.info(f"Processing {len(translations)} samples")
    results = {}
    os.makedirs(args.output_dir, exist_ok = True)
    for i in tqdm(range(0, len(translations), args.batch_size)):
        batch = translations[i:i+args.batch_size]
        
        # Tokenize input to model (batch of strings)
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, 
                          max_length=args.max_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        logger.warning("Model returned logits instead of generated tokens; using greedy decoding.")
        # Perform greedy decoding manually
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                num_beams=1,
                early_stopping=True,
            )

        # Decode outputs
        decoded_outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        # Print or store results
        for src_prompt, generated_text in zip(batch, decoded_outputs):
            print(f"\n--- Translation ---")
            print(f"Prompt:\n{src_prompt}")
            print(f"Generated:\n{generated_text}")
            results[src_prompt] = {"generation": generated_text, "gold": D.get(src_prompt, "")}
            """
            try:
                results["source"] = D[src_prompt]"source"
                results["gold"] = D[src_prompt].get("gold_standard", "")
            except:
                logging.error("error with D")
                results = Dataset.from_dict(results)
                results.to_json(os.path.join("outputs",dataset_key), orient="records", lines=True)
                logging.info("dataset saved urgently because there was a problem")
                sys.exit(1)
            """
    with open(f"./{args.output_dir}/{dataset_key}_generation.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(f"./{args.output_dir}/{dataset_key}_gold.json", "w", encoding="utf-8") as f:
        json.dump(D, f, ensure_ascii=False, indent=2)
    logging.info("dataset saved without any problem")

if __name__ == "__main__":
    #model_name = "CohereForAI/aya-23-8B"
    #model_name = "hfl/chinese-llama-2-7b"
    main()