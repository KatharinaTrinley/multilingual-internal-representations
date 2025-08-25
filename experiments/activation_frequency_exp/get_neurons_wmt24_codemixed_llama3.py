import argparse
import logging
import os
import sys
import torch
import pickle
from collections import defaultdict
from tqdm import tqdm
import datasets
import random
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from huggingface_hub import HfApi, Repository, login

login(token='YOUR_TOKEN') # toshiki's token

random.seed(42)
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("get_neurons_aya23_cohere_tokenizer")


def register_activation_hooks(model, forward_hooks, layer_inputs, layer_outputs):
    """Register hooks to capture activations from MLP layers in transformer blocks"""
    
    for i, layer in enumerate(model.model.layers):
        # Hook for capturing activations from down_proj 
        layer_name = f"layer_{i}"
        
        def make_hook(layer_name, sub_layer_name):
            def hook(module, inp, out):
                layer_inputs[layer_name][sub_layer_name] = inp[0].detach().cpu() # for memory efficiency
                layer_outputs[layer_name][sub_layer_name] = out.detach().cpu()
            return hook
        
        forward_hook = layer.mlp.down_proj.register_forward_hook(
            make_hook(layer_name, "down_proj")
        )
        forward_hooks.append(forward_hook)
    
    return forward_hooks

def neuron_count(layer_inputs, ffn_dim):
    """Count non-zero activations in each layer"""
    temp = {}
    for l in layer_inputs:
        if 'down_proj' in layer_inputs[l]:
            # Get indices of non-zero elements
            nonzero_indices = torch.nonzero(layer_inputs[l]['down_proj'])
            if nonzero_indices.size(0) > 0:
                # Extract the last dimension indices (neuron indices)
                neuron_indices = nonzero_indices[:, -1]
                # Count occurrences of each index
                temp[l] = torch.bincount(neuron_indices, minlength=ffn_dim)
            else:
                temp[l] = torch.zeros(ffn_dim)
    return temp

def main():
    parser = argparse.ArgumentParser(description="Get neuron activations from Aya23 model using Cohere tokenizer")
    parser.add_argument('--model_name', type=str, default="CohereForAI/aya-23-8B", 
                        help='HuggingFace model name or path')
    parser.add_argument('--save_path', type=str, required=True, 
                        help='Path to save activation data')
    parser.add_argument('--language_pair', type=str, default="en-de",
                        help='Language pair to use from EC40 (e.g., en-de)')
    parser.add_argument('--split', type=str, default="test",
                        help='Dataset split to use (train, validation, test)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for processing')
    parser.add_argument('--max_samples', type=int, default=100,
                        help='Maximum number of samples to process')
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum sequence length')
    parser.add_argument('--fp16', action='store_true', 
                        help='Use half precision (fp16)')
    args = parser.parse_args()

    # source and target languages from the language pair
    src_lang, tgt_lang = args.language_pair.split('-')
    
    os.makedirs(os.path.join(args.save_path, args.language_pair), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info(f"Loading model: {args.model_name}")
    try:

        #from transformers import CohereTokenizer, AutoModelForCausalLM
        # had some issues with the CohereTokenizer, so I explicitly load it
        logger.info("Loading Cohere tokenizer explicitly")
        #tokenizer = CohereTokenizer.from_pretrained(args.model_name)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Successfully loaded Cohere tokenizer")
        
        model = AutoModelForCausalLM.from_pretrained(args.model_name)
        logger.info("Successfully loaded model")
    except Exception as e:
        logger.error(f"Error loading with CohereTokenizer: {e}")
        sys.exit(1)
    
    # Move model and set to evaluation mode
    model.to(device)
    model.eval()
    
    if args.fp16 and device.type == "cuda":
        logger.info("Using half precision (fp16)")
        model.half()

    # Get the FFN dimension (hidden size of the MLP)
    ffn_dim = model.model.layers[0].mlp.up_proj.out_features
    logger.info(f"FFN dimension: {ffn_dim}")

    # Setup for recording activations
    layer_inputs = defaultdict(dict)
    layer_outputs = defaultdict(dict)
    forward_hooks = []
    
    # Register hooks
    forward_hooks = register_activation_hooks(model, forward_hooks, layer_inputs, layer_outputs)
    
    # init. counters for alive neurons
    layer_names = [f"layer_{i}" for i in range(len(model.model.layers))]
    alive_neurons = {layer_name: torch.zeros(ffn_dim).long() for layer_name in layer_names}

    logger.info(f"Loading wmt24 dataset for {args.language_pair}")
    try:
        # EC40 always requires the "en-xx" format for loading, even when working with "xx-en" direction
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
        if args.split not in dataset:
            available_splits = list(dataset.keys())
            logger.warning(f"Split '{args.split}' not found in dataset. Available splits: {available_splits}")
            if "test" in available_splits:
                args.split = "test"
                logger.info(f"Using 'test' split instead.")
            else:
                args.split = available_splits[0]
                logger.info(f"Using '{args.split}' split instead.")
        
        split_data = dataset[args.split]
        logger.info(f"Loaded {len(split_data)} examples from '{args.split}' split")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        sys.exit(1)

    #toshiki comment: change below for few-shot
    translations = []
    n = 0
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
        
    #toshiki comment: change below for few-shot
    translations = []
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

        #toshiki comment: the following is the zero shot task instruction
        # prompt the model for translation    
        """
        task_text = f"Translate from {src_lang} to {tgt_lang}: {src_text}"
        translations.append(task_text)
        """
    
    if args.max_samples > 0 and len(translations) > args.max_samples:
        translations = translations[:args.max_samples]
        logger.info(f"Limited to {args.max_samples} samples")

    logger.info(f"Processing {len(translations)} samples")
    for i in tqdm(range(0, len(translations), args.batch_size)):
        batch = translations[i:i+args.batch_size]
        
        # Tokenize input to model (batch of strings)
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, 
                          max_length=args.max_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Count active neurons 
        temp_counts = neuron_count(layer_inputs, ffn_dim)
    
        for layer_name in alive_neurons:
            if layer_name in temp_counts:
                alive_neurons[layer_name] += temp_counts[layer_name]
        layer_inputs.clear() # due to memory
        layer_outputs.clear()

    save_file = os.path.join(args.save_path, args.language_pair, "activations.pkl")
    logger.info(f"Saving activations to: {save_file}")
    with open(save_file, "wb") as f:
        pickle.dump(alive_neurons, f)

    # remove hooks from the model
    for hook in forward_hooks:
        hook.remove()

    logger.info("Done!")

if __name__ == "__main__":
    main()
