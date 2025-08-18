from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
import datetime
import os
from huggingface_hub import login
from datasets import load_dataset, Dataset
from tqdm import tqdm
import ast
import argparse
import pandas as pd
from matplotlib import pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import numpy as np
import os
import json
from matplotlib import colormaps
from sklearn.metrics import auc
import logging

logging.basicConfig(
    filename='aya_logit_lens.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(funcName)s - %(message)s'
)

class Early_Decoding:
    def __init__(self, model_id, mode="standard"):
        self.model_id = model_id
        self.mode = mode

    def load_tokenizer_and_model(self, output_hidden_states=True):
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        model = AutoModelForCausalLM.from_pretrained(self.model_id, output_hidden_states=output_hidden_states)
        return tokenizer, model
    
    def aya_logit_lens_experiment(self, json_path, task_name, top_k=10):
        ds = load_dataset("json", data_files=json_path)["train"]
        ds_trans = load_dataset("csv", data_files="word_translation2.csv")["train"]

        lang_list = ['fr', 'de', 'ru', 'en', 'zh', 'es', 'ja', 'ko', 'et', 'fi', 'nl', 'hi', 'it']
        translation_dict = {}

        for row in ds_trans:
            for lang in lang_list:
                try:
                    row[lang] = ast.literal_eval(row[lang])
                except:
                    #this line is probably unnecessary
                    L = []
                    L.append(row[lang])
                    #row[lang] = [row[lang]]
            translation_dict[row["word_original"]] = row

        tokenizer, model = self.load_tokenizer_and_model()
        '''
        #below is for Llora
        if self.mode == "baseline":
            model = PeftModel.from_pretrained(model, "tona3738/aya_qlora_wo_codemixing_10Mtokens")
        elif self.mode == "codemixing_model":
            model = PeftModel.from_pretrained(model, "tona3738/aya_qlora_with_codemixing_10Mtokens")
        '''
        output_dir = f"{self.mode}{task_name}_final_output"
        os.makedirs(output_dir, exist_ok=True)

        to_write = ""
        to_write2 = ""
        Prompt2topkpred = {}
        Prompt2targetwords = {}
        error_count = 0

        for example in tqdm(ds):
            prompt = example["prompt"]
            word_original = example["word_original"]
            to_write += f"prompt: {prompt}\n\n"
            to_write2 += f"prompt: {prompt}\n\n"

            encoded_input = tokenizer(prompt, return_tensors='pt')
            with torch.no_grad():
                outputs = model(**encoded_input)

            hidden_states = outputs.hidden_states
            last_token_pos = encoded_input["input_ids"].size(1) - 1

            intermediate_predictions = []
            Layer_and_lang = []

            try:
                for layer_idx, hidden_state in enumerate(hidden_states):
                    to_write2 += f"layer:{layer_idx}\n"
                    last_token_hidden = hidden_state[:, last_token_pos, :]

                    if self.mode in ["baseline", "codemixing_model"]:
                        norm_hidden = model.base_model.model.model.norm(last_token_hidden)
                    else:
                        norm_hidden = model.model.norm(last_token_hidden)

                    logits = model.lm_head(norm_hidden)
                    probs = F.softmax(logits, dim=-1)

                    top_k_probs, top_k_indices = torch.topk(probs, k=top_k, dim=-1)
                    top_k_tokens = tokenizer.batch_decode(top_k_indices[0], skip_special_tokens=True)
                    intermediate_predictions.append({
                        "layer": layer_idx,
                        "predictions": [
                            {"token": tok, "probability": prob.item()}
                            for tok, prob in zip(top_k_tokens, top_k_probs[0])
                        ]
                    })
                    Prompt2topkpred[prompt] = intermediate_predictions

                    last_token_probs = probs[0, :]
                    lang2word_probs = {}

                    for lang in lang_list:
                        word_list = translation_dict[word_original][lang]
                        word2prob = {}

                        for word in word_list:
                            to_write2 += f"word:{word}|"
                            tokens = tokenizer.tokenize(word)
                            token_ids = tokenizer.convert_tokens_to_ids(tokens)
                            if not token_ids:
                                continue
                            token_id = token_ids[0]
                            decoded_token = tokenizer.decode(token_id)
                            to_write2 += f"token:{decoded_token}|{tokens[0]}|"
                            prob = last_token_probs[token_id].item()
                            to_write2 += f"probability:{prob}\n"
                            word2prob[word] = (decoded_token, prob)

                        lang2word_probs[lang] = word2prob

                    Layer_and_lang.append({
                        "layer": layer_idx,
                        "predictions": lang2word_probs
                    })

                Prompt2targetwords[prompt] = Layer_and_lang

            except Exception as e:
                error_count += 1
                logging.error(f"Error for word_original='{word_original}' | prompt='{prompt}' | Exception: {str(e)}")
                continue

            for layer_prediction in intermediate_predictions:
                layer = layer_prediction["layer"]
                to_write += f"\nLayer {layer} Predictions:\n"
                for prediction in layer_prediction["predictions"]:
                    token = prediction["token"]
                    prob = prediction["probability"]
                    to_write += f"  Token: '{token}' | Probability: {prob:.4f}\n"
            to_write += "\n\n"
            to_write2 += "\n\n"
            Prompt2topkpred[example["prompt"]] = intermediate_predictions
            Prompt2targetwords[example["prompt"]] = {"word_original":example["word_original"], "Layer_and_lang":Layer_and_lang}

        logging.info(f"Total errors encountered in aya_logit_lens_experiment: {error_count}")

        self.write_json(f"{output_dir}/topk_dict.json", Prompt2topkpred)
        self.write_json(f"{output_dir}/target_dict.json", Prompt2targetwords)
        #self.write_txt(f"{output_dir}/logit_lens_topk.txt", to_write)
        #self.write_txt(f"{output_dir}/logit_lens_target.txt", to_write2)

        task_json_path = json_path
        topk_dict_path = f"{output_dir}/topk_dict.json"
        word_translation_csv_path = "./word_translation2.csv"
        self.clean_average_graph(task_json_path, topk_dict_path, word_translation_csv_path, task_name, tokenizer)

    def load_json(self, file_name):
        d = {}
        with open(file_name, mode="r") as f:
            d = json.load(f)
        return d

    def write_json(self, file_name,L):
        with open(file_name, mode="w") as f:
            d = json.dumps(L)
            f.write(d)

    def clean_dict(self, d, task_ds):
        newD = {}
        for i in task_ds:
            newD[i["word_original"]] = d[i["prompt"]]
        return newD

    def create_translation_dict(self, word_translation_csv_path):
        trans_ds = load_dataset("csv",data_files = word_translation_csv_path)
        lang_list = ['fr', 'de', 'ru', 'en', 'zh', 'es', 'ja', 'ko', 'et', 'fi', 'nl', 'hi', 'it']
        translation_dict = {}
        for i in trans_ds["train"]:
            translation_dict[i["word_original"]] = []
            for lang in lang_list:
                word_list = ast.literal_eval(i[lang])
                for w in word_list:
                    translation_dict[i["word_original"]].append((w, lang))
        return translation_dict

    def plot_and_save(self, layers, lang_list, word_original, task_name):
        if word_original != "average":
            os.makedirs(f"./{task_name}_fixed_graphs", exist_ok=True)
            df = pd.DataFrame(layers, columns=lang_list)
            ax = df.plot()
            ax.set_title(f"{word_original}{task_name}")
            ax.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', borderaxespad=0., ncol=2)
            #plt.savefig(f'./{task_name}_fixed_graphs/{word_original}{task_name}.png', bbox_inches='tight')
            plt.close()
        else:
            os.makedirs("./final_output", exist_ok=True)
            df = pd.DataFrame(layers, columns=lang_list)
            plt.style.use("seaborn-v0_8-muted")
            # Create figure and axis
            fig, ax = plt.subplots(figsize=(7, 6))  # Adjust figure size
            base_cmap = colormaps.get_cmap("tab20")
            colors = base_cmap(np.linspace(0, 1, 13))
            # Plot each column with a different color
            for i, col in enumerate(df.columns):
                ax.plot(df.index, df[col], label=col, color=colors[i])
            # Set title and labels
            ax.set_title(f"{word_original}{task_name}", fontsize=14, fontweight='bold')
            ax.set_xlabel("Layer", fontsize=12)
            ax.set_ylabel("Probability", fontsize=12)
            # Set y-axis limit
            ax.set_ylim(0, 1.0)
            # Customize legend position and styling
            ax.legend(
                bbox_to_anchor=(0.5, -0.15),
                loc='upper center',
                borderaxespad=0.,
                ncol=3,  # Increase number of columns for a compact look
                fontsize=10
            )
            # Add grid for better readability
            ax.grid(True, linestyle="--", alpha=0.6)
            # Save the figure with a tight bounding box
            plt.savefig(f'./final_output/{word_original}{task_name}.png', bbox_inches='tight', dpi=300)
            # Close plot to free memory
            plt.close()

    def clean_average_graph(self, task_json_path, topk_dict_path, word_translation_csv_path, task_name, tokenizer):
        os.makedirs("./final_output", exist_ok=True)
        os.makedirs("./final_output/X", exist_ok=True)

        lang_list = ['fr', 'de', 'ru', 'en', 'zh', 'es', 'ja', 'ko', 'et', 'fi', 'nl', 'hi', 'it']
        lang_idx_dict = {lang: idx for idx, lang in enumerate(lang_list)}

        # Initialize containers
        X, y = [], []
        total_prob = np.zeros((33, 13))
        ambiguity_matrix = np.zeros((13, 13), dtype=int)
        ambiguous_cases = []
        processed_count = 0

        # Load datasets
        task_ds = load_dataset("json", data_files=task_json_path)["train"]
        topk_dict = self.load_json(topk_dict_path)
        filtered_topk_dict = self.clean_dict(topk_dict, task_ds)
        translation_dict = self.create_translation_dict(word_translation_csv_path)

        for word_original in tqdm(filtered_topk_dict):
            try:
                layerwise_probs = np.zeros((33, 13))
                layer_data = filtered_topk_dict[word_original]
                translation_list = translation_dict[word_original]

                for layer in layer_data:
                    layer_idx = layer["layer"]
                    prob_dist = layer["predictions"]

                    for pred in prob_dist:
                        token = pred["token"].strip()
                        prob = pred["probability"]
                        if prob <= 0.1:
                            continue

                        matched_langs = set()
                        for word, lang in translation_list:
                            if lang not in lang_idx_dict:
                                continue

                            try:
                                first_token = tokenizer.tokenize(word)[0]
                                token_id = tokenizer.convert_tokens_to_ids(first_token)
                                decoded_token = tokenizer.decode(token_id)
                            except Exception:
                                continue  # skip problematic tokenizations

                            lang_idx = lang_idx_dict[lang]
                            candidates = [
                                (lang, word),
                                (lang, first_token),
                                (lang, decoded_token)
                            ]

                            for match_lang, match_token in candidates:
                                if token == match_token and (match_lang, match_token) not in matched_langs:
                                    layerwise_probs[layer_idx][lang_idx] += prob
                                    matched_langs.add((match_lang, match_token))
                                    break  # only one match per (lang, token)

                        #   After processing all langs, check if multiple langs matched
                        langs_only = sorted({lang for lang, _ in matched_langs})
                        if len(langs_only) >= 2:
                            for i in range(len(langs_only)):
                                for j in range(i + 1, len(langs_only)):
                                    idx1 = lang_idx_dict[langs_only[i]]
                                    idx2 = lang_idx_dict[langs_only[j]]
                                    ambiguity_matrix[idx1][idx2] += 1
                                    ambiguity_matrix[idx2][idx1] += 1  # symmetric

                            ambiguous_cases.append({
                                "word_original": word_original,
                                "token": token,
                                "languages": ",".join(langs_only)
                            })

                total_prob += layerwise_probs
                X.append(layerwise_probs[:].tolist()) #X.append(layerwise_probs[:-2].tolist()) if want to Exclude last 2 layers
                #y.append(layerwise_probs[-1][6])  # Index 6 = Japanese
                processed_count += 1

                #   Save partial data
                self.write_json(f"./final_output/X/{task_name}_X.json", X)
                #self.write_json(f"./XandY/{task_name}_y.json", y)

            except Exception as e:
                logging.error(f"[clean_average_graph] Error processing '{word_original}': {e}")
                continue

        if processed_count == 0:
            logging.warning("[clean_average_graph] No samples were successfully processed.")
            return

        average_prob = total_prob / processed_count
        self.write_json(f"./final_output/{task_name}_total_prob.json", average_prob.tolist())
        self.plot_and_save(average_prob, lang_list, "average", task_name)

        # Save ambiguity matrix and cases
        df_matrix = pd.DataFrame(ambiguity_matrix, index=lang_list, columns=lang_list)
        df_matrix.to_csv(f"./final_output/{task_name}_ambiguity_matrix.csv")

        df_ambiguous = pd.DataFrame(ambiguous_cases)
        df_ambiguous.to_csv(f"./final_output/{task_name}_ambiguous_cases.csv", index=False)

def build_parser():
    parser = argparse.ArgumentParser(description='logit lens')
    parser.add_argument('--model_id', required=True, help='hfl/chinese-llama-2-7b')
    parser.add_argument('--task_name', required=True, help='transen2zh_chinesellama')
    parser.add_argument('--json_path', required=True, help='translation2_en2zh.json')
    return parser

if __name__ == "__main__":
    huggingface_token = "YOUR_TOKEN"
    login(huggingface_token)
    global args
    parser = build_parser()
    args = parser.parse_args()
    model_id = args.model_id #"hfl/chinese-llama-2-7b"
    task_name = args.task_name #"transen2zh_chinesellama"
    json_path = args.json_path #"translation2_en2zh.json"
    early_decoding = Early_Decoding(model_id)
    early_decoding.aya_logit_lens_experiment(json_path, task_name)
