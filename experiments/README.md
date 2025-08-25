# 🔍 Logit Lens Experiments

This repository provides code and resources for conducting **logit lens experiments** on multilingual models.  
The experiments analyze **language probability distributions across layers** for various models.

---

## 📂 Repository Structure
- `logit_lens/` – Main directory containing logit lens code  
- `logit_lens/logit_lens_graphs/` – Precomputed probability graphs for ~50 tasks across 3 models  

---

## 📊 Example Graphs
Below are sample results from the logit lens experiments:

<img width="370" height="410" alt="averagetransru2ja_llama" src="https://github.com/user-attachments/assets/d6dbdaf4-eb3a-4cc6-84c6-1204014d71dd" />
<img width="370" height="410" alt="averagetransru2ja_chinesellama" src="https://github.com/user-attachments/assets/a1dc1124-e674-48e6-abb5-4458f66fd9e9" />
<img width="370" height="410" alt="averagetransru2ja_aya" src="https://github.com/user-attachments/assets/2a0858f6-83b3-4f2d-9b4f-19626d3a999d" />


---

## ⚙️ How to Run
A Colab notebook is provided for preprocessing and running the experiments:  
👉 [Colab Notebook](https://colab.research.google.com/drive/1jNmV8aH1bI93lj6WjRzlqT76c5dCoWF3?usp=sharing)

### Steps
1. **Preprocess the dataset** from Dumas et al. (2025).  
2. Run the experiment using one of the provided scripts:  
   - `llama_script.sh`  
   - `aya_script.sh`  
   - `chinesellama_script.sh`  
   Each script calls `logit_lens.py` to compute **language probabilities across all layers** and generate probability graphs.  
3. Output files include:  
   - Graphs of probability distributions  
   - Top-k predictions for each input (useful for further analysis)  
4. To run on a **single task only**, slightly modify `logit_lens.py` and rerun.  
   - Expected runtime: ~15 minutes per task on **1 × A100 (40GB)** GPU.  
   - Don’t forget to replace your **HuggingFace token** in the code.  

Additionally, you can compute **AUCs** from the output files.



# 🌐 Codemixing Machine Translation

This repository explores **codemixing in machine translation** using the [Google WMT24pp dataset](https://www.statmt.org/wmt24/) and a codemixing dictionary.  
It provides resources, experiments, and results on **codemixed data generation** and **translation quality**.

---

## 🚀 Overview
- Construct codemixed datasets from WMT24pp using a codemixing dictionary  
- Train and evaluate machine translation systems under codemixing conditions  
- Compare translation results across multiple language pairs  

👉 For step-by-step instructions on dataset construction, check the [Colab Notebook](https://colab.research.google.com/drive/1mN3aUPmxpjqoSOMW9xFRDY55QckjOdIe?usp=sharing).

---

## 📊 Results

### Codemixing MT Experiments
| Metric | Result Snapshot |
|--------|-----------------|
| **BLEU** | ![BLEU Results](https://github.com/user-attachments/assets/320f2892-7e9c-444f-b07c-9e8d6483a123) |
| **ChrF** | ![ChrF Results](https://github.com/user-attachments/assets/7aa8fe94-385c-426f-abd7-2e79a946ee10) |
| **COMET** | ![COMET Results](https://github.com/user-attachments/assets/a9fcb17c-26a6-43af-b7b2-17a2976239a6) |

---


# 🔬 Activation Frequency Experiments

This section investigates **neuron activation frequency and specialization** in multilingual and codemixed machine translation models.  
We build on previous work to analyze how neurons activate differently across **languages** and **codemixing conditions**.

---

## ⚙️ How It Works

- **Step 0:** Preprocess the dataset using **codemixed data** from the [Google WMT24pp dataset](https://www.statmt.org/wmt24/) with a codemixing dictionary.  
  - `data/wmt24ppdict/` contains a word-to-word codemixing dictionary for WMT24pp.  
  - See the [Colab Notebook](https://colab.research.google.com/drive/1mN3aUPmxpjqoSOMW9xFRDY55QckjOdIe?usp=sharing) for step-by-step instructions on converting WMT24pp into codemixed data.  

- **Step 1:** Run `get_neurons_wmt24_codemixed.py`  
  → Saves **activation counts** (non-zero activations) for each neuron.  

- **Step 2:** Run `improved_fig_codemixing_simplified2.py`  
  → Generates **visualizations** of activation frequencies.  
  *(Remember to update the file path to your activation data.)*  

---

🔑 **Key Difference from Tan et al. (2024):**  
We introduced a custom function `remove_shared_neurons` to filter out neurons that are shared across all languages.  
Only neurons that are **not shared globally** are treated as **language-specific neurons**.


---

## 📚 Reference Dataset
For logit lens experiments, we used the dataset introduced in:  

> Dumas et al. (2024). *Separating Tongue from Thought: Activation Patching Reveals Language-Agnostic Concept Representations in Transformers*.  
> [[arXiv link]](https://arxiv.org/abs/2411.08745) | [Official repo](https://github.com/Butanium/llm-lang-agnostic)


> Tan, Shaomu, Wu, Di, & Monz, Christof. (2024). *Neuron Specialization: Leveraging Intrinsic Task Modularity for Multilingual Machine Translation*.  
> In **Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing (EMNLP 2024)**, Miami, Florida, USA.  
> [ACL Anthology](https://aclanthology.org/2024.emnlp-main.374/) | [DOI](https://doi.org/10.18653/v1/2024.emnlp-main.374)

```bibtex
@misc{dumas2025separatingtonguethoughtactivation,
    title        = {Separating Tongue from Thought: Activation Patching Reveals Language-Agnostic Concept Representations in Transformers}, 
    author       = {Clément Dumas and Chris Wendler and Veniamin Veselovsky and Giovanni Monea and Robert West},
    year         = {2025},
    eprint       = {2411.08745},
    archivePrefix= {arXiv},
    primaryClass = {cs.CL},
    url          = {https://arxiv.org/abs/2411.08745}
}

```bibtex
@inproceedings{tan-etal-2024-neuron,
    title     = {Neuron Specialization: Leveraging Intrinsic Task Modularity for Multilingual Machine Translation},
    author    = {Tan, Shaomu and Wu, Di and Monz, Christof},
    booktitle = {Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing},
    month     = nov,
    year      = {2024},
    address   = {Miami, Florida, USA},
    publisher = {Association for Computational Linguistics},
    url       = {https://aclanthology.org/2024.emnlp-main.374/},
    doi       = {10.18653/v1/2024.emnlp-main.374},
    pages     = {6506--6527}
}
