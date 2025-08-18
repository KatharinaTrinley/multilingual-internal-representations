# What Language(s) Does Aya-23 Think In? How Multilinguality Affects Internal Language Representations

## Abstract

Large language models (LLMs) excel at multilingual tasks, yet their internal language processing remains poorly understood. We analyze how Aya-23-8B, a decoder-only LLM trained on balanced multilingual data, handles code-mixed, cloze, and translation tasks compared to predominantly monolingual models like Llama 3 and Chinese-LLaMA-2. Using logit lens and neuron specialization analyses, we find: (1) Aya-23 activates typologically related language representations during translation, unlike English-centric models that rely on a single pivot language; (2) code-mixed neuron activation patterns vary with mixing rates and are shaped more by the base language than the mixed-in one; and (3) Aya-23's language-specific neurons for code-mixed inputs concentrate in final layers, diverging from prior findings on decoder-only models. Neuron overlap analysis further shows that script similarity and typological relations impact processing across model types. These findings reveal how multilingual training shapes LLM internals and inform future cross-lingual transfer research.

## Repository Structure

```
├── README.md
├── src/
│   ├── logit_lens/           # Logit lens analysis code
│   ├── neuron_analysis/      # Neuron specialization experiments
│   ├── code_mixing/          # Code-mixed dataset creation
├── data/
│   ├── code_mixed_dataset/   # Our code-mixed dataset (released on huggingface or here)
│   └── sample_data/         # Sample data for testing (maybe add, if time allows)
├── experiments/
│   ├── logit_lens_results/   # Results from logit lens experiments
│   ├── neuron_results/       # Neuron specialization results
│   └── additional_figures/   # Extra visualizations not in paper
├── requirements.txt
└── LICENSE
```

## Key Findings

### 1. Multilingual Processing Patterns (H1)
- **Aya-23-8B** activates multiple languages simultaneously during translation (e.g., Japanese during Chinese translation)
- **Llama 3.1-8B** follows English-centric processing with English dominance across layers
- **Chinese-LLaMA-2-7B** exhibits Chinese-dominant processing even for English inputs

### 2. Neuron Specialization (H2, H3)
- Language-specific neurons in Aya-23 concentrate in **final layers (27-31)**, unlike prior findings showing early+late distribution
- French-based code-mixed inputs show consistently higher neuron clustering than Chinese-based inputs across all models
- Base language drives neuron sharing more than mixed-in language

### 3. Code-Mixing Behavior (H4)
- Script similarity affects processing effectiveness (Latin > Cross-script pairs)
- Translation quality degrades with increased mixing rates, but Aya-23 shows greater resilience
- Typological relationships influence neuron overlap patterns

## Models Analyzed

| Model | Size | Training Focus | Key Characteristics |
|-------|------|----------------|-------------------|
| **Aya-23-8B** | 8B | Balanced multilingual (23 languages) | Balanced training on multilingual corpus |
| **Llama 3.1-8B** | 8B | English-dominant (~8% multilingual) | Predominantly English training data |
| **Chinese-LLaMA-2-7B** | 7B | Chinese-specialized | Mandarin-adapted with LoRA fine-tuning |

## Languages Studied

**Logit Lens Experiments:** 13 languages (de, en, es, et, fi, fr, hi, it, ja, ko, nl, ru, zh)

**Code-Mixing Experiments:**
- **Base languages:** French (fr), Chinese (zh)
- **Partner languages:** English (en), Spanish (es), Italian (it), Japanese (ja), Korean (ko)
- **Mixing ratios:** 25%, 50%, 75%

## Methodology

### Logit Lens Analysis
- Projects intermediate hidden states into vocabulary space at each layer
- Tracks language-specific token probabilities across 54 translation tasks
- Uses Dumas et al. (2024) dataset with minimal token overlap between languages

### Neuron Specialization
- **Tan et al. approach:** Binary ReLU activation analysis with IoU overlap measurement
- **Kojima et al. approach:** Average Precision scoring for language-specific neuron identification
- Focuses on Feed-Forward Network layers across all 32 transformer layers

### Code-Mixed Dataset
- Created from WMT24++ parallel corpus using rule-based word-to-word translation
- Comprehensive bilingual dictionaries via Google Translate
- Controlled mixing ratios for systematic analysis
- Can be found under [HUGGINGFACE LINK]

## Usage

```bash
# Clone the repository
git clone https://github.com/your-username/aya-23-internal-language.git
cd aya-23-internal-language

# Install dependencies
pip install -r requirements.txt

# Run logit lens analysis
#TODO

# Run neuron specialization analysis
#TODO

# Generate code-mixed dataset
#TODO
```

## Additional Materials

The `experiments/additional_figures/` directory contains:
- Extended logit lens visualizations for all language pairs
- Layer-wise neuron specialization heatmaps
- Translation quality metrics (BLEU, chrF, COMET) across all mixing ratios
- Statistical significance tables for all model comparisons
- Neuron overlap matrices for all language combinations

## Citation

```bibtex
@article{trinley2025language,
  title={What Language (s) Does Aya-23 Think In? How Multilinguality Affects Internal Language Representations},
  author={Trinley, Katharina and Nakai, Toshiki and Anikina, Tatiana and Baeumel, Tanja},
  journal={arXiv preprint arXiv:2507.20279},
  year={2025}
}
```

```
@inproceedings{kojima-etal-2024-multilingual,
  title = "On the Multilingual Ability of Decoder-based Pre-trained Language Models: Finding and Controlling Language-Specific Neurons",
  author = "Kojima, Takeshi and Okimura, Itsuki and Iwasawa, Yusuke and Yanaka, Hitomi and Matsuo, Yutaka",
  editor = "Duh, Kevin and Gomez, Helena and Bethard, Steven",
  booktitle = "Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)",
  month = jun,
  year = "2024",
  address = "Mexico City, Mexico",
  publisher = "Association for Computational Linguistics",
  url = "https://aclanthology.org/2024.naacl-long.384/",
  doi = "10.18653/v1/2024.naacl-long.384",
  pages = "6919--6971"
}
```

```
@inproceedings{tan-etal-2024-neuron,
  title = "Neuron Specialization: Leveraging Intrinsic Task Modularity for Multilingual Machine Translation",
  author = "Tan, Shaomu and Wu, Di and Monz, Christof",
  editor = "Al-Onaizan, Yaser and Bansal, Mohit and Chen, Yun-Nung",
  booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
  month = nov,
  year = "2024",
  address = "Miami, Florida, USA",
  publisher = "Association for Computational Linguistics",
  url = "https://aclanthology.org/2024.emnlp-main.374/",
  doi = "10.18653/v1/2024.emnlp-main.374",
  pages = "6506--6527"
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

- Katharina Trinley: katr00001@stud.uni-saarland.de
- Toshiki Nakai: tona00002@stud.uni-saarland.de
