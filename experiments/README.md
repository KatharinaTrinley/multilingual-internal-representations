# Codemixing Machine Translation

This repository explores **codemixing in machine translation** using the [Google WMT24pp dataset](https://www.statmt.org/wmt24/) and a codemixing dictionary.  
It provides resources, experiments, and results on codemixed data generation and translation quality.

---

## 🚀 Overview
- Construct codemixed datasets from WMT24pp using a codemixing dictionary  
- Train and evaluate machine translation systems under codemixing conditions  
- Compare results across language pairs  

For step-by-step instructions on how to construct codemixed data from WMT24pp, see the [Colab Notebook](https://colab.research.google.com/drive/1mN3aUPmxpjqoSOMW9xFRDY55QckjOdIe?usp=sharing).

---

## 📊 Results

### Codemixing MT Experiments
| Metric | Result Snapshot |
|--------|-----------------|
| **BLEU** | ![Results1](https://github.com/user-attachments/assets/320f2892-7e9c-444f-b07c-9e8d6483a123) |
| **ChrF** | ![Results2](https://github.com/user-attachments/assets/7aa8fe94-385c-426f-abd7-2e79a946ee10) |
| **COMET** | ![Results3](https://github.com/user-attachments/assets/a9fcb17c-26a6-43af-b7b2-17a2976239a6) |

---

## 🔗 Related Work

The codebase is adapted in part from **Tan et al. (2024)**, with minor modifications for our experiments:  

> Tan, Shaomu, Wu, Di, & Monz, Christof. (2024). *Neuron Specialization: Leveraging Intrinsic Task Modularity for Multilingual Machine Translation*.  
> In **Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing (EMNLP 2024)**, Miami, Florida, USA. Association for Computational Linguistics.  
> [ACL Anthology](https://aclanthology.org/2024.emnlp-main.374/) | [DOI](https://doi.org/10.18653/v1/2024.emnlp-main.374)

```bibtex
@inproceedings{tan-etal-2024-neuron,
    title     = "Neuron Specialization: Leveraging Intrinsic Task Modularity for Multilingual Machine Translation",
    author    = "Tan, Shaomu and Wu, Di and Monz, Christof",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month     = nov,
    year      = "2024",
    address   = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url       = "https://aclanthology.org/2024.emnlp-main.374/",
    doi       = "10.18653/v1/2024.emnlp-main.374",
    pages     = "6506--6527"
}
