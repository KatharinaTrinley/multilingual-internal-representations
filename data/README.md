# Codemixing Data Construction (WMT24pp)

This repository provides tools for constructing **codemixed data** from the [Google WMT24pp dataset](https://www.statmt.org/wmt24/) using a codemixing dictionary.  

📓 A ready-to-use [Colab Notebook](https://colab.research.google.com/drive/1mN3aUPmxpjqoSOMW9xFRDY55QckjOdIe?usp=sharing) is provided for running experiments.

---

## Experiments

- **Activation Frequency & Activation Strength**  
  The Colab notebook implements codemixing for experiments analyzing activation frequency and activation strength.  

- **Logit Lens Experiments**  
  For logit lens experiments, we used the dataset introduced in:  
  > Dumas et al. (2024). *Separating Tongue from Thought: Activation Patching Reveals Language-Agnostic Concept Representations in Transformers*  
  [[arXiv link]](https://arxiv.org/abs/2411.08745) | [Official repo](https://github.com/Butanium/llm-lang-agnostic)

  ```bibtex
  @misc{dumas2025separatingtonguethoughtactivation,
        title={Separating Tongue from Thought: Activation Patching Reveals Language-Agnostic Concept Representations in Transformers}, 
        author={Clément Dumas and Chris Wendler and Veniamin Veselovsky and Giovanni Monea and Robert West},
        year={2025},
        eprint={2411.08745},
        archivePrefix={arXiv},
        primaryClass={cs.CL},
        url={https://arxiv.org/abs/2411.08745}, 
  }
