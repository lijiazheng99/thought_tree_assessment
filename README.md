# Calibrating LLMs with Preference Optimization on Thought Trees for Generating Rationale in Science Question Scoring

![Current Version](https://img.shields.io/badge/version-v1.0-blue)
![GitHub contributors](https://img.shields.io/github/contributors/lijiazheng99/thought_tree_assessment)
![GitHub stars](https://img.shields.io/github/stars/lijiazheng99/thought_tree_assessment?style=social)
![GitHub forks](https://img.shields.io/github/forks/lijiazheng99/thought_tree_assessment?style=social)

This repository houses the implementation of the paper titled ["Calibrating LLMs with Preference Optimization on Thought Trees for Generating Rationale in Science Question Scoring,"](https://arxiv.org/abs/2406.19949) which has been accepted for presentation at EMNLP 2024 Findings.

## Abstract
Generating rationales that justify scoring decisions has been a promising way to facilitate explainability in automated scoring systems. However, existing methods do not match the accuracy of classifier-based methods. Plus, the generated rationales often contain hallucinated information. To address these issues, we propose a novel framework capable of generating more faithful rationales and, more importantly, matching performance with classifier-based black-box scoring systems. We first mimic the human assessment process by querying Large Language Models (LLMs) to generate a thought tree. We then summarise intermediate assessment decisions from each thought tree path for creating synthetic rationale data and rationale preference data. Finally, we utilise the generated synthetic data to calibrate LLMs through a two-step training process: supervised fine-tuning and preference optimization. Extensive experimental results demonstrate that our framework achieves a 38% assessment performance improvement in the QWK score compared to prior work while producing higher-quality rationales, as recognised by human evaluators and LLMs. Our work sheds light on the effectiveness of performing preference optimization using synthetic preference data obtained from thought tree paths.

## Open Source Contributions
We are thrilled to make our datasets and models accessible at all stages of our research. Explore our [collections](https://huggingface.co/collections/jiazhengli/mcts-with-preference-optimisation-670bdeaeada59c956f876092) and models via the following links:
- [**Stage 1: MCT Data**](https://huggingface.co/datasets/jiazhengli/Rationale_MCTS)
- [**Stage 2: Synthetic Rationale Data**](https://huggingface.co/datasets/jiazhengli/Synthetic_Rationale)
- [**Stage 3: Rationale to Score Model**](https://huggingface.co/jiazhengli/deberta-v3-large-Rationale-to-Score)
- [**Stage 3: Llama-3-8B SFT Model**](https://huggingface.co/jiazhengli/Meta-Llama-3-8B-QLoRA-Assessment-Rationale-sft)
- [**Stage 3: Llama-3-8B DPO Model**](https://huggingface.co/jiazhengli/Meta-Llama-3-8B-QLoRA-Assessment-Rationale-dpo)
- [**Stage 3: Mixtral-8x7B-Instruct-v0.1 SFT Model**](https://huggingface.co/jiazhengli/Mixtral-8x7B-Instruct-v0.1-QLoRA-Assessment-Rationale-sft)
- [**Stage 3: Mixtral-8x7B-Instruct-v0.1 DPO Model**](https://huggingface.co/jiazhengli/Mixtral-8x7B-Instruct-v0.1-QLoRA-Assessment-Rationale-dpo)

## Usage Instructions

### Environment Setup
```bash
conda env create -f environment.yml
```
### Stage 1: Imitate Human Assessment Process via Thought Trees

#### 1. Configure Thought Trees: 
Edit `configs/tot_query.yaml`
#### 2. Set your API keys for various services (Azure, OpenAI, Mistral, VLLM) to integrate LLM querying.
If you use Azure OpenAI api service: Add your api info in [here](https://github.com/lijiazheng99/thought_tree_assessment/blob/845b26fc323aa93cc4edf93ea262ac598b0abb66/tot_assessment/model.py#L40-L43).   
If you use OpenAI API service: Add your api key in [here](https://github.com/lijiazheng99/thought_tree_assessment/blob/845b26fc323aa93cc4edf93ea262ac598b0abb66/tot_assessment/model.py#L45).  
If you use Mistral API service: Add your api key in [here](https://github.com/lijiazheng99/thought_tree_assessment/blob/845b26fc323aa93cc4edf93ea262ac598b0abb66/tot_assessment/model.py#L50).  
If you use VLLM local api sever: Change your configuration in [here](https://github.com/lijiazheng99/thought_tree_assessment/blob/845b26fc323aa93cc4edf93ea262ac598b0abb66/tot_assessment/model.py#L51-L61).  
Using custom models: You probably need to change model list in [here](https://github.com/lijiazheng99/thought_tree_assessment/blob/845b26fc323aa93cc4edf93ea262ac598b0abb66/tot_assessment/model.py#L8-L32).  

#### 3. Generate Thought Trees
```
python query.py
```

### Stage 2: Summarise Thought Tree Paths as Rationales
#### 1. Configure Generation
Edit `configs/generation.yaml`
#### 2. Generate Batch Query File
```
python generate.py
```
We utilize [OpenAI’s batch API](https://platform.openai.com/docs/guides/batch) to generate synthetic data efficiently.

### Stage 3: Calibrate LLMs to Generate Rationales

We used [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)(Thanks!) to train our models. Please refer to our example training scripts/configs: [[train sft model](https://github.com/lijiazheng99/thought_tree_assessment/blob/main/configs/train_sft.sh)] [[train dpo model](https://github.com/lijiazheng99/thought_tree_assessment/blob/main/configs/lora_dpo.yaml)].


## Cite Our Work
If you find our method useful, please cite our paper as follows:
```bib
@misc{li2024calibratingllmspreferenceoptimization,
      title={Calibrating LLMs with Preference Optimization on Thought Trees for Generating Rationale in Science Question Scoring}, 
      author={Jiazheng Li and Hainiu Xu and Zhaoyue Sun and Yuxiang Zhou and David West and Cesare Aloisi and Yulan He},
      year={2024},
      eprint={2406.19949},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2406.19949}, 
}
```
