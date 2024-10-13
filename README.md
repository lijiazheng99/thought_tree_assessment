# Calibrating LLMs with Preference Optimization on Thought Trees for Generating Rationale in Science Question Scoring

![Current Version](https://img.shields.io/badge/version-v1.0-blue)
![GitHub contributors](https://img.shields.io/github/contributors/lijiazheng99/thought_tree_assessment)
![GitHub stars](https://img.shields.io/github/stars/lijiazheng99/thought_tree_assessment?style=social)
![GitHub forks](https://img.shields.io/github/forks/lijiazheng99/thought_tree_assessment?style=social)

This repo contains the code for the paper [Calibrating LLMs with Preference Optimization on Thought Trees for Generating Rationale in Science Question Scoring](https://arxiv.org/abs/2406.19949), which has been accepted to EMNLP 2024 Findings.

## Abstract
Generating rationales that justify scoring decisions has been a promising way to facilitate explainability in automated scoring systems. However, existing methods do not match the accuracy of classifier-based methods. Plus, the generated rationales often contain hallucinated information. To address these issues, we propose a novel framework capable of generating more faithful rationales and, more importantly, matching performance with classifier-based black-box scoring systems. We first mimic the human assessment process by querying Large Language Models (LLMs) to generate a thought tree. We then summarise intermediate assessment decisions from each thought tree path for creating synthetic rationale data and rationale preference data. Finally, we utilise the generated synthetic data to calibrate LLMs through a two-step training process: supervised fine-tuning and preference optimization. Extensive experimental results demonstrate that our framework achieves a 38% assessment performance improvement in the QWK score compared to prior work while producing higher-quality rationales, as recognised by human evaluators and LLMs. Our work sheds light on the effectiveness of performing preference optimization using synthetic preference data obtained from thought tree paths.

## How to use this code

### install environments
```
conda env create -f environment.yml
```
### Stage 1: Imitate Human Assessment Process via Thought Trees

#### 1. Edit `configs/tot_query.yaml`
#### 2. Put api keys (optional if using hf)
If you use Azure OpenAI api service: Add your api info in [here](https://github.com/lijiazheng99/thought_tree_assessment/blob/845b26fc323aa93cc4edf93ea262ac598b0abb66/tot_assessment/model.py#L40-L43).   
If you use OpenAI API service: Add your api key in [here](https://github.com/lijiazheng99/thought_tree_assessment/blob/845b26fc323aa93cc4edf93ea262ac598b0abb66/tot_assessment/model.py#L45).  
If you use Mistral API service: Add your api key in [here](https://github.com/lijiazheng99/thought_tree_assessment/blob/845b26fc323aa93cc4edf93ea262ac598b0abb66/tot_assessment/model.py#L50).  
If you use VLLM local api sever: Change your configuration in [here](https://github.com/lijiazheng99/thought_tree_assessment/blob/845b26fc323aa93cc4edf93ea262ac598b0abb66/tot_assessment/model.py#L51-L61).  
Using custom models: You probably need to change model list in [here](https://github.com/lijiazheng99/thought_tree_assessment/blob/845b26fc323aa93cc4edf93ea262ac598b0abb66/tot_assessment/model.py#L8-L32).  

#### 3. Query LLMs to generate thought trees
```
python query.py
```
#### Example Tree Data
Coming Soon!!!

### Stage 2: Summarise Thought Tree Paths as Rationales
#### 1. Edit `configs/tot_query.yaml`
#### 2. Generate batch query file
```
python generate.py
```

To save money, we used [openai batch api](https://platform.openai.com/docs/guides/batch) to process synthethic data generation. Please use the stored file to perform batch query.

#### Example Synthethic Rationales
Coming Soon!!!

### Stage 3: Calibrate LLMs to Generate Rationales

We used [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)(Thanks!) to train our models. Please refer to our example training scripts/configs: [[train sft model](https://github.com/lijiazheng99/thought_tree_assessment/blob/main/configs/train_sft.sh)] [[train dpo model](https://github.com/lijiazheng99/thought_tree_assessment/blob/main/configs/lora_dpo.yaml)].

#### Example Models

Coming soon!!!

## Cite our work
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
