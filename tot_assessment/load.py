import json
import os
import pandas as pd
import yaml
from copy import deepcopy

from .tree import importer, WNode

"""
Load yaml config
"""
def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Validate config
    if 'tot_query' in config_path:
        if config["prompt"]["shot_mode"] not in ["few-shot", "zero-shot"]:
            raise ValueError("Invalid shot mode")
        if config["prompt"]["history"] not in ["no_history", "wo_demo_history", "full_history"]:
            raise ValueError("Invalid history mode")
        if config["prompt"]["sum_score"] not in ["by_llm", "by_python"]:
            raise ValueError("Invalid sum score mode")
        if config["prompt"]["shot_mode"] == "zero-shot" and config["prompt"]["history"] == "full_history":
            raise ValueError("Zero-shot mode does not support full history, because it does not have demonstrations")
    elif 'generation' in config_path:
        pass

    return config

"""
Load dataset and prompt
"""

def load_data_prompt(config):
    dataset_name = config["data"]["name"]
    dataset_path = config["data"]["path"]
    input_col = config["data"]["input_column"]
    if "asap" in dataset_name:
        path = dataset_path + "asap/" + dataset_name + "/"
    files = os.listdir(path)
    dataset = {"train":None, "test":None, "validation":None}
    for file in files:
        if "train.jsonl" == file:
            dataset["train"] = pd.read_json(path+file, lines=True, encoding='utf-8')
        elif "test.jsonl" == file:
            dataset["test"] = pd.read_json(path+file, lines=True, encoding='utf-8')
        elif "dev.jsonl" == file or "validation.jsonl" == file:
            dataset["validation"] = pd.read_json(path+file, lines=True, encoding='utf-8')
    
    # Perform data cleansing
    for each in dataset.keys():
        if dataset[each] is not None:
            dataset[each][input_col] = [text.encode('utf-8').decode('unicode-escape') for text in dataset[each][input_col]]

    with open(path+"prompt.json", 'r') as file:
        prompts = json.load(file)
    
    return dataset, prompts

def load_prompt(config):
    dataset_name = config["data"]["name"]
    dataset_path = config["data"]["path"]
    input_col = config["data"]["input_column"]
    if "asap" in dataset_name:
        path = dataset_path + "asap/" + dataset_name + "/"
    with open(path+"prompt.json", 'r') as file:
        prompts = json.load(file)
    
    return prompts

"""
Load dataset and tree of thought result
"""

def load_tot(config):
    tot_path = config["data"]["tree_path"]
    check = os.path.exists(tot_path)
    if not check:
        raise FileNotFoundError(f"Tree of thought result not found: {tot_path}")
    data = pd.read_json(tot_path, lines=True, encoding='utf-8')
    original_data = deepcopy(data)
    for index, row in data.iterrows():
        data.at[index, "key_element_tree"] = importer(row["key_element_tree"])
        data.at[index, "assessment_tree"] = importer(row["assessment_tree"])
    return data, original_data

def load_tot_by_path(tot_path):
    check = os.path.exists(tot_path)
    if not check:
        raise FileNotFoundError(f"Tree of thought result not found: {tot_path}")
    data = pd.read_json(tot_path, lines=True, encoding='utf-8')
    original_data = deepcopy(data)
    for index, row in data.iterrows():
        data.at[index, "key_element_tree"] = importer(row["key_element_tree"])
        data.at[index, "assessment_tree"] = importer(row["assessment_tree"])
    return data, original_data

def load_jsonl_data(data_path):
    return pd.read_json(data_path, lines=True, encoding='utf-8')
