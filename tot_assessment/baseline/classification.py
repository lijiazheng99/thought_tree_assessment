import logging
import os
from ..load import load_data_prompt, load_jsonl_data
from ..eval.evaluation import eval_metrics, compute_metrics
import torch
from torch.utils.data import Dataset
from torch.nn.functional import softmax
import numpy as np
import pandas as pd
from datetime import datetime
from copy import deepcopy
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer, 
    EarlyStoppingCallback
)

def add_prompt(dataset, prompt, input_col):
    for each in dataset.keys():
        dataset[each][input_col] = [f"{prompt} \nstudent answer: {each}" for each in dataset[each][input_col]]
    return dataset

def random_data_split(dataset, random_state):
    shuffled_dataset = {'train': None, 'validation': None, 'test': None}
    for each in dataset.keys():
        if each != 'test':
            shuffled_dataset[each] = dataset[each].sample(frac=1, random_state=random_state).reset_index(drop=True)
        else:
            shuffled_dataset[each] = dataset[each]
    return shuffled_dataset

class StudentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        self.num_labels = len(set(self.labels))

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def tokenize_dataset(dataset, input_col, output_col, tokenizer):

    train_encodings = tokenizer(dataset['train'][input_col].values.tolist(), padding='max_length', truncation=True, max_length=512)
    val_encodings = tokenizer(dataset['validation'][input_col].values.tolist(), padding='max_length', truncation=True, max_length=512)
    test_encodings = tokenizer(dataset['test'][input_col].values.tolist(), padding='max_length', truncation=True, max_length=512)
    
    train_dataset = StudentDataset(train_encodings, dataset['train'][output_col].values.tolist())
    val_dataset = StudentDataset(val_encodings, dataset['validation'][output_col].values.tolist())
    test_dataset = StudentDataset(test_encodings, dataset['test'][output_col].values.tolist())

    return train_dataset, val_dataset, test_dataset

def tokenize_dataset_ablation_pref(dataset, input_cols, output_col, tokenizer):

    if len(input_cols) > 1:
        train_encodings = tokenizer(dataset['train'][input_cols[0]].values.tolist(), dataset['train'][input_cols[1]].values.tolist(), padding='max_length', truncation=True, max_length=512)
        val_encodings = tokenizer(dataset['validation'][input_cols[0]].values.tolist(), dataset['validation'][input_cols[1]].values.tolist(), padding='max_length', truncation=True, max_length=512)
    else:
        train_encodings = tokenizer(dataset['train'][input_cols[0]].values.tolist(), padding='max_length', truncation=True, max_length=256)
        val_encodings = tokenizer(dataset['validation'][input_cols[0]].values.tolist(), padding='max_length', truncation=True, max_length=256)
    
    train_dataset = StudentDataset(train_encodings, dataset['train'][output_col].values.tolist())
    val_dataset = StudentDataset(val_encodings, dataset['validation'][output_col].values.tolist())

    return train_dataset, val_dataset

def train(dataset, config):
    input_col = config['data']['input_column']
    label_col = config['data']['label_column']

    base_model = config['training_args']['base_model']
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    train_dataset, val_dataset, test_dataset = tokenize_dataset(dataset, input_col, label_col, tokenizer)

    num_labels = len(set(dataset['train'][label_col]))
    
    if torch.cuda.is_available():
        device = "cuda"
    
    model = AutoModelForSequenceClassification.from_pretrained(base_model, ignore_mismatched_sizes=True, num_labels=num_labels)

    output_dir = config['training_args']['output_dir']
    if '/' in base_model:
        base_model = base_model.split('/')[-1]
    now = datetime.now()
    timestamp = now.strftime("%m%d-%H%M")
    run_name = f"{config['data']['name']}_{base_model}_{timestamp}"
    output_dir = f"{output_dir}/{run_name}"

    training_args = TrainingArguments(
        output_dir=output_dir,
        run_name=run_name,
        report_to="none",
        num_train_epochs=config['training_args']['num_train_epochs'],
        per_device_train_batch_size=config['training_args']['per_device_train_batch_size'],
        per_device_eval_batch_size=config['training_args']['per_device_eval_batch_size'],    
        warmup_steps=100,                 # number of warmup steps for learning rate scheduler
        weight_decay=0.1,                # strength of weight decay
        logging_dir='./logs',             # directory for storing logs
        logging_steps=100,                 # how often to print logs
        learning_rate=float(config['training_args']['learning_rate']),  # learning rate
        adam_epsilon=1e-8,                # epsilon for Adam optimizer
        save_total_limit=3,               # limit the total amount of checkpoints. Deletes the older checkpoints.
        evaluation_strategy="epoch",     # evaluation strategy to adopt during training
        save_strategy="epoch",
        load_best_model_at_end=True,      # load the best model when finished training
        metric_for_best_model=config['training_args']['metric_for_best_model'], # use accuracy to evaluate the best model
        greater_is_better=True,                     # Define metric direction for best model
        dataloader_drop_last=False                  # Do not drop the last incomplete batch
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,  # Add your metrics function here if needed
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    trainer.train()

    results = trainer.evaluate()
    logger = logging.getLogger(__name__)
    log_msg = f"{run_name}: Accuracy = {results['eval_accuracy']:.4f}, Macro F1 = {results['eval_f1_macro']:.4f}, QWK = {results['eval_qwk']:.4f}"
    logger.info(log_msg)

    # Save validation results
    pred = trainer.predict(test_dataset=test_dataset)
    if 't5' in base_model:
        result_tensor = torch.from_numpy(pred.predictions[0])
    else:
        result_tensor = torch.from_numpy(pred.predictions)
    
    y_pred = torch.argmax(softmax(result_tensor, dim=1), dim=1).numpy()
    test_result = eval_metrics(dataset['test'][label_col], y_pred)

    log_msg = f"{run_name}: Test Accuracy = {test_result['accuracy']:.4f}, Macro F1 = {test_result['f1_macro']:.4f}, QWK = {test_result['qwk']:.4f}"
    logger.info(log_msg)
    return test_result, model, y_pred

def train_classifier(config):
    # Load data and prompt
    dataset, prompts = load_data_prompt(config)
    test_copy = deepcopy(dataset['test'])

    # Add prompt
    if config['data']['with_prompt']:
        selected_prompts = config['data']['selected_prompts']
        prompt = ""
        for each in selected_prompts:
            prompt += f"{prompts[each]} \n"
        dataset = add_prompt(dataset, prompt, config['data']['input_column'])

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    random_seeds = config['random_seeds']
    results = {"accuracy": [], "f1_macro": [], "qwk": []}
    best_model = None
    best_pred = None
    for each in random_seeds:
        rand_dataset = random_data_split(dataset, each)
        result, model, y_pred = train(rand_dataset, config)
        for key in result.keys():
            results[key].append(result[key])
        if result['qwk'] == max(results['qwk']):
            best_model = model.to('cpu')
            best_pred = y_pred
    
    logger = logging.getLogger(__name__)
    message = "Mean Results: "
    for key in results.keys():
        message += f"{key}: {np.mean(results[key]):.4f} ± {np.std(results[key]):.4f}, "
    logger.info(message)

    output_dir = config['training_args']['output_dir']
    base_model = config['training_args']['base_model']
    if '/' in base_model:
        base_model = base_model.split('/')[-1]
    run_name = f"{config['data']['name']}_{base_model}_best"
    best_model.save_pretrained(f"{output_dir}/{run_name}/")
    test_copy['predicted'] = best_pred
    # save to jsonl
    test_copy.to_json(f"{config['data']['saving_path']}{run_name}_{config['data']['with_prompt']}_test_results.jsonl", orient='records', lines=True)

    logging.info(f"Best model saved to {output_dir}{run_name}/")


def train_ablation_pref_classifier(config):
    # Load data and prompt
    dataset_path = config['data']['path']
    print(f"{dataset_path}train.jsonl")
    train = pd.read_json(f"{dataset_path}train.jsonl", lines=True)
    #shuffle the data
    train = train.sample(frac=1).reset_index(drop=True)
    validation = pd.read_json(f"{dataset_path}validation.jsonl", lines=True, encoding='utf-8')
    validation = validation.sample(frac=1).reset_index(drop=True)
    dataset = {"train": train, "validation": validation}

    input_col = config['data']['input_column']
    label_col = config['data']['label_column']

    base_model = config['training_args']['base_model']
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    train_dataset, val_dataset = tokenize_dataset_ablation_pref(dataset, input_col, label_col, tokenizer)

    num_labels = len(set(dataset['train'][label_col]))

    # os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    
    # if torch.cuda.is_available():
        # device = "cuda"
    
    model = AutoModelForSequenceClassification.from_pretrained(base_model, ignore_mismatched_sizes=True, num_labels=num_labels)

    output_dir = config['training_args']['output_dir']
    if '/' in base_model:
        base_model = base_model.split('/')[-1]
    now = datetime.now()
    timestamp = now.strftime("%m%d-%H%M")
    run_name = f"ablation_preference_{base_model}_{timestamp}"
    output_dir = f"{output_dir}/{run_name}"

    print(tokenizer.decode(train_dataset.encodings[0].ids))

    training_args = TrainingArguments(
        output_dir=output_dir,
        run_name=run_name,
        report_to="none",
        num_train_epochs=config['training_args']['num_train_epochs'],
        per_device_train_batch_size=config['training_args']['per_device_train_batch_size'],
        per_device_eval_batch_size=config['training_args']['per_device_eval_batch_size'],    
        warmup_steps=100,                 # number of warmup steps for learning rate scheduler
        weight_decay=0.1,                # strength of weight decay
        logging_dir='./logs',             # directory for storing logs
        logging_steps=100,                 # how often to print logs
        learning_rate=float(config['training_args']['learning_rate']),  # learning rate
        adam_epsilon=1e-8,                # epsilon for Adam optimizer
        save_total_limit=3,               # limit the total amount of checkpoints. Deletes the older checkpoints.
        evaluation_strategy="epoch",     # evaluation strategy to adopt during training
        save_strategy="epoch",
        load_best_model_at_end=True,      # load the best model when finished training
        metric_for_best_model=config['training_args']['metric_for_best_model'], # use accuracy to evaluate the best model
        greater_is_better=True,                     # Define metric direction for best model
        dataloader_drop_last=False                  # Do not drop the last incomplete batch
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,  # Add your metrics function here if needed
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    trainer.train()

    results = trainer.evaluate()
    logger = logging.getLogger(__name__)
    log_msg = f"{run_name}: Accuracy = {results['eval_accuracy']:.4f}, Macro F1 = {results['eval_f1_macro']:.4f}, QWK = {results['eval_qwk']:.4f}"
    logger.info(log_msg)