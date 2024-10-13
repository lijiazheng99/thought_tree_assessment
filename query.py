import logging
logging_name = f"./outputs/logs/all.log"
logging.basicConfig(filename=logging_name, level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')

import argparse
from datetime import datetime
from tqdm import tqdm
import pandas as pd

from tot_assessment.load import load_data_prompt, load_config
from tot_assessment import LLM, tree_of_thought_assessment, get_max, openai_models
from tot_assessment.eval import eval_metrics, plot_confusion_matrix

def query_loop(configs, specfic_index):
    dataset, prompts = load_data_prompt(configs)

    # Select corresponding subset
    dataset = dataset[configs['data']['split']]
 
    print(dataset.head())
    student_answers = dataset[configs['data']['input_column']].tolist()
    labels = dataset[configs['data']['label_column']].tolist()

    if specfic_index is not None:
        specfic_index = int(specfic_index)
        student_answers = [student_answers[specfic_index]]
        labels = [labels[specfic_index]]
        print(f"Index: {specfic_index}, Student Answer: {student_answers}, Label: {labels}")
        verbose = True
    else:
        verbose = False

    # Initialize LLM
    llm = LLM(configs['llm'])
    start_time = datetime.now()
    outputs = llm.query(messages=[{'role': 'user', 'content': "6+7="}], temperature=0.7, candidates=1, max_tokens=4)
    stop_time = datetime.now()
    total_time = stop_time - start_time

    if 'rubric_formula' in prompts:
        exec(str(prompts["rubric_formula"]), globals())
    
    if configs['log']['enable']:
        logger = logging.getLogger(__name__)
        logger.info(f"Configurations: {configs}")
        logger.info(f"LLM intialized: single query time: {total_time}.")
    key_element_trees = []
    assessment_trees = []
    all_pred_labels = []

    python_pred = []
    llm_pred = []
    matched = 0

    print(f"Total number of student answers: {len(student_answers)}; Started querying...")

    # Query Loop
    for index, student_answer in enumerate(tqdm(student_answers)):
        # Text preprocessing
        student_answer = student_answer.replace("\n", " ").replace("\r", " ")
        label = labels[index]
        # Tot Assessment
        try:

            key_element_tree, assessment_tree, pred_labels = \
            tree_of_thought_assessment(configs=configs, prompts=prompts, student_answer=student_answer, llm=llm, iindex=index, sum_score=sum_score,verbose=verbose)

            key_element_trees.append(key_element_tree)
            assessment_trees.append(assessment_tree)
            all_pred_labels.append([pred_labels])

            # Find matched labels
            python_pred.append(get_max(pred_labels['Python']))
            llm_pred.append(get_max(pred_labels['LLM']))
            if label in [list(each.keys())[0] for each in pred_labels['Python']] or label in [list(each.keys())[0] for each in pred_labels['LLM']]:
                matched += 1
        except Exception as e:
            if configs['log']['enable']:
                logger.error(f"Index: {index}, Error: {e}")
            key_element_trees.append("")
            assessment_trees.append("")
            all_pred_labels.append("")
            python_pred.append("")
            llm_pred.append("")
    
    if configs['llm']['model'] in openai_models:
        report_cost = llm.report_cost()
        if configs['log']['enable']:
            logger.info(f"Cost Analysis: Prompt cost: {report_cost['prompt_cost']}, Completion cost: {report_cost['completion_cost']}, Total cost: {report_cost['total_cost']}")
    
    if specfic_index is not None:
        pass
    else:
        # Save results
        dataset['key_element_tree'] = key_element_trees
        dataset['assessment_tree'] = assessment_trees
        dataset['pred_labels'] = all_pred_labels
        dataset['python_pred'] = python_pred
        dataset['llm_pred'] = llm_pred
        now = datetime.now()
        timestamp = now.strftime("%m%d-%H%M")
        if '/' in configs['llm']['model']:
            configs['llm']['model'] = configs['llm']['model'].replace('/', '-')
        dataset.to_json(f"{configs['data']['saving_path']}/{configs['data']['name']}_{configs['data']['split']}_{configs['llm']['model']}_{timestamp}.jsonl", orient='records', lines=True)

        # Evaluation
        python_results = eval_metrics(y_true=labels, y_pred=python_pred)
        llm_results = eval_metrics(y_true=labels, y_pred=llm_pred)
        if configs['log']['enable']:
            logger.info(f"Python results: {python_results}")
            logger.info(f"LLM results: {llm_results}")
            logger.info(f"Matched: {matched}/{len(labels)}")

        # Plot confusion matrix
        ax = plot_confusion_matrix(labels, python_pred, normalize=False, title="Python Confusion Matrix")
        ax.figure.savefig(f"./outputs/confusion_matrix/{configs['data']['split']}_{timestamp}_python.png")
        ax = plot_confusion_matrix(labels, llm_pred, normalize=False, title="LLM Confusion Matrix")
        ax.figure.savefig(f"./outputs/confusion_matrix/{configs['data']['split']}_{timestamp}_llm.png")

def eval_only(eval_path):
    test_data = pd.read_json(eval_path, orient='records', lines=True)
    print(test_data.head())

    print(test_data['pred_labels'][0])

    labels = test_data["Mark"].tolist()
    python_pred = test_data['python_pred'].tolist()
    llm_pred = test_data['llm_pred'].tolist()

    # exclude empty predictions, in parallel with labels
    for index, each in enumerate(python_pred):
        if each == "" or llm_pred[index] == "":
            labels.pop(index)
            python_pred.pop(index)
            llm_pred.pop(index)

    python_results = eval_metrics(y_true=labels, y_pred=python_pred)
    llm_results = eval_metrics(y_true=labels, y_pred=llm_pred)
    print(f"Python results: {python_results}")
    print(f"LLM results: {llm_results}")
    all_pred_labels = test_data['pred_labels'].tolist()
    matched = 0
    for label, pred_labels in zip(labels, all_pred_labels):
        all_labels = []
        print(pred_labels)
        for each in pred_labels[0]['Python']:
            all_labels.append(int(list(each.keys())[0]))
        for each in pred_labels[0]['LLM']:
            all_labels.append(int(list(each.keys())[0]))
        if label in all_labels:
            matched += 1
    print(f"Matched: {matched}/{len(labels)}")

    # Plot confusion matrix
    ax = plot_confusion_matrix(labels, python_pred, normalize=False, title="Python Confusion Matrix")
    ax = plot_confusion_matrix(labels, llm_pred, normalize=False, title="LLM Confusion Matrix")

def main():
    parser = argparse.ArgumentParser(description="Tree of Thought Assessment")
    parser.add_argument("--config", type=str, default="./configs/tot_query.yaml", help="Path to the config file")
    parser.add_argument("--index", type=int, help="Print specific index of the dataset.")
    parser.add_argument("--eval", type=str, default=None, help="Evaluate the results.")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.eval is not None:
        print('Evaluating ...')
        eval_only(args.eval)
    else:
        print('Querying ...')
        query_loop(config, args.index)

if __name__ == "__main__":
    main()
