import logging
import re
from ..eval import eval_metrics
from ..model import LLM
from ..load import load_data_prompt

regex = r"\d+"

def extract_score(output):
    output = output[0]
    if len(output) == 1:
        try:
            return int(output)
        except:
            logger = logging.getLogger(__name__)
            logger.info(f"Invalid output: {output}")
            return "not valid"
    elif len(output) > 1:
        numbers = re.findall(regex, output)
        if len(numbers) == 0:
            logger = logging.getLogger(__name__)
            logger.info(f"Invalid output: {output}")
            return "not valid"
        return int(numbers[0])
    # TODO: investigate more edge cases

def build_prompt(prompt, student_response):
    message = ""
    message += f"{prompt.replace('{{student_answer}}', student_response)} \n"
    return [{'role': 'user', 'content': message}]

def llm_inference(configs):
    dataset, prompts = load_data_prompt(configs)

    # Select corresponding subset
    dataset = dataset[configs['data']['split']]
    student_answers = dataset[configs['data']['input_column']].tolist()
    labels = dataset[configs['data']['label_column']].tolist()

    prompts = prompts["llm_inference"]
    if configs['prompt']['shot_mode'] == 'zero-shot':
        prompt = prompts["zero-shot"]
    elif configs['prompt']['shot_mode'] == 'few-shot':
        prompt = prompts["few-shot"]
    else:
        raise ValueError("Invalid prompt mode")
    
    # Initialize LLM
    llm = LLM(configs['llm'])

    outputs = []
    predicted_labels = []

    for each in student_answers:
        messages = build_prompt(prompt, each)
        output = llm.query(messages=messages)
        outputs.append(output)
        predicted_labels.append(extract_score(output))

    logger = logging.getLogger(__name__)    
    # filter not valid
    for i in range(len(predicted_labels)):
        if predicted_labels[i] == "not valid":
            # randomly get a label that not ground truth label
            ground_truth = labels[i]
            labels_set = list(set(labels))
            labels_set.remove(ground_truth)
            predicted_labels[i] = random.shuffle(labels_set)[0]
            message = f"Invalid output: index {i}, {outputs[i]}, replace with random label: {predicted_labels[i]}"
            logger.info(message)

    results = eval_metrics(labels, predicted_labels)

    message = f"----- model: {configs['llm']['model']} prompt_mode: {configs['prompt']['shot_mode']} -----"
    logger.info(message)
    message = "Results: Accuracy: {:.4f}, F1 Macro: {:.4f}, QWK: {:.4f}".format(results['accuracy'], results['f1_macro'], results['qwk'])
    logger.info(message)

    dataset['llm_output'] = outputs
    dataset['llm_pred'] = predicted_labels
    dataset.to_json(f"{configs['data']['saving_path']}{configs['llm']['model']}_{configs['prompt']['shot_mode']}_test_results.jsonl", orient='records', lines=True)
