import logging
import argparse
from datetime import datetime
from tqdm import tqdm

from tot_assessment import LLM, tree_of_thought_assessment, data_generation, prompt_generation
from tot_assessment.load import load_prompt, load_config, load_tot
from tot_assessment.eval import eval_metrics, plot_confusion_matrix

import pandas as pd
from tokencost import calculate_prompt_cost
import decimal

def construct_prompt(prompt, student_answers):
    return prompt["generation_template"].replace("student_answer", student_answers)

def generation_loop(configs):
    # Load data and prompt
    data, original_data = load_tot(configs)
    data_name = configs['data']['tree_path'].split('/')[-1]
    data_name = '_'.join(data_name.split('_', 3)[:3])
    prompts = load_prompt(configs)

    verbose = False

    if configs['log']['enable']:
        logging.info(f"Configurations: {configs}")
    
    label_range = list(set(data[configs['data']['label_column']]))

    prepared_prompts = []
    full_record = []
    pos_count = []
    neg_count = []
    paired = []
    estimated_cost = decimal.Decimal(0.)
    for index, row in tqdm(data.iterrows()):
        prompts_generated = prompt_generation(configs, row, prompts)
        poss = 0
        negs = 0
        for idx, each in enumerate(prompts_generated):
            messages = [{"role": "user", "content": each['prompt']}]
            id = row['Id']
            pref = 0 if each['preference'] == 'better' else 1
            if each['preference'] == 'better':
                pos_count.append(1)
                poss += 1
            else:
                neg_count.append(1)
                negs += 1
            custom_id = f"{data_name}-{id}_{idx}-{pref}"
            prepared_prompts.append({"custom_id": f"{custom_id}", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-4-turbo-2024-04-09", "messages": messages}})
            full_record.append({"custom_id": f"{custom_id}", "preference": each['preference'], "score": each['score'], 'confidence': each['confidence'], "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-4-turbo-2024-04-09", "messages": messages}})
            estimated_cost += calculate_prompt_cost(each['prompt'], "gpt-4-turbo-2024-04-09")
        if poss > 0 and negs > 0:
            paired.append(1)

    print(f"Total: {len(prepared_prompts)} Paired: {sum(paired)} Pos: {sum(pos_count)} Neg: {sum(neg_count)} Dataset Size: {len(data)} Estimated cost: {estimated_cost}")
    prepared_prompts = pd.DataFrame(prepared_prompts)
    prepared_prompts.to_json(f"{configs['data']['saving_path']}/{data_name}_batched_prompts.jsonl", orient='records',lines=True)
    full_record = pd.DataFrame(full_record)
    full_record.to_json(f"{configs['data']['saving_path']}/{data_name}_full_records.jsonl", orient='records',lines=True)

def main():
    parser = argparse.ArgumentParser(description="Tree of Thought Data Generation")
    parser.add_argument("--config", type=str, default="./configs/generation.yaml", help="Path to the config file")
    args = parser.parse_args()
    # parser.add_argument("--index", type=int, help="Print specific index of the dataset.")

    config = load_config(args.config)
    if config['log']['enable']:
        now = datetime.now()
        timestamp = now.strftime("%m%d-%H%M")
        logging_name = f"./outputs/logs/generation.log"
        logging.basicConfig(filename=logging_name, level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')
    
    generation_loop(config)

if __name__ == "__main__":
    main()
