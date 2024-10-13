
import argparse
import logging
from datetime import datetime
from tot_assessment.load import load_config
from tot_assessment.baseline import train_classifier, llm_inference, train_ablation_pref_classifier

def main():
    parser = argparse.ArgumentParser(description="Run Baseline Tasks")
    parser.add_argument("--task", type=str, default="", help="Task to run")
    parser.add_argument("--config", type=str, default="", help="Path to the config file")
    args = parser.parse_args()

    if args.task == "":
        raise ValueError("Please specify the baseline task")
    print(f"Running {args.task} task")

    if args.task == "classification":
        if args.config == "":
            args.config = "./configs/classifier.yaml"
    elif args.task == "query_llm":
        if args.config == "":
            args.config = "./configs/llm_inference.yaml"
    elif args.task == "ablation_pref":
        if args.config == "":
            args.config = "./configs/classifier_pref.yaml"
    else:
        raise ValueError("Invalid baseline task")

    config = load_config(args.config)
    if config['log']['enable']:
        logging_name = f"{config['log']['path']}"
        logging.basicConfig(filename=logging_name, level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')

    if args.task == "classification":
        train_classifier(config)
    elif args.task == "query_llm":
        llm_inference(config)
    elif args.task == "ablation_pref":
        train_ablation_pref_classifier(config)

if __name__ == "__main__":
    main()