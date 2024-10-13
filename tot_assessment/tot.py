import logging
from datetime import datetime
from random import shuffle
from copy import deepcopy
from .prompt import prompt_selector, group_decisions, group_scores
from .tree import WNode, DotExporter, JsonExporter, print_tree, find_paths, save_graph
import random

logger = logging.getLogger(__name__)

# DFS tree of thought
def key_elements_search(configs, history_stack, llm, key_elements, key_elements_demos, tree_recorder):

    current_key_element = key_elements.pop(0)
    history_stack.append({"prompt_type": "key_element", "prompt": current_key_element})
    history_stack.append({"prompt_type": "key_element_demo", "prompt": key_elements_demos.pop(0)})

    constructed_messages = prompt_selector(configs["prompt"]["history"], "key_element", history_stack)
    # Query LLM
    decisions = llm.query(messages=constructed_messages, max_tokens=4)

    votes = group_decisions(decisions)
    decision_list = list(votes.keys())

    # Generate rationales by each key element
    rationales = []
    for each in decision_list:
        rationale_messages = constructed_messages + [{'role': 'assistant', 'content': f"{each}"},{'role': 'user', 'content': "Please generate a rationale that justifies your above decision. Quote ** part of this student answer ** (e.g. just a few words) that answers this key element with \"...\". Do not include any other additional information that are not relevant to this decision."}]
        rationale = llm.query(messages=rationale_messages, temperature=0.7, candidates=1, max_tokens=120)[0]
        rationale = rationale.replace('\n','')
        rationales.append(rationale)

    if len(decision_list) > 1 and len(key_elements) > 0:
        child0 = WNode(decision_list[0], parent=tree_recorder, weight=votes[decision_list[0]], rationale=rationales[0])
        child1 = WNode(decision_list[1], parent=tree_recorder, weight=votes[decision_list[1]], rationale=rationales[1])
        return key_elements_search(configs, history_stack + [{"prompt_type": "model_output", "prompt": "Yes"}], llm, deepcopy(key_elements), deepcopy(key_elements_demos), child0)\
            + key_elements_search(configs, history_stack + [{"prompt_type": "model_output", "prompt": "No"}], llm, deepcopy(key_elements), deepcopy(key_elements_demos), child1)
    elif len(decision_list) > 1 and len(key_elements) == 0:
        child0 = WNode(decision_list[0], parent=tree_recorder, weight=votes[decision_list[0]], rationale=rationales[0])
        child1 = WNode(decision_list[1], parent=tree_recorder, weight=votes[decision_list[1]], rationale=rationales[1])
        return ""
    elif len(decision_list) == 1 and len(key_elements) > 0:
        child = WNode(decision_list[0], parent=tree_recorder, weight=votes[decision_list[0]], rationale=rationales[0])
        return key_elements_search(configs, history_stack + [{"prompt_type": "model_output", "prompt": decision_list[0]}], llm, deepcopy(key_elements), deepcopy(key_elements_demos), child)
    elif len(decision_list) == 1 and len(key_elements) == 0:
        WNode(decision_list[0], parent=tree_recorder, weight=votes[decision_list[0]], rationale=rationales[0])
        return ""
    else:
        print(f"Invalid decision list: Number of decisions: {len(decision_list)}, Number of key elements: {len(key_elements)}.")
        print(f"Decision list: {decision_list}")
        logger.error(f"Invalid decision list: Number of decisions: {len(decision_list)}, Number of key elements: {len(key_elements)}.")
        raise ValueError("Invalid decision list")

def tree_of_thought_assessment(configs, prompts, student_answer, llm, iindex, sum_score, verbose):
    # Sort out prompts
    prompts = deepcopy(prompts)

    # Control order of key elements
    if configs["prompt"]["order"] == 'shuffle':
        combine = [[key_element,demo] for key_element,demo in zip (prompts["key_elements"], prompts["demonstations"]["key_elements"])]
        shuffle(combine)
        prompts["key_elements"] = [each[0] for each in combine]
        prompts["demonstations"]["key_elements"] = [each[1] for each in combine]
    elif configs["prompt"]["order"] == 'reverse':
        prompts["key_elements"].reverse()
        prompts["demonstations"]["key_elements"].reverse()
    elif configs["prompt"]["order"] == 'default':
        pass
    else:
        potential_order = [int(each) for each in configs["prompt"]["order"].split(",")]
        if len(potential_order) == len(prompts["key_elements"]):
            prompts["key_elements"] = [prompts["key_elements"][each] for each in potential_order]
            prompts["demonstations"]["key_elements"] = [prompts["demonstations"]["key_elements"][each] for each in potential_order]
        else:
            raise ValueError("Invalid order")
    
    if configs["prompt"]["shot_mode"] == "zero-shot":
        prompts["demonstations"]["key_elements"] = ["" for _ in range(len(prompts["key_elements"]))]
        prompts["demonstations"]["rubric"] = ""
    
    # Replace student answer
    prompts["key_elements"] = [key_element.replace("{{student_answer}}", student_answer) for key_element in prompts["key_elements"]]
    prompts["rubric"] = prompts["rubric"].replace("{{student_answer}}", student_answer)

    # Matching key elements
    history_stack = []
    history_stack.append({"prompt_type": "question", "prompt": prompts["question"]})

    key_elements = deepcopy(prompts["key_elements"])
    key_elements_demos = deepcopy(prompts["demonstations"]["key_elements"])

    # Create lead tree recorder
    tree_recorder = WNode(student_answer)

    # Depth first search
    key_elements_search(configs, history_stack, llm, key_elements, key_elements_demos, tree_recorder)

    # Applying rubric
    paths = find_paths(tree_recorder)

    # Second Layer: Perform Rubric Matching
    # Build layerwise tree
    assessment_tree = WNode(student_answer)
    pred_labels = {'Python':[], 'LLM':[]}  

    for each_path in paths:
        node_name = []
        rationales = ""
        node_weight = 1
        # Build prompt
        history_stack_cp = deepcopy(history_stack)
        for key_element, demo, each_decision in zip(prompts["key_elements"], prompts["demonstations"]["key_elements"], each_path):
            decision = list(each_decision.keys())[0]
            rationales += f"{each_decision['rationale']}\n"
            weight = each_decision[decision]
            history_stack_cp.append({"prompt_type": "key_element", "prompt": key_element})
            history_stack_cp.append({"prompt_type": "key_element_demo", "prompt": demo})
            history_stack_cp.append({"prompt_type": "model_output", "prompt": decision})
            node_name += [f"{decision}"]
            node_weight *= weight

        # Add node
        key_element_tree = WNode("; ".join(node_name), parent=assessment_tree, weight=node_weight, rationale=rationales)

        # Calculate rubric
        if configs["prompt"]["sum_score"] == "by_llm":
            rubric = prompts["rubric"].replace("{{demonstation}}", prompts["demonstations"]["rubric"])
            history_stack_cp.append({"prompt_type": "rubric", "prompt": rubric})
            messages = prompt_selector(configs["prompt"]["history"], "rubric", history_stack_cp)

            # Query LLM
            decisions = llm.query(messages=messages, max_tokens=10)
            votes = group_scores(decisions)
            decision_list = list(votes.keys())
            for each in decision_list:
                pred_labels['LLM'].append({each: votes[each]*node_weight})
                WNode(f"LLM: {each}", parent=key_element_tree, weight=votes[each])

        sum_by_python = sum_score(node_name)
        pred_labels['Python'].append({sum_by_python: 1.0*node_weight})
        WNode(f"Python: {sum_by_python}", parent=key_element_tree, weight=1.0)
    
    # Print tree
    # print_tree(assessment_tree)
    if (configs["prompt"]["save_tree"] and len(pred_labels['Python']) > configs["prompt"]["save_tree_threshold"]) or verbose:
        try:
            now = datetime.now()
            timestamp = now.strftime("%m%d-%H%M")
            save_graph(tree_recorder, f"{configs['prompt']['saving_path']}/{configs['data']['name']}_{configs['data']['split']}_{iindex}_{timestamp}_ke.png")
            save_graph(assessment_tree, f"{configs['prompt']['saving_path']}/{configs['data']['name']}_{configs['data']['split']}_{iindex}_{timestamp}_all.png")
        except Exception as e:
            logger.error(f"Error in saving tree: {configs['data']['name']}_{configs['data']['split']}_{iindex}")
            logger.error(e)
    
    # Export tree in json
    key_element_tree = JsonExporter(indent=2).export(assessment_tree)
    assessment_tree = JsonExporter(indent=2).export(assessment_tree)
    return key_element_tree, assessment_tree, pred_labels
