import logging
from .tree import print_tree, find_paths, RenderTree
from random import shuffle

def format_generated(better_choice, worse_choices):
    if len(better_choice) + len(worse_choices) == 0:
        return {'contain': False, 'better_choice': None, 'worse_choice': None}
    return {'contain': True, 'better_choice': better_choice, 'worse_choice': worse_choices}

def llm_selection(llm, good_choices, student_answer):
    messages = [{'role': 'system', 'content': "You are an assessment rationale selector. Please select the best assessment rationale that describe this student's answer, by tell me the number of the rationale you think is the best: "}]
    content = f"Here is the student's answer: \"{student_answer}\". Please select the best assessment rationale for this student's answer: \n"
    for idx, each in enumerate(good_choices):
        content += f"{idx+1}. {each}\n"
    content += "Please tell me the number only: "
    messages.append({'role': 'user', 'content': content})
    output = llm.query(messages=messages, temperature=0.7, candidates=1)
    number = int(output[0])
    return good_choices[number-1]

def template_rationale(student_answer, key_elements, rubric, score):
    prompt = "Here is an assessment for the student's answer:\n"
    prompt += f"\"{student_answer}\".\n\n"
    prompt += "Assessment rationale: "
    rationale = ""
    if len(key_elements) > 0:
        prompt += f"The student's answer matches the following {len(key_elements)} key elements: "
        rationale += f"The student's answer matches the following {len(key_elements)} key elements: "
        for each in key_elements:
            prompt += f"\"{each}\" \n"
            rationale += f"\"{each}\" \n"
    else:
        prompt += "The student's answer does not contain any key elements. \n"
        rationale += "The student's answer does not contain any key elements. \n"
    prompt += f"According to the rubric , \"{rubric}\", the student's answer should get a score of {score}.\n"
    rationale += f"According to the rubric , \"{rubric}\", the student's answer should get a score of {score}.\n"
    return prompt, rationale

def llm_rewrite(templated_rationale, llm):
    messages = [{'role': 'system', 'content': "You are an rationale refiner. Please refine the ** Assessment rationale ** in the format of \"what key element does the student answer matched, what rubric should apply and what score to get\" to make it sounds plausible:"}]
    messages.append({'role': 'user', 'content': templated_rationale})
    return llm.query(messages=messages, temperature=0.7, candidates=1)

def data_generation(configs, row, prompts, llm, label_range):
    input_column = configs['data']['input_column']
    output_column = configs['data']['label_column']
    
    student_answer = row[input_column]
    ground_truth = row[output_column]

    # Check correctness of the data
    pred_labels = row['pred_labels']
    python_pred = [list(each.keys())[0] for each in pred_labels[0]['Python']]
    llm_pred = [list(each.keys())[0] for each in pred_labels[0]['LLM']]
    # python_pred = [int(each) for each in python_pred]
    # llm_pred = [int(each) for each in llm_pred]
    if str(ground_truth) not in python_pred and str(ground_truth) not in llm_pred:
        logger = logging.getLogger(__name__)
        logger.info(f"Ground truth not in the prediction; ground truth: {ground_truth}, Python: {python_pred}, LLM: {llm_pred}")
        return format_generated([], [])
    
    # Prepare prompts
    question = prompts["question"]
    key_elements = prompts["key_elements_splited"]
    rubric = prompts["rubric_splited"]
    
    # Generate Key Elements training data
    rubric_tree = row['assessment_tree']
    good_choices = []
    bad_choices = []

    for key_element_choices in rubric_tree.children:
        choices = key_element_choices.foo.split(";")[:4]
        key_element_weight = key_element_choices.weight

        # Verfiy consistency
        votes = {}
        for each in key_element_choices.children:
            if 'LLM: ' in each.foo:
                each.foo = each.foo.replace("LLM: ", "")
            elif 'Python: ' in each.foo:
                each.foo = each.foo.replace("Python: ", "")
            if each.foo in votes:
                votes[each.foo] += each.weight
            else:
                votes[each.foo] = each.weight
        if len(votes.keys()) > 1:
            logger = logging.getLogger(__name__)
            logger.info(f"Inconsistency found: {votes}")
        
        score = max(votes, key=votes.get)
        selected_key_element = [key_elements[idx] for idx, each in enumerate(choices) if "Yes" in each]
        shuffle(selected_key_element)
        selected_rubric = [each for each in rubric if f"{score} point" in each][0]
        prompt, templated_rationale = template_rationale(student_answer, selected_key_element, selected_rubric, score)
        output = llm_rewrite(templated_rationale, llm)

        if int(score) == ground_truth:
            good_choices.append(output[0])
        else:
            bad_choices.append({"score": score,"content":output[0]})
    
    if len(good_choices) > 1:
        good_choices = llm_selection(llm, good_choices, student_answer)
        print(good_choices)
    
    if len(good_choices) > 0 and len(bad_choices) == 0:
        label_range = [each for each in label_range if each != ground_truth]
        shuffle(label_range)
        fake_label = label_range[0]
        print(fake_label)
        shuffle(key_elements)
        fake_key_element = key_elements[:fake_label]
        print(fake_key_element)
        selected_rubric = [each for each in rubric if f"{fake_label} point" in each][0]
        prompt, templated_rationale = template_rationale(student_answer, fake_key_element, selected_rubric, fake_label)
        print(templated_rationale)
        output = llm_rewrite(templated_rationale, llm)
        print(output)
        logger = logging.getLogger(__name__)
        logger.info(f"Fake label: {fake_label}, output: {output[0]}")
        bad_choices.append({"score": fake_label,"content":output[0]})
    
    return format_generated(good_choices, bad_choices)

def clean_up_key_elements(key_elements):
    prompt = """Please select Yes for matching this key answer element and No for non-matching this key answer element.\n{{demonstation}}\n[Student Answer]: \"{{student_answer}}\"\n[Decision]:"""
    return [key_element.replace(prompt,"") for key_element in key_elements]

def template_rationale(question, student_answer, key_elements, decisions, rationales, rubric, python_score):
    prompt = f"Here is a student answer to the following question:\n\"{question}\"\n"
    prompt += f"[Student Answer]: \"{student_answer}\"\n"
    prompt += f"This question follows a points mark scheme, and the break down assessment by each Key Answer Elements for this student's answer is as follows:\n"
    for idx, (key_element, decision, rationale) in enumerate(zip(key_elements, decisions, rationales)):
        prompt += f"{idx+1}. {key_element} - {decision}: {rationale}\n"
    prompt += f"According to the: \"{rubric}\", the answer should get a score of {python_score}.\n\n"
    prompt += "Please summarize the above rationales and be FAITHFUL to the given assessment decisions for this student's answer briefly and precisely. Give the summarization in JSON format:\n```JSON\n{\n    \"mark\": \"...\", # numeric\n    \"rationale\": \"...\", # including mark awared, which marking rubric applied, and detailed key elements level rationale\n    \"suggestion\": \"...\" # any answer improvement suggestion\n}```\n- The \"mark\" should be the score of the student's answer.\n- The \"rationale\" should be concise, include the assessed score and rubric applied, more importantly, justify the **marking decision-making processes** by **summarizing** the key element-level rationales, you must **quote the exact part** from the student's answer.\n- If the student didn't get a full mark, you can also provide some improvement suggestions; otherwise, leave it blank."
    return prompt

def prompt_generation(configs, row, prompts):
    input_column = configs['data']['input_column']
    output_column = configs['data']['label_column']
    
    student_answer = row[input_column]
    ground_truth = row[output_column]

    # Extract all paths
    rubric_tree = row['assessment_tree']
    paths = find_paths(rubric_tree)

    question = prompts["question"]
    key_elements = clean_up_key_elements(prompts["key_elements"])
    rubric = prompts["rubric"].replace("\n Please assess the student answer with a score based on the above marking rubric: \n{{demonstation}}\n[Student Answer]: \"{{student_answer}}\"\n[Mark]: ", "")

    query_data = []

    for each in paths:
        if 'Python' in list(each[1].keys())[0]:
            keys = list(each[0].keys())
            decisions = [s.strip() for s in keys[0].split(";") if s.strip()]
            prob = each[0][keys[0]]
            rationales = each[0][keys[1]].strip().split('\n')

            if len(decisions) != len(rationales):
                logger = logging.getLogger(__name__)
                logger.info(f"Decision and rationale mismatch: {decisions}, {rationales}")

            python_score = list(each[1].keys())[0].replace("Python: ", "")

            templated_prompt = template_rationale(question, student_answer, key_elements, decisions, rationales, rubric, python_score)

            if int(python_score) == ground_truth:
                query_data.append({'prompt': templated_prompt, 'score': python_score, 'preference': 'better', 'confidence': prob})
            else:
                query_data.append({'prompt': templated_prompt, 'score': python_score, 'preference': 'worse', 'confidence': prob})

    return query_data