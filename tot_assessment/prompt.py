import logging
import re
regex = r"\d+ point[s]?|No point"

def group_decisions(decisions):
    votes = {"Yes": 0, "No": 0, "Not Valid": 0}
    for choice in decisions:
        content = choice.lower()
        if "yes" in content:
            votes["Yes"] += 1
        elif "no" in content:
            votes["No"] += 1
        else:
            votes["Not Valid"] += 1
            logging.warning(f"Invalid decision: {content}")
    # Filter out 0s and Not valid
    votes = {k: v for k, v in votes.items() if v > 0 and k != "Not Valid"}
    # Normalize
    votes = {k: v/len(decisions) for k, v in votes.items()}
    return votes

def group_scores(decisions):
    scores = {}
    for choice in decisions:
        content = choice.lower()
        try:
            score = int(re.search(regex, content).group(0)[0])
            if score in scores:
                scores[score] += 1
            else:
                scores[score] = 1
        except:
            logging.warning(f"Invalid decision: {content}")
            if "Not Valid" in scores:
                scores["Not Valid"] += 1
            else:
                scores["Not Valid"] = 1
    # Filter out 0s and Not valid
    scores = {k: v for k, v in scores.items() if v > 0 and k != "Not Valid"}
    # Normalize
    scores = {k: v/len(decisions) for k, v in scores.items()}
    return scores

def prompt_selector(history_mode, layer, prompt_stack):
    """
    Selects and constructs prompt messages based on the given history mode, prompt stack, and student answer.

    Args:
        history_mode (str): The history mode to determine the construction of prompt messages.
        prompt_stack (list): The stack of prompts containing prompt types and content.
        student_answer (str): The student's answer to be included in the constructed messages.

    Returns:
        list: A list of constructed messages with roles (user or assistant) and content.

    """
    if layer == "key_element":
        constructed_messages = []
        if history_mode == "no_history":
            # Construct messages without history
            # E.g. Only contains Question, current key element, 
            # current demonstrations and student answer
            message = " "
            last_key_element = ""
            for prompt in prompt_stack:
                if prompt['prompt_type'] == 'question':
                    message += prompt['prompt'] + "\n\n"
                if prompt['prompt_type'] == 'key_element':
                    last_key_element = prompt['prompt']
                if prompt['prompt_type'] == 'key_element_demo':
                    last_key_element = last_key_element.replace("{{demonstation}}", prompt['prompt'])
            message += last_key_element
            constructed_messages.append({'role': 'user', 'content': message})
            return constructed_messages
        elif history_mode == "wo_demo_history":
            # Construct messages without demonstration history
            # E.g. Contains Question, all key element, current demonstrations, 
            # student answer, and all previous model output
            message = " "
            last_key_element = ""
            last_key_element_demo = ""
            for prompt in prompt_stack:
                if prompt['prompt_type'] == 'question':
                    message += prompt['prompt'] + "\n\n"
                if prompt['prompt_type'] == 'key_element':
                    last_key_element = prompt['prompt']
                if prompt['prompt_type'] == 'key_element_demo':
                    last_key_element_demo = prompt['prompt']
                if prompt['prompt_type'] == 'model_output':
                    last_key_element = last_key_element.replace("{{demonstation}}", "")
                    message += last_key_element
                    constructed_messages.append({'role': 'user', 'content': message})
                    constructed_messages.append({'role': 'assistant', 'content': prompt['prompt']})
                    message = " "
            last_key_element = last_key_element.replace("{{demonstation}}", last_key_element_demo)
            message += last_key_element
            constructed_messages.append({'role': 'user', 'content': message})
            return constructed_messages
        elif history_mode == "full_history":
            # Construct messages with full history
            # E.g. Contains Question, all key elements, all demonstrations,
            # student answer, and all model outputs
            message = " "
            last_key_element = ""
            for prompt in prompt_stack:
                if prompt['prompt_type'] == 'question':
                    message += prompt['prompt'] + "\n\n"
                if prompt['prompt_type'] == 'key_element':
                    last_key_element = prompt['prompt']
                if prompt['prompt_type'] == 'key_element_demo':
                    last_key_element = last_key_element.replace("{{demonstation}}", prompt['prompt'])
                    message += last_key_element.replace("{{student_answer}}", student_answer)
                if prompt['prompt_type'] == 'model_output':
                    constructed_messages.append({'role': 'user', 'content': message})
                    constructed_messages.append({'role': 'assistant', 'content': prompt['prompt']})
                    message = " "
            constructed_messages.append({'role': 'user', 'content': message})
            return constructed_messages
        else:
            raise ValueError("Invalid history_mode")
    elif layer == "rubric":
        student_answer = ""
        constructed_messages = []
        if history_mode in ["no_history", "wo_demo_history"]:
            # Construct messages without history
            # E.g. Only contains Question, current key element, 
            # current demonstrations and student answer
            message = " "
            last_rubric = ""
            for prompt in prompt_stack:
                if prompt['prompt_type'] == 'question':
                    message += prompt['prompt'] + "\n\n"
                if prompt['prompt_type'] == 'key_element':
                    message += prompt['prompt'].replace("{{demonstation}}", "")
                if prompt['prompt_type'] == 'model_output':
                    constructed_messages.append({'role': 'user', 'content': message})
                    constructed_messages.append({'role': 'assistant', 'content': prompt['prompt']})
                    message = " "
                if prompt['prompt_type'] == 'rubric':
                    last_rubric = prompt['prompt']
            message += last_rubric
            constructed_messages.append({'role': 'user', 'content': message})
            return constructed_messages
        elif history_mode == "full_history":
            # Construct messages with full history
            # E.g. Contains Question, all key elements, all demonstrations,
            # student answer, and all model outputs
            message = " "
            last_rubric = ""
            for prompt in prompt_stack:
                if prompt['prompt_type'] == 'question':
                    message += prompt['prompt'] + "\n\n"
                if prompt['prompt_type'] == 'key_element':
                    message += prompt['prompt']
                if prompt['prompt_type'] == 'key_element_demo':
                    message.replace("{{demonstation}}", prompt['prompt'])
                if prompt['prompt_type'] == 'model_output':
                    constructed_messages.append({'role': 'user', 'content': message})
                    constructed_messages.append({'role': 'assistant', 'content': prompt['prompt']})
                    message = " "
                if prompt['prompt_type'] == 'rubric':
                    last_rubric = prompt['prompt']
            message += last_rubric
            constructed_messages.append({'role': 'user', 'content': message})
            return constructed_messages
        else:
            raise ValueError("Invalid history_mode")
    else:
        raise ValueError("Invalid layer")
