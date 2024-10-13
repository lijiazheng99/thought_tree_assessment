from .model import LLM, openai_models
from .tot import tree_of_thought_assessment
from .generation import data_generation, prompt_generation

def get_max(list_of_dicts):
    max_value = float('-inf')
    max_key = None
    
    # Iterate through each dictionary in the list
    for d in list_of_dicts:
        for key, value in d.items():
            # Update max_value and max_key if the current value is greater than max_value
            if value > max_value:
                max_value = value
                max_key = key
    return max_key
