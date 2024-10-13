import openai
import decimal
from openai import OpenAI
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from tokencost import calculate_prompt_cost, calculate_completion_cost

openai_models = [
    "gpt4",
    "gpt-42", 
    "gpt35",
    "gpt-4-1106-preview",
    "gpt-3.5-turbo-0125"
]

mixtral_models = [
    "open-mixtral-8x7b"
]

hf_models = [
    "meta-llama/Llama-2-7b-chat-hf", 
    "meta-llama/Llama-2-13b-chat-hf", 
    "meta-llama/Llama-2-70b-chat-hf",
    "meta-llama/Meta-Llama-3-8B",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Meta-Llama-3-70B",
    "mistralai/Mixtral-8x7B-Instruct-v0.1", 
    "mistralai/Mixtral-8X7B-Instruct-v0.1",
    "google/gemma-2b-it", 
    "google/gemma-7b-it",
    "google/flan-t5-xxl"
]

class LLM:
    def __init__(self, config):
        self.model = config["model"]
        if self.model in openai_models:
            self.api = True
            if self.model == "gpt-42":
                openai.api_type = "azure"
                openai.api_key = ""
                openai.azure_endpoint = ""
                openai.api_version = ""
            else:
                openai.api_key = ""
            self.api_prompt_cost = decimal.Decimal(0.)
            self.api_completion_cost = decimal.Decimal(0.)
        elif self.model in mixtral_models:
            self.api = True
            self.client = MistralClient(api_key="")
        elif self.model in ["mistralai/Mixtral-8X7B-Instruct-v0.1"]:
            self.api = True
            # Modify OpenAI's API key and API base to use vLLM's API server.
            openai_api_key = "EMPTY"
            openai_api_base = "http://localhost:8000/v1"

            self.client = OpenAI(
                # defaults to os.environ.get("OPENAI_API_KEY")
                api_key=openai_api_key,
                base_url=openai_api_base,
            )

        elif self.model in hf_models:
            self.api = False
            import torch
            import transformers
            from transformers import (
                AutoTokenizer,
                AutoModelForSeq2SeqLM,
                AutoModelForCausalLM,
                BitsAndBytesConfig
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model)
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            if "t5" in self.model:
                mode = "text2text-generation"
            else:
                mode = "text-generation"
            
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            if "lora" in config.keys():
                from peft import PeftModel, PeftConfig
                lora_path = config["lora_path"]
                model = AutoModelForCausalLM.from_pretrained(self.model, device_map="auto", low_cpu_mem_usage=True, quantization_config=quantization_config)
                # self.tokenizer = AutoTokenizer.from_pretrained(lora_path, padding_side='left')
                self.model = PeftModel.from_pretrained(
                    model=model, 
                    model_id=lora_path,
                    adapter_name="peft",
                    device=model.device,
                    device_map="auto",
                    quantization_config=quantization_config,
                    attn_implementation="flash_attention_2"
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(self.model, device_map=0, low_cpu_mem_usage=True, quantization_config=quantization_config)
                # self.model = AutoModelForCausalLM.from_pretrained(self.model, device_map=3, low_cpu_mem_usage=True)
            self.pipeline = transformers.pipeline(
                mode,
                model=self.model,
                tokenizer=self.tokenizer,
                torch_dtype=torch.bfloat16,
                device_map=0,
                max_new_tokens=200
            )
        else:
            raise ValueError("Model not supported. Supported models are ", openai_models + mixtral_models + hf_models)
        # other config
        self.tempreture = config["tempreture"]
        self.candidates = config["candidates"]
    
    def __openai_completion(self, messages=[], temperature=None, candidates=None, max_tokens=None):
        completion = openai.chat.completions.create(
            model=self.model,
            temperature=temperature,
            messages=messages,
            n=candidates,
            max_tokens=max_tokens
        )
        return completion
    
    def __vllm_completion(self, messages=[], temperature=None, candidates=None, max_tokens=None):
        completion = self.client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            messages=messages,
            n=candidates,
            max_tokens=max_tokens
        )
        return completion
    
    def __mistral_completion(self, messages=[], temperature=None, candidates=None, max_tokens=None):
        decisions = []
        for i in range(candidates):
            chat_response = self.client.chat(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=max_tokens
            )
            decisions.append(chat_response.choices[0].message.content)
        return decisions
    
    def __huggingface_completion(self, messages=[], temperature=None, candidates=None, max_tokens=None):
        input_texts = self.tokenizer.apply_chat_template(messages, tokenize=False)
        cut_off = len(input_texts)
        outputs = self.pipeline(
            input_texts,
            do_sample=True,
            top_k=10,
            num_return_sequences=candidates,
            temperature=temperature,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=300 if max_tokens is None else max_tokens
        )
        output_texts = [each['generated_text'][cut_off:] for each in outputs]
        return output_texts
    
    def __openai_cost_analysis(self, prompt, completions):
        if 'gpt-4' in self.model:
            model_config = 'gpt-4'
        else:
            model_config = self.model
        prompt_cost = calculate_prompt_cost(prompt, model_config)
        total_completion_cost = sum([calculate_completion_cost(completion, model_config) for completion in completions])
        self.api_prompt_cost += prompt_cost
        self.api_completion_cost += total_completion_cost
    
    def report_cost(self):
        if self.api:
            print(f"Prompt cost: {self.api_prompt_cost}, Completion cost: {self.api_completion_cost}")
            return {"prompt_cost": self.api_prompt_cost, "completion_cost": self.api_completion_cost, "total_cost": self.api_prompt_cost + self.api_completion_cost}
        else:
            return None

    def query(self, messages=[], temperature=None, candidates=None, max_tokens=None):
        temperature = self.tempreture if temperature is None else temperature
        candidates = self.candidates if candidates is None else candidates

        if self.api:
            if self.model in openai_models:
                decisions = self.__openai_completion(messages=messages, temperature=temperature, candidates=candidates, max_tokens=max_tokens)
                decisions = [choice.message.content for choice in decisions.choices]
                try:
                    self.__openai_cost_analysis(messages, decisions)
                except:
                    if temperature < 1.0:
                        temperature = temperature + 0.1
                    decisions = self.query(messages=messages, temperature=temperature, candidates=candidates, max_tokens=max_tokens)
            elif self.model in mixtral_models:
                decisions = self.__mistral_completion(messages=messages, temperature=temperature, candidates=candidates, max_tokens=max_tokens)
            elif self.model in ["mistralai/Mixtral-8X7B-Instruct-v0.1"]:
                decisions = self.__vllm_completion(messages=messages, temperature=temperature, candidates=candidates, max_tokens=max_tokens)
                decisions = [choice.message.content for choice in decisions.choices]
            return decisions
        else:
            return self.__huggingface_completion(messages=messages, temperature=temperature, candidates=candidates, max_tokens=max_tokens)
