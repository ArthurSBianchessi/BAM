import random
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


def generate_prompt(n_garbage):
    """Generates a text file and inserts an execute line at a random position."""
    n_garbage_prefix = random.randint(0, n_garbage)
    n_garbage_suffix = n_garbage - n_garbage_prefix

    task_description = "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there."
    garbage_inf = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
    while len(garbage_inf) < n_garbage:
        garbage_inf = " ".join([garbage_inf] * 2)
    garbage_prefix = garbage_inf[:n_garbage_prefix]
    garbage_suffix = garbage_inf[:n_garbage_suffix]
    pass_key = random.randint(1, 50000)
    information_line = f"The pass key is {pass_key}. Remember it. {pass_key} is the pass key."
    final_question = "What is the pass key? The pass key is"
    lines = [
        task_description,
        garbage_prefix,
        information_line,
        garbage_suffix,
        final_question
    ]
    return "\n".join(lines), pass_key

class PromptGenerator:
    def __init__(self, tokenizer='mistralai/Mistral-7B-Instruct-v0.3'):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

        self.task_description   = "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there."
        self.garbage_inf        = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
        self.information_line   = "The pass key is {pass_key}. Remember it. {pass_key} is the pass key."
        self.final_question     = "What is the pass key? The pass key is"

        self.task_description_tokens    = self.tokenizer(self.task_description, add_special_tokens=False)['input_ids']
        self.garbage_inf_tokens         = self.tokenizer(self.garbage_inf, add_special_tokens=False)['input_ids']
        self.final_question_tokens      = self.tokenizer(self.final_question, add_special_tokens=False)['input_ids']

        self.task_description_len   = len(self.task_description_tokens)
        self.garbage_inf_len        = len(self.garbage_inf_tokens)
        self.information_line_len   = len(self.tokenizer(self.information_line.format(pass_key=10_000), add_special_tokens=False)['input_ids'])
        self.final_question_len     = len(self.final_question_tokens)

    def generate_prompt(self, length):
        n_garbage = (length -self.task_description_len -self.information_line_len -self.final_question_len) // self.garbage_inf_len
        n_garbage_prefix = random.randint(0, n_garbage)
        n_garbage_suffix = n_garbage - n_garbage_prefix

        pass_key = random.randint(10_000, 99_999)
        information_line = self.information_line.format(pass_key=pass_key)

        information_tokens  = self.tokenizer(information_line, add_special_tokens=False)['input_ids']
        passkey_tokens      = self.tokenizer(' ' + str(pass_key), add_special_tokens=False)['input_ids']
        # passkey_tokens      = self.tokenizer(' ' + str(pass_key) + '.', add_special_tokens=False)['input_ids']

        garbage_prefix = self.garbage_inf_tokens * n_garbage_prefix
        garbage_suffix = self.garbage_inf_tokens * n_garbage_suffix

        prompt = self.task_description_tokens + garbage_prefix + information_tokens + garbage_suffix + self.final_question_tokens
        return prompt, passkey_tokens

    def __call__(self, length, num_prompts, return_tensors=True):
        prompts = []
        pass_keys = []
        for _ in range(num_prompts):
            prompt, pass_key = self.generate_prompt(length)
            prompts.append(prompt + pass_key)
            pass_keys.append(pass_key)

        if return_tensors:
            prompts = torch.tensor(prompts)
            pass_keys = torch.tensor(pass_keys)
        return prompts, pass_keys


        



