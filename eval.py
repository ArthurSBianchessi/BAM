import os
import json
import random
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from models.sinusoidal import SinusoidalModelArgs, SinusoidalTransformer
from models.rotary_local import LocalRotaryTransformer, LocalRotaryModelArgs
from models.rotary import RotaryTransformer, RotaryModelArgs
from models.alibi import ALiBiModelArgs, ALiBiTransformer
from models.bam_uninterpretable import BATransformer0, BATModelArgs0
from models.bam import BATransformer, BATModelArgs 
from models.laplace import LaplaceTransformer, LaplaceModelArgs



class PasskeyEvaluator:
    def __init__(self, seq_lens, device='cpu', pred_digits=5, preffix_digits=1, sampling='equidistant'):
        self.seq_lens = seq_lens
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator = PromptGenerator(digits=pred_digits+preffix_digits)
        self.device = device
        self.pred_digits = pred_digits
        self.preffix_digits = preffix_digits
        self.sampling = sampling

    @torch.inference_mode()
    # def evaluate(self, model, sample_size=100, verbose=True, patience=3):
    def evaluate(self, model, sample_size=100, verbose=True, patience=3):
        model.to(self.device)
        accs = []
        seq_lens = []
        self.patience = patience
        for seq_len in self.seq_lens:
            correct = 0
            seq_lens.append(len(self.generator(seq_len)[0][0]))
            prompts, passkeys = self.generator(seq_len, sample_size, self.sampling)
            for prompt, pass_key in zip(prompts, passkeys):
                if not len(prompt) == seq_lens[-1]:
                    raise ValueError(f"Prompt length {len(prompt)} does not match expected length {seq_lens[-1]}")
                model_input = torch.tensor(prompt+pass_key).unsqueeze(0).to(self.device)
                output = model(model_input)
                pred_pass_key = output.max(-1).indices[0][-self.pred_digits-1:-1].cpu()
                # print(self.generator.tokenizer.decode(pass_key))
                # print(self.generator.tokenizer.decode(pred_pass_key))
                # print()
                if (list(pred_pass_key) == pass_key[self.preffix_digits+1:]):
                    correct += 1
            accs.append(correct/sample_size)
            if verbose:
                print(f"seq_len: {len(prompt)}, acc: {correct/sample_size*100:04.1f}%")
                # print(f"seq_len: {len(prompt)}, acc: {correct}/{sample_size}")
            if correct == 0:
                patience -= 1
            else:
                patience = self.patience
            if patience == 0:
                print(f"Early stopping at seq_len: {seq_lens[-1]}")
                break
        model.to('cpu')
        return seq_lens, accs
        


class PromptGenerator:
    def __init__(self, digits=8, tokenizer='mistralai/Mistral-7B-Instruct-v0.3'):
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
        self.information_line_len   = len(self.tokenizer(self.information_line.format(pass_key=10**digits-1), add_special_tokens=False)['input_ids'])
        self.final_question_len     = len(self.final_question_tokens)
        self.n_digits               = digits

    # def generate_prompt(self, length):
    #     n_garbage = (length -self.task_description_len -self.information_line_len -self.final_question_len) // self.garbage_inf_len
    #     n_garbage_prefix = random.randint(0, n_garbage) if n_garbage > 0 else 0
    #     return self.__generate_prompt(n_garbage, n_garbage_prefix)
    
    # def generate_prompts(self, length, n_prompts):
    #     prompts = []
    #     passkeys = []
    #     n_garbage = (length -self.task_description_len -self.information_line_len -self.final_question_len) // self.garbage_inf_len
    #     paskeys_positions = torch.linspace(0, n_garbage-1, n_prompts).long()
    #     for passkey_position in paskeys_positions:
    #         n_garbage_prefix = passkey_position
    #         prompt, passkey_tokens = self.__generate_prompt(n_garbage, n_garbage_prefix)
    #         prompts.append(prompt)
    #         passkeys.append(passkey_tokens)
    #     return prompts, passkeys
    
    def __generate_prompt(self, n_garbage, n_garbage_prefix):
        n_garbage_suffix = n_garbage - n_garbage_prefix

        pass_key = random.randint(10**(self.n_digits-1), 10**self.n_digits-1)
        information_line = self.information_line.format(pass_key=pass_key)

        information_tokens  = self.tokenizer(information_line, add_special_tokens=False)['input_ids']
        passkey_tokens      = self.tokenizer(' ' + str(pass_key), add_special_tokens=False)['input_ids']

        garbage_prefix = self.garbage_inf_tokens * n_garbage_prefix
        garbage_suffix = self.garbage_inf_tokens * n_garbage_suffix

        prompt = self.task_description_tokens + garbage_prefix + information_tokens + garbage_suffix + self.final_question_tokens
        return prompt, passkey_tokens
    
    def generate_prompt(self, length, sample_size=1, sampling='random'):
        prompts = []
        passkeys = []
        n_garbage = (length -self.task_description_len -self.information_line_len -self.final_question_len) // self.garbage_inf_len
        n_garbage = max(n_garbage, 0)
        if sampling == 'random':
            passkey_positions = random.choices(range(n_garbage+1), k=sample_size)
        elif sampling == 'equidistant':
            passkey_positions = torch.linspace(0, n_garbage, sample_size).long().tolist()
        elif sampling == 'beginning':
            passkey_positions = [0] * sample_size
        elif sampling == 'end':
            passkey_positions = [n_garbage] * sample_size
        else:
            raise ValueError(f"Unknown sampling method: {sampling}")
        for passkey_position in passkey_positions:
            prompt, passkey_tokens = self.__generate_prompt(n_garbage, passkey_position)
            prompts.append(prompt)
            passkeys.append(passkey_tokens)
        return prompts, passkeys
    
    def __call__(self, length, sample_size=1, sampling='random'):
        return self.generate_prompt(length, sample_size, sampling)


    


def load_model(dir, comp=''):
    with open(dir+'args.json') as f:
        args = json.load(f)
    
    ModelArgs, Transformer = {
        "rotary":       (RotaryModelArgs,       RotaryTransformer       ),
        "rotary_local": (LocalRotaryModelArgs,  LocalRotaryTransformer  ),
        "sinusoidal":   (SinusoidalModelArgs,   SinusoidalTransformer   ),
        "alibi":        (ALiBiModelArgs,        ALiBiTransformer        ),
        "bam":          (BATModelArgs,          BATransformer           ),
        "bam_uninterpretable": (BATModelArgs0, BATransformer0),
        "laplace":      (LaplaceModelArgs,      LaplaceTransformer      ),
    }[args['args']['position_encoding']]
    model_dict = torch.load(dir+f'model{comp}.pt')
    model = Transformer(ModelArgs(**args['model_args']))
    model_dict = {k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in model_dict.items()}
    model.load_state_dict(model_dict)
    return model
