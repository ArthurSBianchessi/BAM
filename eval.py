import os
import json
import random
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from time import time

from inference_models.sinusoidal import SinusoidalModelArgs, SinusoidalTransformer
from inference_models.rotary_local import LocalRotaryTransformer, LocalRotaryModelArgs
from inference_models.rotary_ssmax import RotarySSMaxModelArgs, RotarySSMaxTransformer
from inference_models.rotary import RotaryTransformer, RotaryModelArgs
from inference_models.alibi import ALiBiModelArgs, ALiBiTransformer
from inference_models.bam_uninterpretable import BATransformer0, BATModelArgs0
from inference_models.bam import BATransformer, BATModelArgs 
from inference_models.bam_ssmax import SSMaxBATransformer, SSMaxBATModelArgs
from inference_models.laplace import LaplaceTransformer, LaplaceModelArgs
from models.nope import NoPEModelArgs, NoPETransformer



class PasskeyEvaluator:
    def __init__(self, seq_lens, device='cpu', pred_digits=5, preffix_digits=0, sampling='equidistant', patience=float('inf')):
        self.seq_lens = seq_lens
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator = PromptGenerator(digits=pred_digits+preffix_digits)
        self.device = device
        self.pred_digits = pred_digits
        self.preffix_digits = preffix_digits
        self.sampling = sampling
        self.patience = patience

    @torch.inference_mode()
    # def evaluate(self, model, sample_size=100, verbose=True, patience=3):
    def evaluate(self, model, sample_size=10, verbose=True, patience=None):
        model.to(self.device)
        accs = []
        seq_lens = []
        patience = patience or self.patience
        print(8)
        for seq_len in self.seq_lens:
            print(f"                0/0 correct", end='\r')
            correct = 0
            seq_lens.append(len(self.generator(seq_len)[0][0]))
            prompts, passkeys = self.generator(seq_len, sample_size, self.sampling)
            start = time()
            for i, (prompt, pass_key) in enumerate(zip(prompts, passkeys)):
            # for prompt, pass_key in zip(prompts, passkeys):
                if not len(prompt) == seq_lens[-1]:
                    raise ValueError(f"Prompt length {len(prompt)} does not match expected length {seq_lens[-1]}")
                model_input = torch.tensor(prompt+pass_key).unsqueeze(0).to(self.device)
                # output = model(model_input)
                output = model(model_input, seq_batch_size=8)
                # output = model(model_input, seq_batch_size=32)
                # pred_pass_key = output.max(-1).indices[0][-self.pred_digits-1:-1].cpu()
                pred_pass_key = output[0, -self.pred_digits-1:-1].cpu()
                # print(self.generator.tokenizer.decode(pass_key))
                # print(self.generator.tokenizer.decode(pred_pass_key))
                # print()
                if (list(pred_pass_key) == pass_key[self.preffix_digits+1:]):
                    correct += 1
                end = time()
                # print(f"seq_len: {len(prompt)}, acc: {correct}/{i+1} of {sample_size}", end='\r')
                print(f"                seq_len: {len(prompt)}, acc: {correct}/{i+1} of {sample_size} took {(end-start)/(i+1):.2f}s", end='\r')
            accs.append(correct/sample_size)
            if verbose:
                print(f"seq_len: {len(prompt)}, acc: {correct/sample_size*100:04.1f}%                                                                                        ")
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


    




class PerplexityEvaluator:
    def __init__(self, dataset_dir, seq_len, ntokens, window_size=512, window_pos='middle'):
        self.dataset_dir    = os.path.join('data', dataset_dir)
        self.seq_len        = seq_len
        self.ntokens        = ntokens
        self.window_size    = window_size
        self.window_pos     = window_pos
        self.nwindows       = int(seq_len // window_size)
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')

        assert seq_len % window_size == 0, f"seq_len {seq_len} must be divisible by window_size {window_size}"

        self.tokens         = self.get_val_set()
        # glob files that match the pattern
        file = sorted(os.listdir(self.dataset_dir))[0]
        filename = os.path.join(self.dataset_dir, file)
        assert len(self.files) > 0, f"did not find any files that match the pattern {self.dataset_dir}"

        dataset = torch.load(filename)
        tokens     = dataset['input_ids'][:ntokens]

        # Identify EOS and BOS tokens 
        tokenizer      = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.3')
        eos_token_id   = tokenizer.eos_token_id
        bos_token_id   = tokenizer.bos_token_id

        # Remove EOS and BOS tokens from the dataset
        eos_idxs = tokens == eos_token_id
        bos_idxs = tokens == bos_token_id
        idxs = eos_idxs | bos_idxs
        tokens = tokens[~idxs]

        nbatches = len(tokens) // seq_len
        tokens = tokens[:nbatches * seq_len]
        self.tokens = tokens.reshape(nbatches, seq_len)

    def evaluate(self, model, prev_results=None):
        if prev_results is not None:
            perplexity = prev_results['perplexity']
            positions = prev_results['positions']
            return positions, perplexity
        entropies = torch.zeros(self.nwindows)
        for i, tokens in enumerate(self.tokens):
            output = model(tokens.unsqueeze(0))
            entropies += self.cross_entropy(output[:, :-1, :], tokens[1:]).reshape(-1, self.window_size).mean(dim=-1).cpu()
        entropies /= len(self.tokens)
        perplexity = torch.exp(entropies)

        positions = torch.arange(self.nwindows) * self.window_size
        if self.window_pos == 'start':
            pass
        elif self.window_pos == 'middle':
            positions += self.window_size // 2
        elif self.window_pos == 'end':
            positions += self.window_size - 1
        else:
            raise ValueError(f"Unknown window position: {self.window_pos}")
        return positions.tolist(), perplexity.tolist()
    
class Evaluator:
    # def __init__(self, dataset_dir, seq_len, ntokens, window_size=512, window_pos='middle'):
    # def __init__(self, seq_lens, device='cpu', pred_digits=5, preffix_digits=0, sampling='equidistant', patience=float('inf')):
    def __init__(self,
                 passkey_seq_lens=None,
                 passkey_sample_size=10,
                 passkey_pred_digits=5,
                 passkey_preffix_digits=0,
                 passkey_samplings=['equidistant', 'beginning'],
                 passkey_patience=float('inf'),
                 perplexity_dataset_dir='10B',
                 perplexity_seq_len=10_240,
                 perplexity_ntokens=3_932_160,
                 window_size=512,
                 window_pos='middle',
                 device='cpu',
                 ):
        self.passkey_seq_lens = passkey_seq_lens or [0, 1_000, 2_000, 3_000, 4_000, 5_000, 6_000, 7_000, 8_000, 9_000, 10_000, 
                                                     20_000, 30_000, 40_000, 50_000, 60_000, 70_000, 80_000, 90_000, 100_000, 
                                                     200_000, 300_000, 400_000, 500_000]
        self.passkey_evaluator = PasskeyEvaluator(
            seq_lens=self.passkey_seq_lens,
            device=device,
            pred_digits=passkey_pred_digits,
            preffix_digits=passkey_preffix_digits,
            sampling=passkey_samplings[0],
            patience=passkey_patience
        )
        self.perplexity_evaluator = PerplexityEvaluator(
            dataset_dir=perplexity_dataset_dir,
            seq_len=perplexity_seq_len,
            ntokens=perplexity_ntokens,
            window_size=window_size,
            window_pos=window_pos
        )
        self.device = device

    def evaluate(self, model_dir, evals=['passkey', 'perplexity']):
        model = self.load_model(model_dir)
        results = self.load_results(model_dir)



        if 'passkey' in evals:
            passkey_lens, passkey_accs = self.passkey_evaluator.evaluate(model, sample_size=self.passkey_sample_size, prev_results=results['passkey'])

        if 'perplexity' in evals:
            perplexity_lens, perplexity_accs = self.perplexity_evaluator.evaluate(model, prev_results=results['perplexity'])
        


    def load_model(self, dir):
        with open(dir+'args.json') as f:
            args = json.load(f)

        ModelArgs, Transformer = {
            "rotary":       (RotaryModelArgs,       RotaryTransformer       ),
            "rotary_local": (LocalRotaryModelArgs,  LocalRotaryTransformer  ),
            "rotary_ssmax": (RotarySSMaxModelArgs,  RotarySSMaxTransformer  ),
            "sinusoidal":   (SinusoidalModelArgs,   SinusoidalTransformer   ),
            "alibi":        (ALiBiModelArgs,        ALiBiTransformer        ),
            "bam":          (BATModelArgs,          BATransformer           ),
            "bam_ssmax":    (SSMaxBATModelArgs,     SSMaxBATransformer      ),
            "laplace":      (LaplaceModelArgs,      LaplaceTransformer      ),
            "nope":         (NoPEModelArgs,         NoPETransformer         ),
            "bam_uninterpretable": (BATModelArgs0, BATransformer0),
        }[args['args']['position_encoding']]
        # model_dict = torch.load(dir+f'model{comp}.pt')
        model_dict = torch.load(os.path.join(dir, f'model.pt'))
        model = Transformer(ModelArgs(**args['model_args']))
        model_dict = {k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in model_dict.items()}
        model.load_state_dict(model_dict)
        return model
    
    def load_results(self, dir):
        if os.path.exists(os.path.join(dir, 'results.json')):
            with open(os.path.join(dir, 'results.json')) as f:
                results = json.load(f)
        else:
            results = {}
        return results
