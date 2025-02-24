
import argparse
import os
import math
import glob
import inspect
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
import time
from typing import (
    List,
)

import numpy as np
import torch
import torch._inductor.config as config
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist

from torch.utils.data import IterableDataset
from datasets import load_dataset
from transformers import AutoTokenizer

# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

def _peek_data_shard(filename):
    # only reads the header, returns header data
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
    if header[0] != 20240520:
        print("ERROR: magic number mismatch in the data .bin file!")
        exit(1)
    assert header[1] == 1, "unsupported version"
    ntok = header[2] # number of tokens (claimed)
    return ntok # for now just return the number of tokens

def _load_data_shard(filename):
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
        assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        ntok = header[2] # number of tokens (claimed)
        # the rest of it are tokens, stored as uint16
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, "number of tokens read does not match header?"
    return tokens

class DistributedShardedDataLoader:
    """
    This DataLoader is both:
    - distributed (works correctly in case of multiple processes in DDP)
    - sharded (supports datasets that are broken up into multiple data shards)
    It is not *permuted*, meaning that it itearates over the data in the order
    of the dataset on disk, so the user should make sure to shuffle their examples
    during the creation of their data shards for best performance.
    """
    def __init__(self, filename_pattern, B, T, process_rank, num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T

        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"

        # load and validate all data shards, count number of tokens in total
        ntok_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= num_processes * B * T + 1
            ntok_total += shard_ntok
        self.ntok_total = ntok_total
        print0(f"DataLoader: total number of tokens: {ntok_total:24,} across {len(self.files)} files")

        # kick things off
        self.current_shard = None
        self.reset()

    def reset(self):
        # we're being a bit clever here: if we already had shard 0 loaded,
        # then don't do the work to reload it, just reset the pointer
        if self.current_shard != 0:
            self.current_shard = 0
            self.tokens = _load_data_shard(self.files[self.current_shard])
        self.current_position = self.process_rank * self.B * self.T

    def advance(self): # advance to next data shard
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        B = self.B
        T = self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        buf = torch.tensor(buf, dtype=torch.long)
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the start pointer in current shard
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds advance the shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()
        return x, y

class FineWebDistributedDataset(IterableDataset):
    versions = {
        '10B': 'sample-10BT',
        '100B': 'sample-100BT',
        '350B': 'sample-350BT',
        'default': 'default',
    }
    def __init__(self, version, step_token_count, samples_per_fwd, rank=0, world_size=1, shard_size=128*27*5, max_length=1024,
                 tokenizer="mistralai/Mistral-7B-Instruct-v0.3"):
        self.dataset            = load_dataset("HuggingFaceFW/fineweb", name=self.versions[version])['train']
        self.rank               = rank
        self.world_size         = world_size
        self.shard_size         = shard_size
        self.max_length         = max_length
        self.samples_per_fwd    = samples_per_fwd
        self.step_token_count   = step_token_count

        assert self.shard_size % self.samples_per_fwd == 0, "Shard size must be divisible by samples_per_fwd"

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.tokenizer.pad_token = self.tokenizer.unk_token

        self.curr_pos = self.rank * self.shard_size
        self.shard_pos = 0
        # self.current_shard = self.tokenize(self.dataset[self.curr_pos:self.curr_pos+self.shard_size]['text'])
        self.current_shard = self.tokenize_shard(self.curr_pos, self.curr_pos+self.shard_size)

        self.stop = torch.tensor([0])
        
        self.is_ddp = torch.distributed.is_initialized() 

    def next_shard(self):
        self.curr_pos += self.shard_size*self.world_size
        self.shard_pos = 0
        start = self.curr_pos
        end = self.curr_pos + self.shard_size
        if end < len(self.dataset):
            self.current_shard = self.tokenize_shard(start, end)
        else:
            self.stop[0] = 1
    
    def tokenize_shard(self, start, end):
        return self.tokenizer(self.dataset[start:end]['text'], truncation=True, max_length=self.max_length, 
                              return_length=True, padding=False, return_attention_mask=False)
                              
    def pad_token_ids(self, input_ids):
        tokens = {'input_ids': input_ids}
        return self.tokenizer.pad(tokens, return_tensors='pt')
        

    def __iter__(self):
        return self
    
    def __next__(self):
        token_count = 0
        lengths = self.current_shard['length'][self.shard_pos:]
        batched_tokens = []
        while token_count < self.step_token_count:
            if self.shard_pos >= self.shard_size:
                self.next_shard()
            
            if self.is_ddp:
                dist.all_reduce(self.stop, op=dist.ReduceOp.MAX)
            if self.stop[0]:
                raise StopIteration
            
            batched_tokens.append(self.current_shard['input_ids'][self.shard_pos:self.shard_pos+self.samples_per_fwd])
            self.shard_pos += self.samples_per_fwd
            
            # Syncronize the token_count
            current_token_count = torch.tensor(sum(lengths[:self.shard_pos]))
            if self.is_ddp:
                dist.all_reduce(current_token_count, op=dist.ReduceOp.SUM)
            token_count += current_token_count.item()

        batches = []
        for i in range(len(batched_tokens)):
            # batched_tokens[i] = self.pad(batched_tokens[i])
            targets = [input_id[1:]+[self.tokenizer.eos_token_id] for input_id in batched_tokens[i]]
            # batched_targets.append(self.pad(targets))
            batches.append((self.pad_token_ids(batched_tokens[i]), self.pad_token_ids(targets)))
        return batches, token_count
        

                    
# -----------------------------------------------------------------------------
# Python -> C bridge utilities for saving params/grads/activations to .bin files

def write_fp32(tensor, file):
    t = tensor.detach().cpu().to(torch.float32)
    b = t.numpy().tobytes()
    file.write(b)

def write_bf16(tensor, file):
    t = tensor.detach().cpu().to(torch.bfloat16)
    # numpy doesn't have bf16 datatype so we have to trick it
    t = t.view(torch.int16) # trick: reinterpret as int16
    b = t.numpy().tobytes()
    file.write(b)

def write_tensors(model_tensors, L, file, dtype):
    # writes LLaMA 3 model's weights to a binary file
    assert dtype in {"float32", "bfloat16"}
    write_fun = write_fp32 if dtype == "float32" else write_bf16
    write_fun(model_tensors["transformer.wte.weight"], file) # (V, C)
    for i in range(L): # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.ln_1.weight"], file)
    for i in range(L): # (L, 3C, C)
        write_fun(model_tensors[f"transformer.h.{i}.attn.c_attn.weight"], file)
    for i in range(L): # (L, C, C)
        write_fun(model_tensors[f"transformer.h.{i}.attn.c_proj.weight"], file)
    for i in range(L): # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.ln_2.weight"], file)
    for i in range(L): # (L, 4C, C)
        write_fun(model_tensors[f"transformer.h.{i}.mlp.c_fc.weight"], file)
    for i in range(L): # (L, 4C, C)
        write_fun(model_tensors[f"transformer.h.{i}.mlp.c_fc2.weight"], file)
    for i in range(L): # (L, C, 4C)
        write_fun(model_tensors[f"transformer.h.{i}.mlp.c_proj.weight"], file)
    write_fun(model_tensors["transformer.ln_f.weight"], file) # (C, )
    write_fun(model_tensors["lm_head.weight"], file) # (V, C)

def write_model(model, filename, dtype):
    # everything we need to instantiate the model
    # 1) header is: version int, LLaMAConfig ints, padding to 1024 bytes
    assert dtype in {"float32", "bfloat16"}
    version = {
        "float32": 3, # 3: all tensors are fp32
        "bfloat16": 5, # 5: all tensors are bf16
    }[dtype]
    header = torch.zeros(256, dtype=torch.int32)
    header[0] = 20240803 # magic
    header[1] = version # checkpoint version
    header[2] = model.config.block_size
    header[3] = model.config.vocab_size
    header[4] = model.config.n_layer
    header[5] = model.config.n_head
    header[6] = model.config.n_kv_head
    header[7] = model.config.n_embd
    header[8] = model.config.ffn_dim_multiplier
    header[9] = model.config.multiple_of
    header[10] = model.config.norm_eps
    header[11] = model.config.rope_theta
    header[12] = model.config.use_scaled_rope
    header[13] = model.config.max_gen_batch_size
    header[14] = int(model.config.version.split('.')[0]) # major version
    header[15] = int(model.config.version.split('.')[1]) # minor version
    # 2) the parameters follow the header
    params = {name: param.cpu() for name, param in model.named_parameters()}
    # now write to file
    with open(filename, "wb") as file:
        file.write(header.numpy().tobytes()) # header
        write_tensors(params, model.config.n_layer, file, dtype) # params
    print(f"wrote {filename}")

def write_state(model, x, y, logits, loss, filename):
    # the state is used for debugging.
    # it contains information about the input, logits, loss, and the parameter gradients
    # this can be used for checking the computation correctness in C
    header = torch.zeros(256, dtype=torch.int32)
    header[0] = 20240803 # magic
    header[1] = x.size(0) # batch size of the batch, B
    header[2] = x.size(1) # temporal extent of the batch, T
    grads = {name: param.grad.cpu() for name, param in model.named_parameters()}
    with open(filename, "wb") as file:
        # header
        file.write(header.numpy().tobytes())
        # input x
        file.write(x.cpu().numpy().astype("int32").tobytes()) # (B, T)
        # targets y
        file.write(y.cpu().numpy().astype("int32").tobytes()) # (B, T)
        # logits (result of the model forward pass)
        write_fp32(logits.cpu(), file)
        # loss (single float, result of the cross entropy loss)
        write_fp32(loss.cpu(), file)
        # gradients
        write_tensors(grads, model.config.n_layer, file, "float32")
    print(f"wrote {filename}")

# -----------------------------------------------------------------------------
# int main

def print0(*args, **kwargs):
    # modified print that only prints from the master process
    # if this is not a distributed run, it's just a print
    if int(os.environ.get("RANK", 0)) == 0:
        print(*args, **kwargs)


def checkpoint(model, model_name='model', rank=None):
    # state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    state_dict = model.state_dict()
    for k, v in state_dict.items():
        state_dict[k] = v.cpu()
    if rank is not None:
        filename = f'checkpoints/{model_name}_{rank}.pt'
    else:
        filename=f'checkpoints/{model_name}.pt'
    torch.save(state_dict, filename)


class Logger:
    def __init__(self, log_dir='logs', rank=0, model_type=None, num_iterations=None, batch_size=None, model=None):
        assert model_type is not None
        assert num_iterations is not None
        assert batch_size is not None

        self.log_dir = log_dir
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.rank = rank
        self.model = model

        self.train_log_file = f'{self.log_dir}/{model_type}_train.log'
        with open(self.train_log_file, 'w') as f:
            f.write(f'step,time,loss,norm,lr,tok/sec\n')

        self.val_log_file = f'{self.log_dir}/{model_type}_val.log'
        with open(self.val_log_file, 'w') as f:
            f.write(f'step,time,loss,perplexity\n')
        
        self.log_init_time = time.time()
        self.last_log_time = self.log_init_time

        if rank==0:
            self.is_main = True
        else:
            self.is_main = False


    def log(self, step, loss, norm, lr):
        if self.is_main:
            runtime = time.time() - self.log_init_time
            tokens_per_second = self.batch_size / (time.time() - self.last_log_time)
            self.last_log_time = time.time()
            string = f'{step},{runtime},{loss},{norm},{lr},{tokens_per_second}\n'
            with open(self.train_log_file, 'a') as f:
                f.write(string)
            print(f"step {step+1:4d}/{self.num_iterations} | train loss {loss:.6f} | norm {norm:.4f} | lr {lr:.2e} | ({(runtime)*1000:.2f} ms | {tokens_per_second:.0f} tok/s)")
        
        # checkpoint(self.model, rank=self.rank)
        if step % 1000 == 0:
            checkpoint(self.model, rank=self.rank)
        
    
    def log_val(self, step, loss, perplexity):
        if self.is_main:
            runtime = time.time() - self.log_init_time
            string = f'{step},{runtime},{loss},{perplexity}\n'
            with open(self.val_log_file, 'a') as f:
                f.write(string)
            # print0(f"val loss {val_loss}")
            print(f'val loss {loss:.6f} | perplexity {perplexity:.4f}')
            self.last_log_time = time.time()