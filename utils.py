
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

import torch.nn.functional as F
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
    def __init__(self, version, step_token_count, samples_per_fwd, rank=0, world_size=1, shard_size=1024, max_length=1024,
                 tokenizer="mistralai/Mistral-7B-Instruct-v0.3", val_samples=1536):
        self.rank               = rank
        self.world_size         = world_size
        self.shard_size         = shard_size
        self.max_length         = max_length
        self.samples_per_fwd    = samples_per_fwd
        # self.step_token_count   = step_token_count 
        self.step_token_count   = step_token_count / self.world_size
        self.val_samples        = val_samples

        tokenizer = "openai-community/gpt2"
        self.val_samples        = 1280


        assert self.val_samples % (self.world_size * self.samples_per_fwd) == 0, f'Val samples must be divisible by world_size * samples_per_fwd: {self.val_samples} % {self.world_size} * {self.samples_per_fwd}'
        assert self.shard_size % self.samples_per_fwd == 0,     f'Shard size must be divisible by  samples_per_fwd: {self.shard_size} % {self.samples_per_fwd}'

        self.tokenizer              = AutoTokenizer.from_pretrained(tokenizer)
        self.tokenizer.pad_token    = self.tokenizer.unk_token
        self.dataset                = load_dataset("HuggingFaceFW/fineweb", name=self.versions[version])['train']

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
        return self.tokenizer.pad(tokens, return_tensors='pt', padding='max_length', max_length=self.max_length)
    
    def get_val_batches(self):
        val_shard_size = int(self.val_samples // self.world_size)
        start = self.rank * val_shard_size
        end = start + val_shard_size
        # val_data = self.dataset[start:end]
        tokens = self.tokenize_shard(start, end)
        input_ids = tokens['input_ids']

        val_token_count = torch.tensor(sum(tokens['length']))
        if self.is_ddp:
            val_token_count = val_token_count.cuda(self.rank)
            dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
            val_token_count = val_token_count.cpu()
        print0(f"Val data token count: {val_token_count.item():,}")

        val_batches = []
        for i in range(0, len(input_ids), self.samples_per_fwd):
            targets = [input_id[1:]+[self.tokenizer.eos_token_id] for input_id in input_ids[i:i+self.samples_per_fwd]]
            targets = self.pad_token_ids(targets)['input_ids']
            # one_hot_targets = F.one_hot(targets['input_ids'], num_classes=self.tokenizer.vocab_size)
            # one_hot_targets[..., self.tokenizer.pad_token_id] = 0

            inputs = self.pad_token_ids(input_ids[i:i+self.samples_per_fwd])

            val_batches.append((inputs, targets))
        return val_batches, val_token_count.item()
        

    def __iter__(self):
        self.stop = torch.tensor([0]).cuda(self.rank) if self.is_ddp else torch.tensor([0])
        self.curr_pos = self.rank * self.shard_size + self.val_samples
        self.shard_pos = 0
        self.current_shard = self.tokenize_shard(self.curr_pos, self.curr_pos+self.shard_size)
        return self
    
    def __next__(self):
        token_count = 0
        # lengths = self.current_shard['length'][self.shard_pos:]
        batched_tokens = []
        while token_count < self.step_token_count:
            if self.shard_pos >= self.shard_size:
                self.next_shard()
            
            # if self.is_ddp:
            #     dist.all_reduce(self.stop, op=dist.ReduceOp.MAX)
            if self.stop[0]:
                raise StopIteration
            
            batched_tokens.append(self.current_shard['input_ids'][self.shard_pos:self.shard_pos+self.samples_per_fwd])
            
            # Syncronize the token_count
            current_token_count = torch.tensor(self.current_shard['length'][self.shard_pos:self.shard_pos+self.samples_per_fwd]).sum()
            # print(self.current_shard['length'][self.shard_pos:self.shard_pos+self.samples_per_fwd].sum())
            if self.is_ddp:
                current_token_count = current_token_count.cuda(self.rank)
                dist.barrier()
                dist.all_reduce(current_token_count, op=dist.ReduceOp.SUM)
                # dist.all_reduce(current_token_count, op=dist.ReduceOp.SUM, async_op=False)
                current_token_count = current_token_count.cpu()
            token_count += current_token_count.item()
            # token_count += self.current_shard['length'][self.shard_pos:self.shard_pos+self.samples_per_fwd]
            self.shard_pos += self.samples_per_fwd

        batches = []
        for i in range(len(batched_tokens)):
            targets = [input_id[1:]+[self.tokenizer.eos_token_id] for input_id in batched_tokens[i]]
            targets = self.pad_token_ids(targets)['input_ids']
            # one_hot_targets = F.one_hot(targets['input_ids'], num_classes=self.tokenizer.vocab_size)
            # one_hot_targets[..., self.tokenizer.pad_token_id] = 0

            inputs = self.pad_token_ids(batched_tokens[i])

            batches.append((inputs, targets))
            
        return batches, token_count
        

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
            f.write(f'step,time,loss,norm,lr,tokens,runtime,tok/sec\n')

        self.val_log_file = f'{self.log_dir}/{model_type}_val.log'
        with open(self.val_log_file, 'w') as f:
            f.write(f'step,time,loss,perplexity\n')
        
        self.log_init_time = time.time()
        self.last_log_time = self.log_init_time

        if rank==0:
            self.is_main = True
        else:
            self.is_main = False
        self.is_ddp = torch.distributed.is_initialized()


    def log(self, step, loss, norm, lr, token_count):
        if self.is_ddp:
            token_count = torch.tensor(token_count).cuda(self.rank)
            loss = torch.tensor(loss).cuda(self.rank)
            # dist.barrier()
            # dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            token_count = token_count.cpu().item()
            loss = loss.cpu().item()/token_count

        if self.is_main:
            runtime = time.time() - self.last_log_time
            step_runtime = (time.time() - self.last_log_time)
            tokens_per_second = token_count / step_runtime
            self.last_log_time = time.time()
            string = f'{step},{runtime},{loss},{norm},{lr},{token_count},{step_runtime},{tokens_per_second}\n'
            with open(self.train_log_file, 'a') as f:
                f.write(string)
            # print(f"step {step+1:4d}/{self.num_iterations} | train loss {loss:.6f} | norm {norm:.4f} | lr {lr:.2e} | ({(runtime):.2f} s | {tokens_per_second:.0f} tok/s)")
            print(f"step {step+1:4d}/{self.num_iterations} | train loss {loss:.6f} | norm {norm:7.4f} | lr {lr:.2e} | ({token_count:,} tokens | {(step_runtime):.2f} s | {tokens_per_second:,.0f} tok/s)")
        
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

def perplexity(logits, targets):
    return torch.exp(F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=0))