
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

class DistributedShardedDataset(IterableDataset):
    """
    This DataLoader is both:
    - distributed (works correctly in case of multiple processes in DDP)
    - sharded (supports datasets that are broken up into multiple data shards)
    It is not *permuted*, meaning that it itearates over the data in the order
    of the dataset on disk, so the user should make sure to shuffle their examples
    during the creation of their data shards for best performance.
    """
    def __init__(self, dataset_dir, batch_size, seq_len, process_rank, num_processes, 
                 grad_accum_steps, min_val_tokens=0, val_tokens_padding=10**6):
        self.process_rank       = process_rank
        self.num_processes      = num_processes
        self.batch_size         = batch_size
        self.seq_len            = seq_len
        self.tokens_per_batch   = batch_size * seq_len * num_processes
        self.tokens_per_fwd     = batch_size * seq_len
        self.grad_accum_steps   = grad_accum_steps


        self.val_tokens_padding = val_tokens_padding if min_val_tokens > 0 else 0
        self.val_tokens = round_to_multiple(min_val_tokens, multiple=self.tokens_per_batch, up=True)
        print0(f'Valiation tokens:                              {self.val_tokens:16,}')
        print0(f'Valiation tokens padding:                      {self.val_tokens_padding:16,}')


        # glob files that match the pattern
        self.files = sorted(os.listdir(f"data/{dataset_dir}"))
        self.files = [f"data/{dataset_dir}/{f}" for f in self.files]
        # self.files = ['data/10B/sample_000000.pt', 'data/10B/sample_000001.pt']
        assert len(self.files) > 0, f"did not find any files that match the pattern {dataset_dir}"

        # load and validate all data shards, count number of tokens in total
        print0(f"Dataset: loading {len(self.files)} files")
        ntok_total = 0
        for fname in self.files:
            shard_ntok = self.load_data_shard(fname)[0].size(0)
            assert shard_ntok > num_processes * batch_size * seq_len
            ntok_total += shard_ntok
        self.ntok_total = ntok_total
        print0(f"Dataset: total number of tokens:               {ntok_total:16,}")
        print0(f"                                        across {len(self.files):16,} files")

        # kick things off
        self.current_shard      = None
        self.global_position    = None
        self.world_position     = None
        self.reset()

    def reset(self):
        if self.current_shard != 0:
            self.current_shard   = 0
            self.global_position = 0
            self.input_ids, self.seq_codes = self.load_data_shard(self.files[self.current_shard])
        self.world_position    = self.val_tokens_padding + self.val_tokens
        self.current_position  = self.process_rank * self.batch_size * self.seq_len
        self.current_position += self.val_tokens_padding + self.val_tokens
    
    def load_data_shard(self, filename):
        dataset = torch.load(filename)
        return dataset['input_ids'], dataset['seq_codes']
    
    def __iter__(self):
        self.reset()
        return self
    
    def __next__(self):
        # if loading the next batch would be out of bounds load shards until we have enough data
        self.world_position += self.tokens_per_batch*self.grad_accum_steps
        while self.world_position + 1 > len(self.input_ids):
            self.load_next_shard()

        batches = torch.empty((self.grad_accum_steps, 3, self.batch_size, self.seq_len), dtype=torch.int64)
        for i in range(self.grad_accum_steps):
            # # if loading the next batch would be out of bounds load shards until we have enough data
            # while self.current_position + self.tokens_per_fwd + 1 > len(self.input_ids):
            #     self.load_next_shard()
            batches[i] = self.get_batch(self.current_position, self.current_position+self.tokens_per_fwd)
            # advance the start pointer in current shard
            self.current_position += self.tokens_per_batch
        return batches
    
    def load_next_shard(self):
        self.current_shard = (self.current_shard + 1)
        if self.current_shard >= len(self.files):
            raise StopIteration
        self.current_position = self.process_rank * self.tokens_per_fwd
        self.world_position   = self.tokens_per_batch*self.grad_accum_steps

        remainder_pos = round_to_multiple(len(self.input_ids), multiple=self.tokens_per_batch*self.grad_accum_steps, up=False)
        self.input_ids = self.input_ids[remainder_pos:]
        self.seq_codes = self.seq_codes[remainder_pos:]

        new_input_ids, new_seq_codes = self.load_data_shard(self.files[self.current_shard])
        self.input_ids = torch.cat([self.input_ids, new_input_ids], dim=0)
        self.seq_codes = torch.cat([self.seq_codes, new_seq_codes], dim=0)

        self.global_position += new_input_ids.size(0)

    def get_val_dataset(self, device='cpu'):
        if self.val_tokens == 0:
            return None
        else:
            local_val_batches_count = self.val_tokens // self.tokens_per_batch
            val_batches = torch.empty((local_val_batches_count, 3, self.batch_size, self.seq_len), dtype=torch.int64)
            pos = self.process_rank * self.tokens_per_fwd
            for i in range(local_val_batches_count):
                val_batches[i] = self.get_batch(pos, pos+self.batch_size*self.seq_len)
                pos += self.tokens_per_batch
            return val_batches.to(device)
    

    def get_batch(self, start, end):
        input_ids = self.input_ids[start:end+1].long()
        seq_codes = self.seq_codes[start:end].long()

        input       = input_ids[:-1].view(self.batch_size, self.seq_len) 
        targets     = input_ids[1:].view(self.batch_size, self.seq_len)
        seq_codes   = seq_codes.view(self.batch_size, self.seq_len)

        # return input, seq_codes, targets
        return torch.stack([input, seq_codes, targets], dim=0)
    


# -----------------------------------------------------------------------------
# int main

def print0(*args, **kwargs):
    # modified print that only prints from the master process
    # if this is not a distributed run, it's just a print
    if int(os.environ.get("RANK", 0)) == 0:
        print(*args, **kwargs)

def round_to_multiple(x, multiple=1, up=False):
    if x % multiple == 0:
        return x
    return x + up*multiple - (x%multiple)


def checkpoint(model, model_name='model', rank=None):
    # state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    state_dict = model.state_dict()
    for k, v in state_dict.items():
        state_dict[k] = v.cpu()
    if rank is not None:
        filename = f'checkpoints/{model_name}_{rank}_6.pt'
    else:
        filename=f'checkpoints/{model_name}_6.pt'
    torch.save(state_dict, filename)

def get_execution_vars(device=None):
    is_ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    if is_ddp:
        # use of DDP atm demands CUDA, we set the device appropriately according to rank
        assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        # select the device
        if device is None:
            # attempt to autodetect the device
            device = "cpu"
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
    return is_ddp, ddp_rank, ddp_local_rank, ddp_world_size, device


class StateMonitor:
    def __init__(self, log_dir='logs', rank=0, model_type=None, num_iterations=None, 
                 tokens_per_batch=None, model=None, val_tokens=0):
        assert model_type is not None
        assert num_iterations is not None
        assert tokens_per_batch is not None

        # create the logging directory if it does not exist
        os.makedirs(log_dir, exist_ok=True)
        logfile = os.path.join(log_dir, "main.log")
        # create the log file "main.log" inside it, and wipe it clean
        with open(logfile, "w") as f:
            pass

        self.log_dir = log_dir
        self.num_iterations = num_iterations
        self.tokens_per_batch = tokens_per_batch
        
        self.rank = rank
        self.val_tokens = val_tokens
        self.model = model

        self.train_log_file = f'{self.log_dir}/{model_type}_train.log'
        with open(self.train_log_file, 'w') as f:
            f.write(f'step,time,loss,norm,lr,exec_time,tok/sec\n')

        self.val_log_file = f'{self.log_dir}/{model_type}_val.log'
        with open(self.val_log_file, 'w') as f:
            f.write(f'step,time,loss,exec_time,tok/sec\n')
        
        self.log_init_time = time.time()
        self.last_log_time = self.log_init_time

        if rank==0:
            self.is_main = True
        else:
            self.is_main = False


    def log(self, step, loss, norm, lr):
        if self.is_main:
            runtime = time.time() - self.log_init_time
            exec_time = time.time() - self.last_log_time
            tokens_per_second = self.tokens_per_batch / exec_time
            self.last_log_time = time.time()
            string = f'{step},{runtime},{loss},{norm},{lr},{exec_time},{tokens_per_second}\n'
            with open(self.train_log_file, 'a') as f:
                f.write(string)
            print(f"step {step+1:4d}/{self.num_iterations} | train loss {loss:.6f} | norm {norm:.4f} | lr {lr:.2e} | ({(exec_time):.2f} s | {tokens_per_second:.0f} tok/s)")
        
        # checkpoint(self.model, rank=self.rank)
        # if step % 1000 == 0:
        #     checkpoint(self.model, rank=self.rank)
        
    
    def log_val(self, step, loss):
        if self.is_main:
            runtime = time.time() - self.log_init_time
            exec_time = time.time() - self.last_log_time
            tokens_per_second = self.val_tokens / exec_time
            self.last_log_time = time.time()
            string = f'{step},{runtime},{loss},{exec_time},{tokens_per_second}\n'
            with open(self.val_log_file, 'a') as f:
                f.write(string)
            # print0(f"val loss {val_loss}")
            print(f'val loss {loss:.6f} | ({exec_time:.1f}{(runtime):.1f} s | {tokens_per_second:.0f} tok/s)')
            checkpoint(self.model, rank=self.rank)