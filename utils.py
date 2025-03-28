
import os
import json
from dataclasses import dataclass
import time
import math

import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset
from transformers import AutoTokenizer

# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

@dataclass 
class DatasetArgs:
        dataset_dir: str
        batch_size: int
        seq_len: int
        grad_accum_steps: int
        val_tokens: int
        tokens_per_batch: int
        val_tokens_padding: int
        total_tokens: int
        iterations: int
        num_processes: int
    

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
                 grad_accum_steps, val_tokens=0, val_tokens_padding=10**6):
        self.dataset_dir            = dataset_dir
        self.process_rank           = process_rank
        self.num_processes          = num_processes
        self.batch_size             = batch_size
        self.seq_len                = seq_len
        self.tokens_per_world_fwd   = batch_size * seq_len * num_processes
        self.tokens_per_batch       = batch_size * seq_len * num_processes * grad_accum_steps
        self.tokens_per_fwd         = batch_size * seq_len
        self.grad_accum_steps       = grad_accum_steps
        self.val_tokens             = val_tokens


        self.val_tokens_padding = val_tokens_padding if self.val_tokens > 0 else 0
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
        print0(f"Dataset: number of shards:                     {len(self.files):16,}")
        
        self.num_iterations = (ntok_total - self.val_tokens - self.val_tokens_padding) // self.tokens_per_batch
        print0(f"Dataset: number of iterations:                 {self.num_iterations:16,}")

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
        self.world_position += self.tokens_per_world_fwd*self.grad_accum_steps
        while self.world_position + 1 > len(self.input_ids):
            self.load_next_shard()

        batches = torch.empty((self.grad_accum_steps, 3, self.batch_size, self.seq_len), dtype=torch.int64)
        for i in range(self.grad_accum_steps):
            # # if loading the next batch would be out of bounds load shards until we have enough data
            # while self.current_position + self.tokens_per_fwd + 1 > len(self.input_ids):
            #     self.load_next_shard()
            batches[i] = self.get_batch(self.current_position, self.current_position+self.tokens_per_fwd)
            # advance the start pointer in current shard
            self.current_position += self.tokens_per_world_fwd
        return batches
    
    def load_next_shard(self):
        self.current_shard = (self.current_shard + 1)
        if self.current_shard >= len(self.files):
            print(f'Rank {self.process_rank} has finished the dataset')
            raise StopIteration
        self.current_position = self.process_rank * self.tokens_per_fwd
        self.world_position   = self.tokens_per_batch

        remainder_pos = round_to_multiple(len(self.input_ids), multiple=self.tokens_per_world_fwd*self.grad_accum_steps, up=False)
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
            local_val_batches_count = self.val_tokens // self.tokens_per_world_fwd
            val_batches = torch.empty((local_val_batches_count, 3, self.batch_size, self.seq_len), dtype=torch.int64)
            pos = self.process_rank * self.tokens_per_fwd
            for i in range(local_val_batches_count):
                val_batches[i] = self.get_batch(pos, pos+self.batch_size*self.seq_len)
                pos += self.tokens_per_world_fwd
            return val_batches.to(device)
    

    def get_batch(self, start, end):
        input_ids = self.input_ids[start:end+1].long()
        seq_codes = self.seq_codes[start:end].long()

        input       = input_ids[:-1].view(self.batch_size, self.seq_len) 
        targets     = input_ids[1:].view(self.batch_size, self.seq_len)
        seq_codes   = seq_codes.view(self.batch_size, self.seq_len)

        # return input, seq_codes, targets
        return torch.stack([input, seq_codes, targets], dim=0)
    
    def get_args(self):
        args = DatasetArgs(
            dataset_dir=self.dataset_dir,
            batch_size=self.batch_size,
            seq_len=self.seq_len,
            grad_accum_steps=self.grad_accum_steps,
            val_tokens=self.val_tokens,
            tokens_per_batch=self.tokens_per_batch,
            val_tokens_padding=self.val_tokens_padding,
            total_tokens=self.ntok_total,
            iterations=self.num_iterations,
            num_processes=self.num_processes,
        )
        return args


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


# learning rate decay scheduler (cosine with warmup)
def set_lr(optimizer, it, num_iterations, args):
    for param_group in optimizer.param_groups:
        min_lr = param_group['init_lr'] * args.learning_rate_decay_frac
        # 1) linear warmup for warmup_iters steps
        if it < args.warmup_iters:
            param_group['lr'] = param_group['init_lr'] * (it+1) / args.warmup_iters
            continue
        # 2) if it > lr_decay_iters, return min learning rate
        if it > num_iterations:
            param_group['lr'] = min_lr
            continue
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - args.warmup_iters) / (num_iterations - args.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
        param_group['lr'] = min_lr + coeff * (param_group['init_lr'] - min_lr)
    return optimizer

def compute_radam_lr(radam_optimizer):
    def _compute_rect(rho_t, rho_inf):
        return (
            (rho_t - 4)
            * (rho_t - 2)
            * rho_inf
            / ((rho_inf - 4) * (rho_inf - 2) * rho_t)
        ) ** 0.5


    step = radam_optimizer.state[radam_optimizer.param_groups[0]['params'][0]]['step'].item()
    beta2 = radam_optimizer.param_groups[0]['betas'][1]
    bias_correction2 = 1 - beta2**step

    # maximum length of the approximated SMA
    rho_inf = 2 / (1 - beta2) - 1
    # compute the length of the approximated SMA
    rho_t = rho_inf - 2 * step * (beta2**step) / bias_correction2

    if rho_t > 5:
        return _compute_rect(rho_t, rho_inf)*radam_optimizer.param_groups[0]['lr']
    else:
        return None



# # learning rate decay scheduler (cosine with warmup)
# def get_lr(it):
#     min_lr = args.learning_rate * args.learning_rate_decay_frac
#     # 1) linear warmup for warmup_iters steps
#     if it < args.warmup_iters:
#         return args.learning_rate * (it+1) / args.warmup_iters
#     # 2) if it > lr_decay_iters, return min learning rate
#     if it > num_iterations:
#         return min_lr
#     # 3) in between, use cosine decay down to min learning rate
#     decay_ratio = (it - args.warmup_iters) / (num_iterations - args.warmup_iters)
#     assert 0 <= decay_ratio <= 1
#     coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
#     return min_lr + coeff * (args.learning_rate - min_lr)
class StateMonitor:
    # def __init__(self, log_dir='logs', rank=0, model_type=None, model_pos_enc=None, num_iterations=None, 
    #              tokens_per_batch=None, model=None, val_tokens=0):

    def __init__(self, args, dataset_args, model_args, model, optimizer, rank=0):
        self.num_iterations = dataset_args.iterations
        self.tokens_per_batch = dataset_args.tokens_per_batch
        self.rank = rank
        self.val_tokens = dataset_args.val_tokens
        self.model = model
        self.optimizer = optimizer

        if rank==0:
            self.is_main = True
        else:
            self.is_main = False

        if self.is_main:
            # create the logging directory if it does not exist
            # log_dir = os.path.join(log_dir, model_pos_enc, model_type)
            log_dir = os.path.join(args.log_dir, args.model_size, args.position_encoding)
            os.makedirs(log_dir, exist_ok=True)
            for i in range(100):
                if os.path.exists(os.path.join(log_dir, f'version_{i:02d}')):
                    continue
                else:
                    os.makedirs(os.path.join(log_dir, f'version_{i:02d}'))
                    log_dir = os.path.join(log_dir, f'version_{i:02d}')
                    break
            else:
                raise Exception('Too many versions')
            self.log_dir = log_dir

            # self.train_log_file = f'{self.log_dir}/{model_type}_train.log'
            # with open(self.train_log_file, 'w') as f:
            #     f.write(f'step,time,loss,norm,lr,exec_time,tok/sec\n')

            # self.val_log_file = f'{self.log_dir}/{model_type}_val.log'
            # with open(self.val_log_file, 'w') as f:
            #     f.write(f'step,time,loss,exec_time,tok/sec\n')
            self.train_log_file = os.path.join(self.log_dir, f'train.log')
            self.eval_log_file = os.path.join(self.log_dir, f'eval.log')

            # with open(os.path.join(self.log_dir, 'args.log'), 'w') as f:
            #     f.write(f'arg,value\n')
            #     for args in [args, dataset_args, model_args]:
            #         for k, v in vars(args).items():
            #             f.write(f'{k},{v}\n')
            args_dict = {
                'args': vars(args),
                'dataset_args': vars(dataset_args),
                'model_args': vars(model_args)
            }
            with open(os.path.join(self.log_dir, 'args.json'), 'w') as f:
                json.dump(args_dict, f, indent=4)
            # with open(os.path.join(self.log_dir, 'model_args.json'), 'w') as f:
            #     json.dump(model_args.__dict__, f, indent=4)

            with open(self.train_log_file, 'w') as f:
                f.write(f'step,time,loss,norm,lr,exec_time,tok/sec\n')
            with open(self.eval_log_file, 'w') as f:
                f.write(f'step,time,loss,exec_time,tok/sec\n')

            
            self.log_init_time = time.time()
            self.last_log_time = self.log_init_time


    def log(self, step, loss, norm, lr):
        if self.is_main:
            runtime = time.time() - self.log_init_time
            exec_time = time.time() - self.last_log_time
            tokens_per_second = self.tokens_per_batch / exec_time
            self.last_log_time = time.time()
            string = f'{step},{runtime},{loss},{norm},{lr},{exec_time},{tokens_per_second}\n'
            with open(self.train_log_file, 'a') as f:
                f.write(string)
                
            lr = lr if lr is not None else 0
            print(f"step {step+1:4d}/{self.num_iterations} | train loss {loss:.6f} | norm {norm:.4f} | lr {lr:.2e} | ({(exec_time):.2f} s | {tokens_per_second:.0f} tok/s)")
        
            if step % 1000 == 0:
                # checkpoint(self.model, rank=self.rank)
                self.save_model()
                self.save_optimizer()
        
    
    def log_val(self, step, loss):
        if self.is_main:
            runtime = time.time() - self.log_init_time
            exec_time = time.time() - self.last_log_time
            tokens_per_second = self.val_tokens / exec_time
            self.last_log_time = time.time()
            string = f'{step},{runtime},{loss},{exec_time},{tokens_per_second}\n'
            with open(self.eval_log_file, 'a') as f:
                f.write(string)
            # print0(f"val loss {val_loss}")
            print(f'val loss {loss:.6f} | ({exec_time:.1f}{(runtime):.1f} s | {tokens_per_second:.0f} tok/s)')
            # checkpoint(self.model, rank=self.rank)
    

    def save_model(self):
        if self.is_main:
            # state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
            state_dict = self.model.state_dict()
            for k, v in state_dict.items():
                state_dict[k] = v.cpu()
            filename = os.path.join(self.log_dir, 'model.pt')
            torch.save(state_dict, filename)
    
    def save_optimizer(self):
        if self.is_main:
            filename = os.path.join(self.log_dir, 'optimizer.pt')
            torch.save(self.optimizer.state_dict(), filename)

    def max_memory(self):
        if self.is_main:
            mem = torch.cuda.max_memory_allocated() // 1024 / 1024
            print0(f"peak memory consumption:                       {mem:16,.2f} MiB")
            