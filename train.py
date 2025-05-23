"""
Reference code for LLaMA-3.1 training and inference.
Will save the model weights into files, to be read from C as initialization.

This code differs from GPT-2 very slightly, there are three main differences:
1) RoPE: LLaMA uses a different positional encoding scheme called Relative Positional Encoding (RoPE).
2) GQA: Grouped Query Attention (GQA) is used to reduce the number of attention heads.
3) SwiGLU: Swish-Gated Linear Unit (SwiGLU) is used as the activation function in the MLP.

References:
# 1) https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/api/tokenizer.py
# 2) https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/api/model.py
# 3) https://github.com/meta-llama/llama3/blob/11817d47e1ba7a4959b025eb1ca308572e0e3963/llama/generation.py

Example launches to only benchmark the speed of bfloat16 compiled GPU training:
TODO: add the actual commands
"""

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
    List
)
import json

import numpy as np
import torch
import torch.nn.functional as F
import torch._inductor.config as config
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist

########################################################################################
########################################################################################
# from models.model import Transformer, ModelArgs
from models.sinusoidal import SinusoidalModelArgs, SinusoidalTransformer
from models.sinusoidal_ssmax import SinusoidalSSMaxModelArgs, SinusoidalSSMaxTransformer
from models.rotary import RotaryModelArgs, RotaryTransformer
from models.rotary_ssmax import RotarySSMaxModelArgs, RotarySSMaxTransformer
from models.alibi import ALiBiModelArgs, ALiBiTransformer
from models.alibi_ssmax import ALiBiSSMaxModelArgs, ALiBiSSMaxTransformer
from models.bam import BATransformer, BATModelArgs
from models.bam_ssmax import SSMaxBATransformer, SSMaxBATModelArgs
from models.nope import NoPEModelArgs, NoPETransformer
from models.nope_ssmax import NoPESSMaxModelArgs, NoPESSMaxTransformer

from utils import print0, round_to_multiple, set_lr, compute_radam_lr, DistributedShardedDataset, StateMonitor
########################################################################################
########################################################################################


if __name__ == "__main__":
    print0(f"Running pytorch {torch.version.__version__}")

    # default settings will overfit a tiny batch of data
    # and save model weights and debug state to disk on the first iteration
    parser = argparse.ArgumentParser()
    # file system input / output
    parser.add_argument("--dataset", type=str, default="10B", help="data/ directory containing the training data")
    parser.add_argument("--log_dir", type=str, default="logs", help="output directory to which to write logs and checkpoints")
    parser.add_argument("--position_encoding", type=str, default="rotary", help="rotary|sinusoidal|alibi|bam")
    parser.add_argument("--model_size", type=str, default="l12", help="l6|l8|l12|l16|l24|l32")
    # Bayesian Attention Mechanism specific arguments
    parser.add_argument("--global_prior", type=bool, default=True, help="whether to use a global prior for BAM (True|False)")
    # parser.add_argument("--compile", action=argparse.BooleanOptionalAction, help="torch.compile the model")
    # argparse.BooleanOptionalAction but deffault to true
    # parser.add_argument("--global_prior",  
    parser.add_argument("--shape_init", type=str, default=1, help="initial shape exponent for BAM, either a string, float or a list of floats")
    parser.add_argument("--scale_init", type=str, default='slope', help="initial scale multiplier for BAM, either a string, float or a list of floats")
    parser.add_argument("--loc_init", type=str, default=0, help="initial location sum for BAM, either a string, float or a list of floats")
    parser.add_argument("--shape_trainable", type=int, default=1, help="trainable shape exponent for BAM")
    parser.add_argument("--scale_trainable", type=int, default=1, help="trainable scale multiplier for BAM")
    parser.add_argument("--loc_trainable", type=int, default=1, help="trainable location sum for BAM")
    parser.add_argument("--shape_lr", type=float, default=None, help="learning rate for shape exponent")
    parser.add_argument("--scale_lr", type=float, default=None, help="learning rate for scale multiplier")
    parser.add_argument("--loc_lr", type=float, default=None, help="learning rate for location sum")
    parser.add_argument("--no_seq_scale", action=argparse.BooleanOptionalAction, help="whether to disable the SSMax sequence scale in BAM")
    parser.add_argument("--laplace_uniform_heads", type=int, default=1, help="number of uniform heads for Laplace attention")
    # token layout for each step of the optimization
    parser.add_argument("--batch_size", type=int, default=4, help="batch size, in units of #batch dimensions")
    parser.add_argument("--sequence_length", type=int, default=64, help="sequence length")
    parser.add_argument("--min_tokens_per_step", type=int, default=None, help="minimum number of tokens per step")
    parser.add_argument("--tokens_per_step", type=int, default=None, help="exact number of tokens per step")
    parser.add_argument("--max_tokens_per_step", type=int, default=None, help="maximum number of tokens per step")
    # workload (number of steps)
    parser.add_argument("--num_iterations", type=int, default=float('inf'), help="number of iterations to run")
    parser.add_argument("--inference_only", type=bool, default=0, help="only run inference")
    # optimization
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate warmup iterations")
    parser.add_argument("--warmup_iters", type=int, default=0, help="learning rate warmup iterations")
    parser.add_argument("--learning_rate_decay_frac", type=float, default=1.0, help="learning rate warmup iterations")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="maximum gradient magnitude")
    # evaluation
    parser.add_argument("--min_val_tokens", type=int, default=None, help="minimum number of tokens for validation")
    parser.add_argument("--val_tokens", type=int, default=None, help="number of tokens for validation")
    parser.add_argument("--max_val_tokens", type=int, default=None, help="maximum number of tokens for validation")
    parser.add_argument("--val_tokens_padding", type=int, default=1_000_000, help="Number of tokens between validation and training data")
    parser.add_argument("--val_loss_every", type=int, default=0, help="every how mant steps to evaluate val loss?")
    parser.add_argument("--sample_every", type=int, default=0, help="how often to sample from the model?")
    # debugging
    parser.add_argument("--overfit_single_batch", type=bool, default=0, help="overfit just one batch of data")
    # numerics
    # parser.add_argument("--tensorcores", type=bool, default=0, help="use tensorcores")
    parser.add_argument("--tensorcores", action=argparse.BooleanOptionalAction, help="use tensorcores")
    # memory management
    parser.add_argument("--device", type=str, default="", help="by default we autodetect, or set it here")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, help="torch.compile the model")
    # parser.add_argument("--compile", type=bool, default=False, help="torch.compile the model")
    parser.add_argument("--dtype", type=str, default="float32", help="float32|float16|bfloat16")

    args = parser.parse_args()
    # print(args.global_prior)
    args.shape_trainable = bool(args.shape_trainable)
    args.scale_trainable = bool(args.scale_trainable)
    args.loc_trainable = bool(args.loc_trainable)
    print(f'Train Shape:  {args.shape_trainable}')
    print(f'Train Scale:  {args.scale_trainable}')
    print(f'Train Loc:    {args.loc_trainable}')

    # args error checking and convenience variables
    batch_size, seq_len = args.batch_size, args.sequence_length
    assert args.dtype in {"float32", "float16", "bfloat16"}
    assert args.model_size in {"l6", "l8", "l12", "l15", "l16", "l18", "l24", "l32"}
    assert args.position_encoding in {"rotary", "rotary_ssmax", "sinusoidal", "sinusoidal_ssmax", 
                                      "alibi", "alibi_ssmax", "bam", "bam_ssmax", "nope", "nope_ssmax"}
    # assert only one of min_tokens_per_step, tokens_per_step, max_tokens_per_step is set
    assert sum([args.min_tokens_per_step is not None, 
                args.tokens_per_step is not None, 
                args.max_tokens_per_step is not None]) == 1, "only one of min_tokens_per_step, tokens_per_step, max_tokens_per_step can be set"
    assert sum([args.max_val_tokens is not None, 
                args.val_tokens is not None, 
                args.min_val_tokens is not None]) <= 1, "only one of max_val_tokens, val_tokens, min_val_tokens can be set"

    if args.shape_lr is None:
        args.shape_lr = args.learning_rate
    if args.scale_lr is None:
        args.scale_lr = args.learning_rate
    if args.loc_lr is None:
        args.loc_lr = args.learning_rate

    # set up DDP (distributed data parallel). torchrun sets this env variable
    ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    if ddp:
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
        if args.device:
            # provided explicitly by the user
            device = args.device
        else:
            # attempt to autodetect the device
            device = "cpu"
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
    # ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = get_execution_vars(device=args.device)
    print(f"using device: {device}")
    device_type = 'cuda' if 'cuda' in device else 'cpu'

    # calculate gradient accumulation from the desired total batch size and the current run configuration
    
    tokens_per_fwdbwd = int(batch_size * seq_len * ddp_world_size)
    if args.min_tokens_per_step is not None:
        grad_accum_steps = round_to_multiple(args.min_tokens_per_step, multiple=tokens_per_fwdbwd, up=True) // tokens_per_fwdbwd
    elif args.tokens_per_step is not None:
        grad_accum_steps = args.tokens_per_step // tokens_per_fwdbwd
        assert args.tokens_per_step % tokens_per_fwdbwd == 0, f"tokens_per_step {args.tokens_per_step} must be a multiple of {tokens_per_fwdbwd}"
    elif args.max_tokens_per_step is not None:
        grad_accum_steps = round_to_multiple(args.max_tokens_per_step, multiple=tokens_per_fwdbwd, down=True) // tokens_per_fwdbwd
    tokens_per_batch = tokens_per_fwdbwd * grad_accum_steps
    print0(f"Tokens per Forward Pass:                       {tokens_per_fwdbwd:16,}")
    print0(f"Tokens per Batch:                              {tokens_per_batch:16,}")
    print0(f"=> calculated gradient accumulation steps:     {grad_accum_steps:16,}")

    if args.val_tokens is not None:
        val_tokens = args.val_tokens
    elif args.max_val_tokens is not None:
        val_tokens = round_to_multiple(args.max_val_tokens, multiple=tokens_per_fwdbwd, up=False)
    elif args.min_val_tokens is not None:
        val_tokens = round_to_multiple(args.min_val_tokens, multiple=tokens_per_fwdbwd, up=True)
    else:
        val_tokens = 3_932_160
    args.val_tokens = val_tokens
    print0(f"Validation Tokens:                             {val_tokens:16,}")


    # set up a context manager following the desired dtype and device
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.dtype]
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if (device_type == "cuda") else nullcontext()

    # rng / reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # set the torch precision mode to use TensorFloat32 (TF32) for matmuls
    # docs https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
    if args.tensorcores:
        torch.set_float32_matmul_precision('high')

    # -------------------------------------------------------------------------
    ModelArgs, Transformer = {
        "sinusoidal":       (SinusoidalModelArgs,   SinusoidalTransformer        ),
        "sinusoidal_ssmax": (SinusoidalSSMaxModelArgs, SinusoidalSSMaxTransformer),
        "rotary":           (RotaryModelArgs,       RotaryTransformer            ),
        "rotary_ssmax":     (RotarySSMaxModelArgs,  RotarySSMaxTransformer       ),
        "alibi":            (ALiBiModelArgs,        ALiBiTransformer             ),
        "alibi_ssmax":      (ALiBiSSMaxModelArgs,   ALiBiSSMaxTransformer        ),
        "bam":              (BATModelArgs,          BATransformer                ),
        "bam_ssmax":        (SSMaxBATModelArgs,     SSMaxBATransformer           ),
        "nope":             (NoPEModelArgs,         NoPETransformer              ),
        "nope_ssmax":       (NoPESSMaxModelArgs,    NoPESSMaxTransformer         ),
    }[args.position_encoding]

    # init the model
    model_config = {
        "l6":  ModelArgs(dim=512,  n_layers=6,  n_heads=16, ffn_dim_multiplier=2),
        "l8":  ModelArgs(dim=768,  n_layers=8,  n_heads=16, ffn_dim_multiplier=2),
        "l12": ModelArgs(dim=768,  n_layers=12, n_heads=16, ffn_dim_multiplier=2),
        "l15": ModelArgs(dim=1152, n_layers=15, n_heads=24, ffn_dim_multiplier=2),
        "l18": ModelArgs(dim=1536, n_layers=18, n_heads=32, ffn_dim_multiplier=2),
        "l24": ModelArgs(dim=2048, n_layers=24, n_heads=64, ffn_dim_multiplier=2),
    }[args.model_size]
    model_config.max_seq_len = seq_len
    model_config.max_batch_size = batch_size
    if "bam" in args.position_encoding:
        model_config.shape_init = args.shape_init
        model_config.scale_init = args.scale_init
        model_config.loc_init = args.loc_init
        model_config.train_shape = args.shape_trainable
        model_config.train_scale = args.scale_trainable
        model_config.train_loc = args.loc_trainable
        model_config.global_positional_encoding = args.global_prior
    if "ssmax" in args.position_encoding:
        model_config.seq_scale = not args.no_seq_scale
    if "laplace" in args.position_encoding:
        model_config.uniform_heads = args.laplace_uniform_heads

    model = Transformer(model_config)

    param_count = sum(p.numel() for p in model.parameters())
    embed_param_count = sum(p.numel() for p in model.tok_embeddings.parameters())
    print0()
    print0(f"Parameters:                                    {param_count:16,}")
    print0(f"Non-Embedding Parameters:                      {param_count - embed_param_count:16,}")
    print0()
    model.train()
    model = model.to(device)
    if args.compile:
        if hasattr(config, "coordinate_descent_tuning"):
            config.coordinate_descent_tuning = True # suggested by @Chillee
        print0("Compiling the Model...")
        model = torch.compile(model)

    # -------------------------------------------------------------------------
    # Our own version of a simple DistributedDataLoader

    # load tokens
    dataset = DistributedShardedDataset(args.dataset, batch_size, seq_len, ddp_rank, ddp_world_size, grad_accum_steps, 
                                        val_tokens=val_tokens, val_tokens_padding=args.val_tokens_padding)
    dataset_args = dataset.get_args()
    val_dataset = dataset.get_val_dataset(device=device)
    train_loader = torch.utils.data.DataLoader(dataset, num_workers=1, batch_size=None, prefetch_factor=4, pin_memory=True, pin_memory_device=device)
    num_iterations = min(dataset_args.iterations, args.num_iterations)


    # -------------------------------------------------------------------------
    # main training loop

    # here we wrap model into DDP container
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model # always contains the "raw" unwrapped model
    raw_model = raw_model._orig_mod if args.compile else model

    # init the optimizer
    # optimizer = torch.optim.AdamW(raw_model.parameters(), lr=args.learning_rate, 
    #                               betas=(0.9, 0.95), weight_decay=args.weight_decay,
    #                               fused=('fused' in inspect.signature(torch.optim.AdamW).parameters and device_type == 'cuda')
    # )
    # param_dict = {pn: p for pn, p in model.module.named_parameters() if p.requires_grad}
    # optim_groups = [
    #     {'params': [p for n, p in param_dict.items() if p.dim() >= 2], 'weight_decay': args.weight_decay},
    #     {'params': [p for n, p in param_dict.items() if p.dim() < 2], 'weight_decay': 0.0}
    # ]
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    optim_groups = [
        {'params': [p for n, p in param_dict.items() if p.squeeze().dim() >= 2 and 'prior' not in n], 'weight_decay': args.weight_decay,
         'init_lr': args.learning_rate, 'lr': args.learning_rate},
        {'params': [p for n, p in param_dict.items() if p.squeeze().dim() < 2  and 'prior' not in n], 'weight_decay': 0.0, 
         'init_lr': args.learning_rate, 'lr': args.learning_rate},
        {'params': [p for n, p in param_dict.items() if 'prior.shape' in n], 'weight_decay': 0.0, 'lr': args.shape_lr, 
         'init_lr': args.shape_lr, 'lr': args.shape_lr},
        {'params': [p for n, p in param_dict.items() if 'prior.scale' in n], 'weight_decay': 0.0, 'lr': args.scale_lr, 
         'init_lr': args.scale_lr, 'lr': args.scale_lr},
        {'params': [p for n, p in param_dict.items() if 'prior.loc'   in n], 'weight_decay': 0.0, 'lr': args.loc_lr,   
         'init_lr': args.loc_lr, 'lr': args.loc_lr},
    ]
    optimizer = torch.optim.RAdam(optim_groups, betas=(0.9, 0.95), weight_decay=args.weight_decay, decoupled_weight_decay=True)
    # optimizer = torch.optim.AdamW(optim_groups, betas=(0.9, 0.95), weight_decay=args.weight_decay)



    # learning rate decay scheduler (cosine with warmup)
    def get_lr(it):
        min_lr = args.learning_rate * args.learning_rate_decay_frac
        # 1) linear warmup for warmup_iters steps
        if it < args.warmup_iters:
            return args.learning_rate * (it+1) / args.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > num_iterations:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - args.warmup_iters) / (num_iterations - args.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
        return min_lr + coeff * (args.learning_rate - min_lr)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    timings = []
    norm = -1.0   # dummy value to print in inference-only mode
    # monitor0 = StateMonitor(log_dir=args.log_dir, model_type=args.model_size, rank=ddp_rank, 
    #                         num_iterations=num_iterations, val_tokens=dataset.val_tokens,
    #                         tokens_per_batch=tokens_per_batch, model=raw_model)
    
    # def __init__(self, args, dataset_args, model_args, model, rank=0):
    monitor0 = StateMonitor(args, dataset_args, model_config, raw_model, optimizer, rank=ddp_rank)
                            
    # for step in range(args.num_iterations + 1):
    for step, batches in enumerate(train_loader):
        t0 = time.time()
        last_step = (step == args.num_iterations)

        # once in a while evaluate the validation dataset
        if (args.val_loss_every > 0 \
            and (step % args.val_loss_every == 0 or last_step)) \
            and (val_dataset is not None):
            model.eval()
            # checkpoint(model, rank=ddp_rank)
            # val_dataset.reset()
            with torch.no_grad():
                val_loss = 0.0
                for input_ids, seq_codes, targets in val_dataset:
                    logits = model(input_ids, seq_codes=seq_codes)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
                    val_loss += loss.item()
                val_loss /= len(val_dataset)
            if ddp:
                val_loss = torch.tensor(val_loss).to(device)
                dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
                val_loss /= ddp_world_size
                val_loss = val_loss.item()
            # log to console and to file
            monitor0.log_val(step, val_loss)
            if not val_loss == val_loss:
                print0("NaN Loss, Stop Training")
                break

        # once in a while perform model inference on the master process
        if (args.sample_every > 0 \
            and (step % args.sample_every == 0 or last_step)) \
            and ddp_local_rank == 0:
            model.eval()
            prompts: List[str] = [
        "Clearly, the meaning of life is",
        "Simply put, the theory of relativity states that",
        """The repo llm.c on GitHub is""",
        """Translate English to French:

        sea otter => loutre de mer
        peppermint => menthe poivrée
        plush girafe => girafe peluche
        cheese =>""",
            ]
            if args.use_hf:
                prompt_tokens = [model.tokenizer(x).input_ids for x in prompts]
            else:  # Meta
                prompt_tokens = [model.tokenizer.encode(x, bos=True, eos=False) for x in prompts]

            generation_tokens = model.generate(prompt_tokens, max_gen_len=64, temperature=0.6, top_p=0.9, echo=False)
            results = [{"generation": model.tokenizer.decode(t)} for t in generation_tokens]
            for prompt, result in zip(prompts, results):
                print(prompt, end="")
                print(f"{result['generation']}")
                print("\n==================================\n")

        # bit confusing: we want to make sure to eval and sample on 0th iteration
        # but also after the very last iteration. so we loop for step <= num_iterations
        # instead of just < num_iterations (one extra due to <=), only to do
        # the validation/sampling one last time, and then we break right here as we're done.
        if last_step:
            break

        # --------------- TRAINING SECTION BEGIN -----------------
        model.train()
        optimizer.zero_grad(set_to_none=True)
        # if we are trying to overfit a single batch, we reset the loader here
        if args.overfit_single_batch:
            train_loader.reset()
        # micro-batch loop where we do gradient accumulation to reach desired total batch size
        lossf = 0.0 # for getting the mean loss (as simple float) over the accumulation steps
        for micro_step, (input_ids, seq_codes, targets) in enumerate(batches):
            input_ids, seq_codes, targets = input_ids.to(device), seq_codes.to(device), targets.to(device)
            # input_ids, targets = input_ids.to(device), targets.to(device)
            if ddp:
                # we want only the last micro-step to sync grads in a DDP model
                # the official way to do this is with model.no_sync(), but that is a
                # context manager that bloats the code, so we just toggle this variable
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
            # forward pass
            with ctx:
                logits = model(input_ids, seq_codes=seq_codes)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
                # we have to scale the loss to account for gradient accumulation,
                # because the gradients just add on each successive backward().
                # addition of gradients corresponds to a SUM in the objective, but
                # instead of a SUM we want MEAN, so we scale the loss here
                loss = loss / grad_accum_steps
                lossf += loss.detach() # keep track of the mean loss
            # backward pass
            if not args.inference_only:
                loss.backward()
        if ddp:
            dist.all_reduce(lossf, op=dist.ReduceOp.AVG)
        lossf = lossf.item()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        # # determine and set the learning rate for this iteration
        optimizer = set_lr(optimizer, step, num_iterations, args)
        # lr = get_lr(step)
        # all_lrs = set()
        # for param_group in optimizer.param_groups:
        #     # param_group['lr'] = lr
        #     # print(param_group['lr'], end='\t')
        #     all_lrs.add(param_group['lr'])
        # print(all_lrs)

        
        # step the optimizer
        # if lossf == lossf: # check for NaN
        if lossf == lossf and norm == norm: # check for NaN
            optimizer.step()
        else:
            print0("NaN Loss, Stop Training")
            break

        # --------------- TRAINING SECTION END -------------------
        # everything that follows now is just diagnostics, prints, logging, etc.

        # wait on the CPU for all device work to end so we get accurate per-iteration timings below
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()

        # log
        monitor0.log(step, lossf, norm, compute_radam_lr(optimizer))
        
    monitor0.save_model()
    monitor0.max_memory()

    # once in a while evaluate the validation dataset
    if (args.val_loss_every > 0) and (val_dataset is not None):
        model.eval()
        # checkpoint(model, rank=ddp_rank)
        # val_dataset.reset()
        with torch.no_grad():
            val_loss = 0.0
            for input_ids, seq_codes, targets in val_dataset:
                logits = model(input_ids, seq_codes=seq_codes)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
                val_loss += loss.item()
            val_loss /= len(val_dataset)
            if ddp:
                val_loss = torch.tensor(val_loss).to(device)
                dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
                val_loss /= ddp_world_size
                val_loss = val_loss.item()
        # log to console and to file
        monitor0.log_val(step+1, val_loss)


    # -------------------------------------------------------------------------
    # clean up nice
    if ddp:
        destroy_process_group()
