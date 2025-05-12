# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class BATModelArgs:
    dim: int = 1024
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = 32768 
    multiple_of: int = 1  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    max_batch_size: int = 32
    max_seq_len: int = 1024

    # shape_init: float | str = 'linear'
    # scale_init: float | str = 1/16
    shape_init: float | str = 1
    scale_init: float | str = 'slope'
    loc_init:   float = 0

    train_shape: bool = True
    train_scale: bool = True
    train_loc:   bool = False

    global_positional_encoding: bool = True

# loc = exp(loc) - exp(-loc)

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class AttentionPrior(nn.Module):
    def __init__(self, args: BATModelArgs):
        super().__init__()
        self.seq_len = args.max_seq_len
        self.n_heads = args.n_heads
        self.eps = 1e-5

        
        if args.scale_init == 'slope':
            scale = torch.tensor(get_slopes(args.n_heads), dtype=torch.float).reshape(1, args.n_heads, 1, 1)
        else:
            scale = torch.full((1, args.n_heads, 1, 1), float(args.scale_init), dtype=torch.float)
        scale = torch.log(scale)
        
        if args.train_shape and args.shape_init == 'linear':
            shape  = torch.linspace(0, 1, args.n_heads, dtype=torch.float).reshape(1, args.n_heads, 1, 1)
        elif args.train_shape:
            shape   = torch.full((1, args.n_heads, 1, 1), float(args.shape_init), dtype=torch.float)
        else:
            shape   = torch.ones((1, args.n_heads, 1, 1), dtype=torch.float)

        loc     = torch.full((1, args.n_heads, 1, 1), float(args.loc_init),   dtype=torch.float)
        
        self.shape = nn.Parameter(shape, requires_grad = args.train_shape)
        self.scale = nn.Parameter(scale, requires_grad = args.train_scale)
        self.loc   = nn.Parameter(loc,   requires_grad = args.train_loc)
        print0('Shape:', shape.flatten())
        print0('Scale:', scale.flatten())
        print0('Loc:', loc.flatten()) 


        # positions = torch.arange(self.seq_len).float()
        # self.register_buffer("dist_matrix", 
        #                      (positions[None, :] - positions[:, None]).reshape(1, 1, self.seq_len, self.seq_len), 
        #                      persistent=False)
        # self.dist_matrix = (positions[None, :] - positions[:, None]).reshape(1, 1, self.seq_len, self.seq_len)


    def forward(self, seq_len=None):
        # if seq_len == self.dist_matrix.shape[-1] or seq_len is None:
        #     dist_matrix = self.dist_matrix.to(self.scale.device)
        # elif seq_len < self.dist_matrix.shape[-1]:
        #     print('FUUUUCK')
        #     dist_matrix = self.dist_matrix[..., :seq_len, :seq_len].to(self.scale.device)
        # else:
        #     print('FUUUUCK2')
        #     positions = torch.arange(seq_len).float().to(self.scale.device)
        #     dist_matrix = (positions[None, :] - positions[:, None]).reshape(1, 1, seq_len, seq_len)

        # positions = torch.arange(seq_len).float().to(self.scale.device)
        seq_len = seq_len or self.seq_len
        positions = torch.arange(seq_len, device=self.scale.device).float()
        # dist_matrix = (positions[None, :] - positions[:, None]).reshape(1, 1, seq_len, seq_len)
        # return -(dist_matrix.abs() * self.scale).abs()
        # return -(dist_matrix * self.scale + self.loc).abs()
        # loc = self.loc.exp() - (-self.loc).exp()
        # z = (dist_matrix - loc) * self.scale.exp()
        # return -((z.abs()+self.eps)**self.shape )
        b = (positions[None, :] - positions[:, None]).reshape(1, 1, seq_len, seq_len)
        b = b - (self.loc.exp() - (-self.loc).exp())
        return -((b.abs() + self.eps) ** self.shape) * self.scale.exp() 
    

def get_slopes(n):
    def get_slopes_power_of_2(n):
        start = (2**(-2**-(math.log2(n)-3)))
        ratio = start
        return [start*ratio**i for i in range(n)]
    
    if math.log2(n).is_integer():
        return get_slopes_power_of_2(n)              #In the paper, we only train models that have 2^a heads for some a. This function has
    else:                                                 #some good properties that only occur when the input is a power of 2. To maintain that even
        closest_power_of_2 = 2**math.floor(math.log2(n))  #when the number of heads is not a power of 2, we use this workaround. 
        return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class BayesianAttention(nn.Module):
    def __init__(self, args: BATModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.local_positional_encoding = not args.global_positional_encoding
        if self.local_positional_encoding:
            self.prior = AttentionPrior(args)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        bsz, seqlen, _ = x.shape
        queries, keys, values = self.wq(x), self.wk(x), self.wv(x)

        queries = queries.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        keys = keys.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        values = values.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        queries = queries.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        values = values.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        scores = torch.matmul(queries, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if self.local_positional_encoding:
            scores = scores + self.prior(seqlen)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        scores = F.softmax(scores.float(), dim=-1).type_as(queries)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: BATModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = BayesianAttention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        h = x + self.attention(self.attention_norm(x), mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

import os
def print0(*args, **kwargs):
    # modified print that only prints from the master process
    # if this is not a distributed run, it's just a print
    if int(os.environ.get("RANK", 0)) == 0:
        print(*args, **kwargs)

class BATransformer(nn.Module):
    def __init__(self, params: BATModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.global_positional_encoding = params.global_positional_encoding

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        if self.params.global_positional_encoding:
            self.prior = AttentionPrior(params)
        # self.slopes = torch.tensor(get_slopes(params.n_heads)).reshape(1, params.n_heads, 1, 1)
        # self.register_buffer("slopes", torch.tensor(get_slopes(params.n_heads)).reshape(1, params.n_heads, 1, 1))

    def forward(self, tokens: torch.Tensor, seq_codes: Optional[torch.Tensor] = None):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)

        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)

            mask = torch.triu(mask, diagonal=1)

            if seq_codes is not None:
                mask = mask.unsqueeze(0).repeat(_bsz, 1, 1)
                section_mask = seq_codes.unsqueeze(-1) != seq_codes.unsqueeze(-2)
                mask[section_mask] = float("-inf")
                mask = mask.unsqueeze(-3)

            if self.global_positional_encoding:
                # print0()
                # print0(self.prior(seqlen).mean())
                # print0(self.prior(seqlen)[0,0,0])
                # print0()
                mask = mask + self.prior(seqlen)
            # positions = torch.arange(seqlen, device=tokens.device).float()
            # position_encodings = -(positions[None, :] - positions[:, None]).abs() * self.slopes
            # mask = mask + position_encodings

            mask = mask.type_as(h)

        for layer in self.layers:
            h = layer(h, mask)
        h = self.norm(h)
        output = self.output(h).float()
        return output
