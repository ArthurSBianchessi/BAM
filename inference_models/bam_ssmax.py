import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class SSMaxBATModelArgs:
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

    shape_init: float | str = 0
    scale_init: float | str = 5
    loc_init:   float = 0

    train_shape: bool = True
    train_scale: bool = True
    train_loc:   bool = False

    global_positional_encoding: bool = True
    seq_scale: bool = True
    ssmax_prior: bool = True

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
    def __init__(self, args: SSMaxBATModelArgs):
        super().__init__()
        self.seq_len = args.max_seq_len
        self.n_heads = args.n_heads
        self.eps = 1e-5

        
        if args.scale_init == 'slope':
            scale = torch.tensor(get_slopes(args.n_heads), dtype=torch.float).reshape(1, args.n_heads, 1, 1)
        elif args.scale_init == 'sampled':
            scale = torch.randn((1, args.n_heads, 1, 1), dtype=torch.float).exp()
        else:
            scale = torch.full((1, args.n_heads, 1, 1), float(args.scale_init), dtype=torch.float)
        scale = torch.log(scale)
        
        if args.train_shape and args.shape_init == 'linear':
            shape  = torch.linspace(0, 1, args.n_heads, dtype=torch.float).reshape(1, args.n_heads, 1, 1)
        elif args.train_shape and args.shape_init == 'sampled':
            shape  = torch.randn((1, args.n_heads, 1, 1), dtype=torch.float)
        elif args.train_shape:
            shape   = torch.full((1, args.n_heads, 1, 1), float(args.shape_init), dtype=torch.float)
        else:
            shape   = torch.ones((1, args.n_heads, 1, 1), dtype=torch.float)

        loc     = torch.full((1, args.n_heads, 1, 1), float(args.loc_init),   dtype=torch.float)
        
        self.shape = nn.Parameter(shape, requires_grad = args.train_shape)
        self.scale = nn.Parameter(scale, requires_grad = args.train_scale)
        self.loc   = nn.Parameter(loc,   requires_grad = args.train_loc)




    def forward(self, seq_len=None, start_pos=0):
        seq_len = seq_len or self.seq_len
        q_positions = torch.arange(seq_len, device=self.scale.device).float() + start_pos
        k_positions = torch.arange(seq_len+start_pos, device=self.scale.device).float()

        b = (k_positions[None,:] - q_positions[:, None]).reshape(1, 1, seq_len, seq_len+start_pos)
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
    def __init__(self, args: SSMaxBATModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.ssmax_prior = args.ssmax_prior

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.local_positional_encoding = not args.global_positional_encoding
        if self.local_positional_encoding:
            self.prior = AttentionPrior(args)

        seq_scale =  torch.ones((1, args.n_heads, 1, 1), dtype=torch.float)
        self.seq_scale = nn.Parameter(seq_scale, requires_grad=args.seq_scale)

        self.cache_v = torch.zeros((1, 512*1024, self.n_local_kv_heads, self.head_dim))
        self.cache_k = torch.zeros((1, 512*1024, self.n_local_kv_heads, self.head_dim))
        

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        global_prior: Optional[torch.Tensor] = None,
        section_log_len: Optional[torch.Tensor] = None,
        start_pos: int = 0,
    ):
        bsz, seqlen, _ = x.shape
        queries, keys, values = self.wq(x), self.wk(x), self.wv(x)

        queries = queries.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        keys = keys.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        values = values.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        self.cache_k = self.cache_k.to(x.device)
        self.cache_v = self.cache_v.to(x.device)
        if self.cache_k.size(1) < seqlen+start_pos:
            new_size = max(seqlen+start_pos, self.cache_k.size(1)*2)
            temp_cache_k = torch.zeros((1, new_size, self.n_local_kv_heads, self.head_dim), device=x.device)
            temp_cache_v = torch.zeros((1, new_size, self.n_local_kv_heads, self.head_dim), device=x.device)
            temp_cache_k[:, :self.cache_k.size(1), :, :] = self.cache_k
            temp_cache_v[:, :self.cache_v.size(1), :, :] = self.cache_v
            self.cache_k = temp_cache_k
            self.cache_v = temp_cache_v
        self.cache_k[:, start_pos:start_pos+seqlen, :, :] = keys
        self.cache_v[:, start_pos:start_pos+seqlen, :, :] = values
        keys = self.cache_k[:, :start_pos+seqlen, :, :]
        values = self.cache_v[:, :start_pos+seqlen, :, :]

        queries = queries.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        values = values.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)

        scores = torch.matmul(queries, keys.transpose(2, 3)) / math.sqrt(self.head_dim)

        if section_log_len is not None and not self.ssmax_prior:
            scores = scores * (section_log_len * self.seq_scale)

        if self.local_positional_encoding:
            scores = scores + self.prior(seqlen, start_pos)
        elif global_prior is not None:
            scores = scores + global_prior

        if section_log_len is not None and self.ssmax_prior:
            scores = scores * (section_log_len * self.seq_scale)

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
    def __init__(self, layer_id: int, args: SSMaxBATModelArgs):
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
        global_prior: Optional[torch.Tensor] = None,
        section_log_len: Optional[torch.Tensor] = None,
        start_pos: int = 0,
    ):
        h = x + self.attention(self.attention_norm(x), mask, global_prior, section_log_len, start_pos)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class SSMaxBATransformer(nn.Module):
    def __init__(self, params: SSMaxBATModelArgs):
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

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, seq_batch_size: Optional[int] = None, return_logits: bool = False, return_device=None):
        return_device = return_device if return_device is not None else tokens.device
        _bsz, seqlen = tokens.shape
        full_h = self.tok_embeddings(tokens)
        full_output = []

        if seq_batch_size is None:
            seq_batch_size = seqlen
        for start_pos in range(0, seqlen, seq_batch_size):
            h = full_h[:, start_pos:start_pos+seq_batch_size, :].contiguous()
            _bsz, seqlen, h_dim = h.shape

            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1)
            if start_pos > 0:
                mask = torch.hstack([torch.zeros((seqlen, start_pos), device=tokens.device), mask])

            section_log_len = mask.isfinite().float().sum(-1, keepdim=True).log().unsqueeze(-3)

            global_prior = None
            if self.global_positional_encoding:
                global_prior = self.prior(seqlen, start_pos)

            mask = mask.type_as(h)

            for layer in self.layers:
                h = layer(h, mask, global_prior, section_log_len, start_pos)
            h = self.norm(h)
            output = self.output(h).float()
            if return_logits:
                full_output.append(output.to(return_device))
            else:
                full_output.append(output.argmax(-1).to(return_device))
        full_output = torch.cat(full_output, dim=1)
        return full_output