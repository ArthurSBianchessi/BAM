import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention
from torch.nn.attention.flex_attention import create_block_mask


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

    thata_beta_init: float | str = 0
    theta_alpha_init: float | str = 0
    theta_mu_init:   float = 0

    train_theta_beta: bool = True
    train_theta_alpha: bool = True
    train_theta_mu:   bool = False

    global_positional_encoding: bool = False
    seq_scale: bool = True

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

        
        if args.theta_alpha_init == 'slope':
            theta_alpha = torch.tensor(get_slopes(args.n_heads), dtype=torch.float).reshape(args.n_heads, 1)
        elif args.theta_alpha_init == 'sampled':
            theta_alpha = torch.randn((args.n_heads, 1), dtype=torch.float).exp()
        else:
            theta_alpha = torch.full((args.n_heads, 1), float(args.theta_alpha_init), dtype=torch.float)
        
        if args.train_theta_beta and args.thata_beta_init == 'linear':
            theta_beta  = torch.linspace(0, 1, args.n_heads, dtype=torch.float).reshape(args.n_heads, 1)
        elif args.train_theta_beta and args.thata_beta_init == 'sampled':
            theta_beta  = torch.randn((args.n_heads, 1), dtype=torch.float)
        elif args.train_theta_beta:
            theta_beta   = torch.full((args.n_heads, 1), float(args.thata_beta_init), dtype=torch.float)
        else:
            theta_beta   = torch.ones((args.n_heads, 1), dtype=torch.float)

        theta_mu = torch.full((args.n_heads, 1), float(args.theta_mu_init),   dtype=torch.float)
        
        self.theta_beta  = nn.Parameter(theta_beta, requires_grad = args.train_theta_beta)
        self.theta_alpha = nn.Parameter(theta_alpha, requires_grad = args.train_theta_alpha)
        self.theta_mu    = nn.Parameter(theta_mu,   requires_grad = args.train_theta_mu)

    def forward(self, seq_len=None):
        seq_len = seq_len or self.seq_len
        positions = torch.arange(1-seq_len, seq_len, device=self.theta_alpha.device).float()
        b = positions - (self.theta_mu.exp() - (-self.theta_mu).exp())
        return -((b.abs() + self.eps) ** self.theta_beta) * self.theta_alpha.exp() 
    

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

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.local_positional_encoding = not args.global_positional_encoding
        if self.local_positional_encoding:
            self.prior = AttentionPrior(args)

        seq_scale =  torch.ones((1, args.n_heads, 1), dtype=torch.float)
        self.seq_scale = nn.Parameter(seq_scale, requires_grad=args.seq_scale)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        gloabal_prior: Optional[torch.Tensor] = None,
        section_log_len: Optional[torch.Tensor] = None,
    ):
        bsz, seqlen, _ = x.shape

        ssmax_mul = section_log_len * self.seq_scale
        prior = self.prior(seqlen) if self.local_positional_encoding else gloabal_prior

        queries, keys, values = self.wq(x), self.wk(x), self.wv(x)

        queries = queries.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        keys = keys.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        values = values.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        queries = queries.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        values = values.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)

        # ssmax_mul = section_log_len * self.seq_scale
        # prior = self.prior(seqlen) if self.local_positional_encoding else gloabal_prior
        def score_mod(score, b, h, q_idx, kv_idx):
            score = score + prior[h, seqlen-1+kv_idx-q_idx]
            return score * ssmax_mul[b, h, q_idx]

        

        output = flex_attention(queries, keys, values, score_mod=score_mod, block_mask=mask,
                                kernel_options = {
                                    "BLOCK_M":  32,
                                    "BLOCK_N":  32,
                                    "BLOCK_M1": 32,
                                    "BLOCK_N1": 32,
                                    "BLOCK_M2": 32,
                                    "BLOCK_N2": 32,
                                }
                                )

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
        gloabal_prior: Optional[torch.Tensor] = None,
        section_log_len: Optional[torch.Tensor] = None,
    ):
        h = x + self.attention(self.attention_norm(x), mask, gloabal_prior, section_log_len)
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

    def forward(self, tokens: torch.Tensor, seq_codes: Optional[torch.Tensor] = None):
        bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)

        if seq_codes is not None:
            section_log_len = torch.tril((seq_codes.unsqueeze(-1) == seq_codes.unsqueeze(-2))).sum(-1).log().unsqueeze(1)
        else:
            section_log_len = torch.arange(1, seqlen+1).log().unsqueeze(0).unsqueeze(0).to(tokens.device).repeat(bsz, 1, 1)

        def mask_mod(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            seq_mask = seq_codes[b, q_idx] == seq_codes[b, kv_idx]
            return causal_mask & seq_mask
        mask = create_block_mask(mask_mod, B=bsz, H=None, Q_LEN=seqlen, KV_LEN=seqlen, device=tokens.device, BLOCK_SIZE=128)

        global_prior = None
        if self.global_positional_encoding:
            global_prior = self.prior(seqlen)


        for layer in self.layers:
            h = layer(h, mask, global_prior, section_log_len)
        h = self.norm(h)
        output = self.output(h).float()
        return output
