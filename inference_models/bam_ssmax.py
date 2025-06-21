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

    thata_beta_init:  float | str = 0
    theta_alpha_init: float | str = 1
    theta_mu_init:    float = 0

    train_theta_beta:  bool = True
    train_theta_alpha: bool = True
    train_theta_mu:    bool = False

    global_positional_encoding: bool = True
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
            theta_alpha = torch.tensor(get_slopes(args.n_heads), dtype=torch.float).reshape(1, args.n_heads, 1, 1)
        elif args.theta_alpha_init == 'sampled':
            theta_alpha = torch.randn((1, args.n_heads, 1, 1), dtype=torch.float).exp()
        else:
            theta_alpha = torch.full((1, args.n_heads, 1, 1), float(args.theta_alpha_init), dtype=torch.float)
        theta_alpha = torch.log(theta_alpha)
        
        if args.train_theta_beta and args.thata_beta_init == 'linear':
            theta_beta  = torch.linspace(0, 1, args.n_heads, dtype=torch.float).reshape(1, args.n_heads, 1, 1)
        elif args.train_theta_beta and args.thata_beta_init == 'sampled':
            theta_beta  = torch.randn((1, args.n_heads, 1, 1), dtype=torch.float)
        elif args.train_theta_beta:
            theta_beta   = torch.full((1, args.n_heads, 1, 1), float(args.thata_beta_init), dtype=torch.float)
        else:
            theta_beta   = torch.ones((1, args.n_heads, 1, 1), dtype=torch.float)

        theta_mu     = torch.full((1, args.n_heads, 1, 1), float(args.theta_mu_init),   dtype=torch.float)
        
        self.theta_beta = nn.Parameter(theta_beta, requires_grad = args.train_theta_beta)
        self.theta_alpha = nn.Parameter(theta_alpha, requires_grad = args.train_theta_alpha)
        self.theta_mu   = nn.Parameter(theta_mu,   requires_grad = args.train_theta_mu)

    def forward(self, seq_len, start_pos=0):
        q_positions = torch.arange(seq_len, device=self.theta_alpha.device).float() + start_pos
        k_positions = torch.arange(seq_len+start_pos, device=self.theta_alpha.device).float()

        b = (k_positions[None,:] - q_positions[:, None]).reshape(1, 1, seq_len, seq_len+start_pos)
        b = b - (self.theta_mu.exp() - (-self.theta_mu).exp())
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

        seq_scale =  torch.ones((1, args.n_heads, 1, 1), dtype=torch.float)
        self.seq_scale = nn.Parameter(seq_scale, requires_grad=args.seq_scale)
        
        

    def forward(
        self,
        x: torch.Tensor,
        section_log_len: Optional[torch.Tensor] = None,
        seq_batch_size: int = None,
    ):
        bsz, seqlen, _ = x.shape
        seq_batch_size = seq_batch_size or seqlen
        queries, keys, values = self.wq(x), self.wk(x), self.wv(x)

        queries = queries.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        keys = keys.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        values = values.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        outputs = torch.zeros((bsz, self.n_local_heads, seqlen, self.head_dim), device=x.device)

        keys = keys.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        values = values.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        queries = queries.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        # bsz, n_heads, seqlen, head_dim = q.shape
        for start_pos in range(0, seqlen, seq_batch_size):
            local_seqlen = min(seq_batch_size, seqlen - start_pos)
            local_section_log_len = section_log_len[:, :, start_pos:start_pos + local_seqlen, :]
            local_queries = queries[:, :, start_pos:start_pos + local_seqlen, :]
            local_keys = keys[:, :, :start_pos + local_seqlen, :]
            local_values = values[:, :, :start_pos + local_seqlen, :]

            scores = torch.matmul(local_queries, local_keys.transpose(2, 3)) / math.sqrt(self.head_dim)

            scores = scores + self.prior(local_seqlen, start_pos)
            scores = scores * (local_section_log_len * self.seq_scale)

            # if mask is not None:
            #     scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
            # Mask
            mask = torch.full((1, 1, local_seqlen, start_pos+local_seqlen), float('-inf'), device=x.device)
            q_positions = torch.arange(local_seqlen, device=x.device).unsqueeze(0).unsqueeze(0) + start_pos
            k_positions = torch.arange(local_seqlen+start_pos, device=x.device).unsqueeze(0).unsqueeze(0)
            # mask[:, :, q_positions >= k_positions] = 0
            mask[k_positions[:, :, None,:] <= q_positions[:, :, :, None]] = 0
            scores = scores + mask

            scores = F.softmax(scores.float(), dim=-1).type_as(queries)

            output = torch.matmul(scores, local_values)  # (bs, n_local_heads, seqlen, head_dim)
            outputs[:, :, start_pos:start_pos + seq_batch_size, :] = output

        outputs = outputs.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(outputs)


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
        # return self.w2(F.silu(self.w1(x)) * self.w3(x))
        output = []
        for i in range(0, x.shape[1], 32768):
            local_seqlen = min(32768, x.shape[1] - i)
            local_x = x[:, i:i + local_seqlen, :]
            output.append(self.w2(F.silu(self.w1(local_x)) * self.w3(local_x)))
        return torch.cat(output, dim=1)


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
        section_log_len: Optional[torch.Tensor] = None,
        seq_batch_size: int = None,
    ):
        h = x + self.attention(self.attention_norm(x), section_log_len, seq_batch_size)
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
        h = self.tok_embeddings(tokens)

        if seq_batch_size is None:
            seq_batch_size = seqlen

        # section_log_len = mask.isfinite().float().sum(-1, keepdim=True).log().unsqueeze(-3)
        section_log_len = torch.arange(1, seqlen+1, device=h.device).float().unsqueeze(0).unsqueeze(1).unsqueeze(-1).log()


        for i, layer in enumerate(self.layers):
        # for layer in self.layers:
            h = layer(h, section_log_len, seq_batch_size)
        h = self.norm(h)
        # output = self.output(h).float()

        if return_logits:
            return self.output(h).to(return_device)
        else:
            outputs = []
            for i in range(0, seqlen, 32768):
                local_seqlen = min(32768, seqlen - i)
                local_h = h[:, i:i + local_seqlen, :]
                outputs.append(self.output(local_h).argmax(dim=-1).to(return_device))
            return torch.cat(outputs, dim=1)