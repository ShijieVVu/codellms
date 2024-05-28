import torch
import math
import torch.nn as nn

import torch.nn.functional as F

from typing import Tuple, Optional
from .config import TrainArgs as Args


if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


# def compute_position_encoding(args: Args) -> torch.Tensor:
#     head_dim = args.hidden_dims // args.num_heads
#     phase = torch.arange(args.max_length)
#     positions = (torch.arange(head_dim // 2) / (head_dim // 2))[:, None].repeat(1, 2).flatten()
#     freqs = torch.exp(-math.log(args.temperature) * positions)
#     encodings = torch.outer(phase, freqs)
#     return encodings


def precompute_freqs_cis(args: Args):
    theta = args.theta
    dim = args.dims // args.num_heads
    end = args.max_length * 2
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

    t = torch.arange(end, device=freqs.device, dtype=torch.float32)  # type: ignore
    freqs = torch.outer(t, freqs)  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1]), f"{freqs_cis.shape}, {x.shape}"
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not torch.cuda.is_available():
        xq = xq.to('cpu')
        xk = xk.to('cpu')
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq).to(device), xk_out.type_as(xk).to(device)


def repeat_kv(t: torch.Tensor, rep: int) -> torch.Tensor:
    if rep == 1:
        return t
    bsz, length, n_heads, dim = t.shape
    t = t[:, :, :, None, :] \
        .repeat(1, 1, 1, rep, 1) \
        .flatten(2, 3)
    return t

class Attention(nn.Module):
    def __init__(self, args: Args):
        super().__init__()
        self.dims = args.dims
        self.n_heads = args.num_heads
        self.head_dims = self.dims // self.n_heads
        self.n_kv_heads = args.num_kv_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.kv_dims = self.n_kv_heads * self.head_dims
        self.max_length = args.max_length
        self.max_bsz = args.max_bsz

        self.wq = nn.Linear(self.dims, self.dims, bias=False)
        self.wk = nn.Linear(self.dims, self.kv_dims, bias=False)
        self.wv = nn.Linear(self.dims, self.kv_dims, bias=False)
        self.wo = nn.Linear(self.dims, self.dims, bias=False)

        self.cache_k = torch.zeros(self.max_bsz, self.max_length, self.n_kv_heads, self.head_dims).to(device)
        self.cache_v = torch.zeros(self.max_bsz, self.max_length, self.n_kv_heads, self.head_dims).to(device)
    
    def forward(self, 
                x: torch.Tensor, 
                starting_pos: int, 
                position_encoding: torch.Tensor,
                mask: Optional[torch.Tensor]) -> torch.Tensor:
        bsz, length, _ = x.shape
        hq, hk, hv = self.wq(x), self.wk(x), self.wv(x)
        hq = hq.view(bsz, length, self.n_heads, self.head_dims)
        hk = hk.view(bsz, length, self.n_kv_heads, self.head_dims)
        hv = hv.view(bsz, length, self.n_kv_heads, self.head_dims)
        
        hq, hk = apply_rotary_emb(hq, hk, position_encoding)

        self.cache_k[:bsz, starting_pos: starting_pos + length] = hk
        self.cache_v[:bsz, starting_pos: starting_pos + length] = hv

        key = self.cache_k[:bsz, : starting_pos + length]
        value = self.cache_v[:bsz, : starting_pos + length]

        key = repeat_kv(key, self.n_rep)
        value = repeat_kv(value, self.n_rep)

        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        hq = hq.transpose(1, 2)

        scores = torch.matmul(hq, key.transpose(2, 3)) / math.sqrt(self.head_dims)
        if mask is not None:
            scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(hq)

        results = torch.matmul(scores, value)
        results = results.transpose(1, 2).flatten(2, 3)
        return self.wo(results)
    

class FeedForward(nn.Module):
    def __init__(self, args: Args):
        super().__init__()
        self.dim = args.dims
        self.hidden_dims = args.hidden_dims
        
        hidden_dims = int(2 * args.hidden_dims / 3)
        if args.multiple_of is not None:
            hidden_dims = (hidden_dims + args.multiple_of - 1) // args.multiple_of * args.multiple_of

        self.w1 = nn.Linear(self.dim, hidden_dims, bias=False)
        self.w2 = nn.Linear(hidden_dims, self.dim, bias=False)
        self.w3 = nn.Linear(self.dim, hidden_dims, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RMSNorm(nn.Module):
    def __init__(self, args: Args, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(args.dims))
    
    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
    #     return x * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class TransformerBlock(nn.Module):
    def __init__(self, args: Args):
        super().__init__()
        self.attn = Attention(args)
        self.ffn = FeedForward(args)
        self.norm1 = RMSNorm(args)
        self.norm2 = RMSNorm(args)
    
    def forward(self, x: torch.Tensor, starting_position: int, position_encoding: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        h = self.attn(self.norm1(x), starting_position, position_encoding, mask)
        h = x + h
        out = self.ffn(self.norm2(h))
        out = h + out
        return out


class Transformer(nn.Module):
    def __init__(self, args: Args):
        super().__init__()
        self.dims = args.dims
        self.num_layers = args.num_layers
        self.vocab_size = args.vocab_size

        self.embedding = nn.Embedding(self.vocab_size, self.dims)
        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.layers.append(TransformerBlock(args))
        
        self.norm = RMSNorm(args)
        self.output = nn.Linear(self.dims, self.vocab_size, bias=False)
        self.position_encoding = precompute_freqs_cis(args)

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_position: int) -> torch.Tensor:
        bsz, length = tokens.shape
        x = self.embedding(tokens)
        position_encoding = self.position_encoding[start_position: start_position + length].to(device)

        mask = None
        if length > 1:
            mask = torch.full((length, start_position + length), float('-inf'))
            mask = mask.triu((1 + start_position)).to(device)
        
        for layer in self.layers:
            x = layer(x, start_position, position_encoding, mask)
        x = self.norm(x)

        x = self.output(x)
        return x