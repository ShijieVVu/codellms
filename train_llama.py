import os
import time
import torch
import tiktoken
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from glob import glob
from math import cos, sin, pi
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from llms.config import TrainArgs as Args


class RotaryEmbedding(nn.Module):
    def __init__(self, config: Args):
        super().__init__()
        self.base = config.theta
        self.head_dim = config.dims // config.num_heads
        self.n = config.max_length
        self.register_sinusoidal()

    def register_sinusoidal(self):
        freqs = 1 / self.base ** (torch.arange(0, self.head_dim, 2) / self.head_dim)
        phases = torch.outer(torch.arange(self.n), freqs)
        phases = torch.cat([phases, phases], dim=-1)
        self.register_buffer('cos', phases.cos())
        self.register_buffer('sin', phases.sin())

    def rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, x):
        return self.cos * x + self.rotate_half(x) * self.sin

class Attention(nn.Module):
    def __init__(self, config: Args):
        super().__init__()
        dim = config.dims
        self.dim = dim
        self.n_heads = config.num_heads
        self.kv_heads = config.num_kv_heads
        self.rep = self.n_heads // self.kv_heads
        self.head_dim = dim // self.n_heads
        kv_dim = config.num_kv_heads * self.head_dim
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, kv_dim, bias=False)
        self.wv = nn.Linear(dim, kv_dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.rotary = RotaryEmbedding(config)

    def forward(self, x):
        B, L, _ = x.shape
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        q = q.view(B, L, self.n_heads, self.head_dim)
        k = k.view(B, L, self.kv_heads, self.head_dim)
        v = v.view(B, L, self.kv_heads, self.head_dim)

        # B L h C
        # transpose for rotary and sdpa
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q, k = self.rotary(q), self.rotary(k)
        # repeats k and v
        k = k[:, :, None, :, :].expand(B, self.kv_heads, self.rep, L, self.head_dim).reshape(B, self.n_heads, L, self.head_dim)
        v = v[:, :, None, :, :].expand(B, self.kv_heads, self.rep, L, self.head_dim).reshape(B, self.n_heads, L, self.head_dim)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(B, L, self.dim)
        return self.wo(out)

class MLP(nn.Module):
    def __init__(self, config: Args):
        super().__init__()
        dim = config.dims
        hidden = int(8 / 3 * config.dims)
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.w2(self.w1(x).sigmoid() * self.w3(x))


class RMSNorm(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dims))
        self.eps = 1e-6
    
    def forward(self, x):
        # B L C
        input_dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * x.to(input_dtype)


class DecoderLayer(nn.Module):
    def __init__(self, config: Args):
        super().__init__()
        self.attn = Attention(config)
        self.mlp = MLP(config)
        self.norm1 = RMSNorm(config.dims)
        self.norm2 = RMSNorm(config.dims)

    def forward(self, x):
        x = self.attn(self.norm1(x))
        x = self.mlp(self.norm2(x))
        return x


class Llama(nn.Module):
    def __init__(self, config: Args):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.dims)
        self.layers = nn.ModuleList()
        for _ in range(config.num_layers):
            self.layers.append(DecoderLayer(config))
        self.norm = RMSNorm(config.dims)
        self.output = nn.Linear(config.dims, config.vocab_size)

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x, labels=None):
        # max_len x dim
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.output(x)
        if labels is None:
            return logits
        # cross entropy signature
        loss = F.cross_entropy(logits.transpose(1, 2), labels)
        return logits, loss

    def configure_optimizers(self, weight_decay):
        params = {p for n, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for p in params if p.dim() >= 2]
        nodecay_params = [p for p in params if p.dim() == 1]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0}, 
        ]
        if rank == 0:
            print(f'total parameters: {sum(p.numel() for p in params)}')
        return torch.optim.AdamW(optim_groups, fused=True)


init_process_group(backend='nccl')
rank = int(os.environ['RANK'])
world_size = int(os.environ['WORLD_SIZE'])
device = f'cuda:{rank}'
torch.cuda.set_device(device)

args = Args()
model = Llama(config=args)
model.to(device)
model = DDP(model)
raw_model = model.module

B = 8
L = args.max_length
total_batch_size = 524288
grad_accum_steps = total_batch_size // (B * L)


def load_tokens(filename):
    npt = np.load(filename)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class Dataset:
    def __init__(self, split, rank, world_size):
        self.B = B
        self.T = L
        self.rank = rank
        self.num_processes = world_size
        assert split in {'train', 'val'}

        # get the shard filenames
        # data_root = "edu_fineweb10B"
        # data_root = 'starcoderdatajava'
        data_root = '/mnt/c/Codes/build-nanogpt/starcoderdata/'
        # data_root = 'openwebtext'
        # data_root = "starcoderdata"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if rank == 0:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.rank
        return x, y
    

# # dataloader
# class Dataset:
#     def __init__(self, split, rank, world_size):
#         self.rank = rank
#         self.world_size = world_size
#         folder_name = '/mnt/c/Codes/build-nanogpt/starcoderdata/'

#         self.shard_paths = glob(os.path.join(folder_name, f'*{split}*'))
#         self.cur_shard = 0
#         self.cur_token = 0
#         self.tokens = self.load_from_file(self.cur_shard)
#         self.num_shards = len(self.shard_paths)

#     def load_from_file(self, shard_id):
#         npt = np.load(self.shard_paths[shard_id])
#         ptt = torch.tensor(npt, dtype=torch.long)
#         return ptt
    
#     def next_batch(self):
#         if self.cur_token + B * L * world_size < len(self.tokens):
#             buffer = self.tokens[self.cur_token + B * L * self.rank: self.cur_token + B * L * (self.rank + 1) + 1]
#             x, y = buffer[:, :-1], buffer[:, 1:]
#             x, y = x.reshape(B, L), y.reshape(B, L)
#             self.cur_token += B * L * world_size
#             return x, y
        
#         remain = len(self.tokens) - self.cur_token
#         remain_tokens = self.tokens[-remain:]
#         self.cur_shard = (self.cur_shard + 1) % self.num_shards
#         self.tokens = self.load_from_file(self.cur_shard)
#         self.cur_token = B * L * world_size - remain
#         buffer = torch.cat([remain_tokens, self.tokens[:self.cur_token + 1]], dim=-1)
#         buffer = buffer[B * L * self.rank: B * L * (self.rank + 1)]
#         buffer = buffer.reshape(B, L)
#         x, y = buffer[:, :-1], buffer[:, 1:]
#         return x, y

# training loss

# learning rate schedule
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 19073
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + cos(pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

optimizer = raw_model.configure_optimizers(weight_decay=0.1)
train_loader = Dataset(split='train', rank=rank, world_size=world_size)
val_loader = Dataset(split='val', rank=rank, world_size=world_size)

torch.set_float32_matmul_precision('high')

for step in range(max_steps):
    t0 = time.time()
    optimizer.zero_grad()
    model.train()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            _, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        model.require_backward_grad_sync = micro_step == (grad_accum_steps - 1)
        loss.backward()
    dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # determine and set the learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize() # wait for the GPU to finish work
    t1 = time.time()
    dt = t1 - t0 # time difference in seconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * world_size
    tokens_per_sec = tokens_processed / dt
    computation_per_tok = 6 * 162e6 + 12 * args.num_layers * args.dims * args.max_length
    mfu = tokens_per_sec * computation_per_tok / 82.58e12
    if rank == 0:
        print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f} | mfu: {100 * mfu:.2f}%")

destroy_process_group()