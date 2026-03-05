"""Causal Transformer architecture."""
import math, torch, torch.nn as nn, torch.nn.functional as F
from config import TinyGPTConfig

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return self.weight * x / x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads, self.head_dim = cfg.n_heads, cfg.head_dim
        self.qkv  = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)
        self.register_buffer("mask", torch.tril(torch.ones(cfg.context_length, cfg.context_length)).unsqueeze(0).unsqueeze(0))
    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim).unbind(2)
        q, k, v = [t.transpose(1,2) for t in (q, k, v)]
        attn = (q @ k.transpose(-2,-1)) / math.sqrt(self.head_dim)
        attn = attn.masked_fill(self.mask[:,:,:T,:T]==0, float("-inf"))
        attn = self.drop(F.softmax(attn, dim=-1))
        return self.proj((attn @ v).transpose(1,2).reshape(B, T, C))

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(cfg.d_model, cfg.d_ff, bias=False), nn.GELU(),
                                 nn.Dropout(cfg.dropout), nn.Linear(cfg.d_ff, cfg.d_model, bias=False))
    def forward(self, x): return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1, self.attn = RMSNorm(cfg.d_model), CausalSelfAttention(cfg)
        self.norm2, self.ff   = RMSNorm(cfg.d_model), FeedForward(cfg)
        self.drop = nn.Dropout(cfg.dropout)
    def forward(self, x):
        x = x + self.drop(self.attn(self.norm1(x)))
        x = x + self.drop(self.ff(self.norm2(x)))
        return x

class TinyGPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.context_length, cfg.d_model)
        self.drop    = nn.Dropout(cfg.dropout)
        self.blocks  = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm    = RMSNorm(cfg.d_model)
        self.head    = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.tok_emb.weight = self.head.weight
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):   nn.init.normal_(m.weight, std=0.02)
        if isinstance(m, nn.Embedding): nn.init.normal_(m.weight, std=0.02)
    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.drop(self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device)))
        for block in self.blocks: x = block(x)
        logits = self.head(self.norm(x))
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) if targets is not None else None
        return logits, loss
    def num_params(self): return sum(p.numel() for p in self.parameters())
