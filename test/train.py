import math
import random
import time
import json
import pickle
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from datasets import load_dataset
from tqdm import tqdm


def _zeropower_via_newtonschulz5(G, steps=5):
    """Orthogonalize G using Newton-Schulz iteration."""
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    X = X / (X.norm() + 1e-7)
    if X.size(0) > X.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """Muon: Momentum + orthogonalized update for 2D weights; SGD momentum for rest."""
    def __init__(self, params, lr=0.02, momentum=0.95,
                 adamw_params=None, adamw_lr=3e-3, adamw_wd=0.0, adamw_betas=(0.9, 0.95)):
        defaults = dict(lr=lr, momentum=momentum)
        super().__init__(params, defaults)
        # AdamW sub-optimizer for 1D params and embeddings
        if adamw_params is not None:
            self.adamw = torch.optim.AdamW(
                adamw_params, lr=adamw_lr, weight_decay=adamw_wd, betas=adamw_betas
            )
        else:
            self.adamw = None

    @torch.no_grad()
    def step(self):
        if self.adamw is not None:
            self.adamw.step()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if 'buf' not in state:
                    state['buf'] = torch.zeros_like(p)
                buf = state['buf']
                buf.mul_(group['momentum']).add_(p.grad)
                # Nesterov: use g + momentum * v as update direction
                nesterov = p.grad + group['momentum'] * buf
                if p.ndim == 2:
                    update = _zeropower_via_newtonschulz5(nesterov)
                    update *= max(p.size(0), p.size(1)) ** 0.5
                else:
                    update = nesterov
                p.add_(update, alpha=-group['lr'])

# ---------------------------------------------------------------------------
# Hyperparameters — agent modifies these freely
# ---------------------------------------------------------------------------

@dataclass
class Hyperparameters:
    # Data (DO NOT CHANGE — assessment rules)
    num_titles: int = 100_000
    val_frac: float = 0.10
    seed: int = 1337
    epochs: int = 7               # DO NOT CHANGE — assessment rule

    # Model architecture
    block_size: int = 128
    n_layer: int = 24
    n_head: int = 1
    d_model: int = 640
    dropout: float = 0.1

    # Training
    batch_size: int = 256
    lr: float = 1.2e-3
    weight_decay: float = 0.05
    betas: tuple = (0.9, 0.999)
    warmup_frac: float = 0.20
    grad_clip: float = 1.0
    evals_per_epoch: int = 3

    # Logging
    log_file: str = "./logs/mainrun.log"

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def get_titles(num_titles: int, seed: int, val_frac: float):
    ds = load_dataset(
        "julien040/hacker-news-posts",
        split="train",
        cache_dir="./data"
    ).shuffle(seed=seed)
    titles = [row["title"].strip() for row in ds.take(num_titles)]
    n = int(num_titles * (1 - val_frac))
    return titles[:n], titles[n:]

def get_batch(split_ids, ptr, block_size, batch_size, device):
    span = block_size * batch_size + 1
    if ptr + span >= len(split_ids):
        ptr = 0
    batch = split_ids[ptr: ptr + span]
    x = batch[:-1].view(batch_size, block_size).to(device)
    y = batch[1:].view(batch_size, block_size).to(device)
    return x, y, ptr + block_size * batch_size

def iter_full_split(split_ids, block_size, batch_size, device):
    span = block_size * batch_size + 1
    for ptr in range(0, len(split_ids) - span + 1, span):
        batch = split_ids[ptr: ptr + span]
        x = batch[:-1].view(batch_size, block_size).to(device)
        y = batch[1:].view(batch_size, block_size).to(device)
        yield x, y

# ---------------------------------------------------------------------------
# Tokenizer — loaded from disk (trained once by prepare.py)
# ---------------------------------------------------------------------------

TOKENIZER_PATH = "./data/tokenizer.pkl"

class BPETokenizer:
    def __init__(self, tokenizer):
        self.tk = tokenizer

    @classmethod
    def from_file(cls, path=TOKENIZER_PATH):
        assert os.path.exists(path), (
            f"Tokenizer not found at {path}. Run `uv run prepare.py` first."
        )
        with open(path, "rb") as f:
            tokenizer = pickle.load(f)
        return cls(tokenizer)

    def encode(self, s):
        return self.tk.encode(s).ids

    def decode(self, ids):
        return self.tk.decode(ids, skip_special_tokens=True)

    @property
    def vocab_size(self):
        return self.tk.get_vocab_size()

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int
    n_layer: int
    n_head: int
    d_model: int
    dropout: float

def precompute_rope_freqs(head_dim, max_seq_len, base=20, device='cpu'):
    freqs = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(t, freqs)  # [T, head_dim/2]
    return torch.polar(torch.ones_like(freqs), freqs)  # complex [T, head_dim/2]

def apply_rope(q, k, freqs_cis):
    # q, k: [B, n_head, T, head_dim]
    def rotate(x):
        xc = x.float().reshape(*x.shape[:-1], -1, 2)
        xc = torch.view_as_complex(xc.contiguous())  # [B, H, T, head_dim/2]
        xc = xc * freqs_cis.unsqueeze(0).unsqueeze(0)
        return torch.view_as_real(xc).flatten(-2).to(x.dtype)
    return rotate(q), rotate(k)

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.head_dim = cfg.d_model // cfg.n_head
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.proj.RESIDUAL_SCALE_INIT = 0.02 / math.sqrt(2 * cfg.n_layer)
        self.dropout = cfg.dropout
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x, freqs_cis):
        B, T, C = x.size()
        qkv = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim).transpose(1, 3)
        q, k, v = qkv[..., 0, :, :], qkv[..., 1, :, :], qkv[..., 2, :, :]
        q, k = apply_rope(q, k, freqs_cis[:T])
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))

class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        # SwiGLU with 4*d_model hidden (larger capacity)
        hidden = 4 * cfg.d_model
        self.gate = nn.Linear(cfg.d_model, hidden, bias=False)
        self.up   = nn.Linear(cfg.d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)
        self.down.RESIDUAL_SCALE_INIT = 0.02 / math.sqrt(2 * cfg.n_layer)

    def forward(self, x):
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))

class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.mlp = MLP(cfg)
        ls_init = 0.001
        self.ls_attn = nn.Parameter(ls_init * torch.ones(cfg.d_model))
        self.ls_mlp  = nn.Parameter(ls_init * torch.ones(cfg.d_model))

    def forward(self, x, freqs_cis):
        ln_out = self.ln(x)
        return x + self.ls_attn * self.attn(ln_out, freqs_cis) + self.ls_mlp * self.mlp(ln_out)

class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        self.apply(self._init_weights)
        self.head.weight = self.token_emb.weight  # weight tying

        head_dim = cfg.d_model // cfg.n_head
        freqs = precompute_rope_freqs(head_dim, cfg.block_size)
        self.register_buffer("freqs_cis", freqs)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            std = getattr(module, "RESIDUAL_SCALE_INIT", 0.02)
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        x = self.drop(self.token_emb(idx))
        for block in self.blocks:
            x = checkpoint(block, x, self.freqs_cis, use_reentrant=False)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                reduction="mean"
            )
        return logits, loss

# ---------------------------------------------------------------------------
# Evaluation — DO NOT MODIFY (assessment rule)
# ---------------------------------------------------------------------------

def evaluate(model, val_ids, val_text, block_size, batch_size, device):
    model.eval()
    losses = 0.0
    with torch.no_grad():
        for xb, yb in iter_full_split(val_ids, block_size, batch_size, device):
            logits, _ = model(xb, yb)
            B, T, V = logits.size()
            loss = F.cross_entropy(logits.view(-1, V), yb.view(-1), reduction="sum")
            losses += loss.item()
    model.train()
    return losses / len(val_text)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_log(log_file):
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    return open(log_file, "w")

def log(fh, event, **kwargs):
    entry = json.dumps({"event": event, "timestamp": time.time(), **kwargs})
    fh.write(entry + "\n")
    fh.flush()

    prnt = kwargs.pop("prnt", True)
    if prnt:
        parts = [f"{k}={v}" for k, v in kwargs.items()]
        print(f"{event}: {', '.join(parts)}" if parts else event)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = Hyperparameters()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    fh = setup_log(args.log_file)
    log(fh, "hyperparameters", **vars(args))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(fh, "device", device=device)

    # --- Tokenizer (loaded from disk — trained once by prepare.py) ---
    print("Loading tokenizer...")
    tok = BPETokenizer.from_file()
    log(fh, "tokenizer_info", vocab_size=tok.vocab_size)

    # --- Data ---
    print("Loading dataset...")
    train_titles, val_titles = get_titles(args.num_titles, args.seed, args.val_frac)

    eos_token = "<eos>"
    train_text = eos_token.join(train_titles) + eos_token
    val_text = eos_token.join(val_titles) + eos_token
    train_ids = torch.tensor(tok.encode(train_text), dtype=torch.long)
    val_ids = torch.tensor(tok.encode(val_text), dtype=torch.long)

    batches = len(train_ids) // (args.block_size * args.batch_size)
    max_steps = args.epochs * batches
    eval_interval = max(1, batches // args.evals_per_epoch)

    log(fh, "dataset_info",
        train_titles=len(train_titles),
        val_titles=len(val_titles),
        batches_per_epoch=batches,
        max_steps=max_steps,
        vocab_size=tok.vocab_size)

    # --- Model ---
    cfg = GPTConfig(
        vocab_size=tok.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        d_model=args.d_model,
        dropout=args.dropout,
    )
    model = GPT(cfg).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(fh, "model_info", num_params=num_params)

    # --- Optimizer & Scheduler ---
    # Muon for 2D weight matrices; AdamW inside Muon for 1D params & embeddings
    muon_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and p.dim() == 2
                   and 'token_emb' not in n and 'pos_emb' not in n]
    adamw_params = [p for n, p in model.named_parameters()
                    if p.requires_grad and (p.dim() < 2
                    or 'token_emb' in n or 'pos_emb' in n)]
    opt = Muon(
        muon_params, lr=args.lr * 1.3, momentum=0.91,
        adamw_params=[{"params": adamw_params, "weight_decay": 0.0}],
        adamw_lr=args.lr, adamw_wd=0.0, adamw_betas=args.betas,
    )
    warmup_steps = int(args.warmup_frac * max_steps)
    min_lr_ratio = 1.0  # constant LR after warmup
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    # --- SWA setup: average weights over last 20% of training ---
    swa_start = int(0.40 * max_steps)
    swa_model = torch.optim.swa_utils.AveragedModel(model)

    # --- Training loop ---
    ptr = 0
    step = 0
    t0 = time.time()
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        for _ in tqdm(range(batches), desc=f"Epoch {epoch}/{args.epochs}"):
            step += 1
            xb, yb, ptr = get_batch(
                train_ids, ptr, args.block_size, args.batch_size, device
            )
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                _, loss = model(xb, yb)
            opt.zero_grad(set_to_none=True)
            if opt.adamw is not None:
                opt.adamw.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            scheduler.step()
            if step >= swa_start:
                swa_model.update_parameters(model)

            log(fh, "train_step",
                step=step, max_steps=max_steps,
                loss=loss.item(), elapsed=round(time.time() - t0, 2),
                prnt=False)

            if step == 1 or step % eval_interval == 0 or step == max_steps:
                val_loss = evaluate(
                    model, val_ids, val_text,
                    args.block_size, args.batch_size, device
                )
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                log(fh, "val_step",
                    step=step, max_steps=max_steps,
                    val_loss=round(val_loss, 6),
                    elapsed=round(time.time() - t0, 2))

    # --- SWA final evaluation ---
    swa_val_loss = evaluate(
        swa_model.module, val_ids, val_text,
        args.block_size, args.batch_size, device
    )
    if swa_val_loss < best_val_loss:
        best_val_loss = swa_val_loss

    # --- Final summary (agent greps these lines) ---
    total_time = time.time() - t0
    peak_vram_mb = (
        torch.cuda.max_memory_allocated() / 1024 / 1024
        if torch.cuda.is_available() else 0.0
    )

    print("---")
    print(f"val_loss:         {best_val_loss:.6f}")
    print(f"epochs:           {args.epochs}")
    print(f"total_seconds:    {total_time:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"num_params:       {num_params}")
    print(f"n_layer:          {args.n_layer}")
    print(f"d_model:          {args.d_model}")

    log(fh, "final_summary",
        val_loss=best_val_loss,
        total_seconds=round(total_time, 1),
        peak_vram_mb=round(peak_vram_mb, 1),
        num_params=num_params)

    fh.close()

if __name__ == "__main__":
    main()