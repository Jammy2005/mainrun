"""
train.py — GPT training on Hacker News titles.

Trains a small GPT model with modern optimisations:
  - Muon optimizer for 2D weights, AdamW for embeddings and 1D params
  - RoPE positional encoding
  - SwiGLU MLP
  - Parallel attention + MLP block (GPT-J style)
  - LayerScale for training stability
  - Stochastic Weight Averaging (SWA)
  - Gradient checkpointing for memory efficiency
  - bfloat16 mixed precision
"""

import json
import math
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from datasets import load_dataset
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

@dataclass
class Hyperparameters:
    """All tunable parameters for data, architecture, and training."""

    # --- Data (DO NOT CHANGE — assessment rules) ---
    num_titles: int = 100_000
    val_frac: float = 0.10
    seed: int = 1337
    epochs: int = 7
    vocab_size: int = 16_000

    # --- Model architecture ---
    block_size: int = 128
    n_layer: int = 24
    n_head: int = 1
    d_model: int = 640
    dropout: float = 0.1

    # --- Positional encoding ---
    rope_base: int = 20              # RoPE frequency base — lower = better for short seqs

    # --- LayerScale ---
    layer_scale_init: float = 0.001  # Initial scale for attn and MLP residual contributions

    # --- Training ---
    batch_size: int = 256
    lr: float = 1.2e-3
    weight_decay: float = 0.05
    betas: tuple = (0.9, 0.999)
    warmup_frac: float = 0.20        # Fraction of steps used for linear LR warmup
    grad_clip: float = 1.0
    evals_per_epoch: int = 3

    # --- Muon optimizer ---
    muon_lr_ratio: float = 1.5       # Muon LR = lr * muon_lr_ratio
    muon_momentum: float = 0.91
    ns_steps: int = 5                # Newton-Schulz iteration steps for orthogonalisation

    # --- SWA (Stochastic Weight Averaging) ---
    swa_start_frac: float = 0.40     # Fraction of total steps after which SWA begins

    # --- Logging ---
    log_file: str = field(
        default_factory=lambda: f"./logs/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class DualLogger:
    """Writes structured JSON logs to file and human-readable output to stdout."""

    def __init__(self, log_file: str) -> None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        self._file = open(log_file, "w")

    def log(self, event: str, **kwargs) -> None:
        """Log an event with arbitrary keyword metadata."""
        entry = json.dumps({"event": event, "timestamp": time.time(), **kwargs})
        self._file.write(entry + "\n")
        self._file.flush()

        if not kwargs.get("prnt", True):
            return

        if "step" in kwargs and "max_steps" in kwargs:
            loss_val = kwargs.get("loss") or kwargs.get("swa_val_loss", "N/A")
            loss_str = f"{loss_val:.6f}" if isinstance(loss_val, float) else str(loss_val)
            tqdm.write(
                f"[{kwargs['step']:>5}/{kwargs['max_steps']}] "
                f"{event}: loss={loss_str} time={kwargs.get('elapsed_time', 0):.2f}s"
            )
        else:
            parts = [f"{k}={v}" for k, v in kwargs.items() if k not in ("prnt", "timestamp")]
            tqdm.write(f"{event}: {', '.join(parts)}" if parts else event)

    def close(self) -> None:
        """Flush and close the log file."""
        self._file.close()


def configure_logging(log_file: str) -> DualLogger:
    """Create and return a DualLogger writing to log_file."""
    return DualLogger(log_file)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def get_titles(
    num_titles: int,
    seed: int,
    val_frac: float,
) -> tuple[list[str], list[str]]:
    """
    Load and shuffle Hacker News titles, then split into train and validation sets.

    Returns:
        A tuple of (train_titles, val_titles).
    """
    ds = (
        load_dataset("julien040/hacker-news-posts", split="train", cache_dir="./data")
        .shuffle(seed=seed)
    )
    titles = [row["title"].strip() for row in ds.take(num_titles)]
    n = int(num_titles * (1 - val_frac))
    return titles[:n], titles[n:]


def get_batch(
    split_ids: torch.Tensor,
    ptr: int,
    block_size: int,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Slice a (batch_size, block_size) input/target pair from split_ids.

    The target is the input shifted by one token. The pointer wraps to 0
    if there is insufficient data remaining.

    Returns:
        (x, y, new_ptr) where x is input, y is target, new_ptr is the
        updated position in split_ids.
    """
    span = block_size * batch_size + 1
    if ptr + span >= len(split_ids):
        ptr = 0
    batch = split_ids[ptr : ptr + span]
    x = batch[:-1].view(batch_size, block_size).to(device)
    y = batch[1:].view(batch_size, block_size).to(device)
    return x, y, ptr + block_size * batch_size


def iter_full_split(
    split_ids: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: torch.device,
):
    """Yield non-overlapping (x, y) batches covering the full split_ids tensor."""
    span = block_size * batch_size + 1
    for ptr in range(0, len(split_ids) - span + 1, span):
        batch = split_ids[ptr : ptr + span]
        x = batch[:-1].view(batch_size, block_size).to(device)
        y = batch[1:].view(batch_size, block_size).to(device)
        yield x, y


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def train_tokenizer(
    titles: list[str],
    vocab_size: int,
    unk_token: str = "<unk>",
    pad_token: str = "<pad>",
    eos_token: str = "<eos>",
) -> Tokenizer:
    """
    Train a byte-level BPE tokenizer on the provided titles.

    Special tokens are reserved at the start of the vocabulary.
    """
    tokenizer = Tokenizer(models.BPE(unk_token=unk_token))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[pad_token, eos_token, unk_token],
    )
    tokenizer.train_from_iterator(titles, trainer)
    return tokenizer


class BPETokenizer:
    """Thin wrapper around a HuggingFace BPE Tokenizer."""

    def __init__(self, tokenizer: Tokenizer) -> None:
        self._tk = tokenizer

    def encode(self, text: str) -> list[int]:
        """Encode a string to a list of token IDs."""
        return self._tk.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        """Decode a list of token IDs back to a string."""
        return self._tk.decode(ids, skip_special_tokens=True)

    @property
    def vocab_size(self) -> int:
        """Return the vocabulary size."""
        return self._tk.get_vocab_size()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    """Architecture configuration passed to all model components."""

    vocab_size: int
    block_size: int
    n_layer: int
    n_head: int
    d_model: int
    dropout: float
    rope_base: int
    layer_scale_init: float


def precompute_rope_freqs(
    head_dim: int,
    max_seq_len: int,
    base: int = 20,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Precompute complex RoPE rotation frequencies.

    Returns a complex tensor of shape [max_seq_len, head_dim // 2].
    """
    freqs = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )
    t = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Rotary Position Embedding (RoPE) to query and key tensors.

    Args:
        q: Query tensor of shape [B, n_head, T, head_dim].
        k: Key tensor of shape [B, n_head, T, head_dim].
        freqs_cis: Precomputed complex frequencies of shape [T, head_dim // 2].

    Returns:
        Rotated (q, k) tensors with the same shape as input.
    """
    def rotate(x: torch.Tensor) -> torch.Tensor:
        xc = x.float().reshape(*x.shape[:-1], -1, 2)
        xc = torch.view_as_complex(xc.contiguous())
        xc = xc * freqs_cis.unsqueeze(0).unsqueeze(0)
        return torch.view_as_real(xc).flatten(-2).to(x.dtype)

    return rotate(q), rotate(k)


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention with RoPE and FlashAttention.

    Uses a single fused QKV projection for efficiency. Causality is
    enforced via is_causal=True in scaled_dot_product_attention.
    """

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0, "d_model must be divisible by n_head"

        self.head_dim = cfg.d_model // cfg.n_head
        self.n_head = cfg.n_head
        self.dropout = cfg.dropout

        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.proj.RESIDUAL_SCALE_INIT = 0.02 / math.sqrt(2 * cfg.n_layer)  # type: ignore[attr-defined]
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
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
    """
    SwiGLU feed-forward network (LLaMA-style).

    Expands to 4 * d_model hidden units. The gate projection controls
    how much of the up projection passes through, making the nonlinearity
    input-dependent.
    """

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        hidden = 4 * cfg.d_model
        self.gate = nn.Linear(cfg.d_model, hidden, bias=False)
        self.up = nn.Linear(cfg.d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, cfg.d_model, bias=False)
        self.down.RESIDUAL_SCALE_INIT = 0.02 / math.sqrt(2 * cfg.n_layer)  # type: ignore[attr-defined]
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))


class Block(nn.Module):
    """
    GPT-J style transformer block with parallel attention and MLP.

    A single LayerNorm is applied once and its output is fed to both
    attention and MLP in parallel. Their scaled outputs are added to the
    residual stream in one step. LayerScale parameters (initialised near
    zero) stabilise training in deep networks.
    """

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.mlp = MLP(cfg)
        self.ls_attn = nn.Parameter(cfg.layer_scale_init * torch.ones(cfg.d_model))
        self.ls_mlp = nn.Parameter(cfg.layer_scale_init * torch.ones(cfg.d_model))

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        ln_out = self.ln(x)
        return x + self.ls_attn * self.attn(ln_out, freqs_cis) + self.ls_mlp * self.mlp(ln_out)


class GPT(nn.Module):
    """
    GPT language model with modern architectural improvements.

    Key improvements over vanilla GPT-2:
      - RoPE instead of learned absolute positional embeddings
      - SwiGLU MLP instead of GELU
      - Parallel attention + MLP blocks (GPT-J style)
      - LayerScale for training stability
      - Scaled weight initialisation for residual projection layers
      - Gradient checkpointing for memory efficiency
    """

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        self.apply(self._init_weights)
        self.head.weight = self.token_emb.weight  # Weight tying

        head_dim = cfg.d_model // cfg.n_head
        freqs = precompute_rope_freqs(head_dim, cfg.block_size, base=cfg.rope_base)
        self.register_buffer("freqs_cis", freqs)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """
        Initialise weights with a normal distribution.

        Residual projection layers use a scaled std to keep the residual
        stream stable at initialisation across many stacked blocks.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            std = getattr(module, "RESIDUAL_SCALE_INIT", 0.02)
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Forward pass through the full model.

        Args:
            idx: Token ID tensor of shape [B, T].
            targets: Optional target tensor of shape [B, T] for loss computation.

        Returns:
            (logits, loss) where logits has shape [B, T, vocab_size] and
            loss is a scalar or None if targets is not provided.
        """
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
                reduction="mean",
            )
        return logits, loss


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

def _zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """
    Orthogonalise G using Newton-Schulz iteration.

    Produces a matrix with approximately orthonormal columns/rows,
    used by the Muon optimizer to normalise gradient updates for
    2D weight matrices.
    """
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
    """
    Muon optimizer: orthogonalised momentum updates for 2D weights.

    2D weight matrices receive updates orthogonalised via Newton-Schulz
    iteration (approximating steepest descent in the spectral norm).
    1D parameters and embeddings are handled by an internal AdamW instance.

    Reference: https://github.com/KellerJordan/Muon
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        ns_steps: int = 5,
        adamw_params=None,
        adamw_lr: float = 3e-3,
        adamw_wd: float = 0.0,
        adamw_betas: tuple = (0.9, 0.95),
    ) -> None:
        defaults = dict(lr=lr, momentum=momentum, ns_steps=ns_steps)
        super().__init__(params, defaults)
        self.adamw = (
            torch.optim.AdamW(adamw_params, lr=adamw_lr, weight_decay=adamw_wd, betas=adamw_betas)
            if adamw_params is not None
            else None
        )

    @torch.no_grad()
    def step(self) -> None:
        """Perform a single optimisation step."""
        if self.adamw is not None:
            self.adamw.step()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "buf" not in state:
                    state["buf"] = torch.zeros_like(p)
                buf = state["buf"]
                buf.mul_(group["momentum"]).add_(p.grad)
                nesterov = p.grad + group["momentum"] * buf
                if p.ndim == 2:
                    update = _zeropower_via_newtonschulz5(nesterov, steps=group["ns_steps"])
                    update *= max(p.size(0), p.size(1)) ** 0.5
                else:
                    update = nesterov
                p.add_(update, alpha=-group["lr"])


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def build_optimizer(model: GPT, args: Hyperparameters) -> Muon:
    """
    Construct the Muon optimizer with an embedded AdamW for 1D params.

    2D weight matrices (excluding embeddings) use Muon with orthogonalised
    updates. Embeddings and 1D parameters (biases, LayerNorm, LayerScale)
    use AdamW with no weight decay.
    """
    muon_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and p.dim() == 2 and "token_emb" not in n
    ]
    adamw_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and (p.dim() < 2 or "token_emb" in n)
    ]
    return Muon(
        muon_params,
        lr=args.lr * args.muon_lr_ratio,
        momentum=args.muon_momentum,
        ns_steps=args.ns_steps,
        adamw_params=[{"params": adamw_params, "weight_decay": 0.0}],
        adamw_lr=args.lr,
        adamw_wd=0.0,
        adamw_betas=args.betas,
    )


def build_scheduler(
    opt: Muon,
    warmup_steps: int,
    max_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Build a linear warmup followed by constant LR scheduler.

    The LR ramps linearly from 0 to 1× over warmup_steps, then stays
    constant for the remainder of training. SWA provides implicit
    annealing over the final phase.
    """
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def evaluate(
    model: nn.Module,
    val_ids: torch.Tensor,
    val_text: str,
    block_size: int,
    batch_size: int,
    device: torch.device,
) -> float:
    """
    Compute validation loss normalised by the number of characters in val_text.

    NOTE: This function must not be modified — it is used for assessment scoring.
    """
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


def evaluate_with_perplexity(
    model: nn.Module,
    val_ids: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: torch.device,
) -> tuple[float, float]:
    """
    Compute per-token validation loss and perplexity.

    Returns:
        (loss_per_token, perplexity)
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for xb, yb in iter_full_split(val_ids, block_size, batch_size, device):
            logits, _ = model(xb, yb)
            B, T, V = logits.size()
            total_loss += F.cross_entropy(logits.view(-1, V), yb.view(-1), reduction="sum").item()
            total_tokens += B * T
    model.train()
    loss_per_token = total_loss / total_tokens
    perplexity = math.exp(min(loss_per_token, 20))
    return loss_per_token, perplexity


def evaluate_swa(
    swa_model: torch.optim.swa_utils.AveragedModel,
    val_ids: torch.Tensor,
    val_text: str,
    block_size: int,
    batch_size: int,
    device: torch.device,
) -> float:
    """Compute validation loss for the SWA-averaged model."""
    swa_model.eval()
    losses = 0.0
    with torch.no_grad():
        for xb, yb in iter_full_split(val_ids, block_size, batch_size, device):
            logits, _ = swa_model(xb, yb)
            B, T, V = logits.size()
            losses += F.cross_entropy(logits.view(-1, V), yb.view(-1), reduction="sum").item()
    swa_model.train()
    return losses / len(val_text)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point: set up data, model, optimizer, and run the training loop."""
    args = Hyperparameters()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    logger = configure_logging(args.log_file)
    logger.log("hyperparameters_configured", **vars(args))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.log("device_info", device=device)

    # --- Data ---
    train_titles, val_titles = get_titles(args.num_titles, args.seed, args.val_frac)

    eos_token = "<eos>"
    tok = BPETokenizer(
        train_tokenizer(train_titles + val_titles, args.vocab_size, eos_token=eos_token)
    )

    train_text = eos_token.join(train_titles) + eos_token
    val_text = eos_token.join(val_titles) + eos_token
    train_ids = torch.tensor(tok.encode(train_text), dtype=torch.long)
    val_ids = torch.tensor(tok.encode(val_text), dtype=torch.long)

    batches = len(train_ids) // (args.block_size * args.batch_size)
    max_steps = args.epochs * batches
    eval_interval = batches // args.evals_per_epoch
    swa_start = int(args.swa_start_frac * max_steps)
    warmup_steps = int(args.warmup_frac * max_steps)

    logger.log(
        "dataset_info",
        titles_count=len(train_titles),
        epochs=args.epochs,
        batches_per_epoch=batches,
        tokens_per_epoch=len(train_ids),
        vocab_size=tok.vocab_size,
    )

    # --- Model ---
    cfg = GPTConfig(
        vocab_size=tok.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        d_model=args.d_model,
        dropout=args.dropout,
        rope_base=args.rope_base,
        layer_scale_init=args.layer_scale_init,
    )
    model = GPT(cfg).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log("model_info", parameters_count=num_params)

    # --- Optimizer and scheduler ---
    opt = build_optimizer(model, args)
    scheduler = build_scheduler(opt, warmup_steps, max_steps)

    # --- SWA ---
    swa_model = torch.optim.swa_utils.AveragedModel(model)

    # --- Training loop ---
    best_val_loss = float("inf")
    ptr = 0
    step = 0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        for _ in tqdm(range(1, batches + 1), desc=f"Epoch {epoch}/{args.epochs}"):
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

            elapsed = time.time() - t0
            logger.log(
                "training_step",
                step=step,
                max_steps=max_steps,
                loss=loss.item(),
                elapsed_time=elapsed,
                prnt=False,
            )

            if step == 1 or step % eval_interval == 0 or step == max_steps:
                val_loss = evaluate(
                    model, val_ids, val_text,
                    args.block_size, args.batch_size, device,
                )
                _, perplexity = evaluate_with_perplexity(
                    model, val_ids, args.block_size, args.batch_size, device,
                )
                best_val_loss = min(best_val_loss, val_loss)
                logger.log(
                    "validation_step",
                    step=step,
                    max_steps=max_steps,
                    loss=val_loss,
                    perplexity=round(perplexity, 2),
                    elapsed_time=elapsed,
                )

                if step >= swa_start:
                    swa_val_loss = evaluate_swa(
                        swa_model, val_ids, val_text,
                        args.block_size, args.batch_size, device,
                    )
                    logger.log(
                        "swa_validation_step",
                        step=step,
                        max_steps=max_steps,
                        swa_val_loss=round(swa_val_loss, 6),
                        elapsed_time=elapsed,
                    )

    # --- Final SWA evaluation ---
    swa_val_loss = evaluate_swa(
        swa_model, val_ids, val_text,
        args.block_size, args.batch_size, device,
    )
    best_val_loss = min(best_val_loss, swa_val_loss)
    logger.log("final_swa_val_loss", loss=swa_val_loss)
    logger.close()


if __name__ == "__main__":
    main()