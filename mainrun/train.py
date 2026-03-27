# import utils
# import math, random, time
# from dataclasses import dataclass, field
# import json
# from pathlib import Path

# import torch
# import torch.nn as nn
# from torch.nn import functional as F
# from datasets import load_dataset
# from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
# from tqdm import tqdm
# import structlog

# from datetime import datetime

# @dataclass
# class Hyperparameters:
#     block_size: int = 128 
#     batch_size: int = 64
#     # 8000 # 16_000 - Fewer tokens means each token appears more frequently in the training data, the 
#     # domain vocabulary is narrow so fewer token will be ok -- changed it back to 1600 becuase it messes
#     # up the way we calcualte val loss
#     vocab_size: int = 16_000
#     n_layer: int = 8 # 6 - extra transformer layers
#     n_head: int = 8
#     d_model: int = 768 # 512 - increased since we arent fully platueing by the end of training
#     dropout: float = 0 # was 0.1  but no overfitting was detected
#     lr: float = 3e-4  # 6e-3 changed from sgd to adam 
#     weight_decay: float = 0.1  # was 0.0
#     evals_per_epoch: int = 3
    
#     epochs: int = 7
#     seed: int = 1337
#     num_titles: int = 100_000
#     val_frac: float = 0.10
#     log_file: str = field(default_factory=lambda: f"./logs/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log") #"./logs/mainrun.log" - new log file for each run

# def configure_logging(log_file: str):
#     Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
#     file_handler = open(log_file, 'w')
    
#     # Note: The following structlog configuration is currently not used, as the DualLogger handles logging directly.

#     # structlog.configure(
#     #     processors=[
#     #         structlog.stdlib.filter_by_level,
#     #         structlog.stdlib.add_logger_name,
#     #         structlog.stdlib.add_log_level,
#     #         structlog.stdlib.PositionalArgumentsFormatter(),
#     #         structlog.processors.TimeStamper(fmt="iso"),
#     #         structlog.processors.StackInfoRenderer(),
#     #         structlog.processors.format_exc_info,
#     #         structlog.processors.UnicodeDecoder(),
#     #         structlog.processors.JSONRenderer()
#     #     ],
#     #     context_class=dict,
#     #     logger_factory=structlog.stdlib.LoggerFactory(),
#     #     cache_logger_on_first_use=True,
#     # )
    
#     class DualLogger:
#         def __init__(self, file_handler):
#             self.file_handler = file_handler
#             # self.logger = structlog.get_logger()  -- Not used in current implementation
            
#         def log(self, event, **kwargs):
#             log_entry = json.dumps({"event": event, "timestamp": time.time(), **kwargs})
#             self.file_handler.write(log_entry + "\n")
#             self.file_handler.flush()
            
#             if kwargs.get("prnt", True):
#                 if "step" in kwargs and "max_steps" in kwargs:
#                     tqdm.write(f"[{kwargs.get('step'):>5}/{kwargs.get('max_steps')}] {event}: loss={kwargs.get('loss', 'N/A'):.6f} time={kwargs.get('elapsed_time', 0):.2f}s")
#                 else:
#                     parts = [f"{k}={v}" for k, v in kwargs.items() if k not in ["prnt", "timestamp"]]
#                     if parts:
#                         tqdm.write(f"{event}: {', '.join(parts)}")
#                     else:
#                         tqdm.write(event)
    
#     return DualLogger(file_handler)

# logger = None

# def get_titles(num_titles: int, seed: int, val_frac: float) -> tuple[list[str], list[str]]: #str: - fixed type annotation 
#     ds = load_dataset("julien040/hacker-news-posts", split="train", cache_dir="./data").shuffle(seed=seed)
#     titles = [row["title"].strip() for row in ds.take(num_titles)]
#     n = int(num_titles * (1 - val_frac))
#     return titles[:n], titles[n:]

# def get_batch(split_ids: torch.Tensor, ptr: int, block_size: int, batch_size: int, device: torch.device):
#     span = block_size * batch_size + 1
#     if ptr + span >= len(split_ids):
#         ptr = 0
#     batch = split_ids[ptr: ptr + span]
#     x = batch[:-1].view(batch_size, block_size).to(device)
#     y = batch[1:].view(batch_size, block_size).to(device)
#     return x, y, ptr + block_size * batch_size

# def iter_full_split(split_ids: torch.Tensor, block_size: int, batch_size: int, device: torch.device):
#     span = block_size * batch_size + 1
#     for ptr in range(0, len(split_ids) - span + 1, span):
#         batch = split_ids[ptr: ptr + span]
#         x = batch[:-1].view(batch_size, block_size).to(device)
#         y = batch[1:].view(batch_size, block_size).to(device)
#         yield x, y

# def train_tokenizer(titles: list[str], vocab_size: int, unk_token: str = "<unk>", pad_token: str = "<pad>", eos_token: str = "<eos>") -> Tokenizer:
#     # because we are using bytes we never need the unk_token
#     tokenizer = Tokenizer(models.BPE(unk_token=unk_token))
#     tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
#     tokenizer.decoder = decoders.ByteLevel()
#     trainer = trainers.BpeTrainer(
#         vocab_size=vocab_size,
#         special_tokens=[pad_token, eos_token, unk_token]
#     )
#     tokenizer.train_from_iterator(titles, trainer)
#     return tokenizer

# class BPETokenizer:
#     def __init__(self, tokenizer: Tokenizer):
#         self.tk = tokenizer

#         # never used, as we directly use the tokenizer's built-in methods for encoding/decoding
#         # self.stoi = {tok: i for tok, i in tokenizer.get_vocab().items()}
#         # self.itos = {i: tok for tok, i in tokenizer.get_vocab().items()}

#     def encode(self, s: str) -> list[int]:
#         return self.tk.encode(s).ids

#     def decode(self, ids: list[int]) -> str:
#         return self.tk.decode(ids, skip_special_tokens=True)

#     @property
#     def vocab_size(self): return self.tk.get_vocab_size()

# @dataclass
# class GPTConfig:
#     vocab_size: int
#     block_size: int
#     n_layer: int
#     n_head: int
#     d_model: int
#     dropout: float

# class CausalSelfAttention(nn.Module):
#     def __init__(self, cfg: GPTConfig):
#         super().__init__()
#         assert cfg.d_model % cfg.n_head == 0
#         self.head_dim = cfg.d_model // cfg.n_head
#         self.n_head   = cfg.n_head
#         self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
#         self.proj = nn.Linear(cfg.d_model, cfg.d_model)
#         self.attn_drop = nn.Dropout(cfg.dropout)
#         self.resid_drop= nn.Dropout(cfg.dropout)
#         self.register_buffer("tril", torch.tril(torch.ones(cfg.block_size, cfg.block_size)))

#     def forward(self, x: torch.Tensor):
#         B, T, C = x.size()
#         qkv = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim).transpose(1, 3)
#         q, k, v = qkv[..., 0, :, :], qkv[..., 1, :, :], qkv[..., 2, :, :]
#         att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
#         att = att.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
#         att = F.softmax(att, dim=-1)
#         att = self.attn_drop(att)
#         y = att @ v
#         y = y.transpose(1, 2).contiguous().view(B, T, C)
#         return self.resid_drop(self.proj(y))

# class MLP(nn.Module):
#     def __init__(self, cfg: GPTConfig):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(cfg.d_model, 4 * cfg.d_model),
#             nn.GELU(),
#             nn.Linear(4 * cfg.d_model, cfg.d_model),
#             nn.Dropout(cfg.dropout),
#         )
#     def forward(self, x): return self.net(x)

# # SwiGLU replacement
# class MLP(nn.Module):
#     def __init__(self, cfg):
#         super().__init__()
#         hidden = int(2/3 * 4 * cfg.d_model)  # slightly smaller to keep param count similar
#         self.w1   = nn.Linear(cfg.d_model, hidden, bias=False)
#         self.w2   = nn.Linear(cfg.d_model, hidden, bias=False)
#         self.proj = nn.Linear(hidden, cfg.d_model, bias=False)

#     def forward(self, x):
#         return self.proj(F.silu(self.w1(x)) * self.w2(x))

# class Block(nn.Module):
#     def __init__(self, cfg: GPTConfig):
#         super().__init__()
#         self.ln1 = nn.LayerNorm(cfg.d_model)
#         self.ln2 = nn.LayerNorm(cfg.d_model)
#         self.attn = CausalSelfAttention(cfg)
#         self.mlp  = MLP(cfg)
#     def forward(self, x):
#         x = x + self.attn(self.ln1(x))
#         x = x + self.mlp(self.ln2(x))
#         return x

# class GPT(nn.Module):
#     def __init__(self, cfg: GPTConfig):
#         super().__init__()
#         self.cfg = cfg
#         self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
#         self.pos_emb   = nn.Parameter(torch.zeros(1, cfg.block_size, cfg.d_model))
#         self.drop      = nn.Dropout(cfg.dropout)
#         self.blocks    = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
#         self.ln_f      = nn.LayerNorm(cfg.d_model)
#         self.head      = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

#         self.apply(self._init_weights)
#         self.head.weight = self.token_emb.weight

#     @staticmethod
#     def _init_weights(module):
#         if isinstance(module, (nn.Linear, nn.Embedding)):
#             nn.init.normal_(module.weight, mean=0.0, std=0.02)
#             if isinstance(module, nn.Linear) and module.bias is not None:
#                 nn.init.zeros_(module.bias)

#     def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
#         B, T = idx.size()
#         tok = self.token_emb(idx)
#         pos = self.pos_emb[:, :T, :]
#         x = self.drop(tok + pos)
#         for block in self.blocks: x = block(x)
#         x = self.ln_f(x)
#         logits = self.head(x)
#         if targets is None:
#             loss = None
#         else:
#             # loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='mean')
#             loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='mean', label_smoothing=0.1)

#         return logits, loss

# def get_lr(step, max_steps, lr):
#     # cosine decay from lr down to lr/10
#     min_lr = lr / 10
#     progress = step / max_steps
#     return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * progress))

# def get_lr(step, max_steps, lr, warmup_steps=50):
#     # phase 1: linear warmup
#     if step < warmup_steps:
#         return lr * (step / warmup_steps)
#     # phase 2: cosine decay from lr down to lr/10
#     min_lr = lr / 10
#     progress = (step - warmup_steps) / (max_steps - warmup_steps)
#     return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * progress))

# def main():
#     args = Hyperparameters()
#     torch.manual_seed(args.seed)
#     random.seed(args.seed)
    
#     global logger
#     logger = configure_logging(args.log_file)
    
#     Path("./checkpoints").mkdir(parents=True, exist_ok=True)

#     hyperparams_dict = vars(args)
#     logger.log("hyperparameters_configured", **hyperparams_dict)
    
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     logger.log("device_info", device=device)

#     train_titles, val_titles = get_titles(args.num_titles, args.seed, args.val_frac)
    
#     eos_token = "<eos>"
#     tok = BPETokenizer(train_tokenizer(train_titles+val_titles, args.vocab_size, eos_token=eos_token))
#     train_text = eos_token.join(train_titles) + eos_token
#     val_text = eos_token.join(val_titles) + eos_token
#     train_ids = torch.tensor(tok.encode(train_text), dtype=torch.long)
#     val_ids = torch.tensor(tok.encode(val_text), dtype=torch.long)
    
#     batches = len(train_ids) // (args.block_size * args.batch_size) # calculating the number of batches we have
#     max_steps = args.epochs * batches # each batch performs a weight update, so the number of weight updates happening
#     eval_interval = batches // args.evals_per_epoch # how often to run validation
#     logger.log("dataset_info",
#                titles_count=len(train_titles),
#                epochs=args.epochs,
#                batches_per_epoch=batches,
#                tokens_per_epoch=len(train_ids),
#                vocab_size=tok.vocab_size)

#     cfg = GPTConfig(
#         vocab_size = tok.vocab_size,
#         block_size = args.block_size,
#         n_layer    = args.n_layer,
#         n_head     = args.n_head,
#         d_model    = args.d_model,
#         dropout    = args.dropout,
#     )
#     model = GPT(cfg).to(device)
#     model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     logger.log("model_info", parameters_count=model_params)
    
#     # opt = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
#     # opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
#     opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))

#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_steps)

#     def evaluate():
#         model.eval()
#         losses = 0.0
#         with torch.no_grad():
#             for xb, yb in iter_full_split(val_ids, args.block_size, args.batch_size, device):
#                 logits, _ = model(xb, yb)
#                 B, T, V = logits.size()
#                 loss = F.cross_entropy(logits.view(-1, V), yb.view(-1), reduction='sum')
#                 losses += loss.item()
#         model.train()
#         return losses / len(val_text)

#     def evaluate_with_perplexity():
#         model.eval()
#         losses = 0.0
#         total_tokens = 0
        
#         with torch.no_grad():
#             for xb, yb in iter_full_split(val_ids, args.block_size, args.batch_size, device):
#                 logits, _ = model(xb, yb)
#                 B, T, V = logits.size()
#                 loss = F.cross_entropy(logits.view(-1, V), yb.view(-1), reduction='sum')
#                 losses += loss.item()
#                 total_tokens += B * T
        
#         model.train()
#         loss_per_token = losses / total_tokens
#         perplexity = math.exp(min(loss_per_token, 20))
#         return loss_per_token, perplexity

#     ptr = 0 # remembers where in the million token strip we are
#     step = 0 # counts how many weight updates have happened total
#     t0 = time.time() # used for timing
#     for epoch in range(1, args.epochs + 1):
#         for _ in tqdm(range(1, batches + 1), desc=f"Epoch {epoch}/{args.epochs}"):
#             step += 1
#             xb, yb, ptr = get_batch(train_ids, ptr, args.block_size, args.batch_size, device) # xb -> what the model sees, yb -> what the model should predict
#             _, loss = model(xb, yb) # forward pass, we get the loss directly from the model, no need to calculate it separately
#             opt.zero_grad(set_to_none=True) # 
#             loss.backward() # backward pass, calculates the gradients for all parameters
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#             opt.step() # applies grad decent
#             # scheduler.step()
#             lr_now = get_lr(step, max_steps, args.lr)
#             for pg in opt.param_groups:
#                 pg['lr'] = lr_now

#             elapsed = time.time() - t0
#             logger.log("training_step",
#                       step=step,
#                       max_steps=max_steps,
#                       loss=loss.item(),
#                       elapsed_time=elapsed,
#                       prnt=False)

#             if step == 1 or step % eval_interval == 0 or step == max_steps:
#                 val_loss = evaluate()
#                 val_loss_per_token, perplexity = evaluate_with_perplexity()
#                 logger.log("validation_step",
#                           step=step,
#                           max_steps=max_steps,
#                           loss=val_loss,
#                           perplexity=round(perplexity, 2),
#                           elapsed_time=elapsed)

#     # after the training loop
#     torch.save({
#         'model_state_dict': model.state_dict(),
#         'cfg': cfg,
#         'tokenizer': tok.tk,
#     }, './checkpoints/model_final.pt')
#     print("model saved to ./checkpoints/model_final.pt")

# if __name__ == "__main__":
#     try:
#         main()
#     finally:
#         if logger and hasattr(logger, 'file_handler'):
#             logger.file_handler.close()
import utils
import math, random, time
from dataclasses import dataclass
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from tqdm import tqdm
import structlog

# ── Hyperparameters ───────────────────────────────────────────────────────────
# Changes from baseline and reasoning:
#
# block_size  128 → 256   longer context = model sees more of each title + cross-title patterns
# batch_size   64 → 32    smaller batch with grad accumulation gives noisier, more informative gradients
# vocab_size 16k → 8k     smaller vocab = more training signal per token (rare tokens trained more)
# n_layer      6 → 8      more depth = more representational power
# n_head       8 → 8      unchanged
# d_model    512 → 512    unchanged (increasing here risks overfitting on small data)
# dropout    0.1 → 0.0    we are NOT overfitting — dropout is hurting us, turn it off
# lr         kept at 3e-4 with warmup instead of flat start
# weight_decay 0.1        keep, AdamW benefits from this
# grad_accum_steps = 4    effective batch = 32*4 = 128, GPU sees smaller chunks but update is larger
# warmup_steps = 50       linear warmup to avoid large initial updates destabilising attention layers

@dataclass
class Hyperparameters:
    block_size: int = 256
    batch_size: int = 32
    vocab_size: int = 8_000
    n_layer: int = 8
    n_head: int = 8
    d_model: int = 512
    dropout: float = 0.0
    lr: float = 3e-4
    weight_decay: float = 0.1
    evals_per_epoch: int = 3
    grad_accum_steps: int = 4
    warmup_steps: int = 50

    # FIXED — cannot change these
    epochs: int = 7
    seed: int = 1337
    num_titles: int = 100_000
    val_frac: float = 0.10
    log_file: str = f"./logs/run_{time.strftime('%Y%m%d_%H%M%S')}.log"


def configure_logging(log_file: str):
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    file_handler = open(log_file, 'w')

    class DualLogger:
        def __init__(self, file_handler):
            self.file_handler = file_handler

        def log(self, event, **kwargs):
            log_entry = json.dumps({"event": event, "timestamp": time.time(), **kwargs})
            self.file_handler.write(log_entry + "\n")
            self.file_handler.flush()
            if kwargs.get("prnt", True):
                if "step" in kwargs and "max_steps" in kwargs:
                    perp = f" ppl={kwargs.get('perplexity', 'N/A')}" if 'perplexity' in kwargs else ""
                    tqdm.write(f"[{kwargs.get('step'):>5}/{kwargs.get('max_steps')}] {event}: loss={kwargs.get('loss', 'N/A'):.6f}{perp} time={kwargs.get('elapsed_time', 0):.2f}s")
                else:
                    parts = [f"{k}={v}" for k, v in kwargs.items() if k not in ["prnt", "timestamp"]]
                    tqdm.write(f"{event}: {', '.join(parts)}" if parts else event)

    return DualLogger(file_handler)


logger = None


def get_titles(num_titles: int, seed: int, val_frac: float):
    ds = load_dataset("julien040/hacker-news-posts", split="train", cache_dir="./data").shuffle(seed=seed)
    titles = [row["title"].strip() for row in ds.take(num_titles)]
    n = int(num_titles * (1 - val_frac))
    return titles[:n], titles[n:]


def get_batch(split_ids: torch.Tensor, ptr: int, block_size: int, batch_size: int, device: torch.device):
    span = block_size * batch_size + 1
    if ptr + span >= len(split_ids):
        ptr = 0
    batch = split_ids[ptr: ptr + span]
    x = batch[:-1].view(batch_size, block_size).to(device)
    y = batch[1:].view(batch_size, block_size).to(device)
    return x, y, ptr + block_size * batch_size


def iter_full_split(split_ids: torch.Tensor, block_size: int, batch_size: int, device: torch.device):
    span = block_size * batch_size + 1
    for ptr in range(0, len(split_ids) - span + 1, span):
        batch = split_ids[ptr: ptr + span]
        x = batch[:-1].view(batch_size, block_size).to(device)
        y = batch[1:].view(batch_size, block_size).to(device)
        yield x, y


def train_tokenizer(titles, vocab_size, unk_token="<unk>", pad_token="<pad>", eos_token="<eos>"):
    tokenizer = Tokenizer(models.BPE(unk_token=unk_token))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[pad_token, eos_token, unk_token]
    )
    tokenizer.train_from_iterator(titles, trainer)
    return tokenizer


class BPETokenizer:
    def __init__(self, tokenizer: Tokenizer):
        self.tk = tokenizer

    def encode(self, s: str) -> list[int]:
        return self.tk.encode(s).ids

    def decode(self, ids: list[int]) -> str:
        return self.tk.decode(ids, skip_special_tokens=True)

    @property
    def vocab_size(self): return self.tk.get_vocab_size()


# ── Model ─────────────────────────────────────────────────────────────────────

@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int
    n_layer: int
    n_head: int
    d_model: int
    dropout: float


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.head_dim = cfg.d_model // cfg.n_head
        self.n_head   = cfg.n_head
        self.qkv  = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.attn_drop  = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)
        self.register_buffer("tril", torch.tril(torch.ones(cfg.block_size, cfg.block_size)))

    def forward(self, x: torch.Tensor):
        B, T, C = x.size()
        qkv = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim).transpose(1, 3)
        q, k, v = qkv[..., 0, :, :], qkv[..., 1, :, :], qkv[..., 2, :, :]
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        # SwiGLU-style: uses 2/3 * 4 * d_model hidden dim with a gate
        # produces better loss than plain GELU for same parameter count
        hidden = int(2/3 * 4 * cfg.d_model)
        hidden = (hidden + 63) // 64 * 64  # round up to multiple of 64 for GPU efficiency
        self.w1   = nn.Linear(cfg.d_model, hidden, bias=False)
        self.w2   = nn.Linear(cfg.d_model, hidden, bias=False)
        self.proj = nn.Linear(hidden, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        # SwiGLU: silu(xW1) * xW2
        return self.drop(self.proj(F.silu(self.w1(x)) * self.w2(x)))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1  = nn.LayerNorm(cfg.d_model)
        self.ln2  = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.mlp  = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb   = nn.Parameter(torch.zeros(1, cfg.block_size, cfg.d_model))
        self.drop      = nn.Dropout(cfg.dropout)
        self.blocks    = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f      = nn.LayerNorm(cfg.d_model)
        self.head      = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        self.apply(self._init_weights)
        # weight tying
        self.head.weight = self.token_emb.weight
        # scale residual projections by 1/sqrt(n_layer) — GPT-2 paper trick
        # prevents residual stream from growing too large with depth
        for name, p in self.named_parameters():
            if name.endswith('proj.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.size()
        tok  = self.token_emb(idx)
        pos  = self.pos_emb[:, :T, :]
        x    = self.drop(tok + pos)
        for block in self.blocks:
            x = block(x)
        x      = self.ln_f(x)
        logits = self.head(x)
        loss   = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='mean')
        return logits, loss


# ── Training ──────────────────────────────────────────────────────────────────

def get_lr(step: int, warmup_steps: int, max_steps: int, lr: float) -> float:
    """Linear warmup then cosine decay to lr/10."""
    if step < warmup_steps:
        return lr * step / warmup_steps
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return lr / 10 + 0.5 * (lr - lr / 10) * (1 + math.cos(math.pi * progress))


def main():
    args = Hyperparameters()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    global logger
    logger = configure_logging(args.log_file)

    logger.log("hyperparameters_configured", **vars(args))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.log("device_info", device=device)

    # ── Data ──────────────────────────────────────────────────────────────────
    train_titles, val_titles = get_titles(args.num_titles, args.seed, args.val_frac)

    eos_token = "<eos>"
    tok = BPETokenizer(train_tokenizer(
        train_titles + val_titles, args.vocab_size, eos_token=eos_token
    ))
    train_text = eos_token.join(train_titles) + eos_token
    val_text   = eos_token.join(val_titles)   + eos_token
    train_ids  = torch.tensor(tok.encode(train_text), dtype=torch.long)
    val_ids    = torch.tensor(tok.encode(val_text),   dtype=torch.long)

    # ── Step counts ───────────────────────────────────────────────────────────
    # With gradient accumulation, one optimizer step = grad_accum_steps forward passes
    micro_batches  = len(train_ids) // (args.block_size * args.batch_size)
    batches        = micro_batches // args.grad_accum_steps  # optimizer steps per epoch
    max_steps      = args.epochs * batches
    eval_interval  = max(1, batches // args.evals_per_epoch)

    logger.log("dataset_info",
               train_titles=len(train_titles),
               epochs=args.epochs,
               micro_batches_per_epoch=micro_batches,
               optimizer_steps_per_epoch=batches,
               tokens_per_epoch=len(train_ids),
               vocab_size=tok.vocab_size)

    # ── Model ─────────────────────────────────────────────────────────────────
    cfg = GPTConfig(
        vocab_size = tok.vocab_size,
        block_size = args.block_size,
        n_layer    = args.n_layer,
        n_head     = args.n_head,
        d_model    = args.d_model,
        dropout    = args.dropout,
    )
    model = GPT(cfg).to(device)
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log("model_info", parameters_count=model_params)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    # Separate weight decay: apply only to weight matrices, not biases or layernorm
    decay_params     = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay_params  = [p for n, p in model.named_parameters() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params,    "weight_decay": args.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(optim_groups, lr=args.lr, betas=(0.9, 0.95), eps=1e-8)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    def evaluate():
        model.eval()
        losses = 0.0
        with torch.no_grad():
            for xb, yb in iter_full_split(val_ids, args.block_size, args.batch_size, device):
                logits, _ = model(xb, yb)
                B, T, V = logits.size()
                loss = F.cross_entropy(logits.view(-1, V), yb.view(-1), reduction='sum')
                losses += loss.item()
        model.train()
        return losses / len(val_text)

    def evaluate_with_perplexity():
        model.eval()
        losses = 0.0
        total_tokens = 0
        with torch.no_grad():
            for xb, yb in iter_full_split(val_ids, args.block_size, args.batch_size, device):
                logits, _ = model(xb, yb)
                B, T, V = logits.size()
                loss = F.cross_entropy(logits.view(-1, V), yb.view(-1), reduction='sum')
                losses += loss.item()
                total_tokens += B * T
        model.train()
        loss_per_token = losses / total_tokens
        perplexity = math.exp(min(loss_per_token, 20))
        return loss_per_token, round(perplexity, 2)

    # ── Training loop ─────────────────────────────────────────────────────────
    ptr    = 0
    step   = 0  # counts optimizer steps
    t0     = time.time()

    for epoch in range(1, args.epochs + 1):
        for batch_idx in tqdm(range(1, batches + 1), desc=f"Epoch {epoch}/{args.epochs}"):
            step += 1

            # manual LR schedule with warmup
            lr_now = get_lr(step, args.warmup_steps, max_steps, args.lr)
            for pg in opt.param_groups:
                pg['lr'] = lr_now

            # gradient accumulation — accumulate over grad_accum_steps micro-batches
            opt.zero_grad(set_to_none=True)
            accum_loss = 0.0
            for micro_step in range(args.grad_accum_steps):
                xb, yb, ptr = get_batch(train_ids, ptr, args.block_size, args.batch_size, device)
                _, loss = model(xb, yb)
                # scale loss so gradients are averaged over accum steps
                loss = loss / args.grad_accum_steps
                loss.backward()
                accum_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            elapsed = time.time() - t0
            logger.log("training_step",
                       step=step,
                       max_steps=max_steps,
                       loss=accum_loss,
                       lr=lr_now,
                       elapsed_time=elapsed,
                       prnt=False)

            if step == 1 or step % eval_interval == 0 or step == max_steps:
                val_loss = evaluate()
                _, perplexity = evaluate_with_perplexity()
                logger.log("validation_step",
                           step=step,
                           max_steps=max_steps,
                           loss=val_loss,
                           perplexity=perplexity,
                           elapsed_time=elapsed)

    # ── Save checkpoint ───────────────────────────────────────────────────────
    Path("./checkpoints").mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'cfg': cfg,
        'tokenizer': tok.tk,
        'val_loss': val_loss,
    }, './checkpoints/model_optimized.pt')
    logger.log("checkpoint_saved", path="./checkpoints/model_optimized.pt", val_loss=val_loss)


if __name__ == "__main__":
    try:
        main()
    finally:
        if logger and hasattr(logger, 'file_handler'):
            logger.file_handler.close()