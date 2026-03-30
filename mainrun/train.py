import utils
import math, random, time
from dataclasses import dataclass, field
import json
from pathlib import Path

from torch.utils.checkpoint import checkpoint
import torch
import torch.nn as nn
from torch.nn import functional as F
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from tqdm import tqdm
import structlog

from datetime import datetime

"""
changelog:

1. changed adams optimiser
"""

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
# @dataclass
# class Hyperparameters:
#     block_size: int = 128
#     batch_size: int = 128 #64 
#     vocab_size: int = 16_000
#     n_layer: int = 8 #6
#     n_head: int = 8
#     d_model: int = 512
#     dropout: float = 0.1
#     lr: float = 3e-4
#     weight_decay: float = 0.1
#     evals_per_epoch: int = 3
#     warmup_steps: int = 100
#     eta_min: float = 3e-5 
    
#     epochs: int = 7
#     seed: int = 1337
#     num_titles: int = 100_000
#     val_frac: float = 0.10
@dataclass
class Hyperparameters:
    # Data (DO NOT CHANGE — assessment rules)
    num_titles: int = 100_000
    val_frac: float = 0.10
    seed: int = 1337
    epochs: int = 7               # DO NOT CHANGE — assessment rule
    vocab_size: int = 16_000

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
    log_file: str = field(default_factory=lambda: f"./logs/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log") #"./logs/mainrun.log" - new log file for each run

# sets up logging
def configure_logging(log_file: str):
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    file_handler = open(log_file, 'w')
    
    # Note: The following structlog configuration is currently not used, as the DualLogger handles logging directly.

    # structlog.configure(
    #     processors=[
    #         structlog.stdlib.filter_by_level,
    #         structlog.stdlib.add_logger_name,
    #         structlog.stdlib.add_log_level,
    #         structlog.stdlib.PositionalArgumentsFormatter(),
    #         structlog.processors.TimeStamper(fmt="iso"),
    #         structlog.processors.StackInfoRenderer(),
    #         structlog.processors.format_exc_info,
    #         structlog.processors.UnicodeDecoder(),
    #         structlog.processors.JSONRenderer()
    #     ],
    #     context_class=dict,
    #     logger_factory=structlog.stdlib.LoggerFactory(),
    #     cache_logger_on_first_use=True,
    # )
    
    class DualLogger:
        def __init__(self, file_handler):
            self.file_handler = file_handler
            # self.logger = structlog.get_logger()  -- Not used in current implementation
            
        # def log(self, event, **kwargs):
        #     log_entry = json.dumps({"event": event, "timestamp": time.time(), **kwargs})
        #     self.file_handler.write(log_entry + "\n")
        #     self.file_handler.flush()
            
        #     if kwargs.get("prnt", True):
        #         if "step" in kwargs and "max_steps" in kwargs:
        #             tqdm.write(f"[{kwargs.get('step'):>5}/{kwargs.get('max_steps')}] {event}: loss={kwargs.get('loss', 'N/A'):.6f} time={kwargs.get('elapsed_time', 0):.2f}s")
        #         else:
        #             parts = [f"{k}={v}" for k, v in kwargs.items() if k not in ["prnt", "timestamp"]]
        #             if parts:
        #                 tqdm.write(f"{event}: {', '.join(parts)}")
        #             else:
        #                 tqdm.write(event)

        def log(self, event, **kwargs):
            log_entry = json.dumps({"event": event, "timestamp": time.time(), **kwargs})
            self.file_handler.write(log_entry + "\n")
            self.file_handler.flush()
            
            if kwargs.get("prnt", True):
                if "step" in kwargs and "max_steps" in kwargs:
                    loss_val = kwargs.get('loss') or kwargs.get('swa_val_loss', 'N/A')
                    loss_str = f"{loss_val:.6f}" if isinstance(loss_val, float) else str(loss_val)
                    tqdm.write(f"[{kwargs.get('step'):>5}/{kwargs.get('max_steps')}] {event}: loss={loss_str} time={kwargs.get('elapsed_time', 0):.2f}s")
                else:
                    parts = [f"{k}={v}" for k, v in kwargs.items() if k not in ["prnt", "timestamp"]]
                    if parts:
                        tqdm.write(f"{event}: {', '.join(parts)}")
                    else:
                        tqdm.write(event)
    
    return DualLogger(file_handler)

logger = None

def get_titles(num_titles: int, seed: int, val_frac: float) -> tuple[list[str], list[str]]: #str: - fixed type annotation 
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

def train_tokenizer(titles: list[str], vocab_size: int, unk_token: str = "<unk>", pad_token: str = "<pad>", eos_token: str = "<eos>") -> Tokenizer:
    # because we are using bytes we never need the unk_token
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

        # never used, as we directly use the tokenizer's built-in methods for encoding/decoding
        # self.stoi = {tok: i for tok, i in tokenizer.get_vocab().items()}
        # self.itos = {i: tok for tok, i in tokenizer.get_vocab().items()}

    def encode(self, s: str) -> list[int]:
        return self.tk.encode(s).ids

    def decode(self, ids: list[int]) -> str:
        return self.tk.decode(ids, skip_special_tokens=True)

    @property
    def vocab_size(self): return self.tk.get_vocab_size()

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
        # super().__init__()
        # assert cfg.d_model % cfg.n_head == 0
        # self.head_dim = cfg.d_model // cfg.n_head
        # self.n_head   = cfg.n_head
        # self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        # self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        # self.attn_drop = nn.Dropout(cfg.dropout)
        # self.resid_drop= nn.Dropout(cfg.dropout)
        # self.register_buffer("tril", torch.tril(torch.ones(cfg.block_size, cfg.block_size)))
        # self.dropout = cfg.dropout
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.head_dim = cfg.d_model // cfg.n_head
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.proj.RESIDUAL_SCALE_INIT = 0.02 / math.sqrt(2 * cfg.n_layer)
        self.dropout = cfg.dropout
        self.resid_drop = nn.Dropout(cfg.dropout)

    # def forward(self, x: torch.Tensor):
    #     B, T, C = x.size()
    #     qkv = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim).transpose(1, 3)
    #     q, k, v = qkv[..., 0, :, :], qkv[..., 1, :, :], qkv[..., 2, :, :]
        
    #     # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    #     # att = att.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
    #     # att = F.softmax(att, dim=-1)
    #     # att = self.attn_drop(att)
    #     # y = att @ v

    #     y = F.scaled_dot_product_attention(q, k, v,
    #     dropout_p=self.dropout if self.training else 0.0,
    #     is_causal=True)

    #     y = y.transpose(1, 2).contiguous().view(B, T, C)
    #     return self.resid_drop(self.proj(y))

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

        # self.net = nn.Sequential(
        #     nn.Linear(cfg.d_model, 4 * cfg.d_model),
        #     nn.GELU(),
        #     nn.Linear(4 * cfg.d_model, cfg.d_model),
        #     nn.Dropout(cfg.dropout),
        # )

    # def forward(self, x): return self.net(x)
    def forward(self, x):
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))

class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        # super().__init__()
        # self.ln1 = nn.LayerNorm(cfg.d_model)
        # self.ln2 = nn.LayerNorm(cfg.d_model)
        # self.attn = CausalSelfAttention(cfg)
        # self.mlp  = MLP(cfg)
        super().__init__()
        self.ln = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.mlp = MLP(cfg)
        ls_init = 0.001
        self.ls_attn = nn.Parameter(ls_init * torch.ones(cfg.d_model))
        self.ls_mlp  = nn.Parameter(ls_init * torch.ones(cfg.d_model))

    # def forward(self, x):
    #     x = x + self.attn(self.ln1(x))
    #     x = x + self.mlp(self.ln2(x))
    #     return x

    def forward(self, x, freqs_cis):
        ln_out = self.ln(x)
        return x + self.ls_attn * self.attn(ln_out, freqs_cis) + self.ls_mlp * self.mlp(ln_out)

class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        # super().__init__()
        # self.cfg = cfg
        # self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        # # self.pos_emb   = nn.Parameter(torch.zeros(1, cfg.block_size, cfg.d_model))
        # freqs = precompute_rope_freqs(head_dim, cfg.block_size)
        # self.register_buffer("freqs_cis", freqs)
        # self.drop      = nn.Dropout(cfg.dropout)
        # self.blocks    = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        # self.ln_f      = nn.LayerNorm(cfg.d_model)
        # self.head      = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # self.apply(self._init_weights)
        # self.head.weight = self.token_emb.weight
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
    # def _init_weights(module):
    #     if isinstance(module, (nn.Linear, nn.Embedding)):
    #         nn.init.normal_(module.weight, mean=0.0, std=0.02)
    #         if isinstance(module, nn.Linear) and module.bias is not None:
    #             nn.init.zeros_(module.bias)

    # def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
    #     B, T = idx.size()
    #     tok = self.token_emb(idx)
    #     pos = self.pos_emb[:, :T, :]
    #     x = self.drop(tok + pos)
    #     # for block in self.blocks: x = block(x)
    #     for block in self.blocks:
    #         x = checkpoint(block, x, self.freqs_cis, use_reentrant=False)
    #     x = self.ln_f(x)
    #     logits = self.head(x)
    #     if targets is None:
    #         loss = None
    #     else:
    #         loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='mean')
    #     return logits, loss
    def forward(self, idx, targets=None): #blah
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

def main():
    args = Hyperparameters() # get the hyper parameters 

    # for reproducibility, we set the random seed for both torch and random modules
    torch.manual_seed(args.seed) 
    random.seed(args.seed)
    
    # set up logging
    global logger
    logger = configure_logging(args.log_file)
    
    hyperparams_dict = vars(args) # convert the dataclass to a dictionary
    logger.log("hyperparameters_configured", **hyperparams_dict) # logs all the hyper params
    
    device = "cuda" if torch.cuda.is_available() else "cpu" # checks for gpu
    logger.log("device_info", device=device) # logs device info

    train_titles, val_titles = get_titles(args.num_titles, args.seed, args.val_frac) # extracts the titles from the dataset
    
    eos_token = "<eos>" # eos token is used to seperate titles
    tok = BPETokenizer(train_tokenizer(train_titles+val_titles, args.vocab_size, eos_token=eos_token)) 

    # takes all the titles and puts them in a long string sperated by the eos token
    train_text = eos_token.join(train_titles) + eos_token
    val_text = eos_token.join(val_titles) + eos_token

    # makes ids for the words and puts it in a tensor
    train_ids = torch.tensor(tok.encode(train_text), dtype=torch.long)
    val_ids = torch.tensor(tok.encode(val_text), dtype=torch.long)
    
    batches = len(train_ids) // (args.block_size * args.batch_size) # calculating the number of batches we have
    max_steps = args.epochs * batches # each batch performs a weight update, so the number of weight updates happening
    eval_interval = batches // args.evals_per_epoch # how often to run validation
    logger.log("dataset_info",
               titles_count=len(train_titles),
               epochs=args.epochs,
               batches_per_epoch=batches,
               tokens_per_epoch=len(train_ids),
               vocab_size=tok.vocab_size)

    cfg = GPTConfig(
        vocab_size = tok.vocab_size,
        block_size = args.block_size,
        n_layer    = args.n_layer,
        n_head     = args.n_head,
        d_model    = args.d_model,
        dropout    = args.dropout,
    )
    model = GPT(cfg).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # log(fh, "model_info", num_params=num_params)
    logger.log("model_info", parameters_count=num_params)
    
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

    # --- SWA setup: average weights over last 60% of training ---
    swa_start = int(0.40 * max_steps)
    swa_model = torch.optim.swa_utils.AveragedModel(model)

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
        return loss_per_token, perplexity

    def evaluate_x(model, val_ids, val_text, block_size, batch_size, device):
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


    def evaluate_swa():
        swa_model.eval()
        losses = 0.0
        total_tokens = 0
        with torch.no_grad():
            for xb, yb in iter_full_split(val_ids, args.block_size, args.batch_size, device):
                logits, _ = swa_model(xb, yb)
                B, T, V = logits.size()
                loss = F.cross_entropy(logits.view(-1, V), yb.view(-1), reduction='sum')
                losses += loss.item()
                total_tokens += B * T
        swa_model.train()
        return losses / len(val_text) #total_tokens

        def evaluate_swa():
            return evaluate_x(swa_model.module, val_ids, val_text, 
                            args.block_size, args.batch_size, device)

    best_val_loss = float("inf")

    ptr = 0 # remembers where in the million token strip we are
    step = 0 # counts how many weight updates have happened total
    t0 = time.time() # used for timing
    for epoch in range(1, args.epochs + 1):
        for _ in tqdm(range(1, batches + 1), desc=f"Epoch {epoch}/{args.epochs}"):
            step += 1
            xb, yb, ptr = get_batch(train_ids, ptr, args.block_size, args.batch_size, device) # xb -> what the model sees, yb -> what the model should predict
            with torch.autocast(device_type=device, dtype=torch.bfloat16): # using bf16 for faster training on supported hardware
                _, loss = model(xb, yb)
            # _, loss = model(xb, yb) # forward pass, we get the loss directly from the model, no need to calculate it separately
            opt.zero_grad(set_to_none=True) # 
            if opt.adamw is not None:
                opt.adamw.zero_grad(set_to_none=True)
            loss.backward() # backward pass, calculates the gradients for all parameters
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip) # gradient clipping to prevent exploding gradients
            opt.step() # applies grad decent
            scheduler.step() # updates the learning rate according to the schedule
            if step >= swa_start:
                swa_model.update_parameters(model)

            elapsed = time.time() - t0
            logger.log("training_step",
                      step=step,
                      max_steps=max_steps,
                      loss=loss.item(),
                      elapsed_time=elapsed,
                      prnt=False)

            if step == 1 or step % eval_interval == 0 or step == max_steps:
                val_loss = evaluate()
                val_loss_per_token, perplexity = evaluate_with_perplexity()
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                logger.log("validation_step",
                        step=step,
                        max_steps=max_steps,
                        loss=val_loss,
                        perplexity=round(perplexity, 2),
                        elapsed_time=elapsed)
                # add this block
                if step >= swa_start:
                    swa_val_loss = evaluate_swa()
                    logger.log("swa_validation_step",
                            step=step,
                            max_steps=max_steps,
                            swa_val_loss=round(swa_val_loss, 6),
                            elapsed_time=elapsed)

    # --- SWA final evaluation ---
    swa_val_loss = evaluate_swa()
    if swa_val_loss < best_val_loss:
        best_val_loss = swa_val_loss

if __name__ == "__main__":
    try:
        main()
    finally:
        if logger and hasattr(logger, 'file_handler'):
            logger.file_handler.close()
