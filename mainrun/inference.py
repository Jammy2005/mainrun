import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from tokenizers import Tokenizer
from dataclasses import dataclass
from pathlib import Path
import argparse


# ── model definition (must match training code exactly) ──────────────────────

@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int
    n_layer: int
    n_head: int
    d_model: int
    dropout: float

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.head_dim = cfg.d_model // cfg.n_head
        self.n_head   = cfg.n_head
        self.qkv      = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj     = nn.Linear(cfg.d_model, cfg.d_model)
        self.attn_drop  = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)
        self.register_buffer("tril", torch.tril(torch.ones(cfg.block_size, cfg.block_size)))

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim).transpose(1, 3)
        q, k, v = qkv[..., 0, :, :], qkv[..., 1, :, :], qkv[..., 2, :, :]
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))

class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )
    def forward(self, x): return self.net(x)

class Block(nn.Module):
    def __init__(self, cfg):
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
    def __init__(self, cfg):
        super().__init__()
        self.cfg       = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb   = nn.Parameter(torch.zeros(1, cfg.block_size, cfg.d_model))
        self.drop      = nn.Dropout(cfg.dropout)
        self.blocks    = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f      = nn.LayerNorm(cfg.d_model)
        self.head      = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)
        self.head.weight = self.token_emb.weight

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        tok    = self.token_emb(idx)
        pos    = self.pos_emb[:, :T, :]
        x      = self.drop(tok + pos)
        for block in self.blocks:
            x = block(x)
        x      = self.ln_f(x)
        logits = self.head(x)
        loss   = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


# ── tokenizer wrapper ────────────────────────────────────────────────────────

class BPETokenizer:
    def __init__(self, tokenizer: Tokenizer):
        self.tk = tokenizer

    def encode(self, s: str):
        return self.tk.encode(s).ids

    def decode(self, ids):
        return self.tk.decode(ids, skip_special_tokens=True)

    @property
    def vocab_size(self):
        return self.tk.get_vocab_size()


# ── generation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new_tokens: int = 50,
             temperature: float = 1.0, top_k: int = None, device: str = "cpu"):
    """
    Generate text from a prompt.

    temperature: float
        controls randomness. 1.0 = normal, <1.0 = more focused, >1.0 = more random
    top_k: int or None
        if set, only sample from the top k most likely tokens at each step
    """
    model.eval()
    eos_id = tokenizer.tk.token_to_id("<eos>")

    # encode the prompt
    input_ids = tokenizer.encode(prompt)
    if not input_ids:
        # if prompt is empty, start with eos so the model knows to begin a title
        input_ids = [eos_id]

    idx = torch.tensor([input_ids], dtype=torch.long, device=device)

    generated = []
    for _ in range(max_new_tokens):
        # crop context to block_size if too long
        idx_cond = idx[:, -model.cfg.block_size:]

        # forward pass — get logits for the last token position only
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]          # shape [1, vocab_size]

        # apply temperature — lower = sharper distribution
        logits = logits / temperature

        # optionally apply top-k filtering
        if top_k is not None:
            top_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < top_values[:, [-1]]] = float('-inf')

        # convert to probabilities and sample
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # shape [1, 1]

        # stop if we hit <eos>
        if next_token.item() == eos_id:
            break

        generated.append(next_token.item())
        idx = torch.cat([idx, next_token], dim=1)

    return tokenizer.decode(generated)


# ── load checkpoint ──────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: str):
    print(f"loading checkpoint from {checkpoint_path} ...")
    torch.serialization.add_safe_globals([GPTConfig])
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    cfg       = ckpt['cfg']
    tokenizer = BPETokenizer(ckpt['tokenizer'])

    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    params = sum(p.numel() for p in model.parameters())
    print(f"loaded model — {params:,} parameters")
    print(f"vocab size: {cfg.vocab_size} | block size: {cfg.block_size}")
    return model, tokenizer


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate HN titles from trained GPT")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/model_final.pt",
                        help="path to .pt checkpoint file")
    parser.add_argument("--prompt",     type=str, default="",
                        help="prompt to start generation from (leave empty to generate freely)")
    parser.add_argument("--n",          type=int, default=10,
                        help="number of titles to generate")
    parser.add_argument("--max_tokens", type=int, default=50,
                        help="max new tokens per title")
    parser.add_argument("--temperature",type=float, default=0.8,
                        help="sampling temperature (lower = more focused)")
    parser.add_argument("--top_k",      type=int, default=40,
                        help="top-k sampling (set 0 to disable)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using device: {device}")

    if not Path(args.checkpoint).exists():
        print(f"checkpoint not found at {args.checkpoint}")
        print("make sure you have trained the model and saved the weights first")
        return

    model, tokenizer = load_model(args.checkpoint, device)

    top_k = args.top_k if args.top_k > 0 else None
    prompt = args.prompt if args.prompt else "<eos>"

    print(f"\ngenerating {args.n} titles")
    print(f"prompt: '{args.prompt or '(none)'}' | temperature: {args.temperature} | top_k: {top_k}")
    print("-" * 50)

    for i in range(args.n):
        title = generate(
            model, tokenizer,
            prompt      = prompt,
            max_new_tokens = args.max_tokens,
            temperature = args.temperature,
            top_k       = top_k,
            device      = device,
        )
        print(f"{i+1:>3}. {title}")


if __name__ == "__main__":
    main()