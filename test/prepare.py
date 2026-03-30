# prepare.py - run once
# Downloads the Hacker News dataset and trains + saves a BPE tokenizer.

from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
import pickle
import os

# ---------------------------------------------------------------------------
# Config (must match train.py)
# ---------------------------------------------------------------------------

VOCAB_SIZE   = 16_000
NUM_TITLES   = 100_000
SEED         = 1337
VAL_FRAC     = 0.10
DATA_DIR     = "./data"
TOKENIZER_PATH = "./data/tokenizer.pkl"
EOS_TOKEN    = "<eos>"
UNK_TOKEN    = "<unk>"
PAD_TOKEN    = "<pad>"

# ---------------------------------------------------------------------------
# Step 1: Download dataset
# ---------------------------------------------------------------------------

print("Downloading Hacker News dataset...")
ds = load_dataset(
    "julien040/hacker-news-posts",
    split="train",
    cache_dir=DATA_DIR
).shuffle(seed=SEED)

titles = [row["title"].strip() for row in ds.take(NUM_TITLES)]
n = int(NUM_TITLES * (1 - VAL_FRAC))
train_titles = titles[:n]
val_titles   = titles[n:]
print(f"  Train titles: {len(train_titles)}")
print(f"  Val titles:   {len(val_titles)}")

# ---------------------------------------------------------------------------
# Step 2: Train BPE tokenizer
# ---------------------------------------------------------------------------

print(f"Training BPE tokenizer (vocab_size={VOCAB_SIZE})...")

tokenizer = Tokenizer(models.BPE(unk_token=UNK_TOKEN))
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
tokenizer.decoder = decoders.ByteLevel()

trainer = trainers.BpeTrainer(
    vocab_size=VOCAB_SIZE,
    special_tokens=[PAD_TOKEN, EOS_TOKEN, UNK_TOKEN]
)

all_titles = train_titles + val_titles
tokenizer.train_from_iterator(all_titles, trainer)
print(f"  Vocab size: {tokenizer.get_vocab_size()}")

# ---------------------------------------------------------------------------
# Step 3: Save tokenizer
# ---------------------------------------------------------------------------

os.makedirs(DATA_DIR, exist_ok=True)
with open(TOKENIZER_PATH, "wb") as f:
    pickle.dump(tokenizer, f)
print(f"  Tokenizer saved to {TOKENIZER_PATH}")

print("\nDone! Ready to train.")