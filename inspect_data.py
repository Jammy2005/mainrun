from datasets import load_dataset
import random

def load_data(num_titles=100_000, seed=1337, val_frac=0.1, cache_dir="./data"):
    print("Loading dataset...")
    ds = load_dataset("julien040/hacker-news-posts", split="train", cache_dir=cache_dir).shuffle(seed=seed)
    titles = [row["title"].strip() for row in ds.take(num_titles)]
    n = int(num_titles * (1 - val_frac))
    train_titles, val_titles = titles[:n], titles[n:]
    print(f"Loaded {len(train_titles)} train titles, {len(val_titles)} val titles\n")
    return train_titles, val_titles


def show_samples(titles, n=10, label="samples"):
    print(f"--- {label} (random {n}) ---")
    for t in random.sample(titles, n):
        print(f"  {t}")
    print()


def show_stats(titles, label="titles"):
    lengths = [len(t) for t in titles]
    word_counts = [len(t.split()) for t in titles]
    print(f"--- {label} stats ---")
    print(f"  count:          {len(titles):,}")
    print(f"  avg chars:      {sum(lengths)/len(lengths):.1f}")
    print(f"  min/max chars:  {min(lengths)} / {max(lengths)}")
    print(f"  avg words:      {sum(word_counts)/len(word_counts):.1f}")
    print(f"  min/max words:  {min(word_counts)} / {max(word_counts)}")
    print()


def search(titles, query, max_results=20):
    query_lower = query.lower()
    results = [t for t in titles if query_lower in t.lower()]
    print(f"--- search: '{query}' — {len(results)} matches (showing up to {max_results}) ---")
    for t in results[:max_results]:
        print(f"  {t}")
    print()


def show_by_length(titles, shortest=True, n=10):
    label = "shortest" if shortest else "longest"
    sorted_titles = sorted(titles, key=len, reverse=not shortest)
    print(f"--- {n} {label} titles ---")
    for t in sorted_titles[:n]:
        print(f"  [{len(t):3d} chars] {t}")
    print()


def interactive_menu(train_titles, val_titles):
    all_titles = train_titles + val_titles
    while True:
        print("=" * 50)
        print("  1. random samples (train)")
        print("  2. random samples (val)")
        print("  3. stats (train vs val)")
        print("  4. search titles")
        print("  5. shortest titles")
        print("  6. longest titles")
        print("  7. inspect a specific index")
        print("  q. quit")
        print("=" * 50)
        choice = input("choice: ").strip().lower()

        if choice == "1":
            n = input("  how many? [10]: ").strip()
            show_samples(train_titles, int(n) if n else 10, "train")

        elif choice == "2":
            n = input("  how many? [10]: ").strip()
            show_samples(val_titles, int(n) if n else 10, "val")

        elif choice == "3":
            show_stats(train_titles, "train")
            show_stats(val_titles, "val")

        elif choice == "4":
            query = input("  search query: ").strip()
            if query:
                search(all_titles, query)

        elif choice == "5":
            n = input("  how many? [10]: ").strip()
            show_by_length(all_titles, shortest=True, n=int(n) if n else 10)

        elif choice == "6":
            n = input("  how many? [10]: ").strip()
            show_by_length(all_titles, shortest=False, n=int(n) if n else 10)

        elif choice == "7":
            idx = input(f"  index (0-{len(all_titles)-1}): ").strip()
            if idx.isdigit() and int(idx) < len(all_titles):
                print(f"  [{int(idx)}] {all_titles[int(idx)]}\n")
            else:
                print("  invalid index\n")

        elif choice == "q":
            print("bye")
            break
        else:
            print("  unrecognised choice\n")


if __name__ == "__main__":
    train_titles, val_titles = load_data()
    show_samples(train_titles, n=5, label="quick preview")
    interactive_menu(train_titles, val_titles)