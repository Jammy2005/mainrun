import json, sys, os, glob, argparse, math
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
except ImportError:
    print("run: pip install matplotlib")
    sys.exit(1)

def parse_log(filepath):
    train_steps, train_losses = [], []
    val_steps, val_losses, val_perplexities = [], [], []
    hyperparams = {}
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try: entry = json.loads(line)
            except: continue
            event = entry.get("event", "")
            if event == "hyperparameters_configured":
                hyperparams = entry
            elif event == "training_step":
                train_steps.append(entry["step"])
                train_losses.append(entry["loss"])
            elif event == "validation_step":
                val_steps.append(entry["step"])
                val_losses.append(entry["loss"])
                val_perplexities.append(entry.get("perplexity"))
    return dict(filepath=filepath, name=Path(filepath).stem,
                hyperparams=hyperparams, train_steps=train_steps,
                train_losses=train_losses, val_steps=val_steps,
                val_losses=val_losses, val_perplexities=val_perplexities)

def smooth(values, window=30):
    out = []
    for i in range(len(values)):
        s = max(0, i - window + 1)
        out.append(sum(values[s:i+1]) / (i - s + 1))
    return out

def make_label(run):
    h = run["hyperparams"]
    parts = []
    if "block_size" in h:  parts.append(f"blk={h['block_size']}")
    if "n_layer"    in h:  parts.append(f"L={h['n_layer']}")
    if "dropout"    in h:  parts.append(f"drop={h['dropout']}")
    if "vocab_size" in h:  parts.append(f"vocab={h['vocab_size']//1000}k")
    return run["name"] + (f" ({', '.join(parts)})" if parts else "")

def plot(runs, output_path=None):
    has_ppl = any(any(p for p in r["val_perplexities"] if p) for r in runs)
    n = 3 if has_ppl else 2
    fig, axes = plt.subplots(1, n, figsize=(6*n, 5))
    fig.patch.set_facecolor('#0f0f0f')
    for ax in axes:
        ax.set_facecolor('#1a1a1a')
        ax.tick_params(colors='#aaaaaa')
        ax.xaxis.label.set_color('#aaaaaa')
        ax.yaxis.label.set_color('#aaaaaa')
        ax.title.set_color('#dddddd')
        for sp in ax.spines.values(): sp.set_edgecolor('#333333')
        ax.grid(True, color='#2a2a2a', linewidth=0.5)

    colors = ['#1D9E75','#7F77DD','#D85A30','#FAC775','#85B7EB','#F09595']

    for i, run in enumerate(runs):
        c = colors[i % len(colors)]
        label = make_label(run)
        final = run["val_losses"][-1] if run["val_losses"] else None

        # train loss
        if run["train_steps"]:
            sm = smooth(run["train_losses"])
            axes[0].plot(run["train_steps"], run["train_losses"], color=c, alpha=0.15, linewidth=0.5)
            axes[0].plot(run["train_steps"], sm, color=c, linewidth=1.5, label=label)
        axes[0].set_title("train loss (smoothed)")
        axes[0].set_xlabel("step")
        axes[0].set_ylabel("loss")
        axes[0].legend(fontsize=7, facecolor='#111111', labelcolor='#cccccc', edgecolor='#333333')

        # val loss
        if run["val_steps"]:
            fl = f"  final={final:.4f}" if final else ""
            axes[1].plot(run["val_steps"], run["val_losses"], color=c, linewidth=2,
                         marker='o', markersize=3, label=label+fl)
        axes[1].axhline(y=1.754, color='#666666', linewidth=1, linestyle='--', alpha=0.7)
        axes[1].text(2, 1.76, 'baseline 1.754', color='#666666', fontsize=7)
        axes[1].set_title("validation loss")
        axes[1].set_xlabel("step")
        axes[1].set_ylabel("loss")
        axes[1].legend(fontsize=7, facecolor='#111111', labelcolor='#cccccc', edgecolor='#333333')

        # perplexity
        if has_ppl:
            ps = [(s,p) for s,p in zip(run["val_steps"], run["val_perplexities"]) if p]
            if ps:
                xs, ys = zip(*ps)
                axes[2].plot(xs, ys, color=c, linewidth=2, marker='o', markersize=3,
                             label=f"{label}  final={ys[-1]:.0f}")
            axes[2].set_title("validation perplexity")
            axes[2].set_xlabel("step")
            axes[2].set_ylabel("perplexity")
            axes[2].set_yscale('log')
            axes[2].yaxis.set_major_formatter(ticker.ScalarFormatter())
            axes[2].legend(fontsize=7, facecolor='#111111', labelcolor='#cccccc', edgecolor='#333333')

    plt.tight_layout(pad=2.0)
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"saved to {output_path}")
    else:
        plt.show()

def print_summary(runs):
    print("\n" + "="*65)
    print(f"{'run':<32} {'steps':>6} {'final val':>10} {'ppl':>8}")
    print("="*65)
    for r in runs:
        steps = r["val_steps"][-1] if r["val_steps"] else "?"
        vl    = r["val_losses"][-1] if r["val_losses"] else None
        ppls  = [p for p in r["val_perplexities"] if p]
        ppl   = ppls[-1] if ppls else None
        beat  = "  ✓ beats baseline" if vl and vl < 1.754 else ""
        print(f"{r['name'][:31]:<32} {steps:>6} {vl:>10.6f} {str(round(ppl)) if ppl else 'n/a':>8}{beat}")
    print("="*65 + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="*")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    filepaths = args.logs or sorted(glob.glob("./logs/*.log"))
    if not filepaths:
        print("no log files found in ./logs/")
        sys.exit(1)
    runs = [parse_log(f) for f in filepaths if os.path.exists(f)]
    runs = [r for r in runs if r["train_steps"] or r["val_steps"]]
    if not runs:
        print("no valid log data found")
        sys.exit(1)
    print_summary(runs)
    plot(runs, args.out)

if __name__ == "__main__":
    main()
