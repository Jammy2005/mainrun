"""
plot_training.py — visualize a mainrun training log

Usage:
    python plot_training.py                          # latest log in ./logs/
    python plot_training.py logs/run_20260327.log    # specific log file
    python plot_training.py --all                    # overlay all logs in ./logs/
"""

import json
import sys
import glob
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ── helpers ──────────────────────────────────────────────────────────────────

def parse_log(path: str) -> dict:
    """Read a JSONL log file and return dicts of steps → loss values."""
    train_steps, train_losses = [], []
    val_steps,   val_losses   = [], []
    lr_steps,    lr_values    = [], []
    meta = {}

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            event = entry.get("event", "")

            if event == "hyperparameters_configured":
                meta = entry

            elif event == "training_step":
                train_steps.append(entry["step"])
                train_losses.append(entry["loss"])

            elif event == "validation_step":
                val_steps.append(entry["step"])
                val_losses.append(entry["loss"])

            elif event == "lr_step":          # optional — log if you add it
                lr_steps.append(entry["step"])
                lr_values.append(entry["lr"])

    return {
        "path": path,
        "name": Path(path).stem,
        "meta": meta,
        "train": (train_steps, train_losses),
        "val":   (val_steps,   val_losses),
        "lr":    (lr_steps,    lr_values),
    }


def find_latest_log(log_dir: str = "./logs") -> str:
    logs = sorted(glob.glob(f"{log_dir}/*.log"))
    if not logs:
        raise FileNotFoundError(f"No .log files found in {log_dir}/")
    return logs[-1]


def epoch_lines(meta: dict):
    """Return a list of step numbers where each epoch boundary falls."""
    batches = meta.get("batches_per_epoch")
    epochs  = meta.get("epochs")
    if not batches or not epochs:
        return []
    return [batches * e for e in range(1, epochs + 1)]


# ── plotting ──────────────────────────────────────────────────────────────────

COLORS = ["#378ADD", "#E24B4A", "#1D9E75", "#BA7517", "#7F77DD", "#D4537E"]

def plot_single(data: dict, save: bool = True):
    """Full diagnostic plot for one run."""
    train_steps, train_losses = data["train"]
    val_steps,   val_losses   = data["val"]
    lr_steps,    lr_values    = data["lr"]
    meta = data["meta"]

    has_lr = bool(lr_steps)
    n_rows = 3 if has_lr else 2
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 4 * n_rows), sharex=False)
    fig.suptitle(data["name"], fontsize=13, fontweight="normal", y=1.01)

    # ── panel 1: training loss ───────────────────────────────────────────────
    ax = axes[0]
    ax.plot(train_steps, train_losses, color="#E24B4A", linewidth=0.8,
            alpha=0.6, label="training loss (per token)")
    # smoothed version
    if len(train_losses) > 20:
        import numpy as np
        w = max(1, len(train_losses) // 50)
        kernel = np.ones(w) / w
        smoothed = np.convolve(train_losses, kernel, mode="valid")
        offset = len(train_losses) - len(smoothed)
        ax.plot(train_steps[offset:], smoothed, color="#E24B4A",
                linewidth=1.8, label="smoothed")

    for x in epoch_lines(meta):
        ax.axvline(x, color="gray", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.set_ylabel("loss")
    ax.set_title("training loss", fontsize=11, fontweight="normal")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    # ── panel 2: validation loss ─────────────────────────────────────────────
    ax = axes[1]
    ax.plot(val_steps, val_losses, color="#378ADD", linewidth=2,
            marker="o", markersize=4, label="validation loss")

    # annotate first and last
    if val_losses:
        ax.annotate(f"{val_losses[0]:.4f}", (val_steps[0], val_losses[0]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8,
                    color="#378ADD")
        ax.annotate(f"{val_losses[-1]:.4f}", (val_steps[-1], val_losses[-1]),
                    textcoords="offset points", xytext=(-40, -14), fontsize=8,
                    color="#378ADD")

        # shade improvement area
        ax.fill_between(val_steps, val_losses[0], val_losses,
                        alpha=0.08, color="#378ADD")

    for x in epoch_lines(meta):
        ax.axvline(x, color="gray", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.set_ylabel("loss")
    ax.set_title("validation loss", fontsize=11, fontweight="normal")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    # ── panel 3: learning rate (optional) ────────────────────────────────────
    if has_lr:
        ax = axes[2]
        ax.plot(lr_steps, lr_values, color="#1D9E75", linewidth=1.5)
        ax.set_ylabel("lr")
        ax.set_title("learning rate schedule", fontsize=11, fontweight="normal")
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel("step")
    plt.tight_layout()

    if save:
        out = Path(data["path"]).with_suffix(".png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved → {out}")
    else:
        plt.show()

    return fig


def plot_comparison(runs: list[dict], save: bool = True):
    """Overlay validation loss curves from multiple runs."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for i, data in enumerate(runs):
        col = COLORS[i % len(COLORS)]
        label = data["name"]

        train_steps, train_losses = data["train"]
        val_steps,   val_losses   = data["val"]

        # left: val loss
        axes[0].plot(val_steps, val_losses, color=col, linewidth=1.8,
                     marker="o", markersize=3, label=label)

        # right: training loss (smoothed)
        if train_losses:
            try:
                import numpy as np
                w = max(1, len(train_losses) // 60)
                kernel = np.ones(w) / w
                smoothed = np.convolve(train_losses, kernel, mode="valid")
                offset = len(train_losses) - len(smoothed)
                axes[1].plot(train_steps[offset:], smoothed, color=col,
                             linewidth=1.5, alpha=0.8, label=label)
            except ImportError:
                axes[1].plot(train_steps, train_losses, color=col,
                             linewidth=1.0, alpha=0.6, label=label)

    for ax, title in zip(axes, ["validation loss", "training loss (smoothed)"]):
        ax.set_title(title, fontsize=11, fontweight="normal")
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    plt.suptitle("run comparison", fontsize=13, fontweight="normal")
    plt.tight_layout()

    if save:
        out = Path("./logs/comparison.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved → {out}")
    else:
        plt.show()

    return fig


def print_summary(data: dict):
    """Print a quick text summary to the terminal."""
    val_losses = data["val"][1]
    train_losses = data["train"][1]
    meta = data["meta"]

    print(f"\n{'─'*50}")
    print(f"  run      : {data['name']}")
    print(f"  optimizer: {meta.get('optimizer', 'unknown')}")
    print(f"  lr       : {meta.get('lr', '?')}")
    print(f"  epochs   : {meta.get('epochs', '?')}")
    if val_losses:
        print(f"  val loss : {val_losses[0]:.4f} → {val_losses[-1]:.4f}  "
              f"(Δ {val_losses[0]-val_losses[-1]:+.4f})")
        # detect plateau: last 25% of val evals
        quarter = max(1, len(val_losses) // 4)
        tail_drop = val_losses[-quarter] - val_losses[-1]
        print(f"  tail drop: {tail_drop:.4f}  "
              f"({'plateau ⚠' if tail_drop < 0.005 else 'still improving ✓'})")
    if train_losses:
        print(f"  final train loss: {train_losses[-1]:.4f}")
    print(f"{'─'*50}\n")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot mainrun training logs.")
    parser.add_argument("log", nargs="?", help="Path to a specific .log file")
    parser.add_argument("--all", action="store_true",
                        help="Overlay all logs in ./logs/ for comparison")
    parser.add_argument("--show", action="store_true",
                        help="Show plot interactively instead of saving to file")
    args = parser.parse_args()

    save = not args.show

    if args.all:
        logs = sorted(glob.glob("./logs/*.log"))
        if not logs:
            print("No logs found in ./logs/")
            sys.exit(1)
        runs = [parse_log(p) for p in logs]
        for r in runs:
            print_summary(r)
        plot_comparison(runs, save=save)

    else:
        log_path = args.log or find_latest_log()
        print(f"Reading: {log_path}")
        data = parse_log(log_path)
        print_summary(data)
        plot_single(data, save=save)


if __name__ == "__main__":
    main()