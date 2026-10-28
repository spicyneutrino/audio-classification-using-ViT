from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

root = Path(__file__).resolve().parents[1]
source = root / "train_44101.out"
pattern = (
    r"Epoch: (\d+) \| train_loss: [\d.]+ \| train_acc: [\d.]+"
    r" \| val_loss: ([\d.]+) \| val_acc: ([\d.]+)"
)
records = [
    (int(epoch), float(loss), float(accuracy))
    for epoch, loss, accuracy in re.findall(pattern, source.read_text())
]
if not records:
    raise ValueError(f"No epoch metrics found in {source}")

epochs, losses, accuracies = zip(*records)
best_epoch, best_loss, best_accuracy = min(records, key=lambda row: row[1])
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans", "sans-serif"],
    "font.size": 11,
    "svg.fonttype": "none",
    "svg.hashsalt": "urbansound8k-44101",
    "axes.spines.top": False,
    "axes.spines.right": False,
})
fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True, layout="constrained")
fig.suptitle("Job 44101 · Validation on fold 9", fontsize=16, fontweight="bold")
axes[0].plot(epochs, losses, color="#2563a6", linewidth=2)
axes[0].set_ylabel("Validation loss")
axes[1].plot(epochs, accuracies, color="#147d64", linewidth=2)
axes[1].set_ylabel("Validation accuracy")
axes[1].yaxis.set_major_formatter(PercentFormatter(1))
axes[1].set_xlabel("Epoch")
for ax, value in zip(axes, (best_loss, best_accuracy)):
    ax.axvline(best_epoch, color="#73501c", linestyle="--", linewidth=1.4)
    ax.scatter([best_epoch], [value], color="#73501c", zorder=3)
    ax.grid(axis="y", color="#dde3ea")
    ax.set_xlim(1, epochs[-1])
axes[0].annotate(
    f"Selected checkpoint\nEpoch {best_epoch} · Loss {best_loss:.4f}",
    xy=(best_epoch, best_loss),
    xytext=(0.58, 0.72),
    textcoords="axes fraction",
    arrowprops={"arrowstyle": "->", "color": "#73501c"},
    color="#513916",
)
fig.supxlabel(
    "Source: train_44101.out · Checkpoint selected by lowest validation loss",
    fontsize=10,
    color="#4b5563",
)
destination = root / "docs" / "assets" / "training-44101.svg"
destination.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(destination, metadata={"Date": None}, facecolor="white")
plt.close(fig)
print(f"Wrote {destination.relative_to(root)} from {len(records)} epoch records.")
