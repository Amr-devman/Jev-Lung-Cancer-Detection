#!/usr/bin/env python3
"""Generate the README figures from a graded run.

Writes to docs/:
  figures/pr_curves.png    one-vs-rest precision-recall curves per class
  figures/confusion.png    confusion matrix with counts
  figures/views.png        the six views of one tissue field
  samples/{aca,nor,ssc}.jpg   one example image per class

Everything is derived from the run's predictions.json, the index and the
dataset, so the figures reproduce with the rest of the pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jev_cancer import config
from jev_cancer.index import class_root, load_detail, load_rgb, load_tiles

DOCS = config.ROOT / "docs"
CLASS_NAMES = sorted(config.SOURCE_CLASSES)
PROFILE_ORDER = [f"p{i}" for i in range(3)]  # p0..p2 map to sorted classes c0..c2


def load_run(run_dir: Path) -> tuple[dict, dict]:
    payload = json.loads((run_dir / "predictions.json").read_text())
    index = json.loads((config.RUNS / "index.json").read_text())
    return payload, index


def find_latest_run() -> Path:
    candidates = sorted(
        config.RUNS.glob("*/*/*/predictions.json"),
        key=lambda path: path.stat().st_mtime,
    )
    return candidates[-1].parent


def pr_curves(payload: dict, index: dict) -> None:
    class_of = {item["id"]: item["class"] for item in index["images"]}
    y_true = np.array([CLASS_NAMES.index(class_of[p["image_id"]]) for p in payload["predictions"]])
    scores = np.array(
        [[p["probabilities"][k] for k in PROFILE_ORDER] for p in payload["predictions"]]
    )

    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    for k, name in enumerate(CLASS_NAMES):
        s = scores[:, k]
        thresholds = np.unique(s)[::-1]
        precision, recall = [], []
        for t in thresholds:
            pred = s >= t
            tp = int(((pred) & (y_true == k)).sum())
            precision.append(tp / max(1, int(pred.sum())))
            recall.append(tp / int((y_true == k).sum()))
        ap = float(np.trapezoid(precision, recall))
        ax.plot(recall, precision, linewidth=2, label=f"{name} (AP {ap:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(DOCS / "figures" / "pr_curves.png", dpi=150)
    plt.close(fig)


def confusion(payload: dict, index: dict) -> None:
    class_of = {item["id"]: item["class"] for item in index["images"]}
    name_of = {k: n for k, n in zip(PROFILE_ORDER, CLASS_NAMES, strict=True)}
    matrix = np.zeros((3, 3), dtype=int)
    for p in payload["predictions"]:
        truth = CLASS_NAMES.index(class_of[p["image_id"]])
        guess = CLASS_NAMES.index(name_of[p["profile"]])
        matrix[truth, guess] += 1

    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(3), CLASS_NAMES, fontsize=9)
    ax.set_yticks(range(3), CLASS_NAMES, fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(3):
        for j in range(3):
            colour = "white" if matrix[i, j] > matrix.max() / 2 else "black"
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color=colour, fontsize=12)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(DOCS / "figures" / "confusion.png", dpi=150)
    plt.close(fig)


def sample_images(index: dict) -> dict[str, Path]:
    root = class_root(Path(index["source_root"]))
    picked = {}
    for name in CLASS_NAMES:
        files = sorted((root / name).glob("*.jpg"))
        path = files[len(files) // 2]
        picked[name] = path
        image = load_rgb(path, max_side=600)
        image.save(DOCS / "samples" / f"{name}.jpg", quality=82)
    return picked


def views_figure(path: Path) -> None:
    views = {"sample": load_rgb(path, max_side=400), "high_power": load_detail(path, max_side=400)}
    for name, tile in load_tiles(path).items():
        views[name] = tile.resize((400, int(400 * tile.height / tile.width)))
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 6.0))
    for ax, (name, image) in zip(axes.flat, views.items(), strict=False):
        ax.imshow(image)
        ax.set_title(name, fontsize=11)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(DOCS / "figures" / "views.jpg", dpi=100, pil_kwargs={"quality": 85})
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=None)
    args = parser.parse_args()

    for folder in (DOCS / "figures", DOCS / "samples"):
        folder.mkdir(parents=True, exist_ok=True)

    run_dir = args.run or find_latest_run()
    payload, index = load_run(run_dir)
    pr_curves(payload, index)
    confusion(payload, index)
    picked = sample_images(index)
    views_figure(picked[CLASS_NAMES[0]])
    print(f"[report] figures in {DOCS / 'figures'}")
    print(f"[report] samples in {DOCS / 'samples'}: " + ", ".join(p.name for p in picked.values()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
