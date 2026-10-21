"""Download the dataset and build one neutral index over it.

The dataset is LungHist700 (``abdullahhasansajjad/lunghist700`` on Kaggle):
691 JPEG tiles in three class folders under ``LungHist700_combined/data/images``
(``aca``, ``nor``, ``ssc``), plus a ``data.csv`` with subclass, magnification
and patient ids. Only the class folders are read — the CSV metadata is never
used and never sent anywhere.

There is no prepared copy of the pixels. Every image is fed to the model under a
neutral id like ``img_000427`` with no label. The class folder is kept in the
index only so the stratified subset can be drawn and, after predictions are
frozen, so the grader can reconstruct the truth.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from PIL import Image

from . import config

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")

INDEX_JSON = config.RUNS / "index.json"

DATASET_SLUG = "abdullahhasansajjad/lunghist700"


def download_dataset() -> Path:
    """Download the Kaggle dataset via kagglehub and return its root folder."""
    import kagglehub

    return Path(kagglehub.dataset_download(DATASET_SLUG))


def class_root(source: Path) -> Path:
    """Find the directory that holds the class folders.

    Kagglehub layouts vary by version, so search a few levels down for a
    directory containing every class folder rather than hard-coding a path.
    """
    candidates = [source, *sorted(p for p in source.rglob("*") if p.is_dir())]
    for candidate in candidates:
        if all((candidate / name).is_dir() for name in config.SOURCE_CLASSES):
            return candidate
    raise SystemExit(f"could not find {config.SOURCE_CLASSES} under {source}")


def build_index(source: Path | None = None, force: bool = False) -> dict:
    if INDEX_JSON.exists() and not force:
        return json.loads(INDEX_JSON.read_text())

    root = class_root(source or download_dataset())
    images = []
    for class_name in sorted(config.SOURCE_CLASSES):
        for path in sorted((root / class_name).iterdir()):
            if path.suffix.lower() in IMAGE_SUFFIXES:
                images.append({"id": "", "source": str(path), "class": class_name})

    # The neutral id is assigned after a seeded shuffle so the ids carry no
    # information about the class ordering.
    random.Random(config.DEFAULT_SEED).shuffle(images)
    for index, item in enumerate(images):
        item["id"] = f"img_{index:06d}"

    payload = {"total": len(images), "source_root": str(root), "images": images}
    config.RUNS.mkdir(parents=True, exist_ok=True)
    INDEX_JSON.write_text(json.dumps(payload, indent=2))
    print(f"[index] {len(images)} images from {root}")
    return payload


def load_index() -> dict:
    return json.loads(INDEX_JSON.read_text())


def select_subset(
    index: dict,
    fraction: float | None = None,
    seed: int = config.DEFAULT_SEED,
) -> dict:
    """Draw a random subset stratified by class.

    Every class contributes the same number of images, so the subset is class
    balanced. ``fraction`` is the share of the whole dataset to evaluate; left
    as ``None`` it uses the default (the whole dataset), and ``1.0`` evaluates
    every image regardless of class balance.
    """
    fraction = config.DEFAULT_FRACTION if fraction is None else fraction
    if not 0 < fraction <= 1:
        raise SystemExit(f"fraction must be in (0, 1], got {fraction}")

    by_class: dict[str, list[dict]] = {}
    for item in index["images"]:
        by_class.setdefault(item["class"], []).append(item)
    for items in by_class.values():
        items.sort(key=lambda item: item["id"])

    total = len(index["images"])
    if fraction == 1.0:
        picked = sorted(index["images"], key=lambda item: item["id"])
        counts = {name: len(items) for name, items in sorted(by_class.items())}
        print(f"[subset] all {len(picked)} images, seed ignored for full-dataset runs")
        return {
            "fraction": 1.0,
            "seed": seed,
            "per_class": None,
            "class_counts": counts,
            "total": len(picked),
            "images": picked,
        }

    per_class = min(
        round(fraction * total / len(by_class)),
        min(len(items) for items in by_class.values()),
    )
    if per_class < 1:
        raise SystemExit(f"fraction {fraction} is too small to cover {len(by_class)} classes")

    rng = random.Random(seed)
    picked = []
    for class_name in sorted(by_class):
        picked.extend(rng.sample(by_class[class_name], per_class))
    picked.sort(key=lambda item: item["id"])

    counts = {
        class_name: sum(1 for item in picked if item["class"] == class_name)
        for class_name in sorted(by_class)
    }
    print(f"[subset] {len(picked)} images, {per_class} per class, fraction={fraction}, seed={seed}")
    return {
        "fraction": fraction,
        "seed": seed,
        "per_class": per_class,
        "class_counts": counts,
        "total": len(picked),
        "images": picked,
    }


def subset_key(subset: dict) -> str:
    if isinstance(subset["seed"], str):
        return subset["seed"]
    return f"f{subset['fraction']:g}_s{subset['seed']}"


def load_rgb(path: str | Path, max_side: int = config.MAX_SIDE) -> Image.Image:
    with Image.open(path) as source:
        image = source.convert("RGB")
    if max(image.size) > max_side:
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return image
