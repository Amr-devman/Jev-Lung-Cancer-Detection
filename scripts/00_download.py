#!/usr/bin/env python3
"""Download the dataset with kagglehub and verify its layout.

The dataset is **LungHist700** (``abdullahhasansajjad/lunghist700``): 691 JPEG
tiles of lung tissue in three class folders under
``LungHist700_combined/data/images`` — ``aca`` (280), ``nor`` (151), ``ssc``
(260) — plus a ``data.csv`` with subclass, magnification and patient metadata,
which the pipeline never reads. The pixels stay in the kagglehub cache; nothing
is copied or committed.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jev_cancer.index import class_root, download_dataset

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="use an already-downloaded copy instead of kagglehub",
    )
    args = parser.parse_args()

    root = class_root(args.source or download_dataset())
    counts: Counter = Counter()
    for folder in sorted(root.iterdir()):
        if folder.is_dir():
            counts[folder.name] = sum(
                1 for path in folder.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
            )

    print(f"[download] dataset root: {root}")
    for name, count in sorted(counts.items()):
        print(f"[download]   {name:<28} {count:>6} images")
    print(f"[download] total {sum(counts.values())} images in {len(counts)} classes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
