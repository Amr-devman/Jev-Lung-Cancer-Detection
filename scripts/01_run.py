#!/usr/bin/env python3
"""Classify a stratified subset of the dataset with one Jev request per image.

The subset is drawn at random, stratified so every class contributes the same
number of images. ``--fraction`` is the share of the whole dataset to evaluate
(default 0.2, which nets 3,000 of the 15,000 images: 1,000 per class) and
``--seed`` makes the draw reproducible across machines.

The model sees only pixels and the clinical question bank. Class folders are
used to build the subset and, in ``02_grade.py``, to grade it. Grading is a
separate script so no label can reach the model.

Every raw answer is cached under ``runs/<protocol>/raw``, so reruns and
different subsets only pay for images that have not been seen yet.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jev_cancer import config
from jev_cancer.fusion import predict
from jev_cancer.index import build_index, select_subset
from jev_cancer.pipeline import load_records, protocol_dir, run_all, run_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fraction",
        type=float,
        default=None,
        help="fraction of the dataset to evaluate (default 0.2 = 3,000 images)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.DEFAULT_SEED,
        help=f"seed for the stratified draw (default {config.DEFAULT_SEED})",
    )
    parser.add_argument("--workers", type=int, default=config.DEFAULT_WORKERS)
    parser.add_argument("--source", type=Path, default=None, help="local dataset copy")
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="reuse the cached answers and only rebuild predictions.json",
    )
    args = parser.parse_args()

    index = build_index(args.source, force=args.rebuild_index)
    subset = select_subset(index, fraction=args.fraction, seed=args.seed)

    if args.export_only:
        directory = run_dir(subset)
        if not (protocol_dir() / "raw").exists():
            raise SystemExit("no cached answers to export; run without --export-only first")
    else:
        directory, _ = run_all(subset, workers=args.workers)

    records = load_records([item["id"] for item in subset["images"]])
    predictions = []
    for record in records:
        result = predict(record["response"])
        result["image_id"] = record["image_id"]
        predictions.append(result)

    payload = {
        "model": config.MODEL,
        "protocol": directory.parent.name,
        "subset": {key: value for key, value in subset.items() if key != "images"},
        "total": len(predictions),
        "predictions": predictions,
    }
    (directory / "predictions.json").write_text(json.dumps(payload))

    summary_path = directory / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    summary["predicted"] = len(predictions)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[run] {len(predictions)} predictions -> {directory / 'predictions.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
