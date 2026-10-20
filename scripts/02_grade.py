#!/usr/bin/env python3
"""Grade the frozen predictions against the dataset class folders, once.

This is the only script that maps neutral profiles to class names. It runs after
``01_run.py`` has written predictions and never feeds anything back into the
model. By default it grades the most recent run; ``--run`` points at a specific
``runs/<protocol>/<fraction>_<seed>`` directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jev_cancer import config
from jev_cancer.index import INDEX_JSON
from jev_cancer.metrics import bootstrap_accuracy, summarize


def find_latest_run() -> Path:
    candidates = sorted(config.RUNS.glob("*/*/predictions.json"))
    if not candidates:
        raise SystemExit("no runs found; run scripts/01_run.py first")
    return candidates[-1].parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=None, help="run directory to grade")
    parser.add_argument("--predictions", type=Path, default=None)
    args = parser.parse_args()

    directory = args.run or find_latest_run()
    predictions_path = args.predictions or (directory / "predictions.json")
    payload = json.loads(predictions_path.read_text())
    predictions = payload["predictions"]

    index = json.loads(INDEX_JSON.read_text())
    class_of = {item["id"]: item["class"] for item in index["images"]}
    # Sorted folder names -> c0, c1, c2, matching PROFILE_TO_CLASS.
    class_names = {f"c{i}": name for i, name in enumerate(sorted(config.SOURCE_CLASSES))}
    name_to_class = {name: class_id for class_id, name in class_names.items()}

    pred = [config.PROFILE_TO_CLASS[item["profile"]] for item in predictions]
    truth = [name_to_class[class_of[item["image_id"]]] for item in predictions]

    summary = summarize(pred, truth, labels=config.CLASS_IDS)
    summary["class_names"] = class_names
    summary["confusion_named"] = {
        class_names[config.CLASS_IDS[i]]: {
            class_names[config.CLASS_IDS[j]]: summary["matrix"][i][j]
            for j in range(len(config.CLASS_IDS))
        }
        for i in range(len(config.CLASS_IDS))
    }
    summary["per_class_named"] = {
        class_names[label]: values for label, values in summary["per_class"].items()
    }
    summary["accuracy_ci95_bootstrap"] = bootstrap_accuracy(pred, truth)

    benign_class = next(class_id for class_id, name in class_names.items() if name == "benign")
    binary_pred = ["benign" if label == benign_class else "carcinoma" for label in pred]
    binary_true = ["benign" if label == benign_class else "carcinoma" for label in truth]
    summary["binary_benign_vs_carcinoma"] = {
        "accuracy": float(np.mean([p == t for p, t in zip(binary_pred, binary_true, strict=True)])),
        "total": len(pred),
    }

    confidences = np.array([item["confidence"] for item in predictions])
    correct = np.array([p == t for p, t in zip(pred, truth, strict=True)])
    summary["confidence_curve"] = [
        {
            "threshold": threshold,
            "coverage": float((confidences >= threshold).mean()),
            "accuracy": float(correct[confidences >= threshold].mean()),
            "count": int((confidences >= threshold).sum()),
        }
        for threshold in (0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
        if (confidences >= threshold).any()
    ]
    truth_array = np.array(truth)

    def class_accuracy(class_id: str) -> float:
        mask = truth_array == class_id
        return float(correct[mask].mean()) if mask.any() else 0.0

    summary["per_class_accuracy"] = {
        class_names[class_id]: {
            "n": int((truth_array == class_id).sum()),
            "accuracy": class_accuracy(class_id),
        }
        for class_id in config.CLASS_IDS
    }
    summary["subset"] = payload.get("subset", {})
    summary_path = directory / "summary.json"
    if summary_path.exists():
        summary["run"] = json.loads(summary_path.read_text())
    summary["predictions_path"] = str(predictions_path)

    config.REPORTS.mkdir(parents=True, exist_ok=True)
    (config.REPORTS / "run.json").write_text(json.dumps(summary, indent=2))

    print("\n=== graded run ===")
    print(f"  run         {directory}")
    print(f"  images      {summary['total']}")
    print(
        f"  accuracy    {summary['accuracy']:.4f}  "
        f"(95% CI {summary['accuracy_ci95'][0]:.4f}-{summary['accuracy_ci95'][1]:.4f})"
    )
    print(f"  balanced    {summary['balanced_accuracy']:.4f}")
    print(f"  benign/ca   {summary['binary_benign_vs_carcinoma']['accuracy']:.4f}")
    print("  per class:")
    for name, values in summary["per_class_named"].items():
        print(
            f"    {name:<26} n={values['support']:>5}  "
            f"P={values['precision']:.3f} R={values['recall']:.3f} F1={values['f1']:.3f}"
        )
    print("  confusion (rows true, cols predicted):")
    for true_name, row in summary["confusion_named"].items():
        cells = "  ".join(f"{name.split('_')[0][:8]:>9}:{count:>5}" for name, count in row.items())
        print(f"    {true_name:<26} {cells}")
    print("  confidence curve:")
    for row in summary["confidence_curve"]:
        print(
            f"    p>={row['threshold']:.1f}  coverage {row['coverage']:.2f}  "
            f"accuracy {row['accuracy']:.4f}"
        )
    print(f"\n[grade] wrote {config.REPORTS / 'run.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
