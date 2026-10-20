"""Grading: accuracy, per-class metrics and confidence intervals."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .config import CLASS_IDS


def confusion_matrix(
    predictions: list[str], truths: list[str], labels: tuple[str, ...] | list[str] = CLASS_IDS
) -> np.ndarray:
    index = {label: i for i, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for predicted, truth in zip(predictions, truths, strict=True):
        matrix[index[truth], index[predicted]] += 1
    return matrix


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def per_class(
    matrix: np.ndarray, labels: tuple[str, ...] | list[str] = CLASS_IDS
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for i, label in enumerate(labels):
        tp = int(matrix[i, i])
        fn = int(matrix[i, :].sum() - tp)
        fp = int(matrix[:, i].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        result[label] = {
            "support": int(matrix[i, :].sum()),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return result


def summarize(
    predictions: list[str],
    truths: list[str],
    labels: tuple[str, ...] | list[str] = CLASS_IDS,
) -> dict[str, Any]:
    matrix = confusion_matrix(predictions, truths, labels)
    total = int(matrix.sum())
    correct = int(np.trace(matrix))
    recalls = [result["recall"] for result in per_class(matrix, labels).values()]
    low, high = wilson_interval(correct, total)
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "accuracy_ci95": [low, high],
        "balanced_accuracy": float(np.mean(recalls)) if recalls else 0.0,
        "per_class": per_class(matrix, labels),
        "matrix": matrix.tolist(),
    }


def bootstrap_accuracy(
    predictions: list[str], truths: list[str], rounds: int = 2000, seed: int = 7
) -> tuple[float, float]:
    """Stratified bootstrap over images; reported next to the Wilson interval."""
    if not predictions:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    correct = np.array([p == t for p, t in zip(predictions, truths, strict=True)])
    values = []
    for _ in range(rounds):
        sample = rng.integers(0, len(correct), len(correct))
        values.append(float(correct[sample].mean()))
    return (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5)))
