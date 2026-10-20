"""From Jev's answers to a profile.

There is no training and no reference image. Seven signals are combined with
fixed clinical-prior weights:

  profile               the architecture Choice
  profile_recheck       the same Choice with reordered options
  proliferation_pattern soft map of the growth-pattern Choice
  rubric                the weighted 42-observation vector
  cell_shape            soft map of the cell-shape Choice
  cytoplasm             soft map of the cytoplasm Choice
  stroma                soft map of the stroma Choice

The output is a probability over p0/p1/p2 plus confidence and margin.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import PROFILE_IDS
from .pipeline import feature_vector, profile_probabilities
from .protocol import (
    CELL_SHAPE_SIGNAL,
    CYTOPLASM_SIGNAL,
    PROLIFERATION_PATTERN_SIGNAL,
    STROMA_SIGNAL,
    rubric_vector,
)

WEIGHTS = {
    "profile": 0.35,
    "profile_recheck": 0.15,
    "proliferation_pattern": 0.15,
    "rubric": 0.15,
    "cell_shape": 0.08,
    "cytoplasm": 0.07,
    "stroma": 0.05,
}
RUBRIC_TEMPERATURE = 6.0
EPS = 1e-9


def softmax(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(scores, dtype=float) / max(temperature, EPS)
    z -= z.max()
    exp = np.exp(z)
    return exp / exp.sum()


def normalize(scores: np.ndarray) -> np.ndarray:
    scores = np.clip(np.asarray(scores, dtype=float), 0.0, None)
    total = scores.sum()
    return scores / total if total > EPS else np.full(len(scores), 1.0 / len(scores))


def mapped_choice(response: dict[str, Any], key: str, mapping: dict[str, tuple]) -> np.ndarray:
    answer = response["answers"].get(key)
    if not answer or answer.get("type") != "choice":
        return np.full(3, 1 / 3)
    scores = np.zeros(3)
    for option, probability in answer["probabilities"].items():
        scores += probability * np.array(mapping.get(option, (1 / 3,) * 3))
    return normalize(scores)


def build_signals(response: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        "profile": np.array(profile_probabilities(response, "profile")),
        "profile_recheck": np.array(profile_probabilities(response, "profile_recheck")),
        "proliferation_pattern": mapped_choice(
            response, "proliferation_pattern", PROLIFERATION_PATTERN_SIGNAL
        ),
        "rubric": softmax(np.array(rubric_vector(feature_vector(response))), RUBRIC_TEMPERATURE),
        "cell_shape": mapped_choice(response, "cell_shape", CELL_SHAPE_SIGNAL),
        "cytoplasm": mapped_choice(response, "cytoplasm", CYTOPLASM_SIGNAL),
        "stroma": mapped_choice(response, "stroma", STROMA_SIGNAL),
    }


def fuse(signals: dict[str, np.ndarray], weights: dict[str, float] | None = None) -> np.ndarray:
    weights = weights if weights is not None else WEIGHTS
    total = 0.0
    combined = np.zeros(3)
    for name, signal in signals.items():
        weight = weights.get(name, 0.0)
        if weight <= 0:
            continue
        combined += weight * normalize(signal)
        total += weight
    return combined / total if total > EPS else np.full(3, 1 / 3)


def predict(response: dict[str, Any]) -> dict[str, Any]:
    signals = build_signals(response)
    fused = fuse(signals)
    order = np.argsort(fused)[::-1]
    top = int(order[0])
    return {
        "profile": PROFILE_IDS[top],
        "probabilities": {profile: float(fused[i]) for i, profile in enumerate(PROFILE_IDS)},
        "confidence": float(fused[top]),
        "margin": float(fused[top] - fused[order[1]]),
        "signals": {
            name: {profile: float(value[i]) for i, profile in enumerate(PROFILE_IDS)}
            for name, value in signals.items()
        },
    }
