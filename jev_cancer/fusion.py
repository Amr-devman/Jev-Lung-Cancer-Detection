"""From Jev's answers to a class, as a two-stage hierarchical decision.

Stage 1 — normal tissue vs carcinoma. Only signals that can tell untouched
tissue from a proliferation vote here (the architecture Choices, the growth
arrangement Choice, the stroma Choice and the observation rubric), projected to
a binary normal/carcinoma question.

Stage 2 — glandular vs keratinizing, asked only once stage 1 says carcinoma.
Only signals that can separate the two carcinomas vote here (the architecture
Choices, the growth pattern Choice, the differentiation tie-breaker Choice and
the observation rubric), projected to the gland/keratin pair.

The final probabilities are the hierarchical product: p(normal) from stage 1,
p(gland) and p(keratin) as the carcinoma mass split by stage 2. Nothing is
fitted; the weights are fixed clinical priors in ``WEIGHTS``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import PROFILE_IDS
from .pipeline import feature_vector
from .protocol import (
    ARRANGEMENT_SIGNAL,
    CELL_SHAPE_SIGNAL,
    CYTOPLASM_SIGNAL,
    DIFFERENTIATION_SIGNAL,
    KERATIN_SIGNAL,
    LUMEN_SIGNAL,
    PROLIFERATION_PATTERN_SIGNAL,
    STROMA_SIGNAL,
    rubric_vector,
)

WEIGHTS = {
    "profile": 0.28,
    "profile_recheck": 0.10,
    "proliferation_pattern": 0.08,
    "rubric": 0.12,
    "cell_shape": 0.05,
    "cytoplasm": 0.05,
    "stroma": 0.05,
    "arrangement": 0.08,
    "differentiation": 0.09,
    "lumen_type": 0.05,
    "keratin_evidence": 0.05,
}
STAGE1_SIGNALS = ("profile", "profile_recheck", "arrangement", "stroma", "rubric")
STAGE2_SIGNALS = (
    "profile", "profile_recheck", "proliferation_pattern", "differentiation",
    "lumen_type", "keratin_evidence", "rubric", "cell_shape", "cytoplasm",
)
RUBRIC_TEMPERATURE = 6.0
EPS = 1e-9

# Identity map for the profile Choices: option label p0/p1/p2 -> one-hot.
PROFILE_SIGNAL = {
    "p0": (1.0, 0.0, 0.0),
    "p1": (0.0, 1.0, 0.0),
    "p2": (0.0, 0.0, 1.0),
}


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
        "profile": mapped_choice(response, "profile", PROFILE_SIGNAL),
        "profile_recheck": mapped_choice(response, "profile_recheck", PROFILE_SIGNAL),
        "proliferation_pattern": mapped_choice(
            response, "proliferation_pattern", PROLIFERATION_PATTERN_SIGNAL
        ),
        "rubric": softmax(np.array(rubric_vector(feature_vector(response))), RUBRIC_TEMPERATURE),
        "cell_shape": mapped_choice(response, "cell_shape", CELL_SHAPE_SIGNAL),
        "cytoplasm": mapped_choice(response, "cytoplasm", CYTOPLASM_SIGNAL),
        "stroma": mapped_choice(response, "stroma", STROMA_SIGNAL),
        "arrangement": mapped_choice(response, "growth_arrangement", ARRANGEMENT_SIGNAL),
        "differentiation": mapped_choice(response, "differentiation", DIFFERENTIATION_SIGNAL),
        "lumen_type": mapped_choice(response, "lumen_type", LUMEN_SIGNAL),
        "keratin_evidence": mapped_choice(response, "keratin_evidence", KERATIN_SIGNAL),
    }


def _weighted_mean(vectors: list[np.ndarray]) -> np.ndarray:
    total = 0.0
    combined = np.zeros(len(vectors[0]))
    for vector, weight in vectors:
        combined += weight * vector
        total += weight
    return combined / total if total > EPS else normalize(combined)


def stage1_normal_probability(signals: dict[str, np.ndarray]) -> float:
    """P(normal tissue) from the signals that can judge it."""
    votes = []
    for name in STAGE1_SIGNALS:
        signal = signals.get(name)
        if signal is None:
            continue
        binary = normalize(np.array([signal[1], signal[0] + signal[2]]))
        votes.append((binary, WEIGHTS.get(name, 0.0)))
    return float(_weighted_mean(votes)[0])


def stage2_gland_probability(signals: dict[str, np.ndarray]) -> float:
    """P(glandular | carcinoma) from the signals that can separate the two."""
    votes = []
    for name in STAGE2_SIGNALS:
        signal = signals.get(name)
        if signal is None:
            continue
        pair = normalize(np.array([signal[0], signal[2]]))
        votes.append((pair, WEIGHTS.get(name, 0.0)))
    return float(_weighted_mean(votes)[0])


def predict(response: dict[str, Any]) -> dict[str, Any]:
    signals = build_signals(response)
    normal = stage1_normal_probability(signals)
    gland = stage2_gland_probability(signals)
    probabilities = {
        "p1": normal,
        "p0": (1.0 - normal) * gland,
        "p2": (1.0 - normal) * (1.0 - gland),
    }
    order = sorted(probabilities, key=probabilities.get, reverse=True)
    top = order[0]
    return {
        "profile": top,
        "probabilities": probabilities,
        "confidence": probabilities[top],
        "margin": probabilities[top] - probabilities[order[1]],
        "stages": {
            "normal_vs_carcinoma": {"normal": normal, "carcinoma": 1.0 - normal},
            "gland_vs_keratin": {"gland": gland, "keratin": 1.0 - gland},
        },
        "signals": {
            name: {profile: float(value[i]) for i, profile in enumerate(PROFILE_IDS)}
            for name, value in signals.items()
        },
    }
