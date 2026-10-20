"""Question-tuning experiments.

Iterate on the clinical question bank and measure which question wording leads
to the highest accuracy on a fixed random panel of 10 images (4/3/3 stratified
by class). Each iteration runs one variant over the panel and logs overall,
binary and per-class accuracy to ``reports/tuning/log.jsonl``.

Design notes
------------
- The panel is fixed and stratified so variants are compared on identical
  pixels; only the questions change. The panel is drawn with a seed and never
  edited by hand.
- One iteration = one variant tested on 10 images = 10 image-tests. The
  experiment budget is 100 iterations / 1,000 image-tests.
- Raw answers are cached per variant signature under ``runs/tuning/``, so
  re-scoring or re-exporting a variant is free.
- The fusion weights in ``jev_cancer.fusion`` stay fixed unless a variant
  explicitly declares an ablation; question text is the variable under test.
- Every variant passes ``protocol.assert_no_leaks`` before any request: the
  tuning loop must not smuggle diagnosis names into prompts even to win points.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jev_cancer import config, fusion
from jev_cancer.client import Jev, load_json, save_json
from jev_cancer.index import build_index, load_rgb
from jev_cancer.metrics import summarize
from jev_cancer.pipeline import feature_vector
from jev_cancer.protocol import (
    CELL_SHAPE_SIGNAL,
    CYTOPLASM_SIGNAL,
    FEATURES,
    NOUL_CRITERIA,
    PROFILE_TEXT,
    PROLIFERATION_PATTERN_SIGNAL,
    PROLIFERATION_PATTERN_TEXT,
    STROMA_SIGNAL,
    assert_no_leaks,
    rubric_vector,
)

ROOT = Path(__file__).resolve().parent.parent
PANEL_JSON = Path(__file__).resolve().parent / "panel.json"
TUNING_RUNS = config.RUNS / "tuning"
TUNING_REPORTS = config.REPORTS / "tuning"
LOG_JSONL = TUNING_REPORTS / "log.jsonl"

PANEL_SEED = 20261021
PANEL_PER_CLASS = (4, 3, 3)  # sorted class order: adeno 4, benign 3, squamous 3

CLASS_NAMES = sorted(config.SOURCE_CLASSES)
PROFILE_TO_NAME = {
    profile: CLASS_NAMES[int(class_id[1])]
    for profile, class_id in config.PROFILE_TO_CLASS.items()
}


# --------------------------------------------------------------------------
# Panel
# --------------------------------------------------------------------------

def draw_panel(seed: int = PANEL_SEED, per_class: int | tuple[int, ...] = PANEL_PER_CLASS) -> dict:
    """Draw a fixed evaluation panel: random, stratified, seeded."""
    counts = per_class if isinstance(per_class, tuple) else (per_class,) * len(sorted(config.SOURCE_CLASSES))
    suffix = "" if per_class == PANEL_PER_CLASS else f"_n{sum(counts)}"
    path = (
        PANEL_JSON if seed == PANEL_SEED and not suffix
        else PANEL_JSON.with_name(f"panel_s{seed}{suffix}.json")
    )
    if path.exists():
        return json.loads(path.read_text())
    index = build_index()
    rng = random.Random(seed)
    by_class: dict[str, list[dict]] = {}
    for item in index["images"]:
        by_class.setdefault(item["class"], []).append(item)
    picked = []
    for class_name, count in zip(sorted(by_class), counts, strict=True):
        picked.extend(rng.sample(sorted(by_class[class_name], key=lambda i: i["id"]), count))
    picked.sort(key=lambda item: item["id"])
    panel = {
        "seed": seed,
        "per_class": dict(zip(sorted(by_class), counts, strict=True)),
        "images": picked,
    }
    path.write_text(json.dumps(panel, indent=2))
    print(f"[panel] drew {len(picked)} images (seed {seed}) -> {path}")
    return panel


# --------------------------------------------------------------------------
# Variant construction
# --------------------------------------------------------------------------

def build_variant_questions(variant: dict) -> dict[str, Any]:
    """Build a question bank from the base protocol plus variant overrides."""
    from typesafe_sdk import Choice, Noul

    from jev_cancer.protocol import (
        CELL_SHAPE_TEXT,
        CYTOPLASM_TEXT,
        STROMA_TEXT,
    )

    drop = set(variant.get("drop", ()))
    feature_text = variant.get("feature_text", {})
    criteria = variant.get("criteria", NOUL_CRITERIA)
    feature_criteria = variant.get("feature_criteria", {})
    profile_text = {**PROFILE_TEXT, **variant.get("profile_text", {})}
    cell_shape_text = {**CELL_SHAPE_TEXT, **variant.get("cell_shape_text", {})}
    cytoplasm_text = {**CYTOPLASM_TEXT, **variant.get("cytoplasm_text", {})}
    stroma_text = {**STROMA_TEXT, **variant.get("stroma_text", {})}
    prolif_text = {**PROLIFERATION_PATTERN_TEXT, **variant.get("proliferation_text", {})}
    instructions_over = variant.get("instructions", {})

    questions: dict[str, Any] = {}
    for feature in FEATURES:
        if feature.id in drop:
            continue
        ask, look = feature_text.get(feature.id, (feature.ask, feature.look))
        questions[feature.id] = Noul(
            instructions=f"{ask}\n\nWhat to look for: {look}",
            criteria=feature_criteria.get(feature.id, dict(criteria)),
        )
    for feature in variant.get("extra_features", ()):
        questions[feature.id] = Noul(
            instructions=f"{feature.ask}\n\nWhat to look for: {feature.look}",
            criteria=feature_criteria.get(feature.id, dict(criteria)),
        )

    def choice(qid: str, instructions: str, table: dict[str, str]) -> None:
        if qid in drop:
            return
        questions[qid] = Choice(
            instructions=instructions_over.get(qid, instructions),
            criteria=table,
        )

    choice("profile", (
        "Which description best matches the dominant architecture in `sample`? Read the whole "
        "field first, then pick the description whose pattern occupies the largest area."
    ), {p: profile_text[p] for p in config.PROFILE_IDS})
    choice("profile_recheck", (
        "Judge the architecture, not the staining. Which description fits the majority of the "
        "field in `sample`?"
    ), {p: profile_text[p] for p in ("p2", "p0", "p1")})
    choice("cell_shape", (
        "Look only at the epithelial cells (the cells that form structures or sheets). What is "
        "their dominant shape in `sample`?"
    ), cell_shape_text)
    choice("cytoplasm", (
        "Look at the cell body outside the nucleus of the epithelial cells. What is the dominant "
        "cytoplasm in `sample`?"
    ), cytoplasm_text)
    choice("stroma", (
        "Look at what sits between the epithelial structures: the supporting tissue and the "
        "background. What lies between them in `sample`?"
    ), stroma_text)
    choice("proliferation_pattern", (
        "If `sample` contains a crowded epithelial proliferation (many cells growing together "
        "and outgrowing the surrounding tissue), which pattern is the better fit? If it does "
        "not, choose neither_clear."
    ), prolif_text)
    for qid, instructions, table in variant.get("extra_choices", ()):
        questions[qid] = Choice(instructions=instructions, criteria=table)
    return questions


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def variant_signals(response: dict, variant: dict) -> dict[str, np.ndarray]:
    signals = {
        "profile": mapped(response, "profile", PROFILE_SIGNAL),
        "profile_recheck": mapped(response, "profile_recheck", PROFILE_SIGNAL),
        "proliferation_pattern": mapped(response, "proliferation_pattern", PROLIFERATION_PATTERN_SIGNAL),
        "rubric": fusion.softmax(
            np.array(rubric_vector(feature_vector(response))), fusion.RUBRIC_TEMPERATURE
        ),
        "cell_shape": mapped(response, "cell_shape", CELL_SHAPE_SIGNAL),
        "cytoplasm": mapped(response, "cytoplasm", CYTOPLASM_SIGNAL),
        "stroma": mapped(response, "stroma", STROMA_SIGNAL),
    }
    for name in variant.get("drop_signals", ()):
        signals.pop(name, None)
    for name, spec in variant.get("extra_signals", {}).items():
        signals[name] = mapped(response, spec["question"], spec["mapping"])
    return signals


# Identity map for the profile Choices: option label p0/p1/p2 -> one-hot.
PROFILE_SIGNAL = {
    "p0": (1.0, 0.0, 0.0),
    "p1": (0.0, 1.0, 0.0),
    "p2": (0.0, 0.0, 1.0),
}


def mapped(response: dict, key: str, table: dict[str, tuple]) -> np.ndarray:
    answer = response["answers"].get(key)
    if not answer or answer.get("type") != "choice":
        return np.full(3, 1 / 3)
    scores = np.zeros(3)
    for option, probability in answer["probabilities"].items():
        scores += probability * np.array(table.get(option, (1 / 3,) * 3))
    return fusion.normalize(scores)


def score_variant(records: list[dict], variant: dict, panel: dict) -> dict:
    truth_of = {item["id"]: item["class"] for item in panel["images"]}
    preds, truths, rows = [], [], []
    for record in records:
        signals = variant_signals(record["response"], variant)
        weights = {**fusion.WEIGHTS, **variant.get("weights", {})}
        fused = fusion.fuse(signals, weights=weights)
        top = int(np.argmax(fused))
        profile = config.PROFILE_IDS[top]
        pred_name = PROFILE_TO_NAME[profile]
        truth_name = truth_of[record["image_id"]]
        preds.append(pred_name)
        truths.append(truth_name)
        rows.append({
            "image_id": record["image_id"],
            "truth": truth_name,
            "pred": pred_name,
            "confidence": round(float(fused[top]), 3),
            "probabilities": {
                PROFILE_TO_NAME[config.PROFILE_IDS[i]]: round(float(fused[i]), 3) for i in range(3)
            },
            "ok": pred_name == truth_name,
        })

    labels = sorted(set(truths) | set(preds))
    matrix = summarize(preds, truths, labels=labels)
    binary = ["benign" if name == "benign" else "carcinoma" for name in preds]
    binary_true = ["benign" if name == "benign" else "carcinoma" for name in truths]
    return {
        "overall": matrix["accuracy"],
        "binary": float(np.mean([p == t for p, t in zip(binary, binary_true, strict=True)])),
        "per_class": {
            name: matrix["per_class"][name]["recall"]
            for name in labels
        },
        "per_class_accuracy": {
            name: float(np.mean([r["ok"] for r in rows if r["truth"] == name])) for name in labels
        },
        "confusion": matrix["confusion_named"] if "confusion_named" in matrix else matrix["matrix"],
        "rows": rows,
    }


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------

def run_variant(variant: dict, panel: dict, workers: int = 8) -> tuple[dict, Path]:
    from typesafe_sdk import Image as JevImage

    questions = build_variant_questions(variant)
    assert_no_leaks(questions)

    payload = {
        "version": variant["name"],
        "note": variant.get("note", ""),
        "questions": {
            qid: {
                "instructions": getattr(q, "instructions", None),
                "criteria": getattr(q, "criteria", None),
            }
            for qid, q in questions.items()
        },
    }
    import hashlib

    sig = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]
    cache_dir = TUNING_RUNS / f"{variant['name']}_{sig}"
    (cache_dir / "raw").mkdir(parents=True, exist_ok=True)

    jev = Jev()
    pending = [
        item for item in panel["images"]
        if not (cache_dir / "raw" / f"{item['id']}.json").exists()
    ]
    print(f"[tune] {variant['name']}: {len(panel['images'])} images, {len(pending)} to do -> {cache_dir.name}")
    wall = time.perf_counter()
    try:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {}
            for item in pending:
                def work(item=item):
                    image = JevImage.from_pil(load_rgb(item["source"]))
                    response = jev.ask({"sample": image}, questions)
                    save_json(cache_dir / "raw" / f"{item['id']}.json", response)
                futures[pool.submit(work)] = item["id"]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    print(f"[tune] error {futures[future]}: {type(exc).__name__}: {exc}")
    finally:
        jev.close()

    records = [
        {"image_id": item["id"], "response": load_json(cache_dir / "raw" / f"{item['id']}.json")}
        for item in panel["images"]
        if (cache_dir / "raw" / f"{item['id']}.json").exists()
    ]
    scores = score_variant(records, variant, panel)
    scores["cost_usd"] = jev.input_tokens * config.USD_PER_BTOK / 1e9
    scores["wall_seconds"] = round(time.perf_counter() - wall, 1)
    return scores, cache_dir


def log_iteration(variant: dict, scores: dict, panel_seed: int) -> None:
    """Append or refresh this variant's entry. One entry per (variant, panel)
    keeps the trend log honest when a variant is re-scored."""
    TUNING_REPORTS.mkdir(parents=True, exist_ok=True)
    entries = [
        json.loads(line)
        for line in (LOG_JSONL.read_text().splitlines() if LOG_JSONL.exists() else [])
    ]
    entries = [
        entry
        for entry in entries
        if not (entry["variant"] == variant["name"] and entry.get("panel_seed") == panel_seed)
    ]
    entry = {
        "variant": variant["name"],
        "panel_seed": panel_seed,
        "note": variant.get("note", ""),
        "drop": variant.get("drop", ()),
        "image_tests": len(scores["rows"]),
        "overall": scores["overall"],
        "binary": scores["binary"],
        "per_class": scores["per_class_accuracy"],
    }
    entries.append(entry)
    spent = 0
    for index, item in enumerate(entries, start=1):
        item["iteration"] = index
        spent += item.get("image_tests", 10)
        item["total_image_tests"] = spent
    LOG_JSONL.write_text("".join(json.dumps(item) + "\n" for item in entries))
    return entries[-1]


def print_scores(entry: dict, scores: dict) -> None:
    trend = ""
    entries = [json.loads(line) for line in LOG_JSONL.read_text().splitlines()]
    same_panel = [
        item for item in entries
        if item.get("panel_seed") == entry.get("panel_seed") and item["iteration"] != entry["iteration"]
    ]
    if same_panel:
        prev = max(same_panel, key=lambda item: item["iteration"])
        delta = entry["overall"] - prev["overall"]
        trend = f"  ({delta:+.2f} vs iter {prev['iteration']} [{prev['variant']}])"
    print(
        f"\n=== iter {entry['iteration']:03d}  {entry['variant']}  [panel s{entry['panel_seed']}]{trend}\n"
        f"    {entry['note']}\n"
        f"    overall {entry['overall']:.2f}  binary {entry['binary']:.3f}  "
        f"per-class " + "  ".join(f"{k.split('_')[0][:5]}={v:.2f}" for k, v in entry["per_class"].items())
    )
    for row in scores["rows"]:
        mark = "ok " if row["ok"] else "ERR"
        print(
            f"    {mark} {row['image_id']}  truth={row['truth'][:12]:<12} "
            f"pred={row['pred'][:12]:<12} conf={row['confidence']:.2f} "
            f"{row['probabilities']}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, help="variant name from experiments/variants.py")
    parser.add_argument("--panel-seed", type=int, default=PANEL_SEED, help="evaluation panel seed")
    parser.add_argument("--per-class", type=int, default=None, help="images per class (default 4/3/3)")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    from variants import VARIANTS  # experiments/variants.py

    if args.variant not in VARIANTS and args.variant != "all":
        raise SystemExit(f"unknown variant {args.variant!r}; have: {sorted(VARIANTS)}")

    panel = draw_panel(args.panel_seed, args.per_class or PANEL_PER_CLASS)
    names = sorted(VARIANTS) if args.variant == "all" else [args.variant]
    for name in names:
        variant = dict(VARIANTS[name], name=name)
        scores, _ = run_variant(variant, panel, workers=args.workers)
        entry = log_iteration(variant, scores, panel["seed"])
        print_scores(entry, scores)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
