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
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jev_cancer import config
from jev_cancer.index import class_root, load_detail, load_rgb, load_tiles

DOCS = config.ROOT / "docs"
CLASS_NAMES = sorted(config.SOURCE_CLASSES)
PROFILE_ORDER = [f"p{i}" for i in range(3)]  # p0..p2 map to sorted classes c0..c2

PRETTY_CLASS = {
    "aca": "lung adenocarcinoma",
    "nor": "normal lung",
    "ssc": "squamous cell carcinoma",
}
PRETTY_PROFILE = {
    "p0": "rings-around-holes growth",
    "p1": "air-hole layout",
    "p2": "solid pink growth",
}
PRETTY_OPTION = {
    "pre_existing_structures": "cells follow built-in structures",
    "infiltrative_glands": "rings invading tissue",
    "expansile_solid_nests": "sheets replacing tissue",
    "no_proliferation": "no new growth",
    "built_in_structures": "cells follow built-in structures",
    "invading_rings": "rings invading tissue",
    "replacing_sheets": "sheets replacing tissue",
    "no_new_growth": "no new growth",
    "rings_and_goo": "rings and pale goo",
    "hard_pink_material": "hard bright-pink material",
    "both_kinds": "both kinds",
    "neither": "neither",
    "neither_clear": "neither",
    "pink_onion_rings": "onion rings of bright pink",
    "shrivelled_pink_cells": "shrivelled pink cells",
    "lined_holes": "holes lined by a cell layer",
    "unlined_holes": "unlined holes",
    "slit_gaps": "slit-like gaps",
    "no_open_spaces": "no open spaces",
    "gland_type": "rings and pale goo",
    "keratin_type": "hard bright-pink material",
    "both_present": "both kinds",
    "keratin_pearls": "onion rings of bright pink",
    "dyskeratotic_cells": "shrivelled pink cells",
    "lined_gland_lumens": "holes lined by a cell layer",
}
FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")


def fonts() -> dict:
    return {
        "title": ImageFont.truetype(FONT_DIR / "DejaVuSans-Bold.ttf", 26),
        "step": ImageFont.truetype(FONT_DIR / "DejaVuSans-Bold.ttf", 21),
        "body": ImageFont.truetype(FONT_DIR / "DejaVuSans.ttf", 16),
        "small": ImageFont.truetype(FONT_DIR / "DejaVuSans.ttf", 14),
        "label": ImageFont.truetype(FONT_DIR / "DejaVuSans-Bold.ttf", 24),
    }


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


# --- animated webp: image -> Jev's answers -> the label, for three samples ---

W, H = 960, 600
INK = "#16181d"
MUTED = "#6b7280"
ACCENT = "#2563eb"
GOOD = "#047857"


def short_question(ask: str) -> str:
    for lead in ("Does `sample` ", "Does the field ", "Are there ", "Are ", "Do ", "Is ", "Can "):
        if ask.startswith(lead):
            ask = ask[len(lead):]
            break
    return ask.rstrip("?").replace("`sample`", "the field")


def wrap(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    width: int,
) -> list[str]:
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if draw.textlength(trial, font=font) <= width:
            line = trial
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


def paste_field(canvas: Image.Image, path: Path) -> None:
    field = load_rgb(path, max_side=10_000)
    field = field.resize((460, 345), Image.Resampling.LANCZOS)
    canvas.paste(field, (30, 150))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((30, 150, 490, 495), outline="#d1d5db", width=2)


def frame_base(sample_no: int) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    canvas = Image.new("RGB", (W, H), "#fafafa")
    draw = ImageDraw.Draw(canvas)
    f = fonts()
    draw.rectangle((0, 0, W, 70), fill="#16181d")
    draw.text((30, 22), "From pixels to label", font=f["title"], fill="white")
    draw.text((W - 240, 28), f"sample {sample_no} of 3", font=f["body"], fill="#9ca3af")
    return canvas, draw


def frame_image(path: Path, sample_no: int) -> Image.Image:
    canvas, draw = frame_base(sample_no)
    f = fonts()
    paste_field(canvas, path)
    draw.text((530, 165), "Step 1  the image", font=f["step"], fill=INK)
    y = 215
    for line in wrap(draw, "One H&E field of lung tissue goes to the model. "
                     "It carries no class name, no filename and no metadata.", f["body"], 380):
        draw.text((530, y), line, font=f["body"], fill=INK)
        y += 26
    y += 12
    for line in wrap(draw, "Shown six ways in the real pipeline: the whole field, a "
                     "high-power view, and four zoomed tiles.", f["small"], 380):
        draw.text((530, y), line, font=f["small"], fill=MUTED)
        y += 22
    return canvas


def frame_answers(response: dict, sample_no: int, path: Path) -> Image.Image:
    canvas, draw = frame_base(sample_no)
    f = fonts()
    paste_field(canvas, path)
    draw.text((530, 100), "Step 2  Jev answers 55 questions", font=f["step"], fill=INK)
    from jev_cancer.protocol import FEATURES

    asks = {feature.id: feature.ask for feature in FEATURES}
    nouls = sorted(
        ((abs(a["noul"] - 0.5), qid, a["noul"]) for qid, a in response["answers"].items()
         if a.get("type") == "noul"),
        reverse=True,
    )[:5]
    y = 145
    for _, qid, value in nouls:
        yes = value >= 0.5
        draw.ellipse((530, y + 5, 544, y + 19), fill=ACCENT if yes else "#d1d5db")
        word = "yes" if yes else "no"
        draw.text((552, y), f"{word} {value:.2f}", font=f["body"], fill=INK)
        for i, line in enumerate(wrap(draw, short_question(asks[qid]), f["small"], 300)[:2]):
            draw.text((640, y + i * 18), line, font=f["small"], fill=MUTED)
        y += 44
    y += 8
    for qid, name in (("profile", "architecture"), ("growth_arrangement", "growth"),
                      ("differentiation", "tie-breaker"), ("keratin_evidence", "keratin")):
        answer = response["answers"][qid]
        option = max(answer["probabilities"], key=answer["probabilities"].get)
        prob = answer["probabilities"][option]
        text = f"{name}: {PRETTY_OPTION.get(option, PRETTY_PROFILE.get(option, option))}"
        draw.text((530, y), text[:46], font=f["small"], fill=INK)
        draw.text((880, y), f"{prob:.2f}", font=f["small"], fill=ACCENT)
        y += 26
    return canvas


def frame_label(prediction: dict, truth: str, sample_no: int, path: Path) -> Image.Image:
    canvas, draw = frame_base(sample_no)
    f = fonts()
    paste_field(canvas, path)
    stages = prediction["stages"]
    normal = stages["normal_vs_carcinoma"]["normal"]
    gland = stages["gland_vs_keratin"]["gland"]
    draw.text((530, 110), "Step 3  the decision", font=f["step"], fill=INK)
    draw.text((530, 160), f"stage 1   normal tissue vs carcinoma   ->  {1 - normal:.0%} carcinoma",
              font=f["small"], fill=MUTED)
    draw.text((530, 186), f"stage 2   glandular vs hard-pink      ->  {gland:.0%} glandular",
              font=f["small"], fill=MUTED)
    pred_name = CLASS_NAMES[int(prediction["profile"][1])]
    correct = pred_name == truth
    draw.rectangle((530, 230, 930, 330), fill="#eef2ff", outline=ACCENT, width=2)
    draw.text((550, 250), PRETTY_CLASS[pred_name], font=f["label"], fill=INK)
    draw.text((550, 292), f"confidence {prediction['confidence']:.0%}", font=f["body"], fill=ACCENT)
    colour = GOOD if correct else "#b91c1c"
    mark = "prediction correct" if correct else "prediction wrong"
    draw.text((530, 360), f"true label: {PRETTY_CLASS[truth]}", font=f["body"], fill=INK)
    draw.text((530, 390), mark, font=f["body"], fill=colour)
    return canvas


def flow_video(payload: dict, index: dict, picked: dict[str, Path]) -> None:
    id_of = {Path(item["source"]).name: item["id"] for item in index["images"]}
    prediction_of = {p["image_id"]: p for p in payload["predictions"]}
    raw_dir = payload["_run_dir"].parent / "raw"
    frames: list[Image.Image] = []
    durations: list[int] = []
    for no, name in enumerate(CLASS_NAMES, start=1):
        path = picked[name]
        image_id = id_of[path.name]
        prediction = prediction_of[image_id]
        response = json.loads((raw_dir / f"{image_id}.json").read_text())
        frames += [
            frame_image(path, no),
            frame_answers(response, no, path),
            frame_label(prediction, name, no, path),
        ]
        durations += [1700, 3800, 2800]
    frames[0].save(
        DOCS / "figures" / "flow.webp",
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        quality=82,
        method=4,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=None)
    args = parser.parse_args()

    for folder in (DOCS / "figures", DOCS / "samples"):
        folder.mkdir(parents=True, exist_ok=True)

    run_dir = args.run or find_latest_run()
    payload, index = load_run(run_dir)
    payload["_run_dir"] = run_dir
    pr_curves(payload, index)
    confusion(payload, index)
    picked = sample_images(index)
    views_figure(picked[CLASS_NAMES[0]])
    flow_video(payload, index, picked)
    print(f"[report] figures in {DOCS / 'figures'}")
    print(f"[report] samples in {DOCS / 'samples'}: " + ", ".join(p.name for p in picked.values()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
