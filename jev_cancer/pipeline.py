"""One Jev request per image.

State: the image at two views (`sample` = whole field, `detail` = centre at
twice the magnification). Questions: 45 clinical observations with
plain-language descriptions and 8 morphology Choices. No reference images, no
labels, no boxes, no second request.

Raw answers are cached per image under ``runs/<dataset>/<protocol>/raw`` so any
subset or rerun only pays for images that have not been seen yet. The per-run
output directory is keyed by the subset's fraction and seed.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from typesafe_sdk import Image as JevImage

from . import config
from .client import Jev, load_json, save_json
from .index import load_detail, load_rgb, load_tiles, magnification, sibling_at, subset_key
from .protocol import assert_no_leaks, build_questions, question_signature


def protocol_dir() -> Path:
    return config.RUNS / config.DATASET_TAG / question_signature()


def run_dir(subset: dict) -> Path:
    return protocol_dir() / subset_key(subset)


def build_state(source: str) -> dict[str, Any]:
    """Six views of one tissue field.

    ``sample`` is the whole field as given. ``high_power`` is a strictly
    higher-magnification view of the same tissue: the genuine 40x sibling field
    when the dataset ships one (matched by filename, code-side only), otherwise
    a 2x digital zoom of the centre. The four ``tour_*`` tiles are 2x zoom
    tiles covering the whole field at native pixel scale, so no keratin pearl
    or bridge hides in the periphery.
    """
    path = Path(source)
    state: dict[str, Any] = {"sample": JevImage.from_pil(load_rgb(path))}
    pair = sibling_at(path, "40x") if magnification(path) == "20x" else None
    if pair is not None:
        state["high_power"] = JevImage.from_pil(load_rgb(pair))
    else:
        state["high_power"] = JevImage.from_pil(load_detail(path))
    for name, tile in load_tiles(path).items():
        state[name] = JevImage.from_pil(tile)
    return state


def classify_one(jev: Jev, image_id: str, source: str, cache_dir: Path) -> dict[str, Any]:
    path = cache_dir / "raw" / f"{image_id}.json"
    if path.exists():
        return {"image_id": image_id, "cached": True}

    questions = build_questions()
    assert_no_leaks(questions)
    started = time.perf_counter()
    response = jev.ask(build_state(source), questions)
    response["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
    save_json(path, response)
    return {"image_id": image_id, "cached": False}


def run_all(subset: dict, workers: int = config.DEFAULT_WORKERS) -> tuple[Path, list[dict]]:
    images = subset["images"]
    directory = run_dir(subset)
    cache_dir = protocol_dir()
    (cache_dir / "raw").mkdir(parents=True, exist_ok=True)
    directory.mkdir(parents=True, exist_ok=True)

    pending = [
        item
        for item in images
        if not (cache_dir / "raw" / f"{item['id']}.json").exists()
    ]
    print(f"[run] {len(images)} images, {len(pending)} to do, {workers} workers")
    print(f"[run] {directory}")

    jev = Jev()
    wall = time.perf_counter()
    done = 0
    errors = 0
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(classify_one, jev, item["id"], item["source"], cache_dir): item["id"]
                for item in pending
            }
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    errors += 1
                    print(f"[run] error: {type(exc).__name__}: {exc}")
                done += 1
                if done % 25 == 0 or done == len(pending):
                    elapsed = time.perf_counter() - wall
                    rate = done / elapsed if elapsed else 0
                    eta = (len(pending) - done) / rate / 60 if rate else 0
                    print(
                        f"[run] {done}/{len(pending)}  {rate:.2f} img/s  eta {eta:.0f} min  "
                        f"${jev.input_tokens * config.USD_PER_BTOK / 1e9:.4f}",
                        flush=True,
                    )
    finally:
        jev.close()

    summary = {
        "requested": len(images),
        "pending_at_start": len(pending),
        "errors": errors,
        "requests": jev.requests,
        "input_tokens": jev.input_tokens,
        "cost_usd": jev.input_tokens * config.USD_PER_BTOK / 1e9,
        "wall_seconds": round(time.perf_counter() - wall, 1),
        "model": config.MODEL,
        "protocol": question_signature(),
        "subset": {key: value for key, value in subset.items() if key != "images"},
    }
    save_json(directory / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return directory, images


def load_records(image_ids: list[str], cache_dir: Path | None = None) -> list[dict[str, Any]]:
    cache_dir = cache_dir or (protocol_dir() / "raw")
    records = []
    for image_id in image_ids:
        path = cache_dir / f"{image_id}.json"
        if path.exists():
            records.append({"image_id": image_id, "response": load_json(path)})
    return records


def feature_vector(response: dict[str, Any]) -> dict[str, float]:
    return {
        key: value["noul"]
        for key, value in response["answers"].items()
        if value.get("type") == "noul"
    }
