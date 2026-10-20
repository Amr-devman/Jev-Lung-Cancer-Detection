"""Thin TypeSafe client wrapper: one client per worker thread, answers to JSON.

Every response is cached on disk with its token usage and latency, so a rerun
costs nothing and the report can quote the API's own numbers.
"""

from __future__ import annotations

import contextlib
import json
import threading
from pathlib import Path
from typing import Any

from typesafe_sdk import TypeSafeClient

from .config import DANGER_ZONE_HEADERS, MODEL


def answer_to_dict(answer: Any) -> dict[str, Any]:
    if answer.type == "noul":
        return {"type": "noul", "noul": float(answer.noul)}
    if answer.type == "choice":
        return {
            "type": "choice",
            "choice": answer.choice,
            "confidence": float(answer.confidence),
            "probabilities": {k: float(v) for k, v in answer.probabilities.items()},
        }
    return {"type": answer.type}


class Jev:
    """Thread-safe facade over the sync SDK.

    ``system_one`` is called from many worker threads. A client per thread keeps
    each worker's connection pool independent, and the lock only protects the
    shared counters.
    """

    def __init__(self, model: str = MODEL) -> None:
        self.model = model
        self._local = threading.local()
        self._clients: list[TypeSafeClient] = []
        self._lock = threading.Lock()
        self.input_tokens = 0
        self.requests = 0

    def _client(self) -> TypeSafeClient:
        client = getattr(self._local, "client", None)
        if client is None:
            client = TypeSafeClient()
            self._local.client = client
            with self._lock:
                self._clients.append(client)
        return client

    def ask(self, state: dict[str, Any], questions: dict[str, Any]) -> dict[str, Any]:
        """Send one request. ``state`` may hold ``typesafe_sdk.Image`` objects;
        the SDK lifts them into the request's assets."""
        response = self._client().system_one(
            state=state,
            questions=questions,
            model=self.model,
            extra_headers=DANGER_ZONE_HEADERS,
        )
        with self._lock:
            self.input_tokens += response.usage.input_tokens or 0
            self.requests += 1
        return {
            "answers": {name: answer_to_dict(answer) for name, answer in response.answers.items()},
            "usage": {
                "input_tokens": response.usage.input_tokens or 0,
                "output_tokens": response.usage.output_tokens or 0,
            },
            "model": response.model,
        }

    def close(self) -> None:
        for client in self._clients:
            with contextlib.suppress(Exception):
                client.close()
        self._clients.clear()


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())
