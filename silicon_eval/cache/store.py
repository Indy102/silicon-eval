"""Content-addressed result cache.

A cache key is the SHA-256 of the canonical JSON of everything that
determines a variant's results — model, quantization, runtime, profiling and
eval configuration, machine (chip, memory, OS), and the silicon-eval, schema,
and backend (mlx/mlx-lm) versions. The CLI assembles the payload
(``_cache_payload_base``); the full rationale lives in
docs/adr/003-result-cache.md. Change any input and the key changes, so
re-runs only compute what changed.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_ENV_CACHE_DIR = "SILICON_EVAL_CACHE_DIR"


def default_cache_dir() -> Path:
    """Cache root: $SILICON_EVAL_CACHE_DIR or ~/.cache/silicon-eval."""
    override = os.environ.get(_ENV_CACHE_DIR)
    if override:
        return Path(override)
    return Path.home() / ".cache" / "silicon-eval"


def cache_key(payload: Mapping[str, Any]) -> str:
    """Deterministic key for a JSON-representable payload (order-insensitive)."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ResultCache:
    """Stores one JSON document per key; unreadable entries count as misses."""

    def __init__(self, root: Path | None = None) -> None:
        self._root = root if root is not None else default_cache_dir()

    @property
    def root(self) -> Path:
        """Directory this cache reads and writes."""
        return self._root

    def get(self, key: str) -> dict[str, Any] | None:
        """Stored document for ``key``, or None on miss/corruption."""
        path = self._path(key)
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return loaded if isinstance(loaded, dict) else None

    def put(self, key: str, value: dict[str, Any]) -> None:
        """Store ``value`` under ``key`` (atomic tmp-file + rename)."""
        self._root.mkdir(parents=True, exist_ok=True)
        path = self._path(key)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        tmp.replace(path)

    def _path(self, key: str) -> Path:
        return self._root / f"{key}.json"
