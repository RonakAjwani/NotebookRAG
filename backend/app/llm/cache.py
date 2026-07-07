"""Prompt-hash response cache.

A JSON-file-per-entry cache keyed by a stable hash of (provider, model, params,
messages). Its purpose is resume-and-skip: a crashed eval run replays finished
calls for free, and config changes only repay the calls whose inputs changed.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from typing import Any, Optional


class PromptCache:
    def __init__(self, cache_dir: str, enabled: bool = True):
        self.enabled = enabled
        self.dir = os.path.join(cache_dir, "llm")
        self._lock = threading.Lock()
        if self.enabled:
            os.makedirs(self.dir, exist_ok=True)

    @staticmethod
    def make_key(payload: dict[str, Any]) -> str:
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> str:
        return os.path.join(self.dir, f"{key}.json")

    def get(self, key: str) -> Optional[str]:
        if not self.enabled:
            return None
        path = self._path(key)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)["response"]
        except (OSError, KeyError, json.JSONDecodeError):
            return None

    def set(self, key: str, response: str, meta: Optional[dict] = None) -> None:
        if not self.enabled:
            return
        with self._lock:
            tmp = self._path(key) + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"response": response, "meta": meta or {}}, f, ensure_ascii=False)
            os.replace(tmp, self._path(key))
