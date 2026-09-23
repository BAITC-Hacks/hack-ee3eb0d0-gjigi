"""Minimal Open-Meteo HTTP client with on-disk response cache, throttling and retries."""

import gzip
import hashlib
import json
import threading
import time
from pathlib import Path

import requests

SINGLE_RUNS_URL = "https://single-runs-api.open-meteo.com/v1/forecast"
PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"


class OpenMeteoError(RuntimeError):
    pass


class OpenMeteoClient:
    def __init__(self, cache_dir: Path, min_interval_s: float = 0.15,
                 retries: int = 6, timeout_s: float = 60):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_interval_s = min_interval_s
        self.retries = retries
        self.timeout_s = timeout_s
        self._lock = threading.Lock()
        self._last_call = 0.0

    def _cache_path(self, url: str, params: dict) -> Path:
        key = json.dumps([url, sorted(params.items())], sort_keys=True)
        return self.cache_dir / f"{hashlib.sha1(key.encode()).hexdigest()}.json.gz"

    def _throttle(self):
        with self._lock:
            wait = self._last_call + self.min_interval_s - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()

    def get(self, url: str, params: dict, offline: bool = False) -> dict:
        """GET JSON; cached responses are returned without network access.

        Returns {"error": True, "reason": ...} bodies as-is (e.g. run not
        archived) so callers can decide; raises on transport/HTTP failures.
        """
        path = self._cache_path(url, params)
        if path.exists():
            with gzip.open(path, "rt", encoding="utf-8") as f:
                return json.load(f)
        if offline:
            raise OpenMeteoError(f"not in cache (offline mode): {url} {params}")

        delay = 2.0
        for attempt in range(self.retries):
            self._throttle()
            try:
                r = requests.get(url, params=params, timeout=self.timeout_s)
            except requests.RequestException as e:
                err = e
            else:
                if r.status_code == 200 or (r.status_code == 400 and "reason" in r.text):
                    data = r.json()
                    with gzip.open(path, "wt", encoding="utf-8") as f:
                        json.dump(data, f)
                    return data
                err = OpenMeteoError(f"HTTP {r.status_code}: {r.text[:200]}")
                if r.status_code == 429:
                    delay = max(delay, 30.0)  # rate limit: back off harder
            time.sleep(delay)
            delay = min(delay * 2, 300)
        raise OpenMeteoError(f"failed after {self.retries} attempts: {err}")
