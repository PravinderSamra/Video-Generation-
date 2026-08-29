"""Small JSON-over-HTTP helper shared by the service-backed backends.

stdlib only, so the stub/CI path stays dependency-free.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from .base import BackendError


def request_json(
    url: str,
    payload: dict[str, Any] | None = None,
    timeout: int = 180,
    headers: dict[str, str] | None = None,
) -> Any:
    """POST JSON (or GET when payload is None) and decode the JSON response."""
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    all_headers = {"Content-Type": "application/json", **(headers or {})}
    req = urllib.request.Request(url, data=data, headers=all_headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:400]
        raise BackendError(f"{url} returned HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise BackendError(f"cannot reach {url}: {exc}") from exc


def request_bytes(
    url: str,
    payload: dict[str, Any] | None = None,
    timeout: int = 600,
    headers: dict[str, str] | None = None,
) -> bytes:
    """Same, but for endpoints that return binary (an encoded video)."""
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    all_headers = {"Content-Type": "application/json", **(headers or {})}
    req = urllib.request.Request(url, data=data, headers=all_headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:400]
        raise BackendError(f"{url} returned HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise BackendError(f"cannot reach {url}: {exc}") from exc


def is_reachable(url: str, timeout: int = 3) -> bool:
    try:
        urllib.request.urlopen(urllib.request.Request(url), timeout=timeout)
        return True
    except urllib.error.HTTPError:
        return True  # responded, just not with 200 — the service is up
    except Exception:
        return False
