"""Request guard: only this Mac's browser pages and the MacaroniGraph extension can use the backend.

Blocks DNS rebinding (unknown Host), other websites and other extensions (Origin), and cross-site
subresource or navigation requests to the API that browsers send without an Origin header.
"""
from __future__ import annotations

from urllib.parse import urlsplit

from . import config

# Fixed by the "key" in extension/manifest.json (tests/test_security.py checks they match).
EXTENSION_ID = "bigkdakgelhamamdngdhdcbogndohfnl"
LOOPBACK = {"127.0.0.1", "localhost", "::1"}
UNSAFE_NO_ORIGIN_MODES = {"no-cors", "navigate", "nested-navigate"}

def extension_origins() -> list[str]:
    return [f"chrome-extension://{i}" for i in (EXTENSION_ID, *config.EXTRA_EXTENSION_IDS)]

def _hostname(netloc: str) -> str:
    try:
        return (urlsplit(f"//{netloc}").hostname or "").lower()
    except ValueError:
        return ""

def host_allowed(host: str | None) -> bool:
    return bool(host) and _hostname(host) in config.ALLOWED_HOSTS

def origin_allowed(origin: str) -> bool:
    if origin in extension_origins():
        return True
    parts = urlsplit(origin)
    return parts.scheme in ("http", "https") and (parts.hostname or "").lower() in LOOPBACK

def rejection(path: str, headers) -> str | None:
    """Why a request must be refused, or None. `headers` is any case-insensitive mapping."""
    if not host_allowed(headers.get("host")):
        return "unknown host"
    origin = headers.get("origin")
    if origin is not None:
        return None if origin_allowed(origin) else "origin not allowed"
    if path.startswith("/api/") and headers.get("sec-fetch-site") in ("cross-site", "same-site") \
            and headers.get("sec-fetch-mode", "no-cors") in UNSAFE_NO_ORIGIN_MODES:
        return "cross-site request"
    return None