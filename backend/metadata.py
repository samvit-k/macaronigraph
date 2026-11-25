"""Read-only parsing of Instagram's saved_posts.json export."""
from __future__ import annotations

import copy
import json
import re
from datetime import datetime, timezone
from pathlib import Path

_ID_RE = re.compile(r"instagram\.com/(?:[^/]+/)?(reel|reels|p|tv)/([A-Za-z0-9_-]+)|/test-media/page/([A-Za-z0-9_-]+)")


def fix_text(s: str) -> str:
    """Instagram exports UTF-8 bytes as latin-1 code points (mojibake). Reverse it when that's the case."""
    if not s:
        return s
    try:
        return s.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s  # already proper text


def reel_id_from_url(url: str) -> str | None:
    m = _ID_RE.search(url or "")
    if not m:
        return None
    return m.group(2) or m.group(3)


def _dict_entries(block: dict) -> list[dict[str, str]]:
    """[{"dict":[{"label":..,"value":..}],"title":""}, ...] -> [{label: value}, ...]"""
    return [{f["label"]: fix_text(f.get("value", "")) for f in e.get("dict", []) if "label" in f}
            for e in block.get("dict", [])]


def parse_post(item: dict, index: int) -> dict:
    labels: dict[str, str] = {}
    blocks: dict[str, list] = {}
    for lv in item.get("label_values", []):
        if "label" in lv:
            labels.setdefault(lv["label"], lv.get("href") or lv.get("value", ""))
        elif "title" in lv:
            blocks[lv["title"]] = _dict_entries(lv)
    url = labels.get("URL", "")
    owners = blocks.get("Owner") or [{}]
    ts = item.get("timestamp")
    return {
        "reel_id": reel_id_from_url(url),
        "source_url": url,
        "source_index": index,
        "media_type": "reel" if "/reel" in url or "/test-media/" in url else ("post" if "/p/" in url else "unknown"),
        "saved_at": datetime.fromtimestamp(ts, timezone.utc).isoformat() if ts else None,
        "metadata": {
            "caption": fix_text(labels.get("Caption", "")),
            "title": fix_text(labels.get("Title", "")),
            "hashtags": [h.get("Name", "") for h in blocks.get("Hashtags", []) if h.get("Name")],
            "owner": {"name": owners[0].get("Name"), "username": owners[0].get("Username"),
                      "url": owners[0].get("URL")},
            "brand_partner": blocks.get("Brand partner", []),
            "raw": copy.deepcopy(item),  # original record, verbatim
        },
    }


def load_posts(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:  # read-only; the export is never modified
        data = json.load(f)
    if isinstance(data, dict):  # some export versions wrap the list
        data = next((v for v in data.values() if isinstance(v, list)), [])
    return [parse_post(item, i) for i, item in enumerate(data)]
