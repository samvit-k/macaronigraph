"""Package the Chrome extension from an explicit file allowlist (no dependencies: also used by the release workflow).

Usage: python -m backend.extension_package OUT.zip
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

EXTENSION_DIR = Path(__file__).resolve().parent.parent / "extension"
ZIP_ROOT = "macaronigraph-extension"
FILES = (
    "manifest.json", "background.js", "content.js", "crop.js", "offscreen.html", "offscreen.js",
    "ui/popup.html", "ui/popup.css", "ui/popup.js", "ui/logo.svg",
    "ui/icons/icon16.png", "ui/icons/icon32.png", "ui/icons/icon48.png", "ui/icons/icon128.png",
)

def write_zip(target) -> None:
    """Raises FileNotFoundError naming the first missing file; nothing outside FILES is ever included."""
    missing = [rel for rel in FILES if not (EXTENSION_DIR / rel).is_file()]
    if missing:
        raise FileNotFoundError(f"extension file missing: {missing[0]}")
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
        for rel in FILES:
            z.write(EXTENSION_DIR / rel, f"{ZIP_ROOT}/{rel}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    write_zip(sys.argv[1])
    print(f"wrote {sys.argv[1]} ({len(FILES)} files)")