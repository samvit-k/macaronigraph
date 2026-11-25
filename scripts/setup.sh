#!/usr/bin/env bash
# Project-local toolchain (no sudo, nothing system-wide):
#   ffmpeg + tesseract -> .tools/env via micromamba (conda-forge)
#   Python 3.11 deps   -> .venv via uv
# Idempotent: re-running only installs what is missing.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS="$ROOT/.tools"
ENV="$TOOLS/env"
VENV="$ROOT/.venv"
MAMBA="$TOOLS/bin/micromamba"
REQUIRED_PY="3.11"

die() { echo "ERROR: $*" >&2; exit 1; }
step() { echo "==> $*"; }

[ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ] || die "MLX-Whisper requires macOS on Apple Silicon (arm64)."
command -v uv >/dev/null || die "uv not found on PATH (https://docs.astral.sh/uv/)."
command -v curl >/dev/null || die "curl not found."

# 1. micromamba
mkdir -p "$TOOLS/bin"
if [ ! -x "$MAMBA" ]; then
  step "downloading micromamba"
  tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
  curl -fsSL --retry 3 https://micro.mamba.pm/api/micromamba/osx-arm64/latest -o "$tmp/mm.tar.bz2" \
    || die "micromamba download failed"
  tar -xjf "$tmp/mm.tar.bz2" -C "$tmp" bin/micromamba || die "micromamba archive extraction failed"
  mv "$tmp/bin/micromamba" "$MAMBA" && chmod +x "$MAMBA"
fi

# 2. ffmpeg + tesseract
if [ ! -x "$ENV/bin/ffmpeg" ] || [ ! -x "$ENV/bin/tesseract" ]; then
  step "installing ffmpeg + tesseract into .tools/env"
  export MAMBA_ROOT_PREFIX="$TOOLS/mamba"
  if [ -d "$ENV" ]; then
    "$MAMBA" install -y -p "$ENV" -c conda-forge ffmpeg tesseract || die "micromamba install failed"
  else
    "$MAMBA" create -y -p "$ENV" -c conda-forge ffmpeg tesseract || die "micromamba create failed"
  fi
fi

# 3. Python venv (must be 3.11; an incompatible one is moved aside, never deleted)
if [ -x "$VENV/bin/python" ]; then
  have="$("$VENV/bin/python" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")' 2>/dev/null || echo broken)"
  if [ "$have" != "$REQUIRED_PY" ]; then
    backup="$VENV.bak-$(date +%Y%m%d%H%M%S)"
    echo "WARN: existing .venv is Python $have (need $REQUIRED_PY); moving it to $(basename "$backup")"
    mv "$VENV" "$backup"
  fi
elif [ -e "$VENV" ]; then
  backup="$VENV.bak-$(date +%Y%m%d%H%M%S)"
  echo "WARN: .venv exists but has no python; moving it to $(basename "$backup")"
  mv "$VENV" "$backup"
fi
if [ ! -x "$VENV/bin/python" ]; then
  step "creating .venv (Python $REQUIRED_PY)"
  uv venv -p "$REQUIRED_PY" "$VENV" || die "uv could not create a Python $REQUIRED_PY venv"
fi
step "installing Python dependencies"
VIRTUAL_ENV="$VENV" uv pip install -q -r "$ROOT/requirements.txt" || die "pip install failed"

# 4. Verify everything through the same resolution path the backend uses.
step "verifying toolchain"
"$VENV/bin/python" "$ROOT/scripts/check_env.py" || die "environment verification failed"
step "setup complete"
