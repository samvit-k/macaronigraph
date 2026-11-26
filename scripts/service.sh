#!/usr/bin/env bash
# Optional macOS login service (launchd): starts the backend at login on 127.0.0.1 only and restarts it if it exits.
#   scripts/service.sh install | uninstall | [status]
set -euo pipefail

CMD="${1:-status}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LABEL="com.macaronigraph.backend"
AGENTS_DIR="${MACARONIGRAPH_AGENTS_DIR:-$HOME/Library/LaunchAgents}"
PLIST="$AGENTS_DIR/$LABEL.plist"
LAUNCHCTL="${LAUNCHCTL:-launchctl}"
DOMAIN="gui/$(id -u)"
PORT="${PORT:-8765}"
WAIT_S="${SERVICE_WAIT_S:-60}"
UVICORN="${SERVICE_UVICORN:-$ROOT/.venv/bin/uvicorn}"
LOG="${SERVICE_LOG:-$ROOT/output/logs/backend.log}"
URL="http://127.0.0.1:$PORT"

die() { echo "error: $*" >&2; exit 1; }
healthy() { curl -fsS --max-time 2 "$URL/api/health" >/dev/null 2>&1; }
loaded() { "$LAUNCHCTL" print "$DOMAIN/$LABEL" >/dev/null 2>&1; }
xml() { printf '%s' "$1" | sed -e 's/&/\&amp;/g' -e 's/</\&lt;/g' -e 's/>/\&gt;/g'; }

write_plist() {
  cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>$(xml "$UVICORN")</string>
    <string>backend.app:app</string>
    <string>--host</string><string>127.0.0.1</string>
    <string>--port</string><string>$PORT</string>
  </array>
  <key>WorkingDirectory</key><string>$(xml "$ROOT")</string>
  <key>EnvironmentVariables</key>
  <dict><key>PORT</key><string>$PORT</string><key>PYTHONUNBUFFERED</key><string>1</string></dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><dict><key>SuccessfulExit</key><false/></dict>
  <key>ThrottleInterval</key><integer>10</integer>
  <key>StandardOutPath</key><string>$(xml "$LOG")</string>
  <key>StandardErrorPath</key><string>$(xml "$LOG")</string>
</dict>
</plist>
EOF
  plutil -lint "$PLIST" >/dev/null || die "generated plist is invalid: $PLIST"
}

install() {
  [[ -x "$UVICORN" ]] || die "$UVICORN not found: run ./scripts/setup.sh first"
  [[ "$PORT" =~ ^[0-9]+$ ]] || die "PORT must be a number"
  if loaded; then
    "$LAUNCHCTL" bootout "$DOMAIN/$LABEL" >/dev/null 2>&1 || true
  elif healthy; then
    die "a backend is already running on $URL (probably in Terminal): stop it with Ctrl-C, then run install again"
  fi
  mkdir -p "$AGENTS_DIR" "$(dirname "$LOG")"
  write_plist
  "$LAUNCHCTL" bootstrap "$DOMAIN" "$PLIST" || die "launchctl could not load $PLIST"
  echo "installed $LABEL: starts at login, restarts if it stops (log: $LOG)"
  local waited=0
  until healthy; do
    if (( waited >= WAIT_S )); then
      echo "error: the backend did not answer on $URL within ${WAIT_S}s; see $LOG" >&2
      echo "  If the log shows 'Operation not permitted', allow the Python in .venv access to this folder" >&2
      echo "  (System Settings → Privacy & Security), or keep the project outside Desktop/Documents/Downloads." >&2
      exit 1
    fi
    sleep 1; waited=$((waited + 1))
  done
  echo "running: $URL/workspace/"
}

uninstall() {
  if loaded; then "$LAUNCHCTL" bootout "$DOMAIN/$LABEL" >/dev/null 2>&1 || true; fi
  if [[ -f "$PLIST" ]]; then rm -f "$PLIST"; echo "removed $LABEL (the log stays in $LOG)"; else echo "not installed"; fi
}

status() {
  local installed=no running=no answering=no
  [[ -f "$PLIST" ]] && installed=yes
  loaded && running=yes
  healthy && answering=yes
  echo "login service: installed=$installed loaded=$running · backend on $URL: answering=$answering"
}

case "$CMD" in
  install) install ;;
  uninstall) uninstall ;;
  status) status ;;
  *) die "usage: scripts/service.sh install | uninstall | [status]" ;;
esac