#!/usr/bin/env bash
set -euo pipefail

VIEWER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${REPLAY_VIEWER_PYTHON:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x "$VIEWER_DIR/.venv/bin/python" ]]; then
    PYTHON_BIN="$VIEWER_DIR/.venv/bin/python"
  elif [[ -x /16T-2/sudu/code/RLinf/.venv/bin/python ]]; then
    PYTHON_BIN=/16T-2/sudu/code/RLinf/.venv/bin/python
  else
    PYTHON_BIN=python3
  fi
fi

cd "$VIEWER_DIR"
echo "Replay Buffer Viewer: http://127.0.0.1:${REPLAY_VIEWER_PORT:-8088}"
exec "$PYTHON_BIN" app.py
