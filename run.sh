#!/usr/bin/env bash
set -euo pipefail

VIEWER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${REPLAY_VIEWER_PYTHON:-/home/user/miniconda3/envs/rlinf/bin/python}"

cd "$VIEWER_DIR"
echo "Replay Buffer Viewer: http://127.0.0.1:${REPLAY_VIEWER_PORT:-8088}"
exec "$PYTHON_BIN" app.py
