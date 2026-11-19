#!/bin/bash
# Linux: ./launch_viewer.sh  (or double-click if your file manager runs it in a terminal)
cd "$(dirname "$0")" || exit 1
echo "Close this window to stop the viewer."
echo
PY=""
if [ -x ".venv/bin/python" ]; then
  PY=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PY="python3"
elif command -v python >/dev/null 2>&1; then
  PY="python"
fi
if [ -z "$PY" ]; then
  echo "Python 3 was not found. Install Python 3 and try again."
  exit 1
fi
export PYTHONPATH="$PWD/src:$PWD:${PYTHONPATH:-}"
exec "$PY" -u -m wmb.viewer --launch

