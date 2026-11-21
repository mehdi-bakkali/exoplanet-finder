#!/bin/zsh
set -e
cd "$(dirname "$0")"
if [[ -x .venv/bin/python ]]; then
  exo_python=.venv/bin/python
elif [[ -x ../../work/.venv/bin/python ]]; then
  exo_python=../../work/.venv/bin/python
else
  if ! command -v uv >/dev/null 2>&1; then
    echo 'Install uv first: https://docs.astral.sh/uv/getting-started/installation/'
    exit 1
  fi
  uv venv --python 3.12
  uv pip install --python .venv/bin/python -r requirements-lock.txt
  exo_python=.venv/bin/python
fi
printf '\nExoplanet Finder: http://127.0.0.1:8000\nPress Control-C to stop.\n\n'
exec "$exo_python" -m uvicorn exofinder.server:app --host 127.0.0.1 --port 8000
