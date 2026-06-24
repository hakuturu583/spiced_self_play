#!/usr/bin/env bash
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required for this training entrypoint. Install uv, then rerun." >&2
  exit 1
fi

uv run --no-sync python - <<'PY'
import importlib.util
import sys

missing = [name for name in ("torch", "gymnasium", "rich", "rich_argparse") if importlib.util.find_spec(name) is None]
if missing:
    print("Missing Python packages needed for PufferDrive training: " + ", ".join(missing), file=sys.stderr)
    print("Create the uv environment and install training dependencies, then rerun this script:", file=sys.stderr)
    print("  uv venv --python 3.10", file=sys.stderr)
    print("  uv pip install torch --index-url https://download.pytorch.org/whl/cu128", file=sys.stderr)
    print("  uv pip install 'numpy<2.0' Cython wheel", file=sys.stderr)
    print("  uv pip install --no-build-isolation -e .", file=sys.stderr)
    raise SystemExit(1)
PY

uv run --no-sync python scripts/build_town07_drive_maps.py
uv run --no-sync python -m pufferlib.pufferl train puffer_drive_town07 "$@"
