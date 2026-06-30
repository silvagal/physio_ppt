#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR/physio_ppt/src:${PYTHONPATH:-}"

CFG="${1:-physio_ppt/configs/eval/orderness_analysis.yaml}"
SEED="${2:-42}"
DEVICE="${3:-cpu}"

python3 -m physio_ppt.cli analyze --config "$CFG" --seed "$SEED" --device "$DEVICE"
