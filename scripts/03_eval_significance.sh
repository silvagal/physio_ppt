#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR/physio_ppt/src:${PYTHONPATH:-}"

CFG="${1:-physio_ppt/configs/eval/significance_bootstrap.yaml}"
PRED_A="${2:-}"
PRED_B="${3:-}"
SEED="${4:-42}"
DEVICE="${5:-cpu}"

if [[ -z "$PRED_A" || -z "$PRED_B" ]]; then
  echo "Usage: $0 <config> <pred_a.npz> <pred_b.npz> [seed] [device]"
  exit 1
fi

python3 -m physio_ppt.cli eval \
  --config "$CFG" \
  --seed "$SEED" \
  --device "$DEVICE" \
  --override "eval.pred_a=$PRED_A" \
  --override "eval.pred_b=$PRED_B"
