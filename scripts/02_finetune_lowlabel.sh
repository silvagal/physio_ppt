#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR/physio_ppt/src:${PYTHONPATH:-}"

CFG="${1:-physio_ppt/configs/train/physio_ppt.yaml}"
SEED="${2:-42}"
DEVICE="${3:-cuda}"
CKPT="${4:-}"

CMD=(python3 -m physio_ppt.cli run --config "$CFG" --seed "$SEED" --device "$DEVICE" --override pipeline.do_pretrain=false --override pipeline.do_finetune=true)

if [[ -n "$CKPT" ]]; then
  CMD+=(--override "train.pretrained_checkpoint=$CKPT")
fi

"${CMD[@]}"
