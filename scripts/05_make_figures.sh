#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR/physio_ppt/src:${PYTHONPATH:-}"

CFG="${1:-physio_ppt/configs/eval/low_label_1_5_10.yaml}"
SEED="${2:-42}"
DEVICE="${3:-cpu}"

python3 -m physio_ppt.cli make_figures \
  --config "$CFG" \
  --seed "$SEED" \
  --device "$DEVICE" \
  --override paths.figures_dir=physio_ppt/figures \
  --override figures.low_label_csv=physio_ppt/outputs/tables/low_label_summary.csv \
  --override figures.ablation_csv=physio_ppt/outputs/tables/ablation_summary.csv \
  --override figures.delta_csv=physio_ppt/outputs/tables/delta_by_seed.csv \
  --override figures.orderness_csv=physio_ppt/outputs/tables/orderness_vs_gain.csv
