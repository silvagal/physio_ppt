#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# Usar o venv do projeto; permite sobrescrever via variavel de ambiente PYTHON
PYTHON="${PYTHON:-/media/data/eduardo/venv_jackal/bin/python}"
export PYTHONPATH="$ROOT_DIR/physio_ppt/src:${PYTHONPATH:-}"

DEVICE="${DEVICE:-cuda}"
PARALLEL="${PARALLEL:-1}"
SEEDS=(42 1337 2025 7 123)

E1_CFG="physio_ppt/configs/train/physio_ppt.yaml"
E3_CFGS=(
  "physio_ppt/configs/train/supervised_strong.yaml"
  "physio_ppt/configs/train/ppt_classic.yaml"
  "physio_ppt/configs/train/wavepuzzle.yaml"
  "physio_ppt/configs/train/physio_ppt.yaml"
)

CMD_FILE="$(mktemp)"
trap 'rm -f "$CMD_FILE"' EXIT

# E1: Physio-PPT low-label 1/5/10%
for seed in "${SEEDS[@]}"; do
  echo "$PYTHON -m physio_ppt.cli run --config $E1_CFG --seed $seed --device $DEVICE" >> "$CMD_FILE"
done

# E3: 4-way comparison at 10% labels only
for cfg in "${E3_CFGS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    if [[ "$cfg" == *"supervised_strong"* ]]; then
      echo "$PYTHON -m physio_ppt.cli run --config $cfg --seed $seed --device $DEVICE --override pipeline.label_fractions=0.1 --override pipeline.do_pretrain=false" >> "$CMD_FILE"
    else
      echo "$PYTHON -m physio_ppt.cli run --config $cfg --seed $seed --device $DEVICE --override pipeline.label_fractions=0.1" >> "$CMD_FILE"
    fi
  done
done

if [[ "$PARALLEL" -gt 1 ]]; then
  xargs -I{} -P "$PARALLEL" bash -lc "{}" < "$CMD_FILE"
else
  while IFS= read -r cmd; do
    bash -lc "$cmd"
  done < "$CMD_FILE"
fi

# E4: orderness analysis
$PYTHON -m physio_ppt.cli analyze --config physio_ppt/configs/eval/orderness_analysis.yaml --seed 42 --device cpu
$PYTHON -m physio_ppt.cli eval --config physio_ppt/configs/eval/low_label_1_5_10.yaml --seed 42 --device cpu
$PYTHON -m physio_ppt.cli make_figures --config physio_ppt/configs/eval/low_label_1_5_10.yaml --seed 42 --device cpu \
  --override paths.figures_dir=physio_ppt/figures \
  --override figures.low_label_csv=physio_ppt/outputs/tables/low_label_summary.csv \
  --override figures.ablation_csv=physio_ppt/outputs/tables/ablation_summary.csv \
  --override figures.delta_csv=physio_ppt/outputs/tables/delta_by_seed.csv \
  --override figures.orderness_csv=physio_ppt/outputs/tables/orderness_vs_gain.csv
