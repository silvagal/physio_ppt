#!/usr/bin/env bash
# Script para rodar E1 apenas para PPT Clássico
# MIT-BIH pretrain + PTB-XL finetune low-label (1%, 5%, 10%)
# Seeds: [42, 1337, 2025, 7, 123]

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# Usar o venv do projeto; permite sobrescrever via variavel de ambiente PYTHON
PYTHON="${PYTHON:-/media/data/eduardo/venv_jackal/bin/python}"
export PYTHONPATH="$ROOT_DIR/physio_ppt/src:${PYTHONPATH:-}"

DEVICE="${DEVICE:-cuda}"
SEEDS=(42 1337 2025 7 123)

# Config do PPT Clássico
E1_CFG="physio_ppt/configs/train/ppt_classic.yaml"

echo "=========================================="
echo "E1: PPT Clássico - Low-Label (1%, 5%, 10%)"
echo "Device: $DEVICE"
echo "Seeds: ${SEEDS[@]}"
echo "=========================================="
echo ""

# Contador de experimentos
total_experiments=$((${#SEEDS[@]}))
current=0

# E1: PPT Clássico low-label com 1%, 5%, 10% (default do config)
for seed in "${SEEDS[@]}"; do
    current=$((current + 1))
    echo ""
    echo "=========================================="
    echo "[$current/$total_experiments] Running: PPT Classic (Seed: $seed)"
    echo "Fractions: 1%, 5%, 10% (default)"
    echo "=========================================="
    
    $PYTHON -m physio_ppt.cli run \
        --config "$E1_CFG" \
        --seed $seed \
        --device $DEVICE
    
    echo "✓ PPT Classic (Seed: $seed) completed"
done

echo ""
echo "=========================================="
echo "✓ ALL E1 PPT CLASSIC EXPERIMENTS COMPLETED"
echo "=========================================="
echo "Total runs: $total_experiments"
echo "Each run includes pretrain + finetune for 1%, 5%, 10% labels"
echo "Results saved in: physio_ppt/outputs/"
echo ""
