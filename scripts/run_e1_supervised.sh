#!/usr/bin/env bash
# Script para rodar E1 apenas para Supervisionado Forte (baseline sem SSL)
# PTB-XL finetune low-label (1%, 5%, 10%) sem pré-treino
# Seeds: [42, 1337, 2025, 7, 123]

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# Usar o venv do projeto; permite sobrescrever via variavel de ambiente PYTHON
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="$ROOT_DIR/physio_ppt/src:${PYTHONPATH:-}"

DEVICE="${DEVICE:-cuda}"
SEEDS=(42 1337 2025 7 123)

# Config do Supervisionado Forte
E1_CFG="physio_ppt/configs/train/supervised_strong.yaml"

echo "=========================================="
echo "E1: Supervisionado Forte - Low-Label (1%, 5%, 10%)"
echo "Device: $DEVICE"
echo "Seeds: ${SEEDS[@]}"
echo "=========================================="
echo ""

# Contador de experimentos
total_experiments=$((${#SEEDS[@]}))
current=0

# E1: Supervisionado Forte low-label com 1%, 5%, 10%
# Nota: usa as frações default do config OU podemos iterar explicitamente
for seed in "${SEEDS[@]}"; do
    current=$((current + 1))
    echo ""
    echo "=========================================="
    echo "[$current/$total_experiments] Running: Supervised Strong (Seed: $seed)"
    echo "Fractions: 1%, 5%, 10% (default)"
    echo "=========================================="
    
    $PYTHON -m physio_ppt.cli run \
        --config "$E1_CFG" \
        --seed $seed \
        --device $DEVICE \
        --override pipeline.do_pretrain=false
    
    echo "✓ Supervised Strong (Seed: $seed) completed"
done

echo ""
echo "=========================================="
echo "✓ ALL E1 SUPERVISED STRONG EXPERIMENTS COMPLETED"
echo "=========================================="
echo "Total runs: $total_experiments"
echo "Each run includes finetune (no pretrain) for 1%, 5%, 10% labels"
echo "Results saved in: physio_ppt/outputs/"
echo ""
