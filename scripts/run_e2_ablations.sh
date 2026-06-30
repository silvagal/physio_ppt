#!/usr/bin/env bash
# Script para rodar E2: Ablações Physio-PPT
# Testa importância de cada componente: consistency, contrastive, inter-segment
# MIT-BIH pretrain + PTB-XL finetune low-label (1%, 5%, 10%)
# Seeds: [42, 1337, 2025, 7, 123]

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# Usar o venv do projeto; permite sobrescrever via variavel de ambiente PYTHON
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="$ROOT_DIR/physio_ppt/src:${PYTHONPATH:-}"

DEVICE="${DEVICE:-cuda}"
SEEDS=(42 1337 2025 7 123)

# Config base do Physio-PPT
E2_CFG="physio_ppt/configs/train/physio_ppt.yaml"

echo "=========================================="
echo "E2: Ablações Physio-PPT"
echo "Device: $DEVICE"
echo "Seeds: ${SEEDS[@]}"
echo "Variantes: 3 (consistency-only, contrastive-only, no-inter-segment)"
echo "=========================================="
echo ""

# Contador de experimentos
total_experiments=$((${#SEEDS[@]} * 3))
current=0

# Loop sobre todas as seeds
for seed in "${SEEDS[@]}"; do
    echo ""
    echo "=========================================="
    echo "SEED: $seed"
    echo "=========================================="
    
    # 1. Somente Consistency
    current=$((current + 1))
    echo ""
    echo "[$current/$total_experiments] Running: Ablation - Consistency Only (Seed: $seed)"
    echo "---"
    $PYTHON -m physio_ppt.cli run \
        --config "$E2_CFG" \
        --seed $seed \
        --device $DEVICE \
        --override ssl.use_consistency=true \
        --override ssl.use_contrastive=false
    
    echo "✓ Ablation - Consistency Only (Seed: $seed) completed"
    
    # 2. Somente Contrastive
    current=$((current + 1))
    echo ""
    echo "[$current/$total_experiments] Running: Ablation - Contrastive Only (Seed: $seed)"
    echo "---"
    $PYTHON -m physio_ppt.cli run \
        --config "$E2_CFG" \
        --seed $seed \
        --device $DEVICE \
        --override ssl.use_consistency=false \
        --override ssl.use_contrastive=true
    
    echo "✓ Ablation - Contrastive Only (Seed: $seed) completed"
    
    # 3. Ambos + sem inter-segment (apenas intra)
    current=$((current + 1))
    echo ""
    echo "[$current/$total_experiments] Running: Ablation - No Inter-Segment (Seed: $seed)"
    echo "---"
    $PYTHON -m physio_ppt.cli run \
        --config "$E2_CFG" \
        --seed $seed \
        --device $DEVICE \
        --override ssl.perturb_mode=intra
    
    echo "✓ Ablation - No Inter-Segment (Seed: $seed) completed"
    
done

echo ""
echo "=========================================="
echo "✓ ALL E2 ABLATION EXPERIMENTS COMPLETED"
echo "=========================================="
echo "Total experiments run: $total_experiments"
echo "Variants:"
echo "  - Consistency Only: ${#SEEDS[@]} runs"
echo "  - Contrastive Only: ${#SEEDS[@]} runs"
echo "  - No Inter-Segment: ${#SEEDS[@]} runs"
echo ""
echo "Each run includes:"
echo "  - MIT-BIH pretrain with ablated loss"
echo "  - PTB-XL finetune with 1%, 5%, 10% labels"
echo ""
echo "Results saved in: physio_ppt/outputs/"
echo ""
