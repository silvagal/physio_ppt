#!/bin/bash
# Script para rodar experimentos E3 (comparação 4-way) com todas as 5 seeds
# Métodos: Supervisionado Forte, PPT Clássico, WavePuzzle
# Label fraction: 10%

set -e  # Exit on error

# Seeds oficiais do paper
SEEDS=(42 1337 2025 7 123)
DEVICE=${DEVICE:-cuda}
LABEL_FRAC=${LABEL_FRAC:-0.1}

# Diretório base (assumindo execução de physio_ppt/)
BASE_DIR="/media/data/eduardo/ECG_SSL_MultipleLeads"
cd "$BASE_DIR"

echo "=========================================="
echo "E3: Comparação 4-way em ${LABEL_FRAC} labels"
echo "Device: $DEVICE"
echo "Seeds: ${SEEDS[@]}"
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
    
    # 1. Supervisionado Forte
    current=$((current + 1))
    echo ""
    echo "[$current/$total_experiments] Running: Supervised Strong (Seed: $seed)"
    echo "---"
    python3 -m physio_ppt.cli run \
        --config physio_ppt/configs/train/supervised_strong.yaml \
        --seed $seed \
        --device $DEVICE \
        --override pipeline.label_fractions=$LABEL_FRAC \
        --override pipeline.do_pretrain=false
    
    echo "✓ Supervised Strong (Seed: $seed) completed"
    
    # 2. PPT Clássico
    current=$((current + 1))
    echo ""
    echo "[$current/$total_experiments] Running: PPT Classic (Seed: $seed)"
    echo "---"
    python3 -m physio_ppt.cli run \
        --config physio_ppt/configs/train/ppt_classic.yaml \
        --seed $seed \
        --device $DEVICE \
        --override pipeline.label_fractions=$LABEL_FRAC
    
    echo "✓ PPT Classic (Seed: $seed) completed"
    
    # 3. WavePuzzle
    current=$((current + 1))
    echo ""
    echo "[$current/$total_experiments] Running: WavePuzzle (Seed: $seed)"
    echo "---"
    python3 -m physio_ppt.cli run \
        --config physio_ppt/configs/train/wavepuzzle.yaml \
        --seed $seed \
        --device $DEVICE \
        --override pipeline.label_fractions=$LABEL_FRAC
    
    echo "✓ WavePuzzle (Seed: $seed) completed"
    
done

echo ""
echo "=========================================="
echo "✓ ALL E3 EXPERIMENTS COMPLETED"
echo "=========================================="
echo "Total experiments run: $total_experiments"
echo "Results saved in: physio_ppt/outputs/"
echo ""
