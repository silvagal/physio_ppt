#!/usr/bin/env bash
# Script para completar E2: Ablações Physio-PPT — SEED=123 APENAS
# Roda as 3 variantes faltantes: cons_only, contrast_only, intra_only
# Aguarda o run_all.sh (PID passado como arg ou detectado) terminar primeiro.
#
# Uso:
#   ./physio_ppt/scripts/run_e2_seed123.sh [PID_DO_RUN_ALL]
#
# Se nenhum PID for fornecido, roda imediatamente.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-/media/data/eduardo/venv_jackal/bin/python}"
export PYTHONPATH="$ROOT_DIR/physio_ppt/src:${PYTHONPATH:-}"
DEVICE="${DEVICE:-cuda}"
E2_CFG="physio_ppt/configs/train/physio_ppt.yaml"
SEED=123
LOG="physio_ppt/outputs/run_e2_seed123.log"

# Aguardar PID pai se fornecido
if [[ $# -ge 1 && "$1" =~ ^[0-9]+$ ]]; then
    WAIT_PID="$1"
    echo "[$(date)] Aguardando PID $WAIT_PID terminar antes de iniciar..." | tee -a "$LOG"
    while kill -0 "$WAIT_PID" 2>/dev/null; do
        sleep 60
    done
    echo "[$(date)] PID $WAIT_PID terminou. Iniciando E2 seed=123." | tee -a "$LOG"
fi

echo "==========================================" | tee -a "$LOG"
echo "E2 Ablações — SEED=${SEED} (3 variantes)" | tee -a "$LOG"
echo "Device: $DEVICE" | tee -a "$LOG"
echo "Início: $(date)" | tee -a "$LOG"
echo "==========================================" | tee -a "$LOG"

# 1. Somente Consistency (cons_only)
echo "" | tee -a "$LOG"
echo "[1/3] cons_only (seed=${SEED})" | tee -a "$LOG"
$PYTHON -m physio_ppt.cli run \
    --config "$E2_CFG" \
    --seed $SEED \
    --device $DEVICE \
    --override ssl.use_consistency=true \
    --override ssl.use_contrastive=false \
    2>&1 | tee -a "$LOG"
echo "✓ cons_only concluído: $(date)" | tee -a "$LOG"

# 2. Somente Contrastive (contrast_only)
echo "" | tee -a "$LOG"
echo "[2/3] contrast_only (seed=${SEED})" | tee -a "$LOG"
$PYTHON -m physio_ppt.cli run \
    --config "$E2_CFG" \
    --seed $SEED \
    --device $DEVICE \
    --override ssl.use_consistency=false \
    --override ssl.use_contrastive=true \
    2>&1 | tee -a "$LOG"
echo "✓ contrast_only concluído: $(date)" | tee -a "$LOG"

# 3. Ambos + perturb intra-segment (intra_only)
echo "" | tee -a "$LOG"
echo "[3/3] intra_only (seed=${SEED})" | tee -a "$LOG"
$PYTHON -m physio_ppt.cli run \
    --config "$E2_CFG" \
    --seed $SEED \
    --device $DEVICE \
    --override ssl.perturb_mode=intra \
    2>&1 | tee -a "$LOG"
echo "✓ intra_only concluído: $(date)" | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "==========================================" | tee -a "$LOG"
echo "✅ TODOS OS 3 ABLATION RUNS SEED=123 CONCLUÍDOS" | tee -a "$LOG"
echo "Fim: $(date)" | tee -a "$LOG"
echo "==========================================" | tee -a "$LOG"
