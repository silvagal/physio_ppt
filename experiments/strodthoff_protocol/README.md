# Strodthoff-Compatible Downstream Protocol (PTB-XL Superdiagnostic)

Esta linha é **nova e isolada** do pipeline legado.  
Nada do fluxo antigo foi removido ou alterado de forma destrutiva.

## Objetivo científico

Avaliar low-label learning (1%, 5%, 10%) no PTB-XL superdiagnostic com:

- 12 leads
- folds oficiais PTB-XL (`train=1..8`, `val=9`, `test=10`)
- inferência em nível de registro com sliding windows
- métrica primária Macro-AUROC

## Estrutura

```text
physio_ppt/experiments/strodthoff_protocol/
  configs/
    dataset/
    model/
    train/
    experiment/
  scripts/
  outputs/
  splits/
  tables/
  figures/
```

Código de suporte:

```text
physio_ppt/src/physio_ppt/strodthoff_protocol/
  data/
  models/
  training/
  evaluation/
  utils/
```

## Scripts principais

- `prepare_ptbxl_strodthoff.py`: prepara PTB-XL com folds oficiais.
- `make_low_label_splits_strodthoff.py`: gera arquivos persistidos de low-label por seed/fração.
- `make_strodthoff_splits.py`: wrapper que prepara e gera splits low-label.
- `train_supervised_strodthoff.py`: treino supervisionado from scratch.
- `finetune_ssl_strodthoff.py`: fine-tuning com checkpoint SSL (Physio-PPT compatível).
- `evaluate_record_level_strodthoff.py`: avaliação record-level de um checkpoint.
- `eval_strodthoff.py`: alias de avaliação record-level.
- `build_code15_hdf5_index_strodthoff.py`: indexa shards HDF5 do CODE-15 para carregamento lazy.
- `pretrain_ssl_code15_strodthoff.py`: pretreino SSL em CODE-15 (PPT-style) com loader escalável.
- `summarize_strodthoff_results.py`: consolida runs em CSV.
- `plot_strodthoff_results.py`: gera figuras para artigo.
- `compare_models_strodthoff.py`: ranking rápido por Macro-AUROC.
- `inspect_model_capacity_strodthoff.py`: contagem de parâmetros e config do modelo.
- `bootstrap_strodthoff.py`: bootstrap de Macro-AUROC.

## Rodando (ordem recomendada)

1) Preparar dados:

```bash
python3 physio_ppt/experiments/strodthoff_protocol/scripts/prepare_ptbxl_strodthoff.py \
  --config physio_ppt/experiments/strodthoff_protocol/configs/experiment/xresnet_superdiag_100pct.yaml
```

2) Gerar splits low-label persistidos:

```bash
python3 physio_ppt/experiments/strodthoff_protocol/scripts/make_low_label_splits_strodthoff.py \
  --config physio_ppt/experiments/strodthoff_protocol/configs/experiment/xresnet_superdiag_100pct.yaml \
  --fractions 0.01,0.05,0.10 \
  --seeds 42,1337,2025
```

3) Baseline principal (primeiro 100% e 10%):

```bash
python3 physio_ppt/experiments/strodthoff_protocol/scripts/train_supervised_strodthoff.py \
  --config physio_ppt/experiments/strodthoff_protocol/configs/experiment/xresnet_superdiag_100pct.yaml \
  --seed 42 --device cuda

python3 physio_ppt/experiments/strodthoff_protocol/scripts/train_supervised_strodthoff.py \
  --config physio_ppt/experiments/strodthoff_protocol/configs/experiment/xresnet_superdiag_10pct.yaml \
  --seed 42 --device cuda
```

4) Transformer supervisionado:

```bash
python3 physio_ppt/experiments/strodthoff_protocol/scripts/train_supervised_strodthoff.py \
  --config physio_ppt/experiments/strodthoff_protocol/configs/experiment/transformer_superdiag_100pct.yaml \
  --seed 42 --device cuda
```

5) SSL fine-tuning (ajuste `pretrained.checkpoint` no YAML):

```bash
python3 physio_ppt/experiments/strodthoff_protocol/scripts/finetune_ssl_strodthoff.py \
  --config physio_ppt/experiments/strodthoff_protocol/configs/experiment/physioppt_legacy_transformer_10pct.yaml \
  --seed 42 --device cuda
```

6) Consolidação:

```bash
python3 physio_ppt/experiments/strodthoff_protocol/scripts/summarize_strodthoff_results.py
python3 physio_ppt/experiments/strodthoff_protocol/scripts/plot_strodthoff_results.py
```

## CODE-15 para pretreino (escala)

O pipeline agora suporta dois modos de carregamento para CODE-15:

- `format=hdf5_index`: leitura lazy direto dos shards `exams_part*.hdf5` (mais prático quando só há dataset bruto).
- `format=manifest_npy`: leitura via manifest + `.npy` por amostra (padrão Cardio-JEPA).

Configs prontas:

- `configs/pretrain/code15_ppt_hdf5_fs400.yaml`
- `configs/pretrain/code15_ppt_manifest_fs500.yaml`

### HDF5 indexado (sem preprocess pesado)

```bash
python3 physio_ppt/experiments/strodthoff_protocol/scripts/build_code15_hdf5_index_strodthoff.py \
  --config physio_ppt/experiments/strodthoff_protocol/configs/pretrain/code15_ppt_hdf5_fs400.yaml

python3 physio_ppt/experiments/strodthoff_protocol/scripts/pretrain_ssl_code15_strodthoff.py \
  --config physio_ppt/experiments/strodthoff_protocol/configs/pretrain/code15_ppt_hdf5_fs400.yaml \
  --seed 42 --device cuda
```

### Manifest Cardio-JEPA (quando já houver preprocess em .npy)

```bash
python3 physio_ppt/experiments/strodthoff_protocol/scripts/pretrain_ssl_code15_strodthoff.py \
  --config physio_ppt/experiments/strodthoff_protocol/configs/pretrain/code15_ppt_manifest_fs500.yaml \
  --seed 42 --device cuda
```

Observação:
- Foi criado um link local de referência em `references/Cardio-JEPA -> /media/data/eduardo/Cardio-JEPA`.
- Para modo `hdf5_index`, instale `h5py`.

## Observações metodológicas

- Low-label é aplicado **somente no treino oficial (folds 1-8)**.
- Val/test permanecem completos e intactos.
- A amostragem low-label padrão é `record_random` reprodutível e persistida em JSON.
- Seleção de modelo por `val_macro_auroc`.
- Métricas secundárias: Macro-AUPRC, Macro-F1, Micro-F1, Samples-F1.
- Inferência no teste é em nível de registro (janelas + agregação).
