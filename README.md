# Physio-PPT: Physiology-Anchored SSL for Multi-Lead ECG in Low-Label Regimes

Cardiovascular diseases remain a leading cause of morbidity and mortality worldwide, and the electrocardiogram (ECG) is a central, low-cost tool for screening, diagnosis, and monitoring. This repository introduces **Physio-PPT**, a physiology-anchored patch-order SSL method for ECG.

Our goal is to improve ECG classification under low-label regimes, a setting that is increasingly common outside large curated benchmarks: labels are expensive, institution-dependent, and often unavailable at scale, whereas unlabeled ECG streams from wearables and bedside monitoring are abundant.

Physio-PPT retains the Patch-Order Pretext Task (PPT) training recipe but constrains permutations to remain within physiologically meaningful regions (P, QRS, and T) around R-peaks. This ensures that points are shuffled only within each physiological segment, maintaining the global physiological structure of the heartbeat while still introducing a local order disruption sufficient to define a meaningful self-supervised pretext task.

## Structure

```text
physio_ppt/
  configs/
  src/physio_ppt/
  scripts/
  paper_notes/
  figures/
  outputs/
```

## Installation

From the repository root directory:

```bash
pip install -e physio_ppt
```

Optional (extra dependencies for delineation):

```bash
pip install -e "physio_ppt[extras]"
```

For development/testing:

```bash
pip install -e "physio_ppt[dev]"
```

## CLI Commands

Default command:

```bash
python3 -m physio_ppt.cli run --config <yaml> --seed <int> --device <cpu|cuda>
```

Available subcommands:

- `prepare_data`
- `pretrain`
- `finetune`
- `eval`
- `analyze`
- `make_figures`
- `run`

## Data Preparation

### Using pre-processed artifacts (default)
The YAMLs already point to:

- `ecg-patch-order-ssl/data/processed/mitbih/fs500/...`
- `ecg-patch-order-ssl/data/processed/ptbxl/fs500/superclasses`

In this case, no reprocessing is needed.

### Reprocess from scratch

```bash
bash physio_ppt/scripts/00_prepare_data.sh physio_ppt/configs/train/physio_ppt.yaml 42 cpu
```

## Experiments

Official seeds: `[42, 1337, 2025, 7, 123]`

The experiments evaluate four main arms: strong supervised, classic PPT, ECGWavePuzzle, and Physio-PPT. The focus is on the low-label regime (1%, 5%, 10%), paired bootstrap statistics, and orderness analysis.

### E1: Low-label transfer
**Description:** Evaluates the models on a controlled low-label transfer setting to isolate the effect of order-aware SSL pretexts. The models are pretrained on MIT-BIH (labels ignored) and fine-tuned on PTB-XL superclasses with 1%, 5%, and 10% label fractions under leakage-safe patient-wise evaluation.

```bash
python3 -m physio_ppt.cli run --config physio_ppt/configs/train/physio_ppt.yaml --seed 42 --device cuda
```

### E2: Ablation study (objective vs. perturbation scope)
**Description:** Isolates the mechanisms behind Physio-PPT by ablating the objective components (consistency vs contrastive) and the perturbation scope (intra-segment vs inter-segment). This validates that combining consistency and contrastive learning yields the strongest transfer and that preserving the intra-segment physiological structure is beneficial.

Consistency only:
```bash
python3 -m physio_ppt.cli run \
  --config physio_ppt/configs/train/physio_ppt.yaml --seed 42 --device cuda \
  --override ssl.use_consistency=true --override ssl.use_contrastive=false
```

Contrastive only:
```bash
python3 -m physio_ppt.cli run \
  --config physio_ppt/configs/train/physio_ppt.yaml --seed 42 --device cuda \
  --override ssl.use_consistency=false --override ssl.use_contrastive=true
```

Both + without inter-segment (Physio-PPT Intra-only):
```bash
python3 -m physio_ppt.cli run \
  --config physio_ppt/configs/train/physio_ppt.yaml --seed 42 --device cuda \
  --override ssl.perturb_mode=intra
```

### E3: Reliability and seed-dependent collapse
**Description:** Beyond average accuracy, this experiment analyzes reliability across multiple random seeds. It demonstrates the stability of Physio-PPT and ECGWavePuzzle across seeds in a 4-way comparison at 10% labels, and highlights the seed-dependent collapse that can occur with classic PPT when utilizing physiology-agnostic global patch permutations.

Strong supervised:
```bash
python3 -m physio_ppt.cli run \
  --config physio_ppt/configs/train/supervised_strong.yaml --seed 42 --device cuda \
  --override pipeline.label_fractions=0.1 --override pipeline.do_pretrain=false
```

Classic PPT:
```bash
python3 -m physio_ppt.cli run \
  --config physio_ppt/configs/train/ppt_classic.yaml --seed 42 --device cuda \
  --override pipeline.label_fractions=0.1
```

WavePuzzle:
```bash
python3 -m physio_ppt.cli run \
  --config physio_ppt/configs/train/wavepuzzle.yaml --seed 42 --device cuda \
  --override pipeline.label_fractions=0.1
```

Physio-PPT:
```bash
python3 -m physio_ppt.cli run \
  --config physio_ppt/configs/train/physio_ppt.yaml --seed 42 --device cuda \
  --override pipeline.label_fractions=0.1
```

### E4: Orderness analysis (ACF-COS + Beat-Order Score)
```bash
bash physio_ppt/scripts/04_orderness_analysis.sh
```

## Statistical significance (paired bootstrap)

```bash
bash physio_ppt/scripts/03_eval_significance.sh \
  physio_ppt/configs/eval/significance_bootstrap.yaml \
  <pred_method_a.npz> <pred_method_b.npz> 42 cpu
```

## Table and Figure Generation

Low-label aggregation:
```bash
python3 -m physio_ppt.cli eval --config physio_ppt/configs/eval/low_label_1_5_10.yaml --seed 42 --device cpu
```

Main figures:
```bash
bash physio_ppt/scripts/05_make_figures.sh
```

Expected artifacts:
- Tables: `physio_ppt/outputs/tables/`
- Figures: `physio_ppt/figures/`

## Full runner (minimum for paper)

```bash
PARALLEL=1 DEVICE=cuda bash physio_ppt/scripts/run_all.sh
```

With parallelism:
```bash
PARALLEL=4 DEVICE=cuda bash physio_ppt/scripts/run_all.sh
```

## Reproducibility

Checklist:
- Fixed seeds: `[42, 1337, 2025, 7, 123]`
- `config_used.yaml` saved in each run
- Checkpoint with config hash in the run name
- Structured logs in `events.jsonl`
- Metrics per epoch in `metrics.csv`
- Final metrics in `test_metrics.csv`
- Test predictions for bootstrap in `test_predictions.npz`

Hardware:
- CPU: supported (slower)
- GPU: recommended for full matrix

## Threats to Validity (short)
- MIT-BIH has 2 leads; for transfer with PTB-XL the default uses a subset of leads for channel comparability. Mitigation: explicitly report this design and run additional analysis with 12-lead adaptations.
- P/QRS/T delineation depends on a detector and fallback. Mitigation: fallback rate is logged and the fixed offset method maintains operational robustness.
- Distribution differences between datasets can inflate/attenuate SSL gains. Mitigation: multiple seeds, paired bootstrap by exam and strict protocol without leakage.

## Smoke tests

```bash
cd physio_ppt
pytest -q src/physio_ppt/tests
```