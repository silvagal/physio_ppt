# Physio-PPT: SSL Fisiologicamente Ancorado para ECG Multi-Lead em Low-Label

## Pitch
O `physio_ppt` é um subprojeto isolado para investigar SSL em ECG com um viés explícito de fisiologia cardíaca. Em vez de tratar a ordem temporal apenas como um artefato de patches genéricos, o método Physio-PPT respeita a estrutura P/QRS/T e o alinhamento ao R-peak para construir views e perturbações mais clinicamente plausíveis.

A proposta compara, em protocolo único e reproduzível, quatro braços principais: supervisionado forte, PPT clássico, ECGWavePuzzle e Physio-PPT. O foco é regime low-label (1%, 5%, 10%), estatística com bootstrap pareado e análise de orderness (ACF-COS + Beat-Order Score), com rastreabilidade total de config, seed, checkpoint e métricas.

## Estrutura

```text
physio_ppt/
  configs/
  src/physio_ppt/
  scripts/
  paper_notes/
  figures/
  outputs/
```

## Instalação

Do diretório raiz do repositório:

```bash
pip install -e physio_ppt
```

Opcional (dependências extras para delineation):

```bash
pip install -e "physio_ppt[extras]"
```

Para desenvolvimento/testes:

```bash
pip install -e "physio_ppt[dev]"
```

## CLI única

Comando padrão:

```bash
python3 -m physio_ppt.cli run --config <yaml> --seed <int> --device <cpu|cuda>
```

Subcomandos disponíveis:

- `prepare_data`
- `pretrain`
- `finetune`
- `eval`
- `analyze`
- `make_figures`
- `run`

## Preparação de dados

### Usando artefatos já processados (default)
Os YAMLs já apontam para:

- `ecg-patch-order-ssl/data/processed/mitbih/fs500/...`
- `ecg-patch-order-ssl/data/processed/ptbxl/fs500/superclasses`

Nesse caso, não é necessário reprocesar.

### Reprocessar do zero

```bash
bash physio_ppt/scripts/00_prepare_data.sh physio_ppt/configs/train/physio_ppt.yaml 42 cpu
```

Se quiser forçar preprocessamento no YAML, use overrides:

```bash
python3 -m physio_ppt.cli prepare_data \
  --config physio_ppt/configs/train/physio_ppt.yaml \
  --seed 42 --device cpu \
  --override data.prepare_mitbih=true \
  --override data.prepare_ptbxl=true
```

## Experimentos do paper

Seeds oficiais:

```text
[42, 1337, 2025, 7, 123]
```

### E1: MIT-BIH pretrain + PTB-XL finetune low-label (1/5/10)

```bash
python3 -m physio_ppt.cli run --config physio_ppt/configs/train/physio_ppt.yaml --seed 42 --device cuda
```

### E2: Ablações Physio-PPT

Somente consistency:

```bash
python3 -m physio_ppt.cli run \
  --config physio_ppt/configs/train/physio_ppt.yaml --seed 42 --device cuda \
  --override ssl.use_consistency=true --override ssl.use_contrastive=false
```

Somente contrastive:

```bash
python3 -m physio_ppt.cli run \
  --config physio_ppt/configs/train/physio_ppt.yaml --seed 42 --device cuda \
  --override ssl.use_consistency=false --override ssl.use_contrastive=true
```

Ambos + sem inter-segment:

```bash
python3 -m physio_ppt.cli run \
  --config physio_ppt/configs/train/physio_ppt.yaml --seed 42 --device cuda \
  --override ssl.perturb_mode=intra
```

### E3: Comparação 4-way em 10% labels

Supervisionado forte:

```bash
python3 -m physio_ppt.cli run \
  --config physio_ppt/configs/train/supervised_strong.yaml --seed 42 --device cuda \
  --override pipeline.label_fractions=0.1 --override pipeline.do_pretrain=false
```

PPT clássico:

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

Hybrid (opcional):

```bash
python3 -m physio_ppt.cli run --config physio_ppt/configs/train/hybrid_multitask.yaml --seed 42 --device cuda
```

### E4: Análise de orderness (ACF-COS + Beat-Order Score)

```bash
bash physio_ppt/scripts/04_orderness_analysis.sh
```

### E5 (opcional): robustez a ruído/drift

Use overrides para ruído (exemplo simples):

```bash
python3 -m physio_ppt.cli run \
  --config physio_ppt/configs/train/physio_ppt.yaml --seed 42 --device cuda \
  --override ssl.noise_sigma=0.02
```

## Significância estatística (bootstrap pareado)

```bash
bash physio_ppt/scripts/03_eval_significance.sh \
  physio_ppt/configs/eval/significance_bootstrap.yaml \
  <pred_method_a.npz> <pred_method_b.npz> 42 cpu
```

## Geração de tabelas e figuras

Agregação low-label:

```bash
python3 -m physio_ppt.cli eval --config physio_ppt/configs/eval/low_label_1_5_10.yaml --seed 42 --device cpu
```

Figuras principais:

```bash
bash physio_ppt/scripts/05_make_figures.sh
```

Artefatos esperados:

- Tabelas: `physio_ppt/outputs/tables/`
- Figuras: `physio_ppt/figures/`

## Runner completo (mínimo paper)

```bash
PARALLEL=1 DEVICE=cuda bash physio_ppt/scripts/run_all.sh
```

Com paralelismo:

```bash
PARALLEL=4 DEVICE=cuda bash physio_ppt/scripts/run_all.sh
```

## Reprodutibilidade

Checklist:

- Seeds fixas: `[42, 1337, 2025, 7, 123]`
- `config_used.yaml` salvo em cada run
- Checkpoint com hash de config no nome da run
- Logs estruturados em `events.jsonl`
- Métricas por época em `metrics.csv`
- Métricas finais em `test_metrics.csv`
- Predições de teste para bootstrap em `test_predictions.npz`

Hardware:

- CPU: suportado (mais lento)
- GPU: recomendado para matriz completa

## Threats to Validity (curto)

- MIT-BIH possui 2 leads; para transferência com PTB-XL o default usa subconjunto de leads para comparabilidade de canal. Mitigação: reportar explicitamente esse desenho e rodar análise adicional com adaptações 12-lead.
- Delineation P/QRS/T depende de detector e fallback. Mitigação: taxa de fallback é logada e o método fixo por offsets mantém robustez operacional.
- Diferenças de distribuição entre datasets podem inflar/atenuar ganhos SSL. Mitigação: seeds múltiplas, bootstrap pareado por exame e protocolo estrito sem leakage.

## Smoke tests

```bash
cd physio_ppt
pytest -q src/physio_ppt/tests
```
