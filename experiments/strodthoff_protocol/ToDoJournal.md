# ToDoJournal — Extensão para Artigo de Revista (Strodthoff-Compatible + Low-Label SSL)

## 0) Pergunta científica central

**Q\***: Em regime low-label no PTB-XL superdiagnostic, sob protocolo downstream compatível com Strodthoff (12 leads, folds oficiais, inferência record-level, Macro-AUROC), a inicialização SSL fisiologicamente ancorada melhora desempenho e estabilidade em relação a baselines supervisionados fortes?

---

## 1) Regras fixas de protocolo (não negociáveis)

- Tarefa: PTB-XL superdiagnostic (`NORM`, `MI`, `STTC`, `CD`, `HYP`) multilabel.
- Splits: folds oficiais (`train=1..8`, `val=9`, `test=10`).
- Low-label: subamostragem **somente em train** (1%, 5%, 10%), com split persistido por seed/fração.
- Leads downstream: 12.
- Inferência: nível de registro com sliding windows + agregação.
- Janela default: 2.5 s (configurável).
- Métrica primária: Macro-AUROC.
- Métricas secundárias: Macro-AUPRC, Macro-F1, Micro-F1, Samples-F1.
- Seleção de modelo: melhor `val_macro_auroc`.
- Comparações justas: mesmos splits low-label para todos os métodos na mesma seed/fração.

---

## 2) Research Questions (RQs) para a fase revista

### RQ0 — Sanidade do novo protocolo
- O pipeline Strodthoff-compatible está funcional e reproduzível?
- O fluxo salva artefatos completos para auditoria?

### RQ1 — Ganho de SSL sob low-label
- Physio-PPT (pré-treinado) supera baselines supervisionados em 10/5/1%?
- O ganho aumenta quando a fração de rótulos cai?

### RQ2 — Papel da arquitetura no downstream
- xresnet1d101 e Transformer convergem para conclusões consistentes?
- Há interação arquitetura × fração low-label?

### RQ3 — Efeito da fonte de pretreino (escala)
- Pretreino em CODE-15 (grande escala) melhora transferência para PTB-XL low-label?
- CODE-15 vs pré-treino legado (MIT-BIH / outros) altera robustez e estabilidade?

### RQ4 — Efeito do objetivo SSL
- Physio-PPT vs PPT-classic vs ECGWavePuzzle no novo protocolo.
- Diferenças permanecem quando a avaliação é estritamente record-level?

### RQ5 — Estabilidade e significância estatística
- Ganhos se mantêm em múltiplas seeds?
- Diferenças-chave são estatisticamente defensáveis (bootstrap/CI)?

### RQ6 — Sensibilidade de protocolo
- Como variações de janela/overlap/agregação afetam ranking e magnitude dos ganhos?

---

## 3) Pré-requisitos operacionais

### Ambiente
- [ ] `wfdb` instalado (preparo PTB-XL).
- [ ] `scikit-learn` instalado (métricas).
- [ ] `h5py` instalado para fluxo CODE-15 em HDF5 indexado.

### Dados
- [ ] PTB-XL bruto disponível em `ecg-patch-order-ssl/data/raw/ptbxl`.
- [ ] CODE-15 bruto disponível em `ecg-patch-order-ssl/data/raw/code15` (shards `.hdf5`).
- [ ] (Opcional) CODE-15 preprocessado estilo Cardio-JEPA (manifest + `.npy`).

### GPU/IO
- [ ] Definir política de cache e workers para pretreino CODE-15.
- [ ] Monitorar throughput de dataloader (CPU/IO bound vs GPU bound).

---

## 4) Sequência de execução (ordem recomendada)

## Fase A — Congelar protocolo e split
- [ ] A1. Preparar PTB-XL oficial:
```bash
python3 physio_ppt/experiments/strodthoff_protocol/scripts/prepare_ptbxl_strodthoff.py \
  --config physio_ppt/experiments/strodthoff_protocol/configs/experiment/xresnet_superdiag_100pct.yaml
```
- [ ] A2. Gerar splits low-label persistidos (seeds e frações):
```bash
python3 physio_ppt/experiments/strodthoff_protocol/scripts/make_strodthoff_splits.py \
  --config physio_ppt/experiments/strodthoff_protocol/configs/experiment/xresnet_superdiag_100pct.yaml \
  --fractions 0.01,0.05,0.10 \
  --seeds 42,1337,2025,7,123
```
- [ ] A3. Verificar manifesto de splits (`splits_manifest.csv`).

## Fase B — RQ0 (sanidade) + RQ2 (arquitetura em 100%)
- [ ] B1. xresnet1d101 100% seed=42.
- [ ] B2. transformer 100% seed=42.
- [ ] B3. Validar artefatos: `config_resolved.yaml`, `events.jsonl`, `epoch_metrics.csv`, `test_metrics.json`, `predictions`.

## Fase C — RQ1/RQ2 (low-label supervisionado)
- [ ] C1. xresnet: 10%, 5%, 1% (5 seeds).
- [ ] C2. transformer: 10%, 5%, 1% (5 seeds).
- [ ] C3. Consolidar tabelas/figuras e checar variância por seed.

## Fase D — RQ3 (pretreino em CODE-15 em escala)
- [ ] D1. Gerar índice HDF5 lazy:
```bash
python3 physio_ppt/experiments/strodthoff_protocol/scripts/build_code15_hdf5_index_strodthoff.py \
  --config physio_ppt/experiments/strodthoff_protocol/configs/pretrain/code15_ppt_hdf5_fs400.yaml
```
- [ ] D2. Rodar pretreino CODE-15 (PPT-style):
```bash
python3 physio_ppt/experiments/strodthoff_protocol/scripts/pretrain_ssl_code15_strodthoff.py \
  --config physio_ppt/experiments/strodthoff_protocol/configs/pretrain/code15_ppt_hdf5_fs400.yaml \
  --seed 42 --device cuda
```
- [ ] D3. Repetir com seeds adicionais prioritárias (42, 1337, 2025).
- [ ] D4. Catalogar checkpoints candidatos para downstream.

## Fase E — RQ1/RQ3 (SSL low-label com Physio-PPT)
- [ ] E1. Fine-tune 10% com checkpoint SSL escolhido.
- [ ] E2. Fine-tune 5% e 1%.
- [ ] E3. Rodar 5 seeds por fração.
- [ ] E4. Repetir para variações de fonte de pretreino (CODE-15 vs legado), mantendo todo o resto fixo.

## Fase F — RQ4 (comparação de objetivos SSL)
- [ ] F1. Adaptar/checkpoints equivalentes para PPT-classic e ECGWavePuzzle no novo protocolo.
- [ ] F2. Rodar 10/5/1% com mesma arquitetura downstream e mesmas seeds/splits.
- [ ] F3. Consolidar deltas absolutos e relativos vs supervisionado.

## Fase G — RQ5 (estatística)
- [ ] G1. Bootstrap de Macro-AUROC no teste para comparações-chave.
```bash
python3 physio_ppt/experiments/strodthoff_protocol/scripts/bootstrap_strodthoff.py \
  --predictions_npz <run>/test_predictions_record_level.npz \
  --n_bootstrap 10000 --seed 42 --output_json <run>/bootstrap_macro_auroc.json
```
- [ ] G2. Reportar média, desvio, CI95 e tamanho de efeito por fração.

## Fase H — RQ6 (sensibilidade de protocolo)
- [ ] H1. Janela: 2.0s vs 2.5s vs 3.0s.
- [ ] H2. Overlap: 0.25 vs 0.5 vs 0.75.
- [ ] H3. Agregação: `mean` vs `max`.
- [ ] H4. Medir estabilidade de ranking de métodos sob variação de protocolo.

## Fase I — Síntese para manuscrito
- [ ] I1. Gerar tabelas finais:
```bash
python3 physio_ppt/experiments/strodthoff_protocol/scripts/summarize_strodthoff_results.py
```
- [ ] I2. Gerar figuras finais:
```bash
python3 physio_ppt/experiments/strodthoff_protocol/scripts/plot_strodthoff_results.py
```
- [ ] I3. Tabela “paper-ready” e notas de threat-to-validity.

---

## 5) Matriz mínima de experimentos (para resposta completa das RQs)

### Bloco M1 — Supervisão pura (RQ0/RQ2)
- `xresnet` 100%, 10%, 5%, 1% × 5 seeds.
- `transformer` 100%, 10%, 5%, 1% × 5 seeds.

### Bloco M2 — SSL principal (RQ1/RQ3)
- `Physio-PPT(pretrained) -> downstream` em 10%, 5%, 1% × 5 seeds.
- Fontes de pretreino:
- `legacy_pretrain` (baseline histórico)
- `code15_pretrain` (nova escala)

### Bloco M3 — SSL comparativo (RQ4)
- `PPT-classic(pretrained) -> downstream` em 10%, 5%, 1% × 5 seeds.
- `ECGWavePuzzle(pretrained) -> downstream` em 10%, 5%, 1% × 5 seeds.

### Bloco M4 — Estatística e robustez (RQ5/RQ6)
- Bootstrap CI para comparações-chave de Macro-AUROC.
- Sensibilidade janela/overlap/agregação em subset representativo.

---

## 6) Checklist de fairness e reprodutibilidade

- [ ] Mesmo arquivo de split low-label por seed/fração para todos os métodos.
- [ ] Mesmo conjunto de métricas e mesma métrica primária.
- [ ] Mesmo protocolo de inferência record-level.
- [ ] Seleção por `val_macro_auroc` para todos.
- [ ] Config resolvida salva por run.
- [ ] Log completo por run (`seed`, `fraction`, `model`, `checkpoint`, `best_epoch`).
- [ ] Predições por registro salvas para auditoria e bootstrap.

---

## 7) Riscos e mitigação

- Risco: gargalo de IO no CODE-15.
- Mitigação: usar `hdf5_index` lazy, workers adequados e monitorar throughput.

- Risco: incompatibilidade de shape ao carregar checkpoints SSL.
- Mitigação: carregar por compatibilidade de shape e registrar relatório de chaves carregadas.

- Risco: variação alta em 1% por classes raras.
- Mitigação: 5 seeds mínimas + bootstrap CI + análise por classe.

- Risco: custo computacional elevado.
- Mitigação: gate por fases (sanidade -> low-label supervisionado -> SSL).

---

## 8) Critérios de pronto para submissão de revista

- [ ] Todas as RQs respondidas com evidência quantitativa.
- [ ] Tabelas principais com Macro-AUROC (média ± desvio/CI).
- [ ] Figuras de curva por fração + estabilidade por seed.
- [ ] Comparações SSL vs supervisionado em protocolo estritamente compatível.
- [ ] Threats-to-validity explicitados (sampling rate, arquitetura, avaliação record-level).
- [ ] Scripts reproduzíveis e documentação operacional completa.

