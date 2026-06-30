"""High-level experiment pipeline orchestration."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, List

import pandas as pd

from ..utils.io import ensure_dir
from ..utils.logger import build_logger
from .train_finetune import run_finetune
from .train_pretrain import run_pretrain


def _merge(dst: Dict[str, object], patch: Dict[str, object]) -> Dict[str, object]:
    out = copy.deepcopy(dst)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)  # type: ignore[index]
        else:
            out[k] = copy.deepcopy(v)
    return out


def run_pipeline(config: Dict[str, object], seed: int, device: str) -> Dict[str, object]:
    """Run full or partial pipeline defined in config."""
    logger = build_logger()
    cfg = copy.deepcopy(config)

    pipe = cfg.get("pipeline", {})
    do_pretrain = bool(pipe.get("do_pretrain", True))
    do_finetune = bool(pipe.get("do_finetune", True))

    pretrain_result = None
    if do_pretrain:
        logger.info("running pretrain seed=%d", seed)
        pretrain_result = run_pretrain(cfg, seed=seed, device=device)

    finetune_results: List[Dict[str, object]] = []
    if do_finetune:
        fractions = pipe.get("label_fractions", [float(cfg["train"].get("label_fraction", 1.0))])
        if not isinstance(fractions, (list, tuple)):
            fractions = [fractions]
        for frac in fractions:
            frac = float(frac)
            ft_cfg = copy.deepcopy(cfg)
            ft_cfg["train"]["label_fraction"] = frac
            ft_cfg["experiment"]["name"] = f"{cfg['experiment']['name']}_frac{int(frac*100):02d}"
            if pretrain_result is not None:
                ft_cfg["train"]["pretrained_checkpoint"] = pretrain_result["checkpoint"]
            logger.info("running finetune seed=%d frac=%.2f", seed, frac)
            finetune_results.append(run_finetune(ft_cfg, seed=seed, device=device))

    out_root = ensure_dir(str(cfg["paths"]["output_root"]))
    summary_dir = ensure_dir(Path(out_root) / "tables")

    if finetune_results:
        df = pd.DataFrame(finetune_results)
        df.to_csv(summary_dir / f"pipeline_seed{seed}.csv", index=False)

    return {
        "seed": seed,
        "pretrain": pretrain_result,
        "finetune": finetune_results,
        "summary_csv": str(summary_dir / f"pipeline_seed{seed}.csv"),
    }


__all__ = ["run_pipeline"]
