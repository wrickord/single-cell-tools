#!/usr/bin/env python3
"""Slurm worker: train (and optionally evaluate) embedding benchmark with Weights & Biases."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--params-json", required=True, help="Path to benchmark_slurm_params.json")
    args = ap.parse_args()
    pj = Path(args.params_json).resolve()
    cfg = json.loads(pj.read_text(encoding="utf-8"))

    repo = Path(cfg["repo_root"]).resolve()
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "app"))

    complete_path = Path(cfg["complete_json"]).resolve()
    wandb_mod = None
    try:
        import wandb as _wandb

        wandb_mod = _wandb
    except ImportError:
        pass

    def write_complete(ok: bool, **kw: object) -> None:
        data: dict = {"ok": ok}
        for k, v in kw.items():
            if v is not None:
                data[k] = v
        complete_path.parent.mkdir(parents=True, exist_ok=True)
        complete_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    run_url: str | None = None
    proj_url: str | None = None
    step = [0]

    try:
        import anndata as ad

        import benchmark as bench

        adata = ad.read_h5ad(cfg["input_h5ad"])
        src_key = ("path", str(Path(cfg["input_h5ad"]).resolve()))

        wproj = (cfg.get("wandb_project") or "scfms-benchmark").strip() or "scfms-benchmark"
        went = (cfg.get("wandb_entity") or "").strip() or None
        wname = (cfg.get("wandb_run_name") or "").strip() or None

        if wandb_mod is not None:
            w_init: dict = {
                "project": wproj,
                "config": {
                    "classifier_kind": cfg.get("classifier_kind"),
                    "test_fraction": cfg.get("test_fraction"),
                    "random_seed": cfg.get("random_seed"),
                    "max_cells": cfg.get("max_cells"),
                    "mlp_hidden": cfg.get("mlp_hidden"),
                    "split_mode": cfg.get("split_mode"),
                },
            }
            if went:
                w_init["entity"] = went
            if wname:
                w_init["name"] = wname
            wandb_mod.init(**w_init)
            if wandb_mod.run:
                run_url = wandb_mod.run.url
                ent = getattr(wandb_mod.run, "entity", None) or went
                proj = getattr(wandb_mod.run, "project", None) or wproj
                if ent and proj:
                    proj_url = f"https://wandb.ai/{ent}/{proj}"

        def per_model_cb(
            _idx: int,
            spec: str,
            tgt: str,
            mval: dict,
            _entry: dict,
        ) -> None:
            if wandb_mod is None or not wandb_mod.run:
                return
            payload = {
                "val/accuracy": mval["accuracy"],
                "val/balanced_accuracy": mval.get("balanced_accuracy"),
                "val/f1_macro": mval["f1_macro"],
                "val/log_loss": mval.get("log_loss"),
                "model/spec": spec,
                "model/target": tgt,
            }
            wandb_mod.log({k: v for k, v in payload.items() if v is not None}, step=step[0])
            step[0] += 1

        out = bench.train_benchmark_core(
            adata,
            src_key,
            None,
            target_cols=list(cfg.get("target_cols") or []),
            split_mode=str(cfg.get("split_mode") or "random"),
            stratify_col=str(cfg.get("stratify_col") or "(none)"),
            test_fraction=float(cfg.get("test_fraction") or 0.2),
            random_seed=int(cfg.get("random_seed") or 0),
            classifier_kind=str(cfg.get("classifier_kind") or "logistic_regression"),
            mlp_hidden=str(cfg.get("mlp_hidden") or "128,64"),
            mlp_max_iter=float(cfg.get("mlp_max_iter") or 200),
            lr_c=float(cfg.get("lr_c") or 1.0),
            lr_max_iter=float(cfg.get("lr_max_iter") or 2000),
            max_cells=float(cfg.get("max_cells") or 0),
            skip_sources=list(cfg.get("skip_sources") or []),
            per_model_callback=per_model_cb if wandb_mod else None,
        )

        if wandb_mod and wandb_mod.run:
            if out.fig_hm is not None:
                wandb_mod.log({"plots/val_accuracy_heatmap": wandb_mod.Image(out.fig_hm)})
            if out.fig_bar is not None:
                wandb_mod.log({"plots/val_accuracy_by_source": wandb_mod.Image(out.fig_bar)})

        test_path = (cfg.get("test_h5ad_path") or "").strip()
        if test_path:
            te = Path(test_path).expanduser().resolve()
            if te.is_file():
                ad_te = ad.read_h5ad(str(te))
                sk2 = ("path", str(te))
                test_step = [step[0]]

                def row_cb(row: dict) -> None:
                    if wandb_mod is None or not wandb_mod.run:
                        return
                    acc = row.get("accuracy")
                    if acc is None or (isinstance(acc, float) and acc != acc):
                        return
                    wandb_mod.log(
                        {
                            "test/accuracy": acc,
                            "test/f1_macro": row.get("f1_macro"),
                            "test/source": row.get("source"),
                            "test/target": row.get("target"),
                        },
                        step=test_step[0],
                    )
                    test_step[0] += 1

                ev = bench.eval_benchmark_core(
                    ad_te,
                    sk2,
                    None,
                    out.session_dir,
                    per_row_callback=row_cb,
                )
                if wandb_mod and wandb_mod.run:
                    if ev.fig_hm is not None:
                        wandb_mod.log({"plots/test_accuracy_heatmap": wandb_mod.Image(ev.fig_hm)})
                    if ev.fig_bar is not None:
                        wandb_mod.log(
                            {"plots/test_accuracy_by_source": wandb_mod.Image(ev.fig_bar)}
                        )

        if wandb_mod and wandb_mod.run:
            wandb_mod.finish()

        write_complete(
            True,
            session_dir=out.session_dir,
            wandb_url=run_url,
            wandb_project_url=proj_url,
            n_models_trained=len(out.entries),
        )
    except Exception as e:
        if wandb_mod and wandb_mod.run:
            try:
                wandb_mod.finish(exit_code=1)
            except Exception:
                pass
        write_complete(
            False,
            error=str(e),
            traceback=traceback.format_exc(),
            wandb_url=run_url,
            wandb_project_url=proj_url,
        )
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
