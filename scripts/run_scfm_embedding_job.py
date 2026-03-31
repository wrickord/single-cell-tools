#!/usr/bin/env python3
"""Run scFM embedding on a Slurm GPU allocation (invoked by the batch script).

Reads parameters from a JSON file (paths, model, matrix layer, etc.), writes a new
``.h5ad`` with embeddings in ``obsm``.

Example::

    python scripts/run_scfm_embedding_job.py --params-json /path/to/embed_params.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--params-json",
        required=True,
        help="JSON with keys: repo_root, input_h5ad, output_h5ad, model, matrix_spec, "
        "obsm_key, scgpt_ckpt, n_latent_scvi",
    )
    args = ap.parse_args()
    params_path = Path(args.params_json).resolve()
    with open(params_path, encoding="utf-8") as f:
        p = json.load(f)

    repo = Path(p.get("repo_root") or "").resolve()
    if not repo.is_dir() or not (repo / "app" / "preprocess.py").is_file():
        repo = Path(__file__).resolve().parent.parent

    in_path = Path(p["input_h5ad"]).resolve()
    out_path = Path(p["output_h5ad"]).resolve()
    model = str(p["model"])
    matrix_spec = str(p.get("matrix_spec") or "X")
    obsm_key = p.get("obsm_key")
    scgpt_ckpt = p.get("scgpt_ckpt")
    n_latent = int(p.get("n_latent_scvi") or 64)

    app_dir = repo / "app"
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    if str(app_dir) not in sys.path:
        sys.path.insert(0, str(app_dir))

    import importlib.util

    spec = importlib.util.spec_from_file_location("scfms_embed_pre", app_dir / "preprocess.py")
    if spec is None or spec.loader is None:
        raise SystemExit(f"Cannot load preprocess from {app_dir / 'preprocess.py'}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    import anndata as ad

    print(f"Reading {in_path}", flush=True)
    adata = ad.read_h5ad(in_path)
    print(f"Embedding model={model} matrix={matrix_spec} cells={adata.n_obs}", flush=True)
    out, msg, timings = mod.attach_scfm_embedding(
        adata,
        model=model,
        matrix_spec=matrix_spec,
        obsm_key=obsm_key,
        scgpt_ckpt=scgpt_ckpt,
        n_latent_scvi=n_latent,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing {out_path}", flush=True)
    out.write_h5ad(out_path)
    print(msg, flush=True)
    print("timings:", timings, flush=True)


if __name__ == "__main__":
    main()
