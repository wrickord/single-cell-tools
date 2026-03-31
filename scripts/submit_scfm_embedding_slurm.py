#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

import env_bootstrap  # noqa: E402

env_bootstrap.load_repo_dotenv()

from slurm_defaults import effective_slurm_partition  # noqa: E402

from background_jobs import embed_output_base, jobs_root  # noqa: E402
from slurm_gpu import (  # noqa: E402
    build_slurm_gpu_embed_script,
    default_repo_root,
    run_sbatch,
    sanitize_dataset_name,
)


def _estimate_resources(input_h5ad: Path, model: str) -> Dict[str, Any]:
    size_gib = max(input_h5ad.stat().st_size / float(1 << 30), 0.1)
    model_key = str(model or "geneformer").strip().lower()
    cpus = {"scvi": 8}.get(model_key, 4)
    mem_gib = math.ceil(
        max(
            8.0,
            size_gib
            * {"geneformer": 3.0, "transcriptformer": 3.0, "scgpt": 2.5, "scvi": 3.5}.get(model_key, 3.0)
            + {"geneformer": 4.0, "transcriptformer": 5.0, "scgpt": 3.0, "scvi": 6.0}.get(model_key, 4.0),
        )
    )
    hours = {"geneformer": 6.0, "transcriptformer": 6.0, "scgpt": 4.0, "scvi": 6.0}.get(model_key, 6.0)
    try:
        import anndata as ad

        adata = ad.read_h5ad(str(input_h5ad), backed="r")
        n_obs = int(adata.n_obs)
        hours += n_obs / {"geneformer": 80_000.0, "transcriptformer": 70_000.0, "scgpt": 120_000.0, "scvi": 180_000.0}.get(model_key, 100_000.0)
        fobj = getattr(adata, "file", None)
        if fobj is not None:
            try:
                fobj.close()
            except Exception:
                pass
    except Exception:
        pass
    total_sec = max(3600, int(math.ceil(min(72.0, max(2.0, hours)) * 3600.0)))
    hh, rem = divmod(total_sec, 3600)
    mm, ss = divmod(rem, 60)
    return {"cpus": cpus, "mem": f"{mem_gib}G", "time": f"{hh:02d}:{mm:02d}:{ss:02d}"}


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Submit an scFM embedding Slurm job from a login/SSH node.")
    ap.add_argument("--input-h5ad", required=True, help="Shared-filesystem .h5ad path visible to the Slurm worker")
    ap.add_argument("--model", required=True, choices=["geneformer", "transcriptformer", "scgpt", "scvi"])
    ap.add_argument("--matrix-spec", default="X", help="X, raw.X, or layer:<name>")
    ap.add_argument("--obsm-key", default=None, help="Optional obsm key override in the output")
    ap.add_argument("--scgpt-ckpt", default=None, help="Checkpoint dir for scGPT (else use SCGPT_CKPT_DIR)")
    ap.add_argument("--n-latent-scvi", type=int, default=64)
    ap.add_argument("--dataset-name", default=None, help="Output folder label (default: input stem)")
    ap.add_argument("--output-h5ad", default=None, help="Explicit output .h5ad path")
    ap.add_argument(
        "--partition",
        default=None,
        help="Slurm -p (default: SCFMS_SLURM_PARTITION from .env or 'gpu')",
    )
    ap.add_argument("--gres", default="gpu:1", help="Slurm --gres value, e.g. gpu:1 or gpu:a100:2")
    ap.add_argument("--cpus", type=int, default=0, help="0 = auto estimate")
    ap.add_argument("--mem", default="auto", help="Slurm --mem, or auto")
    ap.add_argument("--time", dest="time_limit", default="auto", help="Slurm -t, or auto")
    ap.add_argument("--bash-prologue", default="", help="Shell lines run before python on the worker")
    ap.add_argument("--repo-root", default=None, help="Repo root on shared storage (default: this clone)")
    ap.add_argument("--dry-run", action="store_true", help="Write params/script but do not call sbatch")
    args = ap.parse_args(argv)

    input_h5ad = Path(args.input_h5ad).expanduser().resolve()
    if not input_h5ad.is_file():
        raise SystemExit(f"Input file not found: {input_h5ad}")

    rec = _estimate_resources(input_h5ad, args.model)
    cpus = max(1, int(args.cpus)) if int(args.cpus or 0) > 0 else int(rec["cpus"])
    mem = str(args.mem or "").strip()
    if not mem or mem.lower() == "auto":
        mem = str(rec["mem"])
    time_limit = str(args.time_limit or "").strip()
    if not time_limit or time_limit.lower() == "auto":
        time_limit = str(rec["time"])

    repo = Path(args.repo_root).expanduser().resolve() if args.repo_root else default_repo_root()
    dataset_name = sanitize_dataset_name(args.dataset_name or input_h5ad.stem)
    stage_dir = jobs_root() / f"cli_submit_{uuid.uuid4().hex[:12]}"
    stage_dir.mkdir(parents=True, exist_ok=True)

    if args.output_h5ad:
        output_h5ad = Path(args.output_h5ad).expanduser().resolve()
    else:
        out_dir = embed_output_base() / dataset_name
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        output_h5ad = out_dir / f"{input_h5ad.stem}__{args.model}__{stamp}.h5ad"

    params = {
        "repo_root": str(repo),
        "input_h5ad": str(input_h5ad),
        "output_h5ad": str(output_h5ad),
        "model": str(args.model),
        "matrix_spec": str(args.matrix_spec),
        "obsm_key": args.obsm_key,
        "scgpt_ckpt": args.scgpt_ckpt,
        "n_latent_scvi": int(args.n_latent_scvi),
    }
    params_json = stage_dir / "embed_params.json"
    _write_json(params_json, params)

    script = build_slurm_gpu_embed_script(
        repo_root=repo,
        stage_dir=stage_dir,
        params_json=params_json,
        partition=effective_slurm_partition(args.partition),
        gres=str(args.gres or "gpu:1").strip() or "gpu:1",
        cpus=cpus,
        mem=mem,
        time_limit=time_limit,
        job_name=f"scfms-{args.model}-{stage_dir.name[-6:]}",
        bash_prologue=str(args.bash_prologue or ""),
    )
    script_path = stage_dir / "slurm_gpu_embed.sh"
    script_path.write_text(script, encoding="utf-8")
    script_path.chmod(0o755)

    print(f"stage_dir={stage_dir}")
    print(f"params_json={params_json}")
    print(f"batch_script={script_path}")
    print(f"output_h5ad={output_h5ad}")
    print(f"resolved_request=-c {cpus} --mem={mem} -t {time_limit}")

    if args.dry_run:
        print("dry_run=1")
        return 0

    job_id = run_sbatch(script_path)
    print(f"slurm_job_id={job_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
