"""Persistent background jobs for long-running preprocess / embedding work.

Jobs survive browser disconnect: state is written under ``SCFMS_JOB_DIR`` (default:
``<repo>/scfms_job_store``). Re-open the UI and poll or load by job ID.

Optional env ``SCFMS_ALLOWED_PATH_PREFIXES``: pipe ``|``-separated absolute path
prefixes; server-path loads must resolve under one of them (empty = no restriction).
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from slurm_defaults import effective_slurm_partition

_JOB_ENV = os.environ.get("SCFMS_JOB_DIR", "").strip()


def jobs_root() -> Path:
    if _JOB_ENV:
        root = Path(os.path.expanduser(_JOB_ENV)).resolve()
    else:
        root = Path(__file__).resolve().parent.parent / "scfms_job_store"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _path_allowed_by_prefixes(p: Path) -> None:
    """Raise ``PermissionError`` if ``SCFMS_ALLOWED_PATH_PREFIXES`` is set and *p* is outside it."""
    allow = os.environ.get("SCFMS_ALLOWED_PATH_PREFIXES", "").strip()
    if not allow:
        return
    prefixes = [
        Path(os.path.expandvars(os.path.expanduser(x.strip()))).resolve()
        for x in allow.split("|")
        if x.strip()
    ]
    sp = str(p.resolve())
    if not any(sp.startswith(str(pref)) for pref in prefixes):
        raise PermissionError(
            "Path is outside SCFMS_ALLOWED_PATH_PREFIXES; "
            f"resolved: {p}"
        )


def validate_server_read_path(raw: str) -> Path:
    p = Path(os.path.expandvars(os.path.expanduser(raw.strip()))).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Not a file: {p}")
    _path_allowed_by_prefixes(p)
    return p


def validate_allowed_existing_path(raw: str) -> Path:
    """Resolve *raw* to an existing file or directory; same prefix rules as ``validate_server_read_path``."""
    p = Path(os.path.expandvars(os.path.expanduser(raw.strip()))).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Not found: {p}")
    _path_allowed_by_prefixes(p)
    return p


def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def read_meta(job_id: str) -> Optional[Dict[str, Any]]:
    mp = jobs_root() / job_id / "meta.json"
    if not mp.is_file():
        return None
    return json.loads(mp.read_text(encoding="utf-8"))


def update_meta(job_id: str, **kwargs: Any) -> None:
    mp = jobs_root() / job_id / "meta.json"
    if not mp.is_file():
        return
    data = json.loads(mp.read_text(encoding="utf-8"))
    data.update(kwargs)
    data["updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _atomic_write_json(mp, data)


def list_recent_jobs(limit: int = 25) -> str:
    root = jobs_root()
    rows: List[tuple[float, str]] = []
    for d in root.iterdir():
        if not d.is_dir():
            continue
        meta = d / "meta.json"
        if not meta.is_file():
            continue
        try:
            m = json.loads(meta.read_text(encoding="utf-8"))
            st = m.get("updated") or m.get("created") or ""
            rows.append((meta.stat().st_mtime, m.get("id", d.name)))
        except (json.JSONDecodeError, OSError):
            continue
    rows.sort(key=lambda x: -x[0])
    lines = ["Recent job IDs (newest first):", ""]
    for _, jid in rows[:limit]:
        m = read_meta(jid)
        if not m:
            continue
        status = m.get("status", "?")
        typ = m.get("type", "?")
        msg = (m.get("message") or "")[:60]
        lines.append(f"  • {jid}  [{typ}] {status} — {msg}")
    return "\n".join(lines) if len(lines) > 2 else "(no jobs yet)"


def embed_output_base() -> Path:
    """Directory root for ``<dataset_name>/embedded_<model>_<job>.h5ad``."""
    raw = os.environ.get("SCFMS_SLURM_EMBED_BASE", "").strip()
    if raw:
        return Path(os.path.expandvars(os.path.expanduser(raw))).resolve()
    return jobs_root() / "slurm_embeddings"


def sanitize_dataset_folder(name: str, fallback: str = "dataset") -> str:
    from slurm_gpu import sanitize_dataset_name

    return sanitize_dataset_name(name, fallback)


def sync_slurm_meta(job_id: str) -> None:
    """Update meta from ``squeue`` / ``sacct`` for Slurm-backed jobs."""
    jid = job_id.strip()
    m = read_meta(jid)
    if not m or m.get("type") not in ("scfm_slurm", "benchmark_slurm"):
        return
    if m.get("status") in ("done", "error"):
        return
    if not m.get("slurm_job_id"):
        return
    from slurm_gpu import is_slurm_finished_success, slurm_aggregate_state

    sid = str(m["slurm_job_id"])
    agg = slurm_aggregate_state(sid)
    done, ok = is_slurm_finished_success(agg)
    outp = m.get("result_path")
    typ = m.get("type")
    if not done:
        update_meta(jid, status="running", message=f"slurm: {agg}", step=str(agg).lower())
        return
    if ok:
        if typ == "scfm_slurm":
            if outp and Path(outp).is_file():
                update_meta(
                    jid,
                    status="done",
                    message="slurm COMPLETED",
                    step="complete",
                    step_index=1,
                )
            else:
                update_meta(
                    jid,
                    status="error",
                    error="Slurm COMPLETED but output .h5ad not found",
                    message="missing output",
                )
            return
        if typ == "benchmark_slurm":
            if outp and Path(outp).is_file():
                try:
                    info = json.loads(Path(outp).read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError) as e:
                    update_meta(
                        jid,
                        status="error",
                        error=f"Invalid benchmark_complete.json: {e}",
                        message="bad result file",
                    )
                    return
                if info.get("ok"):
                    update_meta(
                        jid,
                        status="done",
                        message="benchmark COMPLETED",
                        step="complete",
                        step_index=1,
                        wandb_url=info.get("wandb_url"),
                        wandb_project_url=info.get("wandb_project_url"),
                        benchmark_session_dir=info.get("session_dir"),
                        n_models_trained=info.get("n_models_trained"),
                    )
                else:
                    update_meta(
                        jid,
                        status="error",
                        error=info.get("error") or "benchmark reported failure",
                        message="benchmark failed",
                    )
            else:
                update_meta(
                    jid,
                    status="error",
                    error="Slurm COMPLETED but benchmark_complete.json missing",
                    message="missing output",
                )
            return
    update_meta(
        jid,
        status="error",
        error=f"Slurm state: {agg}",
        message="slurm failed",
    )


def start_scfm_slurm_job(
    adata: Any,
    scfm_kwargs: Dict[str, Any],
    dataset_name: str,
    repo_root: Optional[Path],
    partition: str,
    gres: str,
    cpus: int,
    mem: str,
    time_limit: str,
    bash_prologue: str,
) -> str:
    """Write staging files, ``sbatch`` a GPU Slurm script; output under a dataset folder."""
    from slurm_gpu import (
        build_slurm_gpu_embed_script,
        default_repo_root,
        run_sbatch,
        sanitize_dataset_name,
    )

    job_id = uuid.uuid4().hex[:12]
    jd = jobs_root() / job_id
    jd.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(jd / "input.h5ad")

    repo = Path(repo_root).resolve() if repo_root else default_repo_root()
    ds = sanitize_dataset_name(dataset_name)
    out_dir = embed_output_base() / ds
    out_dir.mkdir(parents=True, exist_ok=True)
    model = str(scfm_kwargs["model"])
    out_h5ad = out_dir / f"embedded_{model}_{job_id}.h5ad"

    params: Dict[str, Any] = {
        "repo_root": str(repo),
        "input_h5ad": str((jd / "input.h5ad").resolve()),
        "output_h5ad": str(out_h5ad.resolve()),
        "model": model,
        "matrix_spec": str(scfm_kwargs.get("matrix_spec") or "X"),
        "obsm_key": scfm_kwargs.get("obsm_key"),
        "scgpt_ckpt": scfm_kwargs.get("scgpt_ckpt"),
        "n_latent_scvi": int(scfm_kwargs.get("n_latent_scvi") or 64),
    }
    pj = jd / "embed_params.json"
    _atomic_write_json(pj, params)

    script = build_slurm_gpu_embed_script(
        repo_root=repo,
        stage_dir=jd,
        params_json=pj,
        partition=effective_slurm_partition(partition),
        gres=str(gres or "gpu:1").strip() or "gpu:1",
        cpus=int(cpus),
        mem=str(mem),
        time_limit=str(time_limit),
        job_name=f"scfms-{model}-{job_id[:6]}",
        bash_prologue=bash_prologue or "",
    )
    sh_path = jd / "slurm_gpu_embed.sh"
    sh_path.write_text(script, encoding="utf-8")
    sh_path.chmod(0o755)

    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    base_meta: Dict[str, Any] = {
        "id": job_id,
        "type": "scfm_slurm",
        "dataset_name": ds,
        "result_path": str(out_h5ad.resolve()),
        "batch_script": str(sh_path.resolve()),
        "embed_params": str(pj.resolve()),
        "params": scfm_kwargs,
        "created": now,
        "updated": now,
        "timings": {},
        "step_total": 1,
        "step_index": 0,
    }

    try:
        sid = run_sbatch(sh_path)
    except Exception as e:
        base_meta.update(
            status="error",
            message="sbatch failed",
            error=str(e),
            slurm_job_id=None,
        )
        _atomic_write_json(jd / "meta.json", base_meta)
        return job_id

    base_meta.update(
        status="submitted",
        message=f"Slurm job {sid} queued",
        slurm_job_id=sid,
        step="queued",
        error=None,
    )
    _atomic_write_json(jd / "meta.json", base_meta)
    return job_id


def start_scfm_slurm_job_from_h5ad(
    input_h5ad: str | Path,
    scfm_kwargs: Dict[str, Any],
    dataset_name: str,
    repo_root: Optional[Path],
    *,
    partition: str,
    gres: str,
    cpus: int,
    mem: str,
    time_limit: str,
    bash_prologue: str,
) -> str:
    """Submit a Slurm embedding job using an existing shared-filesystem `.h5ad` path."""
    from slurm_gpu import (
        build_slurm_gpu_embed_script,
        default_repo_root,
        run_sbatch,
        sanitize_dataset_name,
    )

    in_path = Path(input_h5ad).expanduser().resolve()
    if not in_path.is_file():
        raise FileNotFoundError(f"Input h5ad not found: {in_path}")

    job_id = uuid.uuid4().hex[:12]
    jd = jobs_root() / job_id
    jd.mkdir(parents=True, exist_ok=True)

    repo = Path(repo_root).resolve() if repo_root else default_repo_root()
    ds = sanitize_dataset_name(dataset_name)
    out_dir = embed_output_base() / ds
    out_dir.mkdir(parents=True, exist_ok=True)
    model = str(scfm_kwargs["model"])
    out_h5ad = out_dir / f"embedded_{model}_{job_id}.h5ad"

    params: Dict[str, Any] = {
        "repo_root": str(repo),
        "input_h5ad": str(in_path),
        "output_h5ad": str(out_h5ad.resolve()),
        "model": model,
        "matrix_spec": str(scfm_kwargs.get("matrix_spec") or "X"),
        "obsm_key": scfm_kwargs.get("obsm_key"),
        "scgpt_ckpt": scfm_kwargs.get("scgpt_ckpt"),
        "n_latent_scvi": int(scfm_kwargs.get("n_latent_scvi") or 64),
    }
    pj = jd / "embed_params.json"
    _atomic_write_json(pj, params)

    script = build_slurm_gpu_embed_script(
        repo_root=repo,
        stage_dir=jd,
        params_json=pj,
        partition=effective_slurm_partition(partition),
        gres=str(gres or "gpu:1").strip() or "gpu:1",
        cpus=int(cpus),
        mem=str(mem),
        time_limit=str(time_limit),
        job_name=f"scfms-{model}-{job_id[:6]}",
        bash_prologue=bash_prologue or "",
    )
    sh_path = jd / "slurm_gpu_embed.sh"
    sh_path.write_text(script, encoding="utf-8")
    sh_path.chmod(0o755)

    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    base_meta: Dict[str, Any] = {
        "id": job_id,
        "type": "scfm_slurm",
        "dataset_name": ds,
        "result_path": str(out_h5ad.resolve()),
        "batch_script": str(sh_path.resolve()),
        "embed_params": str(pj.resolve()),
        "params": scfm_kwargs,
        "input_h5ad": str(in_path),
        "created": now,
        "updated": now,
        "timings": {},
        "step_total": 1,
        "step_index": 0,
    }

    try:
        sid = run_sbatch(sh_path)
    except Exception as e:
        base_meta.update(
            status="error",
            message="sbatch failed",
            error=str(e),
            slurm_job_id=None,
        )
        _atomic_write_json(jd / "meta.json", base_meta)
        return job_id

    base_meta.update(
        status="submitted",
        message=f"Slurm job {sid} queued",
        slurm_job_id=sid,
        step="queued",
        error=None,
    )
    _atomic_write_json(jd / "meta.json", base_meta)
    return job_id


def start_benchmark_slurm_job(
    adata: Any,
    bench_params: Dict[str, Any],
    repo_root: Optional[Path],
    *,
    partition: str,
    cpus: int,
    mem: str,
    time_limit: str,
    gres: str,
    bash_prologue: str,
    extra_sbatch: str,
) -> str:
    """Stage train AnnData + JSON params, ``sbatch`` GPU benchmark + optional test eval + wandb."""
    from slurm_gpu import (
        build_slurm_gpu_benchmark_script,
        default_repo_root,
        run_sbatch,
    )

    job_id = uuid.uuid4().hex[:12]
    jd = jobs_root() / job_id
    jd.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(jd / "input.h5ad")

    repo = Path(repo_root).resolve() if repo_root else default_repo_root()
    pj = jd / "benchmark_slurm_params.json"
    payload = {
        "repo_root": str(repo.resolve()),
        "input_h5ad": str((jd / "input.h5ad").resolve()),
        "job_stage_dir": str(jd.resolve()),
        "complete_json": str((jd / "benchmark_complete.json").resolve()),
        **bench_params,
    }
    _atomic_write_json(pj, payload)

    script = build_slurm_gpu_benchmark_script(
        repo_root=repo,
        stage_dir=jd,
        params_json=pj,
        partition=effective_slurm_partition(partition),
        cpus=int(cpus),
        mem=str(mem),
        time_limit=str(time_limit),
        job_name=f"scfms-bench-{job_id[:6]}",
        gres=str(gres or "gpu:1").strip() or "gpu:1",
        bash_prologue=bash_prologue or "",
        extra_sbatch=extra_sbatch or "",
    )
    sh_path = jd / "slurm_gpu_benchmark.sh"
    sh_path.write_text(script, encoding="utf-8")
    sh_path.chmod(0o755)

    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    complete_path = jd / "benchmark_complete.json"
    base_meta: Dict[str, Any] = {
        "id": job_id,
        "type": "benchmark_slurm",
        "result_path": str(complete_path.resolve()),
        "batch_script": str(sh_path.resolve()),
        "bench_params_json": str(pj.resolve()),
        "params": bench_params,
        "created": now,
        "updated": now,
        "timings": {},
        "step_total": 1,
        "step_index": 0,
        "wandb_url": None,
        "wandb_project_url": None,
        "benchmark_session_dir": None,
    }

    try:
        sid = run_sbatch(sh_path)
    except Exception as e:
        base_meta.update(
            status="error",
            message="sbatch failed",
            error=str(e),
            slurm_job_id=None,
        )
        _atomic_write_json(jd / "meta.json", base_meta)
        return job_id

    base_meta.update(
        status="submitted",
        message=f"Slurm job {sid} queued",
        slurm_job_id=sid,
        step="queued",
        error=None,
    )
    _atomic_write_json(jd / "meta.json", base_meta)
    return job_id


def start_benchmark_slurm_job_from_h5ad(
    input_h5ad: str | Path,
    bench_params: Dict[str, Any],
    repo_root: Optional[Path],
    *,
    partition: str,
    cpus: int,
    mem: str,
    time_limit: str,
    gres: str,
    bash_prologue: str,
    extra_sbatch: str,
) -> str:
    """Submit benchmark training from an existing shared-filesystem `.h5ad` path."""
    from slurm_gpu import (
        build_slurm_gpu_benchmark_script,
        default_repo_root,
        run_sbatch,
    )

    in_path = Path(input_h5ad).expanduser().resolve()
    if not in_path.is_file():
        raise FileNotFoundError(f"Input h5ad not found: {in_path}")

    job_id = uuid.uuid4().hex[:12]
    jd = jobs_root() / job_id
    jd.mkdir(parents=True, exist_ok=True)

    repo = Path(repo_root).resolve() if repo_root else default_repo_root()
    pj = jd / "benchmark_slurm_params.json"
    payload = {
        "repo_root": str(repo.resolve()),
        "input_h5ad": str(in_path),
        "job_stage_dir": str(jd.resolve()),
        "complete_json": str((jd / "benchmark_complete.json").resolve()),
        **bench_params,
    }
    _atomic_write_json(pj, payload)

    script = build_slurm_gpu_benchmark_script(
        repo_root=repo,
        stage_dir=jd,
        params_json=pj,
        partition=effective_slurm_partition(partition),
        cpus=int(cpus),
        mem=str(mem),
        time_limit=str(time_limit),
        job_name=f"scfms-bench-{job_id[:6]}",
        gres=str(gres or "gpu:1").strip() or "gpu:1",
        bash_prologue=bash_prologue or "",
        extra_sbatch=extra_sbatch or "",
    )
    sh_path = jd / "slurm_gpu_benchmark.sh"
    sh_path.write_text(script, encoding="utf-8")
    sh_path.chmod(0o755)

    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    complete_path = jd / "benchmark_complete.json"
    base_meta: Dict[str, Any] = {
        "id": job_id,
        "type": "benchmark_slurm",
        "result_path": str(complete_path.resolve()),
        "batch_script": str(sh_path.resolve()),
        "bench_params_json": str(pj.resolve()),
        "params": bench_params,
        "input_h5ad": str(in_path),
        "created": now,
        "updated": now,
        "timings": {},
        "step_total": 1,
        "step_index": 0,
        "wandb_url": None,
        "wandb_project_url": None,
        "benchmark_session_dir": None,
    }

    try:
        sid = run_sbatch(sh_path)
    except Exception as e:
        base_meta.update(
            status="error",
            message="sbatch failed",
            error=str(e),
            slurm_job_id=None,
        )
        _atomic_write_json(jd / "meta.json", base_meta)
        return job_id

    base_meta.update(
        status="submitted",
        message=f"Slurm job {sid} queued",
        slurm_job_id=sid,
        step="queued",
        error=None,
    )
    _atomic_write_json(jd / "meta.json", base_meta)
    return job_id


def format_meta_report(job_id: str) -> str:
    jid = job_id.strip()
    sync_slurm_meta(jid)
    m = read_meta(jid)
    if not m:
        return f"No job found: {job_id!r}"
    lines = [
        f"Job {m.get('id')}",
        f"  type:     {m.get('type')}",
        f"  status:   {m.get('status')}",
        f"  message:  {m.get('message')}",
        f"  step:     {m.get('step')} ({m.get('step_index')}/{m.get('step_total')})",
    ]
    if m.get("type") == "scfm_slurm":
        lines.extend(
            [
                f"  Slurm:   {m.get('slurm_job_id')}",
                f"  dataset: {m.get('dataset_name')}",
                f"  batch:   {m.get('batch_script')}",
                f"  check:   squeue -j {m.get('slurm_job_id')}  |  sacct -j {m.get('slurm_job_id')}",
            ]
        )
    if m.get("type") == "benchmark_slurm":
        lines.extend(
            [
                f"  Slurm:   {m.get('slurm_job_id')}",
                f"  batch:   {m.get('batch_script')}",
                f"  check:   squeue -j {m.get('slurm_job_id')}  |  sacct -j {m.get('slurm_job_id')}",
            ]
        )
        if m.get("wandb_url"):
            lines.append(f"  wandb:   {m.get('wandb_url')}")
        if m.get("benchmark_session_dir"):
            lines.append(f"  session: {m.get('benchmark_session_dir')}")
    if m.get("eta_seconds") is not None:
        lines.append(f"  rough ETA remaining: {_fmt_sec(float(m['eta_seconds']))}")
    if m.get("wall_seconds") is not None:
        lines.append(f"  wall:     {_fmt_sec(float(m['wall_seconds']))}")
    if m.get("error"):
        lines.append(f"  error:    {m['error']}")
    timings = m.get("timings") or {}
    if timings:
        lines.append("")
        lines.append("  Timings:")
        for k, v in sorted(timings.items(), key=lambda x: -float(x[1])):
            lines.append(f"    • {k}: {_fmt_sec(float(v))}")
    if m.get("result_path"):
        lines.append("")
        lines.append(f"  result:   {m['result_path']}")
    lines.append("")
    lines.append(f"  updated:  {m.get('updated')}")
    return "\n".join(lines)


def _fmt_sec(s: float) -> str:
    if s >= 3600:
        return f"{s/3600:.2f} h ({s:.0f}s)"
    if s >= 120:
        return f"{s/60:.1f} min ({s:.1f}s)"
    return f"{s:.2f}s"


def _import_preprocess():
    """Load ``preprocess.py`` by path so workers succeed regardless of process cwd."""
    import importlib.util

    app_dir = Path(__file__).resolve().parent
    path = app_dir / "preprocess.py"
    spec = importlib.util.spec_from_file_location("scfms_preprocess_worker", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load preprocess from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _pipeline_progress_sink(job_id: str) -> Callable[..., None]:
    def cb(
        step_name: str,
        step_idx: int,
        step_total: int,
        timings: Dict[str, float],
        wall_sec: float,
        eta_sec: float,
    ) -> None:
        update_meta(
            job_id,
            step=step_name,
            step_index=step_idx,
            step_total=step_total,
            timings=timings,
            wall_seconds=wall_sec,
            eta_seconds=eta_sec,
            message=f"{step_name} ({step_idx}/{step_total})",
        )

    return cb


def start_pipeline_job(adata: Any, pipeline_kwargs: Dict[str, Any]) -> str:
    """Write AnnData + meta and start a daemon thread running the Scanpy pipeline."""
    import anndata as ad

    job_id = uuid.uuid4().hex[:12]
    jd = jobs_root() / job_id
    jd.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(jd / "input.h5ad")
    meta = {
        "id": job_id,
        "type": "pipeline",
        "status": "queued",
        "step": "",
        "step_index": 0,
        "step_total": 0,
        "timings": {},
        "eta_seconds": None,
        "wall_seconds": 0.0,
        "message": "queued",
        "error": None,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "result_path": None,
        "params": pipeline_kwargs,
    }
    _atomic_write_json(jd / "meta.json", meta)

    def worker() -> None:
        try:
            pre = _import_preprocess()
            update_meta(job_id, status="running", message="loading input")
            ad_in = ad.read_h5ad(jd / "input.h5ad")
            cb = _pipeline_progress_sink(job_id)
            pk = dict(pipeline_kwargs)
            use_raw = bool(pk.pop("pipeline_use_raw", False))
            ad_in = pre.pipeline_base_adata(ad_in, use_raw)
            out, _fp, _fu, timings = pre.run_expression_pipeline(
                ad_in, progress_callback=cb, **pk
            )
            outp = jd / "result.h5ad"
            out.write_h5ad(outp)
            update_meta(
                job_id,
                status="done",
                message="finished",
                timings=timings,
                result_path=str(outp),
                step="complete",
                step_index=len(timings),
            )
        except Exception as e:
            update_meta(job_id, status="error", error=str(e), message="failed")

    threading.Thread(target=worker, daemon=True).start()
    return job_id


def start_scfm_job(adata: Any, scfm_kwargs: Dict[str, Any]) -> str:
    import anndata as ad

    job_id = uuid.uuid4().hex[:12]
    jd = jobs_root() / job_id
    jd.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(jd / "input.h5ad")
    meta = {
        "id": job_id,
        "type": "scfm",
        "status": "queued",
        "step": "",
        "step_index": 0,
        "step_total": 1,
        "timings": {},
        "eta_seconds": None,
        "wall_seconds": 0.0,
        "message": "queued",
        "error": None,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "result_path": None,
        "params": scfm_kwargs,
    }
    _atomic_write_json(jd / "meta.json", meta)

    def worker() -> None:
        try:
            pre = _import_preprocess()
            update_meta(job_id, status="running", message="embedding (long step)")
            ad_in = ad.read_h5ad(jd / "input.h5ad")
            t0 = time.perf_counter()
            out, _msg, timings = pre.attach_scfm_embedding(ad_in, **scfm_kwargs)
            wall = time.perf_counter() - t0
            outp = jd / "result.h5ad"
            out.write_h5ad(outp)
            update_meta(
                job_id,
                status="done",
                message="finished",
                timings=timings,
                wall_seconds=wall,
                eta_seconds=0.0,
                result_path=str(outp),
                step="embed_done",
                step_index=1,
                step_total=1,
            )
        except Exception as e:
            update_meta(job_id, status="error", error=str(e), message="failed")

    threading.Thread(target=worker, daemon=True).start()
    return job_id


def start_matrix_view_job(
    adata: Any,
    view_kwargs: Dict[str, Any],
    *,
    result_path: Optional[Path] = None,
) -> str:
    import anndata as ad

    job_id = uuid.uuid4().hex[:12]
    jd = jobs_root() / job_id
    jd.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(jd / "input.h5ad")
    outp = Path(result_path).expanduser().resolve() if result_path else (jd / "result.h5ad").resolve()
    outp.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "id": job_id,
        "type": "matrix_view",
        "status": "queued",
        "step": "",
        "step_index": 0,
        "step_total": 1,
        "timings": {},
        "eta_seconds": None,
        "wall_seconds": 0.0,
        "message": "queued",
        "error": None,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "result_path": str(outp),
        "params": view_kwargs,
    }
    _atomic_write_json(jd / "meta.json", meta)

    def worker() -> None:
        try:
            pre = _import_preprocess()
            update_meta(job_id, status="running", message="building matrix view")
            ad_in = ad.read_h5ad(jd / "input.h5ad")
            t0 = time.perf_counter()
            out = pre.build_matrix_view_adata(ad_in, **view_kwargs)
            wall = time.perf_counter() - t0
            out.write_h5ad(outp, compression="gzip")
            update_meta(
                job_id,
                status="done",
                message="finished",
                timings={"build_matrix_view_adata": wall},
                wall_seconds=wall,
                eta_seconds=0.0,
                result_path=str(outp),
                step="view_done",
                step_index=1,
                step_total=1,
            )
        except Exception as e:
            update_meta(job_id, status="error", error=str(e), message="failed")

    threading.Thread(target=worker, daemon=True).start()
    return job_id


def _copy_optional_text_file(src: str | Path | None, dst: Path) -> Optional[str]:
    if not src:
        return None
    p = Path(src).expanduser().resolve()
    if not p.is_file():
        return None
    dst.write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
    return str(dst.resolve())


def start_de_job(
    view_h5ad: str | Path,
    de_kwargs: Dict[str, Any],
    *,
    result_path: Optional[Path] = None,
) -> str:
    import anndata as ad
    import numpy as np
    import pandas as pd
    from scipy import sparse as sp
    from scipy import stats

    in_path = Path(view_h5ad).expanduser().resolve()
    if not in_path.is_file():
        raise FileNotFoundError(f"View h5ad not found: {in_path}")

    job_id = uuid.uuid4().hex[:12]
    jd = jobs_root() / job_id
    jd.mkdir(parents=True, exist_ok=True)
    outp = Path(result_path).expanduser().resolve() if result_path else (jd / "de_results.csv").resolve()
    outp.parent.mkdir(parents=True, exist_ok=True)

    params = dict(de_kwargs)
    params["view_h5ad"] = str(in_path)
    params["selection_path"] = _copy_optional_text_file(
        params.get("selection_path"), jd / "selection.txt"
    )
    params["selection_a_path"] = _copy_optional_text_file(
        params.get("selection_a_path"), jd / "selection_A.txt"
    )
    params["selection_b_path"] = _copy_optional_text_file(
        params.get("selection_b_path"), jd / "selection_B.txt"
    )

    meta = {
        "id": job_id,
        "type": "de",
        "status": "queued",
        "step": "",
        "step_index": 0,
        "step_total": 1,
        "timings": {},
        "eta_seconds": None,
        "wall_seconds": 0.0,
        "message": "queued",
        "error": None,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "result_path": str(outp),
        "params": params,
        "input_h5ad": str(in_path),
    }
    _atomic_write_json(jd / "meta.json", meta)

    def _load_names(path_str: Optional[str]) -> list[str]:
        if not path_str:
            return []
        p = Path(path_str)
        if not p.is_file():
            return []
        return [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]

    def _compute_df(
        adata: Any,
        fg_mask: Any,
        bg_mask: Any,
        *,
        method: str,
        top_n: int,
    ) -> pd.DataFrame:
        if int(np.sum(fg_mask)) < 5 or int(np.sum(bg_mask)) < 5:
            return pd.DataFrame()
        X = adata.X
        if sp.issparse(X):
            X = X.tocsr()
        X_fg = X[fg_mask]
        X_bg = X[bg_mask]
        if sp.issparse(X_fg):
            mu_fg = np.asarray(X_fg.mean(axis=0)).ravel()
        else:
            mu_fg = np.asarray(X_fg.mean(axis=0)).ravel()
        if sp.issparse(X_bg):
            mu_bg = np.asarray(X_bg.mean(axis=0)).ravel()
        else:
            mu_bg = np.asarray(X_bg.mean(axis=0)).ravel()
        logfc = np.log1p(mu_fg + 1e-9) - np.log1p(mu_bg + 1e-9)
        pvals = np.ones(adata.n_vars, dtype=float)
        use_wilcoxon = str(method or "t-test") == "wilcoxon"
        for j in range(adata.n_vars):
            try:
                a = X_fg[:, j].toarray().ravel() if sp.issparse(X_fg) else np.asarray(X_fg[:, j]).ravel()
                b = X_bg[:, j].toarray().ravel() if sp.issparse(X_bg) else np.asarray(X_bg[:, j]).ravel()
                if use_wilcoxon:
                    _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
                else:
                    _, p = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
                pvals[j] = p if np.isfinite(p) else 1.0
            except Exception:
                pvals[j] = 1.0
        m = len(pvals)
        order_p = np.argsort(pvals)
        ranks = np.arange(1, m + 1, dtype=float)
        qvals = np.empty_like(pvals)
        qvals[order_p] = pvals[order_p] * m / ranks
        for i in range(m - 2, -1, -1):
            qvals[order_p[i]] = min(qvals[order_p[i]], qvals[order_p[i + 1]])
        qvals = np.clip(qvals, 0.0, 1.0)
        order = np.argsort(-np.abs(logfc))
        keep = order[: max(1, int(top_n or 50))]
        return pd.DataFrame(
            {
                "gene": adata.var_names.values[keep],
                "logFC": logfc[keep],
                "p_value": pvals[keep],
                "q_value": qvals[keep],
            }
        )

    def worker() -> None:
        try:
            update_meta(job_id, status="running", message="computing differential expression")
            ad_in = ad.read_h5ad(in_path)
            mode = str(params.get("mode") or "manual")
            method = str(params.get("method") or "t-test")
            top_n = int(params.get("top_n") or 50)
            fg_mask = None
            bg_mask = None
            if mode == "selection":
                selected = _load_names(params.get("selection_path"))
                fg_mask = ad_in.obs.index.isin(selected).to_numpy()
                bg_query = str(params.get("bg_query") or "").strip()
                if bg_query:
                    try:
                        idx = ad_in.obs.query(bg_query).index
                        bg_mask = ad_in.obs.index.isin(idx).to_numpy()
                    except Exception:
                        bg_mask = ~fg_mask
                else:
                    bg_mask = ~fg_mask
            elif mode == "ab":
                sel_a = _load_names(params.get("selection_a_path"))
                sel_b = _load_names(params.get("selection_b_path"))
                fg_mask = ad_in.obs.index.isin(sel_a).to_numpy()
                bg_mask = ad_in.obs.index.isin(sel_b).to_numpy()
            else:
                group_col = str(params.get("group_col") or "")
                if group_col not in ad_in.obs.columns:
                    raise KeyError(f"Group column not found in obs: {group_col}")
                labels = ad_in.obs[group_col].astype(str)
                fg_vals = [str(v) for v in (params.get("fg_vals") or []) if str(v).strip()]
                bg_vals = [str(v) for v in (params.get("bg_vals") or []) if str(v).strip()]
                fg_mask = labels.isin(fg_vals).to_numpy()
                bg_mask = (~fg_mask) if not bg_vals else labels.isin(bg_vals).to_numpy()
            if fg_mask is None or bg_mask is None:
                raise RuntimeError("Failed to construct foreground/background masks.")
            t0 = time.perf_counter()
            df = _compute_df(ad_in, fg_mask, bg_mask, method=method, top_n=top_n)
            df.to_csv(outp, index=False)
            wall = time.perf_counter() - t0
            update_meta(
                job_id,
                status="done",
                message="finished",
                timings={"compute_de": wall},
                wall_seconds=wall,
                eta_seconds=0.0,
                result_path=str(outp),
                step="de_done",
                step_index=1,
                step_total=1,
                fg_count=int(np.sum(fg_mask)),
                bg_count=int(np.sum(bg_mask)),
                n_results=int(len(df)),
            )
        except Exception as e:
            update_meta(job_id, status="error", error=str(e), message="failed")

    threading.Thread(target=worker, daemon=True).start()
    return job_id


def result_h5ad_path(job_id: str) -> Optional[Path]:
    m = read_meta(job_id.strip())
    if not m or m.get("status") != "done":
        return None
    rp = m.get("result_path")
    if not rp:
        return None
    p = Path(rp)
    return p if p.is_file() else None
