#!/usr/bin/env python3
import hashlib
import json
import multiprocessing as mp
import os
import re
import sys
import threading
import time
import traceback
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parent.parent
_APP = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_APP) not in sys.path:
    sys.path.insert(0, str(_APP))

import env_bootstrap

env_bootstrap.load_repo_dotenv()

from slurm_defaults import default_slurm_partition, effective_slurm_partition

import anndata as ad
import background_jobs as bgjobs
from gradio_config import launch_gradio_demo, runtime_info_markdown
from scfm_model_paths import (
    model_weights_choices_and_value,
    model_weights_gr_update,
    normalize_ui_weights_path,
)
import session_results as sess_res
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from sklearn.decomposition import PCA, TruncatedSVD

try:
    from anndata.abc import CSCDataset, CSRDataset

    _DISK_SPARSE_MATRIX_TYPES = (CSCDataset, CSRDataset)
except ImportError:  # pragma: no cover — very old anndata
    _DISK_SPARSE_MATRIX_TYPES = ()

sc.settings.verbosity = "error"


def is_oom_error(exc: BaseException) -> bool:
    """Heuristic: RAM exhaustion, GPU OOM, or common allocator failures."""
    if isinstance(exc, MemoryError):
        return True
    if type(exc).__name__ in ("OutOfMemoryError",):
        return True
    s = str(exc).lower()
    needles = (
        "out of memory",
        "cuda out of memory",
        "cudnn status not initialized",
        "cannot allocate",
        "allocation failed",
        "std::bad_alloc",
        "mkl_malloc",
        "bad allocation",
        "killed process",
        "resource temporarily unavailable",
    )
    return any(n in s for n in needles)


def oom_user_message(exc: BaseException) -> str:
    return (
        f"**Resource limit (often OOM):** `{type(exc).__name__}: {exc}`\n\n"
        "**Retry:** use **Subset cells** (after load / dense), lower **Max cells** in Quick UMAP or preprocessing, "
        "or use **Background** / **Slurm** jobs. If nothing was saved, your previous session object is unchanged."
    )


def subset_adata_random_cells(
    adata: ad.AnnData, n_keep: int, seed: int = 0
) -> ad.AnnData:
    """Random ``n_keep`` rows (cells), stable given ``seed``."""
    n_obs = int(adata.n_obs)
    nk = int(n_keep)
    if nk <= 0 or nk >= n_obs:
        return adata.copy()
    rng = np.random.default_rng(int(seed))
    ix = np.sort(rng.choice(n_obs, size=nk, replace=False))
    return adata[ix].copy()


DENSIFY_STATUS_JSON = "densify_status.json"
DENSE_MATERIALIZED_H5AD = "adata_dense_materialized.h5ad"

# Background dense load: full ``read_h5ad`` in a daemon thread (same process as Gradio).
# No ``adata_dense_materialized.h5ad`` copy — results stay in RAM until the UI consumes them.
_DENSE_JOB_LOCK = threading.Lock()
_DENSE_JOBS: Dict[str, Dict[str, Any]] = {}
# After ``dense_ram_take_if_ready``, the AnnData is kept here so Gradio callbacks can reuse it
# without re-opening backed mode or writing a materialized ``.h5ad``.
_DENSE_SESSION_CACHE_LOCK = threading.Lock()
_DENSE_SESSION_ADATA: Dict[str, ad.AnnData] = {}


def dense_session_get_adata(session_dir: str) -> Optional[ad.AnnData]:
    key = dense_job_key(session_dir)
    if not key:
        return None
    with _DENSE_SESSION_CACHE_LOCK:
        return _DENSE_SESSION_ADATA.get(key)


def dense_session_clear_adata(session_dir: str) -> None:
    key = dense_job_key(session_dir)
    if not key:
        return
    with _DENSE_SESSION_CACHE_LOCK:
        _DENSE_SESSION_ADATA.pop(key, None)


def dense_job_key(session_dir: str) -> str:
    if not (session_dir or "").strip():
        return ""
    try:
        return str(Path(session_dir).expanduser().resolve())
    except OSError:
        return str(session_dir).strip()


def _dense_bg_thread(key: str, src: str, gen: int) -> None:
    err_tb: Optional[str] = None
    adata: Optional[ad.AnnData] = None
    try:
        adata = ad.read_h5ad(src)
    except BaseException:  # noqa: BLE001
        err_tb = traceback.format_exc()
    with _DENSE_JOB_LOCK:
        cur = _DENSE_JOBS.get(key)
        if not cur or int(cur.get("gen", 0)) != gen:
            return
        if cur.get("state") == "cancelled":
            return
        if err_tb is not None:
            _DENSE_JOBS[key] = {
                **cur,
                "state": "error",
                "error": err_tb,
                "adata": None,
                "finished": time.time(),
            }
            return
        assert adata is not None
        adata.uns.pop("_scfms_densify_src", None)
        _DENSE_JOBS[key] = {
            **cur,
            "state": "done",
            "error": None,
            "adata": adata,
            "finished": time.time(),
        }


def dense_load_start(session_dir: str, src_h5ad: str) -> tuple[bool, str]:
    """Start a background full read of *src_h5ad* into RAM for *session_dir*. Returns (ok, error_message)."""
    key = dense_job_key(session_dir)
    if not key:
        return False, "no session directory"
    sp = Path(str(src_h5ad))
    if not sp.is_file():
        return False, f"source missing: {sp}"
    sdp = Path(key)
    for pth in (
        sdp / DENSIFY_STATUS_JSON,
        sdp / DENSE_MATERIALIZED_H5AD,
        sdp / f"{DENSE_MATERIALIZED_H5AD}.writing.h5ad",
    ):
        try:
            pth.unlink(missing_ok=True)
        except OSError:
            pass
    dense_session_clear_adata(key)
    with _DENSE_JOB_LOCK:
        cur = _DENSE_JOBS.get(key)
        if cur and cur.get("state") == "running":
            return False, "already running"
        gen = (int(cur.get("gen", 0)) + 1) if cur else 1
        _DENSE_JOBS[key] = {
            "state": "running",
            "gen": gen,
            "started": time.time(),
            "src": str(sp.resolve()),
        }
    t = threading.Thread(
        target=_dense_bg_thread,
        args=(key, str(sp.resolve()), gen),
        name=f"scfms-dense-{gen}",
        daemon=True,
    )
    t.start()
    return True, ""


def dense_load_cancel(session_dir: str) -> None:
    key = dense_job_key(session_dir)
    if not key:
        return
    with _DENSE_JOB_LOCK:
        cur = _DENSE_JOBS.get(key)
        if cur and cur.get("state") == "running":
            _DENSE_JOBS[key] = {**cur, "state": "cancelled"}


def dense_load_poll(session_dir: str) -> Dict[str, Any]:
    """``state``: idle | running | done | error | cancelled."""
    key = dense_job_key(session_dir)
    if not key:
        return {"state": "idle"}
    with _DENSE_JOB_LOCK:
        cur = _DENSE_JOBS.get(key)
        if not cur:
            return {"state": "idle"}
        st = str(cur.get("state") or "idle")
        if st == "running":
            return {
                "state": "running",
                "started": float(cur.get("started", 0.0)),
                "src": str(cur.get("src", "")),
            }
        if st == "done":
            return {"state": "done", "finished": float(cur.get("finished", 0.0))}
        if st == "error":
            return {"state": "error", "error": str(cur.get("error", ""))}
        if st == "cancelled":
            return {"state": "cancelled"}
        return {"state": "idle"}


def dense_ram_take_if_ready(session_dir: str) -> Optional[ad.AnnData]:
    """If a background dense load finished, pop and return the in-RAM ``AnnData``; else ``None``."""
    key = dense_job_key(session_dir)
    if not key:
        return None
    with _DENSE_JOB_LOCK:
        cur = _DENSE_JOBS.get(key)
        if not cur or cur.get("state") != "done":
            return None
        adata = cur.get("adata")
        del _DENSE_JOBS[key]
    out = adata if isinstance(adata, ad.AnnData) else None
    if out is not None:
        with _DENSE_SESSION_CACHE_LOCK:
            _DENSE_SESSION_ADATA[key] = out
    return out


def dense_load_pop_terminal(session_dir: str) -> tuple[str, Optional[str]]:
    """Remove error or cancelled job; returns (kind, payload) kind in error|cancelled|noop."""
    key = dense_job_key(session_dir)
    if not key:
        return "noop", None
    with _DENSE_JOB_LOCK:
        cur = _DENSE_JOBS.get(key)
        if not cur:
            return "noop", None
        st = cur.get("state")
        if st == "error":
            err = str(cur.get("error", ""))
            del _DENSE_JOBS[key]
            return "error", err
        if st == "cancelled":
            del _DENSE_JOBS[key]
            return "cancelled", None
    return "noop", None
# Max distinct groups (plus "Other") when drawing **Stratify QC histograms by obs**.
DIST_HIST_MAX_GROUPS = 12
# Max matrix entries sampled per group for the value histogram (stratified).
_DIST_VALUE_CAP_PER_GROUP = 150_000

# Client-only: animated Status while /load-from-server runs (cleared in .then after fn finishes).
_LOAD_PATH_STATUS_START_JS = r"""
() => {
    if (window._scfmsLoadPathTicker) {
        clearInterval(window._scfmsLoadPathTicker);
        window._scfmsLoadPathTicker = null;
    }
    const el = document.getElementById("scfms_load_status_anchor");
    if (!el) return;
    el.classList.add("scfms-load-path-pending");
    const ta = el.querySelector("textarea") || el.querySelector('input[type="text"]');
    if (!ta) return;
    let i = 0;
    const tick = () => {
        ta.value = "Loading .h5ad from server" + ".".repeat(i % 4);
        i++;
    };
    tick();
    window._scfmsLoadPathTicker = setInterval(tick, 380);
}
"""
_LOAD_PATH_STATUS_STOP_JS = r"""
() => {
    if (window._scfmsLoadPathTicker) {
        clearInterval(window._scfmsLoadPathTicker);
        window._scfmsLoadPathTicker = null;
    }
    const el = document.getElementById("scfms_load_status_anchor");
    if (el) el.classList.remove("scfms-load-path-pending");
}
"""


def adata_is_backed(adata: Optional[ad.AnnData]) -> bool:
    """True if ``X`` (and large arrays) are memory-mapped from an ``.h5ad`` on disk."""
    if adata is None:
        return False
    return bool(getattr(adata, "isbacked", False))


def _require_dense_adata(adata: Optional[ad.AnnData], action: str) -> None:
    if adata_is_backed(adata):
        raise RuntimeError(
            f"{action} needs fully loaded (dense) data. Use **Load dense in background**, wait until it "
            f"finishes (see **Dense load** log), then retry."
        )


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except (OSError, TypeError, ValueError):
        return False
    return True


def _gradio_bool(v: Any) -> bool:
    """Gradio may pass bool, None, or occasionally string-ish values."""
    if v is None:
        return False
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    return bool(v)


@contextmanager
def _timed(timings: Dict[str, float], name: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        timings[name] = time.perf_counter() - t0


def _fmt_dur(seconds: float) -> str:
    if seconds >= 3600:
        return f"{seconds / 3600:.2f} h ({seconds:.0f} s)"
    if seconds >= 120:
        return f"{seconds / 60:.1f} min ({seconds:.1f} s)"
    return f"{seconds:.2f} s"


def format_timing_report(
    title: str,
    timings: Dict[str, float],
    wall_s: Optional[float] = None,
    footer: str = "",
) -> str:
    lines = [title, ""]
    for k, v in timings.items():
        lines.append(f"  • {k}: {_fmt_dur(float(v))}")
    acc = sum(float(v) for v in timings.values())
    lines.append(f"  — sum(step times): {_fmt_dur(acc)}")
    if wall_s is not None:
        lines.append(f"  — wall clock: {_fmt_dur(float(wall_s))}")
    if footer:
        lines.extend(["", footer])
    return "\n".join(lines)


def _matrix_nbytes(M) -> int:
    """Bytes for main numerical storage of a dense array or sparse matrix."""
    if sp.issparse(M):
        x = M
        nb = int(x.data.nbytes)
        fmt = x.format
        if fmt in ("csr", "csc", "bsr"):
            nb += int(x.indices.nbytes) + int(x.indptr.nbytes)
        elif fmt == "coo":
            nb += int(x.row.nbytes) + int(x.col.nbytes)
        elif hasattr(x, "indices") and hasattr(x, "indptr"):
            nb += int(x.indices.nbytes) + int(x.indptr.nbytes)
        return nb
    a = np.asanyarray(M)
    return int(a.nbytes)


def _shape_kind(M) -> str:
    """Human-readable shape/dtype; must not assume ``X`` is 2D (backed / odd layouts can be 1D)."""
    if sp.issparse(M):
        sh = getattr(M, "shape", ())
        if len(sh) >= 2:
            return f"{sh[0]:,}×{sh[1]:,} {M.format} {M.dtype}"
        return f"shape{sh!r} {getattr(M, 'format', '?')} {getattr(M, 'dtype', '?')}"
    a = np.asanyarray(M)
    sh = a.shape
    if a.ndim == 0:
        return f"scalar {a.dtype}"
    if a.ndim == 1:
        return f"{sh[0]:,} vector {a.dtype}"
    if a.ndim == 2:
        return f"{sh[0]:,}×{sh[1]:,} {a.dtype}"
    return f"{sh!r} {a.dtype}"


def _fmt_bytes(n: int) -> str:
    if n >= 1 << 30:
        return f"{n / (1 << 30):.2f} GiB"
    if n >= 1 << 20:
        return f"{n / (1 << 20):.1f} MiB"
    if n >= 1 << 10:
        return f"{n / (1 << 10):.1f} KiB"
    return f"{n} B"


def _adata_buffer_parts(adata: ad.AnnData) -> List[Tuple[str, int, str]]:
    """Major numerical buffers for RAM sizing (same basis as the Status panel breakdown)."""
    parts: List[Tuple[str, int, str]] = []
    try:
        xb = _matrix_nbytes(adata.X)
    except Exception:
        xb = 0
    parts.append(("X", xb, _shape_kind(adata.X)))
    if adata.raw is not None:
        rb = _matrix_nbytes(adata.raw.X)
        parts.append(("raw.X", rb, _shape_kind(adata.raw.X)))
    for k in adata.layers.keys():
        lay = adata.layers[k]
        parts.append((f"layer:{k}", _matrix_nbytes(lay), _shape_kind(lay)))
    for k in adata.obsm.keys():
        arr = np.asarray(adata.obsm[k])
        nb = int(arr.nbytes)
        sk = f"{arr.shape[0]:,}×{arr.shape[1] if arr.ndim > 1 else 1} {arr.dtype}"
        parts.append((f"obsm:{k}", nb, sk))
    for k in adata.obsp.keys():
        g = adata.obsp[k]
        if sp.issparse(g):
            parts.append((f"obsp:{k}", _matrix_nbytes(g), _shape_kind(g)))
        else:
            arr = np.asarray(g)
            parts.append((f"obsp:{k}", int(arr.nbytes), f"{arr.shape} {arr.dtype}"))
    try:
        om = int(adata.obs.memory_usage(deep=True).sum())
        parts.append(("obs", om, f"{len(adata.obs.columns)} cols"))
    except Exception:
        pass
    try:
        vm = int(adata.var.memory_usage(deep=True).sum())
        parts.append(("var", vm, f"{len(adata.var.columns)} cols"))
    except Exception:
        pass
    return parts


def _compute_ram_plan_lines(
    adata: ad.AnnData,
    *,
    peak_mult: float = 2.0,
    slurm_extra: float = 1.25,
    extras: Optional[List[str]] = None,
) -> List[str]:
    """Plain lines written to ``compute_ram_plan.log`` for node sizing."""
    parts = _adata_buffer_parts(adata)
    total = sum(p[1] for p in parts)
    peak = int(total * peak_mult)
    suggest = int(peak * slurm_extra)
    lines = [
        f"ann_data_mode: {'backed_mmap' if adata_is_backed(adata) else 'dense_in_ram'}",
        f"n_obs: {int(adata.n_obs):,}  n_vars: {int(adata.n_vars):,}",
        f"main_array_bytes (X, raw, layers, obsm/obsp, obs/var table): {total}",
        f"main_array_gib: {total / (1 << 30):.4f}",
        (
            f"rough_peak_compute_gib (×{peak_mult:g} heuristic for copies / scale / PCA / densify): "
            f"{peak / (1 << 30):.4f}"
        ),
        (
            f"suggest_min_node_mem_gib (×{slurm_extra:g} headroom on peak; map to Slurm --mem, GB): "
            f"{suggest / (1 << 30):.4f}"
        ),
        (
            "models_note: scVI / scGPT / Geneformer use extra host RAM (DataLoader) and GPU VRAM; "
            "if only CPU RAM is listed above, add GPU memory separately for CUDA workloads."
        ),
        "top_components:",
    ]
    for name, nb, meta in sorted(parts, key=lambda x: -x[1])[:16]:
        lines.append(f"  - {name} ({meta}): {_fmt_bytes(nb)}")
    if len(parts) > 16:
        lines.append(f"  - … {len(parts) - 16} more")
    if extras:
        lines.append("")
        lines.extend(extras)
    return lines


def _try_append_compute_ram_log(
    session_dir: Optional[str],
    title: str,
    lines: List[str],
) -> None:
    if session_dir is None or not str(session_dir).strip():
        return
    try:
        sess_res.append_compute_ram_log(str(session_dir).strip(), title, lines)
    except Exception:
        pass


def _matrix_var_frame_for_spec(adata: ad.AnnData, spec: str) -> pd.DataFrame:
    if spec == "raw.X":
        if adata.raw is None:
            raise ValueError(".raw is not present in this AnnData")
        return adata.raw.var.copy()
    if spec.startswith("obsm:"):
        raise ValueError(
            "scFM embedding expects a gene-expression matrix (X, raw.X, or layer:*), not obsm."
        )
    return adata.var.copy()


def _build_embedding_input_adata(adata: ad.AnnData, matrix_spec: str) -> ad.AnnData:
    M = _matrix_from_spec(adata, matrix_spec)
    return ad.AnnData(
        X=M,
        obs=adata.obs.copy(),
        var=_matrix_var_frame_for_spec(adata, matrix_spec),
    )


def _list_embedding_matrix_options(adata: ad.AnnData) -> List[str]:
    opts = ["X"]
    if adata.raw is not None:
        opts.append("raw.X")
    opts.extend([f"layer:{k}" for k in adata.layers.keys()])
    return opts


def scfm_embed_matrix_guide(model: str, matrix_spec: str = "") -> str:
    """Short Markdown for the embeddings UI: which ``matrix_spec`` to use per model."""
    m = (model or "geneformer").strip().lower()
    spec = (matrix_spec or "").strip()
    hdr = (
        "**Matrix dropdown (`X`, `raw.X`, `layer:…`):** "
        "Use **`raw.X`** when counts live in ``adata.raw`` and ``adata.X`` is normalized/scaled. "
        "Use **`X`** when the matrix you want is already in ``adata.X``. "
        "The embedder takes that slice **as-is** (no hidden log-normalization).\n\n"
    )
    per = {
        "geneformer": (
            "**Geneformer:** Non-negative **count-like** values so per-cell **ranking** of expressed genes matches pretraining. "
            "Prefer **`raw.X`** for UMIs. **Do not** log-normalize before this step. "
            "``var_names`` should match the tokenizer (often **Ensembl**)."
        ),
        "transcriptformer": (
            "**Transcriptformer:** Same idea as Geneformer — **counts / raw-like** matrix, rank tokenization; match HF gene IDs."
        ),
        "scgpt": (
            "**scGPT:** This repo rank-tokenizes nonzeros like Geneformer — **`raw.X`** counts are fine. "
            "Gene symbols/IDs must match the checkpoint **vocab.json**."
        ),
        "scvi": (
            "**scVI:** Fits a VAE on **your** matrix; **integer UMI counts** are ideal → **`raw.X`** when it holds counts. "
            "Running on log-normalized ``X`` is allowed mechanically but is not the count likelihood setup."
        ),
    }
    body = per.get(m, "**Model:** see repository docs for input expectations.")
    foot = (
        "\n\n**`raw.X` auto-handling:** If you select **`raw.X`**, the code copies that matrix into a temporary "
        "``AnnData`` for the model — no extra preprocessing is applied beyond what each embedder already does."
    )
    if spec:
        foot += f"\n\n*Current selection:* `{spec}`."
    return hdr + body + foot


def _format_hms_from_hours(hours: float) -> str:
    total = max(1, int(np.ceil(float(hours) * 3600.0)))
    hh, rem = divmod(total, 3600)
    mm, ss = divmod(rem, 60)
    return f"{hh:02d}:{mm:02d}:{ss:02d}"


def estimate_scfm_slurm_resources(
    adata: ad.AnnData,
    *,
    model: str,
    matrix_spec: str,
    n_latent_scvi: int = 64,
) -> Dict[str, Any]:
    model_key = str(model or "geneformer").strip().lower()
    _ = _matrix_var_frame_for_spec(adata, matrix_spec)
    M = _matrix_from_spec(adata, matrix_spec)
    total_main = sum(nb for _, nb, _meta in _adata_buffer_parts(adata))
    matrix_bytes = _matrix_nbytes(M)
    n_obs = int(adata.n_obs)

    embed_dim = {
        "geneformer": 512,
        "transcriptformer": 768,
        "scgpt": 512,
        "scvi": max(2, int(n_latent_scvi)),
    }.get(model_key, 256)
    embed_bytes = n_obs * embed_dim * np.dtype(np.float32).itemsize

    model_host_margin_gib = {
        "geneformer": 4.0,
        "transcriptformer": 5.0,
        "scgpt": 3.0,
        "scvi": 6.0,
    }.get(model_key, 3.0)
    peak_factor = {
        "geneformer": 2.0,
        "transcriptformer": 2.0,
        "scgpt": 1.8,
        "scvi": 2.3,
    }.get(model_key, 2.0)
    host_peak = (
        max(total_main, matrix_bytes) * peak_factor
        + embed_bytes * 1.25
        + int(model_host_margin_gib * (1 << 30))
    )
    mem_gib = max(8, int(np.ceil(host_peak * 1.15 / float(1 << 30))))

    cpus = {
        "geneformer": 4,
        "transcriptformer": 4,
        "scgpt": 4,
        "scvi": 8,
    }.get(model_key, 4)

    hours = {
        "geneformer": 4.0 + n_obs / 80_000.0,
        "transcriptformer": 4.0 + n_obs / 70_000.0,
        "scgpt": 3.0 + n_obs / 120_000.0,
        "scvi": 2.5 + n_obs / 180_000.0,
    }.get(model_key, 4.0)
    hours = min(72.0, max(2.0, hours))

    return {
        "cpus": int(cpus),
        "mem_gib": int(mem_gib),
        "mem": f"{int(mem_gib)}G",
        "time": _format_hms_from_hours(hours),
        "notes": [
            f"matrix_spec: {matrix_spec}",
            f"matrix_bytes: {matrix_bytes}",
            f"main_arrays_bytes: {total_main}",
            f"embedding_dim_estimate: {embed_dim}",
            f"model_host_margin_gib: {model_host_margin_gib}",
        ],
    }


def _parse_mem_gib(raw: Any) -> Optional[int]:
    s = str(raw or "").strip().upper()
    if not s:
        return None
    mult = 1.0
    if s.endswith("G"):
        s = s[:-1]
    elif s.endswith("GB"):
        s = s[:-2]
    elif s.endswith("M"):
        mult = 1.0 / 1024.0
        s = s[:-1]
    elif s.endswith("MB"):
        mult = 1.0 / 1024.0
        s = s[:-2]
    try:
        return max(1, int(np.ceil(float(s) * mult)))
    except ValueError:
        return None


def estimate_adata_memory_report(adata: ad.AnnData) -> str:
    """
    Rough **in-RAM** footprint of major ``AnnData`` buffers (not necessarily ``.h5ad`` file size).
    Sparse counts only stored nonzeros; peak compute RAM is often higher (copies, densify).
    """
    lines: List[str] = []
    if adata_is_backed(adata):
        lines.append(
            "**Backed .h5ad:** `X` stays on disk (mmap). You can use **UMAP / obsm** coloring and light QC while "
            "**Load dense in background** runs in another process."
        )
        lines.append("")
    parts = _adata_buffer_parts(adata)

    total = sum(p[1] for p in parts)
    mult = 2.0
    rough_peak = int(total * mult)

    lines.append(
        f"**Estimated in-RAM (main data):** {_fmt_bytes(total)} total  "
        f"(~{_fmt_bytes(rough_peak)} **peak rough guide** at ~{mult:.0f}× for copies / intermediates)"
    )
    lines.append("")
    for name, nb, meta in sorted(parts, key=lambda x: -x[1])[:14]:
        lines.append(f"  • **{name}** ({meta}): {_fmt_bytes(nb)}")
    if len(parts) > 14:
        lines.append(f"  • … {len(parts) - 14} more line(s) omitted")
    lines.append("")
    lines.append(
        "Sparse `X` uses less RAM than dense `n_obs×n_ops×dtype`; steps that densify or duplicate "
        "(PCA, scale, model embeddings) can spike much higher."
    )
    return "\n".join(lines)


def _pack_session_outputs(
    adata: ad.AnnData,
    status_msg: str,
    step_log: str = "",
    job_id_u: Optional[object] = None,
    session_dir: Optional[str] = None,
):
    """Gradio return bundle after a successful ``AnnData`` load or job completion."""
    status_msg = f"{status_msg}\n\n{estimate_adata_memory_report(adata)}"
    matrix_opts = _list_matrix_options(adata)
    oc = list(adata.obs.columns)
    vc = list(adata.var.columns)
    lk = list(adata.layers.keys())
    ok = list(adata.obsm.keys())
    umap_choices = ["(none)"] + oc
    default_mat = _pick_matrix_source_value("X", matrix_opts)
    mat_update = gr.update(choices=matrix_opts, value=default_mat)
    emb_opts = _list_embedding_matrix_options(adata)
    emb_update = gr.update(
        choices=emb_opts,
        value=_pick_matrix_source_value("X", emb_opts),
    )
    jid = gr.skip() if job_id_u is None else job_id_u
    if session_dir:
        sess = str(Path(session_dir).resolve())
        sess_disp = gr.update(value=sess)
    else:
        sess = None
        sess_disp = gr.skip()
    popts = _list_plot_source_options(adata)
    pv = _pick_matrix_source_value("X", popts)
    dist_src_up = gr.update(choices=popts, value=pv)
    dist_by_obs_up = gr.update(choices=["(none)"] + oc, value="(none)")
    obsm_prev_up = gr.update(choices=["(none)"] + ok, value="(none)")
    quick_umap_c_up = gr.update(choices=umap_choices, value="(none)")
    batch_key_up = gr.update(choices=["(none)"] + oc, value="(none)")
    return (
        adata,
        status_msg,
        mat_update,
        gr.update(choices=oc, value=oc),
        gr.update(choices=vc, value=vc),
        gr.update(choices=lk, value=lk),
        gr.update(choices=ok, value=ok),
        gr.update(choices=umap_choices, value="(none)"),
        emb_update,
        step_log,
        jid,
        sess,
        sess_disp,
        dist_src_up,
        dist_by_obs_up,
        obsm_prev_up,
        quick_umap_c_up,
        batch_key_up,
    )


def _pick_session_folder_h5ad(d: Path) -> Optional[Path]:
    """Prefer explicit save, then dense materialization, then a copied job result."""
    for name in ("adata_latest.h5ad", "adata_dense_materialized.h5ad", "result.h5ad"):
        cand = d / name
        if cand.is_file():
            return cand
    return None


def _job_folder_result_h5ad(job_dir: Path) -> Tuple[Optional[Path], Dict[str, Any]]:
    """If *job_dir* has ``meta.json`` for a finished job, return result ``.h5ad`` path and meta dict."""
    meta_path = job_dir / "meta.json"
    if not meta_path.is_file():
        return None, {}
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, {}
    st = meta.get("status", "")
    if st != "done":
        return None, meta
    rp = meta.get("result_path")
    if rp:
        pp = Path(str(rp)).expanduser().resolve()
        if pp.is_file():
            return pp, meta
    r2 = job_dir / "result.h5ad"
    if r2.is_file():
        return r2, meta
    return None, meta


def _matrix_from_spec(adata: ad.AnnData, spec: str) -> sp.spmatrix | np.ndarray:
    if spec == "X":
        M = adata.X
    elif spec == "raw.X":
        if adata.raw is None:
            raise ValueError(".raw is not present in this AnnData")
        M = adata.raw.X
    elif spec.startswith("layer:"):
        key = spec.split(":", 1)[1]
        if key not in adata.layers:
            raise ValueError(f"Layer '{key}' not found in adata.layers")
        M = adata.layers[key]
    elif spec.startswith("obsm:"):
        key = spec.split(":", 1)[1]
        if key not in adata.obsm:
            raise ValueError(
                f"obsm key '{key}' not found; have {list(adata.obsm.keys())}"
            )
        M = np.asarray(adata.obsm[key])
    else:
        raise ValueError(f"Unknown matrix spec: {spec}")
    return M


def _list_matrix_options(adata: ad.AnnData) -> List[str]:
    """All matrix sources for radios: ``X``, ``raw.X``, ``layer:*``, ``obsm:*``."""
    opts = ["X"]
    if adata.raw is not None:
        opts.append("raw.X")
    opts.extend([f"layer:{k}" for k in adata.layers.keys()])
    opts.extend([f"obsm:{k}" for k in adata.obsm.keys()])
    return opts


def _pick_matrix_source_value(
    current: Optional[str], options: List[str]
) -> Optional[str]:
    """Keep the current radio selection if still valid; otherwise first choice."""
    if not options:
        return None
    c = (current or "").strip() if current is not None else ""
    if c in options:
        return c
    return options[0]


def _list_plot_source_options(adata: ad.AnnData) -> List[str]:
    """Same as matrix options (QC / Quick UMAP / apply use the same list)."""
    return _list_matrix_options(adata)


def _matrix_or_obsm_from_spec(adata: ad.AnnData, spec: str) -> sp.spmatrix | np.ndarray:
    if spec.startswith("obsm:"):
        key = spec.split(":", 1)[1]
        if key not in adata.obsm:
            raise ValueError(
                f"obsm key '{key}' not found; have {list(adata.obsm.keys())}"
            )
        return np.asarray(adata.obsm[key])
    return _matrix_from_spec(adata, spec)


def pipeline_base_adata(adata: ad.AnnData, use_raw: bool) -> ad.AnnData:
    """Working copy for Scanpy pipeline: either current ``X`` or ``raw`` counts matrix."""
    if not use_raw:
        return adata.copy()
    if adata.raw is None:
        raise ValueError(
            "Pipeline on raw was requested but AnnData has no .raw (load counts or disable the toggle)."
        )
    Xr = adata.raw.X
    X_copy = Xr.copy() if hasattr(Xr, "copy") else np.asarray(Xr)
    out = ad.AnnData(
        X=X_copy,
        obs=adata.obs.copy(),
        var=adata.raw.var.copy(),
    )
    if adata.raw.obs is not None and len(adata.raw.obs.columns):
        for c in adata.raw.obs.columns:
            if c not in out.obs.columns:
                out.obs[c] = adata.raw.obs[c].values
    for k, v in adata.obsm.items():
        arr = np.asarray(v)
        if arr.shape[0] == out.n_obs:
            out.obsm[k] = arr.copy()
    try:
        out.raw = adata.raw.copy()
    except Exception:
        out.raw = adata.raw
    for k, v in adata.layers.items():
        try:
            lay = v
            if hasattr(lay, "shape") and lay.shape == out.shape:
                out.layers[k] = lay.copy() if hasattr(lay, "copy") else np.asarray(lay)
        except Exception:
            continue
    return out


def _coerce_sparse_matrix_for_reduction(M: Any) -> Any:
    """Backed ``.h5ad`` ``X`` is often a disk-backed CSR/CSC dataset, not a ``scipy.sparse`` type."""
    if sp.issparse(M):
        return M
    if _DISK_SPARSE_MATRIX_TYPES and isinstance(M, _DISK_SPARSE_MATRIX_TYPES):
        return M.to_memory()
    return M


def _reduce_matrix_for_quick_umap(
    M: sp.spmatrix | np.ndarray, n_pcs_requested: int
) -> np.ndarray:
    """Reduce expression matrix to a moderate number of dimensions for sklearn UMAP."""
    max_auto = 50
    M = _coerce_sparse_matrix_for_reduction(M)
    if sp.issparse(M):
        n_samples, n_features = M.shape
        if n_features <= 2:
            return np.asarray(M.toarray(), dtype=np.float32)
        if n_pcs_requested > 0:
            n_comp = max(
                2,
                min(n_pcs_requested, n_features - 1, max(1, n_samples - 1)),
            )
        else:
            n_comp = max(
                2,
                min(max_auto, n_features - 1, max(1, n_samples - 1)),
            )
        svd = TruncatedSVD(n_components=n_comp, random_state=0)
        return np.asarray(svd.fit_transform(M.astype(np.float32)), dtype=np.float32)
    X = np.asarray(M, dtype=np.float64)
    n_samples, n_features = X.shape
    if n_features <= 2:
        return X.astype(np.float32)
    if n_pcs_requested > 0:
        n_comp = max(2, min(n_pcs_requested, n_features - 1, max(1, n_samples - 1)))
    else:
        n_comp = max(2, min(max_auto, n_features - 1, max(1, n_samples - 1)))
    return np.asarray(
        PCA(n_components=n_comp, random_state=0).fit_transform(X), dtype=np.float32
    )


def _import_umap():
    try:
        from umap import UMAP
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Install `umap-learn` for UMAP (e.g. `pip install umap-learn`)."
        ) from e
    return UMAP


def embedding_storage_keys_from_matrix_spec(spec: str) -> Tuple[str, str]:
    """Return ``(pca_key, umap_key)`` for embeddings derived from *matrix_spec*."""
    s = (spec or "X").strip() or "X"
    if s == "X":
        return "X_pca", "X_umap"
    if s == "raw.X":
        return "X_raw_pca", "X_raw_umap"
    if s.startswith("layer:"):
        k = s.split(":", 1)[1].strip()
        k = re.sub(r"[^a-zA-Z0-9_]+", "_", k)[:64]
        return f"X_layer_{k}_pca", f"X_layer_{k}_umap"
    if s.startswith("obsm:"):
        k = s.split(":", 1)[1].strip()
        k = re.sub(r"[^a-zA-Z0-9_]+", "_", k)[:64]
        return f"X_{k}_pca", f"X_{k}_umap"
    tok = re.sub(r"[^a-zA-Z0-9_]+", "_", s)[:64]
    return f"X_{tok}_pca", f"X_{tok}_umap"


def _prepare_view_work_copy(adata: ad.AnnData, max_cells: int) -> ad.AnnData:
    mc = int(max_cells) if max_cells is not None else 0
    if mc > 0 and adata.n_obs > mc:
        rng = np.random.default_rng(42)
        ix = np.sort(rng.choice(adata.n_obs, size=mc, replace=False))
        sub = adata[ix]
        if adata_is_backed(adata):
            return sub.to_memory()
        return sub.copy()
    return adata.copy()


def _compute_z_umap_view(
    work: ad.AnnData,
    spec: str,
    n_pcs: int,
    n_neighbors: int,
    min_dist: float,
) -> Tuple[np.ndarray, np.ndarray, ad.AnnData, int]:
    """Return ``Z, U, view_adata, n_neighbors_used``."""
    npc = int(n_pcs) if n_pcs is not None else 0
    spec = str(spec or "X").strip() or "X"
    if spec.startswith("obsm:"):
        key = spec.split(":", 1)[1]
        if key not in work.obsm:
            raise ValueError(f"obsm key '{key}' not found; have {list(work.obsm.keys())}")
        E = np.asarray(work.obsm[key], dtype=np.float32)
        if E.ndim != 2:
            raise ValueError(f"obsm[{key!r}] must be 2D, got shape {E.shape}")
        n_feat = int(E.shape[1])
        var_df = pd.DataFrame(index=[f"{key}_{i}" for i in range(n_feat)])
        view = ad.AnnData(E.copy(), obs=work.obs.copy(), var=var_df)
        view.obsm[key] = E.copy()
        if E.shape[1] < 2:
            Z = np.column_stack(
                [E, np.zeros((E.shape[0], 2 - E.shape[1]), dtype=np.float32)]
            )
        else:
            Z = _reduce_obsm_for_quick_umap(E, npc)
    else:
        M = _matrix_from_spec(work, spec)
        var_df = _matrix_var_frame_for_spec(work, spec)
        X_copy = M.copy() if hasattr(M, "copy") else np.asarray(M)
        view = ad.AnnData(X=X_copy, obs=work.obs.copy(), var=var_df.copy())
        Z = _reduce_matrix_for_quick_umap(M, npc)

    nn = max(2, int(n_neighbors) if n_neighbors is not None else 15)
    nn = min(nn, max(2, Z.shape[0] - 1))
    UMAP = _import_umap()
    U = np.asarray(
        UMAP(
            n_components=2,
            n_neighbors=nn,
            min_dist=float(min_dist),
            random_state=0,
        ).fit_transform(Z),
        dtype=np.float32,
    )
    return Z, U, view, nn


def _reduce_obsm_for_quick_umap(E: np.ndarray, n_pcs_requested: int) -> np.ndarray:
    """Optional PCA on high-dimensional obsm before UMAP."""
    max_auto = 50
    E = np.asarray(E, dtype=np.float64)
    n_samples, n_features = E.shape
    if n_features < 2:
        raise ValueError("obsm must have at least 2 columns for UMAP.")
    if n_pcs_requested > 0 and n_features > n_pcs_requested:
        n_comp = max(2, min(n_pcs_requested, n_features - 1, max(1, n_samples - 1)))
        return np.asarray(
            PCA(n_components=n_comp, random_state=0).fit_transform(E),
            dtype=np.float32,
        )
    if n_features > max_auto:
        n_comp = max(2, min(max_auto, n_features - 1, max(1, n_samples - 1)))
        return np.asarray(
            PCA(n_components=n_comp, random_state=0).fit_transform(E),
            dtype=np.float32,
        )
    return E.astype(np.float32)


def build_matrix_view_adata(
    adata: ad.AnnData,
    matrix_spec: str,
    *,
    max_cells: int = 0,
    n_pcs: int = 50,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> ad.AnnData:
    """Build an analysis/view AnnData from ``X``, ``raw.X``, ``layer:*`` or ``obsm:*`` with ``X_pca`` and ``X_umap``."""
    spec = str(matrix_spec or "X").strip() or "X"
    work = _prepare_view_work_copy(adata, max_cells)
    Z, U, view, nn = _compute_z_umap_view(work, spec, n_pcs, n_neighbors, min_dist)
    npc = int(n_pcs) if n_pcs is not None else 0
    view.obsm["X_pca"] = np.asarray(Z, dtype=np.float32)
    view.obsm["X_umap"] = U
    view.uns["scfms_view_matrix_spec"] = spec
    view.uns["scfms_view_n_pcs"] = int(npc)
    view.uns["scfms_view_n_neighbors"] = int(nn)
    view.uns["scfms_view_min_dist"] = float(min_dist)
    return view


def compute_matrix_embeddings_adata(
    adata: ad.AnnData,
    matrix_spec: str,
    *,
    max_cells: int = 0,
    n_pcs: int = 50,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> ad.AnnData:
    """
    Same pipeline as :func:`build_matrix_view_adata`, but also writes source-specific keys
    (e.g. ``X_raw_pca`` / ``X_raw_umap``) alongside ``X_pca`` / ``X_umap`` for tools compatibility.
    """
    spec = str(matrix_spec or "X").strip() or "X"
    work = _prepare_view_work_copy(adata, max_cells)
    Z, U, view, nn = _compute_z_umap_view(work, spec, n_pcs, n_neighbors, min_dist)
    pca_k, umap_k = embedding_storage_keys_from_matrix_spec(spec)
    Zf = np.asarray(Z, dtype=np.float32)
    view.obsm["X_pca"] = Zf
    view.obsm["X_umap"] = U.copy()
    view.obsm[pca_k] = Zf.copy()
    view.obsm[umap_k] = U.copy()
    npc = int(n_pcs) if n_pcs is not None else 0
    view.uns["scfms_view_matrix_spec"] = spec
    view.uns["scfms_embed_pca_key"] = pca_k
    view.uns["scfms_embed_umap_key"] = umap_k
    view.uns["scfms_view_n_pcs"] = int(npc)
    view.uns["scfms_view_n_neighbors"] = int(nn)
    view.uns["scfms_view_min_dist"] = float(min_dist)
    return view


# Wider default so categorical legends (Quick UMAP / pipeline UMAP) clip less often.
DEFAULT_UMAP_FIGSIZE: Tuple[float, float] = (10.0, 7.0)


def _coerce_figsize(w: Any, h: Any) -> Tuple[float, float]:
    """Matplotlib figure size in inches; sane bounds for UI / Slurm."""
    try:
        fw = float(w)
        fh = float(h)
    except (TypeError, ValueError):
        return DEFAULT_UMAP_FIGSIZE
    fw = max(1.0, min(40.0, fw))
    fh = max(1.0, min(40.0, fh))
    return (fw, fh)


def _truncate_legend_label(text: Any, n_categories: int) -> str:
    s = str(text)
    lim = 44 if n_categories <= 8 else (30 if n_categories <= 16 else 24)
    if len(s) <= lim:
        return s
    return s[: max(1, lim - 1)] + "…"


def _apply_outside_categorical_legend(
    ax: plt.Axes,
    fig: plt.Figure,
    n_categories: int,
    fig_width_in: float,
) -> None:
    """Multi-column legend outside the axes; `rect` reserves horizontal space."""
    if n_categories <= 0:
        return
    ncol = max(
        1,
        min(
            12,
            int(np.ceil(n_categories / max(3.5, fig_width_in * 0.55))),
        ),
    )
    fontsize = max(5.0, min(8.5, 9.0 - n_categories / 10.0))
    ax.legend(
        bbox_to_anchor=(1.01, 1.0),
        loc="upper left",
        borderaxespad=0.35,
        ncol=ncol,
        fontsize=fontsize,
        frameon=False,
        handlelength=1.0,
        handletextpad=0.4,
        columnspacing=0.55,
        labelspacing=0.22,
    )
    right = max(0.30, min(0.88, 0.90 - 0.038 * ncol - 0.0018 * n_categories))
    fig.tight_layout(rect=(0.06, 0.07, right, 0.96))


def _connectivities_digest(adata: ad.AnnData) -> Optional[str]:
    """Fingerprint Scanpy neighbor connectivities for UMAP cache validation."""
    c = adata.obsp.get("connectivities")
    if c is None:
        return None
    if sp.issparse(c):
        csr = c.tocsr()
        blob = np.concatenate(
            [
                csr.data.astype(np.float64),
                csr.indices.astype(np.int64),
                csr.indptr.astype(np.int64),
            ]
        )
        return hashlib.sha256(blob.tobytes()).hexdigest()
    arr = np.asarray(c, dtype=np.float64)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _obs_index_digest(obs_names: pd.Index) -> str:
    """Stable id for obs rows (handles order and large indices)."""
    h = pd.util.hash_pandas_object(
        pd.Series(np.asarray(obs_names), dtype=object), index=False
    ).to_numpy(dtype=np.uint64)
    return hashlib.sha256(h.tobytes()).hexdigest()


def _umap_pipeline_fingerprint(
    *,
    obs_digest: str,
    n_obs: int,
    n_vars: int,
    pipeline_input_matrix: str,
    pm_cap: Optional[int],
    normalize_total: bool,
    target_sum: float,
    log1p: bool,
    filter_cells_min_counts: Optional[int],
    filter_cells_min_genes: Optional[int],
    filter_genes_min_cells: Optional[int],
    hvg: bool,
    n_top_genes: int,
    hvg_flavor: str,
    subset_hvg: bool,
    scale: bool,
    scale_zero_center: bool,
    scale_max_value: Optional[float],
    n_pcs: int,
    pca_solver: str,
    use_scanpy_pca: bool,
    compute_neighbors: bool,
    neighbor_metric: str,
    n_neighbors: int,
    compute_umap: bool,
    umap_min_dist: float,
    umap_spread: float,
    batch_correction: str = "none",
    batch_key: str = "",
) -> str:
    """Excludes color and plot-only settings; used to reuse X_umap when unchanged."""
    smax = scale_max_value
    if smax is not None and not np.isfinite(float(smax)):
        smax = None
    d = {
        "obs_digest": obs_digest,
        "n_obs": int(n_obs),
        "n_vars": int(n_vars),
        "pipeline_input_matrix": str(pipeline_input_matrix),
        "pm_cap": pm_cap,
        "normalize_total": bool(normalize_total),
        "target_sum": round(float(target_sum), 8),
        "log1p": bool(log1p),
        "filter_cells_min_counts": filter_cells_min_counts,
        "filter_cells_min_genes": filter_cells_min_genes,
        "filter_genes_min_cells": filter_genes_min_cells,
        "hvg": bool(hvg),
        "n_top_genes": int(n_top_genes),
        "hvg_flavor": str(hvg_flavor),
        "subset_hvg": bool(subset_hvg),
        "scale": bool(scale),
        "scale_zero_center": bool(scale_zero_center),
        "scale_max_value": None if smax is None else round(float(smax), 8),
        "n_pcs": int(n_pcs),
        "pca_solver": str(pca_solver),
        "use_scanpy_pca": bool(use_scanpy_pca),
        "compute_neighbors": bool(compute_neighbors),
        "neighbor_metric": str(neighbor_metric),
        "n_neighbors": int(n_neighbors),
        "compute_umap": bool(compute_umap),
        "umap_min_dist": round(float(umap_min_dist), 8),
        "umap_spread": round(float(umap_spread), 8),
        "batch_correction": str(batch_correction),
        "batch_key": str(batch_key),
    }
    return hashlib.sha256(
        json.dumps(d, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _quick_umap_signature(
    spec: str,
    mc: int,
    npc: int,
    nn: int,
    min_dist: float,
    obs_digest: str,
    n_obs: int,
    z_dim: int,
) -> str:
    d = {
        "spec": spec,
        "mc": int(mc),
        "npc": int(npc),
        "nn": int(nn),
        "min_dist": round(float(min_dist), 8),
        "obs_digest": obs_digest,
        "n_obs": int(n_obs),
        "z_dim": int(z_dim),
    }
    return hashlib.sha256(
        json.dumps(d, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _fig_xy_from_obs(
    xy: np.ndarray,
    obs_df: pd.DataFrame,
    color_key: Optional[str],
    title: str,
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """2D scatter with optional obs column coloring (plot-only subsampling)."""
    n_full = xy.shape[0]
    max_plot = sess_res.umap_plot_max_cells()
    plot_idx = np.arange(n_full)
    if n_full > max_plot:
        rng = np.random.default_rng(0)
        plot_idx = np.sort(rng.choice(n_full, size=max_plot, replace=False))
    xyp = xy[plot_idx]
    title_suffix = (
        f" (plot: {xyp.shape[0]:,} / {n_full:,} cells)" if xyp.shape[0] < n_full else ""
    )
    fs = _coerce_figsize(*(figsize if figsize is not None else DEFAULT_UMAP_FIGSIZE))
    fig, ax = plt.subplots(figsize=fs)
    ck = (
        None
        if (not color_key or color_key not in obs_df.columns or color_key == "(none)")
        else str(color_key)
    )
    if ck:
        col = obs_df[ck].iloc[plot_idx]
        if _is_categorical_series(col) or col.nunique() <= 25:
            codes, uniques = pd.factorize(col.astype(str))
            nu = len(uniques)
            cmap = plt.get_cmap("tab20")
            for i, u in enumerate(uniques):
                m = codes == i
                ax.scatter(
                    xyp[m, 0],
                    xyp[m, 1],
                    s=6,
                    alpha=0.75,
                    color=cmap(i % 20),
                    label=_truncate_legend_label(u, nu),
                )
        else:
            scat = ax.scatter(
                xyp[:, 0],
                xyp[:, 1],
                c=np.asarray(col, dtype=float),
                s=6,
                alpha=0.75,
            )
            fig.colorbar(scat, ax=ax, shrink=0.7, label=ck)
    else:
        ax.scatter(xyp[:, 0], xyp[:, 1], s=6, alpha=0.7, color="#495057")
    ax.set_title(title + title_suffix)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    if ck and (_is_categorical_series(col) or col.nunique() <= 25):
        _apply_outside_categorical_legend(ax, fig, len(uniques), fs[0])
    else:
        fig.tight_layout()
    return fig


def _normalize_dist_by_obs_col(raw: Optional[str]) -> Optional[str]:
    c = (raw or "").strip()
    if not c or c.lower() in ("(none)", "none"):
        return None
    return c


def _qc_dist_file_stem(matrix_spec: str, dist_by_obs: Optional[str]) -> str:
    """Filename stem for saved QC distribution PNGs (includes stratify column when set)."""
    safe = matrix_spec.replace("/", "_").replace(":", "_")
    og = _normalize_dist_by_obs_col(dist_by_obs)
    if og:
        suf = (
            og.replace("/", "_")
            .replace(":", "_")
            .replace(" ", "_")
            .replace("%", "")[:48]
        )
        return f"{safe}_by_{suf}"
    return safe


def _compute_distributions(
    adata: ad.AnnData,
    matrix_spec: str,
    sample_cells: Optional[int] = None,
    obs_group_col: Optional[str] = None,
    *,
    max_groups: int = DIST_HIST_MAX_GROUPS,
):
    if sample_cells is None:
        sample_cells = sess_res.dist_sample_cells()
    og = _normalize_dist_by_obs_col(obs_group_col)
    if og is not None and og not in adata.obs.columns:
        raise ValueError(
            f"Stratify column {og!r} is not in `obs` (have {list(adata.obs.columns)[:12]}…)."
        )

    M = _matrix_or_obsm_from_spec(adata, matrix_spec)

    n_obs = int(M.shape[0])
    idx = np.arange(n_obs)
    if sample_cells is not None and n_obs > sample_cells:
        rng0 = np.random.default_rng(0)
        idx = np.sort(rng0.choice(n_obs, size=sample_cells, replace=False))

    Ms = M[idx]
    if sp.issparse(Ms):
        cell_sums = np.asarray(Ms.sum(axis=1)).ravel()
        cell_detected = np.asarray((Ms > 0).sum(axis=1)).ravel()
    else:
        cell_sums = np.asarray(Ms.sum(axis=1)).ravel()
        cell_detected = np.asarray((Ms > 0).sum(axis=1)).ravel()

    short = matrix_spec if len(matrix_spec) <= 40 else matrix_spec[:37] + "..."
    is_obsm = matrix_spec.startswith("obsm:")
    feat_word = "features" if is_obsm else "genes"

    if og is None:
        if sp.issparse(Ms):
            gene_means = np.asarray(Ms.mean(axis=0)).ravel()
            vals = Ms.data
            if vals.shape[0] > 1_000_000:
                vals = vals[:1_000_000]
            x_vals = vals
        else:
            gene_means = np.asarray(Ms.mean(axis=0)).ravel()
            flat = np.asarray(Ms).reshape(-1)
            if flat.shape[0] > 1_000_000:
                flat = flat[:1_000_000]
            x_vals = flat

        figs = []
        fig1, ax1 = plt.subplots(figsize=(5, 3.5))
        ax1.hist(cell_sums, bins=50, color="#4C6EF5", alpha=0.8)
        ax1.set_title(f"Per-cell row sums ({short})")
        ax1.set_xlabel("sum / L1 per cell")
        ax1.set_ylabel("cells")
        fig1.tight_layout()
        figs.append(fig1)

        fig2, ax2 = plt.subplots(figsize=(5, 3.5))
        ax2.hist(cell_detected, bins=50, color="#12B886", alpha=0.8)
        ax2.set_title(f"Per-cell nonzero {feat_word} ({short})")
        ax2.set_xlabel(f"#{feat_word} > 0")
        ax2.set_ylabel("cells")
        fig2.tight_layout()
        figs.append(fig2)

        fig3, ax3 = plt.subplots(figsize=(5, 3.5))
        ax3.hist(gene_means, bins=50, color="#F59F00", alpha=0.8)
        ax3.set_title(f"Per-{feat_word[:-1]} column means ({short})")
        ax3.set_xlabel("mean along cells")
        ax3.set_ylabel(feat_word)
        fig3.tight_layout()
        figs.append(fig3)

        fig4, ax4 = plt.subplots(figsize=(5, 3.5))
        ax4.hist(x_vals, bins=50, color="#E64980", alpha=0.8)
        ax4.set_title(f"Value distribution ({short})")
        ax4.set_xlabel("matrix / embedding values")
        ax4.set_ylabel("frequency")
        fig4.tight_layout()
        figs.append(fig4)

        return figs

    gl = adata.obs[og].where(pd.notna(adata.obs[og]), "NA").astype(str)
    labs_raw = np.asarray(gl.iloc[idx], dtype=object)
    vc_all = pd.Series(labs_raw).value_counts()
    nlevels = int(vc_all.shape[0])
    if nlevels > max_groups:
        top_cats = set(vc_all.index[: max_groups - 1].tolist())
        labs_plot = np.array(
            [x if x in top_cats else "Other" for x in labs_raw],
            dtype=object,
        )
        strat_title = (
            f"{short} | by {og} (top {max_groups - 1} + Other; {nlevels} levels)"
        )
    else:
        labs_plot = labs_raw
        strat_title = f"{short} | by {og} ({nlevels} groups)"

    cats = sorted(np.unique(labs_plot), key=lambda x: (x == "Other", str(x)))
    cmap = plt.get_cmap("tab20")
    ncat = len(cats)

    figs = []
    # Fig 1 — per-cell sums
    fig1, ax1 = plt.subplots(figsize=(7.5, 4.5))
    b1 = np.histogram_bin_edges(cell_sums, bins=50)
    for i, cat in enumerate(cats):
        m = labs_plot == cat
        if not np.any(m):
            continue
        ax1.hist(
            cell_sums[m],
            bins=b1,
            color=cmap(i % 20),
            alpha=0.48,
            label=_truncate_legend_label(cat, ncat),
        )
    ax1.set_title(f"Per-cell row sums — {strat_title}")
    ax1.set_xlabel("sum / L1 per cell")
    ax1.set_ylabel("cells")
    ax1.legend(fontsize=8, framealpha=0.92, loc="best", ncol=2 if ncat > 6 else 1)
    fig1.tight_layout()
    figs.append(fig1)

    # Fig 2 — detected
    fig2, ax2 = plt.subplots(figsize=(7.5, 4.5))
    b2 = np.histogram_bin_edges(cell_detected.astype(float), bins=50)
    for i, cat in enumerate(cats):
        m = labs_plot == cat
        if not np.any(m):
            continue
        ax2.hist(
            cell_detected[m],
            bins=b2,
            color=cmap(i % 20),
            alpha=0.48,
            label=_truncate_legend_label(cat, ncat),
        )
    ax2.set_title(f"Per-cell nonzero {feat_word} — {strat_title}")
    ax2.set_xlabel(f"#{feat_word} > 0")
    ax2.set_ylabel("cells")
    ax2.legend(fontsize=8, framealpha=0.92, loc="best", ncol=2 if ncat > 6 else 1)
    fig2.tight_layout()
    figs.append(fig2)

    # Fig 3 — per-{gene|feature} means within each group
    fig3, ax3 = plt.subplots(figsize=(7.5, 4.5))
    pooled_gm: List[np.ndarray] = []
    for cat in cats:
        m = labs_plot == cat
        if not np.any(m):
            continue
        sub = Ms[m]
        if sp.issparse(sub):
            gm = np.asarray(sub.mean(axis=0)).ravel()
        else:
            gm = np.asarray(sub.mean(axis=0)).ravel()
        pooled_gm.append(gm)
    if pooled_gm:
        b3 = np.histogram_bin_edges(np.concatenate(pooled_gm), bins=50)
    else:
        b3 = np.linspace(0.0, 1.0, 51)
    for i, cat in enumerate(cats):
        m = labs_plot == cat
        if not np.any(m):
            continue
        sub = Ms[m]
        if sp.issparse(sub):
            gm = np.asarray(sub.mean(axis=0)).ravel()
        else:
            gm = np.asarray(sub.mean(axis=0)).ravel()
        ax3.hist(
            gm,
            bins=b3,
            color=cmap(i % 20),
            alpha=0.48,
            label=_truncate_legend_label(cat, ncat),
        )
    ax3.set_title(f"Per-{feat_word[:-1]} column means (within group) — {strat_title}")
    ax3.set_xlabel("mean along cells in group")
    ax3.set_ylabel(feat_word)
    ax3.legend(fontsize=8, framealpha=0.92, loc="best", ncol=2 if ncat > 6 else 1)
    fig3.tight_layout()
    figs.append(fig3)

    # Fig 4 — value samples per group
    rngv = np.random.default_rng(1)
    pooled_v_chunks: List[np.ndarray] = []
    per_cat_vals: List[Tuple[int, str, np.ndarray]] = []
    sparse_ms = sp.issparse(Ms)
    for ci, cat in enumerate(cats):
        m = labs_plot == cat
        if not np.any(m):
            continue
        sub = Ms[m]
        if sparse_ms:
            v = np.asarray(sub.data, dtype=np.float64)
        else:
            v = np.asarray(sub, dtype=np.float64).ravel()
        if v.size > _DIST_VALUE_CAP_PER_GROUP:
            v = rngv.choice(v, _DIST_VALUE_CAP_PER_GROUP, replace=False)
        pooled_v_chunks.append(v)
        per_cat_vals.append((ci, str(cat), v))
    if pooled_v_chunks:
        b4 = np.histogram_bin_edges(np.concatenate(pooled_v_chunks), bins=50)
    else:
        b4 = np.linspace(0.0, 1.0, 51)
    fig4, ax4 = plt.subplots(figsize=(7.5, 4.5))
    for ci, cat, v in per_cat_vals:
        ax4.hist(
            v,
            bins=b4,
            color=cmap(ci % 20),
            alpha=0.48,
            label=_truncate_legend_label(cat, ncat),
        )
    ax4.set_title(
        f"Value distribution (≤{_DIST_VALUE_CAP_PER_GROUP:,} draws / group) — {strat_title}"
    )
    ax4.set_xlabel("matrix / embedding values")
    ax4.set_ylabel("frequency")
    ax4.legend(fontsize=8, framealpha=0.92, loc="best", ncol=2 if ncat > 6 else 1)
    fig4.tight_layout()
    figs.append(fig4)

    return figs


def _parse_renames(s: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if not s:
        return mapping
    parts = [p.strip() for p in s.split(",") if p.strip()]
    for p in parts:
        if "=" in p:
            old, new = p.split("=", 1)
            old = old.strip()
            new = new.strip()
            if old and new:
                mapping[old] = new
    return mapping


def _apply_edits(
    adata: ad.AnnData,
    use_matrix: str,
    keep_obs: List[str],
    obs_renames: Dict[str, str],
    keep_var: List[str],
    var_renames: Dict[str, str],
    keep_layers: List[str],
    keep_obsm: List[str],
    keep_raw: bool,
    set_raw_from_current_x: bool,
) -> ad.AnnData:
    adata = adata.copy()

    # Set ``X`` from the selected matrix. obsm-backed ``X`` changes ``n_vars``; rebuild ``AnnData``
    # so ``var`` and ``X`` stay consistent (in-place assignment would violate shape checks).
    use_m = str(use_matrix).strip()
    M = _matrix_from_spec(adata, use_m)
    if use_m.startswith("obsm:"):
        ok = use_m.split(":", 1)[1]
        xnew = M.copy() if hasattr(M, "copy") else np.asarray(M)
        n_f = int(
            xnew.shape[1] if hasattr(xnew, "shape") else int(np.asarray(xnew).shape[1])
        )
        var_df = pd.DataFrame(index=[f"obsm_{ok}_{i}" for i in range(n_f)])
        nx = ad.AnnData(xnew, obs=adata.obs.copy(), var=var_df)
        try:
            nx.uns = dict(adata.uns)
        except Exception:
            nx.uns = adata.uns.copy()
        nx.obsp = adata.obsp.copy()
        nx.obsm = {
            k: (v.copy() if hasattr(v, "copy") else np.asarray(v))
            for k, v in adata.obsm.items()
        }
        adata = nx
    else:
        adata.X = M.copy() if hasattr(M, "copy") else np.array(M)

    # Handle raw
    if keep_raw:
        if set_raw_from_current_x:
            adata.raw = ad.AnnData(adata.X.copy(), var=adata.var.copy())
    else:
        adata.raw = None

    # Filter and rename obs
    if keep_obs:
        adata.obs = adata.obs[keep_obs].copy()
    if obs_renames:
        cols = adata.obs.columns.tolist()
        new_cols = [obs_renames.get(c, c) for c in cols]
        adata.obs.columns = new_cols

    # Filter and rename var (intersection only — e.g. after switching X to an obsm embedding,
    # checkboxes may still list old gene names)
    if keep_var:
        cols = [c for c in keep_var if c in adata.var.columns]
        if cols:
            adata.var = adata.var[cols].copy()
    if var_renames:
        cols = adata.var.columns.tolist()
        new_cols = [var_renames.get(c, c) for c in cols]
        adata.var.columns = new_cols

    # Keep selected layers only
    if keep_layers is not None:
        layers_to_drop = [k for k in adata.layers.keys() if k not in keep_layers]
        for k in layers_to_drop:
            del adata.layers[k]

    # Keep selected obsm only
    if keep_obsm is not None:
        obsm_to_drop = [k for k in adata.obsm.keys() if k not in keep_obsm]
        for k in obsm_to_drop:
            del adata.obsm[k]

    return adata


def _is_categorical_series(s: pd.Series) -> bool:
    return isinstance(s.dtype, pd.CategoricalDtype) or s.dtype == object


def _fig_pca_variance(adata: ad.AnnData, n_show: int = 30) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    if "pca" not in adata.uns or "variance_ratio" not in adata.uns["pca"]:
        ax.text(0.5, 0.5, "No PCA in AnnData.uns['pca']", ha="center", va="center")
        ax.set_axis_off()
        return fig
    vr = np.asarray(adata.uns["pca"]["variance_ratio"], dtype=float)
    k = min(n_show, vr.shape[0])
    xs = np.arange(1, k + 1)
    ax.bar(xs, vr[:k], color="#4C6EF5", alpha=0.85)
    ax.set_xlabel("PC")
    ax.set_ylabel("Variance ratio")
    ax.set_title("PCA variance ratio")
    fig.tight_layout()
    return fig


def _fig_umap(
    adata: ad.AnnData,
    color_key: Optional[str] = None,
    max_plot_cells: Optional[int] = None,
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> Optional[plt.Figure]:
    if "X_umap" not in adata.obsm:
        return None
    xy_full = np.asarray(adata.obsm["X_umap"])
    n_full = xy_full.shape[0]
    if max_plot_cells is None:
        max_plot_cells = sess_res.umap_plot_max_cells()
    plot_idx = np.arange(n_full)
    if n_full > max_plot_cells:
        rng = np.random.default_rng(0)
        plot_idx = np.sort(rng.choice(n_full, size=max_plot_cells, replace=False))
    xy = xy_full[plot_idx]
    title_suffix = (
        f" (plot: {xy.shape[0]:,} / {n_full:,} cells)" if xy.shape[0] < n_full else ""
    )
    fs = _coerce_figsize(*(figsize if figsize is not None else DEFAULT_UMAP_FIGSIZE))
    fig, ax = plt.subplots(figsize=fs)
    col = None
    cat_legend = False
    n_cat = 0
    if color_key and color_key in adata.obs.columns:
        col = adata.obs[color_key].iloc[plot_idx]
        if _is_categorical_series(col) or col.nunique() <= 25:
            codes, uniques = pd.factorize(col.astype(str))
            n_cat = len(uniques)
            cat_legend = True
            cmap = plt.get_cmap("tab20")
            for i, u in enumerate(uniques):
                m = codes == i
                ax.scatter(
                    xy[m, 0],
                    xy[m, 1],
                    s=6,
                    alpha=0.75,
                    color=cmap(i % 20),
                    label=_truncate_legend_label(u, n_cat),
                )
        else:
            scat = ax.scatter(xy[:, 0], xy[:, 1], c=np.asarray(col), s=6, alpha=0.75)
            fig.colorbar(scat, ax=ax, shrink=0.7, label=color_key)
    else:
        ax.scatter(xy[:, 0], xy[:, 1], s=6, alpha=0.7, color="#495057")
    ax.set_title(f"UMAP{title_suffix}")
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    if cat_legend:
        _apply_outside_categorical_legend(ax, fig, n_cat, fs[0])
    else:
        fig.tight_layout()
    return fig


def _fig_obsm_scatter(
    adata: ad.AnnData,
    obsm_key: str,
    color_key: Optional[str] = None,
    max_plot_cells: Optional[int] = None,
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> Optional[plt.Figure]:
    if obsm_key not in adata.obsm:
        return None
    E = np.asarray(adata.obsm[obsm_key], dtype=np.float64)
    if E.ndim == 1:
        E = E.reshape(-1, 1)
    n_full = E.shape[0]
    if max_plot_cells is None:
        max_plot_cells = sess_res.umap_plot_max_cells()
    plot_idx = np.arange(n_full)
    if n_full > max_plot_cells:
        rng = np.random.default_rng(1)
        plot_idx = np.sort(rng.choice(n_full, size=max_plot_cells, replace=False))
    Ep = E[plot_idx]
    fs = _coerce_figsize(*(figsize if figsize is not None else DEFAULT_UMAP_FIGSIZE))
    fig, ax = plt.subplots(figsize=fs)
    title_suffix = (
        f" ({Ep.shape[0]:,} / {n_full:,} cells)" if Ep.shape[0] < n_full else ""
    )
    cat_legend = False
    n_cat = 0
    if E.shape[1] >= 2:
        xy = Ep[:, :2]
        ck = (
            None
            if (
                not color_key
                or color_key not in adata.obs.columns
                or color_key == "(none)"
            )
            else str(color_key)
        )
        if ck:
            col = adata.obs[ck].iloc[plot_idx]
            if _is_categorical_series(col) or col.nunique() <= 25:
                codes, uniques = pd.factorize(col.astype(str))
                n_cat = len(uniques)
                cat_legend = True
                cmap = plt.get_cmap("tab20")
                for i, u in enumerate(uniques):
                    m = codes == i
                    ax.scatter(
                        xy[m, 0],
                        xy[m, 1],
                        s=6,
                        alpha=0.75,
                        color=cmap(i % 20),
                        label=_truncate_legend_label(u, n_cat),
                    )
            else:
                scat = ax.scatter(
                    xy[:, 0], xy[:, 1], c=np.asarray(col), s=6, alpha=0.75
                )
                fig.colorbar(scat, ax=ax, shrink=0.7, label=ck)
        else:
            ax.scatter(xy[:, 0], xy[:, 1], s=6, alpha=0.7, color="#495057")
        ax.set_xlabel(f"{obsm_key} dim 0")
        ax.set_ylabel(f"{obsm_key} dim 1")
        ax.set_title(f"{obsm_key} scatter{title_suffix}")
    else:
        ax.hist(Ep.ravel(), bins=50, color="#4C6EF5", alpha=0.85)
        ax.set_title(f"{obsm_key} values (1D){title_suffix}")
        ax.set_xlabel("value")
        ax.set_ylabel("count")
    if cat_legend:
        _apply_outside_categorical_legend(ax, fig, n_cat, fs[0])
    else:
        fig.tight_layout()
    return fig


def _scanorama_integrate_reordered(
    adata: ad.AnnData,
    key: str,
    *,
    basis: str = "X_pca",
    adjusted_basis: str = "X_scanorama",
) -> None:
    """
    ``scanpy.external.pp.scanorama_integrate`` requires cells from the same batch
    to be **contiguous** in ``adata``. Sort by batch, integrate, then map rows back.
    """
    import scanpy.external as sce

    n = adata.n_obs
    if n == 0:
        return
    batch = adata.obs[key].astype(str).to_numpy()
    perm = np.argsort(batch, kind="mergesort")
    inv = np.empty(n, dtype=np.int64)
    inv[perm] = np.arange(n, dtype=np.int64)
    sub = adata[perm].copy()
    sce.pp.scanorama_integrate(sub, key, basis=basis, adjusted_basis=adjusted_basis)
    E = np.asarray(sub.obsm[adjusted_basis], dtype=np.float32)
    adata.obsm[adjusted_basis] = E[inv]


def run_expression_pipeline(
    adata: ad.AnnData,
    *,
    normalize_total: bool,
    target_sum: float,
    log1p: bool,
    filter_cells_min_counts: Optional[int],
    filter_cells_min_genes: Optional[int],
    filter_genes_min_cells: Optional[int],
    hvg: bool,
    n_top_genes: int,
    hvg_flavor: str,
    subset_hvg: bool,
    scale: bool,
    scale_zero_center: bool,
    scale_max_value: Optional[float],
    n_pcs: int,
    pca_solver: str,
    use_scanpy_pca: bool,
    compute_neighbors: bool,
    neighbor_metric: str,
    n_neighbors: int,
    compute_umap: bool,
    umap_min_dist: float,
    umap_spread: float,
    umap_color_obs: Optional[str] = None,
    pipeline_max_cells: Optional[int] = None,
    umap_plot_max_cells: Optional[int] = None,
    umap_fig_width: float = DEFAULT_UMAP_FIGSIZE[0],
    umap_fig_height: float = DEFAULT_UMAP_FIGSIZE[1],
    pipeline_input_matrix: str = "X",
    batch_correction: str = "none",
    batch_key: Optional[str] = None,
    progress_callback: Optional[
        Callable[[str, int, int, Dict[str, float], float, float], None]
    ] = None,
) -> Tuple[ad.AnnData, Optional[plt.Figure], Optional[plt.Figure], Dict[str, float]]:
    """Scanpy-style workflow on a copy: optional QC, norm/log1p, HVG, scale, PCA, neighbors, UMAP."""
    timings: Dict[str, float] = {}

    def _nz(v: Optional[int]) -> Optional[int]:
        if v is None:
            return None
        try:
            i = int(v)
        except (TypeError, ValueError):
            return None
        return i if i > 0 else None

    fc = _nz(filter_cells_min_counts)
    fgene = _nz(filter_cells_min_genes)
    fgcell = _nz(filter_genes_min_cells)
    n_pcs_i = int(n_pcs)
    need_neighbors = bool(compute_neighbors or compute_umap)
    bc_method = (batch_correction or "none").strip().lower()
    bc_key_use = (batch_key or "").strip()
    if bc_key_use == "(none)":
        bc_key_use = ""
    if bc_method not in ("", "none", "harmony2", "scanorama"):
        raise ValueError(
            f"Unknown batch_correction {batch_correction!r}; use 'none', 'harmony2', or 'scanorama'."
        )
    use_bc = bc_method not in ("", "none") and bool(bc_key_use)
    if bc_method not in ("", "none") and not bc_key_use:
        raise ValueError(
            "Batch effect correction is enabled — choose a **batch** column in `obs` (not “(none)”)."
        )
    if use_bc and n_pcs_i <= 0:
        raise ValueError("Batch effect correction needs **PCA components > 0**.")

    planned: List[str] = []
    if fc or fgene or fgcell:
        planned.append("qc_filters")
    if normalize_total:
        planned.append("normalize_total")
    if log1p:
        planned.append("log1p")
    if hvg:
        planned.append("hvg")
    if hvg and subset_hvg:
        planned.append("subset_hvg")
    if scale:
        planned.append("scale")
    if n_pcs_i > 0:
        planned.append("pca")
    if use_bc:
        if "pca" not in planned:
            raise ValueError("Internal: batch correction without PCA.")
        planned.insert(planned.index("pca") + 1, "batch_correction")
    if need_neighbors:
        planned.append("neighbors")
    if compute_umap:
        planned.append("umap")
    n_plan = max(1, len(planned))
    si = [0]
    wall0 = time.perf_counter()

    def _tick(name: str) -> None:
        si[0] += 1
        if progress_callback:
            wall = time.perf_counter() - wall0
            done = si[0]
            avg = wall / max(1, done)
            eta = max(0.0, avg * (n_plan - done))
            progress_callback(name, done, n_plan, dict(timings), wall, eta)

    @contextmanager
    def _step(name: str):
        with _timed(timings, name):
            yield
        _tick(name)

    pm_cap: Optional[int] = None
    if pipeline_max_cells is not None:
        try:
            pm_cap = int(pipeline_max_cells)
        except (TypeError, ValueError):
            pm_cap = None
        if pm_cap is not None and pm_cap <= 0:
            pm_cap = None

    def _fp_for_adata(a: ad.AnnData) -> str:
        return _umap_pipeline_fingerprint(
            obs_digest=_obs_index_digest(a.obs_names),
            n_obs=a.n_obs,
            n_vars=a.n_vars,
            pipeline_input_matrix=str(pipeline_input_matrix),
            pm_cap=pm_cap,
            normalize_total=bool(normalize_total),
            target_sum=float(target_sum),
            log1p=bool(log1p),
            filter_cells_min_counts=fc,
            filter_cells_min_genes=fgene,
            filter_genes_min_cells=fgcell,
            hvg=bool(hvg),
            n_top_genes=int(n_top_genes),
            hvg_flavor=str(hvg_flavor),
            subset_hvg=bool(subset_hvg),
            scale=bool(scale),
            scale_zero_center=bool(scale_zero_center),
            scale_max_value=scale_max_value,
            n_pcs=int(n_pcs_i),
            pca_solver=str(pca_solver),
            use_scanpy_pca=bool(use_scanpy_pca),
            compute_neighbors=bool(compute_neighbors),
            neighbor_metric=str(neighbor_metric),
            n_neighbors=int(n_neighbors),
            compute_umap=bool(compute_umap),
            umap_min_dist=float(umap_min_dist),
            umap_spread=float(umap_spread),
            batch_correction=bc_method if use_bc else "none",
            batch_key=bc_key_use if use_bc else "",
        )

    if (
        compute_umap
        and planned
        and adata.n_obs > 0
        and adata.uns.get("_scfms_umap_fp") == _fp_for_adata(adata)
        and "X_umap" in adata.obsm
        and np.asarray(adata.obsm["X_umap"]).shape[0] == adata.n_obs
        and np.asarray(adata.obsm["X_umap"]).ndim == 2
        and np.asarray(adata.obsm["X_umap"]).shape[1] >= 2
        and (not need_neighbors or "connectivities" in adata.obsp)
        and (n_pcs_i == 0 or "X_pca" in adata.obsm)
        and adata.uns.get("_scfms_umap_conn_digest") == _connectivities_digest(adata)
    ):
        out = adata
        if n_pcs_i > 0:
            fig_pca = _fig_pca_variance(out)
        else:
            fig_pca = None
        ckey = umap_color_obs if umap_color_obs else None
        if ckey == "":
            ckey = None
        with _timed(timings, "cache_recolor"):
            fig_umap = _fig_umap(
                out,
                ckey,
                umap_plot_max_cells,
                figsize=_coerce_figsize(umap_fig_width, umap_fig_height),
            )
        if progress_callback:
            wall = time.perf_counter() - wall0
            progress_callback(
                "cache_recolor",
                n_plan,
                n_plan,
                dict(timings),
                wall,
                0.0,
            )
        return out, fig_pca, fig_umap, timings

    out = adata.copy()
    if pm_cap is not None and out.n_obs > pm_cap:
        rng = np.random.default_rng(42)
        ix = np.sort(rng.choice(out.n_obs, size=pm_cap, replace=False))
        out = out[ix].copy()
    if not planned:
        if progress_callback:
            wall = time.perf_counter() - wall0
            progress_callback("no_steps", 1, 1, timings, wall, 0.0)
        return out, None, None, timings

    if fc or fgene or fgcell:
        with _step("qc_filters"):
            if fc:
                sc.pp.filter_cells(out, min_counts=int(fc))
            if fgene:
                sc.pp.filter_cells(out, min_genes=int(fgene))
            if fgcell:
                sc.pp.filter_genes(out, min_cells=int(fgcell))

    if normalize_total:
        with _step("normalize_total"):
            sc.pp.normalize_total(out, target_sum=float(target_sum), inplace=True)
    if log1p:
        with _step("log1p"):
            sc.pp.log1p(out)

    if hvg:
        with _step("hvg"):
            nt = min(int(n_top_genes), max(1, out.n_vars - 1))
            try:
                sc.pp.highly_variable_genes(
                    out, flavor=hvg_flavor, n_top_genes=nt, subset=False
                )
            except Exception:
                sc.pp.highly_variable_genes(
                    out, flavor="seurat", n_top_genes=nt, subset=False
                )
        if subset_hvg and "highly_variable" in out.var.columns:
            with _step("subset_hvg"):
                out = out[:, out.var["highly_variable"].to_numpy()].copy()

    if scale:
        with _step("scale"):
            zc = bool(scale_zero_center)
            if (
                zc
                and sp.issparse(out.X)
                and os.environ.get("SCFMS_ALLOW_SPARSE_ZERO_CENTER", "").strip().lower()
                not in ("1", "true", "yes")
            ):
                warnings.warn(
                    "Sparse `X`: zero-center for `sc.pp.scale` was **skipped** to avoid densifying "
                    "the full matrix (large RAM spike; often ends with **Killed** / OOM). "
                    "Scaling still z-scores per gene without subtracting the mean in sparse form. "
                    "Set **`SCFMS_ALLOW_SPARSE_ZERO_CENTER=1`** to force the old behavior if you "
                    "have enough RAM.",
                    UserWarning,
                    stacklevel=2,
                )
                zc = False
            sc.pp.scale(
                out,
                zero_center=zc,
                max_value=scale_max_value,
            )

    neighbor_rep: Optional[str] = None
    fig_pca: Optional[plt.Figure] = None
    if n_pcs_i > 0:
        with _step("pca"):
            n_comp = min(n_pcs_i, max(1, out.n_obs - 1), max(1, out.n_vars - 1))
            if use_scanpy_pca:
                try:
                    sc.tl.pca(out, n_comps=n_comp, svd_solver=pca_solver)
                except TypeError:
                    sc.tl.pca(out, n_comps=n_comp)
            else:
                X = out.X
                if sp.issparse(X):
                    X = X.toarray()
                else:
                    X = np.asarray(X, dtype=np.float64)
                pca = PCA(n_components=n_comp, random_state=0)
                out.obsm["X_pca"] = np.asarray(pca.fit_transform(X), dtype=np.float32)
                out.uns["pca"] = {
                    "variance_ratio": pca.explained_variance_ratio_.astype(np.float64),
                    "variance": pca.explained_variance_.astype(np.float64),
                }
        fig_pca = _fig_pca_variance(out)
        if use_bc:
            if bc_key_use not in out.obs:
                raise ValueError(
                    f"Batch column {bc_key_use!r} not found in adata.obs after preprocessing."
                )
            with _step("batch_correction"):
                import scanpy.external as sce

                if bc_method == "harmony2":
                    try:
                        import harmonypy
                    except ImportError as e:
                        raise ImportError(
                            "Harmony 2 integration needs **harmonypy** "
                            "(``pip install harmonypy``)."
                        ) from e
                    x_pc = out.obsm["X_pca"].astype(np.float64)
                    ho = harmonypy.run_harmony(x_pc, out.obs, bc_key_use, verbose=False)
                    Z = np.asarray(ho.Z_corr, dtype=np.float32)
                    if Z.ndim != 2:
                        raise ValueError("harmonypy returned non-2D Z_corr")
                    # harmonypy ≥0.2 is (n_obs, n_pcs); older builds were (n_pcs, n_obs).
                    if Z.shape[0] != out.n_obs and Z.shape[1] == out.n_obs:
                        Z = Z.T
                    if Z.shape[0] != out.n_obs:
                        raise ValueError(
                            f"Harmony Z_corr shape {Z.shape} incompatible with n_obs={out.n_obs}"
                        )
                    out.obsm["X_pca_harmony"] = Z
                    neighbor_rep = "X_pca_harmony"
                else:
                    try:
                        _scanorama_integrate_reordered(
                            out,
                            bc_key_use,
                            basis="X_pca",
                            adjusted_basis="X_scanorama",
                        )
                    except ImportError as e:
                        raise ImportError(
                            "Scanorama integration needs **scanorama** "
                            "(``pip install scanorama``)."
                        ) from e
                    neighbor_rep = "X_scanorama"

    fig_umap: Optional[plt.Figure] = None
    if need_neighbors:
        with _step("neighbors"):
            if neighbor_rep is not None:
                if neighbor_rep not in out.obsm:
                    raise ValueError(
                        f"Expected obsm['{neighbor_rep}'] after batch correction."
                    )
                sc.pp.neighbors(
                    out,
                    n_neighbors=int(n_neighbors),
                    use_rep=neighbor_rep,
                    metric=neighbor_metric,
                )
            else:
                if "X_pca" not in out.obsm:
                    raise ValueError(
                        "PCA is required for neighbors/UMAP; set PCA components > 0."
                    )
                use_pcs = min(int(out.obsm["X_pca"].shape[1]), n_pcs_i)
                sc.pp.neighbors(
                    out,
                    n_neighbors=int(n_neighbors),
                    n_pcs=use_pcs,
                    metric=neighbor_metric,
                )
        cd = _connectivities_digest(out)
        if cd is not None:
            out.uns["_scfms_neighbors_conn_digest"] = cd
    if compute_umap:
        with _step("umap"):
            sc.tl.umap(out, min_dist=float(umap_min_dist), spread=float(umap_spread))
            ckey = umap_color_obs if umap_color_obs else None
            if ckey == "":
                ckey = None
            fig_umap = _fig_umap(
                out,
                ckey,
                umap_plot_max_cells,
                figsize=_coerce_figsize(umap_fig_width, umap_fig_height),
            )
        out.uns["_scfms_umap_fp"] = _fp_for_adata(out)
        cd_umap = _connectivities_digest(out)
        if cd_umap is not None:
            out.uns["_scfms_umap_conn_digest"] = cd_umap

    return out, fig_pca, fig_umap, timings


def _normalize_obsm_key(model: str, user_key: Optional[str]) -> str:
    """Use Scanpy-style `X_*` keys in obsm (e.g. `X_geneformer`)."""
    s = (user_key or "").strip()
    if not s:
        return f"X_{model}"
    if s.startswith("X_"):
        return s
    return f"X_{s}"


# Registry: name -> (embed_fn, stable order for UI). Extend via ``register_scfm_embedder``.
# Each ``embed_fn(ad_emb, weights_path, n_latent_scvi) -> np.ndarray`` receives the same
# ``AnnData`` slice as the legacy path. Add matching entries in ``scfm_compatibility`` for PDF checks.
ScfmEmbedFn = Callable[[ad.AnnData, Optional[str], int], np.ndarray]
_SCFM_EMBED_ORDER: List[str] = []
_SCFM_EMBED_REGISTRY: Dict[str, ScfmEmbedFn] = {}


def register_scfm_embedder(name: str, fn: ScfmEmbedFn) -> None:
    """Register a foundation-model embedder (or override an existing name)."""
    n = str(name).strip()
    if not n:
        raise ValueError("embedder name must be non-empty")
    if n not in _SCFM_EMBED_REGISTRY:
        _SCFM_EMBED_ORDER.append(n)
    _SCFM_EMBED_REGISTRY[n] = fn


def list_scfm_model_names() -> List[str]:
    """Ordered model keys for Gradio radios and Slurm scripts."""
    return list(_SCFM_EMBED_ORDER)


def _default_embed_geneformer(
    ad_emb: ad.AnnData, weights_path: Optional[str], _n_latent: int
) -> np.ndarray:
    from scripts.generate_embeddings import embed_geneformer

    p = (weights_path or "").strip() or None
    return embed_geneformer(ad_emb, pretrained_name_or_path=p)


def _default_embed_transcriptformer(
    ad_emb: ad.AnnData, weights_path: Optional[str], _n_latent: int
) -> np.ndarray:
    from scripts.generate_embeddings import embed_transcriptformer

    p = (weights_path or "").strip() or None
    return embed_transcriptformer(ad_emb, pretrained_name_or_path=p)


def _default_embed_scgpt(
    ad_emb: ad.AnnData, scgpt_ckpt: Optional[str], _n_latent: int
) -> np.ndarray:
    from scripts.generate_embeddings import embed_scgpt

    ckpt = (scgpt_ckpt or "").strip() or os.environ.get("SCGPT_CKPT_DIR")
    return embed_scgpt(ad_emb, ckpt_dir=ckpt)


def _default_embed_scvi(
    ad_emb: ad.AnnData, _scgpt_ckpt: Optional[str], n_latent: int
) -> np.ndarray:
    from scripts.generate_embeddings import embed_scvi

    return embed_scvi(ad_emb, n_latent=int(n_latent))


def _init_default_scfm_embedders() -> None:
    if _SCFM_EMBED_REGISTRY:
        return
    register_scfm_embedder("geneformer", _default_embed_geneformer)
    register_scfm_embedder("transcriptformer", _default_embed_transcriptformer)
    register_scfm_embedder("scgpt", _default_embed_scgpt)
    register_scfm_embedder("scvi", _default_embed_scvi)


_init_default_scfm_embedders()


def _resolve_scfm_weights_path(model: str, ui_path: Optional[str]) -> Optional[str]:
    """Prefer explicit UI path; else model-specific env default (Slurm JSON uses ``scgpt_ckpt`` for all)."""
    p = (ui_path or "").strip()
    if p:
        return p
    m = str(model or "").strip().lower()
    if m == "geneformer":
        return (os.environ.get("GENEFORMER_MODEL") or "").strip() or None
    if m == "scgpt":
        return (os.environ.get("SCGPT_CKPT_DIR") or "").strip() or None
    if m == "transcriptformer":
        return (os.environ.get("TRANSCRIPTFORMER_MODEL") or "").strip() or None
    return None


def attach_scfm_embedding(
    adata: ad.AnnData,
    model: str,
    matrix_spec: str,
    obsm_key: Optional[str],
    scgpt_ckpt: Optional[str],
    n_latent_scvi: int,
    compat_report_dir: Optional[str] = None,
) -> Tuple[ad.AnnData, str, Dict[str, float]]:
    """Compute registered scFM embeddings and store in ``obsm`` (full n_obs)."""
    import scfm_compatibility as sc_compat

    _init_default_scfm_embedders()
    model = str(model).strip()
    if model not in _SCFM_EMBED_REGISTRY:
        known = ", ".join(list_scfm_model_names()) or "(none)"
        raise ValueError(f"Unknown model: {model}. Registered: {known}")

    timings: Dict[str, float] = {}
    key = _normalize_obsm_key(model, obsm_key)
    ad_emb = _build_embedding_input_adata(adata, matrix_spec)
    strict = os.environ.get("SCFMS_SC_FM_COMPAT_STRICT", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )
    ckpt_res = _resolve_scfm_weights_path(model, scgpt_ckpt)
    report_path: Optional[Path] = None
    crd = (compat_report_dir or "").strip()
    if crd:
        rd = Path(crd).expanduser().resolve()
        rd.mkdir(parents=True, exist_ok=True)
        report_path = rd / (
            f"scfms_compat_{model}_{time.strftime('%Y%m%d_%H%M%S')}.pdf"
        )
    findings, compat_summary, pdf_p = sc_compat.validate_run_and_pdf(
        model,
        ad_emb,
        matrix_spec=matrix_spec,
        scgpt_ckpt=ckpt_res,
        report_path=report_path,
        strict=strict,
    )
    tkey = f"embed_{model}"
    with _timed(timings, tkey):
        E = _SCFM_EMBED_REGISTRY[model](ad_emb, ckpt_res, int(n_latent_scvi))
    E = np.asarray(E, dtype=np.float32)
    if E.shape[0] != adata.n_obs:
        raise ValueError(
            f"Embedding rows {E.shape[0]} != n_obs {adata.n_obs}; model must return one row per cell."
        )
    out = adata.copy()
    out.uns["scfms_validation"] = compat_summary
    out.uns["scfms_compat_report_pdf"] = str(pdf_p.resolve())
    out.uns["scfms_embed_matrix_spec"] = str(matrix_spec)
    out.uns["scfms_embed_model"] = str(model)
    out.obsm[key] = E
    nwarn = sum(1 for f in findings if f.level == "warn")
    tail = f" Compatibility PDF: {pdf_p}"
    if nwarn:
        tail += f" ({nwarn} warning(s); see PDF and uns['scfms_validation'].)"
    return (
        out,
        f"Saved obsm['{key}'] shape {E.shape} ({model}).{tail}",
        timings,
    )


def build_ui():
    def _empty_load_ret(msg: str):
        return (
            None,
            msg,
            gr.update(choices=[], value=None),
            gr.update(choices=[], value=[]),
            gr.update(choices=[], value=[]),
            gr.update(choices=[], value=[]),
            gr.update(choices=[], value=[]),
            gr.update(choices=[], value=None),
            gr.update(choices=[], value=None),
            "",
            gr.skip(),
            None,
            gr.update(value=""),
            gr.update(choices=["X"], value="X"),
            gr.update(choices=["(none)"], value="(none)"),  # stratify QC by obs
            gr.update(choices=["(none)"], value="(none)"),
            gr.update(choices=["(none)"], value="(none)"),
            gr.update(choices=["(none)"], value="(none)"),  # pipeline batch column
        )

    def load_from_server_path(path_str, open_backed):
        raw = str(path_str or "").strip()
        if not raw:
            return _empty_load_ret("Enter an absolute path on the compute server.")
        use_backed = _gradio_bool(open_backed)
        try:
            p = bgjobs.validate_server_read_path(raw)
        except FileNotFoundError as e:
            return _empty_load_ret(
                f"{e}\n\nPath is checked on the **machine running this app** "
                f"(`expanduser` / `expandvars` applied). Must be a regular file."
            )
        except PermissionError as e:
            return _empty_load_ret(
                f"{e}\n\nSet **`SCFMS_ALLOWED_PATH_PREFIXES`** to pipe-separated roots "
                f"(e.g. `/scratch/your_project/|/shared/data/`) that include your file, or ask your admin."
            )
        pstr = os.fspath(p)
        try:
            t0 = time.perf_counter()
            if use_backed:
                adata = ad.read_h5ad(pstr, backed="r")
                adata.uns["_scfms_densify_src"] = pstr
                slog = f"Opened **backed** `.h5ad` (mmap) in {_fmt_dur(time.perf_counter() - t0)}."
                msg = (
                    f"Backed from server: **{p}** — cells={adata.n_obs:,}, genes={adata.n_vars:,}\n\n"
                    f"**Load dense** re-reads this file in a **separate process**. "
                    f"Inspect UMAP meanwhile. Before **subset / apply / pipeline / embeddings**, finish dense load."
                )
            else:
                adata = ad.read_h5ad(pstr)
                dt = time.perf_counter() - t0
                msg = f"Loaded from server: {p} — cells={adata.n_obs}, genes={adata.n_vars}"
                slog = f"Read `.h5ad` from disk in {_fmt_dur(dt)}."
            session = sess_res.create_dataset_session(
                p.stem,
                source_kind="server",
                source_path=pstr,
                n_obs=int(adata.n_obs),
                n_vars=int(adata.n_vars),
            )
            msg = f"{msg}\n\n**Results folder:** `{session}`"
            try:
                fz = int(p.stat().st_size)
            except OSError:
                fz = None
            _ex: List[str] = [
                f"source_h5ad: {pstr}",
                f"ui_open_mode: {'backed' if use_backed else 'dense_full_read'}",
            ]
            if fz is not None:
                _ex.extend(
                    [
                        f"source_file_bytes: {fz}",
                        f"source_file_gib: {fz / (1 << 30):.4f}",
                    ]
                )
            _ex.append(
                f"offline_log: `{session / sess_res.COMPUTE_RAM_LOG}` (RAM GiB hints for preprocessing & embeddings)"
            )
            _try_append_compute_ram_log(
                str(session),
                "h5ad_open_server",
                _compute_ram_plan_lines(adata, extras=_ex),
            )
            return _pack_session_outputs(adata, msg, slog, session_dir=str(session))
        except BaseException as e:
            if is_oom_error(e):
                return _empty_load_ret(oom_user_message(e))
            return _empty_load_ret(
                f"Server load error: {e}\n\n"
                f"If paths look correct, try **`Open backed`** for large files or check file permissions."
            )

    def load_resume_path(path_str, open_backed):
        """Reload after disconnect: dataset session dir, job store folder, or any ``.h5ad`` (reuses **Open backed**)."""
        raw = str(path_str or "").strip()
        if not raw:
            return _empty_load_ret(
                "Paste a **session folder** (has `session_meta.json`), a **job folder** under the job store "
                "(has `meta.json`), or a **`.h5ad`** file path."
            )
        use_backed = _gradio_bool(open_backed)
        try:
            root = bgjobs.validate_allowed_existing_path(raw)
        except FileNotFoundError as e:
            return _empty_load_ret(str(e))
        except PermissionError as e:
            return _empty_load_ret(
                f"{e}\n\nSet **`SCFMS_ALLOWED_PATH_PREFIXES`** to pipe-separated roots that include this path."
            )

        h5ad_file: Optional[Path] = None
        session_dir_str: Optional[str] = None
        job_id_ui: Optional[object] = None
        resumed_job_id: Optional[str] = None
        log_extras: List[str] = []
        resume_kind = ""

        if root.is_file():
            if root.suffix.lower() != ".h5ad":
                return _empty_load_ret(
                    "Path is a file but not `.h5ad`. Paste a directory or an AnnData `.h5ad` path."
                )
            h5ad_file = root
            parent = root.parent
            if (parent / "session_meta.json").is_file():
                session_dir_str = str(parent.resolve())
                resume_kind = "h5ad_in_session_folder"
                log_extras.append(f"reuse_session_dir: {session_dir_str}")
            else:
                resume_kind = "h5ad_standalone"
        else:
            if (root / "session_meta.json").is_file():
                h5ad_file = _pick_session_folder_h5ad(root)
                if h5ad_file is None:
                    return _empty_load_ret(
                        f"No `.h5ad` found under `{root}`. Use **Save** (writes `adata_latest.h5ad`), "
                        f"run **Load dense** (in-RAM, no extra copy) then continue from this folder, or copy a `result.h5ad` here."
                    )
                session_dir_str = str(root.resolve())
                resume_kind = "session_folder"
            elif (root / "meta.json").is_file():
                h5ad_file, meta = _job_folder_result_h5ad(root)
                if not meta:
                    return _empty_load_ret(
                        f"Folder `{root}` has `meta.json` but it could not be read. Check JSON / permissions."
                    )
                jid = str(meta.get("id", root.name))
                st = meta.get("status", "?")
                typ = meta.get("type", "?")
                if st != "done":
                    return _empty_load_ret(
                        f"Job `{jid}` is not finished yet (status={st!r}, type={typ!r}). "
                        f"When **done**, load via **Job ID** here or **Resume** on this folder again."
                    )
                if h5ad_file is None:
                    return _empty_load_ret(
                        f"Job `{jid}` is **done** but no result `.h5ad` was found "
                        f"(`result_path` in meta: {meta.get('result_path')!r})."
                    )
                resumed_job_id = jid
                job_id_ui = gr.update(value=jid)
                log_extras.extend(
                    [
                        f"job_folder: {root}",
                        f"job_id: {jid}",
                        f"job_type: {typ}",
                    ]
                )
                resume_kind = "job_folder"
            else:
                return _empty_load_ret(
                    f"Unrecognized folder `{root}`. Expected **`session_meta.json`** (dataset session under "
                    f"`{sess_res.sessions_base()}`) or **`meta.json`** (background job under `{bgjobs.jobs_root()}`). "
                    f"Or paste the path to a `.h5ad` file."
                )

        assert h5ad_file is not None
        try:
            bgjobs.validate_server_read_path(os.fspath(h5ad_file.resolve()))
        except FileNotFoundError as e:
            return _empty_load_ret(str(e))
        except PermissionError as e:
            return _empty_load_ret(
                f"{e}\n\nThe result `.h5ad` path must fall under **`SCFMS_ALLOWED_PATH_PREFIXES`** when that env is set."
            )

        pstr = os.fspath(h5ad_file.resolve())
        try:
            t0 = time.perf_counter()
            if use_backed:
                adata = ad.read_h5ad(pstr, backed="r")
                adata.uns["_scfms_densify_src"] = pstr
                slog = (
                    f"Resumed **backed** (mmap) from `{h5ad_file.name}` in "
                    f"{_fmt_dur(time.perf_counter() - t0)}."
                )
            else:
                adata = ad.read_h5ad(pstr)
                slog = f"Resumed dense `.h5ad` in {_fmt_dur(time.perf_counter() - t0)}."
        except BaseException as e:
            if is_oom_error(e):
                return _empty_load_ret(oom_user_message(e))
            return _empty_load_ret(f"Read error: {e}")

        if resume_kind == "job_folder":
            jid_str = resumed_job_id or root.name
            session = sess_res.create_dataset_session(
                f"job_{jid_str}",
                source_kind="job_folder",
                source_path=pstr,
                n_obs=int(adata.n_obs),
                n_vars=int(adata.n_vars),
                extra={"job_id": jid_str, "resumed_from_folder": str(root)},
            )
            session_dir_str = str(session)
            msg = (
                f"Loaded finished job **`{jid_str}`** from `{h5ad_file.name}`\n\n"
                f"**Results folder:** `{session}`"
            )
            _try_append_compute_ram_log(
                session_dir_str,
                "resume_job_folder",
                _compute_ram_plan_lines(
                    adata,
                    extras=log_extras + [f"result_h5ad: {pstr}"],
                ),
            )
            return _pack_session_outputs(
                adata, msg, slog, job_id_u=job_id_ui, session_dir=session_dir_str
            )

        if resume_kind == "h5ad_standalone":
            session = sess_res.create_dataset_session(
                h5ad_file.stem,
                source_kind="resume_h5ad",
                source_path=pstr,
                n_obs=int(adata.n_obs),
                n_vars=int(adata.n_vars),
                extra={"resume": True},
            )
            session_dir_str = str(session)
            msg = (
                f"Opened `.h5ad` in a **new** session folder (file was not under a dataset session dir).\n\n"
                f"**Results folder:** `{session}`"
            )
        else:
            sdp = Path(session_dir_str or "").resolve()
            try:
                sm = json.loads((sdp / "session_meta.json").read_text(encoding="utf-8"))
                lab = sm.get("dataset_label", sdp.name)
            except (OSError, json.JSONDecodeError):
                lab = sdp.name
            msg = (
                f"Resumed **`{lab}`** — `{h5ad_file.name}` in `{sdp.name}/` "
                f"(same session folder; plots and **`{sess_res.COMPUTE_RAM_LOG}`** append here)."
            )

        assert session_dir_str is not None
        _try_append_compute_ram_log(
            session_dir_str,
            "session_resumed",
            _compute_ram_plan_lines(
                adata,
                extras=log_extras
                + [
                    f"resume_kind: {resume_kind}",
                    f"h5ad: {pstr}",
                ],
            ),
        )
        return _pack_session_outputs(adata, msg, slog, session_dir=session_dir_str)

    def random_subset_session(adata, n_keep, seed, session_dir):
        """Randomly subset cells in-session (same session folder when possible)."""
        if adata is None:
            return _empty_load_ret("Load data first, then subset.")
        try:
            nk = int(n_keep) if n_keep is not None else 0
        except (TypeError, ValueError):
            return _empty_load_ret("Invalid N cells to keep.")
        if nk <= 0:
            return _empty_load_ret("Set **N cells to keep** to a positive integer.")
        sd = (session_dir or "").strip() or None
        if nk >= adata.n_obs:
            msg = (
                f"**No change:** N cells to keep ({nk:,}) ≥ **n_obs** ({adata.n_obs:,}). "
                f"Pick a smaller N or load a larger file."
            )
            return _pack_session_outputs(adata, msg, "", session_dir=sd)
        try:
            _require_dense_adata(adata, "Random subset")
        except RuntimeError as e:
            return _empty_load_ret(str(e))
        try:
            sg = int(seed) if seed is not None else 0
            sub = subset_adata_random_cells(adata, nk, sg)
        except BaseException as e:
            if is_oom_error(e):
                return _empty_load_ret(oom_user_message(e))
            return _empty_load_ret(f"Subset failed: {e}")
        prev = adata.n_obs
        msg = (
            f"**Random subset:** **{sub.n_obs:,}** cells retained (from **{prev:,}**), seed=**{sg}**.\n\n"
            f"{estimate_adata_memory_report(sub)}"
        )
        _try_append_compute_ram_log(
            sd,
            "random_cell_subset_applied",
            _compute_ram_plan_lines(
                sub,
                extras=[
                    f"cells_before: {prev:,}",
                    f"cells_after: {sub.n_obs:,}",
                    f"seed: {sg}",
                ],
            ),
        )
        return _pack_session_outputs(sub, msg, "", session_dir=sd)

    def recompute_plots(adata, dist_plot_source, dist_by_obs, session_dir):
        if adata is None:
            return None, None, None, None, "Load a file first"
        try:
            spec = dist_plot_source or "X"
            og = _normalize_dist_by_obs_col(dist_by_obs)
            f1, f2, f3, f4 = _compute_distributions(adata, spec, obs_group_col=og)
            stem = _qc_dist_file_stem(spec, dist_by_obs)
            sess_res.save_figures_if_session(
                {
                    f"dist_cell_sums_{stem}": f1,
                    f"dist_detected_{stem}": f2,
                    f"dist_col_means_{stem}": f3,
                    f"dist_values_{stem}": f4,
                },
                session_dir,
            )
            strat = f" stratified by **{og}**" if og else ""
            note = (
                f"QC from {spec}{strat}; figures saved under session/plots/"
                if session_dir and str(session_dir).strip()
                else f"QC from {spec}{strat}"
            )
            return f1, f2, f3, f4, note
        except BaseException as e:
            em = oom_user_message(e) if is_oom_error(e) else f"Error: {e}"
            return None, None, None, None, em

    def preview_obsm_scatter(
        adata, obsm_key, color_obs, session_dir, umap_fig_w, umap_fig_h
    ):
        if adata is None:
            return None, "Load data first"
        if not obsm_key or obsm_key == "(none)":
            return None, "Pick an obsm key (not “(none)”)."
        try:
            ckey = None if (not color_obs or color_obs == "(none)") else str(color_obs)
            fig = _fig_obsm_scatter(
                adata,
                str(obsm_key),
                ckey,
                figsize=_coerce_figsize(umap_fig_w, umap_fig_h),
            )
            if fig is None:
                return None, f"Key {obsm_key!r} not in obsm."
            stem = f"obsm_scatter_{str(obsm_key).replace(' ', '_')}"
            sess_res.save_figures_if_session({stem: fig}, session_dir)
            return fig, f"Scatter for {obsm_key} saved to session/plots/."
        except BaseException as e:
            return None, (oom_user_message(e) if is_oom_error(e) else f"Error: {e}")

    def run_quick_umap_plot(
        adata,
        source_spec,
        max_cells,
        n_pcs_in,
        n_neighbors,
        min_dist,
        color_obs,
        session_dir,
        umap_fig_w,
        umap_fig_h,
    ):
        if adata is None:
            return None, "Load data first."
        try:
            from umap import UMAP
        except ImportError:
            return None, "Install `umap-learn` to use quick UMAP."
        try:
            spec = (source_spec or "X").strip()
            mc = int(max_cells) if max_cells is not None else 0
            work = adata
            if mc > 0 and adata.n_obs > mc:
                rng = np.random.default_rng(42)
                ix = np.sort(rng.choice(adata.n_obs, size=mc, replace=False))
                sub = adata[ix]
                # Backed mode forbids ``.copy()`` without ``filename=``; load the cell subset into RAM.
                if adata_is_backed(adata):
                    work = sub.to_memory()
                else:
                    work = sub.copy()
            npc = int(n_pcs_in) if n_pcs_in is not None else 0

            if spec.startswith("obsm:"):
                key = spec[5:]
                if key not in work.obsm:
                    return None, f"No obsm[{key!r}] on (subsampled) data."
                E = np.asarray(work.obsm[key], dtype=np.float64)
                Z = _reduce_obsm_for_quick_umap(E, npc)
            else:
                M = _matrix_from_spec(work, spec)
                Z = _reduce_matrix_for_quick_umap(M, npc)

            nn = max(2, int(n_neighbors))
            nn = min(nn, max(2, Z.shape[0] - 1))
            obs_d = _obs_index_digest(work.obs_names)
            qsig = _quick_umap_signature(
                spec, mc, npc, nn, float(min_dist), obs_d, work.n_obs, Z.shape[1]
            )
            U = None
            qu = work.obsm.get("_scfms_quick_umap")
            if (
                qu is not None
                and work.uns.get("_scfms_quick_umap_sig") == qsig
                and np.asarray(qu).shape[0] == work.n_obs
                and np.asarray(qu).shape[1] == 2
            ):
                U = np.asarray(qu, dtype=np.float32)
            reused_quick = U is not None
            if not reused_quick:
                U = np.asarray(
                    UMAP(
                        n_components=2,
                        n_neighbors=nn,
                        min_dist=float(min_dist),
                        random_state=0,
                    ).fit_transform(Z),
                    dtype=np.float32,
                )
                # Backed AnnData is read-only: cannot stash cache in obsm/uns.
                if not adata_is_backed(work):
                    if getattr(work, "is_view", False):
                        work = work.copy()
                    work.obsm["_scfms_quick_umap"] = U
                    work.uns["_scfms_quick_umap_sig"] = qsig
            ckey = None if (not color_obs or color_obs == "(none)") else str(color_obs)
            fig = _fig_xy_from_obs(
                U,
                work.obs,
                ckey,
                f"Quick UMAP ← {spec}",
                figsize=_coerce_figsize(umap_fig_w, umap_fig_h),
            )
            safe = spec.replace(":", "_").replace("/", "_")
            sess_res.save_figures_if_session({f"quick_umap_{safe}": fig}, session_dir)
            reuse_note = (
                " **(reused embedding; recolored / resized only).**"
                if reused_quick
                else ""
            )
            return (
                fig,
                f"Quick UMAP from **{spec}** on **{work.n_obs:,}** cells "
                f"(reduced to {Z.shape[1]}D then UMAP; **no** normalization / HVG / scaling). "
                f"Saved PNG under session **plots/**." + reuse_note,
            )
        except BaseException as e:
            return None, (
                oom_user_message(e) if is_oom_error(e) else f"Quick UMAP error: {e}"
            )

    def apply_and_preview(
        adata,
        use_matrix,
        obs_keep,
        obs_map_str,
        var_keep,
        var_map_str,
        layers_keep,
        obsm_keep,
        keep_raw,
        set_raw_from_x,
        session_dir,
        dist_plot_source,
        dist_by_obs,
        pip_batch_key_sel,
        embed_matrix_sel,
    ):
        if adata is None:
            return (
                gr.skip(),
                None,
                None,
                None,
                None,
                "Load a file first",
                None,
                None,
                gr.skip(),
                gr.skip(),
                gr.skip(),
                "",
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
            )
        try:
            _require_dense_adata(adata, "Apply edits (preview)")
            obs_map = _parse_renames(obs_map_str)
            var_map = _parse_renames(var_map_str)
            new_ad = _apply_edits(
                adata,
                use_matrix,
                obs_keep or list(adata.obs.columns),
                obs_map,
                var_keep or list(adata.var.columns),
                var_map,
                layers_keep or list(adata.layers.keys()),
                obsm_keep or list(adata.obsm.keys()),
                keep_raw,
                set_raw_from_x,
            )
            po = _list_plot_source_options(new_ad)
            dps = _pick_matrix_source_value(dist_plot_source, po)
            og = _normalize_dist_by_obs_col(dist_by_obs)
            f1, f2, f3, f4 = _compute_distributions(new_ad, dps, obs_group_col=og)
            stem = _qc_dist_file_stem(dps, dist_by_obs)
            sess_res.save_figures_if_session(
                {
                    f"dist_cell_sums_{stem}": f1,
                    f"dist_detected_{stem}": f2,
                    f"dist_col_means_{stem}": f3,
                    f"dist_values_{stem}": f4,
                },
                session_dir,
            )
            summary = (
                f"Preview: cells={new_ad.n_obs}, genes={new_ad.n_vars}; "
                f"layers={list(new_ad.layers.keys())}; raw={'yes' if new_ad.raw is not None else 'no'}\n\n"
                f"{estimate_adata_memory_report(new_ad)}"
            )
            umap_choices = ["(none)"] + list(new_ad.obs.columns)
            mopts = _list_matrix_options(new_ad)
            emb_opts = _list_embedding_matrix_options(new_ad)
            ms_u = gr.update(
                choices=mopts,
                value=_pick_matrix_source_value(use_matrix, mopts),
            )
            emb_u = gr.update(
                choices=emb_opts,
                value=_pick_matrix_source_value(embed_matrix_sel, emb_opts),
            )
            dist_up = gr.update(choices=po, value=dps)
            doc_cols = ["(none)"] + list(new_ad.obs.columns)
            db_val = dist_by_obs if dist_by_obs in doc_cols else "(none)"
            dist_by_up = gr.update(choices=doc_cols, value=db_val)
            obsm_ok = list(new_ad.obsm.keys())
            obsm_up = gr.update(choices=["(none)"] + obsm_ok, value="(none)")
            quick_uc = gr.update(choices=umap_choices, value="(none)")
            bk_val = pip_batch_key_sel if pip_batch_key_sel in doc_cols else "(none)"
            batch_key_up = gr.update(choices=doc_cols, value=bk_val)
            return (
                new_ad,
                f1,
                f2,
                f3,
                f4,
                summary,
                None,
                None,
                gr.update(choices=umap_choices, value="(none)"),
                ms_u,
                emb_u,
                "",
                gr.skip(),
                dist_up,
                dist_by_up,
                obsm_up,
                quick_uc,
                batch_key_up,
            )
        except BaseException as e:
            em = oom_user_message(e) if is_oom_error(e) else f"Error: {e}"
            return (
                gr.skip(),
                None,
                None,
                None,
                None,
                em,
                None,
                None,
                gr.skip(),
                gr.skip(),
                gr.skip(),
                "",
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
            )

    def run_full_pipeline(
        adata,
        normalize_total,
        target_sum,
        log1p,
        fc_min_counts,
        fc_min_genes,
        fg_min_cells,
        do_hvg,
        n_top_genes,
        hvg_flavor,
        subset_hvg,
        do_scale,
        scale_zero_center,
        scale_max_value,
        n_pcs,
        pca_solver,
        use_scanpy_pca,
        do_neighbors,
        neighbor_metric,
        n_neighbors_graph,
        do_umap,
        umap_min_dist,
        umap_spread,
        umap_color_obs,
        umap_fig_w,
        umap_fig_h,
        pip_pipeline_max_cells,
        pip_use_raw,
        pip_run_background,
        session_dir,
        dist_plot_source,
        dist_by_obs,
        pip_batch_corr,
        pip_batch_key,
        matrix_spec_ui,
        embed_matrix_ui,
    ):
        if adata is None:
            return (
                gr.skip(),
                None,
                None,
                None,
                None,
                "Load data first",
                None,
                None,
                gr.skip(),
                gr.skip(),
                gr.skip(),
                "",
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
            )
        try:
            _require_dense_adata(adata, "Preprocessing pipeline")

            def _nz_int(v) -> Optional[int]:
                if v is None:
                    return None
                try:
                    i = int(v)
                except (TypeError, ValueError):
                    return None
                return i if i > 0 else None

            smax = float(scale_max_value) if scale_max_value is not None else 0.0
            scale_cap = smax if smax > 0 else None
            umap_col = (
                None
                if (not umap_color_obs or umap_color_obs == "(none)")
                else str(umap_color_obs)
            )
            u_fw, u_fh = _coerce_figsize(umap_fig_w, umap_fig_h)
            pmax = _nz_int(pip_pipeline_max_cells)
            use_raw = bool(pip_use_raw)
            pipeline_kwargs = dict(
                normalize_total=bool(normalize_total),
                target_sum=float(target_sum),
                log1p=bool(log1p),
                filter_cells_min_counts=_nz_int(fc_min_counts),
                filter_cells_min_genes=_nz_int(fc_min_genes),
                filter_genes_min_cells=_nz_int(fg_min_cells),
                hvg=bool(do_hvg),
                n_top_genes=int(n_top_genes),
                hvg_flavor=str(hvg_flavor),
                subset_hvg=bool(subset_hvg),
                scale=bool(do_scale),
                scale_zero_center=bool(scale_zero_center),
                scale_max_value=scale_cap,
                n_pcs=int(n_pcs),
                pca_solver=str(pca_solver),
                use_scanpy_pca=bool(use_scanpy_pca),
                compute_neighbors=bool(do_neighbors),
                neighbor_metric=str(neighbor_metric),
                n_neighbors=int(n_neighbors_graph),
                compute_umap=bool(do_umap),
                umap_min_dist=float(umap_min_dist),
                umap_spread=float(umap_spread),
                umap_color_obs=umap_col,
                umap_fig_width=u_fw,
                umap_fig_height=u_fh,
                pipeline_input_matrix=("raw.X" if use_raw else "X"),
                pipeline_max_cells=pmax,
                pipeline_use_raw=use_raw,
                batch_correction=str(pip_batch_corr or "none"),
                batch_key=(
                    None
                    if (not pip_batch_key or pip_batch_key == "(none)")
                    else str(pip_batch_key)
                ),
            )

            if pip_run_background:
                jid = bgjobs.start_pipeline_job(adata, pipeline_kwargs)
                _try_append_compute_ram_log(
                    (session_dir or "").strip(),
                    "preprocessing_pipeline_background_queued",
                    _compute_ram_plan_lines(
                        adata,
                        extras=[
                            f"background_job_id: {jid}",
                            "note: After **Load finished result into session**, a new log entry records the output AnnData.",
                        ],
                    ),
                )
                root = bgjobs.jobs_root()
                slog = format_timing_report(
                    "Background preprocessing — job queued on the compute server",
                    {},
                    footer=(
                        f"**Job ID:** `{jid}`\n\n"
                        f"**Store:** `{root}`  (env **SCFMS_JOB_DIR**)\n\n"
                        "Paste the ID into **Job ID** below; enable **Auto-poll** or click **Refresh job status**. "
                        "When status is **done**, click **Load finished result into session**.\n\n"
                        "While the job runs, each completed step updates **rough ETA remaining** in the job record "
                        "(average pace × steps left)."
                    ),
                )
                return (
                    gr.skip(),
                    gr.skip(),
                    gr.skip(),
                    gr.skip(),
                    gr.skip(),
                    f"Background job running: {jid}",
                    gr.skip(),
                    gr.skip(),
                    gr.skip(),
                    gr.skip(),
                    gr.skip(),
                    slog,
                    gr.update(value=jid),
                    gr.skip(),
                    gr.skip(),
                    gr.skip(),
                    gr.skip(),
                    gr.skip(),
                )

            t_wall0 = time.perf_counter()
            run_kw = {
                k: v for k, v in pipeline_kwargs.items() if k != "pipeline_use_raw"
            }
            work = pipeline_base_adata(adata, use_raw)
            new_ad, fig_pca, fig_umap, timings = run_expression_pipeline(
                work,
                progress_callback=None,
                **run_kw,
            )
            wall = time.perf_counter() - t_wall0
            po = _list_plot_source_options(new_ad)
            dps = _pick_matrix_source_value(dist_plot_source, po)
            og = _normalize_dist_by_obs_col(dist_by_obs)
            f1, f2, f3, f4 = _compute_distributions(new_ad, dps, obs_group_col=og)
            stem = _qc_dist_file_stem(dps, dist_by_obs)
            sess_res.save_figures_if_session(
                {
                    f"dist_cell_sums_{stem}": f1,
                    f"dist_detected_{stem}": f2,
                    f"dist_col_means_{stem}": f3,
                    f"dist_values_{stem}": f4,
                    "pca_variance_ratio": fig_pca,
                    "umap": fig_umap,
                },
                session_dir,
            )
            msg = (
                f"Pipeline OK (input={'raw.X' if use_raw else 'X'}): n_obs={new_ad.n_obs}, n_vars={new_ad.n_vars}; "
                f"obsm={list(new_ad.obsm.keys())}; obsp={list(new_ad.obsp.keys())}\n\n"
                f"{estimate_adata_memory_report(new_ad)}"
            )
            if "cache_recolor" in timings:
                msg = (
                    "**Reused stored UMAP** (same preprocessing settings as last run); **only the figure "
                    "(color / size)** was updated.\n\n"
                ) + msg
            mopts = _list_matrix_options(new_ad)
            emb_opts = _list_embedding_matrix_options(new_ad)
            ms_u = gr.update(
                choices=mopts,
                value=_pick_matrix_source_value(matrix_spec_ui, mopts),
            )
            emb_u = gr.update(
                choices=emb_opts,
                value=_pick_matrix_source_value(embed_matrix_ui, emb_opts),
            )
            slog = format_timing_report(
                "Preprocessing step durations (wall clock includes plotting & overhead)",
                timings,
                wall_s=wall,
            )
            _psn = (session_dir or "").strip()
            _snap = format_timing_report(
                "preprocessing_timings",
                timings,
                wall_s=wall,
            ).split("\n")
            _try_append_compute_ram_log(
                _psn,
                "preprocessing_scanpy_pipeline_done",
                _compute_ram_plan_lines(
                    new_ad,
                    extras=[
                        f"pipeline_input_matrix: {'raw.X' if use_raw else 'X'}",
                        f"batch_correction: {pipeline_kwargs.get('batch_correction')}",
                        f"pip_batch_key: {pipeline_kwargs.get('batch_key')}",
                        "",
                        "timings:",
                        *_snap,
                    ],
                ),
            )
            dist_up = gr.update(choices=po, value=dps)
            doc_cols = ["(none)"] + list(new_ad.obs.columns)
            db_val = dist_by_obs if dist_by_obs in doc_cols else "(none)"
            dist_by_up = gr.update(choices=doc_cols, value=db_val)
            obsm_ok = list(new_ad.obsm.keys())
            obsm_up = gr.update(choices=["(none)"] + obsm_ok, value="(none)")
            umap_choices_p = ["(none)"] + list(new_ad.obs.columns)
            quick_uc = gr.update(choices=umap_choices_p, value="(none)")
            pb_val = pip_batch_key if pip_batch_key in doc_cols else "(none)"
            batch_key_up = gr.update(choices=doc_cols, value=pb_val)
            return (
                new_ad,
                f1,
                f2,
                f3,
                f4,
                msg,
                fig_pca,
                fig_umap,
                gr.skip(),
                ms_u,
                emb_u,
                slog,
                gr.skip(),
                dist_up,
                dist_by_up,
                obsm_up,
                quick_uc,
                batch_key_up,
            )
        except BaseException as e:
            em = oom_user_message(e) if is_oom_error(e) else f"Pipeline error: {e}"
            return (
                gr.skip(),
                None,
                None,
                None,
                None,
                em,
                None,
                None,
                gr.skip(),
                gr.skip(),
                gr.skip(),
                "",
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
            )

    def redraw_umap_only(adata, umap_color_obs, session_dir, umap_fig_w, umap_fig_h):
        if adata is None:
            return None, "Load data first"
        if "X_umap" not in adata.obsm:
            return None, "No UMAP coordinates; run the pipeline with UMAP enabled."
        try:
            ckey = (
                None
                if (not umap_color_obs or umap_color_obs == "(none)")
                else str(umap_color_obs)
            )
            fig = _fig_umap(
                adata,
                ckey,
                figsize=_coerce_figsize(umap_fig_w, umap_fig_h),
            )
            sess_res.save_figures_if_session({"umap_colored": fig}, session_dir)
            return fig, "UMAP colors updated"
        except BaseException as e:
            return None, (
                oom_user_message(e) if is_oom_error(e) else f"Redraw error: {e}"
            )

    def _embed_dataset_folder(name_tb: str, server_path_tb: str) -> str:
        if (name_tb or "").strip():
            return bgjobs.sanitize_dataset_folder((name_tb or "").strip())
        s = (server_path_tb or "").strip()
        if s:
            return bgjobs.sanitize_dataset_folder(Path(s).stem)
        return "uploaded"

    def run_scfm_embed(
        adata,
        model,
        embed_matrix,
        matrix_spec_ui,
        dist_plot_source_ui,
        obsm_key_tb,
        scgpt_ckpt,
        n_latent,
        obsm_keep_sel,
        scfm_run_background,
        scfm_slurm_gpu,
        server_path_for_name,
        scfm_dataset_name,
        slurm_partition,
        slurm_cpus,
        slurm_mem,
        slurm_time,
        slurm_bash_prologue,
        scfms_repo_root_tb,
        session_dir,
    ):
        if adata is None:
            return (
                gr.skip(),
                "Load data first",
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                "",
                gr.skip(),
                gr.skip(),
            )
        if not embed_matrix:
            return (
                gr.skip(),
                "Select a matrix to embed (reload the file if choices are empty).",
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                "",
                gr.skip(),
                gr.skip(),
            )
        try:
            _require_dense_adata(adata, "scFM embedding")
            rec = estimate_scfm_slurm_resources(
                adata,
                model=str(model),
                matrix_spec=str(embed_matrix),
                n_latent_scvi=int(n_latent) if n_latent is not None else 64,
            )
            cpus_req = rec["cpus"]
            if slurm_cpus is not None:
                try:
                    user_cpus = int(slurm_cpus)
                except (TypeError, ValueError):
                    user_cpus = 0
                if user_cpus > 0:
                    cpus_req = max(user_cpus, int(rec["cpus"]))
            mem_req = str(rec["mem"])
            user_mem_gib = _parse_mem_gib(slurm_mem)
            if user_mem_gib is not None:
                mem_req = f"{max(user_mem_gib, int(rec['mem_gib']))}G"
            time_raw = str(slurm_time or "").strip()
            time_req = str(rec["time"]) if not time_raw or time_raw.lower() == "auto" else time_raw
            if scfm_slurm_gpu:
                wpath = normalize_ui_weights_path(str(model), scgpt_ckpt)
                sc_kwargs = dict(
                    model=str(model),
                    matrix_spec=str(embed_matrix),
                    obsm_key=(obsm_key_tb or "").strip() or None,
                    scgpt_ckpt=wpath,
                    n_latent_scvi=int(n_latent) if n_latent is not None else 64,
                )
                rr = (scfms_repo_root_tb or "").strip()
                repo = Path(rr).resolve() if rr else None
                dname = _embed_dataset_folder(scfm_dataset_name, server_path_for_name)
                jid = bgjobs.start_scfm_slurm_job(
                    adata,
                    sc_kwargs,
                    dataset_name=dname,
                    repo_root=repo,
                    partition=effective_slurm_partition(slurm_partition),
                    gres="gpu:1",
                    cpus=cpus_req,
                    mem=mem_req,
                    time_limit=time_req,
                    bash_prologue=str(slurm_bash_prologue or ""),
                )
                base_out = bgjobs.embed_output_base()
                slog = format_timing_report(
                    "Slurm GPU embedding submitted",
                    {},
                    footer=(
                        f"**UI job ID:** `{jid}`\n"
                        f"**Resolved request:** `-c {cpus_req}` · `--mem={mem_req}` · `-t {time_req}`\n"
                        f"**Output (when done):** `{base_out / dname}`\n"
                        f"**Staging:** `{bgjobs.jobs_root() / jid}`\n\n"
                        "Refresh **Job status** or enable auto-poll. When **status** is **done**, "
                        "load the result .h5ad into the session.\n\n"
                        "Set **SCFMS_SLURM_EMBED_BASE** to put embeddings on shared storage "
                        "(default: under the job store's `slurm_embeddings/`)."
                    ),
                )
                _try_append_compute_ram_log(
                    (session_dir or "").strip(),
                    "scfm_slurm_job_submitted",
                    _compute_ram_plan_lines(
                        adata,
                        extras=[
                            f"ui_job_id: {jid}",
                            f"model: {model}",
                            f"matrix_spec: {embed_matrix}",
                            f"slurm_cpus_resolved: {cpus_req}",
                            f"slurm_mem_resolved: {mem_req}",
                            f"slurm_time_resolved: {time_req}",
                            *[f"resource_note: {x}" for x in rec["notes"]],
                            "note: When the Slurm run completes, load the output .h5ad to append fresh RAM estimates.",
                        ],
                    ),
                )
                return (
                    gr.skip(),
                    f"Slurm-GPU scFM job queued (UI id {jid})",
                    gr.skip(),
                    gr.skip(),
                    gr.skip(),
                    gr.skip(),
                    slog,
                    gr.update(value=jid),
                    gr.skip(),
                )

            if scfm_run_background:
                wpath = normalize_ui_weights_path(str(model), scgpt_ckpt)
                sc_kwargs = dict(
                    model=str(model),
                    matrix_spec=str(embed_matrix),
                    obsm_key=(obsm_key_tb or "").strip() or None,
                    scgpt_ckpt=wpath,
                    n_latent_scvi=int(n_latent) if n_latent is not None else 64,
                )
                jid = bgjobs.start_scfm_job(adata, sc_kwargs)
                _try_append_compute_ram_log(
                    (session_dir or "").strip(),
                    "scfm_embedding_background_queued",
                    _compute_ram_plan_lines(
                        adata,
                        extras=[
                            f"background_job_id: {jid}",
                            f"model: {model}",
                            f"matrix_spec: {embed_matrix}",
                            "note: Load the finished .h5ad via **Load finished result** to log embeddings on disk.",
                        ],
                    ),
                )
                slog = format_timing_report(
                    "scFM background job queued",
                    {},
                    footer=(
                        f"**Job ID:** `{jid}`  |  **store:** `{bgjobs.jobs_root()}`\n\n"
                        "Embedding runs as one long step; status file updates when it finishes."
                    ),
                )
                return (
                    gr.skip(),
                    f"Background embedding job: {jid}",
                    gr.skip(),
                    gr.skip(),
                    gr.skip(),
                    gr.skip(),
                    slog,
                    gr.update(value=jid),
                    gr.skip(),
                )

            t0 = time.perf_counter()
            crd = (session_dir or "").strip() or None
            wpath = normalize_ui_weights_path(str(model), scgpt_ckpt)
            new_ad, msg, tmap = attach_scfm_embedding(
                adata,
                model=str(model),
                matrix_spec=str(embed_matrix),
                obsm_key=obsm_key_tb if obsm_key_tb else None,
                scgpt_ckpt=wpath,
                n_latent_scvi=int(n_latent) if n_latent is not None else 64,
                compat_report_dir=crd,
            )
            tw = time.perf_counter() - t0
            ok = list(new_ad.obsm.keys())
            nk = _normalize_obsm_key(str(model), obsm_key_tb if obsm_key_tb else None)
            if not obsm_keep_sel:
                val_t = ok
            else:
                val_t = [k for k in obsm_keep_sel if k in ok]
                if nk not in val_t:
                    val_t = val_t + [nk]
            slog = format_timing_report(f"scFM — {model}", tmap, wall_s=tw)
            pdf_path = (new_ad.uns.get("scfms_compat_report_pdf") or "").strip()
            mopts = _list_matrix_options(new_ad)
            emb_opts = _list_embedding_matrix_options(new_ad)
            ms_u = gr.update(
                choices=mopts,
                value=_pick_matrix_source_value(matrix_spec_ui, mopts),
            )
            emb_u = gr.update(
                choices=emb_opts,
                value=_pick_matrix_source_value(embed_matrix, emb_opts),
            )
            dist_u = gr.update(
                choices=mopts,
                value=_pick_matrix_source_value(dist_plot_source_ui, mopts),
            )
            _esh = tuple(np.asarray(new_ad.obsm[nk]).shape)
            _try_append_compute_ram_log(
                (session_dir or "").strip(),
                f"scfm_embedding_done_{model}",
                _compute_ram_plan_lines(
                    new_ad,
                    extras=[
                        f"model: {model}",
                        f"input_matrix_spec: {embed_matrix}",
                        f"obsm_key: {nk}",
                        f"embedding_shape: {_esh}",
                        "",
                        "timings:",
                        *slog.split("\n")[:48],
                    ],
                ),
            )
            return (
                new_ad,
                msg,
                gr.update(choices=ok, value=val_t),
                ms_u,
                emb_u,
                dist_u,
                slog,
                gr.skip(),
                gr.update(value=pdf_path or None),
            )
        except BaseException as e:
            em = oom_user_message(e) if is_oom_error(e) else f"scFM error: {e}"
            return (
                gr.skip(),
                em,
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                "",
                gr.skip(),
                gr.skip(),
            )

    def refresh_job_ui(job_id: str):
        jid = (job_id or "").strip()
        if not jid:
            return "(enter a job ID above)"
        return bgjobs.format_meta_report(jid)

    def load_job_into_session(job_id: str):
        jid = (job_id or "").strip()
        if not jid:
            return _empty_load_ret("Enter a job ID to load its result.")
        p = bgjobs.result_h5ad_path(jid)
        if not p:
            m = bgjobs.read_meta(jid) or {}
            st = m.get("status", "unknown")
            return _empty_load_ret(
                f"Job {jid!r} is not finished (status={st}). Refresh job status."
            )
        try:
            t0 = time.perf_counter()
            adata = ad.read_h5ad(p)
            dt = time.perf_counter() - t0
            meta_txt = bgjobs.format_meta_report(jid)
            session = sess_res.create_dataset_session(
                f"job_{jid}",
                source_kind="job",
                source_path=str(p),
                n_obs=int(adata.n_obs),
                n_vars=int(adata.n_vars),
                extra={"job_id": jid},
            )
            msg = f"Loaded result for job {jid} from {p}\n\n**Results folder:** `{session}`"
            slog = f"Read result in {_fmt_dur(dt)}.\n\n---\n{meta_txt}"
            _try_append_compute_ram_log(
                str(session),
                "background_job_result_h5ad_loaded",
                _compute_ram_plan_lines(
                    adata,
                    extras=[
                        f"job_id: {jid}",
                        f"result_path: {p}",
                    ],
                ),
            )
            return _pack_session_outputs(adata, msg, slog, session_dir=str(session))
        except BaseException as e:
            if is_oom_error(e):
                return _empty_load_ret(oom_user_message(e))
            return _empty_load_ret(f"Failed to read result: {e}")

    def poll_job_ui(job_id: str, enabled: bool):
        if not enabled:
            return gr.skip()
        return refresh_job_ui(job_id)

    def list_recent_jobs_ui():
        return bgjobs.list_recent_jobs()

    def save_file(adata, out_path, session_dir):
        if adata is None:
            return "Load and apply changes first"
        op = (out_path or "").strip()
        if not op:
            cr = (session_dir or "").strip()
            if not cr:
                return "Set output path or load a dataset first (session folder required for default path)."
            op = str(Path(cr).expanduser().resolve() / "adata_latest.h5ad")
        try:
            out_dir = os.path.dirname(op)
            if out_dir and not os.path.exists(out_dir):
                os.makedirs(out_dir, exist_ok=True)
            adata.write_h5ad(op)
            return f"Saved to {op}"
        except BaseException as e:
            if is_oom_error(e):
                return oom_user_message(e)
            return f"Save error: {e}"

    def _active_session_dir(session_dir, session_folder_text: str) -> str:
        """Prefer Gradio ``State``; fall back to **Active results folder** (same value after load)."""
        for raw in ((session_dir or "").strip(), (session_folder_text or "").strip()):
            if not raw:
                continue
            try:
                return str(Path(raw).expanduser().resolve())
            except OSError:
                return raw
        return ""

    def start_dense_load(adata, session_dir, session_folder_text: str):
        """Background thread: full ``read_h5ad`` into RAM (no extra ``.h5ad`` on disk)."""
        sd = _active_session_dir(session_dir, session_folder_text)
        if not sd:
            return (
                "No session folder yet — click **Load from server path** after entering a path "
                "(**Active results folder** should show a directory once load succeeds)."
            )
        if adata is None:
            return "No AnnData in session."
        if not adata_is_backed(adata):
            return "Current object is already **dense** in memory (not backed)."
        src = (adata.uns or {}).get("_scfms_densify_src")
        if not src:
            return "No densify source in `uns['_scfms_densify_src']` — reload with **Open backed** checked."
        sp = Path(str(src))
        if not sp.is_file():
            return f"Source file missing: {sp}"
        sdp = Path(sd).resolve()
        ok, err = dense_load_start(str(sdp), str(sp.resolve()))
        if not ok:
            return f"Could not start dense load: {err}"
        _try_append_compute_ram_log(
            str(sdp),
            "dense_load_started",
            [
                f"reads_h5ad: {sp.resolve()}",
                "mode: background_thread_full_read (no materialized .h5ad file)",
                "note: Same Python process as the app; peak RSS rises while the thread loads.",
            ],
        )
        return (
            "**Started** dense load — **background thread** (full read into RAM, **no disk copy**). "
            f"Append log: **`{sess_res.COMPUTE_RAM_LOG}`**. "
            "Keep using **Redraw UMAP** on backed data; when the log shows **Finished**, the session switches to dense."
        )

    with gr.Blocks(
        title="H5AD Preprocessing",
        css=r"""
@keyframes scfms-dots-opacity {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.4; }
}
#scfms_load_status_anchor.scfms-load-path-pending textarea,
#scfms_load_status_anchor.scfms-load-path-pending input[type="text"] {
  color: #1864ab;
  animation: scfms-dots-opacity 0.85s ease-in-out infinite;
}
""",
    ) as demo:
        # Must be created inside `with gr.Blocks()` so render() registers the State in
        # demo.blocks; otherwise SessionState.__setitem__ sees blocks.get(id) is None and
        # overwrites the session blocks map with AnnData → AttributeError on get_block_name.
        state_adata = gr.State(None)
        state_session_dir = gr.State(None)

        gr.Markdown("# H5AD Preprocessing App")
        gr.Markdown(
            "Set **Path on compute server** below and click **Load from server path** (or press **Enter** — both show a loading indicator while the file opens). "
            "Use **Resume** to reopen a **`.assets/…`** session folder, a **background job** directory under the job store, or any **`.h5ad`** after you disconnect. "
            "**Open backed** is on by default; use **Load dense in background** when you need the full matrix for subsetting or embeddings. Open **Details** for paths, RAM, jobs, and session folders."
        )
        with gr.Accordion(
            "Details — paths, jobs, RAM, backed mode, sessions", open=False
        ):
            gr.Markdown(
                f"**Runs on the compute server.** Paths are checked on the **machine running this app** "
                f"(`expanduser` / `expandvars`). Optional **`SCFMS_ALLOWED_PATH_PREFIXES`**: pipe-separated allowed roots "
                f"(e.g. `/scratch/your_project/|/shared/data/`). Jobs & caches: **`{bgjobs.jobs_root()}`** (set **SCFMS_JOB_DIR** to move). "
                "After each successful **load** / **job result**, **Status** includes an **estimated in-RAM size** of the main arrays "
                "(sparse `X` counts nonzeros only; real peak RAM can be much higher during normalization, PCA, or embeddings).\n\n"
                "**Background jobs** keep running if you close the browser; reopen this page, paste the **Job ID**, poll or load result. "
                "**Resume** accepts the **Active results folder** path (`.assets/<segment>/<timestamp>_…/` with `session_meta.json`), a **job folder** (`…/job_store/<id>/` with `meta.json`), or a **`.h5ad`** file; dataset sessions reuse the same folder so plots and the RAM log keep appending. "
                "**Rough ETA** uses average seconds per finished step × steps remaining (pipeline only).\n\n"
                "**OOM / RAM:** heavy steps are wrapped so **MemoryError** and common **CUDA/dataloader OOM** messages surface in **Status** "
                "without killing the app — use **Subset cells** or lower **Max cells**, then retry.\n\n"
                "**Backed `.h5ad`:** loads in **backed** mode by default (mmap `X`, lower RAM). Uncheck **Open backed** to load the full matrix into RAM. Use **Load dense in background** "
                "to materialize the full object in a **separate process** while UMAP/obsm plots keep using the backed handle.\n\n"
                "**Quick UMAP** (accordion below) builds a 2D embedding from the **QC source** without normalization or HVG.\n\n"
                "**Large datasets (e.g. millions of cells):** use **Max cells for pipeline** to randomly subsample before PCA / neighbors / UMAP "
                "(full `.h5ad` in memory may still be large). **UMAP figures** only plot a random subset (see title); "
                "QC histograms sample cells for speed (**`SCFMS_DIST_SAMPLE_CELLS`**). Each **load** creates a timestamped folder under "
                f"**`{sess_res.sessions_base()}/<dataset-bucket>/`** (managed `.data/` name or parent of `data/` in the path; or flat **`SCFMS_SESSION_DIR`**); PNGs and compatibility PDFs go into that session.\n\n"
                f"**Compute / RAM log:** every session folder gets an append-only **`{sess_res.COMPUTE_RAM_LOG}`** with **GiB** estimates "
                "(main arrays, rough peak, suggested minimum node RAM) after **load**, **dense materialize**, **subset**, "
                "**preprocessing**, **scFM embed**, and **job result** loads — use it to size Slurm **`--mem`** and interactive nodes."
            )
        with gr.Accordion("Where is this running? (server vs laptop)", open=False):
            gr.Markdown(runtime_info_markdown())
        session_path_display = gr.Textbox(
            label="Active results folder (this dataset session)",
            interactive=False,
        )
        with gr.Row():
            server_path = gr.Textbox(
                label="Path on compute server (.h5ad)",
                placeholder="/path/on/cluster/data.h5ad",
                scale=3,
            )
            load_server_btn = gr.Button(
                "Load from server path", variant="primary", scale=1
            )
            cancel_load_btn = gr.Button(
                "Cancel loading",
                variant="stop",
                scale=1,
                visible=False,
            )
            load_dense_btn = gr.Button(
                "Load dense in background",
                variant="secondary",
                scale=1,
            )
            cancel_dense_btn = gr.Button(
                "Cancel dense load",
                variant="stop",
                scale=1,
                visible=False,
            )
        open_backed = gr.Checkbox(
            label="Open .h5ad in **backed** mode (mmap `X`, lower RAM — then **Load dense in background** for pipeline / subset / embeddings)",
            value=True,
        )
        with gr.Row():
            resume_path_input = gr.Textbox(
                label="Resume — session folder, job folder, or .h5ad (uses **Open backed** above)",
                placeholder=".../.assets/<segment>/<timestamp>_<name>/  or  .../scfms_job_store/<id>/  or  file.h5ad",
                scale=3,
            )
            resume_btn = gr.Button("Resume", variant="secondary", scale=1)
            cancel_resume_btn = gr.Button(
                "Cancel resume",
                variant="stop",
                scale=1,
                visible=False,
            )
        dense_load_log = gr.Textbox(
            label="Dense load status (auto-refresh every 2s)",
            lines=5,
            interactive=False,
        )
        dense_timer = gr.Timer(value=2.0)
        status = gr.Textbox(
            label="Status",
            interactive=False,
            elem_id="scfms_load_status_anchor",
        )
        with gr.Accordion(
            "Subset cells — random sample (row-slices X, obs, layers, obsm, obsp, raw)",
            open=False,
        ):
            gr.Markdown(
                "**Yes — this subsets cells (rows), not genes.** The app replaces the in-session object with "
                "**`adata[random_indices].copy()`**, so **`X`** loses rows to match **`n_obs`**, together with "
                "**`obs`**, **`layers`**, **`obsm`**, **`obsp`**, and **`raw`** (if present). **`var` / genes are unchanged.**\n\n"
                "Use this to shrink RAM before pipeline / embeddings. **Backed mode:** run **Load dense in background** "
                "first (subset needs a writable in-memory `AnnData`). The **active session folder** path stays the same."
            )
            with gr.Row():
                subset_n_cells = gr.Number(
                    label="N cells to keep",
                    value=50_000,
                    precision=0,
                    minimum=1,
                )
                subset_seed = gr.Number(
                    label="Random seed",
                    value=0,
                    precision=0,
                )
            with gr.Row():
                btn_subset_cells = gr.Button(
                    "Apply random cell subset",
                    variant="secondary",
                )
                cancel_subset_btn = gr.Button(
                    "Cancel subset",
                    variant="stop",
                    visible=False,
                )
        step_timings_log = gr.Textbox(
            label="Step timings & job hints",
            lines=12,
            interactive=False,
        )

        with gr.Accordion(
            "Background jobs — reconnect with Job ID or Resume (job folder)", open=False
        ):
            job_id_input = gr.Textbox(
                label="Job ID",
                placeholder="Returned when you start a background pipeline / scFM run",
            )
            gr.Markdown(
                "When **status** is **done**, you can **Load finished result** here or paste the same job directory into **Resume** (above)."
            )
            with gr.Row():
                refresh_job_btn = gr.Button("Refresh job status")
                cancel_refresh_job_btn = gr.Button(
                    "Cancel refresh",
                    variant="stop",
                    visible=False,
                )
                load_job_result_btn = gr.Button("Load finished result into session")
                cancel_load_job_btn = gr.Button(
                    "Cancel load result",
                    variant="stop",
                    visible=False,
                )
                list_jobs_btn = gr.Button("List recent job IDs")
                cancel_list_jobs_btn = gr.Button(
                    "Cancel list jobs",
                    variant="stop",
                    visible=False,
                )
            job_poll_enable = gr.Checkbox(
                label="Auto-poll job status every 2s (server must keep this app process alive)",
                value=False,
            )
            job_status_log = gr.Textbox(
                label="Job status (from disk)",
                lines=18,
                interactive=False,
            )
            recent_jobs_list = gr.Textbox(
                label="Recent jobs",
                lines=8,
                interactive=False,
            )
            job_timer = gr.Timer(value=2.0)

        with gr.Row():
            matrix_spec = gr.Radio(
                label="Matrix to analyze/set as X", choices=[], value=None
            )
            keep_raw = gr.Checkbox(label="Keep .raw in output", value=True)
            set_raw_from_x = gr.Checkbox(
                label="Set .raw from current X before switching", value=False
            )

        with gr.Row():
            obs_keep = gr.CheckboxGroup(
                choices=[], label="obs columns to keep (leave empty to keep all)"
            )
            obs_map = gr.Textbox(label="obs renames (format: old1=new1, old2=new2)")
        with gr.Row():
            var_keep = gr.CheckboxGroup(
                choices=[], label="var columns to keep (leave empty to keep all)"
            )
            var_map = gr.Textbox(label="var renames (format: old1=new1, old2=new2)")
        with gr.Row():
            layers_keep = gr.CheckboxGroup(
                choices=[], label="layers to keep (leave empty to keep all)"
            )
            obsm_keep = gr.CheckboxGroup(
                choices=[], label="obsm keys to keep (leave empty to keep all)"
            )

        dist_plot_source = gr.Radio(
            label="QC / histogram source — X, raw.X, layers, or obsm embeddings",
            choices=["X"],
            value="X",
        )
        dist_by_obs = gr.Dropdown(
            choices=["(none)"],
            value="(none)",
            label="Stratify QC histograms by obs column",
            info=(
                f"Overlaid histograms per group to compare shifts (same cell subsample as QC). "
                f"If there are more than {DIST_HIST_MAX_GROUPS} levels, the top "
                f"{DIST_HIST_MAX_GROUPS - 1} by count plus **Other** are shown."
            ),
        )
        with gr.Row():
            recompute = gr.Button("Recompute QC histograms (uses source above)")
            cancel_recompute_btn = gr.Button(
                "Cancel QC recompute",
                variant="stop",
                visible=False,
            )
        with gr.Row():
            plot1 = gr.Plot(label="Per-cell row sums")
            plot2 = gr.Plot(label="Per-cell nonzero features")
        with gr.Row():
            plot3 = gr.Plot(label="Per-feature column means")
            plot4 = gr.Plot(label="Value distribution")

        with gr.Row():
            apply_btn = gr.Button("Apply Edits (Preview)")
            cancel_apply_btn = gr.Button(
                "Cancel apply",
                variant="stop",
                visible=False,
            )
        preview_status = gr.Textbox(
            label="Preview / pipeline summary",
            interactive=False,
        )
        gr.Markdown(
            "**2D embedding figure size** (matplotlib inches) — Quick UMAP, pipeline UMAP, obsm scatter, "
            "and **Redraw UMAP colors only**. "
            "**Tip:** increase **Width** when categorical legends are crowded or clipped (legends use multiple columns automatically)."
        )
        with gr.Row():
            umap_fig_w = gr.Number(
                label="Width (in)",
                value=DEFAULT_UMAP_FIGSIZE[0],
                minimum=1,
                maximum=40,
                precision=1,
            )
            umap_fig_h = gr.Number(
                label="Height (in)",
                value=DEFAULT_UMAP_FIGSIZE[1],
                minimum=1,
                maximum=40,
                precision=1,
            )

        with gr.Accordion("Quick UMAP (no normalization / HVG / scaling)", open=False):
            gr.Markdown(
                "Runs **UMAP** (``umap-learn``) on a **PCA** (dense) or **truncated SVD** (sparse) reduction of the "
                "**QC / histogram source** selected above — same options as **X**, **raw.X**, **layers**, or **obsm** entries. "
                "This does **not** run the Scanpy preprocessing block, does **not** normalize or select HVGs, and does **not** "
                "write **`obsm['X_umap']`** (only shows a figure; PNG saved under the session **`plots/`** folder). "
                "Set **Max cells** to limit RAM/time on large objects."
            )
            with gr.Row():
                quick_max_cells = gr.Number(
                    label="Max cells (0 = use all; set e.g. 50k–250k if slow / OOM)",
                    value=0,
                    precision=0,
                    minimum=0,
                )
                quick_n_pcs = gr.Number(
                    label="PCA/SVD comps (0 = auto, cap 50)",
                    value=50,
                    precision=0,
                    minimum=0,
                )
            with gr.Row():
                quick_n_neighbors = gr.Number(
                    label="n_neighbors",
                    value=15,
                    precision=0,
                    minimum=2,
                )
                quick_min_dist = gr.Slider(0.001, 0.99, value=0.1, label="min_dist")
            quick_umap_color = gr.Dropdown(
                choices=["(none)"],
                value="(none)",
                label="Color by obs column",
            )
            with gr.Row():
                btn_quick_umap = gr.Button("Compute quick UMAP from QC source above")
                cancel_quick_umap_btn = gr.Button(
                    "Cancel quick UMAP",
                    variant="stop",
                    visible=False,
                )
            plot_quick_umap = gr.Plot(label="Quick UMAP")

        with gr.Accordion("Preprocessing: normalization, HVG, PCA, UMAP", open=False):
            gr.Markdown(
                "Runs on the **current** AnnData in session (after *Apply Edits* or raw *Load*). "
                "Toggle **Pipeline from raw.X** to copy **`adata.raw`** (counts, `raw.var`) into a working object so "
                "normalization/HVG/PCA run on raw counts; leave off to use **current `X`**. "
                "`seurat_v3` HVGs expect approximate **counts** in the pipeline input matrix when possible."
            )
            pip_use_raw = gr.Checkbox(
                label="Pipeline from raw.X (requires .raw; uses raw.var / raw.X as input matrix)",
                value=False,
            )
            with gr.Row():
                pip_norm_total = gr.Checkbox(label="Normalize total counts", value=True)
                pip_target_sum = gr.Number(label="Target sum", value=10000, minimum=1)
                pip_log1p = gr.Checkbox(label="log1p", value=True)
            gr.Markdown("**QC** — leave at 0 to skip")
            with gr.Row():
                pip_fc_min_counts = gr.Number(
                    label="Cell min total counts", value=0, precision=0
                )
                pip_fc_min_genes = gr.Number(
                    label="Cell min detected genes", value=0, precision=0
                )
                pip_fg_min_cells = gr.Number(
                    label="Gene min cells", value=0, precision=0
                )
            with gr.Row():
                pip_hvg = gr.Checkbox(label="Highly variable genes", value=True)
                pip_n_top_hvg = gr.Number(label="n_top_genes", value=2000, precision=0)
                pip_hvg_flavor = gr.Dropdown(
                    choices=["seurat", "cell_ranger", "seurat_v3"],
                    value="seurat",
                    label="HVG flavor",
                )
                pip_subset_hvg = gr.Checkbox(label="Subset to HVGs only", value=False)
            with gr.Row():
                pip_scale = gr.Checkbox(label="Scale genes (z-score)", value=True)
                pip_scale_zc = gr.Checkbox(
                    label="Zero-center when scaling (ignored for sparse X unless SCFMS_ALLOW_SPARSE_ZERO_CENTER=1)",
                    value=True,
                )
                pip_scale_max = gr.Number(
                    label="Scale clip max (0 = no clip)", value=10, minimum=0
                )
            with gr.Row():
                pip_n_pcs = gr.Number(label="PCA components", value=50, precision=0)
                pip_pca_solver = gr.Dropdown(
                    choices=["arpack", "randomized", "auto"],
                    value="arpack",
                    label="Scanpy PCA svd_solver",
                )
                pip_scanpy_pca = gr.Checkbox(
                    label="Use Scanpy PCA (sparse-friendly)", value=True
                )
            with gr.Row():
                pip_batch_corr = gr.Radio(
                    choices=[
                        ("None", "none"),
                        ("Harmony 2 (harmonypy)", "harmony2"),
                        ("Scanorama", "scanorama"),
                    ],
                    value="none",
                    label="Batch effect correction",
                    info=(
                        "Applied **after PCA**, before neighbors / UMAP. "
                        "**Harmony 2** → `obsm['X_pca_harmony']`. "
                        "**Scanorama** → `obsm['X_scanorama']` (PCA basis; rows reordered internally for contiguity)."
                    ),
                )
                pip_batch_key = gr.Dropdown(
                    choices=["(none)"],
                    value="(none)",
                    label="Batch column (obs)",
                )
            with gr.Row():
                pip_neighbors = gr.Checkbox(label="Neighbor graph", value=True)
                pip_neighbor_metric = gr.Dropdown(
                    choices=["euclidean", "cosine"],
                    value="euclidean",
                    label="Neighbor metric",
                )
                pip_n_neighbors = gr.Number(
                    label="n_neighbors", value=15, precision=0, minimum=2
                )
            with gr.Row():
                pip_umap = gr.Checkbox(label="Compute UMAP", value=True)
                pip_umap_min_dist = gr.Slider(
                    0.0, 0.99, value=0.5, label="UMAP min_dist"
                )
                pip_umap_spread = gr.Slider(0.25, 4.0, value=1.0, label="UMAP spread")
            umap_color = gr.Dropdown(
                choices=[],
                label="UMAP / obsm scatter color by obs column",
                value=None,
            )
            with gr.Row():
                obsm_scatter_key = gr.Dropdown(
                    choices=["(none)"],
                    value="(none)",
                    label="Plot obsm — key (first 2 dimensions if dim ≥ 2)",
                    scale=3,
                )
                btn_obsm_scatter = gr.Button("Draw obsm scatter / 1D hist", scale=1)
                cancel_obsm_btn = gr.Button(
                    "Cancel obsm plot",
                    variant="stop",
                    scale=1,
                    visible=False,
                )
            plot_obsm = gr.Plot(label="obsm embedding view")
            pip_pipeline_max_cells = gr.Number(
                label="Max cells for pipeline (0 = use all; subsample before PCA/neighbors/UMAP for speed)",
                value=0,
                precision=0,
                minimum=0,
            )
            pip_run_background = gr.Checkbox(
                label="Run in background (survive disconnect — use Job ID below)",
                value=False,
            )
            with gr.Row():
                pipeline_btn = gr.Button(
                    "Run preprocessing pipeline", variant="primary"
                )
                cancel_pipeline_btn = gr.Button(
                    "Cancel pipeline",
                    variant="stop",
                    visible=False,
                )
                umap_recolor_btn = gr.Button("Redraw UMAP colors only")
                cancel_umap_recolor_btn = gr.Button(
                    "Cancel UMAP recolor",
                    variant="stop",
                    visible=False,
                )
            with gr.Row():
                plot_pca = gr.Plot(label="PCA variance ratio")
                plot_umap = gr.Plot(label="UMAP")

        with gr.Accordion("Foundation models (Geneformer, Transcriptformer, scGPT, scVI)", open=False):
            gr.Markdown(
                "Runs **`scripts.generate_embeddings`** on the current AnnData. Pick the **expression matrix** "
                "used as input (counts or normalized — match what the model expects). Results are written to "
                "`obsm` under **`X_<name>`** (default `X_geneformer` / `X_scgpt` / `X_scvi`). "
                "**Slurm GPU** submits a job with **`-p`** (partition) and **`--gres`** for GPUs — use values that match **your cluster** "
                "(see your site docs for partition names and `gres` syntax). "
                "It writes a **new** `.h5ad` under **`<SCFMS_SLURM_EMBED_BASE>/<dataset>/embedded_<model>_<id>.h5ad`** "
                "(`SCFMS_SLURM_EMBED_BASE` defaults to `scfms_job_store/slurm_embeddings`). "
                "Set **Shell lines** to `conda activate …` or `module load` on the compute node. "
                "**Model weights / checkpoint** lists `./models/…` and env defaults for the selected model.\n\n"
                "If **Slurm -c / --mem / -t** are left at **auto** (or below the app's minimum estimate), "
                "the submit path resolves them from the current `AnnData` size and selected model.\n\n"
                "Before every embedding, **compatibility checks** run (gene vocabulary, non-negativity, count-like heuristics) "
                "and a **PDF report** is written into the **active session folder** after you load data (or under **`scfms_reports/`** / "
                "**`SCFMS_COMPAT_REPORT_DIR`** if no session). "
                "Set **`SCFMS_SC_FM_COMPAT_STRICT=0`** to warn-only (no hard stop on errors). "
                "Outputs also store **`uns['scfms_validation']`** on the returned AnnData.\n\n"
                "**Adding a model in code:** call **`preprocess.register_scfm_embedder(name, fn)`** where ``fn(ad_emb, weights_path, n_latent)`` "
                "returns a ``(n_obs, n_latent_dim)`` float array; extend **`scfm_compatibility.FORMAT_REQUIREMENTS`** / "
                "**`MODEL_TRAINING_CORPUS`** for PDF text; the Gradio **Model** radio lists registered names automatically."
            )
            with gr.Row():
                _scfm_names = list_scfm_model_names()
                _scfm_init = _scfm_names[0] if _scfm_names else "geneformer"
                _scfm_w_ch, _scfm_w_val = model_weights_choices_and_value(_scfm_init)
                scfm_model = gr.Radio(
                    _scfm_names,
                    value=_scfm_init,
                    label="Model",
                )
                embed_matrix = gr.Radio(
                    label="Matrix for embedding",
                    choices=[],
                    value=None,
                )
            obsm_key_custom = gr.Textbox(
                label="obsm key suffix or full key",
                placeholder="blank → X_<model>; or e.g. scvi_v1 → X_scvi_v1",
            )
            with gr.Row():
                scfm_ckpt = gr.Dropdown(
                    choices=_scfm_w_ch,
                    value=_scfm_w_val,
                    allow_custom_value=True,
                    label="Model weights / checkpoint",
                    info="Scans ./models and env; type any path or Hugging Face id.",
                    scale=3,
                )
                scfm_n_latent = gr.Number(
                    label="scVI n_latent",
                    value=64,
                    precision=0,
                    minimum=2,
                )
            scfm_run_background = gr.Checkbox(
                label="Run embedding in background (same machine as UI)",
                value=False,
            )
            scfm_slurm_gpu = gr.Checkbox(
                label="Submit Slurm GPU job (writes separate .h5ad on shared FS; overrides in-process background)",
                value=False,
            )
            scfm_dataset_name = gr.Textbox(
                label="Dataset folder name (optional; else server-path stem or 'uploaded')",
                placeholder="e.g. organoid_batch1",
            )
            with gr.Row():
                slurm_partition = gr.Textbox(
                    value=default_slurm_partition(),
                    label="Slurm -p",
                )
                slurm_cpus = gr.Number(
                    value=0, precision=0, label="Slurm -c (0 = auto floor)", minimum=0
                )
                slurm_mem = gr.Textbox(value="auto", label="Slurm --mem (auto = estimate)")
                slurm_time = gr.Textbox(value="auto", label="Slurm -t (auto = estimate)")
            scfms_repo_root_tb = gr.Textbox(
                label="scFMs repo root on cluster (blank = auto from this install)",
                placeholder="/path/to/scFMs",
            )
            slurm_bash_prologue = gr.Textbox(
                label="Bash Lines before Python (conda / modules on GPU node)",
                placeholder="source $HOME/miniconda3/etc/profile.d/conda.sh\nconda activate scvi-env",
                lines=3,
            )
            with gr.Row():
                scfm_btn = gr.Button("Compute embedding → obsm", variant="primary")
                cancel_scfm_btn = gr.Button(
                    "Cancel embedding",
                    variant="stop",
                    visible=False,
                )
            scfms_compat_pdf = gr.File(
                label="Latest compatibility report (PDF)",
                interactive=False,
            )

        save_path = gr.Textbox(
            label="Output .h5ad path (blank → <session>/adata_latest.h5ad)",
            value="",
        )
        with gr.Row():
            save_btn = gr.Button("Save")
            cancel_save_btn = gr.Button(
                "Cancel save",
                variant="stop",
                visible=False,
            )
        save_msg = gr.Textbox(label="Save Status", interactive=False)

        # Wire up
        _load_outputs = [
            state_adata,
            status,
            matrix_spec,
            obs_keep,
            var_keep,
            layers_keep,
            obsm_keep,
            umap_color,
            embed_matrix,
            step_timings_log,
            job_id_input,
            state_session_dir,
            session_path_display,
            dist_plot_source,
            dist_by_obs,
            obsm_scatter_key,
            quick_umap_color,
            pip_batch_key,
        ]
        _dense_poll_outputs = _load_outputs + [dense_load_log]

        _apply_pipeline_outputs = [
            state_adata,
            plot1,
            plot2,
            plot3,
            plot4,
            preview_status,
            plot_pca,
            plot_umap,
            umap_color,
            matrix_spec,
            embed_matrix,
            step_timings_log,
            job_id_input,
            dist_plot_source,
            dist_by_obs,
            obsm_scatter_key,
            quick_umap_color,
            pip_batch_key,
        ]

        def _wire_cancel_pair(
            triggers,
            primary_btn,
            cancel_btn,
            fn,
            inputs,
            outputs,
            *,
            cancel_message: str,
            msg_component_idx: int,
            workload_progress: str = "full",
        ):
            """Show **Cancel** while `fn` runs; cancel clears the queue and writes `cancel_message` to one output."""

            def _busy():
                return gr.update(visible=False), gr.update(visible=True)

            def _idle():
                return gr.update(visible=True), gr.update(visible=False)

            outs = list(outputs)

            def _on_cancel():
                tail = [gr.skip()] * len(outs)
                tail[msg_component_idx] = gr.update(value=cancel_message)
                return (gr.update(visible=True), gr.update(visible=False), *tail)

            run_inputs = [] if inputs is None else inputs
            running = gr.on(
                triggers=triggers,
                fn=_busy,
                inputs=None,
                outputs=[primary_btn, cancel_btn],
                queue=False,
                show_progress="hidden",
            ).then(
                fn,
                inputs=run_inputs,
                outputs=outs,
                show_progress=workload_progress,
            )
            running.then(
                _idle,
                inputs=None,
                outputs=[primary_btn, cancel_btn],
                queue=False,
                show_progress="hidden",
                api_visibility="undocumented",
            )
            cancel_btn.click(
                fn=_on_cancel,
                inputs=None,
                outputs=[primary_btn, cancel_btn] + outs,
                cancels=running,
                queue=False,
                show_progress="hidden",
            )
            return running

        def poll_dense_load(session_dir: str, session_folder_text: str):
            sk = (gr.skip(),) * len(_load_outputs)
            sd = _active_session_dir(session_dir, session_folder_text)
            if not sd:
                return (*sk, gr.skip())
            sdp = Path(sd).expanduser().resolve()

            kind, payload = dense_load_pop_terminal(str(sdp))
            if kind == "error" and payload:
                _try_append_compute_ram_log(
                    str(sdp),
                    "dense_load_failed",
                    ["dense_thread_error", f"error: {payload}"],
                )
                return (*sk, gr.update(value=f"Dense load **failed**:\n{payload}"))
            if kind == "cancelled":
                return (*sk, gr.update(value="**Cancelled** — dense load discarded."))

            st = dense_load_poll(str(sdp))
            if st.get("state") == "idle":
                return (*sk, gr.skip())
            if st.get("state") == "running":
                t0 = float(st.get("started", 0.0))
                elapsed = int(max(0.0, time.time() - t0))
                return (
                    *sk,
                    gr.update(
                        value=(
                            f"Dense load **running** — **{elapsed}s** elapsed… "
                            f"(background thread, full read into RAM; **no extra .h5ad** written)"
                        )
                    ),
                )

            if st.get("state") == "done":
                fin = float(st.get("finished", 0.0))
                adata = dense_ram_take_if_ready(str(sdp))
                if adata is None:
                    return (*sk, gr.skip())
                _dex: List[str] = [
                    "dense_mode: in_memory_only",
                    f"worker_finished_unix_time: {fin}",
                ]
                _try_append_compute_ram_log(
                    str(sdp),
                    "dense_load_finished",
                    _compute_ram_plan_lines(adata, extras=_dex),
                )
                msg = (
                    "**Dense load complete** — full **in-memory** AnnData (no materialized file).\n\n"
                    f"**RAM plan log:** `{sdp / sess_res.COMPUTE_RAM_LOG}`"
                )
                return (
                    *_pack_session_outputs(adata, msg, "", session_dir=str(sdp)),
                    gr.update(
                        value=(
                            "**Finished** — session switched to dense AnnData. "
                            f"See **`{sess_res.COMPUTE_RAM_LOG}`** for GiB hints."
                        )
                    ),
                )

            return (*sk, gr.skip())

        _server_load_inputs = [server_path, open_backed]

        def _load_path_ui_busy():
            return gr.update(visible=False), gr.update(visible=True)

        def _load_path_ui_idle():
            return gr.update(visible=True), gr.update(visible=False)

        def _load_path_cancelled():
            return (
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(
                    value=(
                        "**Cancelled** — load removed from the queue or stopped between steps. "
                        "If the server was already inside `read_h5ad`, that call may still run until the OS returns."
                    )
                ),
            )

        # Swap Load ↔ Cancel; `cancels=` must target the Dependency that ends right after `load_from_server_path`.
        _load_path_running = gr.on(
            triggers=[load_server_btn.click, server_path.submit],
            fn=_load_path_ui_busy,
            inputs=None,
            outputs=[load_server_btn, cancel_load_btn],
            queue=False,
            js=_LOAD_PATH_STATUS_START_JS,
            show_progress="hidden",
        ).then(
            load_from_server_path,
            inputs=_server_load_inputs,
            outputs=_load_outputs,
            show_progress="full",
        )
        _load_path_running.then(
            _load_path_ui_idle,
            inputs=None,
            outputs=[load_server_btn, cancel_load_btn],
            queue=False,
            show_progress="hidden",
            api_visibility="undocumented",
        ).then(
            fn=None,
            inputs=None,
            outputs=None,
            js=_LOAD_PATH_STATUS_STOP_JS,
        )
        cancel_load_btn.click(
            fn=_load_path_cancelled,
            inputs=None,
            outputs=[load_server_btn, cancel_load_btn, status],
            cancels=_load_path_running,
            queue=False,
            show_progress="hidden",
        ).then(
            fn=None,
            inputs=None,
            outputs=None,
            js=_LOAD_PATH_STATUS_STOP_JS,
        )

        def _dense_ui_busy():
            return gr.update(visible=False), gr.update(visible=True)

        def _dense_ui_idle():
            return gr.update(visible=True), gr.update(visible=False)

        def _dense_cancelled(session_dir, session_folder_text):
            sd = _active_session_dir(session_dir, session_folder_text)
            if sd:
                dense_load_cancel(str(Path(sd).expanduser().resolve()))
            return (
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(
                    value="**Cancelled** — dense load discarded. A thread may still finish reading; the result will not be applied."
                ),
            )

        _dense_running = gr.on(
            triggers=[load_dense_btn.click],
            fn=_dense_ui_busy,
            inputs=None,
            outputs=[load_dense_btn, cancel_dense_btn],
            queue=False,
            show_progress="hidden",
        ).then(
            start_dense_load,
            inputs=[state_adata, state_session_dir, session_path_display],
            outputs=[dense_load_log],
            show_progress="minimal",
        )
        _dense_running.then(
            _dense_ui_idle,
            inputs=None,
            outputs=[load_dense_btn, cancel_dense_btn],
            queue=False,
            show_progress="hidden",
            api_visibility="undocumented",
        )
        cancel_dense_btn.click(
            fn=_dense_cancelled,
            inputs=[state_session_dir, session_path_display],
            outputs=[load_dense_btn, cancel_dense_btn, dense_load_log],
            cancels=_dense_running,
            queue=False,
            show_progress="hidden",
        )
        dense_timer.tick(
            fn=poll_dense_load,
            inputs=[state_session_dir, session_path_display],
            outputs=_dense_poll_outputs,
        )

        _wire_cancel_pair(
            [btn_subset_cells.click],
            btn_subset_cells,
            cancel_subset_btn,
            random_subset_session,
            [state_adata, subset_n_cells, subset_seed, state_session_dir],
            _load_outputs,
            cancel_message="**Cancelled** — cell subset stopped.",
            msg_component_idx=1,
            workload_progress="full",
        )

        _wire_cancel_pair(
            [recompute.click],
            recompute,
            cancel_recompute_btn,
            recompute_plots,
            [state_adata, dist_plot_source, dist_by_obs, state_session_dir],
            [plot1, plot2, plot3, plot4, status],
            cancel_message="**Cancelled** — QC histogram recompute stopped.",
            msg_component_idx=4,
            workload_progress="full",
        )

        _wire_cancel_pair(
            [btn_obsm_scatter.click],
            btn_obsm_scatter,
            cancel_obsm_btn,
            preview_obsm_scatter,
            [
                state_adata,
                obsm_scatter_key,
                umap_color,
                state_session_dir,
                umap_fig_w,
                umap_fig_h,
            ],
            [plot_obsm, status],
            cancel_message="**Cancelled** — obsm scatter stopped.",
            msg_component_idx=1,
            workload_progress="full",
        )

        _wire_cancel_pair(
            [btn_quick_umap.click],
            btn_quick_umap,
            cancel_quick_umap_btn,
            run_quick_umap_plot,
            [
                state_adata,
                dist_plot_source,
                quick_max_cells,
                quick_n_pcs,
                quick_n_neighbors,
                quick_min_dist,
                quick_umap_color,
                state_session_dir,
                umap_fig_w,
                umap_fig_h,
            ],
            [plot_quick_umap, preview_status],
            cancel_message="**Cancelled** — quick UMAP stopped.",
            msg_component_idx=1,
            workload_progress="full",
        )

        _wire_cancel_pair(
            [apply_btn.click],
            apply_btn,
            cancel_apply_btn,
            apply_and_preview,
            [
                state_adata,
                matrix_spec,
                obs_keep,
                obs_map,
                var_keep,
                var_map,
                layers_keep,
                obsm_keep,
                keep_raw,
                set_raw_from_x,
                state_session_dir,
                dist_plot_source,
                dist_by_obs,
                pip_batch_key,
                embed_matrix,
            ],
            _apply_pipeline_outputs,
            cancel_message="**Cancelled** — apply edits stopped.",
            msg_component_idx=5,
            workload_progress="full",
        )

        _wire_cancel_pair(
            [pipeline_btn.click],
            pipeline_btn,
            cancel_pipeline_btn,
            run_full_pipeline,
            [
                state_adata,
                pip_norm_total,
                pip_target_sum,
                pip_log1p,
                pip_fc_min_counts,
                pip_fc_min_genes,
                pip_fg_min_cells,
                pip_hvg,
                pip_n_top_hvg,
                pip_hvg_flavor,
                pip_subset_hvg,
                pip_scale,
                pip_scale_zc,
                pip_scale_max,
                pip_n_pcs,
                pip_pca_solver,
                pip_scanpy_pca,
                pip_neighbors,
                pip_neighbor_metric,
                pip_n_neighbors,
                pip_umap,
                pip_umap_min_dist,
                pip_umap_spread,
                umap_color,
                umap_fig_w,
                umap_fig_h,
                pip_pipeline_max_cells,
                pip_use_raw,
                pip_run_background,
                state_session_dir,
                dist_plot_source,
                dist_by_obs,
                pip_batch_corr,
                pip_batch_key,
                matrix_spec,
                embed_matrix,
            ],
            _apply_pipeline_outputs,
            cancel_message="**Cancelled** — preprocessing pipeline stopped.",
            msg_component_idx=5,
            workload_progress="full",
        )

        scfm_model.change(
            model_weights_gr_update,
            inputs=[scfm_model],
            outputs=[scfm_ckpt],
        )

        _scfm_embed_outputs = [
            state_adata,
            preview_status,
            obsm_keep,
            matrix_spec,
            embed_matrix,
            dist_plot_source,
            step_timings_log,
            job_id_input,
            scfms_compat_pdf,
        ]
        _wire_cancel_pair(
            [scfm_btn.click],
            scfm_btn,
            cancel_scfm_btn,
            run_scfm_embed,
            [
                state_adata,
                scfm_model,
                embed_matrix,
                matrix_spec,
                dist_plot_source,
                obsm_key_custom,
                scfm_ckpt,
                scfm_n_latent,
                obsm_keep,
                scfm_run_background,
                scfm_slurm_gpu,
                server_path,
                scfm_dataset_name,
                slurm_partition,
                slurm_cpus,
                slurm_mem,
                slurm_time,
                slurm_bash_prologue,
                scfms_repo_root_tb,
                state_session_dir,
            ],
            _scfm_embed_outputs,
            cancel_message="**Cancelled** — embedding / Slurm submission stopped.",
            msg_component_idx=1,
            workload_progress="full",
        )

        _wire_cancel_pair(
            [refresh_job_btn.click],
            refresh_job_btn,
            cancel_refresh_job_btn,
            refresh_job_ui,
            [job_id_input],
            [job_status_log],
            cancel_message="**Cancelled** — job status refresh stopped.",
            msg_component_idx=0,
            workload_progress="minimal",
        )
        _wire_cancel_pair(
            [load_job_result_btn.click],
            load_job_result_btn,
            cancel_load_job_btn,
            load_job_into_session,
            [job_id_input],
            _load_outputs,
            cancel_message="**Cancelled** — loading job result stopped.",
            msg_component_idx=1,
            workload_progress="full",
        )
        _wire_cancel_pair(
            [resume_btn.click, resume_path_input.submit],
            resume_btn,
            cancel_resume_btn,
            load_resume_path,
            [resume_path_input, open_backed],
            _load_outputs,
            cancel_message="**Cancelled** — resume load stopped.",
            msg_component_idx=1,
            workload_progress="full",
        )
        _wire_cancel_pair(
            [list_jobs_btn.click],
            list_jobs_btn,
            cancel_list_jobs_btn,
            list_recent_jobs_ui,
            None,
            [recent_jobs_list],
            cancel_message="**Cancelled** — listing jobs stopped.",
            msg_component_idx=0,
            workload_progress="minimal",
        )
        job_timer.tick(
            fn=poll_job_ui,
            inputs=[job_id_input, job_poll_enable],
            outputs=[job_status_log],
        )

        _wire_cancel_pair(
            [umap_recolor_btn.click],
            umap_recolor_btn,
            cancel_umap_recolor_btn,
            redraw_umap_only,
            [
                state_adata,
                umap_color,
                state_session_dir,
                umap_fig_w,
                umap_fig_h,
            ],
            [plot_umap, preview_status],
            cancel_message="**Cancelled** — UMAP recolor stopped.",
            msg_component_idx=1,
            workload_progress="minimal",
        )

        _wire_cancel_pair(
            [save_btn.click],
            save_btn,
            cancel_save_btn,
            save_file,
            [state_adata, save_path, state_session_dir],
            [save_msg],
            cancel_message="**Cancelled** — save stopped (file may be partial if write had started).",
            msg_component_idx=0,
            workload_progress="minimal",
        )

    return demo


if __name__ == "__main__":
    ui = build_ui()
    launch_gradio_demo(
        ui,
        default_port=int(os.environ.get("PORT", "7861")),
        app_label="scFMs preprocess",
    )
