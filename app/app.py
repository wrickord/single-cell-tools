import os
import re
import sys
import tempfile
import time
import json
import multiprocessing as mp
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
import gradio as gr
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy import sparse as sp
from scipy import stats
from umap import UMAP
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

import background_jobs as bgjobs
import benchmark as bench
from background_jobs import validate_server_read_path
from gradio_config import launch_gradio_demo, repo_root, runtime_info_markdown
from scripts.dataset_sources import (
    download_dataset,
    managed_data_root,
    sanitize_dataset_label,
)
from scripts.generate_embeddings import load_adata
import session_results as sess_res
import preprocess as pre


def _session_src_key(
    file_obj,
    server_h5ad_path: str,
) -> Tuple[str, str]:
    sp = (server_h5ad_path or "").strip()
    if sp:
        return ("server", str(Path(sp).expanduser().resolve()))
    if file_obj is None:
        return ("none", "")
    read_fn = getattr(file_obj, "read", None)
    if callable(read_fn):
        return ("upload", str(getattr(file_obj, "name", "upload")))
    return ("path", str(Path(str(file_obj)).resolve()))


def _load_adata_from_inputs(
    file_obj,
    server_h5ad_path: str,
    transpose: bool,
    *,
    backed: bool = False,
    preferred_h5ad_path: str = "",
    sess_bundle: Optional[Dict[str, Any]] = None,
    consume_dense_ram: bool = False,
) -> ad.AnnData:
    if consume_dense_ram and isinstance(sess_bundle, dict):
        sd = str(sess_bundle.get("dir") or "").strip()
        if sd:
            ram = pre.dense_ram_take_if_ready(sd)
            if ram is not None:
                return ram
    sp = (server_h5ad_path or "").strip()
    if sp:
        if preferred_h5ad_path:
            p = validate_server_read_path(preferred_h5ad_path)
        else:
            p = validate_server_read_path(sp)
        if backed and p.suffix.lower() == ".h5ad":
            out = ad.read_h5ad(str(p), backed="r")
            try:
                out.uns["_scfms_densify_src"] = str(p)
            except Exception:
                pass
            return out
        return load_adata(str(p), transpose=transpose)
    if file_obj is None:
        raise ValueError(
            "Set a server path to an .h5ad / .csv / .tsv."
        )
    read_fn = getattr(file_obj, "read", None)
    if callable(read_fn):
        suffix = Path(getattr(file_obj, "name", "upload.bin")).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(read_fn())
            tmp = Path(f.name)
        try:
            return load_adata(str(tmp), transpose=transpose)
        finally:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
    src = Path(str(file_obj))
    if not src.is_file():
        raise FileNotFoundError(f"File not found: {src}")
    return load_adata(str(src), transpose=transpose)


def _resolved_dense_h5ad_from_bundle(sess_bundle: Optional[Dict[str, Any]]) -> str:
    if not isinstance(sess_bundle, dict):
        return ""
    dense = str(sess_bundle.get("dense_h5ad_path") or "").strip()
    if dense and Path(dense).is_file():
        return dense
    session_dir = str(sess_bundle.get("dir") or "").strip()
    if not session_dir:
        return ""
    candidate = Path(session_dir).expanduser() / pre.DENSE_MATERIALIZED_H5AD
    if candidate.is_file():
        return str(candidate.resolve())
    return ""


def _preferred_h5ad_path_for_session(
    server_h5ad_path: str,
    sess_bundle: Optional[Dict[str, Any]],
    *,
    prefer_dense: bool = True,
) -> str:
    if prefer_dense:
        dense = _resolved_dense_h5ad_from_bundle(sess_bundle)
        if dense:
            return dense
    sp = str(server_h5ad_path or "").strip()
    if not sp:
        return ""
    p = validate_server_read_path(sp)
    return str(p)


def _bundle_with_materialized_paths(
    bundle: Optional[Dict[str, Any]],
    *,
    source_h5ad_path: str = "",
) -> Dict[str, Any]:
    out = dict(bundle or {})
    if source_h5ad_path:
        out["source_h5ad_path"] = str(Path(source_h5ad_path).expanduser().resolve())
    dense = _resolved_dense_h5ad_from_bundle(out)
    if dense:
        out["dense_h5ad_path"] = dense
    else:
        out.pop("dense_h5ad_path", None)
    return out


def _dense_status_from_bundle(sess_bundle: Optional[Dict[str, Any]]) -> str:
    if not isinstance(sess_bundle, dict):
        return "Load a dataset first."
    session_dir = str(sess_bundle.get("dir") or "").strip()
    if not session_dir:
        return "Load a dataset first."
    sdp = Path(session_dir).expanduser().resolve()
    dense_path = sdp / pre.DENSE_MATERIALIZED_H5AD
    if dense_path.is_file():
        return (
            f"Legacy materialized file on disk: `{dense_path}`\n"
            "`Load Data` will prefer it until removed. New **Load Dense** uses RAM only (no new copy)."
        )
    st = pre.dense_load_poll(str(sdp))
    if st.get("state") == "running":
        elapsed = int(max(0.0, time.time() - float(st.get("started", 0.0))))
        return (
            f"Dense load running — **{elapsed}s** (background thread, full read into RAM; no extra `.h5ad`). "
            "Click **Load Data** when finished to attach the dense object to this session."
        )
    if st.get("state") == "done":
        return (
            "Dense load finished in RAM — click **Load Data** to use the full matrix for this session."
        )
    if st.get("state") == "error":
        kind, payload = pre.dense_load_pop_terminal(str(sdp))
        err = str(payload or st.get("error") or "").strip() or "unknown error"
        if kind == "error":
            return f"Dense load failed:\n{err}"
        return f"Dense load failed:\n{err}"
    return (
        "Current dataset is backed-only.\n"
        "Click **Load Dense** for a full in-RAM read in the background (no disk copy). "
        "Then click **Load Data** to apply."
    )


def start_dense_load_ui(server_h5ad_path, sess_bundle):
    try:
        if not isinstance(sess_bundle, dict):
            return "Load a dataset first.", sess_bundle
        session_dir = str(sess_bundle.get("dir") or "").strip()
        if not session_dir:
            return "Load a dataset first.", sess_bundle
        source_path = str(sess_bundle.get("source_h5ad_path") or "").strip()
        if not source_path:
            source_path = _preferred_h5ad_path_for_session(
                server_h5ad_path,
                sess_bundle,
                prefer_dense=False,
            )
        if not source_path:
            return "Set a server `.h5ad` path first.", sess_bundle
        src = validate_server_read_path(source_path)
        if src.suffix.lower() != ".h5ad":
            return "Load Dense currently supports only `.h5ad` inputs.", sess_bundle
        sdp = Path(session_dir).expanduser().resolve()
        sdp.mkdir(parents=True, exist_ok=True)
        dense_path = sdp / pre.DENSE_MATERIALIZED_H5AD
        if dense_path.is_file():
            new_bundle = _bundle_with_materialized_paths(
                sess_bundle,
                source_h5ad_path=str(src),
            )
            return (
                f"Legacy materialized file exists: `{dense_path}` — `Load Data` already prefers it.\n"
                "Start **Load Dense** again to drop that file and reload fully in RAM (no new copy).",
                new_bundle,
            )
        poll = pre.dense_load_poll(str(sdp))
        if poll.get("state") == "running":
            return (
                "Dense load **already running** — wait for it to finish, then click **Load Data**.",
                sess_bundle,
            )
        ok, err = pre.dense_load_start(str(sdp), str(src))
        if not ok:
            return (f"Could not start dense load: {err}", sess_bundle)
        new_bundle = _bundle_with_materialized_paths(
            sess_bundle,
            source_h5ad_path=str(src),
        )
        new_bundle.pop("dense_h5ad_path", None)
        return (
            "Started **Load Dense** — background thread reads the full `.h5ad` into **RAM** "
            "(no `adata_dense_materialized.h5ad`).\n"
            "When status says finished, click **Load Data** to attach the dense object.",
            new_bundle,
        )
    except Exception as e:
        return f"Dense load error: {e}", sess_bundle


def refresh_dense_load_ui(sess_bundle):
    try:
        new_bundle = _bundle_with_materialized_paths(sess_bundle)
        return _dense_status_from_bundle(new_bundle), new_bundle
    except Exception as e:
        return f"Dense status refresh error: {e}", sess_bundle


def _norm_src_key(raw: Any) -> Optional[Tuple[str, str]]:
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        return (str(raw[0]), str(raw[1]))
    return None


def _ensure_session(
    sess_bundle: Optional[Dict[str, Any]],
    src_key: Tuple[str, str],
    adata: ad.AnnData,
) -> Tuple[str, Dict[str, Any]]:
    prev = (
        _norm_src_key(sess_bundle.get("key")) if isinstance(sess_bundle, dict) else None
    )
    d = (sess_bundle or {}).get("dir") if isinstance(sess_bundle, dict) else None
    if prev == src_key and d and str(d).strip():
        return str(d), sess_bundle  # type: ignore[return-value]

    if src_key[0] in ("server", "path") and src_key[1]:
        label = Path(src_key[1]).stem
    elif src_key[0] == "upload" and src_key[1]:
        label = Path(src_key[1]).stem
    else:
        label = "embedding_run"
    session: Path
    if src_key[0] in ("server", "path") and src_key[1]:
        ctx = _dataset_context_from_source_path(src_key[1])
        if ctx is not None:
            session = _create_session_dir_with_meta(
                label,
                source_kind=src_key[0],
                source_path=src_key[1] or label,
                n_obs=int(adata.n_obs),
                n_vars=int(adata.n_vars),
            )
        else:
            session = sess_res.create_dataset_session(
                label,
                source_kind=src_key[0],
                source_path=src_key[1] or label,
                n_obs=int(adata.n_obs),
                n_vars=int(adata.n_vars),
            )
    else:
        session = sess_res.create_dataset_session(
            label,
            source_kind=src_key[0],
            source_path=src_key[1] or label,
            n_obs=int(adata.n_obs),
            n_vars=int(adata.n_vars),
        )
    bundle = {"key": list(src_key), "dir": str(session)}
    return str(session), bundle


_DATASET_PICKER_EXTS = (".h5ad",)


def _dataset_search_roots() -> list[Path]:
    raw = os.environ.get("SCFMS_SERVER_DATA_DIRS", "").strip()
    roots: list[Path] = []
    if raw:
        for chunk in raw.split("|"):
            s = chunk.strip()
            if s:
                roots.append(Path(s).expanduser())
        roots.append(managed_data_root())
    else:
        roots.extend(
            [
                repo_root() / ".data",
                managed_data_root(),
                repo_root() / "data",
            ]
        )
    out: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        try:
            rp = root.resolve()
        except OSError:
            continue
        key = str(rp)
        if key not in seen:
            seen.add(key)
            out.append(rp)
    return out


def _dataset_context_from_source_path(raw_path: str) -> Optional[Dict[str, Path]]:
    try:
        source = Path(raw_path).expanduser().resolve()
    except OSError:
        return None
    for base in _dataset_search_roots():
        try:
            rel = source.relative_to(base)
        except ValueError:
            continue
        parts = rel.parts
        if len(parts) < 2:
            continue
        if parts[1] not in ("data", "processed"):
            continue
        dataset_root = (base / parts[0]).resolve()
        data_dir = dataset_root / "data"
        processed_dir = dataset_root / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)
        return {
            "search_root": base,
            "dataset_root": dataset_root,
            "data_dir": data_dir.resolve(),
            "processed_dir": processed_dir.resolve(),
        }
    return None


def _create_session_dir_with_meta(
    dataset_label: str,
    *,
    source_kind: str,
    source_path: str,
    n_obs: int,
    n_vars: int,
) -> Path:
    base = sess_res.session_storage_dir(source_path, dataset_label)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe = sanitize_dataset_label(dataset_label, "dataset")[:60]
    d = base / f"{ts}_{safe}"
    d.mkdir(parents=True, exist_ok=False)
    (d / "plots").mkdir(exist_ok=True)
    meta: Dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_label": dataset_label,
        "source_kind": source_kind,
        "source_path": source_path,
        "n_obs": int(n_obs),
        "n_vars": int(n_vars),
    }
    (d / "session_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return d.resolve()


def _find_first_h5ad_under(root: Path) -> str:
    try:
        for p in sorted(root.rglob("*.h5ad")):
            if p.is_file():
                return str(p.resolve())
    except OSError:
        return ""
    return ""


def _download_method_help(method: str):
    key = str(method or "direct_url")
    if key == "zenodo_record":
        return (
            "Enter a Zenodo record id. Optional extra field: shell-style file filter like `*.h5ad`."
        )
    if key == "geo_accession":
        return (
            "Enter a GEO accession like `GSE252510`. Files download under the managed "
            "dataset `.data/<dataset>/data/` folder and then auto-convert to `.h5ad` when possible."
        )
    if key == "cellxgene_url":
        return "Enter a direct CELLxGENE asset URL. Optional extra field overrides the saved filename."
    return "Enter a direct file URL. Optional extra field overrides the saved filename."


def _format_download_status(res: Dict[str, Any]) -> tuple[str, str]:
    ds_root = Path(res["dataset_root"])
    first_h5ad = _find_first_h5ad_under(ds_root)
    downloaded = [str(Path(p).resolve()) for p in (res.get("downloaded") or [])]
    extracted = [str(Path(p).resolve()) for p in (res.get("extracted") or [])]
    status_lines = [
        f"Downloaded dataset into `{ds_root}`",
        f"- data dir: `{res['data_dir']}`",
        f"- processed dir: `{res['processed_dir']}`",
        f"- files downloaded: {len(downloaded)}",
    ]
    if extracted:
        status_lines.append(f"- extracted outputs: {len(extracted)}")
    if res.get("attempted"):
        status_lines.append(f"- conversion attempts: {res['attempted']}")
    created_h5ad = [str(Path(p).resolve()) for p in (res.get("created_h5ad") or [])]
    if created_h5ad:
        status_lines.append(f"- new `.h5ad` files created: {len(created_h5ad)}")
    if first_h5ad:
        status_lines.append(f"- first discovered .h5ad: `{first_h5ad}`")
    else:
        status_lines.append("- no `.h5ad` discovered yet; raw files may still need processing.")
    conversion_errors = [str(x) for x in (res.get("conversion_errors") or [])]
    if conversion_errors:
        status_lines.append(f"- conversion notes: {len(conversion_errors)}")
    if downloaded:
        status_lines.append("")
        status_lines.extend(f"  - `{p}`" for p in downloaded[:8])
        if len(downloaded) > 8:
            status_lines.append(f"  - ... {len(downloaded) - 8} more")
    if created_h5ad:
        status_lines.append("")
        status_lines.extend(f"  - new h5ad: `{p}`" for p in created_h5ad[:8])
        if len(created_h5ad) > 8:
            status_lines.append(f"  - ... {len(created_h5ad) - 8} more")
    if conversion_errors:
        status_lines.append("")
        status_lines.extend(f"  - note: `{msg}`" for msg in conversion_errors[:6])
        if len(conversion_errors) > 6:
            status_lines.append(f"  - ... {len(conversion_errors) - 6} more")
    return "\n".join(status_lines), first_h5ad


def _download_job_root() -> Path:
    root = bgjobs.jobs_root() / "downloads"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _download_job_status_path(job_id: str) -> Path:
    return _download_job_root() / job_id / "status.json"


def _download_worker_main(
    method: str,
    dataset_name: str,
    identifier: str,
    extra: str,
    status_path: str,
) -> None:
    import time as _time
    import traceback

    st = Path(status_path)
    try:
        result = download_dataset(method, dataset_name, identifier, extra=extra)
        payload = {
            "ok": True,
            "finished": _time.time(),
            "result": result,
        }
        st.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception:
        err = traceback.format_exc()
        st.write_text(
            json.dumps(
                {
                    "ok": False,
                    "finished": _time.time(),
                    "error": err,
                },
                indent=2,
            ),
            encoding="utf-8",
        )


def start_download_dataset_ui(download_method, download_name, download_identifier, download_extra):
    try:
        method = str(download_method or "direct_url").strip() or "direct_url"
        dataset_name = str(download_name or "").strip()
        identifier = str(download_identifier or "").strip()
        extra = str(download_extra or "").strip()
        if not dataset_name:
            return "Set a dataset name first.", None
        if not identifier:
            return "Set a download source first.", None
        job_id = f"download_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        job_dir = _download_job_root() / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        status_path = job_dir / "status.json"
        ctx = mp.get_context("spawn")
        proc = ctx.Process(
            target=_download_worker_main,
            args=(method, dataset_name, identifier, extra, str(status_path)),
            daemon=False,
        )
        proc.start()
        status_path.write_text(
            json.dumps(
                {
                    "status": "running",
                    "pid": int(proc.pid or 0),
                    "method": method,
                    "dataset_name": dataset_name,
                    "identifier": identifier,
                    "extra": extra,
                    "started": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return (
            f"Started dataset download in background.\n"
            f"Job id: {job_id}\n"
            f"PID: {proc.pid}\n"
            f"Method: {method}\n"
            f"Dataset name: {dataset_name}",
            {"job_id": job_id},
        )
    except Exception as e:
        return f"Download error: {e}", None


def refresh_download_dataset_ui(download_state):
    if not isinstance(download_state, dict):
        return (
            "No background download started yet.",
            gr.update(choices=_list_data_options()),
            _dataset_picker_info(),
            gr.skip(),
            None,
        )
    job_id = str(download_state.get("job_id") or "").strip()
    if not job_id:
        return (
            "No background download started yet.",
            gr.update(choices=_list_data_options()),
            _dataset_picker_info(),
            gr.skip(),
            None,
        )
    st_path = _download_job_status_path(job_id)
    if not st_path.is_file():
        return (
            f"Download status file missing for {job_id}.",
            gr.update(choices=_list_data_options()),
            _dataset_picker_info(),
            gr.skip(),
            download_state,
        )
    try:
        meta = json.loads(st_path.read_text(encoding="utf-8"))
    except Exception as e:
        return (
            f"Download status unreadable for {job_id}: {e}",
            gr.update(choices=_list_data_options()),
            _dataset_picker_info(),
            gr.skip(),
            download_state,
        )
    if meta.get("status") == "running":
        pid = int(meta.get("pid", 0) or 0)
        if pid > 0 and pre._pid_alive(pid):
            return (
                f"Dataset download running in background.\nJob id: {job_id}\nPID: {pid}",
                gr.update(choices=_list_data_options()),
                _dataset_picker_info(),
                gr.skip(),
                download_state,
            )
    if meta.get("ok") is True:
        res = meta.get("result") or {}
        status_txt, first_h5ad = _format_download_status(res)
        return (
            status_txt,
            gr.update(choices=_list_data_options(), value=first_h5ad or None),
            _dataset_picker_info(),
            gr.update(value=first_h5ad) if first_h5ad else gr.skip(),
            download_state,
        )
    if meta.get("ok") is False:
        err = str(meta.get("error") or "").strip() or "unknown error"
        return (
            f"Download failed for {job_id}:\n{err}",
            gr.update(choices=_list_data_options()),
            _dataset_picker_info(),
            gr.skip(),
            download_state,
        )
    return (
        f"Download status for {job_id} is still pending.",
        gr.update(choices=_list_data_options()),
        _dataset_picker_info(),
        gr.skip(),
        download_state,
    )


def _list_data_options() -> list[tuple[str, str]]:
    opts: list[tuple[str, str]] = []
    seen: set[str] = set()
    for base in _dataset_search_roots():
        if not base.exists() or not base.is_dir():
            continue
        for root, _dirs, files in os.walk(base, followlinks=True):
            root_path = Path(root)
            for name in sorted(files):
                path = root_path / name
                if path.suffix.lower() not in _DATASET_PICKER_EXTS:
                    continue
                resolved = str(path.resolve())
                if resolved in seen:
                    continue
                seen.add(resolved)
                try:
                    rel = path.relative_to(base)
                    label = rel.as_posix()
                except ValueError:
                    label = path.name
                opts.append((label, resolved))
    return opts


def _dataset_picker_info() -> str:
    roots = _dataset_search_roots()
    found = _list_data_options()
    root_lines = [f"- `{p}`" for p in roots]
    if not root_lines:
        root_lines = ["- `(none configured)`"]
    return (
        f"Datasets found: **{len(found)}**\n\n"
        "Search roots:\n"
        + "\n".join(root_lines)
        + "\n\nSet `SCFMS_SERVER_DATA_DIRS` to pipe-separated directories on the server to change this."
    )


def _column_choices(adata: ad.AnnData) -> list[str]:
    cols = [str(c) for c in adata.obs.columns]
    if "cell_type" in cols:
        cols.insert(0, cols.pop(cols.index("cell_type")))
    return cols


def _obsm_choices(adata: ad.AnnData) -> list[str]:
    keys = [str(k) for k in adata.obsm.keys()]
    return keys


def _matrix_choices(adata: ad.AnnData) -> list[str]:
    try:
        return list(pre._list_matrix_options(adata))
    except Exception:
        return ["X"]


def _embedding_matrix_choices(adata: ad.AnnData) -> list[str]:
    try:
        return list(pre._list_embedding_matrix_options(adata))
    except Exception:
        return ["X"]


def _unique_obs_values(adata: ad.AnnData, column: str, *, max_values: int = 500) -> list[str]:
    if not column or column == "(none)" or column not in adata.obs.columns:
        return []
    series = adata.obs[column]
    vals = pd.Series(series).dropna()
    if vals.empty:
        return []
    try:
        ordered = pd.unique(vals.astype(str))
    except Exception:
        ordered = pd.unique(vals.map(str))
    out = [str(v) for v in ordered if str(v).strip()]
    out.sort()
    return out[:max_values]


def _load_obs_value_choices(
    server_h5ad_path: str,
    transpose: bool,
    sess_bundle: Optional[Dict[str, Any]],
    view_bundle: Optional[Dict[str, Any]],
    group_col: str,
) -> list[str]:
    gcol = str(group_col or "").strip()
    if not gcol or gcol == "(none)":
        return []
    view_path = _resolved_h5ad_path_from_bundle(view_bundle)
    adata: Optional[ad.AnnData] = None
    if view_path:
        adata = ad.read_h5ad(view_path, backed="r")
        try:
            return _unique_obs_values(adata, gcol)
        finally:
            _close_adata_handle(adata)
    preferred = _preferred_h5ad_path_for_session(
        server_h5ad_path,
        sess_bundle,
        prefer_dense=True,
    )
    if not preferred:
        return []
        adata = _load_adata_from_inputs(
            None,
            server_h5ad_path,
            transpose,
            backed=True,
            preferred_h5ad_path=preferred,
            sess_bundle=sess_bundle,
        )
        try:
            return _unique_obs_values(adata, gcol)
    finally:
        _close_adata_handle(adata)


def update_de_value_choices_ui(
    server_h5ad_path,
    transpose,
    sess_bundle,
    view_bundle,
    group_col,
    current_fg,
    current_bg,
):
    try:
        values = _load_obs_value_choices(
            str(server_h5ad_path or ""),
            bool(transpose),
            sess_bundle,
            view_bundle,
            str(group_col or ""),
        )
        fg_now = [str(x).strip() for x in (current_fg or []) if str(x).strip()]
        bg_now = [str(x).strip() for x in (current_bg or []) if str(x).strip()]
        fg_keep = [x for x in fg_now if x in values]
        bg_keep = [x for x in bg_now if x in values]
        return (
            gr.update(choices=values, value=fg_keep),
            gr.update(choices=values, value=bg_keep),
        )
    except Exception:
        return (
            gr.update(choices=[], value=[]),
            gr.update(choices=[], value=[]),
        )


def _normalize_groupby_input(raw: Any) -> list[str]:
    if isinstance(raw, (list, tuple)):
        return [str(x).strip() for x in raw if str(x).strip()]
    try:
        return [x.strip() for x in str(raw or "").split(",") if x.strip()]
    except Exception:
        return []


def load_data_options(server_h5ad_path, transpose, sess_bundle):
    try:
        preferred = _preferred_h5ad_path_for_session(
            server_h5ad_path,
            sess_bundle,
            prefer_dense=False,
        )
        adata = _load_adata_from_inputs(
            None,
            server_h5ad_path,
            transpose,
            backed=True,
            preferred_h5ad_path=preferred,
            sess_bundle=sess_bundle,
            consume_dense_ram=True,
        )
        try:
            src_key = _session_src_key(None, server_h5ad_path)
            session_dir, bundle = _ensure_session(sess_bundle, src_key, adata)
            bundle = _bundle_with_materialized_paths(
                bundle,
                source_h5ad_path=preferred,
            )
            obs_cols = _column_choices(adata)
            obsm_cols = _obsm_choices(adata)
            matrix_choices = _matrix_choices(adata)
            embed_choices = _embedding_matrix_choices(adata)
            color_default = "cell_type" if "cell_type" in obs_cols else "(none)"
            facet_default = "dataset" if "dataset" in obs_cols else "(none)"
            group_default = "cell_type" if "cell_type" in obs_cols else "(none)"
            obsm_default = "(default)"
            de_value_choices = _unique_obs_values(adata, group_default)
            mode_label = "backed" if pre.adata_is_backed(adata) else "dense"
            status = (
                f"Loaded dataset metadata in {mode_label} mode: n_obs={adata.n_obs:,}, n_vars={adata.n_vars:,}\n"
                f"obs columns={len(obs_cols)}, obsm keys={len(obsm_cols)}\n"
                f"default output dir: `{session_dir}`\n"
                f"{_dense_status_from_bundle(bundle)}"
            )
            return (
                status,
                bundle,
                gr.update(value=session_dir),
                gr.update(choices=["(none)"] + obs_cols, value=color_default),
                gr.update(choices=["(none)"] + obs_cols, value=facet_default),
                gr.update(choices=obs_cols, value=[]),
                gr.update(choices=["(default)"] + obsm_cols, value=obsm_default),
                gr.update(choices=obs_cols, value=[]),
                gr.update(choices=["(none)"] + obs_cols, value=group_default),
                gr.update(choices=["(none)"] + obs_cols, value="(none)"),
                gr.update(
                    choices=matrix_choices,
                    value=pre._pick_matrix_source_value("X", matrix_choices),
                ),
                gr.update(
                    choices=embed_choices,
                    value=pre._pick_matrix_source_value("X", embed_choices),
                ),
                _dense_status_from_bundle(bundle),
                gr.update(choices=de_value_choices, value=[]),
                gr.update(choices=de_value_choices, value=[]),
                gr.update(
                    choices=matrix_choices,
                    value=pre._pick_matrix_source_value("X", matrix_choices),
                ),
                gr.update(choices=["(none)"] + obs_cols, value="(none)"),
            )
        finally:
            _close_adata_handle(adata)
    except Exception as e:
        return (
            f"Error loading data for options: {e}",
            sess_bundle,
            gr.skip(),
            gr.update(choices=["(none)"], value="(none)"),
            gr.update(choices=["(none)"], value="(none)"),
            gr.update(choices=[], value=[]),
            gr.update(choices=["(default)"], value="(default)"),
            gr.update(choices=[], value=[]),
            gr.update(choices=["(none)"], value="(none)"),
            gr.update(choices=["(none)"], value="(none)"),
            gr.update(choices=["X"], value="X"),
            gr.update(choices=["X"], value="X"),
            "Load a dataset first.",
            gr.update(choices=[], value=[]),
            gr.update(choices=[], value=[]),
            gr.update(choices=["X"], value="X"),
            gr.update(choices=["(none)"], value="(none)"),
        )


def _close_adata_handle(adata_obj: Any) -> None:
    fobj = getattr(adata_obj, "file", None)
    if fobj is not None:
        try:
            fobj.close()
        except Exception:
            pass


def _copy_matrix_payload(M: Any) -> Any:
    if sp.issparse(M):
        return M.copy()
    if hasattr(M, "copy"):
        try:
            return M.copy()
        except Exception:
            pass
    return np.asarray(M).copy()


def _resolved_h5ad_path_from_bundle(bundle: Optional[Dict[str, Any]]) -> str:
    if not isinstance(bundle, dict):
        return ""
    path = str(bundle.get("path") or "").strip()
    if path and Path(path).is_file():
        return path
    jid = str(bundle.get("job_id") or "").strip()
    if jid:
        rp = bgjobs.result_h5ad_path(jid)
        if rp is not None and rp.is_file():
            return str(rp.resolve())
    return ""


def _add_catalog_entry(
    catalog: Dict[str, Dict[str, Any]],
    *,
    final_spec: str,
    display_label: str,
    kind: str,
    source_spec: Optional[str] = None,
    path: str = "",
    key: str = "",
    origin: str = "",
) -> None:
    if final_spec in catalog:
        return
    catalog[final_spec] = {
        "final_spec": final_spec,
        "display_label": display_label,
        "kind": kind,
        "source_spec": source_spec or final_spec,
        "path": path,
        "key": key,
        "origin": origin,
    }


def _build_benchmark_source_catalog(
    base_adata: ad.AnnData,
    *,
    view_bundle: Optional[Dict[str, Any]] = None,
    run_bundle: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    catalog: Dict[str, Dict[str, Any]] = {}
    for spec in pre._list_plot_source_options(base_adata):
        _add_catalog_entry(
            catalog,
            final_spec=str(spec),
            display_label=str(spec),
            kind="base_spec",
            source_spec=str(spec),
            origin="loaded dataset",
        )

    def add_external_obsm_sources(path_str: str, *, origin: str) -> None:
        raw = str(path_str or "").strip()
        if not raw:
            return
        p = Path(raw).expanduser().resolve()
        if not p.is_file():
            return
        adx = None
        try:
            adx = ad.read_h5ad(str(p), backed="r")
            keys = [str(k) for k in adx.obsm.keys()]
            view_spec = str(getattr(adx, "uns", {}).get("scfms_view_matrix_spec") or "").strip()
            for key in keys:
                base_spec = f"obsm:{key}"
                final_spec = base_spec
                if final_spec in catalog:
                    suffix = re.sub(r"[^a-zA-Z0-9_]+", "_", origin.strip().lower()).strip("_") or "session"
                    final_spec = f"obsm:{key}__{suffix}"
                label = final_spec
                if view_spec and origin == "current view":
                    label = f"{final_spec} [{origin}: {view_spec}]"
                else:
                    label = f"{final_spec} [{origin}]"
                _add_catalog_entry(
                    catalog,
                    final_spec=final_spec,
                    display_label=label,
                    kind="external_obsm",
                    path=str(p),
                    key=str(key),
                    origin=origin,
                )
        except Exception:
            return
        finally:
            if adx is not None:
                _close_adata_handle(adx)

    add_external_obsm_sources(
        _resolved_h5ad_path_from_bundle(view_bundle),
        origin="current view",
    )
    add_external_obsm_sources(
        _resolved_h5ad_path_from_bundle(run_bundle),
        origin="embedding result",
    )
    return catalog


def refresh_benchmark_sources_ui(
    server_h5ad_path,
    transpose,
    sess_bundle,
    view_bundle,
    run_bundle,
    selected_sources,
):
    try:
        preferred = _preferred_h5ad_path_for_session(
            server_h5ad_path,
            sess_bundle,
            prefer_dense=True,
        )
        adata = _load_adata_from_inputs(
            None,
            server_h5ad_path,
            transpose,
            backed=True,
            preferred_h5ad_path=preferred,
        )
        try:
            catalog = _build_benchmark_source_catalog(
                adata,
                view_bundle=view_bundle,
                run_bundle=run_bundle,
            )
            values = [str(x).strip() for x in (selected_sources or []) if str(x).strip()]
            valid = [x for x in values if x in catalog]
            choices = [
                (meta["display_label"], spec)
                for spec, meta in sorted(catalog.items(), key=lambda kv: kv[0])
            ]
            obs_cols = _column_choices(adata)
            n_base = sum(1 for meta in catalog.values() if meta["kind"] == "base_spec")
            n_external = len(catalog) - n_base
            lines = [
                f"Benchmark sources discovered: {len(catalog)}",
                f"- base dataset sources: {n_base}",
                f"- session-derived sources: {n_external}",
            ]
            if n_external:
                lines.append("- render a view to expose `obsm:X_pca` / `obsm:X_umap`; finish an embedding run to expose `obsm:X_scgpt`, `obsm:X_scvi`, etc.")
            return (
                "\n".join(lines),
                gr.update(choices=choices, value=valid),
                gr.update(choices=obs_cols, value=[]),
                gr.update(choices=["(none)"] + obs_cols, value="(none)"),
                catalog,
            )
        finally:
            _close_adata_handle(adata)
    except Exception as e:
        return (
            f"Benchmark source refresh error: {e}",
            gr.update(choices=[], value=[]),
            gr.update(choices=[], value=[]),
            gr.update(choices=["(none)"], value="(none)"),
            {},
        )


def _build_benchmark_ready_adata(
    base_adata: ad.AnnData,
    catalog: Dict[str, Dict[str, Any]],
    selected_sources: List[str],
) -> Tuple[ad.AnnData, List[str]]:
    specs = [str(s).strip() for s in (selected_sources or []) if str(s).strip() and str(s).strip() in catalog]
    if not specs:
        raise ValueError("Select at least one benchmark source.")
    need_base_x = any(
        catalog[s]["kind"] == "base_spec" and (
            catalog[s]["source_spec"] == "X" or str(catalog[s]["source_spec"]).startswith("layer:")
        )
        for s in specs
    )
    if need_base_x:
        out = ad.AnnData(
            X=_copy_matrix_payload(base_adata.X),
            obs=base_adata.obs.copy(),
            var=base_adata.var.copy(),
        )
        skip_default: List[str] = []
        if "X" not in specs:
            skip_default.append("X")
    else:
        out = ad.AnnData(
            X=np.zeros((base_adata.n_obs, 1), dtype=np.float32),
            obs=base_adata.obs.copy(),
            var=pd.DataFrame(index=["__placeholder__"]),
        )
        skip_default = ["X"]

    for spec in specs:
        meta = catalog[spec]
        if meta["kind"] == "base_spec":
            source_spec = str(meta["source_spec"])
            if source_spec == "X":
                continue
            if source_spec == "raw.X":
                if base_adata.raw is None:
                    raise ValueError("Selected raw.X but `.raw` is missing.")
                out.raw = ad.AnnData(
                    X=_copy_matrix_payload(base_adata.raw.X),
                    obs=base_adata.obs.copy(),
                    var=base_adata.raw.var.copy(),
                )
                continue
            if source_spec.startswith("layer:"):
                layer_key = source_spec.split(":", 1)[1]
                out.layers[layer_key] = _copy_matrix_payload(base_adata.layers[layer_key])
                continue
            if source_spec.startswith("obsm:"):
                obsm_key = source_spec.split(":", 1)[1]
                out.obsm[obsm_key] = np.asarray(base_adata.obsm[obsm_key], dtype=np.float32).copy()
                continue
            raise ValueError(f"Unsupported benchmark source spec: {source_spec}")

        if meta["kind"] == "external_obsm":
            path = str(meta.get("path") or "").strip()
            key = str(meta.get("key") or "").strip()
            final_key = str(meta["final_spec"]).split(":", 1)[1]
            adx = ad.read_h5ad(path, backed="r")
            try:
                if key not in adx.obsm:
                    raise KeyError(f"{key!r} not found in {path}")
                arr = np.asarray(adx.obsm[key], dtype=np.float32)
                if arr.shape[0] != out.n_obs:
                    raise ValueError(
                        f"{meta['final_spec']}: row count {arr.shape[0]} does not match loaded dataset rows {out.n_obs}"
                    )
                out.obsm[final_key] = arr.copy()
            finally:
                _close_adata_handle(adx)
            continue

        raise ValueError(f"Unsupported benchmark source kind: {meta['kind']}")

    out.uns["scfms_benchmark_selected_sources"] = specs
    out.uns["scfms_benchmark_skip_default_sources"] = skip_default
    return out, skip_default


def prepare_benchmark_dataset_ui(
    server_h5ad_path,
    transpose,
    sess_bundle,
    catalog_state,
    selected_sources,
):
    try:
        preferred = _preferred_h5ad_path_for_session(
            server_h5ad_path,
            sess_bundle,
            prefer_dense=True,
        )
        adata = _load_adata_from_inputs(
            None,
            server_h5ad_path,
            transpose,
            backed=True,
            preferred_h5ad_path=preferred,
        )
        src_key = _session_src_key(None, server_h5ad_path)
        session_dir, bundle = _ensure_session(sess_bundle, src_key, adata)
        bundle = _bundle_with_materialized_paths(bundle, source_h5ad_path=preferred)
        catalog = catalog_state or {}
        selected = [str(x).strip() for x in (selected_sources or []) if str(x).strip()]
        if not selected:
            return (
                bundle,
                "Select at least one benchmark source.",
                gr.update(value=""),
                gr.update(value=None),
                None,
            )
        out, skip_default = _build_benchmark_ready_adata(adata, catalog, selected)
        out_path = Path(session_dir) / "benchmark_ready.h5ad"
        out.write_h5ad(out_path, compression="gzip")
        rows = []
        for spec in selected:
            meta = catalog.get(spec) or {}
            rows.append(
                {
                    "source": spec,
                    "kind": meta.get("kind"),
                    "origin": meta.get("origin") or "loaded dataset",
                }
            )
        prepared = {
            "path": str(out_path.resolve()),
            "sources": selected,
            "skip_default_sources": skip_default,
            "session_dir": session_dir,
        }
        msg = (
            f"Prepared benchmark dataset.\n"
            f"Path: {out_path}\n"
            f"Sources included: {len(selected)}\n"
            f"Default skipped sources: {', '.join(skip_default) if skip_default else '(none)'}"
        )
        return (
            bundle,
            msg,
            gr.update(value=str(out_path.resolve())),
            pd.DataFrame(rows),
            prepared,
        )
    except Exception as e:
        return (
            sess_bundle,
            f"Benchmark dataset prep error: {e}",
            gr.update(value=""),
            gr.update(value=None),
            None,
        )


def _parse_hms_to_seconds(raw: str) -> int:
    s = str(raw or "").strip()
    if not s:
        return 0
    parts = [int(p) for p in s.split(":")]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 1:
        return parts[0]
    raise ValueError(f"Invalid time string: {raw}")


def _format_seconds_hms(sec: int) -> str:
    total = max(1, int(sec))
    hh, rem = divmod(total, 3600)
    mm, ss = divmod(rem, 60)
    return f"{hh:02d}:{mm:02d}:{ss:02d}"


def _benchmark_batch_groups(
    sources: List[str],
    targets: List[str],
    mode: str,
) -> List[Tuple[str, List[str], List[str]]]:
    srcs = [str(s) for s in sources if str(s).strip()]
    tgts = [str(t) for t in targets if str(t).strip()]
    if not srcs:
        raise ValueError("Select at least one benchmark source.")
    if not tgts:
        raise ValueError("Select at least one target obs column.")
    key = str(mode or "single_job_all_sources")
    if key == "job_per_source":
        return [(f"source:{src}", [src], tgts) for src in srcs]
    if key == "job_per_target":
        return [(f"target:{tgt}", srcs, [tgt]) for tgt in tgts]
    if key == "job_per_source_target":
        return [(f"source:{src}|target:{tgt}", [src], [tgt]) for src in srcs for tgt in tgts]
    return [("all_sources_all_targets", srcs, tgts)]


def recommend_benchmark_settings_ui(
    prepared_state,
    target_cols,
    classifier_kind,
    batch_mode,
):
    if not isinstance(prepared_state, dict):
        return 4, "16G", "04:00:00", "Prepare a benchmark dataset first."
    path = str(prepared_state.get("path") or "").strip()
    if not path:
        return 4, "16G", "04:00:00", "Prepare a benchmark dataset first."
    p = Path(path)
    if not p.is_file():
        return 4, "16G", "04:00:00", f"Prepared benchmark dataset missing: {p}"
    targets = [str(x).strip() for x in (target_cols or []) if str(x).strip()]
    try:
        adata = ad.read_h5ad(str(p), backed="r")
        try:
            rec = bench.estimate_benchmark_slurm_resources(
                adata,
                max_cells=0,
                skip_sources=list(prepared_state.get("skip_default_sources") or []),
                classifier_kind=str(classifier_kind or "logistic_regression"),
            )
        finally:
            _close_adata_handle(adata)
        groups = _benchmark_batch_groups(
            list(prepared_state.get("sources") or []),
            targets or ["(placeholder)"],
            str(batch_mode or "single_job_all_sources"),
        )
        models_per_job = max(len(srcs) * len(tgts) for _name, srcs, tgts in groups)
        base_sec = _parse_hms_to_seconds(str(rec["time"]))
        denom = 2.0 if str(classifier_kind or "").strip().lower() == "mlp" else 6.0
        scaled = int(np.ceil(base_sec * max(1.0, models_per_job / denom)))
        note = (
            f"Prepared benchmark dataset: `{p.name}`\n"
            f"- selected sources: {len(prepared_state.get('sources') or [])}\n"
            f"- selected targets: {len(targets)}\n"
            f"- batch groups: {len(groups)}\n"
            f"- models per job (worst case): {models_per_job}\n"
            f"- recommended floor: {rec['cpus']} CPU / {rec['mem_gib']}G / {_format_seconds_hms(scaled)}"
        )
        return int(rec["cpus"]), f"{int(rec['mem_gib'])}G", _format_seconds_hms(scaled), note
    except Exception as e:
        return 4, "16G", "04:00:00", f"Benchmark recommendation error: {e}"


def _benchmark_jobs_df(batch_state: Optional[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for job in list((batch_state or {}).get("jobs") or []):
        jid = str(job.get("job_id") or "").strip()
        if not jid:
            continue
        bgjobs.sync_slurm_meta(jid)
        meta = bgjobs.read_meta(jid) or {}
        rows.append(
            {
                "ui_job_id": jid,
                "slurm_job_id": meta.get("slurm_job_id"),
                "status": meta.get("status"),
                "sources": ", ".join(job.get("sources") or []),
                "targets": ", ".join(job.get("targets") or []),
                "wandb_url": meta.get("wandb_url") or "",
                "session_dir": meta.get("benchmark_session_dir") or "",
            }
        )
    return pd.DataFrame(rows)


def submit_benchmark_batch_ui(
    prepared_state,
    target_cols,
    split_mode,
    stratify_col,
    test_fraction,
    random_seed,
    classifier_kind,
    mlp_hidden,
    mlp_max_iter,
    lr_c,
    lr_max_iter,
    batch_mode,
    slurm_partition,
    slurm_gres,
    slurm_cpus,
    slurm_mem,
    slurm_time,
    slurm_extra_sbatch,
    slurm_bash_prologue,
    scfms_repo_root_tb,
    wandb_project,
    wandb_entity,
    wandb_run_prefix,
):
    if not isinstance(prepared_state, dict):
        return (
            "Prepare a benchmark dataset first.",
            gr.update(value=None),
            gr.update(choices=[], value=None),
            "",
            "<p>Prepare a benchmark dataset first.</p>",
            *_artifact_updates([]),
            None,
        )
    prepared_path = str(prepared_state.get("path") or "").strip()
    if not prepared_path or not Path(prepared_path).is_file():
        return (
            f"Prepared benchmark dataset missing: {prepared_path}",
            gr.update(value=None),
            gr.update(choices=[], value=None),
            "",
            "<p>Prepared benchmark dataset missing.</p>",
            *_artifact_updates([]),
            None,
        )
    selected_sources = list(prepared_state.get("sources") or [])
    targets = [str(x).strip() for x in (target_cols or []) if str(x).strip()]
    try:
        groups = _benchmark_batch_groups(selected_sources, targets, batch_mode)
        adata = ad.read_h5ad(prepared_path, backed="r")
        jobs: List[Dict[str, Any]] = []
        status_lines = [
            f"Prepared benchmark file: {prepared_path}",
            f"Submitting {len(groups)} benchmark job(s).",
        ]
        try:
            for idx, (group_name, group_sources, group_targets) in enumerate(groups, start=1):
                skip_sources = sorted(
                    set(str(x) for x in (prepared_state.get("skip_default_sources") or []))
                    | (set(selected_sources) - set(group_sources))
                )
                rec = bench.estimate_benchmark_slurm_resources(
                    adata,
                    max_cells=0,
                    skip_sources=skip_sources,
                    classifier_kind=str(classifier_kind or "logistic_regression"),
                )
                cpus = int(rec["cpus"])
                try:
                    user_cpus = int(slurm_cpus)
                except (TypeError, ValueError):
                    user_cpus = 0
                if user_cpus > 0:
                    cpus = max(cpus, user_cpus)
                mem = str(slurm_mem or "").strip()
                if not mem or mem.lower() == "auto":
                    mem = f"{int(rec['mem_gib'])}G"
                time_limit = str(slurm_time or "").strip()
                if not time_limit or time_limit.lower() == "auto":
                    base_sec = _parse_hms_to_seconds(str(rec["time"]))
                    models_per_job = max(1, len(group_sources) * len(group_targets))
                    denom = 2.0 if str(classifier_kind or "").strip().lower() == "mlp" else 6.0
                    time_limit = _format_seconds_hms(
                        int(np.ceil(base_sec * max(1.0, models_per_job / denom)))
                    )
                run_prefix = (wandb_run_prefix or "").strip()
                run_slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", group_name).strip("_")[:80] or f"group_{idx:03d}"
                run_name = f"{run_prefix}_{run_slug}" if run_prefix else run_slug
                bench_params: Dict[str, Any] = {
                    "target_cols": group_targets,
                    "split_mode": str(split_mode or "random"),
                    "stratify_col": str(stratify_col or "(none)"),
                    "test_fraction": float(test_fraction),
                    "random_seed": int(random_seed) if random_seed is not None else 0,
                    "classifier_kind": str(classifier_kind or "logistic_regression"),
                    "mlp_hidden": str(mlp_hidden or "128,64"),
                    "mlp_max_iter": float(mlp_max_iter or 200),
                    "lr_c": float(lr_c or 1.0),
                    "lr_max_iter": float(lr_max_iter or 2000),
                    "max_cells": 0.0,
                    "skip_sources": skip_sources,
                    "wandb_project": (wandb_project or "scfms-benchmark").strip() or "scfms-benchmark",
                    "wandb_entity": (wandb_entity or "").strip(),
                    "wandb_run_name": run_name,
                    "test_h5ad_path": "",
                }
                repo = Path((scfms_repo_root_tb or "").strip()).resolve() if (scfms_repo_root_tb or "").strip() else None
                jid = bgjobs.start_benchmark_slurm_job_from_h5ad(
                    prepared_path,
                    bench_params,
                    repo,
                    partition=effective_slurm_partition(slurm_partition),
                    cpus=cpus,
                    mem=mem,
                    time_limit=time_limit,
                    gres=str(slurm_gres or "gpu:1").strip() or "gpu:1",
                    bash_prologue=str(slurm_bash_prologue or ""),
                    extra_sbatch=str(slurm_extra_sbatch or ""),
                )
                jobs.append(
                    {
                        "job_id": jid,
                        "group": group_name,
                        "sources": group_sources,
                        "targets": group_targets,
                        "cpus": cpus,
                        "mem": mem,
                        "time": time_limit,
                    }
                )
                status_lines.append(
                    f"- {jid}: sources={len(group_sources)} targets={len(group_targets)} request=-c {cpus} --mem={mem} -t {time_limit}"
                )
        finally:
            _close_adata_handle(adata)
        batch_state = {
            "prepared_path": prepared_path,
            "session_dir": prepared_state.get("session_dir"),
            "jobs": jobs,
        }
        df = _benchmark_jobs_df(batch_state)
        first_job = jobs[0]["job_id"] if jobs else ""
        detail, html = bench.refresh_benchmark_slurm_panel(first_job) if first_job else ("", "<p>No job selected.</p>")
        artifacts = _job_artifacts_from_job(
            first_job,
            session_dir=str(prepared_state.get("session_dir") or ""),
            extra_paths=[("Prepared benchmark h5ad", prepared_path)],
        ) if first_job else []
        choices = [(f"{job['job_id']} [{job['group']}]", job["job_id"]) for job in jobs]
        return (
            "\n".join(status_lines),
            df,
            gr.update(choices=choices, value=first_job or None),
            detail,
            html,
            *_artifact_updates(artifacts),
            batch_state,
        )
    except Exception as e:
        return (
            f"Benchmark batch submission error: {e}",
            gr.update(value=None),
            gr.update(choices=[], value=None),
            str(e),
            f"<p>Error: {e}</p>",
            *_artifact_updates([]),
            None,
        )


def refresh_benchmark_batch_ui(batch_state, selected_job_id):
    if not isinstance(batch_state, dict):
        return (
            "No benchmark jobs submitted yet.",
            gr.update(value=None),
            gr.update(choices=[], value=None),
            "",
            "<p>No benchmark jobs submitted yet.</p>",
            *_artifact_updates([]),
            None,
        )
    jobs = list(batch_state.get("jobs") or [])
    if not jobs:
        return (
            "No benchmark jobs submitted yet.",
            gr.update(value=None),
            gr.update(choices=[], value=None),
            "",
            "<p>No benchmark jobs submitted yet.</p>",
            *_artifact_updates([]),
            batch_state,
        )
    df = _benchmark_jobs_df(batch_state)
    choices = [(f"{job['job_id']} [{job['group']}]", job["job_id"]) for job in jobs]
    chosen = str(selected_job_id or "").strip()
    if not chosen or chosen not in {job["job_id"] for job in jobs}:
        chosen = jobs[0]["job_id"]
    detail, html = bench.refresh_benchmark_slurm_panel(chosen)
    artifacts = _job_artifacts_from_job(
        chosen,
        session_dir=str(batch_state.get("session_dir") or ""),
        extra_paths=[("Prepared benchmark h5ad", str(batch_state.get("prepared_path") or ""))],
    )
    summary = (
        f"Benchmark jobs tracked: {len(jobs)}\n"
        f"Selected job: {chosen}"
    )
    return (
        summary,
        df,
        gr.update(choices=choices, value=chosen),
        detail,
        html,
        *_artifact_updates(artifacts),
        batch_state,
    )


def _compose_gres(gpu_count: Any, gpu_type: str) -> str:
    try:
        n = max(1, int(gpu_count))
    except (TypeError, ValueError):
        n = 1
    gt = str(gpu_type or "").strip()
    if gt:
        return f"gpu:{gt}:{n}"
    return f"gpu:{n}"


def _local_output_artifacts(session_dir: str, *, compat_pdf: str = "", de_csv: str = "") -> list[tuple[str, str]]:
    base = Path(session_dir).expanduser().resolve()
    items: list[tuple[str, str]] = []
    candidates = [
        ("Session folder", str(base)),
        ("Embedded subset h5ad", str(base / "embedding_subset.h5ad")),
        ("UMAP map csv", str(base / "umap_plot_map.csv")),
    ]
    if compat_pdf:
        candidates.append(("Compatibility PDF", compat_pdf))
    if de_csv:
        candidates.append(("DE CSV", de_csv))
    for label, path in candidates:
        if Path(path).exists():
            items.append((label, path))
    return items


def _slurm_artifacts_from_job(job_id: str) -> list[tuple[str, str]]:
    meta = bgjobs.read_meta(job_id) or {}
    items: list[tuple[str, str]] = []
    for label, key in (
        ("Stage dir", "id"),
        ("Batch script", "batch_script"),
        ("Params JSON", "embed_params"),
        ("Input h5ad", "input_h5ad"),
        ("Output h5ad", "result_path"),
    ):
        if key == "id":
            path = str((bgjobs.jobs_root() / job_id).resolve())
        else:
            path = str(meta.get(key) or "").strip()
        if path:
            items.append((label, path))
    return items


def _job_artifacts_from_job(
    job_id: str,
    *,
    session_dir: str = "",
    extra_paths: Optional[list[tuple[str, str]]] = None,
) -> list[tuple[str, str]]:
    meta = bgjobs.read_meta(job_id) or {}
    jd = (bgjobs.jobs_root() / job_id).resolve()
    items: list[tuple[str, str]] = []
    if jd.exists():
        items.append(("Job stage dir", str(jd)))
    for label, key in (
        ("Input h5ad", "input_h5ad"),
        ("Result file", "result_path"),
        ("Params JSON", "embed_params"),
        ("Batch script", "batch_script"),
    ):
        path = str(meta.get(key) or "").strip()
        if path:
            items.append((label, path))
    if session_dir:
        sdp = Path(session_dir).expanduser().resolve()
        if sdp.exists():
            items.append(("Session folder", str(sdp)))
            pdf = sdp / "scfms_compatibility_report.pdf"
            if pdf.exists():
                items.append(("Compatibility PDF", str(pdf)))
    for label, path in (extra_paths or []):
        if path and Path(path).exists():
            items.append((label, path))
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for label, path in items:
        if path not in seen:
            seen.add(path)
            out.append((label, path))
    return out


def _artifact_updates(items: list[tuple[str, str]]) -> tuple[Any, Any]:
    choices = [(label, path) for label, path in items]
    if choices:
        return (
            gr.update(choices=choices, value=choices[0][1]),
            gr.update(value=choices[0][1]),
        )
    return (
        gr.update(choices=[], value=None),
        gr.update(value=""),
    )


def _save_plotly_figure_if_session(
    fig: Any,
    session_dir: Optional[str],
    stem: str,
) -> Optional[Path]:
    if fig is None or session_dir is None or not str(session_dir).strip():
        return None
    try:
        sd = Path(str(session_dir).strip()).expanduser().resolve()
        plots = sd / "plots"
        plots.mkdir(parents=True, exist_ok=True)
        out = plots / f"{stem}.html"
        fig.write_html(str(out), include_plotlyjs="cdn")
        return out
    except Exception:
        return None


def _save_distribution_figures_if_session(
    figs: list[Any],
    session_dir: Optional[str],
    matrix_spec: str,
    dist_by_obs: Optional[str],
) -> list[Path]:
    stems = [
        f"dist_cell_sums_{pre._qc_dist_file_stem(matrix_spec, dist_by_obs)}",
        f"dist_detected_{pre._qc_dist_file_stem(matrix_spec, dist_by_obs)}",
        f"dist_col_means_{pre._qc_dist_file_stem(matrix_spec, dist_by_obs)}",
        f"dist_values_{pre._qc_dist_file_stem(matrix_spec, dist_by_obs)}",
    ]
    saved: list[Path] = []
    if session_dir is None or not str(session_dir).strip():
        return saved
    for stem, fig in zip(stems, figs):
        path = sess_res.save_matplotlib_figure(fig, Path(str(session_dir)), stem)
        if path is not None:
            saved.append(path)
    return saved


def recompute_distributions_ui(
    server_h5ad_path,
    transpose,
    sess_bundle_dict,
    matrix_spec,
    dist_by_obs,
):
    try:
        preferred = _preferred_h5ad_path_for_session(
            server_h5ad_path,
            sess_bundle_dict,
            prefer_dense=True,
        )
        adata = _load_adata_from_inputs(
            None,
            server_h5ad_path,
            transpose,
            backed=True,
            preferred_h5ad_path=preferred,
        )
        try:
            spec = str(matrix_spec or "X").strip() or "X"
            og = None if str(dist_by_obs or "(none)") in ("", "(none)", "none") else str(dist_by_obs)
            figs = pre._compute_distributions(adata, spec, obs_group_col=og)
        finally:
            _close_adata_handle(adata)
        session_dir = str((sess_bundle_dict or {}).get("dir") or "")
        saved = _save_distribution_figures_if_session(figs, session_dir, spec, og)
        strat = f", stratified by `{og}`" if og else ""
        status = (
            f"Computed distributions for `{spec}`{strat}."
            + (f"\nSaved {len(saved)} figure(s) under `{Path(session_dir).expanduser().resolve() / 'plots'}`." if saved else "")
        )
        return figs[0], figs[1], figs[2], figs[3], status
    except Exception as e:
        return gr.skip(), gr.skip(), gr.skip(), gr.skip(), f"Distribution viewer error: {e}"


def _recommend_run_settings(
    server_h5ad_path: str,
    model: str,
    n_latent: Any,
    matrix_spec: str = "X",
) -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    raw = str(server_h5ad_path or "").strip()
    if not raw:
        return (
            1,
            "",
            default_slurm_partition(),
            4,
            "auto",
            "auto",
            "Set a server `.h5ad` path, then click **Load Data** for recommendations.",
        )
    try:
        p = validate_server_read_path(raw)
        if p.suffix.lower() != ".h5ad":
            return (
                1,
                "",
                default_slurm_partition(),
                4,
                "auto",
                "auto",
                "Recommendations currently target `.h5ad` server paths.",
            )
        adata = ad.read_h5ad(str(p), backed="r")
        try:
            rec = pre.estimate_scfm_slurm_resources(
                adata,
                model=str(model),
                matrix_spec=str(matrix_spec or "X"),
                n_latent_scvi=int(n_latent) if n_latent is not None else 64,
            )
        finally:
            fobj = getattr(adata, "file", None)
            if fobj is not None:
                try:
                    fobj.close()
                except Exception:
                    pass
        gpu_count = 1
        gpu_type = ""
        partition = default_slurm_partition()
        cpus = int(rec["cpus"])
        note = (
            f"Recommended Slurm floor for **{Path(raw).name}** with **{model}**:\n"
            f"- Matrix: `{matrix_spec or 'X'}`\n"
            f"- GPUs: `{gpu_count}`\n"
            f"- CPUs: `{cpus}`\n"
            f"- Mem: `{rec['mem']}`\n"
            f"- Time: `{rec['time']}`\n"
            "\nThese are conservative minimums meant to avoid failure while staying fairly tight."
        )
        return gpu_count, gpu_type, partition, cpus, rec["mem"], rec["time"], note
    except Exception as e:
        return 1, "", default_slurm_partition(), 4, "auto", "auto", f"Recommendation error: {e}"


def recommend_settings_for_ui(server_h5ad_path: str, model: str, n_latent: Any, matrix_spec: str):
    return _recommend_run_settings(server_h5ad_path, model, n_latent, matrix_spec)


def _reset_run_outputs():
    return (
        "No run yet.",
        *_artifact_updates([]),
        None,
    )


def _toggle_exec_mode(mode: str):
    is_sbatch = str(mode or "current_node") == "sbatch"
    return gr.update(visible=not is_sbatch), gr.update(visible=is_sbatch)


def refresh_run_status(run_bundle: Optional[Dict[str, Any]]):
    if not isinstance(run_bundle, dict):
        return "No run yet.", *_artifact_updates([])
    mode = str(run_bundle.get("mode") or "")
    if mode == "sbatch":
        jid = str(run_bundle.get("job_id") or "").strip()
        if not jid:
            return "No Slurm job id recorded.", *_artifact_updates([])
        bgjobs.sync_slurm_meta(jid)
        txt = bgjobs.format_meta_report(jid)
        return txt, *_artifact_updates(_slurm_artifacts_from_job(jid))
    if mode == "background_local":
        jid = str(run_bundle.get("job_id") or "").strip()
        if not jid:
            return "No local background job id recorded.", *_artifact_updates([])
        txt = bgjobs.format_meta_report(jid)
        return txt, *_artifact_updates(
            _job_artifacts_from_job(
                jid,
                session_dir=str(run_bundle.get("session_dir") or ""),
            )
        )
    artifacts = [
        (str(label), str(path))
        for label, path in (run_bundle.get("artifacts") or [])
        if str(path).strip()
    ]
    return str(run_bundle.get("status") or "Local run complete."), *_artifact_updates(artifacts)


def _reset_view_outputs():
    return (
        gr.update(value=None),
        gr.update(value=None),
        "No view yet.",
        gr.update(choices=[], value=None),
        gr.update(value=""),
        None,
    )


def _reset_de_outputs():
    return (
        "No DE job yet.",
        *_artifact_updates([]),
        gr.update(value=None),
        gr.update(value=None),
        go.Figure(),
        go.Figure(),
        gr.update(value=None),
        gr.update(value=None),
        go.Figure(),
        go.Figure(),
        None,
    )


def _reset_benchmark_outputs():
    return (
        "No benchmark dataset prepared.",
        gr.update(value=""),
        gr.update(value=None),
        None,
        "No benchmark jobs submitted yet.",
        gr.update(value=None),
        gr.update(choices=[], value=None),
        "",
        "<p>No benchmark jobs submitted yet.</p>",
        *_artifact_updates([]),
        None,
        {},
    )


def _load_view_h5ad(view_bundle: Optional[Dict[str, Any]]):
    if not isinstance(view_bundle, dict):
        return None
    path = str(view_bundle.get("path") or "").strip()
    if not path:
        jid = str(view_bundle.get("job_id") or "").strip()
        if jid:
            rp = bgjobs.result_h5ad_path(jid)
            if rp is not None:
                path = str(rp)
    if not path:
        return None
    try:
        return ad.read_h5ad(path)
    except Exception:
        return None


def _render_view_outputs(
    adata: ad.AnnData,
    session_dir: str,
    color_by,
    gene_symbol,
    facet_by,
    hover_cols,
    point_size,
    alpha,
    dragmode,
):
    U = np.asarray(adata.obsm.get("X_umap"))
    if U.ndim != 2 or U.shape[1] < 2:
        raise ValueError("Current view does not contain obsm['X_umap'].")
    max_tbl = sess_res.embed_table_max_rows()
    n_cells = U.shape[0]
    max_plot = sess_res.umap_plot_max_cells()
    plot_idx = np.arange(n_cells)
    if n_cells > max_plot:
        rng = np.random.default_rng(0)
        plot_idx = np.sort(rng.choice(n_cells, size=max_plot, replace=False))
    color_vec_full, color_title = _get_color_vector(adata, gene_symbol, color_by)
    cvec = color_vec_full[plot_idx] if color_vec_full is not None else None
    if isinstance(hover_cols, (list, tuple)):
        hv_list = [str(x).strip() for x in hover_cols if str(x).strip()]
    else:
        hv_list = [x.strip() for x in str(hover_cols or "").split(",") if x.strip()]
    fig = _build_plotly_umap(
        U[plot_idx],
        adata,
        plot_idx,
        cvec,
        color_title,
        hv_list,
        None if (not facet_by or facet_by == "(none)") else str(facet_by),
        int(point_size) if point_size is not None else 4,
        float(alpha) if alpha is not None else 0.8,
        str(dragmode) if dragmode else "zoom",
    )
    view_spec = str(adata.uns.get("scfms_view_matrix_spec") or "X")
    safe_view = view_spec.replace("/", "_").replace(":", "_")
    _save_plotly_figure_if_session(fig, session_dir, f"view_umap_{safe_view}")
    X_show = adata.X[:max_tbl]
    if sp.issparse(X_show):
        df_arr = np.asarray(X_show.toarray())
    else:
        df_arr = np.asarray(X_show)
    df_show = pd.DataFrame(df_arr, index=adata.obs_names[: len(df_arr)])
    try:
        sdp = Path(session_dir).expanduser().resolve()
        pd.DataFrame({"cell": adata.obs_names[plot_idx]}).to_csv(
            sdp / "umap_plot_map.csv", index=False
        )
    except Exception:
        pass
    return df_show, fig


def refresh_view_status(
    view_bundle: Optional[Dict[str, Any]],
    session_bundle: Optional[Dict[str, Any]],
    color_by,
    gene_symbol,
    facet_by,
    hover_cols,
    point_size,
    alpha,
    dragmode,
):
    if not isinstance(view_bundle, dict):
        return (
            gr.skip(),
            gr.skip(),
            "No view yet.",
            *_artifact_updates([]),
            None,
        )
    jid = str(view_bundle.get("job_id") or "").strip()
    session_dir = str((session_bundle or {}).get("dir") or "")
    status_txt = bgjobs.format_meta_report(jid) if jid else str(view_bundle.get("status") or "No view yet.")
    path = str(view_bundle.get("path") or "").strip()
    rp = bgjobs.result_h5ad_path(jid) if jid else None
    if rp is not None:
        path = str(rp)
    updates = _artifact_updates(
        _job_artifacts_from_job(
            jid,
            session_dir=session_dir,
            extra_paths=[("Current view", path)] if path else None,
        )
    )
    if not path:
        return gr.skip(), gr.skip(), status_txt, *updates, view_bundle
    try:
        adx = ad.read_h5ad(path)
        df_show, fig = _render_view_outputs(
            adx,
            session_dir,
            color_by,
            gene_symbol,
            facet_by,
            hover_cols,
            point_size,
            alpha,
            dragmode,
        )
        new_bundle = dict(view_bundle)
        new_bundle["path"] = path
        new_bundle["status"] = status_txt
        if session_dir:
            new_bundle["dir"] = session_dir
        return df_show, fig, status_txt, *updates, new_bundle
    except Exception as e:
        return gr.skip(), gr.skip(), f"{status_txt}\n\nView load/render error: {e}", *updates, view_bundle


def _artifact_path_choice(path: str):
    return gr.update(value=str(path or ""))


def start_view_job(
    server_h5ad_path,
    transpose,
    matrix_spec,
    max_cells,
    n_pcs,
    n_neighbors,
    min_dist,
    color_by,
    gene_symbol,
    facet_by,
    hover_cols,
    point_size,
    alpha,
    dragmode,
    sess_bundle,
):
    try:
        spath = str(server_h5ad_path or "").strip()
        if not spath:
            return (
                sess_bundle,
                gr.update(value=None),
                gr.update(value=None),
                "Set a server path first.",
                *_artifact_updates([]),
                None,
            )
        preferred = _preferred_h5ad_path_for_session(
            server_h5ad_path,
            sess_bundle,
            prefer_dense=True,
        )
        adata = _load_adata_from_inputs(
            None,
            server_h5ad_path,
            transpose,
            backed=True,
            preferred_h5ad_path=preferred,
        )
        try:
            src_key = _session_src_key(None, server_h5ad_path)
            session_dir, bundle = _ensure_session(sess_bundle, src_key, adata)
            bundle = _bundle_with_materialized_paths(bundle, source_h5ad_path=preferred)
            spec = str(matrix_spec or "X").strip() or "X"
            safe = spec.replace(":", "_").replace("/", "_")
            result_path = Path(session_dir) / f"view_{safe}.h5ad"
            view_adata = pre.build_matrix_view_adata(
                adata,
                matrix_spec=spec,
                max_cells=int(max_cells or 0),
                n_pcs=int(n_pcs or 50),
                n_neighbors=int(n_neighbors or 15),
                min_dist=float(min_dist or 0.1),
            )
        finally:
            _close_adata_handle(adata)
        view_adata.write_h5ad(result_path, compression="gzip")
        df_show, fig = _render_view_outputs(
            view_adata,
            session_dir,
            color_by,
            gene_symbol,
            facet_by,
            hover_cols,
            point_size,
            alpha,
            dragmode,
        )
        txt = (
            f"Rendered view.\n"
            f"Source: {spec}\n"
            f"Cells in view: {view_adata.n_obs:,}\n"
            f"Features in view: {view_adata.n_vars:,}\n"
            f"Saved view file: {result_path}\n"
            f"Session: {session_dir}"
        )
        view_bundle = {
            "mode": "foreground",
            "matrix_spec": spec,
            "path": str(result_path),
            "status": txt,
            "dir": session_dir,
        }
        return (
            bundle,
            df_show,
            fig,
            txt,
            *_artifact_updates(
                [
                    ("Current view", str(result_path)),
                    ("Session folder", str(Path(session_dir).expanduser().resolve())),
                ]
            ),
            view_bundle,
        )
    except Exception as e:
        return (
            sess_bundle,
            gr.update(value=None),
            gr.update(value=None),
            f"View render error: {e}",
            *_artifact_updates([]),
            None,
        )


def _get_color_vector(
    adata: ad.AnnData, gene_symbol: str | None, color_by: str | None
) -> tuple[np.ndarray | None, str]:
    # Gene expression takes precedence if present
    if gene_symbol:
        gs = str(gene_symbol).strip()
        if gs:
            # Match case-insensitively
            var_low = pd.Index(adata.var_names.astype(str).str.lower())
            m = (var_low == gs.lower()).to_numpy()
            if m.any():
                idx = int(np.where(m)[0][0])
                X = adata.X
                if sp.issparse(X):
                    expr = np.asarray(X[:, idx].toarray()).ravel()
                else:
                    expr = np.asarray(X[:, idx]).ravel()
                return expr, f"Expression: {gs}"
    # Otherwise use metadata column
    if color_by and color_by in adata.obs.columns:
        s = adata.obs[color_by]
        # Return as numpy; caller decides discrete/continuous
        vals = s.values
        try:
            vals = s.astype(float).values  # type: ignore[assignment]
        except Exception:
            vals = s.astype(str).values
        return np.asarray(vals), f"Color by: {color_by}"
    return None, ""


def _obs_color_options(adata: ad.AnnData, max_unique: int = 50) -> list[str]:
    opts: list[str] = []
    for c in adata.obs.columns:
        s = adata.obs[c]
        try:
            if pd.api.types.is_numeric_dtype(s) and s.notna().sum() > 0:
                opts.append(c)
                continue
        except Exception:
            pass
        try:
            nunq = s.astype(str).nunique(dropna=True)
            if nunq <= max_unique:
                opts.append(c)
        except Exception:
            pass
    seen = set()
    out: list[str] = []
    for c in opts:
        if c not in seen:
            seen.add(c)
            out.append(c)
    if "cell_type" in out:
        out.remove("cell_type")
        out.insert(0, "cell_type")
    return out


def _build_plotly_umap(
    U: np.ndarray,
    adata: ad.AnnData,
    plot_idx: np.ndarray,
    color_vec: np.ndarray | None,
    color_title: str,
    hover_cols: list[str],
    facet_by: str | None,
    point_size: int,
    alpha: float,
    dragmode: str,
) -> go.Figure:
    df = pd.DataFrame(
        {"UMAP1": U[:, 0], "UMAP2": U[:, 1]}, index=adata.obs_names[plot_idx]
    )
    if color_vec is not None:
        df["__color__"] = pd.Series(color_vec, index=df.index)
    else:
        df["__color__"] = ""
    if facet_by and facet_by in adata.obs.columns:
        df["__facet__"] = adata.obs.loc[df.index, facet_by].astype(str).values
    else:
        df["__facet__"] = "all"
    for hc in hover_cols[:8]:
        if hc in adata.obs.columns:
            df[f"hover:{hc}"] = adata.obs.loc[df.index, hc].astype(str).values
    hover_data = {c: True for c in df.columns if c.startswith("hover:")}
    if facet_by and df["__facet__"].nunique() > 1:
        fig = px.scatter(
            df,
            x="UMAP1",
            y="UMAP2",
            color="__color__" if color_vec is not None else None,
            facet_col="__facet__",
            facet_col_wrap=4,
            opacity=max(0.0, min(1.0, alpha)),
            hover_data=hover_data,
            title="UMAP",
        )
    else:
        fig = px.scatter(
            df,
            x="UMAP1",
            y="UMAP2",
            color="__color__" if color_vec is not None else None,
            opacity=max(0.0, min(1.0, alpha)),
            hover_data=hover_data,
            title="UMAP",
        )
    fig.update_traces(marker=dict(size=int(point_size), line=dict(width=0)))
    fig.update_layout(
        dragmode=dragmode if dragmode in ("zoom", "select", "lasso") else "zoom"
    )
    if color_vec is not None and color_title:
        fig.update_layout(
            legend_title_text=color_title, coloraxis_colorbar_title=color_title
        )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def _compute_de(
    adata: ad.AnnData,
    group_col: str,
    fg_vals: list[str],
    bg_vals: list[str] | None,
    method: str,
    top_n: int,
) -> pd.DataFrame:
    if group_col not in adata.obs.columns:
        return pd.DataFrame()
    s = adata.obs[group_col].astype(str)
    fg_mask = s.isin(fg_vals)
    if bg_vals is None or len(bg_vals) == 0:
        bg_mask = ~fg_mask
    else:
        bg_mask = s.isin(bg_vals)
    if fg_mask.sum() < 5 or bg_mask.sum() < 5:
        return pd.DataFrame()
    X = adata.X
    if sp.issparse(X):
        X = X.tocsr()
    X_fg = X[fg_mask]
    X_bg = X[bg_mask]
    if sp.issparse(X_fg):
        mu_fg = np.asarray(X_fg.mean(axis=0)).ravel()
    else:
        mu_fg = X_fg.mean(axis=0)
    if sp.issparse(X_bg):
        mu_bg = np.asarray(X_bg.mean(axis=0)).ravel()
    else:
        mu_bg = X_bg.mean(axis=0)
    logfc = np.log1p(mu_fg + 1e-9) - np.log1p(mu_bg + 1e-9)
    pvals = np.ones(adata.n_vars, dtype=float)
    if method == "wilcoxon":
        for j in range(adata.n_vars):
            try:
                a = X_fg[:, j].toarray().ravel() if sp.issparse(X_fg) else X_fg[:, j]
                b = X_bg[:, j].toarray().ravel() if sp.issparse(X_bg) else X_bg[:, j]
                _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
                pvals[j] = p
            except Exception:
                pvals[j] = 1.0
    else:
        for j in range(adata.n_vars):
            try:
                a = X_fg[:, j].toarray().ravel() if sp.issparse(X_fg) else X_fg[:, j]
                b = X_bg[:, j].toarray().ravel() if sp.issparse(X_bg) else X_bg[:, j]
                _, p = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
                pvals[j] = p if np.isfinite(p) else 1.0
            except Exception:
                pvals[j] = 1.0
    # Benjamini-Hochberg adjustment over all p-values
    m = len(pvals)
    order_p = np.argsort(pvals)
    ranks = np.arange(1, m + 1, dtype=float)
    qvals = np.empty_like(pvals)
    qvals[order_p] = pvals[order_p] * m / ranks
    for i in range(m - 2, -1, -1):
        qvals[order_p[i]] = min(qvals[order_p[i]], qvals[order_p[i + 1]])
    qvals = np.clip(qvals, 0.0, 1.0)

    order = np.argsort(-np.abs(logfc))
    keep = order[: max(1, int(top_n))]
    df = pd.DataFrame(
        {
            "gene": adata.var_names.values[keep],
            "logFC": logfc[keep],
            "p_value": pvals[keep],
            "q_value": qvals[keep],
        }
    )
    return df


def run_embed(
    model,
    server_h5ad_path,
    transpose,
    scgpt_ckpt,
    n_latent,
    embed_matrix_spec,
    exec_mode,
    sbatch_partition,
    sbatch_gpu_count,
    sbatch_gpu_type,
    sbatch_cpus,
    sbatch_mem,
    sbatch_time,
    sbatch_bash_prologue,
    sbatch_repo_root,
    n_neighbors,
    min_dist,
    color_by,
    gene_symbol,
    filter_query,
    obsm_key_override,
    facet_by,
    hover_cols,
    point_size,
    alpha,
    dragmode,
    de_group_col,
    de_fore_vals,
    de_bg_mode,
    de_back_vals,
    de_method,
    de_topn,
    sess_bundle,
):
    try:
        spath = str(server_h5ad_path or "").strip()
        if not spath:
            return sess_bundle, gr.skip(), "Set a server path first.", *_artifact_updates([]), None
        src_key = _session_src_key(None, server_h5ad_path)
        mode = str(exec_mode or "current_node")
        embed_spec = str(embed_matrix_spec or "X").strip() or "X"
        if mode == "sbatch":
            p = validate_server_read_path(
                _preferred_h5ad_path_for_session(
                    server_h5ad_path,
                    sess_bundle,
                    prefer_dense=True,
                )
            )
            if p.suffix.lower() != ".h5ad":
                raise ValueError("`sbatch` mode currently requires a server `.h5ad` path.")
            ad_meta = ad.read_h5ad(str(p), backed="r")
            try:
                session_dir, bundle = _ensure_session(sess_bundle, src_key, ad_meta)
                bundle = _bundle_with_materialized_paths(bundle, source_h5ad_path=str(p))
            finally:
                fobj = getattr(ad_meta, "file", None)
                if fobj is not None:
                    try:
                        fobj.close()
                    except Exception:
                        pass
            rec_meta = ad.read_h5ad(str(p), backed="r")
            try:
                rec = pre.estimate_scfm_slurm_resources(
                    rec_meta,
                    model=str(model),
                    matrix_spec=embed_spec,
                    n_latent_scvi=int(n_latent) if n_latent is not None else 64,
                )
            finally:
                fobj = getattr(rec_meta, "file", None)
                if fobj is not None:
                    try:
                        fobj.close()
                    except Exception:
                        pass
            try:
                mem_floor = int(rec.get("mem_gib") or 0)
            except (TypeError, ValueError):
                mem_floor = 0
            mem_raw = str(sbatch_mem or "").strip()
            mem_raw_up = mem_raw.upper()
            user_mem_gib = 0
            if mem_raw and mem_raw.lower() != "auto":
                try:
                    if mem_raw_up.endswith("G"):
                        user_mem_gib = int(float(mem_raw_up[:-1]))
                    elif mem_raw_up.endswith("GB"):
                        user_mem_gib = int(float(mem_raw_up[:-2]))
                except (TypeError, ValueError):
                    user_mem_gib = 0
            if user_mem_gib > 0 and mem_floor > 0:
                mem = f"{max(mem_floor, user_mem_gib)}G"
            elif mem_raw and mem_raw.lower() != "auto":
                mem = mem_raw
            else:
                mem = str(rec["mem"])
            try:
                user_cpus = int(sbatch_cpus)
            except (TypeError, ValueError):
                user_cpus = 0
            cpus = max(int(rec["cpus"]), user_cpus if user_cpus > 0 else 0)
            time_raw = str(sbatch_time or "").strip()
            time_limit = str(rec["time"]) if not time_raw or time_raw.lower() == "auto" else time_raw
            try:
                gpu_floor = max(1, int(sbatch_gpu_count))
            except (TypeError, ValueError):
                gpu_floor = 1
            gres = _compose_gres(gpu_floor, sbatch_gpu_type)
            repo_root_val = (sbatch_repo_root or "").strip()
            sc_kwargs = dict(
                model=str(model),
                matrix_spec=embed_spec,
                obsm_key=None if obsm_key_override in (None, "", "(default)") else str(obsm_key_override),
                scgpt_ckpt=(scgpt_ckpt or "").strip() or None,
                n_latent_scvi=int(n_latent) if n_latent is not None else 64,
            )
            jid = bgjobs.start_scfm_slurm_job_from_h5ad(
                p,
                sc_kwargs,
                dataset_name=Path(spath).stem,
                repo_root=Path(repo_root_val).resolve() if repo_root_val else None,
                partition=effective_slurm_partition(sbatch_partition),
                gres=gres,
                cpus=cpus,
                mem=mem,
                time_limit=time_limit,
                bash_prologue=str(sbatch_bash_prologue or ""),
            )
            run_text = (
                f"Submitted Slurm embedding job.\n"
                f"Mode: sbatch\n"
                f"UI job id: {jid}\n"
                f"Resolved request: -p {effective_slurm_partition(sbatch_partition)} --gres={gres} -c {cpus} --mem={mem} -t {time_limit}\n"
                f"Embedding matrix: {embed_spec}\n"
                f"Input: {p}\n"
                f"Session: {session_dir}"
            )
            run_bundle = {
                "mode": "sbatch",
                "job_id": jid,
                "status": run_text,
                "session_dir": session_dir,
            }
            return (
                bundle,
                gr.update(value=session_dir),
                run_text,
                *_artifact_updates(_slurm_artifacts_from_job(jid)),
                run_bundle,
            )

        preferred = _preferred_h5ad_path_for_session(
            server_h5ad_path,
            sess_bundle,
            prefer_dense=True,
        )
        adata = _load_adata_from_inputs(
            None,
            server_h5ad_path,
            transpose,
            backed=True,
            preferred_h5ad_path=preferred,
        )
        session_dir, bundle = _ensure_session(sess_bundle, src_key, adata)
        bundle = _bundle_with_materialized_paths(bundle, source_h5ad_path=preferred)
        sc_kwargs = dict(
            model=str(model),
            matrix_spec=embed_spec,
            obsm_key=None if obsm_key_override in (None, "", "(default)") else str(obsm_key_override),
            scgpt_ckpt=(scgpt_ckpt or "").strip() or None,
            n_latent_scvi=int(n_latent) if n_latent is not None else 64,
            compat_report_dir=session_dir,
        )
        jid = bgjobs.start_scfm_job(adata, sc_kwargs)
        run_text = (
            f"Queued local embedding job.\n"
            f"Mode: current_node background\n"
            f"Job id: {jid}\n"
            f"Embedding matrix: {embed_spec}\n"
            f"Session: {session_dir}"
        )
        run_bundle = {
            "mode": "background_local",
            "job_id": jid,
            "status": run_text,
            "session_dir": session_dir,
        }
        return (
            bundle,
            gr.update(value=session_dir),
            run_text,
            *_artifact_updates(
                _job_artifacts_from_job(
                    jid,
                    session_dir=session_dir,
                )
            ),
            run_bundle,
        )
    except Exception as e:
        return (
            sess_bundle,
            gr.skip(),
            f"Error: {e}",
            *_artifact_updates([]),
            None,
        )


with gr.Blocks(title="scFMs: Organoid Embeddings") as demo:
    session_state = gr.State(None)
    download_state = gr.State(None)
    run_state = gr.State(None)
    view_state = gr.State(None)
    de_state = gr.State(None)
    benchmark_catalog_state = gr.State({})
    benchmark_prepared_state = gr.State(None)
    benchmark_batch_state = gr.State(None)
    gr.Markdown("# scFMs: Single-Cell Foundation Models for Organoids")
    with gr.Accordion("Where is this running? (server vs laptop)", open=False):
        gr.Markdown(runtime_info_markdown())
    gr.Markdown(
        "Compute embeddings with **Geneformer**, **scGPT**, or **scVI** (same pipeline as the preprocess app: "
        "compatibility checks + PDF). Prefer **Server path** on the cluster so large `.h5ad` files are not uploaded through the browser. "
        "For large cell counts, **UMAP** is fit on a **random subset** for responsiveness; full embeddings are still computed for all cells. "
        "New dataset sessions (plots, logs, `session_meta.json`) are created under "
        "**`<repo>/.assets/<dataset bucket>/<timestamp>_<label>/`** — the bucket is the managed-dataset folder: "
        "path under **`.data/`** (first component), or the parent of a **`data/`** or **`processed/`** segment in the path "
        "(e.g. `…/2025-05-HNOCA/data/file.h5ad` → `2025-05-HNOCA`). Set **`SCFMS_SESSION_DIR`** to use a single flat root instead."
    )
    session_path_disp = gr.Textbox(
        label="Active results folder",
        interactive=False,
    )
    gr.Markdown(
        "### Dataset Selection\n"
        "Pick a `.h5ad` from configured server dataset directories or paste a direct server path."
    )
    with gr.Row():
        dataset_pick = gr.Dropdown(
            choices=_list_data_options(),
            label="Server dataset",
            value=None,
            allow_custom_value=False,
            scale=4,
        )
        refresh_datasets = gr.Button("Refresh Datasets", scale=1)
    dataset_dir_info = gr.Markdown(_dataset_picker_info())
    server_path = gr.Textbox(
        label="Path on server (.h5ad / .csv / .tsv)",
        placeholder="/cluster/path/data.h5ad",
    )
    with gr.Accordion("Download Dataset", open=False):
        with gr.Row():
            download_method = gr.Dropdown(
                choices=[
                    ("Direct file URL", "direct_url"),
                    ("Zenodo record", "zenodo_record"),
                    ("GEO accession", "geo_accession"),
                    ("CELLxGENE asset URL", "cellxgene_url"),
                ],
                value="direct_url",
                label="Download method",
            )
            download_name = gr.Textbox(
                label="Dataset name",
                placeholder="e.g. HNOCA",
            )
        download_identifier = gr.Textbox(
            label="Download source",
            placeholder="URL, Zenodo record id, or GEO accession",
        )
        download_extra = gr.Textbox(
            label="Optional filename override / Zenodo file glob",
            placeholder="e.g. my_file.h5ad or *.h5ad",
        )
        download_method_help = gr.Markdown(_download_method_help("direct_url"))
        with gr.Row():
            download_btn = gr.Button("Start Download In Background", variant="secondary")
            refresh_download_btn = gr.Button("Refresh Download Status", variant="secondary")
        download_status = gr.Textbox(label="Download status", interactive=False, lines=8)
    with gr.Row():
        load_data_btn = gr.Button("Load Data", variant="secondary", scale=1)
        load_dense_btn = gr.Button("Load Dense", variant="secondary", scale=1)
        refresh_dense_btn = gr.Button("Refresh Dense Status", scale=1)
    data_status = gr.Textbox(label="Loaded dataset status", interactive=False)
    dense_status = gr.Textbox(label="Dense load status", interactive=False)
    with gr.Row():
        _m = pre.list_scfm_model_names()
        model = gr.Radio(
            _m,
            value=_m[0] if _m else "geneformer",
            label="Model",
        )
        transpose = gr.Checkbox(label="Transpose CSV/TSV", value=False)
        scgpt_ckpt = gr.Textbox(label="scGPT checkpoint dir (if using scGPT)")
    n_latent = gr.Number(
        label="scVI n_latent (ignored for other models)",
        value=64,
        precision=0,
        minimum=2,
    )
    embed_matrix_spec = gr.Dropdown(
        choices=["X"],
        value="X",
        label="Embedding input matrix",
    )
    exec_mode = gr.Dropdown(
        choices=["current_node", "sbatch"],
        value="sbatch",
        label="Execution target",
    )
    run_recommendation = gr.Markdown(
        "Load a server `.h5ad` to get a tight resource recommendation for the selected model."
    )
    with gr.Group(visible=False) as current_node_group:
        gr.Markdown(
            "Current-node mode runs inside this Gradio process on the node hosting the app. "
            "Use it for smaller datasets or when the current node already has the recommended GPU/RAM."
        )
        local_run_btn = gr.Button("Generate Embeddings", variant="primary")
    with gr.Group(visible=True) as sbatch_group:
        gr.Markdown(
            "Use `auto` for `RAM` and `Wall time` to keep the request at the recommended minimum safe floor."
        )
        with gr.Row():
            sbatch_partition = gr.Textbox(
                label="Slurm partition",
                value=default_slurm_partition(),
            )
            sbatch_gpu_count = gr.Number(
                label="# GPUs",
                value=1,
                precision=0,
                minimum=1,
            )
            sbatch_gpu_type = gr.Textbox(
                label="GPU type (optional)",
                placeholder="e.g. a100",
            )
            sbatch_cpus = gr.Number(
                label="# CPUs",
                value=4,
                precision=0,
                minimum=1,
            )
        with gr.Row():
            sbatch_mem = gr.Textbox(label="RAM", value="auto", placeholder="e.g. 24G")
            sbatch_time = gr.Textbox(
                label="Wall time",
                value="auto",
                placeholder="e.g. 08:00:00",
            )
            sbatch_repo_root = gr.Textbox(
                label="Repo root override (optional)",
                placeholder="leave blank to use this checkout",
            )
        sbatch_bash_prologue = gr.Textbox(
            label="SBATCH bash prologue (optional)",
            lines=3,
            placeholder="source ~/miniconda3/etc/profile.d/conda.sh\nconda activate scfms",
        )
        sbatch_run_btn = gr.Button("Generate Embeddings", variant="primary")
    run_status = gr.Textbox(label="Run status", interactive=False, lines=8)
    with gr.Row():
        refresh_run_btn = gr.Button("Refresh Run Status", variant="secondary")
        run_artifacts = gr.Dropdown(
            label="Run outputs / Slurm artifacts",
            choices=[],
            value=None,
            allow_custom_value=False,
            scale=3,
        )
    run_artifact_path = gr.Textbox(label="Selected output path", interactive=False)
    gr.Markdown("### Independent View / Plot Source")
    with gr.Row():
        view_matrix_spec = gr.Dropdown(
            choices=["X"],
            value="X",
            label="View source matrix",
        )
        view_max_cells = gr.Number(
            label="View max cells (0 = all)",
            value=0,
            precision=0,
            minimum=0,
        )
        view_n_pcs = gr.Number(
            label="View PCA/SVD comps",
            value=50,
            precision=0,
            minimum=0,
        )
    with gr.Row():
        render_view_btn = gr.Button("Render PCA/UMAP View", variant="secondary")
        refresh_view_btn = gr.Button("Refresh View", variant="secondary")
    view_status = gr.Textbox(label="View status", interactive=False, lines=8)
    with gr.Row():
        view_artifacts = gr.Dropdown(
            label="View outputs",
            choices=[],
            value=None,
            allow_custom_value=False,
            scale=3,
        )
        view_artifact_path = gr.Textbox(label="Selected view path", interactive=False)
    with gr.Accordion("Distribution Viewer", open=False):
        with gr.Row():
            dist_matrix_spec = gr.Dropdown(
                choices=["X"],
                value="X",
                label="Distribution source",
            )
            dist_by_obs = gr.Dropdown(
                choices=["(none)"],
                value="(none)",
                label="Stratify by obs column",
            )
            recompute_dist_btn = gr.Button("Compute Distributions", variant="secondary")
        dist_status = gr.Textbox(label="Distribution status", interactive=False, lines=4)
        with gr.Row():
            dist_plot1 = gr.Plot(label="Per-cell row sums")
            dist_plot2 = gr.Plot(label="Per-cell nonzero genes/features")
        with gr.Row():
            dist_plot3 = gr.Plot(label="Per-feature/gene column means")
            dist_plot4 = gr.Plot(label="Value distribution")
    with gr.Row():
        n_neighbors = gr.Slider(
            label="UMAP n_neighbors", value=15, minimum=5, maximum=50, step=1
        )
        min_dist = gr.Slider(
            label="UMAP min_dist", value=0.1, minimum=0.0, maximum=0.99, step=0.01
        )
    with gr.Row():
        point_size = gr.Slider(
            label="Point size", value=4, minimum=1, maximum=10, step=1
        )
        alpha = gr.Slider(
            label="Opacity", value=0.8, minimum=0.1, maximum=1.0, step=0.05
        )
        dragmode = gr.Radio(
            ["zoom", "select", "lasso"], value="zoom", label="Drag mode"
        )
    with gr.Row():
        color_by = gr.Dropdown(
            choices=["(none)"],
            value="(none)",
            label="Color by obs column",
        )
        gene_symbol = gr.Textbox(
            label="Gene symbol to overlay (optional)", placeholder="e.g., GAPDH"
        )
    facet_by = gr.Dropdown(
        choices=["(none)"],
        value="(none)",
        label="Facet by obs column (small multiples)",
    )
    hover_cols = gr.Dropdown(
        choices=[],
        value=[],
        multiselect=True,
        label="Hover columns",
    )
    with gr.Row():
        filter_query = gr.Textbox(
            label="Filter (pandas query on obs)",
            placeholder="e.g., tissue == 'brain' and cell_type in ['Astrocyte','Neuron']",
        )
        obsm_key_override = gr.Dropdown(
            choices=["(default)"],
            value="(default)",
            label="obsm key to plot (override)",
        )

    def _set_server_path(choice):
        return gr.update(value=choice or "")

    dataset_pick.change(_set_server_path, inputs=[dataset_pick], outputs=[server_path])
    download_method.change(
        _download_method_help,
        inputs=[download_method],
        outputs=[download_method_help],
    )
    refresh_datasets.click(
        lambda: (
            gr.update(choices=_list_data_options(), value=None),
            _dataset_picker_info(),
        ),
        None,
        [dataset_pick, dataset_dir_info],
    )
    download_btn.click(
        start_download_dataset_ui,
        inputs=[download_method, download_name, download_identifier, download_extra],
        outputs=[download_status, download_state],
    )
    refresh_download_btn.click(
        refresh_download_dataset_ui,
        inputs=[download_state],
        outputs=[download_status, dataset_pick, dataset_dir_info, server_path, download_state],
    )
    out = gr.Dataframe(label="Current View Matrix (truncated for display)", wrap=True)
    plot = gr.Plot(label="UMAP (interactive)")
    compat_pdf = gr.File(label="Compatibility report (PDF)", interactive=False)
    de_table = gr.Dataframe(label="Differential Expression (top genes)", wrap=True)
    de_csv = gr.File(label="Download DE CSV", interactive=False)
    volcano_plot = gr.Plot(label="Volcano Plot")
    dot_plot = gr.Plot(label="Dot Plot (FG vs BG)")

    with gr.Accordion("Selection & Inspector", open=False):
        sel_mode = gr.Radio(
            ["query", "umap_rect", "roi_plot"], value="query", label="Selection mode"
        )
        sel_query = gr.Textbox(
            label="Selection query (obs)",
            placeholder="e.g., dataset == 'CHOOSE' and cell_type == 'Neuron'",
        )
        with gr.Row():
            sel_xmin = gr.Number(label="UMAP x min", value=0.0)
            sel_xmax = gr.Number(label="UMAP x max", value=0.0)
            sel_ymin = gr.Number(label="UMAP y min", value=0.0)
            sel_ymax = gr.Number(label="UMAP y max", value=0.0)
        inspector_group_by = gr.Dropdown(
            choices=[],
            value=[],
            multiselect=True,
            label="Inspector: group by obs columns",
        )
        apply_sel = gr.Button("Apply Selection")
        sel_summary = gr.Dataframe(label="Selection Summary (counts)", wrap=True)
        sel_count = gr.Textbox(label="Selection Size", interactive=False)

    with gr.Accordion("Differential Expression", open=False):
        with gr.Row():
            run_de_btn = gr.Button("Run DE In Background")
            refresh_de_btn = gr.Button("Refresh DE", variant="secondary")
        de_status = gr.Textbox(label="DE status", interactive=False, lines=8)
        with gr.Row():
            de_artifacts = gr.Dropdown(
                label="DE outputs",
                choices=[],
                value=None,
                allow_custom_value=False,
                scale=3,
            )
            de_artifact_path = gr.Textbox(label="Selected DE path", interactive=False)
        de_group_col = gr.Dropdown(
            choices=["(none)"],
            value="(none)",
            label="Group column (obs)",
        )
        de_fore_vals = gr.Dropdown(
            choices=[],
            value=[],
            multiselect=True,
            label="Foreground values",
        )
        de_bg_mode = gr.Radio(
            ["all_others", "explicit"], value="all_others", label="Background"
        )
        de_back_vals = gr.Dropdown(
            choices=[],
            value=[],
            multiselect=True,
            label="Background values",
            interactive=False,
        )
        de_method = gr.Radio(["t-test", "wilcoxon"], value="t-test", label="Method")
        de_topn = gr.Slider(
            label="Top N genes", value=50, minimum=10, maximum=500, step=10
        )
        de_rank_by = gr.Radio(
            ["q_value", "p_value", "abs_logFC"], value="q_value", label="Rank by"
        )
        de_sig_axis = gr.Radio(
            ["q_value", "p_value"], value="q_value", label="Volcano significance axis"
        )
        de_sig_thresh = gr.Number(label="Significance threshold", value=0.05)
        de_fc_thresh = gr.Number(label="|logFC| threshold", value=0.25)
        de_label_topn = gr.Slider(
            label="Volcano: label top N", value=10, minimum=0, maximum=200, step=1
        )
        dot_facet_by = gr.Dropdown(
            choices=["(none)"],
            value="(none)",
            label="Dot plot facet by (obs)",
        )
        use_selection_fg = gr.Checkbox(
            label="Use current selection as foreground", value=False
        )
        de_bg_query = gr.Textbox(label="Background query (obs, optional)")
        gr.Markdown("---")
        gr.Markdown("Compare Two Saved Selections (A vs B)")
        save_sel_A = gr.Button("Save Current Selection as A")
        save_sel_B = gr.Button("Save Current Selection as B")
        run_de_ab_btn = gr.Button("Run DE (A vs B) In Background")
        de_table_ab = gr.Dataframe(label="DE A vs B", wrap=True)
        de_csv_ab = gr.File(label="Download DE A vs B CSV", interactive=False)
        volcano_plot_ab = gr.Plot(label="Volcano Plot (A vs B)")
        dot_plot_ab = gr.Plot(label="Dot Plot (A vs B)")

    with gr.Accordion("Export Bundle", open=False):
        mk_bundle = gr.Button("Prepare Download Bundle (UMAP + selection + DE)")
        bundle_file = gr.File(label="Bundle zip", interactive=False)

    with gr.Accordion("Benchmark (W&B / Slurm)", open=False):
        gr.Markdown(
            "Benchmark any mix of loaded-dataset sources plus session-derived sources. "
            "Use **Refresh benchmark sources** after rendering a view to expose `obsm:X_pca` / `obsm:X_umap`, "
            "or after an embedding run to expose `obsm:X_scgpt`, `obsm:X_scvi`, etc. "
            "Then prepare one shared benchmark-ready `.h5ad` and reuse it across many Slurm jobs."
        )
        with gr.Row():
            refresh_benchmark_sources_btn = gr.Button(
                "Refresh Benchmark Sources",
                variant="secondary",
            )
            prepare_benchmark_btn = gr.Button(
                "Prepare Benchmark Dataset",
                variant="secondary",
            )
        benchmark_source_info = gr.Textbox(
            label="Benchmark source discovery",
            interactive=False,
            lines=5,
        )
        benchmark_sources = gr.Dropdown(
            label="Benchmark sources",
            choices=[],
            value=[],
            multiselect=True,
        )
        with gr.Row():
            benchmark_target_cols = gr.Dropdown(
                label="Target obs columns",
                choices=[],
                value=[],
                multiselect=True,
            )
            benchmark_stratify_col = gr.Dropdown(
                label="Optional stratify obs column",
                choices=["(none)"],
                value="(none)",
            )
        with gr.Row():
            benchmark_classifier_kind = gr.Radio(
                ["logistic_regression", "mlp"],
                value="logistic_regression",
                label="Classifier",
            )
            benchmark_split_mode = gr.Dropdown(
                choices=[
                    ("Random split", "random"),
                    ("Stratify by obs column", "stratify_obs"),
                ],
                value="random",
                label="Train/validation split",
            )
            benchmark_batch_mode = gr.Dropdown(
                choices=[
                    ("Single job: all sources x all targets", "single_job_all_sources"),
                    ("Job per source", "job_per_source"),
                    ("Job per target", "job_per_target"),
                    ("Job per source x target", "job_per_source_target"),
                ],
                value="single_job_all_sources",
                label="Submission split",
            )
            benchmark_test_fraction = gr.Number(
                label="Validation fraction",
                value=0.2,
            )
            benchmark_random_seed = gr.Number(
                label="Random seed",
                value=0,
                precision=0,
            )
        with gr.Row():
            benchmark_mlp_hidden = gr.Textbox(
                label="MLP hidden layers",
                value="128,64",
            )
            benchmark_mlp_max_iter = gr.Number(
                label="MLP max_iter",
                value=200,
                precision=0,
            )
            benchmark_lr_c = gr.Number(label="LR C", value=1.0)
            benchmark_lr_max_iter = gr.Number(
                label="LR max_iter",
                value=2000,
                precision=0,
            )
        benchmark_prepare_status = gr.Textbox(
            label="Benchmark dataset status",
            interactive=False,
            lines=6,
        )
        benchmark_dataset_path = gr.Textbox(
            label="Prepared benchmark dataset path",
            interactive=False,
        )
        benchmark_prepared_sources = gr.Dataframe(
            label="Prepared sources",
            wrap=True,
        )
        benchmark_recommendation = gr.Markdown(
            "Prepare a benchmark dataset to get Slurm recommendations."
        )
        with gr.Row():
            benchmark_slurm_partition = gr.Textbox(
                label="Slurm partition",
                value=default_slurm_partition(),
            )
            benchmark_slurm_gres = gr.Textbox(
                label="Slurm gres",
                value="gpu:1",
            )
            benchmark_slurm_cpus = gr.Number(
                label="# CPUs",
                value=4,
                precision=0,
                minimum=1,
            )
        with gr.Row():
            benchmark_slurm_mem = gr.Textbox(
                label="RAM",
                value="auto",
            )
            benchmark_slurm_time = gr.Textbox(
                label="Wall time",
                value="auto",
            )
            benchmark_repo_root = gr.Textbox(
                label="Repo root override (optional)",
                placeholder="leave blank to use this checkout",
            )
        benchmark_slurm_extra = gr.Textbox(
            label="Extra #SBATCH lines (optional)",
            lines=2,
            placeholder="#SBATCH --constraint=a100",
        )
        benchmark_slurm_prologue = gr.Textbox(
            label="SBATCH bash prologue (optional)",
            lines=3,
            placeholder="source ~/miniconda3/etc/profile.d/conda.sh\nconda activate scfms",
        )
        with gr.Row():
            benchmark_wandb_project = gr.Textbox(
                label="wandb project",
                value="scfms-benchmark",
            )
            benchmark_wandb_entity = gr.Textbox(
                label="wandb entity (optional)",
            )
            benchmark_wandb_prefix = gr.Textbox(
                label="wandb run prefix",
                placeholder="e.g. hnoca_bench",
            )
        submit_benchmark_btn = gr.Button(
            "Submit Benchmark Batch",
            variant="primary",
        )
        benchmark_submit_status = gr.Textbox(
            label="Benchmark submit status",
            interactive=False,
            lines=10,
        )
        benchmark_jobs_table = gr.Dataframe(
            label="Benchmark jobs",
            wrap=True,
        )
        with gr.Row():
            refresh_benchmark_jobs_btn = gr.Button(
                "Refresh Benchmark Jobs",
                variant="secondary",
            )
            benchmark_job_pick = gr.Dropdown(
                label="Selected benchmark job",
                choices=[],
                value=None,
                allow_custom_value=False,
                scale=3,
            )
        benchmark_job_detail = gr.Textbox(
            label="Selected benchmark job detail",
            interactive=False,
            lines=12,
        )
        benchmark_wandb_html = gr.HTML(
            label="Weights & Biases",
        )
        with gr.Row():
            benchmark_artifacts = gr.Dropdown(
                label="Benchmark artifacts",
                choices=[],
                value=None,
                allow_custom_value=False,
                scale=3,
            )
            benchmark_artifact_path = gr.Textbox(
                label="Selected benchmark artifact path",
                interactive=False,
            )

    load_data_btn.click(
        load_data_options,
        inputs=[server_path, transpose, session_state],
        outputs=[
            data_status,
            session_state,
            session_path_disp,
            color_by,
            facet_by,
            hover_cols,
            obsm_key_override,
            inspector_group_by,
            de_group_col,
            dot_facet_by,
            view_matrix_spec,
            embed_matrix_spec,
            dense_status,
            de_fore_vals,
            de_back_vals,
            dist_matrix_spec,
            dist_by_obs,
        ],
    )
    load_data_btn.click(
        recommend_settings_for_ui,
        inputs=[server_path, model, n_latent, embed_matrix_spec],
        outputs=[
            sbatch_gpu_count,
            sbatch_gpu_type,
            sbatch_partition,
            sbatch_cpus,
            sbatch_mem,
            sbatch_time,
            run_recommendation,
        ],
    )
    load_data_btn.click(
        _reset_run_outputs,
        inputs=None,
        outputs=[
            run_status,
            run_artifacts,
            run_artifact_path,
            run_state,
        ],
    )
    load_data_btn.click(
        _reset_view_outputs,
        inputs=None,
        outputs=[
            out,
            plot,
            view_status,
            view_artifacts,
            view_artifact_path,
            view_state,
        ],
    )
    load_data_btn.click(
        lambda: (gr.update(value=None), gr.update(value=None), gr.update(value=None), gr.update(value=None), "No distributions yet."),
        inputs=None,
        outputs=[dist_plot1, dist_plot2, dist_plot3, dist_plot4, dist_status],
    )
    load_data_btn.click(
        _reset_de_outputs,
        inputs=None,
        outputs=[
            de_status,
            de_artifacts,
            de_artifact_path,
            de_table,
            de_csv,
            volcano_plot,
            dot_plot,
            de_table_ab,
            de_csv_ab,
            volcano_plot_ab,
            dot_plot_ab,
            de_state,
        ],
    )
    load_data_btn.click(
        _reset_benchmark_outputs,
        inputs=None,
        outputs=[
            benchmark_prepare_status,
            benchmark_dataset_path,
            benchmark_prepared_sources,
            benchmark_prepared_state,
            benchmark_submit_status,
            benchmark_jobs_table,
            benchmark_job_pick,
            benchmark_job_detail,
            benchmark_wandb_html,
            benchmark_artifacts,
            benchmark_artifact_path,
            benchmark_batch_state,
            benchmark_catalog_state,
        ],
    )
    load_data_btn.click(
        lambda: (
            gr.update(value="Benchmark sources will appear after you click `Refresh Benchmark Sources`."),
            gr.update(choices=[], value=[]),
            gr.update(choices=[], value=[]),
            gr.update(choices=["(none)"], value="(none)"),
            "Prepare a benchmark dataset to get Slurm recommendations.",
        ),
        inputs=None,
        outputs=[
            benchmark_source_info,
            benchmark_sources,
            benchmark_target_cols,
            benchmark_stratify_col,
            benchmark_recommendation,
        ],
    )
    load_dense_btn.click(
        start_dense_load_ui,
        inputs=[server_path, session_state],
        outputs=[dense_status, session_state],
    )
    refresh_dense_btn.click(
        refresh_dense_load_ui,
        inputs=[session_state],
        outputs=[dense_status, session_state],
    )
    model.change(
        recommend_settings_for_ui,
        inputs=[server_path, model, n_latent, embed_matrix_spec],
        outputs=[
            sbatch_gpu_count,
            sbatch_gpu_type,
            sbatch_partition,
            sbatch_cpus,
            sbatch_mem,
            sbatch_time,
            run_recommendation,
        ],
    )
    de_group_col.change(
        update_de_value_choices_ui,
        inputs=[
            server_path,
            transpose,
            session_state,
            view_state,
            de_group_col,
            de_fore_vals,
            de_back_vals,
        ],
        outputs=[de_fore_vals, de_back_vals],
    )
    de_bg_mode.change(
        lambda mode: gr.update(interactive=str(mode or "all_others") == "explicit"),
        inputs=[de_bg_mode],
        outputs=[de_back_vals],
    )
    n_latent.change(
        recommend_settings_for_ui,
        inputs=[server_path, model, n_latent, embed_matrix_spec],
        outputs=[
            sbatch_gpu_count,
            sbatch_gpu_type,
            sbatch_partition,
            sbatch_cpus,
            sbatch_mem,
            sbatch_time,
            run_recommendation,
        ],
    )
    embed_matrix_spec.change(
        recommend_settings_for_ui,
        inputs=[server_path, model, n_latent, embed_matrix_spec],
        outputs=[
            sbatch_gpu_count,
            sbatch_gpu_type,
            sbatch_partition,
            sbatch_cpus,
            sbatch_mem,
            sbatch_time,
            run_recommendation,
        ],
    )
    exec_mode.change(
        _toggle_exec_mode,
        inputs=[exec_mode],
        outputs=[current_node_group, sbatch_group],
    )

    local_run_btn.click(
        run_embed,
        inputs=[
            model,
            server_path,
            transpose,
            scgpt_ckpt,
            n_latent,
            embed_matrix_spec,
            exec_mode,
            sbatch_partition,
            sbatch_gpu_count,
            sbatch_gpu_type,
            sbatch_cpus,
            sbatch_mem,
            sbatch_time,
            sbatch_bash_prologue,
            sbatch_repo_root,
            n_neighbors,
            min_dist,
            color_by,
            gene_symbol,
            filter_query,
            obsm_key_override,
            facet_by,
            hover_cols,
            point_size,
            alpha,
            dragmode,
            de_group_col,
            de_fore_vals,
            de_bg_mode,
            de_back_vals,
            de_method,
            de_topn,
            session_state,
        ],
        outputs=[
            session_state,
            session_path_disp,
            run_status,
            run_artifacts,
            run_artifact_path,
            run_state,
        ],
    )
    sbatch_run_btn.click(
        run_embed,
        inputs=[
            model,
            server_path,
            transpose,
            scgpt_ckpt,
            n_latent,
            embed_matrix_spec,
            exec_mode,
            sbatch_partition,
            sbatch_gpu_count,
            sbatch_gpu_type,
            sbatch_cpus,
            sbatch_mem,
            sbatch_time,
            sbatch_bash_prologue,
            sbatch_repo_root,
            n_neighbors,
            min_dist,
            color_by,
            gene_symbol,
            filter_query,
            obsm_key_override,
            facet_by,
            hover_cols,
            point_size,
            alpha,
            dragmode,
            de_group_col,
            de_fore_vals,
            de_bg_mode,
            de_back_vals,
            de_method,
            de_topn,
            session_state,
        ],
        outputs=[
            session_state,
            session_path_disp,
            run_status,
            run_artifacts,
            run_artifact_path,
            run_state,
        ],
    )
    refresh_run_btn.click(
        refresh_run_status,
        inputs=[run_state],
        outputs=[run_status, run_artifacts, run_artifact_path],
    )
    run_artifacts.change(
        _artifact_path_choice,
        inputs=[run_artifacts],
        outputs=[run_artifact_path],
    )
    render_view_btn.click(
        start_view_job,
        inputs=[
            server_path,
            transpose,
            view_matrix_spec,
            view_max_cells,
            view_n_pcs,
            n_neighbors,
            min_dist,
            color_by,
            gene_symbol,
            facet_by,
            hover_cols,
            point_size,
            alpha,
            dragmode,
            session_state,
        ],
        outputs=[
            session_state,
            out,
            plot,
            view_status,
            view_artifacts,
            view_artifact_path,
            view_state,
        ],
    )
    recompute_dist_btn.click(
        recompute_distributions_ui,
        inputs=[server_path, transpose, session_state, dist_matrix_spec, dist_by_obs],
        outputs=[dist_plot1, dist_plot2, dist_plot3, dist_plot4, dist_status],
    )
    refresh_view_btn.click(
        refresh_view_status,
        inputs=[
            view_state,
            session_state,
            color_by,
            gene_symbol,
            facet_by,
            hover_cols,
            point_size,
            alpha,
            dragmode,
        ],
        outputs=[
            out,
            plot,
            view_status,
            view_artifacts,
            view_artifact_path,
            view_state,
        ],
    )
    view_artifacts.change(
        _artifact_path_choice,
        inputs=[view_artifacts],
        outputs=[view_artifact_path],
    )
    refresh_benchmark_sources_btn.click(
        refresh_benchmark_sources_ui,
        inputs=[
            server_path,
            transpose,
            session_state,
            view_state,
            run_state,
            benchmark_sources,
        ],
        outputs=[
            benchmark_source_info,
            benchmark_sources,
            benchmark_target_cols,
            benchmark_stratify_col,
            benchmark_catalog_state,
        ],
    )
    prepare_benchmark_btn.click(
        prepare_benchmark_dataset_ui,
        inputs=[
            server_path,
            transpose,
            session_state,
            benchmark_catalog_state,
            benchmark_sources,
        ],
        outputs=[
            session_state,
            benchmark_prepare_status,
            benchmark_dataset_path,
            benchmark_prepared_sources,
            benchmark_prepared_state,
        ],
    )
    benchmark_target_cols.change(
        recommend_benchmark_settings_ui,
        inputs=[
            benchmark_prepared_state,
            benchmark_target_cols,
            benchmark_classifier_kind,
            benchmark_batch_mode,
        ],
        outputs=[
            benchmark_slurm_cpus,
            benchmark_slurm_mem,
            benchmark_slurm_time,
            benchmark_recommendation,
        ],
    )
    benchmark_classifier_kind.change(
        recommend_benchmark_settings_ui,
        inputs=[
            benchmark_prepared_state,
            benchmark_target_cols,
            benchmark_classifier_kind,
            benchmark_batch_mode,
        ],
        outputs=[
            benchmark_slurm_cpus,
            benchmark_slurm_mem,
            benchmark_slurm_time,
            benchmark_recommendation,
        ],
    )
    benchmark_batch_mode.change(
        recommend_benchmark_settings_ui,
        inputs=[
            benchmark_prepared_state,
            benchmark_target_cols,
            benchmark_classifier_kind,
            benchmark_batch_mode,
        ],
        outputs=[
            benchmark_slurm_cpus,
            benchmark_slurm_mem,
            benchmark_slurm_time,
            benchmark_recommendation,
        ],
    )
    submit_benchmark_btn.click(
        submit_benchmark_batch_ui,
        inputs=[
            benchmark_prepared_state,
            benchmark_target_cols,
            benchmark_split_mode,
            benchmark_stratify_col,
            benchmark_test_fraction,
            benchmark_random_seed,
            benchmark_classifier_kind,
            benchmark_mlp_hidden,
            benchmark_mlp_max_iter,
            benchmark_lr_c,
            benchmark_lr_max_iter,
            benchmark_batch_mode,
            benchmark_slurm_partition,
            benchmark_slurm_gres,
            benchmark_slurm_cpus,
            benchmark_slurm_mem,
            benchmark_slurm_time,
            benchmark_slurm_extra,
            benchmark_slurm_prologue,
            benchmark_repo_root,
            benchmark_wandb_project,
            benchmark_wandb_entity,
            benchmark_wandb_prefix,
        ],
        outputs=[
            benchmark_submit_status,
            benchmark_jobs_table,
            benchmark_job_pick,
            benchmark_job_detail,
            benchmark_wandb_html,
            benchmark_artifacts,
            benchmark_artifact_path,
            benchmark_batch_state,
        ],
    )
    refresh_benchmark_jobs_btn.click(
        refresh_benchmark_batch_ui,
        inputs=[benchmark_batch_state, benchmark_job_pick],
        outputs=[
            benchmark_submit_status,
            benchmark_jobs_table,
            benchmark_job_pick,
            benchmark_job_detail,
            benchmark_wandb_html,
            benchmark_artifacts,
            benchmark_artifact_path,
            benchmark_batch_state,
        ],
    )
    benchmark_job_pick.change(
        refresh_benchmark_batch_ui,
        inputs=[benchmark_batch_state, benchmark_job_pick],
        outputs=[
            benchmark_submit_status,
            benchmark_jobs_table,
            benchmark_job_pick,
            benchmark_job_detail,
            benchmark_wandb_html,
            benchmark_artifacts,
            benchmark_artifact_path,
            benchmark_batch_state,
        ],
    )
    benchmark_artifacts.change(
        _artifact_path_choice,
        inputs=[benchmark_artifacts],
        outputs=[benchmark_artifact_path],
    )

    def _load_current_view_h5ad(view_bundle_dict):
        return _load_view_h5ad(view_bundle_dict)

    def apply_selection_fn(
        sess_bundle_dict, mode, query, xmin, xmax, ymin, ymax, group_by_csv
    ):
        adx = _load_current_view_h5ad(sess_bundle_dict)
        if adx is None:
            return gr.update(value=None), "0"
        sel_mask = np.zeros(adx.n_obs, dtype=bool)
        if mode == "query" and query:
            try:
                idx = adx.obs.query(str(query)).index
                sel_mask = adx.obs.index.isin(idx).to_numpy()
            except Exception:
                pass
        elif mode == "umap_rect":
            U = adx.obsm.get("X_umap")
            if U is not None:
                try:
                    xmin = float(xmin)
                    xmax = float(xmax)
                    ymin = float(ymin)
                    ymax = float(ymax)
                    m = (
                        (U[:, 0] >= min(xmin, xmax))
                        & (U[:, 0] <= max(xmin, xmax))
                        & (U[:, 1] >= min(ymin, ymax))
                        & (U[:, 1] <= max(ymin, ymax))
                    )
                    sel_mask = m
                except Exception:
                    pass
        # Persist selection
        try:
            d = (sess_bundle_dict or {}).get("dir")
            if d:
                sel_path = Path(d) / "selection.txt"
                with open(sel_path, "w") as f:
                    for name in adx.obs.index[sel_mask].tolist():
                        f.write(str(name) + "\n")
        except Exception:
            pass
        gb = _normalize_groupby_input(group_by_csv)
        if gb:
            df_sel = adx.obs.loc[sel_mask, gb].copy()
            for c in gb:
                df_sel[c] = df_sel[c].astype(str)
            summary = df_sel.value_counts().reset_index(name="count")
        else:
            summary = pd.DataFrame({"selected_count": [int(sel_mask.sum())]})
        return summary, str(int(sel_mask.sum()))

    apply_sel.click(
        apply_selection_fn,
        inputs=[
            view_state,
            sel_mode,
            sel_query,
            sel_xmin,
            sel_xmax,
            sel_ymin,
            sel_ymax,
            inspector_group_by,
        ],
        outputs=[sel_summary, sel_count],
    )

    def on_plot_select(evt, sess_bundle_dict, group_by_csv):
        adx = _load_current_view_h5ad(sess_bundle_dict)
        if adx is None:
            return gr.update(value=None), "0"
        d = (sess_bundle_dict or {}).get("dir")
        try:
            mp = Path(d) / "umap_plot_map.csv"
            if not mp.exists():
                return gr.update(value=None), "0"
            dfm = pd.read_csv(mp)
            idxs = []
            try:
                idxs = list(getattr(evt, "index", []) or [])
            except Exception:
                try:
                    idxs = list((evt or {}).get("index", []) or [])
                except Exception:
                    idxs = []
            idxs = [int(i) for i in idxs if i is not None]
            cells = dfm.iloc[idxs]["cell"].astype(str).tolist()
            sel_mask = adx.obs.index.isin(cells).to_numpy()
            sel_path = Path(d) / "selection.txt"
            with open(sel_path, "w") as f:
                for name in adx.obs.index[sel_mask].tolist():
                    f.write(str(name) + "\n")
        except Exception:
            return gr.update(value=None), "0"
        gb = _normalize_groupby_input(group_by_csv)
        if gb:
            df_sel = adx.obs.loc[sel_mask, gb].copy()
            for c in gb:
                df_sel[c] = df_sel[c].astype(str)
            summary = df_sel.value_counts().reset_index(name="count")
        else:
            summary = pd.DataFrame({"selected_count": [int(sel_mask.sum())]})
        return summary, str(int(sel_mask.sum()))

    try:
        plot.select(
            on_plot_select,
            inputs=[view_state, inspector_group_by],
            outputs=[sel_summary, sel_count],
        )
    except Exception:
        pass

    def _rank_df(df: pd.DataFrame, how: str) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        how = (how or "q_value").lower()
        if how == "q_value" and "q_value" in df.columns:
            return df.sort_values("q_value", ascending=True)
        if how == "p_value" and "p_value" in df.columns:
            return df.sort_values("p_value", ascending=True)
        if "logFC" in df.columns:
            return df.reindex(df.index[np.argsort(-np.abs(df["logFC"].to_numpy()))])
        return df

    def _make_volcano(
        df: pd.DataFrame,
        axis: str,
        sig_thresh: float,
        fc_thresh: float,
        label_topn: int,
    ) -> go.Figure:
        if df is None or df.empty:
            return px.scatter(
                pd.DataFrame({"logFC": [], "neglog10": []}),
                x="logFC",
                y="neglog10",
                title="Volcano",
            )
        axis = axis if axis in ("q_value", "p_value") else "q_value"
        sig = df[axis].to_numpy()
        y = -np.log10(np.clip(sig, 1e-300, None))
        passed = (sig <= sig_thresh) & (np.abs(df["logFC"].to_numpy()) >= fc_thresh)
        tmp = df.copy()
        tmp["neglog10"] = y
        tmp["signif"] = np.where(passed, "sig", "nsig")
        fig = px.scatter(
            tmp,
            x="logFC",
            y="neglog10",
            color="signif",
            hover_data=["gene", axis],
            color_discrete_map={"sig": "crimson", "nsig": "lightgray"},
        )
        fig.update_layout(xaxis_title="logFC", yaxis_title=f"-log10({axis})")
        if label_topn and label_topn > 0:
            idx = np.where(passed)[0]
            if idx.size > 0:
                order = np.argsort(sig[idx])[: min(label_topn, idx.size)]
                pick = idx[order]
                for i in pick:
                    fig.add_annotation(
                        x=float(df.iloc[i]["logFC"]),
                        y=float(y[i]),
                        text=str(df.iloc[i]["gene"]),
                        showarrow=False,
                        font=dict(size=10),
                    )
        return fig

    def _make_dotplot(
        adx: ad.AnnData,
        genes: list[str],
        fg_mask: np.ndarray,
        bg_mask: np.ndarray,
        facet_by: str | None,
    ) -> go.Figure:
        if adx is None or adx.n_vars == 0 or len(genes) == 0:
            return px.scatter(
                pd.DataFrame(
                    {"group": [], "gene": [], "mean": [], "frac": [], "facet": []}
                ),
                x="group",
                y="gene",
                size="frac",
                color="mean",
            )
        var_index = {g: i for i, g in enumerate(adx.var_names.astype(str))}
        use = [g for g in genes if g in var_index]
        if not use:
            return px.scatter(
                pd.DataFrame(
                    {"group": [], "gene": [], "mean": [], "frac": [], "facet": []}
                ),
                x="group",
                y="gene",
                size="frac",
                color="mean",
            )
        X = adx.X.tocsr() if sp.issparse(adx.X) else np.asarray(adx.X)
        rows = []
        facets = None
        if facet_by and facet_by in adx.obs.columns:
            facets = adx.obs[facet_by].astype(str).values
        for grp, m in (("FG", fg_mask), ("BG", bg_mask)):
            sub = X[m]
            fac = facets[m] if facets is not None else None
            for g in use:
                j = var_index[g]
                col = sub[:, j].toarray().ravel() if sp.issparse(sub) else sub[:, j]
                if fac is None:
                    mean = float(np.mean(col)) if col.size else 0.0
                    frac = float(np.mean(col > 0)) if col.size else 0.0
                    rows.append(
                        {
                            "group": grp,
                            "gene": g,
                            "mean": mean,
                            "frac": frac,
                            "facet": "all",
                        }
                    )
                else:
                    dfc = pd.DataFrame({"val": col, "facet": fac})
                    for fac_val, subdf in dfc.groupby("facet"):
                        v = subdf["val"].to_numpy()
                        rows.append(
                            {
                                "group": grp,
                                "gene": g,
                                "mean": float(np.mean(v)),
                                "frac": float(np.mean(v > 0)),
                                "facet": str(fac_val),
                            }
                        )
        dfm = pd.DataFrame(rows)
        if facets is not None:
            fig = px.scatter(
                dfm,
                x="group",
                y="gene",
                size="frac",
                color="mean",
                color_continuous_scale="Viridis",
                facet_col="facet",
                facet_col_wrap=4,
            )
        else:
            fig = px.scatter(
                dfm,
                x="group",
                y="gene",
                size="frac",
                color="mean",
                color_continuous_scale="Viridis",
            )
        fig.update_layout(xaxis_title="Group", yaxis_title="Gene")
        return fig

    def _read_name_list(path_like: Path) -> list[str]:
        if not path_like.exists():
            return []
        return [line.strip() for line in path_like.read_text().splitlines() if line.strip()]

    def _de_masks_from_params(adata: ad.AnnData, params: Dict[str, Any]):
        mode = str(params.get("mode") or "manual")
        if mode == "selection":
            sel_raw = str(params.get("selection_path") or "").strip()
            cells = _read_name_list(Path(sel_raw)) if sel_raw else []
            fg_mask = adata.obs.index.isin(cells).to_numpy()
            bg_query = str(params.get("bg_query") or "").strip()
            if bg_query:
                try:
                    idx = adata.obs.query(bg_query).index
                    bg_mask = adata.obs.index.isin(idx).to_numpy()
                except Exception:
                    bg_mask = ~fg_mask
            else:
                bg_mask = ~fg_mask
            return fg_mask, bg_mask
        if mode == "ab":
            a_raw = str(params.get("selection_a_path") or "").strip()
            b_raw = str(params.get("selection_b_path") or "").strip()
            sel_a = _read_name_list(Path(a_raw)) if a_raw else []
            sel_b = _read_name_list(Path(b_raw)) if b_raw else []
            return (
                adata.obs.index.isin(sel_a).to_numpy(),
                adata.obs.index.isin(sel_b).to_numpy(),
            )
        group_col = str(params.get("group_col") or "")
        if group_col not in adata.obs.columns:
            return np.zeros(adata.n_obs, dtype=bool), np.zeros(adata.n_obs, dtype=bool)
        labels = adata.obs[group_col].astype(str)
        fg_vals = [str(v) for v in (params.get("fg_vals") or []) if str(v).strip()]
        bg_vals = [str(v) for v in (params.get("bg_vals") or []) if str(v).strip()]
        fg_mask = labels.isin(fg_vals).to_numpy()
        bg_mask = (~fg_mask) if not bg_vals else labels.isin(bg_vals).to_numpy()
        return fg_mask, bg_mask

    def start_de_job_ui(
        use_sel,
        view_bundle,
        method,
        topn,
        bgq,
        gcol,
        fgvals,
        bmode,
        bgvals,
        sess_bundle_dict,
    ):
        adx = _load_current_view_h5ad(view_bundle)
        if adx is None:
            return "Render a view first.", *_artifact_updates([]), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), None
        session_dir = str((sess_bundle_dict or {}).get("dir") or (view_bundle or {}).get("dir") or "")
        if not session_dir:
            return "No session directory available.", *_artifact_updates([]), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), None
        result_name = "de_selection.csv" if bool(use_sel) else "de_results_manual.csv"
        params: Dict[str, Any]
        if bool(use_sel):
            sel_path = Path(session_dir) / "selection.txt"
            if not sel_path.exists():
                return "Save or apply a selection first.", *_artifact_updates([]), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), None
            params = {
                "mode": "selection",
                "method": str(method or "t-test"),
                "top_n": int(topn or 50),
                "bg_query": str(bgq or "").strip(),
                "selection_path": str(sel_path),
            }
        else:
            fg = [str(x).strip() for x in (fgvals or []) if str(x).strip()]
            bg = [] if str(bmode or "all_others") == "all_others" else [str(x).strip() for x in (bgvals or []) if str(x).strip()]
            params = {
                "mode": "manual",
                "method": str(method or "t-test"),
                "top_n": int(topn or 50),
                "group_col": str(gcol or ""),
                "fg_vals": fg,
                "bg_vals": bg,
            }
        result_path = Path(session_dir) / result_name
        jid = bgjobs.start_de_job(
            str((view_bundle or {}).get("path") or ""),
            params,
            result_path=result_path,
        )
        status = (
            f"Queued DE job.\n"
            f"Job id: {jid}\n"
            f"View: {str((view_bundle or {}).get('path') or '')}\n"
            f"Output: {result_path}"
        )
        bundle = {
            "job_id": jid,
            "target": "primary",
            "status": status,
            "session_dir": session_dir,
            "view_path": str((view_bundle or {}).get("path") or ""),
        }
        return (
            status,
            *_artifact_updates(
                _job_artifacts_from_job(
                    jid,
                    session_dir=session_dir,
                    extra_paths=[("DE CSV target", str(result_path))],
                )
            ),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            bundle,
        )

    def _copy_selection(sess_bundle_dict, tag: str):
        d = (sess_bundle_dict or {}).get("dir")
        if not d:
            return
        src = Path(d) / "selection.txt"
        if not src.exists():
            return
        (Path(d) / f"selection_{tag}.txt").write_text(src.read_text())

    save_sel_A.click(
        lambda s: _copy_selection(s, "A"), inputs=[view_state], outputs=[]
    )
    save_sel_B.click(
        lambda s: _copy_selection(s, "B"), inputs=[view_state], outputs=[]
    )

    def start_de_ab_job_ui(
        view_bundle,
        method,
        topn,
        sess_bundle_dict,
    ):
        adx = _load_current_view_h5ad(view_bundle)
        if adx is None:
            return "Render a view first.", *_artifact_updates([]), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), None
        session_dir = str((sess_bundle_dict or {}).get("dir") or (view_bundle or {}).get("dir") or "")
        if not session_dir:
            return "No session directory available.", *_artifact_updates([]), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), None
        a_path = Path(session_dir) / "selection_A.txt"
        b_path = Path(session_dir) / "selection_B.txt"
        if not a_path.exists() or not b_path.exists():
            return "Save selections A and B first.", *_artifact_updates([]), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), None
        result_path = Path(session_dir) / "de_ab.csv"
        jid = bgjobs.start_de_job(
            str((view_bundle or {}).get("path") or ""),
            {
                "mode": "ab",
                "method": str(method or "t-test"),
                "top_n": int(topn or 50),
                "selection_a_path": str(a_path),
                "selection_b_path": str(b_path),
            },
            result_path=result_path,
        )
        status = (
            f"Queued A vs B DE job.\n"
            f"Job id: {jid}\n"
            f"View: {str((view_bundle or {}).get('path') or '')}\n"
            f"Output: {result_path}"
        )
        bundle = {
            "job_id": jid,
            "target": "ab",
            "status": status,
            "session_dir": session_dir,
            "view_path": str((view_bundle or {}).get("path") or ""),
        }
        return (
            status,
            *_artifact_updates(
                _job_artifacts_from_job(
                    jid,
                    session_dir=session_dir,
                    extra_paths=[("DE CSV target", str(result_path))],
                )
            ),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            bundle,
        )

    def refresh_de_status(
        de_bundle,
        rank_by,
        sig_axis,
        sig_thresh,
        fc_thresh,
        dot_facet,
    ):
        if not isinstance(de_bundle, dict):
            return "No DE job yet.", *_artifact_updates([]), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), None
        jid = str(de_bundle.get("job_id") or "").strip()
        if not jid:
            return str(de_bundle.get("status") or "No DE job yet."), *_artifact_updates([]), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), de_bundle
        meta = bgjobs.read_meta(jid) or {}
        status_txt = bgjobs.format_meta_report(jid)
        session_dir = str(de_bundle.get("session_dir") or "")
        result_path = str(meta.get("result_path") or "").strip()
        updates = _artifact_updates(
            _job_artifacts_from_job(
                jid,
                session_dir=session_dir,
                extra_paths=[("DE CSV", result_path)] if result_path else None,
            )
        )
        if meta.get("status") != "done" or not result_path:
            return status_txt, *updates, gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), {**de_bundle, "status": status_txt}
        try:
            df = pd.read_csv(result_path)
        except Exception as e:
            return f"{status_txt}\n\nDE load error: {e}", *updates, gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), {**de_bundle, "status": status_txt}
        df = _rank_df(df, rank_by)
        if df.empty:
            bundle = {**de_bundle, "status": status_txt, "path": result_path}
            if str(de_bundle.get("target") or "primary") == "ab":
                return (
                    status_txt,
                    *updates,
                    gr.skip(),
                    gr.skip(),
                    gr.skip(),
                    gr.skip(),
                    df,
                    gr.update(value=result_path),
                    go.Figure(),
                    go.Figure(),
                    bundle,
                )
            return (
                status_txt,
                *updates,
                df,
                gr.update(value=result_path),
                go.Figure(),
                go.Figure(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                bundle,
            )
        view_path = str(meta.get("input_h5ad") or de_bundle.get("view_path") or "")
        adx = ad.read_h5ad(view_path)
        fg_mask, bg_mask = _de_masks_from_params(adx, meta.get("params") or {})
        volc = _make_volcano(
            df,
            sig_axis,
            float(sig_thresh or 0.05),
            float(fc_thresh or 0.25),
            10,
        )
        dotp = _make_dotplot(
            adx,
            df["gene"].astype(str).tolist()[:50],
            fg_mask,
            bg_mask,
            dot_facet if isinstance(dot_facet, str) else None,
        )
        suffix = "ab" if str(de_bundle.get("target") or "primary") == "ab" else "primary"
        _save_plotly_figure_if_session(volc, session_dir, f"de_volcano_{suffix}")
        _save_plotly_figure_if_session(dotp, session_dir, f"de_dotplot_{suffix}")
        bundle = {**de_bundle, "status": status_txt, "path": result_path}
        if str(de_bundle.get("target") or "primary") == "ab":
            return (
                status_txt,
                *updates,
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                df,
                gr.update(value=result_path),
                volc,
                dotp,
                bundle,
            )
        return (
            status_txt,
            *updates,
            df,
            gr.update(value=result_path),
            volc,
            dotp,
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            bundle,
        )

    run_de_ab_btn.click(
        start_de_ab_job_ui,
        inputs=[
            view_state,
            de_method,
            de_topn,
            session_state,
        ],
        outputs=[
            de_status,
            de_artifacts,
            de_artifact_path,
            de_table,
            de_csv,
            volcano_plot,
            dot_plot,
            de_table_ab,
            de_csv_ab,
            volcano_plot_ab,
            dot_plot_ab,
            de_state,
        ],
    )

    run_de_btn.click(
        start_de_job_ui,
        inputs=[
            use_selection_fg,
            view_state,
            de_method,
            de_topn,
            de_bg_query,
            de_group_col,
            de_fore_vals,
            de_bg_mode,
            de_back_vals,
            session_state,
        ],
        outputs=[
            de_status,
            de_artifacts,
            de_artifact_path,
            de_table,
            de_csv,
            volcano_plot,
            dot_plot,
            de_table_ab,
            de_csv_ab,
            volcano_plot_ab,
            dot_plot_ab,
            de_state,
        ],
    )
    refresh_de_btn.click(
        refresh_de_status,
        inputs=[
            de_state,
            de_rank_by,
            de_sig_axis,
            de_sig_thresh,
            de_fc_thresh,
            dot_facet_by,
        ],
        outputs=[
            de_status,
            de_artifacts,
            de_artifact_path,
            de_table,
            de_csv,
            volcano_plot,
            dot_plot,
            de_table_ab,
            de_csv_ab,
            volcano_plot_ab,
            dot_plot_ab,
            de_state,
        ],
    )
    de_artifacts.change(
        _artifact_path_choice,
        inputs=[de_artifacts],
        outputs=[de_artifact_path],
    )

    def make_bundle(sess_bundle_dict, view_bundle_dict):
        import shutil
        import zipfile

        d = (sess_bundle_dict or {}).get("dir")
        if not d:
            return gr.update(value=None)
        base = Path(d)
        zip_path = base / "bundle.zip"
        try:
            current_view = str((view_bundle_dict or {}).get("path") or "")
            if not current_view:
                current_view = str(base / "embedding_subset.h5ad")
            adx = ad.read_h5ad(current_view)
            U = adx.obsm.get("X_umap")
            if U is not None:
                umap_csv = base / "umap_coords.csv"
                pd.DataFrame(
                    {"cell": adx.obs_names, "UMAP1": U[:, 0], "UMAP2": U[:, 1]}
                ).to_csv(umap_csv, index=False)
            if current_view and Path(current_view).exists():
                target = base / "current_view.h5ad"
                if Path(current_view).resolve() != target.resolve():
                    shutil.copyfile(current_view, target)
        except Exception:
            pass
        with zipfile.ZipFile(zip_path, "w") as zf:
            for name in [
                "embedding_subset.h5ad",
                "current_view.h5ad",
                "umap_coords.csv",
                "selection.txt",
                "selection_A.txt",
                "selection_B.txt",
                "de_results.csv",
                "de_selection.csv",
                "de_results_manual.csv",
                "de_ab.csv",
                "umap_plot_map.csv",
            ]:
                p = base / name
                if p.exists():
                    zf.write(p, arcname=name)
        return gr.update(value=str(zip_path))

    mk_bundle.click(make_bundle, inputs=[session_state, view_state], outputs=[bundle_file])

    # ---------------------- Concept Bottleneck Page ----------------------
    with gr.Accordion("Concept Bottleneck (GO-based)", open=False):
        gr.Markdown(
            "Compute GO concept activations per cell and train a concept-bottleneck classifier for cell type."
        )
        cbn_label_col = gr.Textbox(label="Label column (obs)", value="cell_type")
        cbn_protocol = gr.Radio(
            ["logreg", "mlp"], value="logreg", label="Classifier protocol"
        )
        cbn_test_frac = gr.Slider(
            label="Test fraction", value=0.2, minimum=0.05, maximum=0.5, step=0.05
        )
        cbn_min_genes = gr.Slider(
            label="Min genes per GO set (present in data)",
            value=5,
            minimum=1,
            maximum=100,
            step=1,
        )
        cbn_topk = gr.Slider(
            label="Top K concepts per class", value=10, minimum=3, maximum=50, step=1
        )

        # GO gene set picker
        def _list_gmt():
            base = Path(__file__).resolve().parent / "data" / "go"
            opts = []
            if base.exists():
                for f in sorted(base.glob("*.gmt")):
                    opts.append((f.name, str(f)))
            return opts

        cbn_gmt_pick = gr.Dropdown(
            choices=_list_gmt(), label="GO gene set (GMT in app/data/go)"
        )
        cbn_gmt_path = gr.Textbox(label="Or path to GMT file")
        cbn_run = gr.Button("Run Concept Bottleneck")
        cbn_status = gr.Textbox(label="Status", interactive=False)
        cbn_metrics = gr.Dataframe(label="Metrics (F1 macro/micro; per-class)")
        cbn_bar = gr.Plot(label="Top Concepts Per Class (bar)")
        cbn_heat = gr.Plot(label="Concept Importances Heatmap")
        cbn_coeffs_csv = gr.File(label="Coefficients CSV", interactive=False)

        def _parse_gmt(path: str) -> dict:
            gs = {}
            with open(path, "r") as fh:
                for line in fh:
                    parts = line.strip().split("\t")
                    if len(parts) < 3:
                        continue
                    name = parts[0]
                    genes = [g.strip().upper() for g in parts[2:] if g.strip()]
                    if genes:
                        gs[name] = genes
            return gs

        def _compute_concepts(
            adata: ad.AnnData, gene_sets: dict, min_genes: int
        ) -> tuple[np.ndarray, list[str]]:
            # Map var_names to indices (case-insensitive)
            var_upper = pd.Index(adata.var_names.astype(str).str.upper())
            name_to_idx = {}
            for i, g in enumerate(var_upper):
                if g not in name_to_idx:
                    name_to_idx[g] = i
            keep_names = []
            idx_lists = []
            for name, genes in gene_sets.items():
                idxs = [name_to_idx[g] for g in genes if g in name_to_idx]
                if len(idxs) >= max(1, int(min_genes)):
                    keep_names.append(name)
                    idx_lists.append(np.array(idxs, dtype=int))
            if not idx_lists:
                return np.zeros((adata.n_obs, 0), dtype=np.float32), []
            X = adata.X
            if sp.issparse(X):
                X = X.tocsr()
            C = np.zeros((adata.n_obs, len(idx_lists)), dtype=np.float32)
            for j, idxs in enumerate(idx_lists):
                if sp.issparse(adata.X):
                    sub = X[:, idxs]
                    m = np.asarray(sub.mean(axis=1)).ravel()
                else:
                    m = X[:, idxs].mean(axis=1)
                C[:, j] = m.astype(np.float32)
            return C, keep_names

        def run_cbn_fn(
            server_h5ad_path,
            transpose,
            sess_bundle_dict,
            label_col,
            protocol,
            test_frac,
            min_genes,
            topk,
            gmt_choice,
            gmt_path,
        ):
            # Load data via same mechanism
            try:
                adata = _load_adata_from_inputs(None, server_h5ad_path or "", bool(transpose))
            except Exception as e:
                return (
                    "Error loading data: %s" % e,
                    gr.update(value=None),
                    go.Figure(),
                    go.Figure(),
                    gr.update(value=None),
                )
            # Choose GMT
            gmt_file = None
            if (
                gmt_choice
                and isinstance(gmt_choice, str)
                and len(gmt_choice.strip()) > 0
            ):
                # dropdown provides (label,value); Gradio passes value
                gmt_file = gmt_choice
            if (not gmt_file) and gmt_path:
                gmt_file = gmt_path
            if not gmt_file or not Path(gmt_file).exists():
                return (
                    "Missing GO GMT file (pick from dropdown or set path)",
                    gr.update(value=None),
                    go.Figure(),
                    go.Figure(),
                    gr.update(value=None),
                )
            try:
                gene_sets = _parse_gmt(str(gmt_file))
            except Exception as e:
                return (
                    "Error reading GMT: %s" % e,
                    gr.update(value=None),
                    go.Figure(),
                    go.Figure(),
                    gr.update(value=None),
                )
            # Build concepts
            C, names = _compute_concepts(adata, gene_sets, int(min_genes))
            if C.shape[1] == 0:
                return (
                    "No GO sets passed min_genes threshold.",
                    gr.update(value=None),
                    go.Figure(),
                    go.Figure(),
                    gr.update(value=None),
                )
            # Labels
            if label_col not in adata.obs.columns:
                return (
                    f"Label column '{label_col}' not found in obs.",
                    gr.update(value=None),
                    go.Figure(),
                    go.Figure(),
                    gr.update(value=None),
                )
            y = adata.obs[label_col].astype(str).values
            try:
                Xtr, Xte, ytr, yte = train_test_split(
                    C, y, test_size=float(test_frac), random_state=0, stratify=y
                )
            except Exception:
                Xtr, Xte, ytr, yte = train_test_split(
                    C, y, test_size=float(test_frac), random_state=0
                )
            if protocol == "mlp":
                clf = make_pipeline(
                    StandardScaler(with_mean=True),
                    MLPClassifier(
                        hidden_layer_sizes=(256,),
                        activation="relu",
                        early_stopping=True,
                        max_iter=200,
                        random_state=0,
                    ),
                )
            else:
                clf = make_pipeline(
                    StandardScaler(with_mean=True),
                    LogisticRegression(max_iter=1000, multi_class="auto"),
                )
            clf.fit(Xtr, ytr)
            yhat = clf.predict(Xte)
            f1_ma = f1_score(yte, yhat, average="macro")
            f1_mi = f1_score(yte, yhat, average="micro")
            # Per-class F1
            rep = classification_report(yte, yhat, output_dict=True)
            rows = [
                {"metric": "f1_macro", "value": f1_ma},
                {"metric": "f1_micro", "value": f1_mi},
            ]
            for cls, dd in rep.items():
                if isinstance(dd, dict) and "f1-score" in dd:
                    rows.append({"metric": f"f1_{cls}", "value": dd["f1-score"]})
            dfm = pd.DataFrame(rows)
            # Get coefficients for interpretability (works for LR; for MLP, use permutation of last layer weights via feature importances is complex; we approximate with absolute SHAP-like with coef_ fallback not available -> fall back to logistic if mlp)
            coefs = None
            classes_ = None
            try:
                lr = clf.named_steps.get("logisticregression", None)
                sc = clf.named_steps.get("standardscaler", None)
                if lr is not None and sc is not None:
                    coef = lr.coef_
                    classes_ = lr.classes_
                    # scale back to concept feature scale: coefficients already applied on standardized features; for ranking, relative values suffice
                    coefs = coef
            except Exception:
                coefs = None
            # Build plots
            bar_fig = go.Figure()
            heat_fig = go.Figure()
            coeff_csv_path = None
            if coefs is not None and classes_ is not None:
                k = max(1, int(topk))
                # Bar: top K per class
                for i, cls in enumerate(classes_):
                    w = coefs[i]
                    order = np.argsort(-np.abs(w))[:k]
                    bar_fig.add_trace(
                        go.Bar(name=str(cls), x=[names[j] for j in order], y=w[order])
                    )
                bar_fig.update_layout(
                    barmode="group",
                    title="Top Concepts Per Class",
                    xaxis_title="GO set",
                    yaxis_title="Coefficient",
                )
                # Heatmap across classes x concepts (top concepts union)
                union = set()
                for i in range(coefs.shape[0]):
                    union.update(np.argsort(-np.abs(coefs[i]))[:k])
                union = list(union)
                heat = np.vstack([coefs[i, union] for i in range(coefs.shape[0])])
                heat_fig = px.imshow(
                    heat,
                    x=[names[j] for j in union],
                    y=[str(c) for c in classes_],
                    color_continuous_scale="RdBu",
                    origin="lower",
                    aspect="auto",
                    title="Concept Importances (abs signed coeffs)",
                )
                # Save coefficients CSV
                try:
                    d = (sess_bundle_dict or {}).get("dir")
                    if d:
                        coeff_csv_path = str(Path(d) / "cbn_coefficients.csv")
                        dfc = pd.DataFrame(
                            coefs, index=[str(c) for c in classes_], columns=names
                        )
                        dfc.to_csv(coeff_csv_path)
                except Exception:
                    coeff_csv_path = None
            status = f"Train/test split done. F1_macro={f1_ma:.3f}, F1_micro={f1_mi:.3f}. Concepts used: {C.shape[1]}"
            return (
                status,
                dfm,
                bar_fig,
                heat_fig,
                gr.update(value=coeff_csv_path)
                if coeff_csv_path
                else gr.update(value=None),
            )

        cbn_run.click(
            run_cbn_fn,
            inputs=[
                server_path,
                transpose,
                session_state,
                cbn_label_col,
                cbn_protocol,
                cbn_test_frac,
                cbn_min_genes,
                cbn_topk,
                cbn_gmt_pick,
                cbn_gmt_path,
            ],
            outputs=[cbn_status, cbn_metrics, cbn_bar, cbn_heat, cbn_coeffs_csv],
        )

if __name__ == "__main__":
    launch_gradio_demo(
        demo,
        default_port=int(os.environ.get("PORT", "7860")),
        app_label="scFMs embeddings",
    )
