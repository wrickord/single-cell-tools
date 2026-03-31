"""Per-dataset session folders: timestamps, saved figures, compatibility PDFs.

Default layout (when ``SCFMS_SESSION_DIR`` is unset):
  ``<repo>/.assets/<dataset_bucket>/<YYYYMMDD_HHMMSS>_<dataset_label>/``

  *dataset_bucket* is derived from ``source_path`` in order:

  1. Path relative to ``<repo>/.data/`` → first component (e.g. ``2025-05-HNOCA/data/a.h5ad`` → ``2025-05-HNOCA``).
  2. Path relative to ``<repo>/`` with leading ``.data/`` → the dataset folder after ``.data``.
  3. Any path containing a ``data`` or ``processed`` directory segment → parent of that segment
     (e.g. ``…/2025-05-HNOCA/data/hnoca_extended.h5ad`` → ``2025-05-HNOCA``).
  4. Else sanitized ``dataset_label`` (not the first ``/n`` of an absolute path).

Environment (optional):
  SCFMS_SESSION_DIR           — if set, session folders are created directly under this path (no ``.assets`` grouping)
  SCFMS_UMAP_PLOT_MAX_CELLS   — max points drawn on UMAP scatter plots (default: 200000)
  SCFMS_DIST_SAMPLE_CELLS     — cells sampled for QC distribution histograms (default: 5000)
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from gradio_config import repo_root

try:
    from matplotlib.figure import Figure
except ImportError:  # pragma: no cover
    Figure = Any  # type: ignore


def sessions_base() -> Path:
    """Root used in help text: ``SCFMS_SESSION_DIR`` or ``<repo>/.assets`` (per-session dirs add a path segment + timestamp)."""
    raw = os.environ.get("SCFMS_SESSION_DIR", "").strip()
    if raw:
        p = Path(os.path.expandvars(os.path.expanduser(raw))).resolve()
    else:
        p = repo_root() / ".assets"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _assets_segment_from_repo_relative(path: Path, *, data_root: Path, repo: Path) -> Optional[str]:
    """First path component under ``.data``, or under repo with managed layout."""
    try:
        rel = path.resolve().relative_to(data_root.resolve())
        if rel.parts:
            return rel.parts[0]
    except ValueError:
        pass
    try:
        rel = path.resolve().relative_to(repo.resolve())
        parts = rel.parts
        if len(parts) >= 2 and parts[0] == ".data":
            return parts[1]
        if parts:
            return parts[0]
    except ValueError:
        pass
    return None


def _assets_segment_from_data_or_processed_parent(path: Path) -> Optional[str]:
    """``…/<dataset>/data/…`` or ``…/<dataset>/processed/…`` → ``<dataset>``."""
    parts = path.parts
    for marker in ("data", "processed"):
        for i, part in enumerate(parts):
            if part == marker and i >= 1:
                parent = parts[i - 1]
                if parent not in (".", "..", "") and parent != marker:
                    return parent
    return None


def _assets_path_segment(source_path: str, dataset_label: str) -> str:
    """Stable folder name under ``.assets/`` from managed paths, not leading ``/n``."""
    from background_jobs import sanitize_dataset_folder

    raw = (source_path or "").strip()
    if not raw:
        flat = sanitize_dataset_folder(dataset_label, "dataset")[:80]
        return flat if flat else "dataset"

    try:
        path = Path(os.path.expandvars(os.path.expanduser(raw))).resolve()
    except OSError:
        path = None

    if path is not None:
        repo = repo_root()
        data_root = repo / ".data"
        seg = _assets_segment_from_repo_relative(path, data_root=data_root, repo=repo)
        if seg:
            flat = sanitize_dataset_folder(seg, "dataset")[:80]
            if flat:
                return flat
        seg2 = _assets_segment_from_data_or_processed_parent(path)
        if seg2:
            flat = sanitize_dataset_folder(seg2, "dataset")[:80]
            if flat:
                return flat

    flat = sanitize_dataset_folder(dataset_label, "dataset")[:80]
    return flat if flat else "dataset"


def session_storage_dir(source_path: str, dataset_label: str) -> Path:
    """Parent directory for new timestamped session folders (see module docstring)."""
    raw = os.environ.get("SCFMS_SESSION_DIR", "").strip()
    if raw:
        p = Path(os.path.expandvars(os.path.expanduser(raw))).resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p
    seg = _assets_path_segment(source_path, dataset_label)
    out = (repo_root() / ".assets" / seg).resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out


def umap_plot_max_cells() -> int:
    raw = os.environ.get("SCFMS_UMAP_PLOT_MAX_CELLS", "200000").strip()
    try:
        n = int(raw)
    except ValueError:
        n = 200_000
    return max(5_000, min(n, 5_000_000))


def dist_sample_cells() -> int:
    raw = os.environ.get("SCFMS_DIST_SAMPLE_CELLS", "5000").strip()
    try:
        n = int(raw)
    except ValueError:
        n = 5_000
    return max(500, min(n, 500_000))


def embed_table_max_rows() -> int:
    raw = os.environ.get("SCFMS_EMBED_TABLE_MAX_ROWS", "5000").strip()
    try:
        n = int(raw)
    except ValueError:
        n = 5_000
    return max(100, min(n, 100_000))


def session_dir_optional(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    t = str(s).strip()
    if not t:
        return None
    p = Path(t).expanduser()
    return str(p.resolve()) if p.is_dir() else None


def create_dataset_session(
    dataset_label: str,
    *,
    source_kind: str,
    source_path: str,
    n_obs: int,
    n_vars: int,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    from background_jobs import sanitize_dataset_folder

    base = session_storage_dir(source_path, dataset_label)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe = sanitize_dataset_folder(dataset_label, "dataset")[:60]
    d = base / f"{ts}_{safe}"
    d.mkdir(parents=False, exist_ok=False)
    plots = d / "plots"
    plots.mkdir()
    meta: Dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_label": dataset_label,
        "source_kind": source_kind,
        "source_path": source_path,
        "n_obs": int(n_obs),
        "n_vars": int(n_vars),
    }
    if extra:
        meta["extra"] = extra
    (d / "session_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return d.resolve()


def save_matplotlib_figure(
    fig: Any,
    session_dir: Path,
    stem: str,
    *,
    dpi: int = 150,
) -> Optional[Path]:
    if fig is None:
        return None
    sd = Path(session_dir)
    plots = sd / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    path = plots / f"{stem}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def save_figures_if_session(
    figs: Dict[str, Any],
    session_dir: Optional[str],
    *,
    dpi: int = 150,
) -> None:
    """Save multiple figures keyed by filename stem (under ``<session>/plots/``)."""
    if session_dir is None or not str(session_dir).strip():
        return
    sd = Path(str(session_dir).strip()).expanduser().resolve()
    for stem, fig in figs.items():
        if fig is not None:
            save_matplotlib_figure(fig, sd, stem, dpi=dpi)


COMPUTE_RAM_LOG = "compute_ram_plan.log"


def append_compute_ram_log(
    session_dir: Path | str,
    title: str,
    lines: list[str],
) -> Optional[Path]:
    """
    Append a timestamped block to ``<session>/compute_ram_plan.log`` for offline sizing
    (Slurm ``--mem``, interactive RAM, GPU host RAM).
    """
    if not str(session_dir).strip():
        return None
    try:
        sd = Path(str(session_dir).strip()).expanduser().resolve()
    except OSError:
        return None
    if not sd.is_dir():
        return None
    logf = sd / COMPUTE_RAM_LOG
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    block = [
        "",
        "=" * 72,
        f"# {ts}  {title}",
        "",
        *lines,
        "",
    ]
    try:
        with logf.open("a", encoding="utf-8") as fh:
            fh.write("\n".join(block))
    except OSError:
        return None
    return logf
