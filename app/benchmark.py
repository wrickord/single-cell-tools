#!/usr/bin/env python3
"""Gradio app: benchmark sklearn classifiers (logistic regression, MLP) on every embedding source.

Trains one model per (matrix / layer / obsm source) × (categorical obs column), saves artifacts
under a session folder (same pattern as preprocess), and evaluates a separate test AnnData using
saved bundles (gene alignment for expression matrices, dim check for obsm).

When **Classifier = mlp** and PyTorch sees **CUDA** (or **Apple MPS**), the MLP is trained with
**torch** on that device; otherwise sklearn's **MLPClassifier** on CPU is used. Logistic regression
stays CPU (sklearn). Set **SCFMS_BENCH_USE_GPU_MLP=0** to force sklearn MLP on CPU.
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import anndata as ad
import gradio as gr
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
)
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

_ROOT = Path(__file__).resolve().parent.parent
_APP = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_APP) not in sys.path:
    sys.path.insert(0, str(_APP))

import session_results as sess_res
import background_jobs as bgjobs
from background_jobs import validate_server_read_path
from gradio_config import launch_gradio_demo, runtime_info_markdown
from slurm_defaults import default_slurm_partition, effective_slurm_partition

from preprocess import (
    _adata_buffer_parts,
    _compute_ram_plan_lines,
    _list_plot_source_options,
    _matrix_from_spec,
    _matrix_or_obsm_from_spec,
    _matrix_nbytes,
    _try_append_compute_ram_log,
    estimate_adata_memory_report,
)

from scripts.generate_embeddings import load_adata

MANIFEST_NAME = "benchmark_manifest.json"
MODELS_SUBDIR = "models"
MLP_FEATURE_CAP = int(os.environ.get("SCFMS_BENCH_MLP_MAX_FEATURES", "8192"))
# Set to 0 to force sklearn MLP on CPU even when CUDA is available.
_USE_GPU_MLP_ENV = os.environ.get("SCFMS_BENCH_USE_GPU_MLP", "1").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
)


def _accelerated_mlp_available() -> bool:
    """True if PyTorch can use CUDA or Apple MPS for the custom MLP (not sklearn's MLP)."""
    if not _USE_GPU_MLP_ENV:
        return False
    try:
        import torch

        if torch.cuda.is_available():
            return True
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return True
        return False
    except Exception:
        return False


def _torch_train_device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class TorchMLPClassifier(BaseEstimator, ClassifierMixin):
    """Sklearn-like MLP trained with PyTorch on CUDA when available (CPU otherwise).

    Drop-in replacement for the final step of a ``Pipeline`` after dense scaling.
    """

    def __init__(
        self,
        hidden_layer_sizes: Tuple[int, ...],
        n_classes: int,
        max_iter: int = 200,
        random_state: int = 0,
        validation_fraction: float = 0.1,
        n_iter_no_change: int = 12,
        batch_size: int = 2048,
    ):
        self.hidden_layer_sizes = tuple(hidden_layer_sizes)
        self.n_classes = int(n_classes)
        self.max_iter = int(max_iter)
        self.random_state = int(random_state)
        self.validation_fraction = float(validation_fraction)
        self.n_iter_no_change = int(n_iter_no_change)
        self.batch_size = int(max(batch_size, 1))
        self._module: Any = None
        self.classes_: Optional[np.ndarray] = None
        self.n_features_in_: Optional[int] = None
        self.n_iter_: int = 0
        self._train_device_str: str = "cpu"

    def _build_module(self, n_in: int, n_classes: int) -> Any:
        import torch
        import torch.nn as nn

        layers: List[Any] = []
        prev = n_in
        for h in self.hidden_layer_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        return nn.Sequential(*layers)

    def fit(self, X, y) -> "TorchMLPClassifier":
        import torch
        import torch.nn as nn

        Xd = np.asarray(X, dtype=np.float32, order="C")
        y = np.asarray(y, dtype=np.int64)
        if Xd.ndim != 2:
            raise ValueError("X must be 2D")
        self.n_features_in_ = int(Xd.shape[1])
        n_classes = int(self.n_classes)
        if n_classes < 2:
            raise ValueError("Need at least 2 classes")
        self.classes_ = np.arange(n_classes, dtype=np.int64)

        device = _torch_train_device()
        self._train_device_str = str(device)

        torch.manual_seed(self.random_state)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(self.random_state)

        net = self._build_module(self.n_features_in_, n_classes).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        crit = nn.CrossEntropyLoss()

        n = Xd.shape[0]
        vf = self.validation_fraction
        if vf > 0 and n >= 5 and int(n * vf) >= 1:
            X_tr, X_va, y_tr, y_va = train_test_split(
                Xd,
                y,
                test_size=vf,
                random_state=self.random_state,
                stratify=y if n_classes > 1 else None,
            )
        else:
            X_tr, y_tr = Xd, y
            X_va = y_va = None

        def batches(Xa: np.ndarray, ya: np.ndarray):
            bs = min(self.batch_size, max(1, Xa.shape[0]))
            for i in range(0, Xa.shape[0], bs):
                yield Xa[i : i + bs], ya[i : i + bs]

        best_val = float("inf")
        stall = 0
        self.n_iter_ = 0

        for epoch in range(self.max_iter):
            net.train()
            for xb, yb in batches(X_tr, y_tr):
                xbt = torch.from_numpy(xb).to(device)
                ybt = torch.from_numpy(yb).to(device)
                opt.zero_grad(set_to_none=True)
                logits = net(xbt)
                loss = crit(logits, ybt)
                loss.backward()
                opt.step()

            self.n_iter_ = epoch + 1

            if X_va is not None:
                net.eval()
                with torch.no_grad():
                    xv = torch.from_numpy(X_va).to(device)
                    yv = torch.from_numpy(y_va).to(device)
                    vloss = float(crit(net(xv), yv).item())
                if vloss < best_val - 1e-7:
                    best_val = vloss
                    stall = 0
                else:
                    stall += 1
                if stall >= self.n_iter_no_change:
                    break

        self._module = net
        return self

    def _predict_logits(self, X) -> np.ndarray:
        import torch

        if self._module is None:
            raise RuntimeError("Model not fitted")
        Xd = np.asarray(X, dtype=np.float32, order="C")
        device = _torch_train_device()
        self._module = self._module.to(device)
        self._module.eval()
        out: List[np.ndarray] = []
        bs = min(self.batch_size, max(1, Xd.shape[0]))
        with torch.no_grad():
            for i in range(0, Xd.shape[0], bs):
                chunk = torch.from_numpy(Xd[i : i + bs]).to(device)
                out.append(self._module(chunk).float().cpu().numpy())
        return np.vstack(out)

    def predict(self, X) -> np.ndarray:
        logits = self._predict_logits(X)
        return np.argmax(logits, axis=1).astype(np.int64)

    def predict_proba(self, X) -> np.ndarray:
        import torch

        logits = self._predict_logits(X)
        t = torch.from_numpy(logits)
        prob = torch.softmax(t, dim=1).numpy()
        return prob.astype(np.float64)

    def __getstate__(self) -> Dict[str, Any]:
        import torch

        sd = None
        if self._module is not None:
            sd = {k: v.detach().cpu() for k, v in self._module.state_dict().items()}
        return {
            "hidden_layer_sizes": self.hidden_layer_sizes,
            "n_classes": self.n_classes,
            "max_iter": self.max_iter,
            "random_state": self.random_state,
            "validation_fraction": self.validation_fraction,
            "n_iter_no_change": self.n_iter_no_change,
            "batch_size": self.batch_size,
            "classes_": self.classes_,
            "n_features_in_": self.n_features_in_,
            "n_iter_": self.n_iter_,
            "_train_device_str": self._train_device_str,
            "_state_dict": sd,
            "_arch_n_in": self.n_features_in_,
            "_arch_n_classes": int(self.n_classes),
        }

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._module = None
        sd = state.get("_state_dict")
        n_in = state.get("_arch_n_in") or state.get("n_features_in_")
        n_cls = state.get("_arch_n_classes") or state.get("n_classes")
        if sd is not None and n_in and n_cls:
            import torch

            self._module = self._build_module(int(n_in), int(n_cls))
            self._module.load_state_dict(sd)
            self._module.train(False)


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
) -> ad.AnnData:
    sp = (server_h5ad_path or "").strip()
    if sp:
        p = validate_server_read_path(sp)
        return load_adata(str(p), transpose=transpose)
    if file_obj is None:
        raise ValueError("Upload a file or set a server path to an .h5ad / .csv / .tsv.")
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


def _norm_sess_bundle(raw: Any) -> Optional[Dict[str, Any]]:
    if raw is None or not isinstance(raw, dict):
        return None
    return raw


def _ensure_session(
    sess_bundle: Optional[Dict[str, Any]],
    src_key: Tuple[str, str],
    adata: ad.AnnData,
    *,
    tag: str,
) -> Tuple[str, Dict[str, Any]]:
    prev = sess_bundle.get("key") if isinstance(sess_bundle, dict) else None
    d = (sess_bundle or {}).get("dir") if isinstance(sess_bundle, dict) else None
    if (
        isinstance(prev, (list, tuple))
        and len(prev) == 2
        and tuple(prev) == src_key
        and d
        and str(d).strip()
    ):
        return str(d), sess_bundle  # type: ignore[return-value]

    if src_key[0] in ("server", "path") and src_key[1]:
        label = Path(src_key[1]).stem
    elif src_key[0] == "upload" and src_key[1]:
        label = Path(src_key[1]).stem
    else:
        label = "benchmark"
    session = sess_res.create_dataset_session(
        f"{tag}_{label}",
        source_kind=src_key[0],
        source_path=src_key[1] or label,
        n_obs=int(adata.n_obs),
        n_vars=int(adata.n_vars),
        extra={"app": "embedding_benchmark", "role": tag},
    )
    bundle = {"key": list(src_key), "dir": str(session)}
    return str(session), bundle


def _var_names_for_spec(adata: ad.AnnData, spec: str) -> Optional[pd.Index]:
    if spec.startswith("obsm:"):
        return None
    if spec == "raw.X":
        if adata.raw is None:
            raise ValueError("raw.X requires AnnData.raw")
        return adata.raw.var_names
    return adata.var_names


def _align_expression_matrix(
    adata: ad.AnnData,
    spec: str,
    var_names_train: Sequence[str],
) -> sp.spmatrix | np.ndarray:
    M = _matrix_from_spec(adata, spec)
    v_idx = _var_names_for_spec(adata, spec)
    name_to_col = {str(g): i for i, g in enumerate(v_idx)}
    n = M.shape[0]
    nt = len(var_names_train)
    if sp.issparse(M):
        from scipy.sparse import lil_matrix

        L = lil_matrix((n, nt), dtype=np.float64)
        for j, g in enumerate(var_names_train):
            ii = name_to_col.get(str(g))
            if ii is not None:
                L[:, j] = M[:, ii]
        return L.tocsr()
    X = np.zeros((n, nt), dtype=np.float64)
    arr = np.asarray(M)
    for j, g in enumerate(var_names_train):
        ii = name_to_col.get(str(g))
        if ii is not None:
            X[:, j] = arr[:, ii]
    return X


def _materialize_X(
    adata: ad.AnnData,
    spec: str,
    *,
    var_names_train: Optional[Sequence[str]] = None,
    expected_obsm_dim: Optional[int] = None,
) -> Tuple[sp.spmatrix | np.ndarray, Dict[str, Any]]:
    meta: Dict[str, Any] = {"spec": spec}
    if spec.startswith("obsm:"):
        X = np.asarray(_matrix_or_obsm_from_spec(adata, spec), dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"{spec} must be 2D, got shape {X.shape}")
        if expected_obsm_dim is not None and X.shape[1] != expected_obsm_dim:
            raise ValueError(
                f"{spec}: expected {expected_obsm_dim} columns, got {X.shape[1]}"
            )
        meta["n_features"] = int(X.shape[1])
        if np.isnan(X).any():
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X, meta
    if var_names_train is None:
        X = _matrix_from_spec(adata, spec)
    else:
        X = _align_expression_matrix(adata, spec, var_names_train)
        meta["aligned_var_names"] = list(var_names_train)
    meta["n_features"] = int(X.shape[1]) if X.ndim == 2 else 0
    if sp.issparse(X):
        return X, meta
    Xd = np.asarray(X, dtype=np.float64)
    if np.isnan(Xd).any():
        Xd = np.nan_to_num(Xd, nan=0.0, posinf=0.0, neginf=0.0)
    return Xd, meta


def _encode_y(series: pd.Series) -> Tuple[np.ndarray, LabelEncoder, np.ndarray]:
    s = pd.Series(series).astype("string")
    mask = s.notna() & (s.str.strip() != "")
    y_raw = s[mask].astype(str)
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    return y, le, np.where(mask.values)[0]


def _maybe_subsample_idx(n: int, max_cells: int, seed: int) -> np.ndarray:
    if max_cells <= 0 or n <= max_cells:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=max_cells, replace=False))


def _parse_hidden(s: str) -> Tuple[int, ...]:
    t = (s or "").strip()
    if not t:
        return (100,)
    parts = [int(x.strip()) for x in re.split(r"[,;\s]+", t) if x.strip()]
    if not parts:
        return (100,)
    return tuple(max(1, p) for p in parts)


def _build_estimator(
    kind: str,
    X_train,
    n_classes: int,
    *,
    seed: int,
    mlp_hidden: str,
    mlp_max_iter: int,
    lr_c: float,
    lr_max_iter: int,
) -> Any:
    kind = (kind or "logistic_regression").strip().lower()
    is_sparse = sp.issparse(X_train)
    n_features = X_train.shape[1]
    if kind == "logistic_regression":
        if is_sparse:
            solver = "saga"
            dual = False
        else:
            solver = "lbfgs" if n_classes <= 2 else "lbfgs"
            dual = False
        return LogisticRegression(
            max_iter=int(lr_max_iter),
            random_state=seed,
            C=float(lr_c),
            n_jobs=-1,
            solver=solver,
            dual=dual,
            multi_class="auto",
        )
    if kind == "mlp":
        hidden = _parse_hidden(mlp_hidden)
        use_torch = _accelerated_mlp_available()
        if use_torch:
            mlp: Any = TorchMLPClassifier(
                hidden_layer_sizes=hidden,
                n_classes=n_classes,
                max_iter=int(mlp_max_iter),
                random_state=seed,
                validation_fraction=0.1,
                n_iter_no_change=12,
            )
        else:
            mlp = MLPClassifier(
                hidden_layer_sizes=hidden,
                max_iter=int(mlp_max_iter),
                random_state=seed,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=12,
            )
        need_svd = is_sparse or n_features > MLP_FEATURE_CAP
        n_samples = X_train.shape[0]
        if need_svd:
            n_comp = min(256, max(2, n_features - 1), max(2, n_samples - 1))
            reduc = TruncatedSVD(n_components=n_comp, random_state=seed)
            if is_sparse:
                return Pipeline(
                    [("svd", reduc), ("scaler", StandardScaler(with_mean=False)), ("mlp", mlp)]
                )
            return Pipeline([("svd", reduc), ("scaler", StandardScaler()), ("mlp", mlp)])
        return Pipeline([("scaler", StandardScaler()), ("mlp", mlp)])
    raise ValueError(f"Unknown classifier: {kind}")


def _metrics_dict(y_true, y_pred, y_proba, labels) -> Dict[str, float]:
    out: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    if y_proba is not None and len(labels) > 1:
        try:
            out["log_loss"] = float(
                log_loss(y_true, y_proba, labels=np.arange(len(labels)))
            )
        except Exception:
            pass
    return out


def _slug(s: str) -> str:
    x = re.sub(r"[^a-zA-Z0-9._-]+", "_", s.strip())[:120]
    return x or "src"


@dataclass
class TrainBenchmarkResult:
    df: Optional[pd.DataFrame]
    status_md: str
    fig_hm: Any
    fig_bar: Any
    bundle: Dict[str, Any]
    session_dir: str
    entries: List[Dict[str, Any]]
    rows_log: List[str]


@dataclass
class EvalBenchmarkResult:
    df: pd.DataFrame
    status_md: str
    fig_hm: Any
    fig_bar: Any
    bundle: Dict[str, Any]
    test_session_dir: str


def train_benchmark_core(
    adata: ad.AnnData,
    src_key: Tuple[str, str],
    train_sess_bundle: Optional[Dict[str, Any]],
    *,
    target_cols: List[str],
    split_mode: str,
    stratify_col: str,
    test_fraction: float,
    random_seed: int,
    classifier_kind: str,
    mlp_hidden: str,
    mlp_max_iter: float,
    lr_c: float,
    lr_max_iter: float,
    max_cells: float,
    skip_sources: List[str],
    per_model_callback: Optional[Callable[[int, str, str, Dict[str, float], Dict[str, Any]], None]] = None,
) -> TrainBenchmarkResult:
    """Train all source×target models; optional callback after each successful model (for wandb, etc.)."""
    session_dir, bundle = _ensure_session(
        _norm_sess_bundle(train_sess_bundle),
        src_key,
        adata,
        tag="train",
    )
    models_dir = Path(session_dir) / MODELS_SUBDIR
    models_dir.mkdir(parents=True, exist_ok=True)

    specs_all = _list_plot_source_options(adata)
    skip_set = set(skip_sources or [])
    specs = [s for s in specs_all if s not in skip_set]
    if not specs:
        raise ValueError("No embedding sources left (check skip list).")

    targets = [str(c) for c in (target_cols or []) if str(c).strip()]
    if not targets:
        raise ValueError("Select at least one target obs column.")

    seed = int(random_seed)
    max_c = int(max_cells) if max_cells is not None else 0
    tf = float(test_fraction)
    if not (0.05 <= tf < 0.95):
        raise ValueError("test_fraction should be in [0.05, 0.95).")

    clf_kind = str(classifier_kind or "logistic_regression")
    entries: List[Dict[str, Any]] = []
    rows_log: List[str] = []
    model_idx = 0

    for spec in specs:
        var_train: Optional[List[str]] = None
        if not spec.startswith("obsm:"):
            try:
                vn = _var_names_for_spec(adata, spec)
                var_train = [str(x) for x in vn]
            except Exception as e:
                rows_log.append(f"skip {spec}: {e}")
                continue

        try:
            X_full, xmeta = _materialize_X(adata, spec, var_names_train=None)
        except Exception as e:
            rows_log.append(f"skip {spec}: features — {e}")
            continue

        ix_all = np.arange(X_full.shape[0])
        ix_sub = _maybe_subsample_idx(len(ix_all), max_c, seed)
        X_full = X_full[ix_sub]
        obs_sub = adata.obs.iloc[ix_sub]

        for tgt in targets:
            if tgt not in adata.obs.columns:
                rows_log.append(f"skip {spec} / {tgt}: missing column")
                continue
            try:
                y_enc, le, row_ix = _encode_y(obs_sub[tgt])
            except Exception as e:
                rows_log.append(f"skip {spec} / {tgt}: labels — {e}")
                continue
            if len(np.unique(y_enc)) < 2:
                rows_log.append(f"skip {spec} / {tgt}: need ≥2 classes")
                continue

            Xi = X_full[row_ix]
            yi = y_enc
            strat = None
            mode = (split_mode or "random").strip().lower()
            if mode == "stratify_obs" and stratify_col and stratify_col != "(none)":
                if stratify_col not in adata.obs.columns:
                    rows_log.append(f"skip {spec} / {tgt}: stratify col missing")
                    continue
                strat = obs_sub.iloc[row_ix][stratify_col].astype(str).values
                u, c = np.unique(strat, return_counts=True)
                if np.any(c < 2) or len(u) < 2:
                    strat = None

            try:
                X_tr, X_va, y_tr, y_va = train_test_split(
                    Xi,
                    yi,
                    test_size=tf,
                    random_state=seed,
                    stratify=strat,
                )
            except ValueError:
                X_tr, X_va, y_tr, y_va = train_test_split(
                    Xi, yi, test_size=tf, random_state=seed, stratify=None
                )

            try:
                est = _build_estimator(
                    clf_kind,
                    X_tr,
                    len(le.classes_),
                    seed=seed,
                    mlp_hidden=mlp_hidden,
                    mlp_max_iter=int(mlp_max_iter),
                    lr_c=float(lr_c),
                    lr_max_iter=int(lr_max_iter),
                )
                est.fit(X_tr, y_tr)
            except Exception as e:
                rows_log.append(f"skip {spec} / {tgt}: fit — {e}")
                continue

            y_pred = est.predict(X_va)
            try:
                y_proba = (
                    est.predict_proba(X_va)
                    if hasattr(est, "predict_proba")
                    else None
                )
            except Exception:
                y_proba = None
            mval = _metrics_dict(y_va, y_pred, y_proba, le.classes_)

            stem = f"{_slug(spec)}__{_slug(tgt)}__{_slug(clf_kind)}"
            bundle_path = models_dir / f"{stem}.joblib"
            payload = {
                "estimator": est,
                "label_encoder": le,
                "spec": spec,
                "target": tgt,
                "classifier": clf_kind,
                "var_names_train": var_train,
                "obsm_n_features": xmeta.get("n_features")
                if spec.startswith("obsm:")
                else None,
                "train_rows": int(Xi.shape[0]),
                "classes": [str(c) for c in le.classes_],
            }
            joblib.dump(payload, bundle_path)

            entry = {
                "spec": spec,
                "target": tgt,
                "classifier": clf_kind,
                "model_relpath": f"{MODELS_SUBDIR}/{bundle_path.name}",
                "metrics_val": mval,
                "n_train_fit": int(X_tr.shape[0]),
                "n_val": int(X_va.shape[0]),
            }
            entries.append(entry)
            rows_log.append(
                f"ok {spec} × {tgt}: acc={mval['accuracy']:.4f} f1_macro={mval['f1_macro']:.4f}"
            )
            if per_model_callback is not None:
                try:
                    per_model_callback(model_idx, spec, tgt, mval, entry)
                except Exception:
                    pass
            model_idx += 1

    manifest: Dict[str, Any] = {
        "version": 1,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "session_dir": session_dir,
        "split": {
            "mode": split_mode,
            "stratify_col": stratify_col
            if stratify_col != "(none)"
            else None,
            "test_fraction": tf,
            "random_seed": seed,
        },
        "classifier": clf_kind,
        "mlp_hidden": mlp_hidden,
        "max_cells_subsample": max_c,
        "targets": targets,
        "entries": entries,
    }
    (Path(session_dir) / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    df = pd.DataFrame(
        [
            {
                "source": e["spec"],
                "target": e["target"],
                **e["metrics_val"],
                "n_val": e["n_val"],
            }
            for e in entries
        ]
    )
    fig_hm = None
    fig_bar = None
    if len(entries):
        pivot = df.pivot_table(
            index="source", columns="target", values="accuracy", aggfunc="mean"
        )
        fig_hm, ax = plt.subplots(figsize=(max(6, pivot.shape[1] * 0.9), max(4, pivot.shape[0] * 0.35)))
        im = ax.imshow(pivot.values, aspect="auto", vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(np.arange(pivot.shape[1]), labels=list(pivot.columns), rotation=45, ha="right")
        ax.set_yticks(np.arange(pivot.shape[0]), labels=list(pivot.index))
        ax.set_title("Validation accuracy (holdout from train)")
        fig_hm.colorbar(im, ax=ax, fraction=0.02)
        fig_hm.tight_layout()

        by_src = df.groupby("source")["accuracy"].mean().sort_values()
        fig_bar, axb = plt.subplots(figsize=(7, max(3, len(by_src) * 0.25)))
        axb.barh(by_src.index.astype(str), by_src.values)
        axb.set_xlim(0, 1)
        axb.set_xlabel("Mean validation accuracy across targets")
        axb.set_title("By embedding source")
        fig_bar.tight_layout()

        sess_res.save_figures_if_session(
            {
                "benchmark_val_accuracy_heatmap": fig_hm,
                "benchmark_val_accuracy_by_source": fig_bar,
            },
            session_dir,
        )

    _try_append_compute_ram_log(
        session_dir,
        "benchmark_train_run_complete",
        _compute_ram_plan_lines(
            adata,
            extras=[
                f"classifier: {clf_kind}",
                f"n_models_trained: {len(entries)}",
                f"manifest: {MANIFEST_NAME}",
            ],
        ),
    )

    status = (
        f"**Train benchmark** — {len(entries)} model(s) saved.\n"
        f"Session: `{session_dir}`\n"
        f"Manifest: `{MANIFEST_NAME}`\n\n"
        + "\n".join(rows_log[-40:])
        + ("\n\n" + estimate_adata_memory_report(adata) if adata is not None else "")
    )
    return TrainBenchmarkResult(
        df=df if len(df) else None,
        status_md=status,
        fig_hm=fig_hm,
        fig_bar=fig_bar,
        bundle=bundle,
        session_dir=session_dir,
        entries=entries,
        rows_log=rows_log,
    )


def eval_benchmark_core(
    adata: ad.AnnData,
    src_key: Tuple[str, str],
    test_sess_bundle: Optional[Dict[str, Any]],
    benchmark_session_dir: str,
    *,
    per_row_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> EvalBenchmarkResult:
    _, tb = _ensure_session(
        _norm_sess_bundle(test_sess_bundle),
        src_key,
        adata,
        tag="test",
    )

    root = Path((benchmark_session_dir or "").strip()).expanduser().resolve()
    man_p = root / MANIFEST_NAME
    if not man_p.is_file():
        raise FileNotFoundError(f"Manifest not found: {man_p}")
    manifest = json.loads(man_p.read_text(encoding="utf-8"))
    models_root = root

    rows: List[Dict[str, Any]] = []
    for e in manifest.get("entries", []):
        spec = e.get("spec")
        tgt = e.get("target")
        rel = e.get("model_relpath")
        if not spec or not tgt or not rel:
            continue
        bp = models_root / rel
        if not bp.is_file():
            row = {
                "source": spec,
                "target": tgt,
                "accuracy": np.nan,
                "note": "missing model file",
            }
            rows.append(row)
            if per_row_callback:
                try:
                    per_row_callback(row)
                except Exception:
                    pass
            continue
        if tgt not in adata.obs.columns:
            row = {
                "source": spec,
                "target": tgt,
                "accuracy": np.nan,
                "note": "target column absent on test",
            }
            rows.append(row)
            if per_row_callback:
                try:
                    per_row_callback(row)
                except Exception:
                    pass
            continue
        try:
            payload = joblib.load(bp)
            le: LabelEncoder = payload["label_encoder"]
            est = payload["estimator"]
            var_names_train = payload.get("var_names_train")
            obsm_n = payload.get("obsm_n_features")
            X_te, _ = _materialize_X(
                adata,
                str(spec),
                var_names_train=var_names_train,
                expected_obsm_dim=obsm_n,
            )
            y_raw = adata.obs[tgt].astype("string")
            mask = y_raw.notna() & (y_raw.str.strip() != "")
            y_strings = y_raw[mask].astype(str).values
            ix = np.where(mask.values)[0]
            Xe = X_te[ix]
            known = set(str(c) for c in le.classes_)
            eval_mask = np.array([s in known for s in y_strings])
            if not np.any(eval_mask):
                row = {
                    "source": spec,
                    "target": tgt,
                    "accuracy": np.nan,
                    "note": "no test labels overlap train classes",
                }
                rows.append(row)
                if per_row_callback:
                    try:
                        per_row_callback(row)
                    except Exception:
                        pass
                continue
            Xe = Xe[eval_mask]
            y_strings = y_strings[eval_mask]
            y_true = le.transform(y_strings)
            y_pred = est.predict(Xe)
            acc = float(accuracy_score(y_true, y_pred))
            f1m = float(
                f1_score(y_true, y_pred, average="macro", zero_division=0)
            )
            row = {
                "source": spec,
                "target": tgt,
                "accuracy": acc,
                "balanced_accuracy": float(
                    balanced_accuracy_score(y_true, y_pred)
                ),
                "f1_macro": f1m,
                "n_test": int(Xe.shape[0]),
                "note": "",
            }
            rows.append(row)
            if per_row_callback:
                try:
                    per_row_callback(row)
                except Exception:
                    pass
        except Exception as ex:
            row = {
                "source": spec,
                "target": tgt,
                "accuracy": np.nan,
                "note": str(ex)[:200],
            }
            rows.append(row)
            if per_row_callback:
                try:
                    per_row_callback(row)
                except Exception:
                    pass

    df = pd.DataFrame(rows)
    fig_hm = None
    fig_bar = None
    ok = df[df["accuracy"].notna()]
    if len(ok):
        pivot = ok.pivot_table(
            index="source", columns="target", values="accuracy", aggfunc="mean"
        )
        fig_hm, ax = plt.subplots(
            figsize=(max(6, pivot.shape[1] * 0.9), max(4, pivot.shape[0] * 0.35))
        )
        im = ax.imshow(pivot.values, aspect="auto", vmin=0, vmax=1, cmap="magma")
        ax.set_xticks(np.arange(pivot.shape[1]), labels=list(pivot.columns), rotation=45, ha="right")
        ax.set_yticks(np.arange(pivot.shape[0]), labels=list(pivot.index))
        ax.set_title("Test accuracy (loaded models)")
        fig_hm.colorbar(im, ax=ax, fraction=0.02)
        fig_hm.tight_layout()
        by_src = ok.groupby("source")["accuracy"].mean().sort_values()
        fig_bar, axb = plt.subplots(figsize=(7, max(3, len(by_src) * 0.25)))
        axb.barh(by_src.index.astype(str), by_src.values)
        axb.set_xlim(0, 1)
        axb.set_xlabel("Mean test accuracy across targets")
        axb.set_title("By embedding source (test)")
        fig_bar.tight_layout()
        out_sess = str(Path(tb["dir"]).resolve())
        sess_res.save_figures_if_session(
            {
                "benchmark_test_accuracy_heatmap": fig_hm,
                "benchmark_test_accuracy_by_source": fig_bar,
            },
            out_sess,
        )

    _try_append_compute_ram_log(
        str(Path(tb["dir"]).resolve()),
        "benchmark_eval_complete",
        _compute_ram_plan_lines(
            adata,
            extras=[
                f"train_benchmark_session: {(benchmark_session_dir or '').strip()}",
                f"n_metric_rows: {len(rows)}",
            ],
        ),
    )

    status = (
        f"**Eval** on `{Path(tb['dir']).name}` (test session folder for plots).\n"
        f"Models from: `{root}`\n"
        f"Rows: {len(rows)}\n\n"
        + estimate_adata_memory_report(adata)
    )
    return EvalBenchmarkResult(
        df=df,
        status_md=status,
        fig_hm=fig_hm,
        fig_bar=fig_bar,
        bundle=tb,
        test_session_dir=str(Path(tb["dir"]).resolve()),
    )


def run_benchmark_train(
    file_obj,
    server_path,
    transpose,
    target_cols: List[str],
    split_mode: str,
    stratify_col: str,
    test_fraction: float,
    random_seed: int,
    classifier_kind: str,
    mlp_hidden: str,
    mlp_max_iter: float,
    lr_c: float,
    lr_max_iter: float,
    max_cells: float,
    skip_sources: List[str],
    train_sess_bundle,
):
    try:
        adata = _load_adata_from_inputs(file_obj, server_path, transpose)
        src_key = _session_src_key(file_obj, server_path)
        out = train_benchmark_core(
            adata,
            src_key,
            _norm_sess_bundle(train_sess_bundle),
            target_cols=target_cols,
            split_mode=split_mode,
            stratify_col=stratify_col,
            test_fraction=test_fraction,
            random_seed=random_seed,
            classifier_kind=classifier_kind,
            mlp_hidden=mlp_hidden,
            mlp_max_iter=mlp_max_iter,
            lr_c=lr_c,
            lr_max_iter=lr_max_iter,
            max_cells=max_cells,
            skip_sources=skip_sources,
        )
        return (
            out.df,
            out.status_md,
            out.fig_hm,
            out.fig_bar,
            out.bundle,
            gr.update(value=out.session_dir),
        )
    except Exception as e:
        return None, f"Error: {e}", None, None, train_sess_bundle, gr.skip()


def run_benchmark_eval(
    file_obj,
    server_path,
    transpose,
    benchmark_session_dir: str,
    test_sess_bundle,
):
    try:
        adata = _load_adata_from_inputs(file_obj, server_path, transpose)
        src_key = _session_src_key(file_obj, server_path)
        out = eval_benchmark_core(
            adata,
            src_key,
            _norm_sess_bundle(test_sess_bundle),
            benchmark_session_dir,
        )
        return (
            out.df,
            out.status_md,
            out.fig_hm,
            out.fig_bar,
            out.bundle,
            gr.update(value=out.test_session_dir),
        )
    except Exception as e:
        return None, f"Error: {e}", None, None, test_sess_bundle, gr.skip()


def _obs_choices(adata: Optional[ad.AnnData]) -> List[str]:
    if adata is None:
        return []
    return [str(c) for c in adata.obs.columns]


def _format_benchmark_wandb_html(job_id: str) -> str:
    jid = (job_id or "").strip()
    if not jid:
        return "<p>Submit a Slurm benchmark job, then enter its <strong>Job ID</strong> here and refresh (or enable auto-poll).</p>"
    m = bgjobs.read_meta(jid)
    if not m:
        return f"<p>No job metadata for <code>{jid}</code> under <code>{bgjobs.jobs_root()}</code>.</p>"
    st = m.get("status", "?")
    wu = (m.get("wandb_url") or "").strip()
    pu = (m.get("wandb_project_url") or "").strip()
    lines = [
        "<div style='font-family:system-ui,sans-serif;line-height:1.5'>",
        f"<p><strong>Job</strong> <code>{jid}</code> — <strong>status:</strong> {st}</p>",
    ]
    if wu:
        lines.append(
            f"<p><a href=\"{wu}\" target=\"_blank\" rel=\"noopener noreferrer\">Open this run in Weights &amp; Biases</a></p>"
        )
        lines.append(
            "<p style='color:#666;font-size:0.9em'>Embedded workspace (may be blocked by W&amp;B headers on some browsers; use the link if the frame is empty).</p>"
        )
        lines.append(
            f"<iframe src=\"{wu}\" title=\"wandb\" style=\"width:100%;min-height:640px;border:1px solid #ddd;border-radius:6px\""
            " sandbox=\"allow-scripts allow-same-origin allow-popups allow-popups-to-escape-sandbox\"></iframe>"
        )
    elif st == "done":
        lines.append(
            "<p>No run URL was recorded. On the compute node, install <code>wandb</code>, run <code>wandb login</code> "
            "or set <code>WANDB_API_KEY</code>, then resubmit.</p>"
        )
    else:
        lines.append("<p>When Slurm finishes successfully, the W&amp;B run link and embed appear here.</p>")
    if pu:
        lines.append(
            f"<p><a href=\"{pu}\" target=\"_blank\" rel=\"noopener noreferrer\">Project on wandb.ai</a></p>"
        )
    lines.append("</div>")
    return "\n".join(lines)


def refresh_benchmark_slurm_panel(job_id: str) -> Tuple[str, str]:
    jid = (job_id or "").strip()
    if jid:
        bgjobs.sync_slurm_meta(jid)
    txt = bgjobs.format_meta_report(jid) if jid else "(no job id)"
    if jid:
        m = bgjobs.read_meta(jid)
        if m and m.get("status") == "done" and m.get("benchmark_session_dir"):
            txt += (
                "\n\n**Benchmark session dir** (paste into Test → Benchmark session dir):\n`"
                + str(m["benchmark_session_dir"])
                + "`"
            )
    html = _format_benchmark_wandb_html(jid)
    return txt, html


def poll_benchmark_slurm_panel(job_id: str, enabled: bool) -> Tuple[Any, Any]:
    if not enabled or not (job_id or "").strip():
        return gr.skip(), gr.skip()
    t, h = refresh_benchmark_slurm_panel(job_id)
    return t, h


def estimate_benchmark_slurm_resources(
    adata: ad.AnnData,
    *,
    max_cells: float,
    skip_sources: List[str],
    classifier_kind: str,
) -> Dict[str, Any]:
    skip_set = set(skip_sources or [])
    total_main = sum(nb for _, nb, _meta in _adata_buffer_parts(adata))
    source_bytes = 0
    for spec in _list_plot_source_options(adata):
        if spec in skip_set:
            continue
        try:
            source_bytes = max(source_bytes, _matrix_nbytes(_matrix_or_obsm_from_spec(adata, spec)))
        except Exception:
            continue
    max_cells_i = int(max_cells) if max_cells is not None else 0
    cell_fraction = 1.0
    if max_cells_i > 0 and int(adata.n_obs) > 0:
        cell_fraction = min(1.0, max_cells_i / float(int(adata.n_obs)))
    working_bytes = int(source_bytes * max(0.2, cell_fraction))
    model_margin_gib = 6.0 if str(classifier_kind or "").strip().lower() == "mlp" else 3.0
    peak = total_main * 1.5 + working_bytes * 1.5 + int(model_margin_gib * (1 << 30))
    mem_gib = max(8, int(np.ceil(peak * 1.15 / float(1 << 30))))
    cpus = 8 if str(classifier_kind or "").strip().lower() == "mlp" else 4
    hours = 3.0 + int(adata.n_obs) / 200_000.0
    if str(classifier_kind or "").strip().lower() == "mlp":
        hours += 2.0
    hours = min(48.0, max(2.0, hours))
    total_sec = max(1, int(np.ceil(hours * 3600.0)))
    hh, rem = divmod(total_sec, 3600)
    mm, ss = divmod(rem, 60)
    return {
        "cpus": int(cpus),
        "mem_gib": int(mem_gib),
        "time": f"{hh:02d}:{mm:02d}:{ss:02d}",
    }


def submit_benchmark_slurm_job(
    train_state,
    train_sess_bundle,
    target_cols: List[str],
    split_mode: str,
    stratify_col: str,
    test_fraction: float,
    random_seed: float,
    classifier_kind: str,
    mlp_hidden: str,
    mlp_max_iter: float,
    lr_c: float,
    lr_max_iter: float,
    max_cells: float,
    skip_sources: List[str],
    slurm_partition: str,
    slurm_cpus: float,
    slurm_mem_gb: float,
    slurm_time: str,
    slurm_gres: str,
    slurm_extra_sbatch: str,
    slurm_bash_prologue: str,
    scfms_repo_root_tb: str,
    wandb_project: str,
    wandb_entity: str,
    wandb_run_name: str,
    test_h5ad_cluster_path: str,
    train_session_dir: str,
):
    if train_state is None:
        return (
            "Load train data first (same as interactive benchmark).",
            gr.update(value=""),
            "(no job)",
            "<p>Load data, then submit.</p>",
        )
    try:
        adata = train_state
        rr = (scfms_repo_root_tb or "").strip()
        repo = Path(rr).resolve() if rr else None
        rec = estimate_benchmark_slurm_resources(
            adata,
            max_cells=max_cells,
            skip_sources=skip_sources,
            classifier_kind=classifier_kind,
        )
        cpus = int(rec["cpus"])
        if slurm_cpus is not None:
            try:
                user_cpus = int(slurm_cpus)
            except (TypeError, ValueError):
                user_cpus = 0
            if user_cpus > 0:
                cpus = max(user_cpus, int(rec["cpus"]))
        mem_gib_req = int(rec["mem_gib"])
        if slurm_mem_gb is not None:
            try:
                user_mem = int(np.ceil(float(slurm_mem_gb)))
            except (TypeError, ValueError):
                user_mem = 0
            if user_mem > 0:
                mem_gib_req = max(user_mem, int(rec["mem_gib"]))
        mem = f"{mem_gib_req}G"
        time_raw = str(slurm_time or "").strip()
        time_limit = str(rec["time"]) if not time_raw or time_raw.lower() == "auto" else time_raw
        bench_params: Dict[str, Any] = {
            "target_cols": [str(c) for c in (target_cols or []) if str(c).strip()],
            "split_mode": str(split_mode or "random"),
            "stratify_col": str(stratify_col or "(none)"),
            "test_fraction": float(test_fraction),
            "random_seed": int(random_seed) if random_seed is not None else 0,
            "classifier_kind": str(classifier_kind or "logistic_regression"),
            "mlp_hidden": str(mlp_hidden or "128,64"),
            "mlp_max_iter": float(mlp_max_iter or 200),
            "lr_c": float(lr_c or 1.0),
            "lr_max_iter": float(lr_max_iter or 2000),
            "max_cells": float(max_cells or 0),
            "skip_sources": list(skip_sources or []),
            "wandb_project": (wandb_project or "scfms-benchmark").strip() or "scfms-benchmark",
            "wandb_entity": (wandb_entity or "").strip(),
            "wandb_run_name": (wandb_run_name or "").strip(),
            "test_h5ad_path": (test_h5ad_cluster_path or "").strip(),
        }
        if not bench_params["target_cols"]:
            return (
                "Select at least one target obs column before submitting to Slurm.",
                gr.update(value=""),
                "(no job)",
                "<p>Choose targets.</p>",
            )
        gres_s = str(slurm_gres or "gpu:1").strip() or "gpu:1"
        jid = bgjobs.start_benchmark_slurm_job(
            adata,
            bench_params,
            repo,
            partition=effective_slurm_partition(slurm_partition),
            cpus=cpus,
            mem=mem,
            time_limit=time_limit,
            gres=gres_s,
            bash_prologue=str(slurm_bash_prologue or ""),
            extra_sbatch=str(slurm_extra_sbatch or ""),
        )
        msg = (
            f"**Slurm GPU benchmark queued** — UI job id `{jid}`.\n"
            f"Staging: `{bgjobs.jobs_root() / jid}`\n"
            f"Slurm script: `slurm_gpu_benchmark.sh`\n\n"
            "Set **WANDB_API_KEY** (or `wandb login`) on the compute node so runs upload.\n"
            "Poll status below; when **done**, copy the **benchmark session dir** into the Test tab.\n\n"
            f"Resolved request: **-c {cpus}** · **--mem={mem}** · **-t {time_limit}** · **--gres={gres_s}**\n\n"
            "_Sklearn fits on CPU; the GPU allocation reserves an accelerator node for your site’s policy "
            "and for mixed workflows (e.g. CUDA-enabled stacks)._"
        )
        sess = (train_session_dir or "").strip()
        if sess:
            _try_append_compute_ram_log(
                sess,
                "benchmark_slurm_submitted",
                _compute_ram_plan_lines(
                    adata,
                    extras=[
                        f"ui_job_id: {jid}",
                        f"slurm_mem: {mem}",
                        f"slurm_cpus: {cpus}",
                        f"slurm_time: {time_limit}",
                        f"slurm_gres: {gres_s}",
                    ],
                ),
            )
        t, h = refresh_benchmark_slurm_panel(jid)
        return msg, gr.update(value=jid), t, h
    except Exception as e:
        return f"Error: {e}", gr.update(value=""), str(e), f"<p>Error: {e}</p>"


def load_train_adata_only(file_obj, server_path, transpose, train_sess_bundle):
    try:
        adata = _load_adata_from_inputs(file_obj, server_path, transpose)
        src_key = _session_src_key(file_obj, server_path)
        session_dir, bundle = _ensure_session(
            _norm_sess_bundle(train_sess_bundle),
            src_key,
            adata,
            tag="train",
        )
        oc = _obs_choices(adata)
        specs = _list_plot_source_options(adata)
        msg = f"Loaded train: n_obs={adata.n_obs}, n_vars={adata.n_vars}\n`{session_dir}`"
        _try_append_compute_ram_log(
            session_dir,
            "benchmark_train_adata_loaded",
            _compute_ram_plan_lines(
                adata,
                extras=[f"session_tag: train", f"source_key: {src_key}"],
            ),
        )
        return (
            adata,
            msg,
            bundle,
            gr.update(value=session_dir),
            gr.update(choices=oc, value=[]),
            gr.update(choices=["(none)"] + oc, value="(none)"),
            gr.update(choices=specs, value=[]),
        )
    except Exception as e:
        return None, str(e), train_sess_bundle, gr.skip(), gr.skip(), gr.skip(), gr.skip()


def load_test_adata_only(file_obj, server_path, transpose, test_sess_bundle):
    try:
        adata = _load_adata_from_inputs(file_obj, server_path, transpose)
        src_key = _session_src_key(file_obj, server_path)
        session_dir, bundle = _ensure_session(
            _norm_sess_bundle(test_sess_bundle),
            src_key,
            adata,
            tag="test",
        )
        msg = f"Loaded test: n_obs={adata.n_obs}\n`{session_dir}`"
        _try_append_compute_ram_log(
            session_dir,
            "benchmark_test_adata_loaded",
            _compute_ram_plan_lines(
                adata,
                extras=[f"session_tag: test", f"source_key: {src_key}"],
            ),
        )
        return (
            adata,
            msg,
            bundle,
            gr.update(value=session_dir),
        )
    except Exception as e:
        return None, str(e), test_sess_bundle, gr.skip()


def build_ui():
    with gr.Blocks(title="scFMs: Embedding benchmark") as demo:
        train_state = gr.State(None)
        train_sess = gr.State(None)
        test_state = gr.State(None)
        test_sess = gr.State(None)

        gr.Markdown("# Embedding benchmark — LR / MLP on all matrix & obsm sources")
        with gr.Accordion("Where is this running? (server vs laptop)", open=False):
            gr.Markdown(runtime_info_markdown())
        gr.Markdown(
            "**Train:** load an `.h5ad` (or CSV/TSV), choose categorical **obs** columns to predict, "
            "configure a **holdout split** (random or stratified by another obs column), pick **logistic regression** or **MLP**. "
            "The app trains one model per **embedding source** (`X`, `raw.X`, layers, every `obsm`) × target, saves **`benchmark_manifest.json`** "
            "and **`models/*.joblib`** under a timestamped session (same root as preprocess: **`SCFMS_SESSION_DIR`**).\n\n"
            "**Test:** load a separate AnnData, paste the **train benchmark session directory** that contains the manifest, and run **Evaluate**. "
            "Expression matrices are **gene-aligned** to the training `var_names` (missing genes filled with zeros). **obsm** rows must match the "
            "training latent dimension.\n\n"
            f"**MLP** on very wide sparse inputs uses a **`TruncatedSVD`** front-end (cap **{MLP_FEATURE_CAP}** features; override with **`SCFMS_BENCH_MLP_MAX_FEATURES`**). "
            f"If **PyTorch** reports **CUDA** or **MPS**, the MLP trains on that accelerator automatically; set **`SCFMS_BENCH_USE_GPU_MLP=0`** to use sklearn's CPU MLP instead.\n\n"
            f"Each session directory also appends **`{sess_res.COMPUTE_RAM_LOG}`** with **GiB** RAM estimates whenever you load data or finish train/eval — use it with the preprocess app logs to size cluster nodes."
        )

        with gr.Accordion("Train — load data & run benchmark", open=True):
            train_session_disp = gr.Textbox(
                label="Train session folder",
                interactive=False,
            )
            with gr.Row():
                tr_server = gr.Textbox(
                    label="Server path (train)",
                    placeholder="/path/train.h5ad",
                    scale=3,
                )
                tr_file = gr.File(
                    label="Or upload train",
                    scale=2,
                )
            tr_transpose = gr.Checkbox(label="Transpose CSV/TSV", value=False)
            with gr.Row():
                btn_load_tr = gr.Button("Load train only (refresh column lists)")
                btn_run_tr = gr.Button("Run full benchmark train", variant="primary")

            tr_status = gr.Textbox(label="Train status", interactive=False, lines=6)
            target_cols = gr.CheckboxGroup(
                choices=[],
                label="Target obs columns (categorical labels)",
            )
            split_mode = gr.Radio(
                ["random", "stratify_obs"],
                value="random",
                label="Holdout split",
            )
            stratify_col = gr.Dropdown(
                choices=["(none)"],
                value="(none)",
                label="Stratify split by obs (when mode = stratify_obs)",
            )
            with gr.Row():
                test_fraction = gr.Slider(
                    0.05, 0.5, value=0.2, step=0.05, label="Validation fraction"
                )
                random_seed = gr.Number(value=0, precision=0, label="Random seed")
            classifier_kind = gr.Radio(
                ["logistic_regression", "mlp"],
                value="logistic_regression",
                label="Classifier",
            )
            with gr.Row():
                mlp_hidden = gr.Textbox(
                    label="MLP hidden sizes (comma-separated)",
                    value="128,64",
                )
                mlp_max_iter = gr.Number(value=200, precision=0, label="MLP max_iter", minimum=10)
            with gr.Row():
                lr_c = gr.Number(value=1.0, label="LR C", minimum=0.0001)
                lr_max_iter = gr.Number(value=2000, precision=0, label="LR max_iter", minimum=100)
            max_cells = gr.Number(
                value=0,
                precision=0,
                label="Max train cells (0 = all; subsample rows before split)",
                minimum=0,
            )
            skip_sources = gr.CheckboxGroup(
                choices=[],
                label="Skip embedding sources (e.g. drop huge raw.X)",
            )

            with gr.Accordion(
                "Slurm GPU + Weights & Biases — queue train / optional test eval",
                open=False,
            ):
                gr.Markdown(
                    "Submits **`sbatch`** on a **GPU partition** with **`#SBATCH --gres=…`** (GPU count and type depend on your site; "
                    "examples: `gpu:1`, `gpu:a100:1`). **Node RAM (GiB)** maps to **`#SBATCH --mem=NG`**. "
                    "Training uses **sklearn** on **CPU** inside that allocation unless you swap estimators later.\n\n"
                    "**Load train** must be run first so the same AnnData is staged as `input.h5ad`. "
                    "Optional **test .h5ad path** must be visible on the **compute** node. "
                    "For W&B uploads on the node, set **`WANDB_API_KEY`** or run **`wandb login`** once. "
                    "If **CPUs / RAM / time** stay at **auto** (or below the app's minimum estimate), the submit path raises them to a dataset-aware floor."
                )
                with gr.Row():
                    slurm_partition = gr.Textbox(
                        value=default_slurm_partition(),
                        label="Slurm partition (-p)",
                    )
                    slurm_cpus = gr.Number(
                        value=0,
                        precision=0,
                        label="CPUs per task (-c, 0 = auto floor)",
                        minimum=0,
                    )
                    slurm_mem_gb = gr.Number(
                        value=0,
                        precision=0,
                        label="Node RAM (GiB, 0 = auto floor)",
                        minimum=0,
                    )
                with gr.Row():
                    slurm_time = gr.Textbox(value="auto", label="Time limit (-t, auto = estimate)")
                    slurm_gres = gr.Textbox(
                        value="gpu:1",
                        label="GPU request (--gres=…)",
                        info="Cluster-specific; encodes device type/count at many sites.",
                    )
                slurm_extra_sbatch = gr.Textbox(
                    label="Extra #SBATCH lines (optional, one per line)",
                    placeholder="#SBATCH --constraint=a100\n#SBATCH --gres=gpu:2",
                    lines=2,
                )
                slurm_bash_prologue = gr.Textbox(
                    label="Shell prologue after cd repo (modules, conda, etc.)",
                    lines=2,
                )
                scfms_repo_root_tb = gr.Textbox(
                    label="scFMs repo root (empty = auto)",
                    placeholder="/path/to/scFMs",
                )
                with gr.Row():
                    wandb_project = gr.Textbox(value="scfms-benchmark", label="wandb project")
                    wandb_entity = gr.Textbox(label="wandb entity (optional)")
                    wandb_run_name = gr.Textbox(label="wandb run name (optional)")
                test_h5ad_cluster_path = gr.Textbox(
                    label="Optional test .h5ad on cluster (eval + test/* metrics in same W&B run)",
                    placeholder="/n/.../held_out.h5ad",
                )
                btn_slurm = gr.Button(
                    "Queue benchmark on Slurm (GPU allocation)",
                    variant="secondary",
                )
                gr.Markdown(
                    "**Job tracking** uses the same on-disk store as the preprocess app (`SCFMS_JOB_DIR`). "
                    "After submit, the **Job ID** is filled automatically; use **Refresh** or auto-poll."
                )
                with gr.Row():
                    bench_job_id = gr.Textbox(
                        label="Benchmark / Slurm UI job ID",
                        placeholder="filled when you queue a run",
                    )
                    btn_bench_refresh = gr.Button("Refresh status + W&B panel")
                bench_job_poll = gr.Checkbox(
                    label="Auto-poll job status every 2s (keep this browser session open)",
                    value=False,
                )
                bench_slurm_log = gr.Textbox(
                    label="Job status (from disk)",
                    lines=14,
                    interactive=False,
                )
                bench_wandb_html = gr.HTML(label="Weights & Biases (link + embedded run page)")
                bench_timer = gr.Timer(2.0)

            tr_table = gr.Dataframe(label="Validation metrics (wide)", wrap=True)
            tr_plot_hm = gr.Plot(label="Accuracy heatmap (val)")
            tr_plot_bar = gr.Plot(label="Mean accuracy by source (val)")

            btn_load_tr.click(
                load_train_adata_only,
                [tr_file, tr_server, tr_transpose, train_sess],
                [
                    train_state,
                    tr_status,
                    train_sess,
                    train_session_disp,
                    target_cols,
                    stratify_col,
                    skip_sources,
                ],
            )
            btn_run_tr.click(
                run_benchmark_train,
                [
                    tr_file,
                    tr_server,
                    tr_transpose,
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
                    max_cells,
                    skip_sources,
                    train_sess,
                ],
                [tr_table, tr_status, tr_plot_hm, tr_plot_bar, train_sess, train_session_disp],
            )
            btn_slurm.click(
                submit_benchmark_slurm_job,
                [
                    train_state,
                    train_sess,
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
                    max_cells,
                    skip_sources,
                    slurm_partition,
                    slurm_cpus,
                    slurm_mem_gb,
                    slurm_time,
                    slurm_gres,
                    slurm_extra_sbatch,
                    slurm_bash_prologue,
                    scfms_repo_root_tb,
                    wandb_project,
                    wandb_entity,
                    wandb_run_name,
                    test_h5ad_cluster_path,
                    train_session_disp,
                ],
                [tr_status, bench_job_id, bench_slurm_log, bench_wandb_html],
            )
            btn_bench_refresh.click(
                refresh_benchmark_slurm_panel,
                [bench_job_id],
                [bench_slurm_log, bench_wandb_html],
            )
            bench_timer.tick(
                poll_benchmark_slurm_panel,
                [bench_job_id, bench_job_poll],
                [bench_slurm_log, bench_wandb_html],
            )

        with gr.Accordion("Test — load data & evaluate saved models", open=True):
            test_session_disp = gr.Textbox(
                label="Test session folder (plots saved here)",
                interactive=False,
            )
            benchmark_dir = gr.Textbox(
                label="Benchmark session dir (contains benchmark_manifest.json)",
                placeholder="paste train session path",
            )
            with gr.Row():
                te_server = gr.Textbox(
                    label="Server path (test)",
                    placeholder="/path/test.h5ad",
                    scale=3,
                )
                te_file = gr.File(label="Or upload test", scale=2)
            te_transpose = gr.Checkbox(label="Transpose CSV/TSV", value=False)
            with gr.Row():
                btn_load_te = gr.Button("Load test only")
                btn_run_te = gr.Button("Evaluate on test", variant="primary")

            te_status = gr.Textbox(label="Test status", interactive=False, lines=6)
            te_table = gr.Dataframe(label="Test metrics per source × target", wrap=True)
            te_plot_hm = gr.Plot(label="Accuracy heatmap (test)")
            te_plot_bar = gr.Plot(label="Mean accuracy by source (test)")

            btn_load_te.click(
                load_test_adata_only,
                [te_file, te_server, te_transpose, test_sess],
                [test_state, te_status, test_sess, test_session_disp],
            )
            btn_run_te.click(
                run_benchmark_eval,
                [te_file, te_server, te_transpose, benchmark_dir, test_sess],
                [te_table, te_status, te_plot_hm, te_plot_bar, test_sess, test_session_disp],
            )

    return demo


if __name__ == "__main__":
    ui = build_ui()
    launch_gradio_demo(
        ui,
        default_port=int(os.environ.get("PORT", "7862")),
        app_label="scFMs embedding benchmark",
    )
