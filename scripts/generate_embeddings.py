#!/usr/bin/env python3
import argparse
import contextlib
import inspect
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse as sp
from tqdm import tqdm


def load_adata(path: str, transpose: bool = False) -> ad.AnnData:
    if path.endswith(".h5ad"):
        return ad.read_h5ad(path)
    elif path.endswith(".csv") or path.endswith(".tsv"):
        sep = "," if path.endswith(".csv") else "\t"
        df = pd.read_csv(path, sep=sep, index_col=0)
        if transpose:
            df = df.T
        return ad.AnnData(df)
    else:
        raise ValueError("Unsupported input format. Use .h5ad, .csv, or .tsv")


def _resolve_device(device: Optional[str] = None) -> str:
    if device:
        return str(device)
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _sequence_token_cap(*, model_cap: Optional[int], default_cap: int = 2048) -> int:
    env_raw = os.environ.get("SCFMS_SCFM_MAX_TOKENS", "").strip()
    if env_raw:
        try:
            env_cap = max(1, int(env_raw))
        except ValueError:
            env_cap = default_cap
        if model_cap is None:
            return env_cap
        return min(env_cap, model_cap)
    if model_cap is None:
        return default_cap
    return max(1, int(model_cap))


def _model_seq_cap(model: Any, tokenizer: Optional[Any] = None) -> Optional[int]:
    caps = []
    cfg = getattr(model, "config", None)
    for attr in ("max_position_embeddings", "n_positions", "seq_length", "max_seq_len"):
        val = getattr(cfg, attr, None) if cfg is not None else None
        if isinstance(val, int) and 0 < val < 1_000_000:
            caps.append(int(val))
    tok_cap = getattr(tokenizer, "model_max_length", None) if tokenizer is not None else None
    if isinstance(tok_cap, int) and 0 < tok_cap < 1_000_000:
        caps.append(int(tok_cap))
    if not caps:
        return None
    return min(caps)


def _iter_ranked_nonzero_gene_indices(
    X: Any,
    n_obs: int,
    *,
    max_cells: Optional[int] = None,
) -> Iterable[np.ndarray]:
    n_use = n_obs if max_cells is None else min(n_obs, int(max_cells))
    if sp.issparse(X):
        X = X.tocsr()
        for i in range(n_use):
            start = int(X.indptr[i])
            end = int(X.indptr[i + 1])
            idx = X.indices[start:end]
            if idx.size == 0:
                yield np.empty(0, dtype=np.int64)
                continue
            vals = X.data[start:end]
            order = np.argsort(-vals, kind="stable")
            yield np.asarray(idx[order], dtype=np.int64)
        return

    arr = np.asarray(X)
    for i in range(n_use):
        row = np.asarray(arr[i]).ravel()
        nz = np.flatnonzero(row > 0)
        if nz.size == 0:
            yield np.empty(0, dtype=np.int64)
            continue
        order = np.argsort(-row[nz], kind="stable")
        yield np.asarray(nz[order], dtype=np.int64)


def _extract_hidden_matrix(model_out: Any) -> np.ndarray:
    if hasattr(model_out, "last_hidden_state"):
        hidden = model_out.last_hidden_state
    elif isinstance(model_out, tuple) and model_out:
        hidden = model_out[0]
    else:
        hidden = model_out
    if hidden is None:
        raise ValueError("Model output did not include hidden states.")
    return hidden


def _truncate_tokens(tokens: list[int], max_tokens: int, fallback_id: int) -> list[int]:
    if not tokens:
        return [int(fallback_id)]
    if len(tokens) > max_tokens:
        return tokens[:max_tokens]
    return tokens


def _filter_model_kwargs(cls: Any, cfg: Dict[str, Any]) -> Dict[str, Any]:
    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return dict(cfg)
    allowed = {
        name
        for name, param in sig.parameters.items()
        if name != "self" and param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    return {k: v for k, v in cfg.items() if k in allowed}


def _unwrap_state_dict(state: Any) -> Dict[str, Any]:
    if isinstance(state, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            sub = state.get(key)
            if isinstance(sub, dict):
                return sub
    if not isinstance(state, dict):
        raise TypeError(f"Unexpected checkpoint payload type: {type(state).__name__}")
    return state


def _parse_nonnegative_int_env(name: str) -> Optional[int]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        return max(0, int(raw))
    except ValueError:
        return None


def _detect_available_cpu_budget() -> tuple[int, str]:
    override = _parse_nonnegative_int_env("SCFMS_SCVI_NUM_WORKERS")
    if override is not None:
        return override, "SCFMS_SCVI_NUM_WORKERS"

    for env_name in ("SLURM_CPUS_PER_TASK", "OMP_NUM_THREADS"):
        val = _parse_nonnegative_int_env(env_name)
        if val is not None and val > 0:
            return val, env_name

    if hasattr(os, "sched_getaffinity"):
        try:
            n_aff = len(os.sched_getaffinity(0))
            if n_aff > 0:
                return n_aff, "sched_getaffinity"
        except OSError:
            pass

    cpu_total = os.cpu_count() or 1
    return max(1, int(cpu_total)), "os.cpu_count"


def _auto_scvi_compute_threads(visible_cpu_budget: int, num_workers: int) -> tuple[int, str]:
    override = _parse_nonnegative_int_env("SCFMS_SCVI_COMPUTE_THREADS")
    if override is not None:
        return max(1, override), "SCFMS_SCVI_COMPUTE_THREADS"
    remaining = max(1, int(visible_cpu_budget) - max(0, int(num_workers)))
    return min(4, remaining), "auto_visible_minus_workers"


def _resolve_scvi_trainer_hardware(device: Optional[str] = None) -> tuple[str, int, str]:
    import torch

    resolved = _resolve_device(device)
    if resolved.startswith("cuda") and torch.cuda.is_available():
        return "gpu", 1, resolved
    return "cpu", 1, "cpu"


def _should_mask_lightning_slurm_hint() -> bool:
    if shutil.which("srun") is None:
        return False
    if "SLURM_NTASKS" not in os.environ:
        return False
    if os.environ.get("SLURM_JOB_NAME") in ("bash", "interactive"):
        return False
    return "SLURM_STEP_ID" not in os.environ


@contextlib.contextmanager
def _temporary_env(updates: Dict[str, Optional[str]]):
    old: Dict[str, Optional[str]] = {}
    try:
        for key, value in updates.items():
            old[key] = os.environ.get(key)
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def embed_geneformer(
    adata: ad.AnnData,
    device: Optional[str] = None,
    max_cells: Optional[int] = None,
    pretrained_name_or_path: Optional[str] = None,
) -> np.ndarray:
    from transformers import AutoModel, AutoTokenizer
    import torch

    model_name = (pretrained_name_or_path or "").strip() or os.environ.get(
        "GENEFORMER_MODEL", "ctheodoris/Geneformer"
    )
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    device = _resolve_device(device)
    model.to(device)

    genes = list(adata.var_names.astype(str))
    gene_to_id = {g: tok.convert_tokens_to_ids(g) for g in genes}
    seq_cap = _sequence_token_cap(
        model_cap=_model_seq_cap(model, tok),
        default_cap=2048,
    )
    fallback_id = getattr(tok, "unk_token_id", None)
    if fallback_id is None:
        fallback_id = getattr(tok, "pad_token_id", 0)

    embs = []
    with torch.no_grad():
        rows = _iter_ranked_nonzero_gene_indices(adata.X, adata.n_obs, max_cells=max_cells)
        total = adata.n_obs if max_cells is None else min(adata.n_obs, int(max_cells))
        for order in tqdm(rows, total=total, desc="Geneformer embeddings"):
            tokens = [gene_to_id.get(genes[j], fallback_id) for j in order]
            tokens = _truncate_tokens(tokens, seq_cap, int(fallback_id))
            input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
            out = model(input_ids=input_ids)
            hidden = _extract_hidden_matrix(out).squeeze(0)
            embs.append(hidden.mean(dim=0).cpu().numpy())
    return np.stack(embs)


def _repo_root_for_embeddings() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_transcriptformer_checkpoint() -> str:
    p = _repo_root_for_embeddings() / "models" / "transcriptformer" / "tf_sapiens"
    return str(p.resolve()) if p.is_dir() else ""


def embed_transcriptformer(
    adata: ad.AnnData,
    device: Optional[str] = None,
    max_cells: Optional[int] = None,
    pretrained_name_or_path: Optional[str] = None,
) -> np.ndarray:
    """
    Lightweight path: Hugging Face ``transformers`` AutoModel cell embedding.

    Official CZI checkpoints from ``download-weights`` (S3) are **PyTorch Lightning**
    bundles (``model_weights.pt``), not HF repos — use the ``transcriptformer``
    package / CLI for those. See ``models/README.md``.
    """
    from transformers import AutoModel, AutoTokenizer
    import torch

    explicit = (pretrained_name_or_path or "").strip()
    if explicit:
        model_name = explicit
    else:
        model_name = (os.environ.get("TRANSCRIPTFORMER_MODEL") or "").strip()
    if not model_name:
        model_name = _default_transcriptformer_checkpoint()
    if not model_name:
        raise FileNotFoundError(
            "TranscriptFormer: set TRANSCRIPTFORMER_MODEL to a Hugging Face model id or "
            "local HF-format checkpoint, or run `uv run python main.py download-weights -- "
            "--models transcriptformer` and use the official `transcriptformer` CLI for CZI weights "
            "(see models/README.md)."
        )

    ckpt = Path(model_name)
    if ckpt.is_dir() and (ckpt / "model_weights.pt").is_file():
        raise RuntimeError(
            f"Path {ckpt} is an official CZI TranscriptFormer checkpoint (model_weights.pt). "
            "This helper only supports Hugging Face–style checkpoints. "
            "Install `transcriptformer` from PyPI or GitHub and run inference with Hydra, "
            "or see models/README.md."
        )
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    device = _resolve_device(device)
    model.to(device)

    genes = list(adata.var_names.astype(str))
    try:
        unk_id = tok.unk_token_id
    except Exception:
        unk_id = None
    gene_to_id = {}
    for g in genes:
        try:
            gene_to_id[g] = tok.convert_tokens_to_ids(g)
        except Exception:
            gene_to_id[g] = unk_id
    seq_cap = _sequence_token_cap(
        model_cap=_model_seq_cap(model, tok),
        default_cap=2048,
    )
    fallback_id = getattr(tok, "pad_token_id", None)
    if fallback_id is None:
        fallback_id = unk_id if unk_id is not None else 0

    embs = []
    with torch.no_grad():
        rows = _iter_ranked_nonzero_gene_indices(adata.X, adata.n_obs, max_cells=max_cells)
        total = adata.n_obs if max_cells is None else min(adata.n_obs, int(max_cells))
        for order in tqdm(rows, total=total, desc="Transcriptformer embeddings"):
            tokens = [gene_to_id.get(genes[j], fallback_id) for j in order]
            tokens = _truncate_tokens(tokens, seq_cap, int(fallback_id))
            input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
            out = model(input_ids=input_ids)
            hidden = _extract_hidden_matrix(out).squeeze(0)
            embs.append(hidden.mean(dim=0).cpu().numpy())
    return np.stack(embs)


def embed_scgpt(
    adata: ad.AnnData,
    ckpt_dir: Optional[str] = None,
    device: Optional[str] = None,
    max_cells: Optional[int] = None,
) -> np.ndarray:
    import json
    import torch

    try:
        from scgpt.model import TransformerModel
        from scgpt.tokenizer.gene_tokenizer import GeneTokenizer
    except Exception as e:
        raise ImportError(
            "scgpt package is required for scGPT embeddings. Install scgpt (may require CUDA/flash-attn) or use Geneformer/scVI."
        ) from e

    if ckpt_dir is None:
        ckpt_dir = os.environ.get("SCGPT_CKPT_DIR")
    if ckpt_dir is None or not os.path.exists(ckpt_dir):
        raise FileNotFoundError(
            "scGPT checkpoint directory not found. Set --ckpt or SCGPT_CKPT_DIR."
        )

    vocab_path = os.path.join(ckpt_dir, "vocab.json")
    tok = GeneTokenizer(vocab_path)

    args_json = os.path.join(ckpt_dir, "args.json")
    if os.path.exists(args_json):
        with open(args_json) as f:
            cfg = json.load(f)
    else:
        cfg = torch.load(os.path.join(ckpt_dir, "model_args.pt"), map_location="cpu")
    if not isinstance(cfg, dict):
        raise TypeError("scGPT args payload must be a dict.")
    model = TransformerModel(**_filter_model_kwargs(TransformerModel, cfg))
    state_path = None
    for cand in [
        os.path.join(ckpt_dir, "pytorch_model.bin"),
        os.path.join(ckpt_dir, "best_model.pt"),
    ]:
        if os.path.exists(cand):
            state_path = cand
            break
    if state_path is None:
        raise FileNotFoundError(
            "scGPT checkpoint weights not found (expected pytorch_model.bin or best_model.pt)"
        )
    state = torch.load(state_path, map_location="cpu")
    model.load_state_dict(_unwrap_state_dict(state), strict=False)
    model.eval()
    device = _resolve_device(device)
    model.to(device)

    genes = list(adata.var_names.astype(str))
    gene_tokens = []
    for g in genes:
        try:
            gene_tokens.append(tok.encode(g))
        except Exception:
            gene_tokens.append(None)
    seq_cap = _sequence_token_cap(
        model_cap=min(
            x
            for x in (
                int(cfg.get("max_seq_len", 0) or 0),
                int(cfg.get("max_len", 0) or 0),
                int(cfg.get("max_length", 0) or 0),
            )
            if x > 0
        )
        if any(int(cfg.get(k, 0) or 0) > 0 for k in ("max_seq_len", "max_len", "max_length"))
        else None,
        default_cap=1200,
    )
    pad_id = getattr(tok, "pad_token_id", None)
    if pad_id is None:
        pad_id = 0
    embs = []
    with torch.no_grad():
        rows = _iter_ranked_nonzero_gene_indices(adata.X, adata.n_obs, max_cells=max_cells)
        total = adata.n_obs if max_cells is None else min(adata.n_obs, int(max_cells))
        for order in tqdm(rows, total=total, desc="scGPT embeddings"):
            tokens = [gene_tokens[j] for j in order if gene_tokens[j] is not None]
            tokens = _truncate_tokens(tokens, seq_cap, int(pad_id))
            input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
            out = model(input_ids)
            hidden = _extract_hidden_matrix(out).squeeze(0)
            embs.append(hidden.mean(dim=0).cpu().numpy())
    return np.stack(embs)


def embed_scvi(
    adata: ad.AnnData, device: Optional[str] = None, n_latent: int = 64
) -> np.ndarray:
    import scvi
    import torch
    from scvi.model import SCVI

    adata = adata.copy()
    scvi.model.SCVI.setup_anndata(adata)
    m = SCVI(adata, n_latent=n_latent)
    train_kw: Dict[str, Any] = {}
    bs = os.environ.get("SCFMS_SCVI_BATCH_SIZE", "").strip()
    if bs:
        train_kw["batch_size"] = max(1, int(bs))
    me = os.environ.get("SCFMS_SCVI_MAX_EPOCHS", "").strip()
    if me:
        train_kw["max_epochs"] = max(1, int(me))
    if os.environ.get("SCFMS_SCVI_LOAD_SPARSE_TENSOR", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        train_kw["load_sparse_tensor"] = True
    visible_cpu_budget, worker_source = _detect_available_cpu_budget()
    if worker_source == "SCFMS_SCVI_NUM_WORKERS":
        num_workers = visible_cpu_budget
    elif visible_cpu_budget <= 1:
        num_workers = 0
    else:
        num_workers = max(1, visible_cpu_budget - 1)
    compute_threads, compute_thread_source = _auto_scvi_compute_threads(
        visible_cpu_budget, num_workers
    )
    accelerator, devices, resolved_device = _resolve_scvi_trainer_hardware(device)
    persistent_workers = bool(num_workers > 0)
    try:
        scvi.settings.dl_num_workers = int(num_workers)
        scvi.settings.dl_persistent_workers = persistent_workers
        scvi.settings.num_threads = int(compute_threads)
    except Exception:
        pass
    sig = inspect.signature(m.train)
    accepts_datasplitter_kwargs = (
        "datasplitter_kwargs" in sig.parameters
        or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    )
    if accepts_datasplitter_kwargs:
        ds_kwargs = dict(train_kw.get("datasplitter_kwargs") or {})
        ds_kwargs["num_workers"] = int(num_workers)
        ds_kwargs["persistent_workers"] = persistent_workers
        ds_kwargs["pin_memory"] = accelerator == "gpu"
        train_kw["datasplitter_kwargs"] = ds_kwargs
    train_kw["accelerator"] = accelerator
    train_kw["devices"] = devices
    train_kw["logger"] = False
    train_kw["enable_checkpointing"] = False
    train_kw["enable_model_summary"] = False
    print(
        f"scVI dataloader num_workers={num_workers} "
        f"(source={worker_source}; visible_cpu_budget={max(1, visible_cpu_budget)}; "
        f"datasplitter_kwargs_supported={accepts_datasplitter_kwargs})",
        flush=True,
    )
    print(
        f"scVI trainer accelerator={accelerator} devices={devices} resolved_device={resolved_device} "
        f"torch_threads={compute_threads} (source={compute_thread_source}) "
        f"persistent_workers={persistent_workers}",
        flush=True,
    )
    env_updates: Dict[str, Optional[str]] = {
        "OMP_NUM_THREADS": str(compute_threads),
        "MKL_NUM_THREADS": str(compute_threads),
        "OPENBLAS_NUM_THREADS": str(compute_threads),
        "NUMEXPR_NUM_THREADS": str(compute_threads),
    }
    if _should_mask_lightning_slurm_hint():
        env_updates["SLURM_JOB_NAME"] = "interactive"
        print(
            "scVI single-process run detected inside a Slurm allocation; "
            "masking Lightning's `srun` hint for this local session.",
            flush=True,
        )
    with _temporary_env(env_updates):
        try:
            torch.set_num_threads(int(compute_threads))
        except Exception:
            pass
        try:
            torch.set_num_interop_threads(1)
        except Exception:
            pass
        m.train(**train_kw)
    z = m.get_latent_representation()
    return z


def main():
    ap = argparse.ArgumentParser(
        description="Generate embeddings from single-cell foundation models"
    )
    ap.add_argument(
        "--model",
        required=True,
        choices=["geneformer", "transcriptformer", "scgpt", "scvi"],
        help="Which model to use",
    )
    ap.add_argument("--input", required=True, help="Path to .h5ad, .csv, or .tsv")
    ap.add_argument("--output", required=True, help="Path to output .npy or .csv")
    ap.add_argument("--ckpt", help="Checkpoint dir for scGPT", default=None)
    ap.add_argument(
        "--subset", type=int, default=None, help="Optionally embed only first N cells"
    )
    ap.add_argument(
        "--transpose", action="store_true", help="Transpose CSV/TSV input if needed"
    )
    ap.add_argument(
        "--skip-compat",
        action="store_true",
        help="Skip PDF compatibility validation (not recommended).",
    )
    args = ap.parse_args()

    adata = load_adata(args.input, transpose=args.transpose)

    if not args.skip_compat:
        app_dir = Path(__file__).resolve().parent.parent / "app"
        sys.path.insert(0, str(app_dir))
        try:
            import scfm_compatibility as sc  # noqa: E402

            strict = os.environ.get(
                "SCFMS_SC_FM_COMPAT_STRICT", "1"
            ).strip().lower() not in (
                "0",
                "false",
                "no",
            )
            _, _, pdf = sc.validate_run_and_pdf(
                args.model,
                adata,
                matrix_spec="X",
                scgpt_ckpt=args.ckpt,
                strict=strict,
            )
            print(f"Compatibility PDF: {pdf}", flush=True)
        except ValueError as e:
            print(str(e), file=sys.stderr)
            raise SystemExit(2) from e
        except Exception as e:
            print(f"Compatibility check error: {e}", file=sys.stderr)
            raise SystemExit(1) from e

    if args.model == "geneformer":
        E = embed_geneformer(adata, max_cells=args.subset)
    elif args.model == "transcriptformer":
        E = embed_transcriptformer(adata, max_cells=args.subset)
    elif args.model == "scgpt":
        E = embed_scgpt(adata, ckpt_dir=args.ckpt, max_cells=args.subset)
    else:
        E = embed_scvi(adata)

    if args.output.endswith(".npy"):
        np.save(args.output, E)
    else:
        pd.DataFrame(E, index=adata.obs_names).to_csv(args.output)


if __name__ == "__main__":
    main()
