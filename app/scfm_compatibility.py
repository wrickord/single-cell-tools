"""
scFM data format validation, training-data reference notes, and PDF compatibility reports.

Used before embedding to enforce minimal correctness and document whether user data plausibly
matches each model's pre-training regime (heuristic; not a guarantee).
"""

from __future__ import annotations

import json
import math
import os
import re
import textwrap
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import anndata as ad
import numpy as np
import scipy.sparse as sp

# ---------------------------------------------------------------------------
# Reference text: what each model was trained on (high-level, for the PDF).
# Sources: public model cards / primary papers; scVI path here trains de novo.
# ---------------------------------------------------------------------------

MODEL_TRAINING_CORPUS: Dict[str, str] = {
    "geneformer": (
        "Geneformer (Hugging Face checkpoints such as `ctheodoris/Geneformer`) is a transformer "
        "pretrained on very large human single-cell RNA-seq compendia (tens of millions of cells "
        "in aggregate; CellxGene / broad atlas-style data in the original Geneformer work). "
        "Input during pre-training is gene-token sequences where cell expression is represented "
        "as **rank-ordered genes** (non-zero signal), typically from **raw or raw-like UMI counts** "
        "so ranking reflects count magnitude. Gene identifiers are expected to match the model "
        "vocabulary (**often Ensembl gene IDs** for public checkpoints). "
        "Similarity to your data: good if genes are in vocabulary, values are non-negative suitable "
        "for per-cell ranking, and biology is human single-cell; organoids are often in-distribution "
        "but domain shift (species, modality, heavy batch effects) can still reduce quality."
    ),
    "scgpt": (
        "scGPT foundation checkpoints are pretrained on large-scale scRNA-seq collections (e.g. "
        "CELLxGENE-scale data; exact corpus depends on the **specific checkpoint** you load). "
        "This repo passes **rank-ordered expressed genes** with values > 0, using your "
        "`var_names` **must appear in that checkpoint's vocab.json**. "
        "Training typically used count-like matrices for ranking. "
        "Similarity to your data: best when gene symbols/IDs match the checkpoint vocabulary, "
        "matrix is non-negative, and tissue/domain is close to the checkpoint (whole-human vs "
        "organ-specific). Mismatch in gene naming (symbols vs Ensembl) is a common failure mode."
    ),
    "scvi": (
        "**Important for this repository:** `embed_scvi` calls `SCVI.setup_anndata` and **trains "
        "SCVI from scratch on your matrix** — it does **not** load a public 'foundation' weight "
        "snapshot. The latent you get is a **dataset-specific VAE** fit. "
        "scVI is statistically motivated for **count data** (UMI) with a latent representation; "
        "using log-normalized or scaled data is supported mechanically but can violate count "
        "assumptions. "
        "Similarity to your data: compatibility is mainly about **count structure** (non-negative, "
        "appropriate depth, enough cells/genes) rather than similarity to an external pre-training corpus."
    ),
}

FORMAT_REQUIREMENTS: Dict[str, List[str]] = {
    "geneformer": [
        "AnnData cells × genes; expression matrix used must be finite, non-negative.",
        "Per cell, genes with expression > 0 are ranked (descending); ties follow array order.",
        "Gene labels in `var_names` should match the tokenizer vocabulary (often Ensembl IDs).",
        "Very high fraction of unknown tokens ⇒ embedding quality may be poor — use counts & matching IDs.",
    ],
    "scgpt": [
        "Non-negative finite expression; same rank-tokenization pattern as Geneformer in this code.",
        "Each `var_name` must encode via the checkpoint's `GeneTokenizer` / `vocab.json`.",
        "Use the checkpoint matched to your domain (whole-human vs organ) when possible.",
    ],
    "scvi": [
        "Non-negative matrix strongly preferred (integer UMI counts ideal for likelihood).",
        "Training is on **your** data only in this pipeline; allow enough cells for stable training.",
        "Avoid highly sparse tiny panels unless you adjust `n_latent` / expectations accordingly.",
        "If the process prints **Killed** during Lightning training, the host likely ran out of RAM — "
        "use a larger node, subset cells, set **`SCFMS_SCVI_BATCH_SIZE`** (e.g. 64), and optionally "
        "**`SCFMS_SCVI_MAX_EPOCHS`** for shorter runs.",
    ],
}


@dataclass
class Finding:
    level: str  # "ok" | "warn" | "error"
    code: str
    message: str

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


def _fmt(x: float) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "n/a"
    if abs(x) >= 1e6 or (abs(x) > 0 and abs(x) < 1e-4):
        return f"{x:.3e}"
    return f"{x:.4g}"


def matrix_quick_stats(
    X: np.ndarray | sp.spmatrix, max_sample: int = 8000, rng: Optional[np.random.Generator] = None
) -> Dict[str, Any]:
    """Lightweight stats for validation / PDF (does not densify full matrix)."""
    rng = rng or np.random.default_rng(0)
    if sp.issparse(X):
        n_obs, n_var = X.shape
        row_idx = np.arange(n_obs)
        if n_obs > max_sample:
            row_idx = np.sort(rng.choice(n_obs, size=max_sample, replace=False))
        sub = X[row_idx]
        data = sub.data.astype(np.float64, copy=False) if sub.nnz else np.array([])
        nnz_per_row = np.diff(sub.indptr)
    else:
        Xd = np.asarray(X, dtype=np.float64)
        n_obs, n_var = Xd.shape
        if n_obs > max_sample:
            row_idx = np.sort(rng.choice(n_obs, size=max_sample, replace=False))
            Xd = Xd[row_idx]
        else:
            row_idx = np.arange(n_obs)
        data = Xd.ravel()
        nnz_per_row = (Xd > 0).sum(axis=1).astype(np.int64)

    finite = np.isfinite(data) if data.size else np.array([True])
    n_negative = int((data < 0).sum()) if data.size else 0
    n_nan = int((~np.isfinite(data)).sum()) if data.size else 0
    frac_zero_rows = float(np.mean(nnz_per_row == 0)) if len(nnz_per_row) else 0.0

    # Integer-ish (on expressed entries)
    expr = data[data > 0] if data.size else np.array([])
    frac_int = (
        float(np.mean(np.isclose(expr, np.round(expr), rtol=0, atol=1e-6)))
        if expr.size
        else 0.0
    )
    median_nz = float(np.median(nnz_per_row)) if len(nnz_per_row) else 0.0
    max_val = float(np.max(data)) if data.size else 0.0
    median_pos = float(np.median(expr)) if expr.size else 0.0

    log_like = max_val < 25 and median_pos < 8 and frac_int < 0.3 and expr.size > 0

    return {
        "n_obs_full": int(X.shape[0]),
        "n_var_full": int(X.shape[1]),
        "sampled_rows": int(len(row_idx)),
        "nnz_sample": int(data.size),
        "n_negative_sample": n_negative,
        "n_nonfinite_sample": n_nan,
        "max_sampled_value": max_val,
        "median_positive_sampled": median_pos,
        "median_genes_detected_sample": median_nz,
        "frac_zero_rows_sampled": frac_zero_rows,
        "fraction_positive_values_integerish": frac_int,
        "heuristic_looks_log_normalized": bool(log_like),
    }


def gene_name_style(genes: List[str]) -> Tuple[str, float]:
    """Rough heuristic: ensembl vs symbol-like."""
    g = [str(x) for x in genes[:5000]]
    en = sum(1 for x in g if re.match(r"^ENSG[0-9]{9,}(?:\.\d+)?$", x, re.I))
    sym = sum(1 for x in g if re.match(r"^[A-Z0-9][A-Z0-9\.\-]{1,15}$", x) and not x.startswith("ENS"))
    n = max(len(g), 1)
    if en / n > 0.45:
        return "ensembl_like", en / n
    if sym / n > 0.45:
        return "symbol_like", sym / n
    return "mixed_or_other", max(en, sym) / n


def geneformer_vocab_check(
    var_names: List[str], model_name: Optional[str] = None, max_genes: int = 6000
) -> Tuple[Optional[float], Optional[str]]:
    """Fraction of genes that map to a non-UNK token (sampled). None if tokenizer unavailable."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return None, "transformers not installed"
    name = model_name or os.environ.get("GENEFORMER_MODEL", "ctheodoris/Geneformer")
    try:
        tok = AutoTokenizer.from_pretrained(name, local_files_only=False)
    except Exception as e:
        return None, f"tokenizer load failed: {e}"
    unk = getattr(tok, "unk_token_id", None)
    if unk is None:
        return None, "no unk_token_id"
    genes = [str(v) for v in var_names[:max_genes]]
    if not genes:
        return None, "no genes"
    bad = 0
    for g in genes:
        tid = tok.convert_tokens_to_ids(g)
        if tid == unk:
            bad += 1
    return 1.0 - bad / len(genes), None


def scgpt_vocab_check(var_names: List[str], ckpt_dir: Optional[str]) -> Tuple[Optional[float], Optional[str]]:
    if not ckpt_dir or not Path(ckpt_dir).is_dir():
        return None, "no checkpoint dir"
    vp = Path(ckpt_dir) / "vocab.json"
    if not vp.is_file():
        return None, "vocab.json missing"
    try:
        with open(vp, encoding="utf-8") as f:
            vocab = json.load(f)
        # scgpt vocab is often gene -> index or special structure
        if isinstance(vocab, dict):
            keys = set(vocab.keys())
        else:
            return None, "unexpected vocab format"
    except Exception as e:
        return None, str(e)
    genes = [str(v) for v in var_names[:8000]]
    if not genes:
        return None, "no genes"
    hit = sum(1 for g in genes if g in keys)
    return hit / len(genes), None


def validate_for_embedding(
    model: str,
    ad_emb: ad.AnnData,
    *,
    scgpt_ckpt: Optional[str],
    strict: bool = True,
) -> Tuple[List[Finding], Dict[str, Any]]:
    """Run all checks; return findings + numeric summary. Errors should block embedding if strict."""
    findings: List[Finding] = []
    X = ad_emb.X
    stats = matrix_quick_stats(X)

    if ad_emb.n_vars == 0:
        findings.append(Finding("error", "no_genes", "AnnData has zero genes."))
    if ad_emb.n_obs == 0:
        findings.append(Finding("error", "no_cells", "AnnData has zero cells."))

    if stats["n_negative_sample"] > 0:
        findings.append(
            Finding(
                "error",
                "negative_values",
                f"Expression sample contains {stats['n_negative_sample']} negative entries; models expect non-negative ranks / counts.",
            )
        )
    if stats["n_nonfinite_sample"] > 0:
        findings.append(
            Finding(
                "error",
                "non_finite",
                "Matrix contains NaN or Inf in the sampled values.",
            )
        )

    if stats["frac_zero_rows_sampled"] > 0.9:
        findings.append(
            Finding(
                "warn",
                "sparse_rows",
                f"{_fmt(stats['frac_zero_rows_sampled']*100)}% of sampled rows have zero genes detected — check layer / filtering.",
            )
        )

    if model in ("geneformer", "transcriptformer", "scgpt"):
        if stats["heuristic_looks_log_normalized"]:
            findings.append(
                Finding(
                    "warn",
                    "log_like",
                    "Values look more like **log-normalized** than raw counts (small max, low integer fraction). "
                    "Rank embeddings still run, but ordering may differ from pre-training on counts.",
                )
            )
        gstyle, conf = gene_name_style(list(map(str, ad_emb.var_names)))
        stats["gene_name_style"] = gstyle
        stats["gene_name_style_confidence"] = conf
        if model == "geneformer":
            gf_ck = (scgpt_ckpt or "").strip() or None
            frac_ok, err = geneformer_vocab_check(
                list(map(str, ad_emb.var_names)), model_name=gf_ck
            )
            stats["geneformer_vocab_hit_rate"] = frac_ok
            stats["geneformer_vocab_note"] = err
            if frac_ok is not None:
                if frac_ok < 0.15:
                    findings.append(
                        Finding(
                            "error",
                            "geneformer_vocab",
                            f"Only ~{_fmt(frac_ok*100)}% of sampled genes map to **known** tokens — Geneformer checkpoint likely mismatched (use Ensembl IDs matching the model card).",
                        )
                    )
                elif frac_ok < 0.45:
                    findings.append(
                        Finding(
                            "warn",
                            "geneformer_vocab",
                            f"~{_fmt((1-frac_ok)*100)}% tokenization fall back to UNK; embeddings may be weak.",
                        )
                    )
                else:
                    findings.append(
                        Finding(
                            "ok",
                            "geneformer_vocab",
                            f"~{_fmt(frac_ok*100)}% of sampled genes map to known tokens (rough check).",
                        )
                    )
            else:
                findings.append(
                    Finding(
                        "warn",
                        "geneformer_vocab",
                        f"Could not verify vocabulary overlap ({err or 'unknown'}).",
                    )
                )
        if model == "scgpt":
            ckpt = (scgpt_ckpt or "").strip() or os.environ.get("SCGPT_CKPT_DIR")
            frac_ok, err = scgpt_vocab_check(list(map(str, ad_emb.var_names)), ckpt)
            stats["scgpt_vocab_hit_rate"] = frac_ok
            stats["scgpt_vocab_note"] = err
            if frac_ok is not None:
                if frac_ok < 0.2:
                    findings.append(
                        Finding(
                            "error",
                            "scgpt_vocab",
                            f"Only ~{_fmt(frac_ok*100)}% of sampled genes are in vocab.json — wrong checkpoint or gene naming.",
                        )
                    )
                elif frac_ok < 0.5:
                    findings.append(
                        Finding(
                            "warn",
                            "scgpt_vocab",
                            f"Moderate gene coverage in scGPT vocab (~{_fmt(frac_ok*100)}%); consider harmonizing IDs.",
                        )
                    )
                else:
                    findings.append(
                        Finding(
                            "ok",
                            "scgpt_vocab",
                            f"~{_fmt(frac_ok*100)}% of sampled genes found in checkpoint vocabulary.",
                        )
                    )
            else:
                findings.append(
                    Finding(
                        "warn",
                        "scgpt_vocab",
                        f"Could not check scGPT vocabulary ({err or 'no checkpoint'}).",
                    )
                )

    if model == "scvi":
        if stats["fraction_positive_values_integerish"] < 0.5 and stats["median_positive_sampled"] < 20:
            findings.append(
                Finding(
                    "warn",
                    "scvi_not_counts",
                    "Matrix does not look like raw UMI counts (low integer fraction / small magnitudes). "
                    "SCVI trains on this matrix anyway, but likelihood assumptions may be off.",
                )
            )
        if ad_emb.n_obs < 200:
            findings.append(
                Finding(
                    "warn",
                    "scvi_small_n",
                    f"Only {ad_emb.n_obs} cells — SCVI is trained from scratch here; latent quality may be unstable.",
                )
            )

    if not any(f.level in ("error", "warn") for f in findings):
        findings.append(
            Finding("ok", "baseline", "Core numeric checks passed (finite, non-negative sample)."),
        )

    summary = {
        "model": model,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "matrix_stats": stats,
        "findings": [f.to_dict() for f in findings],
    }
    summary["strict"] = strict
    return findings, summary


def errors_blocking(findings: List[Finding]) -> List[Finding]:
    return [f for f in findings if f.level == "error"]


def default_report_path(model: str) -> Path:
    base = os.environ.get("SCFMS_COMPAT_REPORT_DIR", "").strip()
    root = Path(os.path.expanduser(base)) if base else Path(__file__).resolve().parent.parent / "scfms_reports"
    root.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    return root / f"scfms_compat_{model}_{ts}.pdf"


def write_compatibility_pdf(
    path: Path,
    *,
    model: str,
    matrix_spec: str,
    ad_emb: ad.AnnData,
    findings: List[Finding],
    summary: Dict[str, Any],
) -> Path:
    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stats = summary.get("matrix_stats", {})

    def page_text(title: str, body: str):
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        ax.set_title(title, fontsize=13, fontweight="bold", loc="left", pad=12)
        ax.text(
            0.03,
            0.97,
            textwrap.fill(body, width=96),
            transform=ax.transAxes,
            fontsize=8,
            va="top",
            family="sans-serif",
        )
        return fig

    with PdfPages(str(path)) as pdf:
        cap = (
            f"scFM data compatibility report\n"
            f"Model: {model}   |   matrix: {matrix_spec}\n"
            f"AnnData: n_obs={ad_emb.n_obs}, n_vars={ad_emb.n_vars}\n"
            f"Generated UTC: {summary.get('timestamp_utc')}\n"
        )
        fig = page_text("Overview", cap)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        req = "\n".join(f"• {line}" for line in FORMAT_REQUIREMENTS.get(model, []))
        fig = page_text(f"Expected data format — {model}", req)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        st_lines = [
            f"n_obs (full): {stats.get('n_obs_full')}",
            f"n_vars (full): {stats.get('n_var_full')}",
            f"sampled rows: {stats.get('sampled_rows')}",
            f"median genes >0 (sample rows): {_fmt(stats.get('median_genes_detected_sample'))}",
            f"max sampled value: {_fmt(stats.get('max_sampled_value'))}",
            f"median positive value (sample): {_fmt(stats.get('median_positive_sampled'))}",
            f"fraction integer-like (>0): {_fmt(stats.get('fraction_positive_values_integerish'))}",
            f"heuristic log-normalized look: {stats.get('heuristic_looks_log_normalized')}",
            f"gene name style (heuristic): {stats.get('gene_name_style', 'n/a')}",
            f"geneformer vocab hit (sample): {stats.get('geneformer_vocab_hit_rate')}",
            f"scgpt vocab hit (sample): {stats.get('scgpt_vocab_hit_rate')}",
        ]
        fig = page_text("Matrix / gene diagnostics (sample-based)", "\n".join(st_lines))
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fd_lines = []
        for f in findings:
            fd_lines.append(f"[{f.level.upper()}] {f.code}: {f.message}")
        fig = page_text("Validation findings", "\n".join(fd_lines) if fd_lines else "(none)")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        corp = MODEL_TRAINING_CORPUS.get(model, "")
        fig = page_text("Training corpus & similarity to your data", textwrap.fill(corp, width=96))
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Optional histogram of sampled values
        X = ad_emb.X
        if sp.issparse(X):
            row_idx = np.arange(min(500, X.shape[0]))
            sub = X[row_idx]
            vals = sub.data.astype(np.float64) if sub.nnz else np.array([])
        else:
            Xd = np.asarray(X, dtype=np.float64)[:500]
            vals = Xd.ravel()
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if vals.size > 50:
            fig, ax = plt.subplots(figsize=(8.5, 4))
            ax.hist(np.minimum(vals, np.percentile(vals, 99)), bins=50, color="#4C6EF5", alpha=0.85)
            ax.set_title("Histogram of positive expression values (sample, capped at 99th pct)")
            ax.set_xlabel("value")
            ax.set_ylabel("count")
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    return path


def validate_run_and_pdf(
    model: str,
    ad_emb: ad.AnnData,
    *,
    matrix_spec: str,
    scgpt_ckpt: Optional[str],
    report_path: Optional[Path] = None,
    strict: bool = True,
) -> Tuple[List[Finding], Dict[str, Any], Path]:
    """Validate, write PDF, return findings + summary dict + pdf path."""
    findings, summary = validate_for_embedding(
        model, ad_emb, scgpt_ckpt=scgpt_ckpt, strict=strict
    )
    blockers = errors_blocking(findings)
    pdf_p = Path(report_path) if report_path else default_report_path(model)
    write_compatibility_pdf(
        pdf_p,
        model=model,
        matrix_spec=matrix_spec,
        ad_emb=ad_emb,
        findings=findings,
        summary=summary,
    )
    summary["report_pdf"] = str(pdf_p.resolve())
    if strict and blockers:
        msg = "\n".join(f"• {b.message}" for b in blockers)
        raise ValueError(
            "scFM compatibility checks failed — fix data or set SCFMS_SC_FM_COMPAT_STRICT=0 to warn-only.\n"
            f"{msg}\nPDF written (includes details): {pdf_p}"
        )
    return findings, summary, pdf_p
