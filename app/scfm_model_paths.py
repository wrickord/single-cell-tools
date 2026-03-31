"""Shared UI helpers: model-specific weight / checkpoint paths under ``./models`` and env."""

from __future__ import annotations

import os
from typing import Any, List, Optional, Tuple

import gradio as gr

from gradio_config import repo_root

_SCFM_WEIGHTS_USE_ENV = "(use default from environment)"
_SCFM_WEIGHTS_SCVI_NOTE = "(scVI — no fixed checkpoint)"


def _dedupe_paths_preserve(paths: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for x in paths:
        s = str(x).strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def model_weights_choices_and_value(model_name: str) -> Tuple[List[str], str]:
    """Discover ``./models`` layouts + env defaults for the embedding model radio."""
    m = str(model_name or "").strip().lower()
    models_dir = repo_root() / "models"
    if m == "scvi":
        return [_SCFM_WEIGHTS_SCVI_NOTE], _SCFM_WEIGHTS_SCVI_NOTE
    if m == "geneformer":
        opts: List[str] = []
        env_p = (os.environ.get("GENEFORMER_MODEL") or "").strip()
        if env_p:
            opts.append(env_p)
        base = models_dir / "geneformer"
        if base.is_dir():
            subdirs = sorted([p for p in base.iterdir() if p.is_dir()])
            if subdirs:
                opts.extend(str(p.resolve()) for p in subdirs)
            else:
                opts.append(str(base.resolve()))
        opts = _dedupe_paths_preserve(opts)
        choices = (
            [_SCFM_WEIGHTS_USE_ENV] + opts if opts else [_SCFM_WEIGHTS_USE_ENV]
        )
        val = (
            env_p
            if env_p in opts
            else (opts[0] if opts else _SCFM_WEIGHTS_USE_ENV)
        )
        if val not in choices:
            val = choices[0]
        return choices, val
    if m == "scgpt":
        opts = []
        env_p = (os.environ.get("SCGPT_CKPT_DIR") or "").strip()
        if env_p:
            opts.append(env_p)
        base = models_dir / "scgpt"
        if base.is_dir():
            for sub in sorted(base.iterdir()):
                if sub.is_dir():
                    opts.append(str(sub.resolve()))
        opts = _dedupe_paths_preserve(opts)
        choices = (
            [_SCFM_WEIGHTS_USE_ENV] + opts if opts else [_SCFM_WEIGHTS_USE_ENV]
        )
        val = (
            env_p
            if env_p in opts
            else (opts[0] if opts else _SCFM_WEIGHTS_USE_ENV)
        )
        if val not in choices:
            val = choices[0]
        return choices, val
    if m == "transcriptformer":
        opts = []
        env_p = (os.environ.get("TRANSCRIPTFORMER_MODEL") or "").strip()
        if env_p:
            opts.append(env_p)
        base = models_dir / "transcriptformer"
        if base.is_dir():
            for sub in sorted(base.iterdir()):
                if sub.is_dir():
                    opts.append(str(sub.resolve()))
        opts = _dedupe_paths_preserve(opts)
        choices = (
            [_SCFM_WEIGHTS_USE_ENV] + opts if opts else [_SCFM_WEIGHTS_USE_ENV]
        )
        val = (
            env_p
            if env_p in opts
            else (opts[0] if opts else _SCFM_WEIGHTS_USE_ENV)
        )
        if val not in choices:
            val = choices[0]
        return choices, val
    return [_SCFM_WEIGHTS_USE_ENV], _SCFM_WEIGHTS_USE_ENV


def model_weights_gr_update(model_name: str):
    ch, val = model_weights_choices_and_value(model_name)
    return gr.update(choices=ch, value=val)


def normalize_ui_weights_path(model: str, raw: Any) -> Optional[str]:
    s = str(raw or "").strip()
    if not s or s in (_SCFM_WEIGHTS_USE_ENV, _SCFM_WEIGHTS_SCVI_NOTE):
        return None
    return s
