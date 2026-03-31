"""Slurm defaults from environment (see ``.env.example``)."""

from __future__ import annotations

import os

_ENV_PARTITION = "SCFMS_SLURM_PARTITION"
_FALLBACK_PARTITION = "gpu"


def default_slurm_partition() -> str:
    """Partition prefill when the UI/CLI does not override (from ``SCFMS_SLURM_PARTITION`` or ``gpu``)."""
    v = (os.environ.get(_ENV_PARTITION) or "").strip()
    return v or _FALLBACK_PARTITION


def effective_slurm_partition(user_value: str | None) -> str:
    """Use a non-empty user/UI value, otherwise :func:`default_slurm_partition`."""
    u = (user_value or "").strip()
    return u if u else default_slurm_partition()
