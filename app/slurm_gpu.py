"""Slurm helpers for GPU jobs using ``#SBATCH --gres=…`` and ``srun`` (common on many HPC sites)."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Tuple

_ENV_SLURM_ACCOUNT = "SCFMS_SLURM_ACCOUNT"


def _slurm_account_sbatch_line() -> str:
    """``#SBATCH -A …`` from ``SCFMS_SLURM_ACCOUNT`` if set; empty otherwise."""
    acc = (os.environ.get(_ENV_SLURM_ACCOUNT) or "").strip()
    if not acc:
        return ""
    if not re.fullmatch(r"[\w.@+-]+", acc):
        return ""
    return f"#SBATCH -A {acc}\n"


def sanitize_dataset_name(name: str, fallback: str = "dataset") -> str:
    s = (name or "").strip() or fallback
    s = re.sub(r"[^\w\-.]+", "_", s, flags=re.UNICODE)
    s = s.strip("._") or fallback
    return s[:120]


def sh_quote(s: str) -> str:
    """POSIX single-quote literal."""
    return "'" + s.replace("'", "'\\''") + "'"


def default_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _runtime_env_block(*, stage_dir: Path, cpus: int) -> str:
    st = stage_dir.resolve()
    n_cpu = max(1, int(cpus))
    return f"""export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${{MPLCONFIGDIR:-{st.as_posix()}/.mplconfig}}"
mkdir -p "$MPLCONFIGDIR"
export SCFMS_EFFECTIVE_CPUS="${{SLURM_CPUS_PER_TASK:-{n_cpu}}}"
export OMP_NUM_THREADS="${{OMP_NUM_THREADS:-$SCFMS_EFFECTIVE_CPUS}}"
export MKL_NUM_THREADS="${{MKL_NUM_THREADS:-$SCFMS_EFFECTIVE_CPUS}}"
export OPENBLAS_NUM_THREADS="${{OPENBLAS_NUM_THREADS:-$SCFMS_EFFECTIVE_CPUS}}"
export NUMEXPR_NUM_THREADS="${{NUMEXPR_NUM_THREADS:-$SCFMS_EFFECTIVE_CPUS}}"
"""


def build_slurm_gpu_embed_script(
    *,
    repo_root: Path,
    stage_dir: Path,
    params_json: Path,
    partition: str,
    gres: str,
    cpus: int,
    mem: str,
    time_limit: str,
    job_name: str,
    bash_prologue: str,
) -> str:
    """Shell script body suitable for ``sbatch`` on a GPU partition (configurable ``--gres``)."""
    rr = repo_root.resolve()
    st = stage_dir.resolve()
    pj = params_json.resolve()
    runner = rr / "scripts" / "run_scfm_embedding_job.py"
    jn = re.sub(r"[^\w\-]+", "-", job_name).strip("-")[:40] or "scfms-embed"
    prologue = (bash_prologue or "").rstrip()
    if prologue and not prologue.endswith("\n"):
        prologue += "\n"
    gres_l = (gres or "gpu:1").strip() or "gpu:1"
    runtime_env = _runtime_env_block(stage_dir=st, cpus=cpus)
    acct = _slurm_account_sbatch_line()
    return f"""#!/bin/bash
{acct}#SBATCH -p {partition}
#SBATCH --gres={gres_l}
#SBATCH -c {int(cpus)}
#SBATCH --mem={mem}
#SBATCH -t {time_limit}
#SBATCH -J {jn}
#SBATCH -o {st.as_posix()}/slurm-%j.out
#SBATCH -e {st.as_posix()}/slurm-%j.err

set -euo pipefail
cd {sh_quote(rr.as_posix())}
{runtime_env}{prologue}srun --ntasks=1 --nodes=1 --cpu-bind=cores python3 {sh_quote(runner.as_posix())} \\
  --params-json {sh_quote(pj.as_posix())}
"""


def build_slurm_gpu_benchmark_script(
    *,
    repo_root: Path,
    stage_dir: Path,
    params_json: Path,
    partition: str,
    cpus: int,
    mem: str,
    time_limit: str,
    job_name: str,
    gres: str,
    bash_prologue: str,
    extra_sbatch: str,
) -> str:
    """Shell script for ``sbatch`` on GPU partitions (``--gres``).

    ``mem`` is typically node RAM, e.g. ``64G``. ``gres`` examples: ``gpu:1``,
    ``gpu:a100:1`` — set from the UI to match your site's Slurm grammar.
    ``extra_sbatch`` is appended after the standard headers (one directive per line,
    each line should start with ``#SBATCH``).
    """
    rr = repo_root.resolve()
    st = stage_dir.resolve()
    pj = params_json.resolve()
    runner = rr / "scripts" / "run_benchmark_slurm_job.py"
    jn = re.sub(r"[^\w\-]+", "-", job_name).strip("-")[:40] or "scfms-bench"
    prologue = (bash_prologue or "").rstrip()
    if prologue and not prologue.endswith("\n"):
        prologue += "\n"
    gres_l = (gres or "gpu:1").strip() or "gpu:1"
    extras = (extra_sbatch or "").strip()
    if extras and not extras.endswith("\n"):
        extras += "\n"
    extra_block = extras if extras else ""
    runtime_env = _runtime_env_block(stage_dir=st, cpus=cpus)
    acct = _slurm_account_sbatch_line()
    return f"""#!/bin/bash
{acct}#SBATCH -p {partition}
#SBATCH --gres={gres_l}
#SBATCH -c {int(cpus)}
#SBATCH --mem={mem}
#SBATCH -t {time_limit}
#SBATCH -J {jn}
#SBATCH -o {st.as_posix()}/slurm-%j.out
#SBATCH -e {st.as_posix()}/slurm-%j.err
{extra_block}
set -euo pipefail
cd {sh_quote(rr.as_posix())}
{runtime_env}{prologue}srun --ntasks=1 --nodes=1 --cpu-bind=cores python3 {sh_quote(runner.as_posix())} \\
  --params-json {sh_quote(pj.as_posix())}
"""


def run_sbatch(script_path: Path) -> str:
    """Return numeric Slurm job id from ``sbatch`` stdout."""
    r = subprocess.run(
        ["sbatch", str(script_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    out = (r.stdout or "") + (r.stderr or "")
    if r.returncode != 0:
        raise RuntimeError(f"sbatch failed ({r.returncode}): {out.strip()}")
    # Submitted batch job 12345
    parts = out.strip().split()
    if not parts:
        raise RuntimeError("sbatch produced no output")
    job_id = parts[-1]
    if not job_id.isdigit():
        raise RuntimeError(f"Could not parse Slurm job id from: {out!r}")
    return job_id


def slurm_aggregate_state(slurm_job_id: str) -> str:
    """Return a coarse state: PENDING, RUNNING, COMPLETED, FAILED, UNKNOWN, etc."""
    jid = str(slurm_job_id).strip()
    if not jid:
        return "UNKNOWN"
    rq = subprocess.run(
        ["squeue", "-j", jid, "-h", "-t", "all", "-o", "%T"],
        capture_output=True,
        text=True,
    )
    if rq.returncode == 0 and rq.stdout.strip():
        # First line (main step)
        return rq.stdout.strip().split("\n")[0].strip() or "UNKNOWN"
    sa = subprocess.run(
        ["sacct", "-j", jid, "-n", "-X", "-o", "State", "-P", "--noheader"],
        capture_output=True,
        text=True,
    )
    if sa.returncode != 0 or not sa.stdout.strip():
        return "UNKNOWN"
    lines = [ln.strip() for ln in sa.stdout.strip().splitlines() if ln.strip()]
    if not lines:
        return "UNKNOWN"
    st = lines[0].split("|")[0].strip()
    if not st:
        return "UNKNOWN"
    if st.endswith("+"):  # e.g. COMPLETING+
        st = st.rstrip("+")
    return st


def is_slurm_finished_success(state: str) -> Tuple[bool, bool]:
    """(done, success) — success iff COMPLETED."""
    s = state.upper()
    if s in ("RUNNING", "PENDING", "COMPLETING", "PREEMPTED", "SUSPENDED", "CONFIGURING"):
        return False, False
    if s == "COMPLETED":
        return True, True
    return True, False
