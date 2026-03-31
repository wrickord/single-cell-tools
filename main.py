#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _run_module(module: str, extra_args: list[str]) -> int:
    cmd = [sys.executable, "-m", module, *extra_args]
    return subprocess.run(cmd, cwd=_repo_root(), check=False).returncode


def main() -> int:
    try:
        from app.env_bootstrap import load_repo_dotenv

        load_repo_dotenv()
    except ImportError:
        pass

    ap = argparse.ArgumentParser(
        description="scFMs entrypoint: launch apps, download weights, or submit Slurm jobs."
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    for name in ("embeddings",):
        sp = sub.add_parser(name, help=f"Launch the {name} app")
        sp.add_argument("args", nargs=argparse.REMAINDER, help="Extra args passed through")

    dl = sub.add_parser("download-weights", help="Download model weights into ./models")
    dl.add_argument("args", nargs=argparse.REMAINDER, help="Extra args passed through")

    sj = sub.add_parser(
        "submit-embedding-slurm",
        help="Submit a headless scFM embedding Slurm job from a login/SSH node",
    )
    sj.add_argument("args", nargs=argparse.REMAINDER, help="Extra args passed through")

    args = ap.parse_args()

    if args.cmd == "embeddings":
        return _run_module("app.app", args.args)
    if args.cmd == "download-weights":
        return _run_module("scripts.download_weights", args.args)
    if args.cmd == "submit-embedding-slurm":
        return _run_module("scripts.submit_scfm_embedding_slurm", args.args)
    raise SystemExit(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
