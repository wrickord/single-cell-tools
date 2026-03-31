"""Shared Gradio launch settings and runtime hints for remote / tunnel setups."""

from __future__ import annotations

import os
import socket
import sys
from pathlib import Path
from typing import Any, Dict


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def app_dir() -> Path:
    return Path(__file__).resolve().parent


def launch_kwargs(default_port: int) -> Dict[str, Any]:
    """
    Environment:
      GRADIO_SERVER_NAME  — bind address (default 0.0.0.0)
      PORT                — port
      GRADIO_SHARE        — if 1/true, create a temporary public gradio.live link
      GRADIO_ROOT_PATH    — subpath when behind a reverse proxy (e.g. /scfms)
    """
    name = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0").strip()
    port = int(os.environ.get("PORT", str(default_port)))
    share = os.environ.get("GRADIO_SHARE", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    root = os.environ.get("GRADIO_ROOT_PATH", "").strip()
    kw: Dict[str, Any] = dict(
        server_name=name,
        server_port=port,
        share=share,
        show_error=True,
    )
    if root:
        kw["root_path"] = root
    return kw


def print_bind_banner(app_label: str, default_port: int) -> None:
    kw = launch_kwargs(default_port)
    host = kw["server_name"]
    port = kw["server_port"]
    hn = socket.gethostname()
    fq = socket.getfqdn()
    print("\n" + "=" * 72, flush=True)
    print(f"  {app_label}", flush=True)
    print(f"  Process host: {hn}  ({fq})", flush=True)
    print(f"  CWD: {Path.cwd().resolve()}", flush=True)
    print(f"  Python: {sys.executable}", flush=True)
    print(f"  Bind: http://{host}:{port}/  (0.0.0.0 = all interfaces on this machine)", flush=True)
    if kw.get("share"):
        print("  GRADIO_SHARE=1 — Gradio will print a temporary public URL.", flush=True)
    if kw.get("root_path"):
        print(f"  Behind proxy: root_path={kw['root_path']!r}", flush=True)
    print(
        "  Tip: ‘Load from server path’ reads THIS machine’s disk. Browser “localhost” via",
        flush=True,
    )
    print(
        "  SSH -L only forwards ports; it does not move the Python process to your laptop.",
        flush=True,
    )
    print("=" * 72 + "\n", flush=True)


def runtime_info_markdown() -> str:
    """Static snapshot at import / first page build — refresh by restarting the app."""
    try:
        hn = socket.gethostname()
        fq = socket.getfqdn()
    except OSError:
        hn, fq = "?", "?"
    rr = repo_root()
    return (
        "### Where is this app running?\n\n"
        f"- **Host:** `{hn}` · **FQDN:** `{fq}`\n"
        f"- **Repo:** `{rr}`\n"
        f"- **Process CWD:** `{Path.cwd().resolve()}`\n\n"
        "**Server path** (in the preprocess app) and **`sbatch`** run on **this host**, not on your laptop. "
        "If you use **SSH port forwarding** (`ssh -L 7861:localhost:7861 ...`), your browser’s `localhost` "
        "forwards to this machine — paths like `/scratch/...` on the server are still **cluster paths**.\n\n"
        "**If paths look like your Mac** (`/Users/...`), the Python process is running locally on the Mac; "
        "SSH to the cluster, `cd` to the repo there, activate the venv, and start the app **on the compute** node.\n\n"
        "Environment: `GRADIO_SERVER_NAME`, `PORT`, `GRADIO_SHARE`, `GRADIO_ROOT_PATH` — see `app/gradio_config.py`."
    )


def launch_gradio_demo(demo, *, default_port: int, app_label: str) -> None:
    print_bind_banner(app_label, default_port)
    demo.launch(**launch_kwargs(default_port))
