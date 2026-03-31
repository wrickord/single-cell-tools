#!/usr/bin/env python3
"""
Download scFM model weights into the local repo after clone.

Examples
--------
python scripts/download_weights.py --models geneformer scgpt transcriptformer
python scripts/download_weights.py --models all --env-file .scfms.env
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable

from rich.console import Console

console = Console()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def models_root(models_dir: str | None) -> Path:
    if models_dir:
        return Path(models_dir).expanduser().resolve()
    return repo_root() / "models"


def dl_geneformer(dest: Path) -> None:
    from huggingface_hub import snapshot_download

    repo_id = os.environ.get("GENEFORMER_REPO", "ctheodoris/Geneformer")
    console.print(f"[bold]Geneformer[/bold] {repo_id} -> {dest}")
    ensure_dir(dest)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(dest),
        local_dir_use_symlinks=False,
    )


def dl_transcriptformer(dest: Path) -> None:
    from huggingface_hub import snapshot_download

    repo_id = os.environ.get(
        "TRANSCRIPTFORMER_REPO",
        "cziscience/Transcriptformer-homo-sapiens-entire",
    )
    console.print(f"[bold]Transcriptformer[/bold] {repo_id} -> {dest}")
    ensure_dir(dest)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(dest),
        local_dir_use_symlinks=False,
    )


def dl_scgpt(dest: Path) -> None:
    import gdown

    folder_url = os.environ.get(
        "SCGPT_URL",
        "https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y?usp=sharing",
    )
    console.print(f"[bold]scGPT[/bold] {folder_url} -> {dest}")
    ensure_dir(dest)
    gdown.download_folder(
        folder_url,
        output=str(dest),
        quiet=False,
        use_cookies=False,
    )


def dl_xtrimo(dest: Path) -> None:
    from huggingface_hub import snapshot_download

    repo_id = os.environ.get(
        "XTRIMO_REPO",
        "TencentAILabHealthcare/xTrimoGene-356M-Entrez",
    )
    token = os.environ.get("HF_TOKEN")
    console.print(f"[bold]xTrimoGene[/bold] {repo_id} -> {dest}")
    ensure_dir(dest)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(dest),
        local_dir_use_symlinks=False,
        token=token,
    )


def selected_models(raw_models: Iterable[str]) -> list[str]:
    vals = [str(x).strip().lower() for x in raw_models if str(x).strip()]
    if not vals or "all" in vals:
        return ["geneformer", "transcriptformer", "scgpt"]
    out: list[str] = []
    for name in vals:
        if name not in out:
            out.append(name)
    return out


def write_env_file(path: Path, root: Path) -> None:
    lines = [
        f"export GENEFORMER_MODEL={str((root / 'geneformer').resolve())}",
        f"export TRANSCRIPTFORMER_MODEL={str((root / 'transcriptformer').resolve())}",
        f"export SCGPT_CKPT_DIR={str((root / 'scgpt' / 'whole-human').resolve())}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    console.print(f"[green]Wrote env file:[/green] {path}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help="Subset of models: geneformer, transcriptformer, scgpt, xtrimo, all",
    )
    ap.add_argument(
        "--models-dir",
        default=None,
        help="Destination root (default: <repo>/models)",
    )
    ap.add_argument(
        "--env-file",
        default=None,
        help="Optional file to write export lines for local model paths",
    )
    args = ap.parse_args(argv)

    root = models_root(args.models_dir)
    ensure_dir(root)

    actions = {
        "geneformer": lambda: dl_geneformer(root / "geneformer"),
        "transcriptformer": lambda: dl_transcriptformer(root / "transcriptformer"),
        "scgpt": lambda: dl_scgpt(root / "scgpt" / "whole-human"),
        "xtrimo": lambda: dl_xtrimo(root / "xtrimo-356m-entrez"),
    }

    failures = 0
    for model in selected_models(args.models):
        action = actions.get(model)
        if action is None:
            console.print(f"[red]Unknown model:[/red] {model}")
            failures += 1
            continue
        try:
            action()
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]Download failed[/red] for {model}: {exc}")
            failures += 1

    if args.env_file:
        write_env_file(Path(args.env_file).expanduser().resolve(), root)

    if failures:
        console.print(f"[yellow]{failures} download step(s) failed.[/yellow]")
        return 1

    console.print("[green]All requested downloads completed.[/green]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
