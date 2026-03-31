#!/usr/bin/env python3
"""
Download scFM model weights into the local repo after clone.

Examples
--------
python scripts/download_weights.py --models geneformer scgpt transcriptformer
python scripts/download_weights.py --models all --env-file .scfms.env
python scripts/download_weights.py --models transcriptformer --transcriptformer-variant tf-exemplar
"""

from __future__ import annotations

import argparse
import os
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
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


# Official CZI weights (public S3). Maps CLI flag / env to tarball name on the bucket.
TRANSCRIPTFORMER_VARIANT_TO_KEY: dict[str, str] = {
    "tf-sapiens": "tf_sapiens",
    "tf-exemplar": "tf_exemplar",
    "tf-metazoa": "tf_metazoa",
}


def _transcriptformer_s3_key(variant: str) -> str:
    v = str(variant or "").strip().lower().replace("_", "-")
    if v in TRANSCRIPTFORMER_VARIANT_TO_KEY:
        return TRANSCRIPTFORMER_VARIANT_TO_KEY[v]
    env_v = os.environ.get("TRANSCRIPTFORMER_VARIANT", "").strip().lower().replace("_", "-")
    if env_v in TRANSCRIPTFORMER_VARIANT_TO_KEY:
        return TRANSCRIPTFORMER_VARIANT_TO_KEY[env_v]
    raise ValueError(
        f"Unknown TranscriptFormer variant {variant!r}; "
        f"use one of {list(TRANSCRIPTFORMER_VARIANT_TO_KEY)} (or set TRANSCRIPTFORMER_VARIANT)."
    )


def _download_transcriptformer_s3(checkpoint_dir: Path, model_key: str) -> None:
    """Download and extract one TranscriptFormer bundle (same layout as CZI ``download_artifacts``)."""
    s3_url = f"https://czi-transcriptformer.s3.amazonaws.com/weights/{model_key}.tar.gz"
    checkpoint_dir = checkpoint_dir.expanduser().resolve()
    ensure_dir(checkpoint_dir)
    output_dir = checkpoint_dir / model_key

    def report_hook(count: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        done = min(count * block_size, total_size)
        pct = int(100 * done / total_size)
        if count % 128 == 0 or done >= total_size:
            mb = done / (1024 * 1024)
            tot_mb = total_size / (1024 * 1024)
            console.print(
                f"[dim]TranscriptFormer download… {pct:3d}% ({mb:.1f} / {tot_mb:.1f} MiB)[/dim]",
                end="\r",
            )

    console.print(f"[bold]TranscriptFormer[/bold] {s3_url} -> {checkpoint_dir}/")
    try:
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=True) as tmp:
            urllib.request.urlretrieve(s3_url, filename=tmp.name, reporthook=report_hook)
            console.print()
            tmp.seek(0)
            with tarfile.open(fileobj=tmp, mode="r:gz") as tar:
                members = tar.getmembers()
                for i, member in enumerate(members, 1):
                    tar.extract(member, path=str(checkpoint_dir))
                    if i == 1 or i == len(members) or i % 500 == 0:
                        console.print(
                            f"[dim]Extracting… {i}/{len(members)}[/dim]",
                            end="\r",
                        )
            console.print()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise RuntimeError(f"Model tarball not found at {s3_url}") from e
        raise RuntimeError(f"HTTP {e.code} while downloading TranscriptFormer weights") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Download failed: {e}") from e
    except tarfile.ReadError as e:
        raise RuntimeError("Downloaded file is not a valid .tar.gz") from e

    console.print(f"[green]TranscriptFormer weights ready under[/green] {output_dir}")


def dl_transcriptformer(dest: Path, variant: str) -> None:
    key = _transcriptformer_s3_key(variant)
    _download_transcriptformer_s3(dest, key)


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


def write_env_file(path: Path, root: Path, transcriptformer_subdir: str) -> None:
    tf_path = (root / "transcriptformer" / transcriptformer_subdir).resolve()
    lines = [
        f"export GENEFORMER_MODEL={str((root / 'geneformer').resolve())}",
        f"export TRANSCRIPTFORMER_MODEL={str(tf_path)}",
        f"export SCGPT_CKPT_DIR={str((root / 'scgpt' / 'whole-human').resolve())}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    console.print(f"[green]Wrote env file:[/green] {path}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
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
    ap.add_argument(
        "--transcriptformer-variant",
        default=os.environ.get("TRANSCRIPTFORMER_VARIANT", "tf-sapiens"),
        choices=sorted(TRANSCRIPTFORMER_VARIANT_TO_KEY.keys()),
        help=(
            "Which TranscriptFormer checkpoint to download from CZI S3 "
            "(tf-sapiens = human-only, ~smallest; see models/README.md)."
        ),
    )
    args = ap.parse_args(argv)

    root = models_root(args.models_dir)
    ensure_dir(root)

    tf_key = _transcriptformer_s3_key(args.transcriptformer_variant)

    actions = {
        "geneformer": lambda: dl_geneformer(root / "geneformer"),
        "transcriptformer": lambda: dl_transcriptformer(root / "transcriptformer", args.transcriptformer_variant),
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
        write_env_file(Path(args.env_file).expanduser().resolve(), root, tf_key)

    if failures:
        console.print(f"[yellow]{failures} download step(s) failed.[/yellow]")
        return 1

    console.print("[green]All requested downloads completed.[/green]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
