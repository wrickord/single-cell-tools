#!/usr/bin/env python3
from __future__ import annotations

import fnmatch
import gzip
import shutil
import tarfile
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from scripts.generate_embeddings import load_adata

def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent

def managed_data_root() -> Path:
    root = repo_root() / ".data"
    root.mkdir(parents=True, exist_ok=True)
    return root


def sanitize_dataset_label(name: str, fallback: str = "dataset") -> str:
    raw = (name or "").strip() or fallback
    out = []
    for ch in raw:
        if ch.isalnum():
            out.append(ch)
        elif ch in ("-", "_", "."):
            out.append(ch)
        elif ch.isspace():
            out.append("-")
    label = "".join(out).strip("._-") or fallback
    return label[:80]


def managed_dataset_dir_name(name: str, *, when: datetime | None = None) -> str:
    now = when or datetime.now()
    safe = sanitize_dataset_label(name)
    return f"{now:%Y}-{now:%m}-{safe}"


def create_managed_dataset_layout(name: str) -> dict[str, Path]:
    ds_root = managed_data_root() / managed_dataset_dir_name(name)
    data_dir = ds_root / "data"
    processed_dir = ds_root / "processed"
    data_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    return {
        "dataset_root": ds_root.absolute(),
        "data_dir": data_dir.absolute(),
        "processed_dir": processed_dir.absolute(),
    }


def _conversion_candidates(data_dir: Path) -> list[Path]:
    supported = {".h5ad", ".csv", ".tsv"}
    return sorted(
        p.resolve()
        for p in data_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in supported
    )


def _output_h5ad_path(src: Path, processed_dir: Path) -> Path:
    safe_parent = "__".join(src.relative_to(src.anchor).parts[-3:-1]).strip("_")
    stem = src.stem
    if safe_parent:
        stem = f"{safe_parent}__{stem}"
    return (processed_dir / f"{stem}.h5ad").resolve()


def convert_download_tree_to_h5ad(data_dir: Path, processed_dir: Path) -> dict[str, Any]:
    created_h5ad: list[Path] = []
    existing_h5ad: list[Path] = []
    errors: list[str] = []
    attempted = 0

    for src in _conversion_candidates(data_dir):
        if src.suffix.lower() == ".h5ad":
            existing_h5ad.append(src)
            continue
        attempted += 1
        out_path = _output_h5ad_path(src, processed_dir)
        if out_path.exists():
            existing_h5ad.append(out_path)
            continue
        try:
            adata = load_adata(str(src))
            adata.write_h5ad(str(out_path))
            created_h5ad.append(out_path)
        except Exception as exc:
            errors.append(f"{src.name}: {exc}")

    return {
        "created_h5ad": sorted({p.resolve() for p in created_h5ad}),
        "existing_h5ad": sorted({p.resolve() for p in existing_h5ad}),
        "attempted": attempted,
        "conversion_errors": errors,
    }


def _filename_from_url(url: str, fallback: str = "download.bin") -> str:
    name = Path(urlparse(url).path).name
    return name or fallback


def _stream_download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        with dest.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fh.write(chunk)
    return dest.resolve()


def _download_url_any_scheme(url: str, dest: Path) -> Path:
    parsed = urlparse(url)
    if parsed.scheme.lower() in ("http", "https"):
        return _stream_download(url, dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    return dest.resolve()


def _extract_if_archive(path: Path) -> list[Path]:
    out: list[Path] = []
    lower = path.name.lower()
    if lower.endswith(".zip"):
        extract_dir = path.parent / path.stem
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path) as zf:
            zf.extractall(extract_dir)
        out.append(extract_dir.resolve())
    elif lower.endswith((".tar.gz", ".tgz", ".tar")):
        stem = path.name
        if stem.endswith(".tar.gz"):
            stem = stem[:-7]
        elif stem.endswith(".tgz"):
            stem = stem[:-4]
        elif stem.endswith(".tar"):
            stem = stem[:-4]
        extract_dir = path.parent / stem
        extract_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(path, "r:*") as tf:
            tf.extractall(extract_dir)
        out.append(extract_dir.resolve())
    elif lower.endswith(".gz") and not lower.endswith(".tar.gz"):
        out_path = path.with_suffix("")
        with gzip.open(path, "rb") as fin, out_path.open("wb") as fout:
            shutil.copyfileobj(fin, fout)
        out.append(out_path.resolve())
    return out


_GEO_SUPPL_SKIP_SUFFIXES = (
    ".bam",
    ".bai",
    ".cram",
    ".crai",
    ".fastq",
    ".fastq.gz",
    ".fq",
    ".fq.gz",
    ".fa",
    ".fa.gz",
    ".fasta",
    ".fasta.gz",
)


def _geo_should_download_supplementary(url: str) -> bool:
    name = _filename_from_url(url).lower()
    if not name:
        return False
    if name.endswith(_GEO_SUPPL_SKIP_SUFFIXES):
        return False
    if name.endswith(
        (
            ".h5ad",
            ".h5",
            ".h5mu",
            ".loom",
            ".csv",
            ".csv.gz",
            ".tsv",
            ".tsv.gz",
            ".txt",
            ".txt.gz",
            ".mtx",
            ".mtx.gz",
            ".zip",
            ".tar",
            ".tar.gz",
            ".tgz",
            ".rds",
        )
    ):
        return True
    return "raw" in name


def download_direct_url(
    url: str,
    data_dir: Path,
    *,
    filename_override: str = "",
    extract_archives: bool = True,
) -> dict[str, Any]:
    name = filename_override.strip() or _filename_from_url(url)
    target = data_dir / name
    saved = _stream_download(url, target)
    extracted = _extract_if_archive(saved) if extract_archives else []
    return {
        "downloaded": [saved],
        "extracted": extracted,
    }


def download_zenodo_record(
    record_id: str,
    data_dir: Path,
    *,
    file_glob: str = "",
    extract_archives: bool = True,
) -> dict[str, Any]:
    meta_url = f"https://zenodo.org/api/records/{record_id.strip()}"
    resp = requests.get(meta_url, timeout=60)
    resp.raise_for_status()
    meta = resp.json()
    downloaded: list[Path] = []
    extracted: list[Path] = []
    patt = file_glob.strip()
    for item in meta.get("files", []):
        file_name = str(item.get("key") or "")
        if patt and not fnmatch.fnmatch(file_name, patt):
            continue
        file_url = str((item.get("links") or {}).get("self") or "")
        if not file_url:
            continue
        saved = _stream_download(file_url, data_dir / file_name)
        downloaded.append(saved)
        if extract_archives:
            extracted.extend(_extract_if_archive(saved))
    if not downloaded:
        raise FileNotFoundError("Zenodo record returned no matching files to download.")
    return {
        "downloaded": downloaded,
        "extracted": extracted,
    }


def download_geo_accession(accession: str, data_dir: Path) -> dict[str, Any]:
    import GEOparse

    acc = accession.strip()
    if not acc:
        raise ValueError("GEO accession is required.")
    dest = data_dir / acc
    dest.mkdir(parents=True, exist_ok=True)
    geo_obj = GEOparse.get_GEO(
        geo=acc,
        destdir=str(dest),
        how="full",
        include_data=True,
        silent=False,
    )
    downloaded: list[Path] = sorted(p.resolve() for p in dest.rglob("*") if p.is_file())
    extracted: list[Path] = []
    metadata = getattr(geo_obj, "metadata", {}) or {}
    supp_urls = [str(u).strip() for u in metadata.get("supplementary_file", []) if str(u).strip()]
    for url in supp_urls:
        if not _geo_should_download_supplementary(url):
            continue
        try:
            saved = _download_url_any_scheme(url, dest / _filename_from_url(url))
            downloaded.append(saved)
            extracted.extend(_extract_if_archive(saved))
        except Exception:
            continue
    uniq_downloaded = sorted({p.resolve() for p in downloaded})
    uniq_extracted = sorted({p.resolve() for p in extracted})
    return {
        "downloaded": uniq_downloaded,
        "extracted": uniq_extracted,
        "geo_title": str(getattr(geo_obj, "metadata", {}).get("title", [""])[0] or ""),
    }


def download_dataset(
    method: str,
    dataset_name: str,
    identifier: str,
    *,
    extra: str = "",
) -> dict[str, Any]:
    layout = create_managed_dataset_layout(dataset_name)
    data_dir = layout["data_dir"]
    processed_dir = layout["processed_dir"]
    method_key = str(method or "").strip()
    if method_key in ("direct_url", "cellxgene_url"):
        result = download_direct_url(identifier, data_dir, filename_override=extra)
    elif method_key == "zenodo_record":
        result = download_zenodo_record(identifier, data_dir, file_glob=extra)
    elif method_key == "geo_accession":
        result = download_geo_accession(identifier, data_dir)
    else:
        raise ValueError(f"Unsupported download method: {method_key}")
    try:
        conversion = convert_download_tree_to_h5ad(data_dir, processed_dir)
    except Exception as exc:
        conversion = {
            "created_h5ad": [],
            "existing_h5ad": [],
            "attempted": 0,
            "conversion_errors": [str(exc)],
        }
    result.update(layout)
    result.update(conversion)
    return result
