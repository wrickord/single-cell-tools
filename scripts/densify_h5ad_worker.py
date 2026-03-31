#!/usr/bin/env python3
"""Minimal process: full read of ``.h5ad`` → write materialized copy.

Invoked via ``subprocess`` so the child does **not** import the Gradio / scanpy app stack
(``multiprocessing`` ``spawn`` would reload ``preprocess.py`` and take minutes on large envs).
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 4:
        print(
            "usage: densify_h5ad_worker.py <src.h5ad> <dst.h5ad> <status.json>",
            file=sys.stderr,
        )
        return 2
    src, dst_final, status_path = sys.argv[1:4]
    import anndata as ad

    partial = str(Path(dst_final).with_suffix(".writing.h5ad"))
    st = Path(status_path)
    try:
        st.parent.mkdir(parents=True, exist_ok=True)
        a = ad.read_h5ad(src)
        # Avoid categorical conversion pass (can dominate wall time on wide ``obs``).
        a.write_h5ad(partial, convert_strings_to_categoricals=False)
        Path(partial).replace(dst_final)
        st.write_text(
            json.dumps({"ok": True, "finished": time.time()}),
            encoding="utf-8",
        )
    except Exception:  # noqa: BLE001
        err = traceback.format_exc()
        try:
            st.write_text(json.dumps({"ok": False, "error": err}), encoding="utf-8")
        except OSError:
            pass
        try:
            Path(partial).unlink(missing_ok=True)
        except OSError:
            pass
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
