#!/usr/bin/env python3
import os
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad


def make_toy(n_cells=200, n_genes=500, seed=0):
    rng = np.random.default_rng(seed)
    # Two clusters with different mean expression on two gene blocks
    means_a = np.concatenate(
        [
            np.full(n_genes // 5, 3.0),  # up block A
            np.full(n_genes // 5, 0.2),  # down block B
            np.full(n_genes - 2 * (n_genes // 5), 0.5),
        ]
    )
    means_b = np.concatenate(
        [
            np.full(n_genes // 5, 0.2),
            np.full(n_genes // 5, 3.0),
            np.full(n_genes - 2 * (n_genes // 5), 0.5),
        ]
    )

    n_a = n_cells // 2
    n_b = n_cells - n_a
    X_a = rng.poisson(lam=means_a, size=(n_a, n_genes))
    X_b = rng.poisson(lam=means_b, size=(n_b, n_genes))
    X = np.vstack([X_a, X_b]).astype(np.float32)
    obs = pd.DataFrame({"cluster": ["A"] * n_a + ["B"] * n_b})
    var = pd.DataFrame(index=[f"G{i}" for i in range(n_genes)])
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    return adata


def main():
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "test"
    out_dir.mkdir(parents=True, exist_ok=True)
    adata = make_toy()
    path = out_dir / "toy.h5ad"
    adata.write_h5ad(path)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
