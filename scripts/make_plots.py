#!/usr/bin/env python3
import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from umap import UMAP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True, help="Path to embeddings (.csv or .npy)")
    ap.add_argument("--out", required=True, help="Output PNG path")
    ap.add_argument("--adata", help="Optional .h5ad with obs to color", default=None)
    ap.add_argument("--obs_key", help="obs key for color", default=None)
    args = ap.parse_args()

    # Load embeddings
    if args.emb.endswith(".npy"):
        X = np.load(args.emb)
        idx = None
    else:
        df = pd.read_csv(args.emb, index_col=0)
        X = df.values
        idx = df.index.tolist()

    # UMAP
    Xn = StandardScaler().fit_transform(X)
    U = UMAP(
        n_components=2, n_neighbors=15, min_dist=0.1, random_state=0
    ).fit_transform(Xn)

    # Colors
    labels = None
    if args.adata and args.obs_key:
        adata = ad.read_h5ad(args.adata)
        if idx is not None:
            adata = adata[idx]
        labels = adata.obs[args.obs_key].astype(str).tolist()

    # Plot
    plt.figure(figsize=(5, 4))
    if labels is None:
        plt.scatter(U[:, 0], U[:, 1], s=6, alpha=0.8)
    else:
        # Simple categorical coloring
        cats = pd.Categorical(labels)
        colors = plt.get_cmap("tab10")(cats.codes % 10)
        plt.scatter(U[:, 0], U[:, 1], s=6, alpha=0.8, c=colors)
        # Legend
        handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=cl,
                markerfacecolor=plt.get_cmap("tab10")(i % 10),
                markersize=6,
            )
            for i, cl in enumerate(cats.categories)
        ]
        plt.legend(
            handles=handles, bbox_to_anchor=(1.04, 1), loc="upper left", borderaxespad=0
        )
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=180)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
