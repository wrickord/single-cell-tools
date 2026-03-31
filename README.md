# single-cell-tools

`single-cell-tools` is a small toolkit for working with single-cell datasets locally or on a shared cluster. The repo centers around a Gradio app for generating embeddings, exploring datasets, downloading public data into a managed `.data/` workspace, and kicking off Slurm jobs when the workload is too large for an interactive session.

## Prerequisites

- **Python 3.12+** (see `requires-python` in `pyproject.toml`)
- **[uv](https://docs.astral.sh/uv/getting-started/installation/)** for environments and runs (`uv sync`, `uv run …`)

Optional: copy **`.env.example`** to **`.env`** and fill in only what you need (tokens, paths). `.env` is gitignored.

If you want to try it quickly, the shortest path is:

```bash
git clone <repository-url>
cd single-cell-tools
uv sync
uv run python main.py embeddings
```

Then open the URL Gradio prints (often `http://127.0.0.1:7861`), point the app at a server-side `.h5ad`, `.csv`, or `.tsv`, choose a model, and run embeddings on the current node or through `sbatch`. Adjust **Slurm partition** and **`--gres`** in the UI to match your cluster.

## What you can do with it

- Compute embeddings with `geneformer`, `scGPT`, or `scVI`
- Load data from a server path instead of uploading large files through the browser
- Download datasets from direct URLs, Zenodo, GEO, or CELLxGENE into `.data/`
- Convert downloaded `.csv` and `.tsv` files into `.h5ad` inside the repo
- Inspect PCA and UMAP views
- Submit embedding jobs to Slurm with automatic resource recommendations
- Run benchmark-related workflows from the app when your environment supports them

## Installation

This project is set up for `uv`:

```bash
uv sync
```

That creates the environment from `pyproject.toml` and `uv.lock`.

## Quick start

Launch the app:

```bash
uv run python main.py embeddings
```

From there, a typical workflow looks like this:

1. Pick a dataset from the managed `.data/` area or paste a server path to a `.h5ad`, `.csv`, or `.tsv`.
2. If you do not already have a dataset handy, use the app's "Download Dataset" section to pull one from a URL, Zenodo, GEO, or CELLxGENE.
3. Choose a model.
4. Run embeddings either on the current node or through Slurm.
5. Explore the outputs in the app, including dimensionality reduction views and saved artifacts.

## Downloading model weights

Some models need local weights or checkpoints. You can fetch them with:

```bash
uv run python main.py download-weights -- --models geneformer scgpt transcriptformer
```

You can also call the script directly:

```bash
uv run python scripts/download_weights.py --models geneformer scgpt transcriptformer
```

Available downloads include:

- `geneformer`
- `transcriptformer`
- `scgpt`
- `xtrimo`

Model files are stored under `models/` (see `models/README.md`). File contents are gitignored so the clone stays small.

**Hugging Face:** some downloads (e.g. `xtrimo`) may need a token. Set **`HF_TOKEN`** in the environment or in `.env` — never commit it.

Useful environment variables if you want to point the app at existing local checkpoints:

- `GENEFORMER_MODEL`
- `TRANSCRIPTFORMER_MODEL`
- `SCGPT_CKPT_DIR`

A fuller list of optional variables (paths, `SCFMS_ALLOWED_PATH_PREFIXES`, W&B, etc.) is in **`.env.example`**.

## Working with datasets

The app has a built-in dataset download panel. Downloaded files are organized under:

```text
.data/<dataset-name>/
```

Inside each dataset folder:

- `data/` holds the raw downloaded files
- `processed/` holds derived `.h5ad` files when conversion succeeds

Right now the in-repo conversion step handles `.h5ad`, `.csv`, and `.tsv` inputs. If a download does not convert automatically, the raw files are still preserved in `.data/` so you can process them yourself.

## Running on Slurm

For larger jobs, you can submit embeddings from the CLI:

```bash
uv run python main.py submit-embedding-slurm -- \
  --input-h5ad /path/to/my_dataset.h5ad \
  --model scgpt \
  --matrix-spec X \
  --mem auto \
  --time auto
```

Or call the helper directly:

```bash
uv run python scripts/submit_scfm_embedding_slurm.py \
  --input-h5ad /path/to/my_dataset.h5ad \
  --model scgpt \
  --matrix-spec X \
  --mem auto \
  --time auto
```

This writes staging files under `scfms_job_store/` (override with **`SCFMS_JOB_DIR`**) and, unless you override it, saves outputs under `scfms_job_store/slurm_embeddings/`. The generated batch script is named `slurm_gpu_embed.sh`; use **`--partition`** and **`--gres`** that your site supports.

**Slurm defaults from `.env`:** set **`SCFMS_SLURM_PARTITION`** and optional **`SCFMS_SLURM_ACCOUNT`** in `.env` (see `.env.example`). The UI and CLI prefill the partition from that variable, and a non-empty account adds `#SBATCH -A …` to generated scripts. Values you type in the UI or pass with **`--partition`** still override the partition default.

If you just want to see what would be submitted:

```bash
uv run python scripts/submit_scfm_embedding_slurm.py \
  --input-h5ad /path/to/my_dataset.h5ad \
  --model geneformer \
  --dry-run
```

## Repo layout

- `app/`: the Gradio application and related UI logic
- `scripts/`: download helpers, embedding utilities, and Slurm helpers
- `main.py`: a simple CLI entrypoint for the most common commands
- `.data/`: managed dataset workspace created at runtime
- `models/`: local model weights and checkpoints

## Notes

- `models/`, `.data/`, logs, and runtime job artifacts are ignored by git
- the CLI entrypoints available today are `embeddings`, `download-weights`, and `submit-embedding-slurm`
- benchmark and W&B-related features are available in the app, but they depend on your local environment being set up for them
