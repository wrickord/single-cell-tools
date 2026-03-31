# models/

Downloaded checkpoints and weights live here. The `models/` directory is tracked in git, but **file contents are gitignored** so clones stay small.

## What `main.py download-weights` does

From the **repository root**, `uv run python main.py download-weights -- …` runs `scripts/download_weights.py` (everything after `--` is passed through). It can fetch:

| Target | Source | Default layout under `models/` |
|--------|--------|--------------------------------|
| **geneformer** | Hugging Face (`GENEFORMER_REPO`, default `ctheodoris/Geneformer`) | `geneformer/` |
| **transcriptformer** | **CZI public S3** (`czi-transcriptformer.s3.amazonaws.com/weights/…`) | `transcriptformer/<variant_dir>/` |
| **scgpt** | Google Drive folder (`SCGPT_URL`) | `scgpt/whole-human/` |
| **xtrimo** | Hugging Face (may need `HF_TOKEN`) | `xtrimo-356m-entrez/` |

`--models all` downloads **geneformer**, **transcriptformer**, and **scgpt** (not xtrimo).

### TranscriptFormer (CZI)

TranscriptFormer weights are **not** on Hugging Face under the old `cziscience/…` id (that URL was invalid). This repo downloads the official **`.tar.gz`** from AWS, matching the [CZI TranscriptFormer](https://github.com/czi-ai/transcriptformer) CLI.

- **Default variant:** `tf-sapiens` (human-only checkpoint; smallest of the three).
- **Other variants:** `tf-exemplar`, `tf-metazoa`.

```bash
uv run python main.py download-weights -- --models transcriptformer
uv run python main.py download-weights -- --models transcriptformer --transcriptformer-variant tf-exemplar
```

Equivalent without `main.py`:

```bash
uv run python scripts/download_weights.py --models transcriptformer
```

Artifacts unpack to:

```text
models/transcriptformer/tf_sapiens/    # or tf_exemplar / tf_metazoa
```

You can also set **`TRANSCRIPTFORMER_VARIANT`** (`tf-sapiens`, `tf-exemplar`, or `tf-metazoa`) instead of using the flag.

### Optional: env file for paths

```bash
uv run python main.py download-weights -- --models all --env-file .scfms.env
# then: source .scfms.env
```

`TRANSCRIPTFORMER_MODEL` in that file points at `models/transcriptformer/<variant>/` (same folder as `--transcriptformer-variant`).

### Using official CZI inference (recommended for downloaded weights)

The checkpoints above include **`model_weights.pt`** and Hydra-style configs for the **`transcriptformer`** Python package, not Hugging Face `AutoModel`. For full inference / embeddings with those files, install from [PyPI](https://pypi.org/project/transcriptformer/) or clone [czi-ai/transcriptformer](https://github.com/czi-ai/transcriptformer) and follow their docs (`transcriptformer download …`, Hydra configs).

If you keep local Hydra snippets or Slurm wrappers, a typical checkout for configs is:

`/n/data1/hms/dbmi/zitnik/lab/users/war013/Artifacts/Utilities/TranscriptFormer`

Point **`model.checkpoint_path`** at the downloaded directory (e.g. `…/models/transcriptformer/tf_sapiens`), consistent with CZI’s layout (`./checkpoints/tf_sapiens` in their examples).

### In-repo `scripts/generate_embeddings.py` note

The **`embed_transcriptformer`** helper there only loads **Hugging Face–style** checkpoints via `transformers`. If `TRANSCRIPTFORMER_MODEL` is set to a CZI S3 layout (directory containing `model_weights.pt`), it will error with a pointer to this README — use the official package for those weights.

## Environment variables

See the root **README** and **`.env.example`** for `HF_TOKEN`, `GENEFORMER_REPO`, `SCGPT_URL`, `XTRIMO_REPO`, `TRANSCRIPTFORMER_VARIANT`, and path overrides (`GENEFORMER_MODEL`, `TRANSCRIPTFORMER_MODEL`, `SCGPT_CKPT_DIR`).
