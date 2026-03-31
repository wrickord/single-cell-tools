# models/

Downloaded checkpoints and weights live here. The directory is listed in git, but **contents are gitignored** so clones stay small.

After cloning, fetch weights (from the repo root):

```bash
uv run python main.py download-weights -- --models geneformer scgpt transcriptformer
```

Optional: write path exports for your shell:

```bash
uv run python main.py download-weights -- --models all --env-file .scfms.env
# then: source .scfms.env
```

See the root **README** and **`.env.example`** for `HF_TOKEN` and other environment variables.
