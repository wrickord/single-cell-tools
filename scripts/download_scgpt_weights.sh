#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)

python3 "$ROOT/scripts/download_weights.py" \
  --models scgpt \
  --env-file "$ROOT/.scfms.env"

echo
echo "scGPT download complete."
echo "To use the local checkpoint path:"
echo "  source \"$ROOT/.scfms.env\""
