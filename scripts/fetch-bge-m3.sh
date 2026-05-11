#!/usr/bin/env bash
# fetch-bge-m3.sh — download BGE-M3 weights for the image bake.
#
# Why this exists: baking the model via `python -c BGEM3FlagModel(...)`
# inside the Dockerfile hangs unpredictably on slow / flaky links —
# docker daemon network calls are invisible and unkillable. Curl on the
# host gives you a visible progress bar and resumable downloads.
#
# Usage:
#   scripts/fetch-bge-m3.sh              # downloads to ./bge-m3
#   scripts/fetch-bge-m3.sh /tmp/bge-m3  # custom dir
#
# Idempotent: re-run safely. curl -C - resumes partial files.
# After this, `docker build -t orion-bge-m3 .` COPYs the directory into
# the image — no network calls inside the build.
set -euo pipefail

OUT_DIR="${1:-$(cd "$(dirname "$0")/.." && pwd)/bge-m3}"
REPO="BAAI/bge-m3"
REVISION="main"
BASE_URL="https://huggingface.co/${REPO}/resolve/${REVISION}"

# Inference files for BGEM3FlagModel. Junk (imgs/, README, .DS_Store)
# intentionally omitted.
FILES=(
  "config.json"
  "tokenizer.json"
  "tokenizer_config.json"
  "special_tokens_map.json"
  "sentencepiece.bpe.model"
  "pytorch_model.bin"
  "colbert_linear.pt"
  "sparse_linear.pt"
  "modules.json"
  "sentence_bert_config.json"
  "config_sentence_transformers.json"
  "1_Pooling/config.json"
)

mkdir -p "${OUT_DIR}"
echo "→ ${REPO} → ${OUT_DIR}"
echo "  ${#FILES[@]} files, ~2.2 GB total. Resumable: re-run on failure."
echo

for rel in "${FILES[@]}"; do
  dest="${OUT_DIR}/${rel}"
  mkdir -p "$(dirname "${dest}")"
  if [[ -f "${dest}" ]]; then
    size=$(wc -c < "${dest}")
    if [[ "${size}" -gt 0 ]]; then
      echo "[skip] ${rel} (${size} bytes)"
      continue
    fi
  fi
  echo "[fetch] ${rel}"
  curl -L --fail --retry 5 --retry-delay 5 -C - --progress-bar \
       -o "${dest}" "${BASE_URL}/${rel}"
done

echo
echo "✓ Done."
echo "  Total size: $(du -sh "${OUT_DIR}" | cut -f1)"
echo "  Build now:  docker build -t orion-bge-m3 ."
