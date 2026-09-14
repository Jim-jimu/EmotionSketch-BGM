#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$ROOT_DIR"

echo "[info] preparing project layout under: $ROOT_DIR"

mkdir -p \
  data \
  Experiment/datasets/raw/bgm909/new_raw_video_feats \
  Experiment/datasets/raw/bgm909/caption_feats \
  Experiment/datasets/raw/bgm909/detection \
  Experiment/datasets/raw/POP909_4_bin_pnt_8bar \
  Experiment/datasets/pretrained/pnotree_20 \
  Experiment/datasets/pretrained/polydis \
  Experiment/datasets/pretrained/a2s \
  Experiment/datasets/pretrained/chd8bar \
  Experiment/datasets/emopia \
  Experiment/datasets/curated_emotionbgm \
  Experiment/core_code/data \
  Experiment/core_code/emotionsketch_diffbgm \
  Experiment/core_code/scripts \
  Experiment/core_code/logs \
  Experiment/core_code/checkpoints

# Match upstream Diff-BGM path expectations without editing the reference repo.
ln -sfn ../Experiment/datasets/raw/bgm909 data/bgm909
ln -sfn ../Experiment/datasets/raw/POP909_4_bin_pnt_8bar data/POP909_4_bin_pnt_8bar
ln -sfn Experiment/datasets/pretrained pretrained

echo "[info] layout ready"
echo "[info] symlink status:"
ls -ld data/bgm909 data/POP909_4_bin_pnt_8bar pretrained

echo "[info] top-level dataset directories:"
find Experiment/datasets -maxdepth 2 -type d | sort
