#!/usr/bin/env bash
set -euo pipefail
source /home/qinghao/profile_spec/env.sh
cd /home/qinghao/profile_spec/third_party/JetSpec
python bench/jetspec/engine/tps_walltime.py \
  --model "$JETSPEC_MODEL" --draft-head "$JETSPEC_DRAFT_HEAD" \
  --prompt-set gsm8k --samples 4 --max-tokens 128 \
  --tree-depth 15 --tree-width 7 --budget 128 \
  --drafter graphed --warm-all --session "$@"
