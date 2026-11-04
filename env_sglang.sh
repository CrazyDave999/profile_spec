#!/usr/bin/env bash
# Source for the isolated SGLang runtime; no change to the JetSpec environment.
export PROFILE_SPEC_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$PROFILE_SPEC_ROOT/.venv-sglang/bin/activate"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=GPU-eed4707e-7a00-58e0-4123-1ccbe4326b7a
export CUDA_HOME="$PROFILE_SPEC_ROOT/.venv-sglang/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HOME=/home/qinghao/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export SGLANG_CACHE_DIR="$PROFILE_SPEC_ROOT/.cache/sglang-cu130"
export SGLANG_RAGGED_VERIFY_MODE=static
