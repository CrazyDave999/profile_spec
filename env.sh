#!/usr/bin/env bash
# Source this file to select the RTX 5090 and the isolated JetSpec environment.
export PROFILE_SPEC_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export JETSPEC_REPO="$PROFILE_SPEC_ROOT/third_party/JetSpec"
source "$PROFILE_SPEC_ROOT/.venv/bin/activate"
export PATH="$HOME/.local/bin:$PATH"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=GPU-eed4707e-7a00-58e0-4123-1ccbe4326b7a
export HF_HOME=/home/qinghao/.cache/huggingface
export JETSPEC_MODEL=/home/qinghao/models/Qwen3-8B
export JETSPEC_DRAFT_HEAD=/home/qinghao/models/jetspec-qwen3-8b
export JETSPEC_BACKEND=triton_paged_tree_cudagraph
# Fusing GEMMs copies projection weights into several verify stacks.
# Start without these copies on a 32 GiB card.
export JETSPEC_FUSE_GEMMS=0
export TOKENIZERS_PARALLELISM=false
