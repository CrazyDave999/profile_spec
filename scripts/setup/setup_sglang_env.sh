#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
uv venv .venv-sglang --python 3.12 --allow-existing
# CPU Rust extensions are optional; current measurements must record this setting.
# This does not disable CUDA Graphs or the GPU SD kernels.
SGLANG_BUILD_RUST_EXTS=none uv pip sync --python .venv-sglang/bin/python requirements-sglang.lock
uv pip check --python .venv-sglang/bin/python
# FlashInfer's JIT expects a toolkit layout, while CUDA pip wheels use lib/.
sglang_cuda_prefix="$PWD/.venv-sglang/lib/python3.12/site-packages/nvidia/cu13"
if [[ ! -e "$sglang_cuda_prefix/lib64" ]]; then ln -s lib "$sglang_cuda_prefix/lib64"; fi
if [[ ! -e "$sglang_cuda_prefix/lib/libcudart.so" ]]; then ln -s libcudart.so.13 "$sglang_cuda_prefix/lib/libcudart.so"; fi
if [[ ! -e "$sglang_cuda_prefix/lib/libnvrtc.so" ]]; then ln -s libnvrtc.so.13 "$sglang_cuda_prefix/lib/libnvrtc.so"; fi
CUDA_HOME="$sglang_cuda_prefix" .venv-sglang/bin/python scripts/setup/patch_cuda_glibc.py
