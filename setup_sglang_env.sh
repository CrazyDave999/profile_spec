#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
uv venv .venv-sglang --python 3.12 --allow-existing
# CPU Rust extensions are optional; current measurements must record this setting.
# This does not disable CUDA Graphs or the GPU SD kernels.
SGLANG_BUILD_RUST_EXTS=none uv pip sync --python .venv-sglang/bin/python requirements-sglang.lock
uv pip check --python .venv-sglang/bin/python
