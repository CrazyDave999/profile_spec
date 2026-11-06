#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
git submodule update --init --recursive
if [[ ! -d .venv ]]; then
  uv venv --python 3.12 .venv
fi
uv pip install --python .venv/bin/python \
  --index https://download.pytorch.org/whl/cu128 'torch==2.8.0'
if [[ -f /home/qinghao/profile_spec/requirements-5090.lock ]]; then
  uv pip install --python .venv/bin/python -r /home/qinghao/profile_spec/requirements-5090.lock
else
  uv pip install --python .venv/bin/python \
    -e './third_party/JetSpec[bench,kernel]' 'transformers==4.57.1' 'numpy==2.3.3' \
    'datasets==4.6.1' 'huggingface_hub==0.36.2' 'hf_transfer==0.1.9'
fi
uv pip check --python .venv/bin/python
