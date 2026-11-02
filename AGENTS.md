# Project instructions

- Keep this project's code, experiment launchers and environment under this repository (`~/profile_spec`). Model weights stay in the external Hugging Face cache, with aliases under `~/models`.
- JetSpec is a submodule pinned in `third_party/JetSpec`. Read its applicable `AGENTS.md` before changing upstream code. Record the submodule revision with experimental results.
- Source `env.sh` to select the RTX 5090 and the root `.venv`. Do not use the RTX 5080 for these experiments unless requested.
- `bench_5090.sh` measures throughput. Its AR baseline is JetSpec's own compiled, CUDA-graphed single-token decoder; it is not an SGLang comparison.
- `profile_5090.sh` collects a warmed CUDA/NVTX trace with graph-node detail. `profile_workload.py` adds instrumentation overhead and must not be used for headline throughput claims.
- Preserve the recorded BF16 AR/tree divergence observation when reporting correctness. Report enabled and disabled features explicitly.
- Keep tokens outside the repository. Local Git credentials read `~/.x6_spec_token` through a repository-scoped helper. Never print, commit, embed in a URL, or copy the token into a profiler's environment.
- Large profiler artifacts belong in ignored `profiles/`. Keep small reproducibility manifests and summary statistics in `logs/`.
