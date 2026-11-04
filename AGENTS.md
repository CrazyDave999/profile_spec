# Project instructions

- Keep this project's code, experiment launchers and environment under this repository (`~/profile_spec`). Model weights stay in the external Hugging Face cache, with aliases under `~/models`.
- Push to GitHub only when the user explicitly instructs a push. Having credentials or a local commit does not authorize a push.
- New HTML reports live at `reports/YYYY-MM-DD_topic.html`, with assets in the same-named directory and links from `reports/index.html`. Preserve prior dated reports and the initial snapshot.
- Keep the SGLang experiments in `.venv-sglang`; do not upgrade the validated JetSpec `.venv` to satisfy SGLang dependencies. Profile only warmed, explicitly admitted GPU workloads and record actual CUDA Graph replay and backend selection.
- JetSpec is a submodule pinned in `third_party/JetSpec`. Read its applicable `AGENTS.md` before changing upstream code. Record the submodule revision with experimental results.
- Source `env.sh` to select the RTX 5090 and the root `.venv`. Do not use the RTX 5080 for these experiments unless requested.
- `bench_5090.sh` measures throughput. Its AR baseline is JetSpec's own compiled, CUDA-graphed single-token decoder; it is not an SGLang comparison.
- `profile_5090.sh` collects a warmed CUDA/NVTX trace with graph-node detail. `profile_workload.py` adds instrumentation overhead and must not be used for headline throughput claims.
- Preserve the recorded BF16 AR/tree divergence observation when reporting correctness. Report enabled and disabled features explicitly.
- Keep tokens outside the repository. Local Git credentials read `~/.x6_spec_token` through a repository-scoped helper. Never print, commit, embed in a URL, or copy the token into a profiler's environment.
- Large profiler artifacts belong in ignored `profiles/`. Keep small reproducibility manifests and summary statistics in `logs/`.
- The completed overnight report is `reports/2026-11-03/report.html`; `audit_overnight.py` checks the full artifact matrix. Preserve the initial 06:29 report snapshot as deadline provenance.
- Keep native optimized BS=1 results separate from `batch_reference.py` results. The latter performs genuine batched learned drafting and target verification with eager SDPA/dense padded KV, and must not be described as optimized production batching.
- Run `run_overnight.py` only after checking the authoritative supervisor PID; it yields its own GPU process group when foreign compute is detected. Never restart solely because a status read or tool wait timed out.
