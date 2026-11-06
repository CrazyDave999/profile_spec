Parallel speculative decoding research on RTX 5090

Reports: reports/index.html
MoE / GPT-OSS20B MXFP4: reports/2026-11-06_moe_parallel_sd.html
SD state machines: reports/2026-11-06_sd_systems_flow.html
JetSpec visualization report: reports/2026-11-04_jetspec_profile.html
Parallel SD overview: reports/2026-11-04_parallel_sd.html
Matched SGLang matrix: reports/2026-11-04_psd_scaling.html
Open locally in a browser after cloning/downloading; GitHub's file view displays source.

Measured native JetSpec BS=1 and experimental eager batch reference are separate cohorts.
The native AR baseline is JetSpec's compiled CUDA Graph decoder, not SGLang or vLLM.
BF16 AR/tree token divergence is recorded; no claim of bitwise equality is made.
The SGLang experiments have their own same-runtime AR baseline. DFlash,
Domino, DSpark and DFlash2 results preserve their checkpoint/width differences.
Main-matrix DSpark uses static verify; adaptive results are a separate cohort.
Fixed output, BF16 token differences, profiling overhead and active-BS tails
are documented in the reports; no task-quality claim is made.

Rebuild the CPU-only visualization report:
  source env.sh
  python scripts/reports/build_jetspec_summary.py

The validated JetSpec environment stays in .venv (requirements-5090.lock).
SGLang has a separate .venv-sglang (requirements-sglang.lock).
Weights stay outside Git in the Hugging Face cache, with aliases under ~/models.
Large .nsys-rep, SQLite and ZIP artifacts remain local and ignored.
See SETUP.txt for hardware, commands, profiler setup and provenance.

Script layout (organized 2026-11-07):
  scripts/experiments/   Workers, benchmark/profile launchers and queue supervisors.
  scripts/analysis/      Trace analysis, numerical checks, audits and packaging.
  scripts/reports/       HTML/SVG/PNG report builders and figure helpers.
  scripts/setup/         Environment setup, model downloads and input preparation.
  env.sh / env_sglang.sh and bench_5090.sh / profile_5090.sh stay at the root.
See scripts/README.txt for current commands. Recorded historical commands and
worker hashes remain unchanged; archived sources preserve the measured code.

New HTML reports use reports/YYYY-MM-DD_topic.html and same-named asset folders.
Prior reports and the initial 2026-11-03 snapshot are preserved.
Push only when explicitly instructed by the user.
