Parallel speculative decoding research on RTX 5090

Reports: reports/index.html
JetSpec visualization report: reports/2026-11-04_jetspec_profile.html
Open locally in a browser after cloning/downloading; GitHub's file view displays source.

Measured native JetSpec BS=1 and experimental eager batch reference are separate cohorts.
The native AR baseline is JetSpec's compiled CUDA Graph decoder, not SGLang or vLLM.
BF16 AR/tree token divergence is recorded; no claim of bitwise equality is made.

Rebuild the CPU-only visualization report:
  source env.sh
  python build_jetspec_summary.py

The validated JetSpec environment stays in .venv (requirements-5090.lock).
SGLang has a separate .venv-sglang (requirements-sglang.lock).
Weights stay outside Git in the Hugging Face cache, with aliases under ~/models.
Large .nsys-rep, SQLite and ZIP artifacts remain local and ignored.
See SETUP.txt for hardware, commands, profiler setup and provenance.

New HTML reports use reports/YYYY-MM-DD_topic.html and same-named asset folders.
Prior reports and the initial 2026-11-03 snapshot are preserved.
Push only when explicitly instructed by the user.
