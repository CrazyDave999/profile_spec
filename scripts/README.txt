Research scripts

Run commands from the repository root. Each Python script can be executed
directly; the small shared _bootstrap module resolves repository paths and
imports between the four script folders. Environment isolation is unchanged.

experiments/  GPU workers, experiment matrices, admission guards and supervisors
analysis/     CPU analysis, result audits, numerical comparisons and trace ZIPs
reports/      HTML report builders, SVG state machines and plotting helpers
setup/        Environment setup, pinned model downloads and frozen inputs

Common commands
  source env.sh
  python scripts/reports/build_moe_sd_report.py
  python scripts/reports/build_sd_state_report.py
  python scripts/reports/build_report_index.py
  python scripts/analysis/audit_script_layout.py

  source env_sglang.sh
  python scripts/analysis/analyze_moe_sd.py --sequence native03
  python scripts/analysis/audit_moe_sd_campaign.py
  python scripts/experiments/run_moe_sd_campaign.py --help

  bash scripts/setup/setup_env.sh
  bash scripts/setup/setup_sglang_env.sh
  bash scripts/setup/setup_nsys.sh

Historical evidence
  layout.json maps the previous root filenames to the new paths.
  logs/script_layout_20261107/ preserves the pre-move worker source files and
  an AST comparison proving that the workers' computation was not changed.
  Existing experiment JSON, command lines and worker SHA256 values keep their
  recorded meaning. Use fresh run IDs for new experiments; old campaign guards
  intentionally reject a different source hash or command plan.
  No GPU workload is launched by report rebuilding or layout validation.

Environment and measurement notes
Measured native JetSpec BS=1 and experimental eager batch reference are separate cohorts.
The native AR baseline is JetSpec's compiled CUDA Graph decoder, not SGLang or vLLM.
BF16 AR/tree token divergence is recorded; no claim of bitwise equality is made.
The SGLang experiments have their own same-runtime AR baseline. DFlash,
Domino, DSpark and DFlash2 results preserve their checkpoint/width differences.
Main-matrix DSpark uses static verify; adaptive results are a separate cohort.
Fixed output, BF16 token differences, profiling overhead and active-BS tails
are documented in the reports; no task-quality claim is made.


Pinned dependencies: requirements/5090.txt for .venv (JetSpec), and
requirements/sglang.txt for .venv-sglang. Weights stay outside Git; raw
profiler artifacts remain ignored under profiles/. See SETUP.txt for details.
