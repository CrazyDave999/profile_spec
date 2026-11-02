#!/usr/bin/env bash
set -euo pipefail
source /home/qinghao/profile_spec/env.sh
cd "$PROFILE_SPEC_ROOT"
mkdir -p profiles
report="${NSYS_OUTPUT:-$PROFILE_SPEC_ROOT/profiles/jetspec_5090_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)}"
nsys --version
nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --cuda-graph-trace="${NSYS_GRAPH_TRACE:-node}" \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --force-overwrite=true --output="$report" \
  python -u profile_workload.py --capture "$@"
nsys export --type=sqlite --force-overwrite=true --output="$report.sqlite" "$report.nsys-rep"
nsys stats --report cuda_api_sum,cuda_gpu_kern_sum,nvtx_sum --format csv \
  "$report.nsys-rep" > "$report.stats.csv"
python summarize_nsys.py "$report.sqlite" --graph-trace="${NSYS_GRAPH_TRACE:-node}" \
  --output="$report.summary.json"
echo "Report: $report.nsys-rep"
echo "SQLite: $report.sqlite"
echo "Summary: $report.stats.csv"
