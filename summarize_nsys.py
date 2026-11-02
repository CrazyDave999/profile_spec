"""Summarize and verify CUDA Graph/NVTX evidence in an nsys SQLite export."""
import argparse
import json
from pathlib import Path
import sqlite3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("database", type=Path)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--graph-trace", choices=["node", "graph"], default="node")
    args = parser.parse_args()
    with sqlite3.connect(f"file:{args.database.resolve()}?mode=ro", uri=True) as con:
        runtime = dict(con.execute(
            "SELECT s.value, COUNT(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
            "JOIN StringIds s ON r.nameId=s.id GROUP BY s.value"))
        nvtx = dict(con.execute(
            "SELECT COALESCE(n.text,s.value),COUNT(*) FROM NVTX_EVENTS n "
            "LEFT JOIN StringIds s ON n.textId=s.id GROUP BY COALESCE(n.text,s.value)"))
        total_kernels, graph_kernels = con.execute(
            "SELECT COUNT(*),SUM(CASE WHEN graphId IS NOT NULL AND graphId!=0 THEN 1 ELSE 0 END) "
            "FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()
        graph_calls = sum(v for k, v in runtime.items() if "GraphLaunch" in k)
        graph_capture_calls = sum(v for k, v in runtime.items() if "StreamBeginCapture" in k)
        report = {"database": str(args.database.resolve()),
                  "cuda_graph_launch_calls": graph_calls,
                  "cuda_graph_capture_calls_in_measured_region": graph_capture_calls,
                  "gpu_kernel_events": total_kernels, "gpu_graph_kernel_events": graph_kernels,
                  "nvtx_event_count": sum(nvtx.values()),
                  "named_nvtx_counts": {k: v for k, v in nvtx.items() if k},
                  "capture_scope": "warmed AR and JetSpec requests only",
                  "graph_trace_granularity": args.graph_trace}
    assert graph_calls > 0, report
    if args.graph_trace == "node":
        assert graph_kernels > 0 and total_kernels > 0, report
    assert graph_capture_calls == 0, report
    if args.audit:
        counters = json.loads(args.audit.read_text())["measured_counts"]
        expected = sum(counters.get(k, 0) for k in ["AR/graph_replay", "JetSpec/draft_graph_replay",
                                                   "JetSpec/target_verify_graph_replay"])
        report["expected_cuda_graph_launch_calls_from_python_audit"] = expected
        assert graph_calls == expected, (graph_calls, expected)
    result = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result)
    print(result, end="")


if __name__ == "__main__":
    main()
