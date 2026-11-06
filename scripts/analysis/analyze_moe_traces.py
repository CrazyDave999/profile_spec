"""Export warmed traces, correlate kernels, and retain explicit FP4 evidence."""
import _bootstrap
from bisect import bisect_right
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import subprocess

from analyze_sglang_trace import analyze

ROOT=_bootstrap.ROOT
LOGS=ROOT/"logs/moe_sd_20261106"
PROFILES=ROOT/"profiles/moe_sd_20261106"
OUT=ROOT/"reports/2026-11-06_moe_parallel_sd"


def classification(name):
    lower=name.lower()
    if all(token in name for token in ('cutlass::device_kernel','MainloopSm120','float_e2m1_t','float_e4m3_t','SM120::BLOCKSCALED')):
        return 'native_SM120_MXFP4_GEMM'
    if 'tinygemm_kernel' in lower:
        return 'router_linear'
    if '_router_triton_kernel' in lower:
        return 'router_select'
    if 'computestridestmawarpspecializedkernel' in lower:
        return 'moe_metadata'
    if 'doactivationkernel' in lower:
        return 'moe_activation'
    if 'expandinputrowskernel' in lower:
        return 'moe_dispatch'
    if 'finalizemoeroutingkernel' in lower:
        return 'moe_combine'
    if 'quantize' in lower or 'quantization' in lower:
        return 'activation_quantization'
    if "e2m1" in lower or "mxfp4" in lower or ("sm120" in lower and "fp4" in lower):
        return "FP4_kernel_candidate"
    if "gemm" in lower or "gemv" in lower:
        return "other_GEMM"
    if "flash_fwd" in lower or "fmha_" in lower:
        return "attention_core"
    if any(s in lower for s in ("moe","topk","top_k","routing","permute","quantiz")):
        return "routing_reorder_quantization_candidate"
    return "other"


def kernel_details(database):
    connection=sqlite3.connect(f"file:{database.resolve()}?mode=ro",uri=True)
    strings=dict(connection.execute("SELECT id,value FROM StringIds"))
    steps=defaultdict(list)
    for start,end,tid,text,text_id in connection.execute("SELECT start,end,globalTid,text,textId FROM NVTX_EVENTS WHERE end IS NOT NULL ORDER BY start"):
        name=text or strings.get(text_id,"")
        if name.startswith("step["):
            phase=name[5:].split()[0].rstrip("]");match=re.search(r"bs=(\d+)",name)
            steps[tid].append((start,end,phase,int(match.group(1)) if match else -1))
    starts={tid:[r[0] for r in items] for tid,items in steps.items()}
    correlations={}
    for start,corr,tid in connection.execute("SELECT start,correlationId,globalTid FROM CUPTI_ACTIVITY_KIND_RUNTIME"):
        if tid not in starts:continue
        i=bisect_right(starts[tid],start)-1
        if i>=0 and start<=steps[tid][i][1]:correlations[corr]=steps[tid][i][2:]
    totals=defaultdict(lambda:{"calls":0,"kernel_ms":0.})
    for start,end,corr,name_id in connection.execute("SELECT start,end,correlationId,demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL"):
        phase,bs=correlations.get(corr,("outside_step",-1));name=strings[name_id]
        row=totals[phase,bs,name];row["calls"]+=1;row["kernel_ms"]+=(end-start)/1e6
    connection.close()
    return [dict(phase=phase,observed_bs=bs,category=classification(name),kernel=name,**values)
            for (phase,bs,name),values in totals.items()]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--sequence')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    phase_rows=[];batch_rows=[];kernel_rows=[];counter_rows=[];evidence=[]
    for campaign_path in sorted(LOGS.glob((args.sequence or '*')+"_traces.campaign.json")):
        campaign=json.loads(campaign_path.read_text())
        for cell in campaign["commands"]:
            name=cell["name"];job=LOGS/(name+".job.json")
            if not job.exists() or json.loads(job.read_text())["state"]!="success":continue
            result=json.loads((LOGS/(name+".json")).read_text());a=result["arguments"]
            report=PROFILES/(name+".nsys-rep");database=PROFILES/(name+".sqlite")
            if not database.exists():
                with (LOGS/(name+".export.log")).open("w") as stream:
                    subprocess.run(["nsys","export","--type","sqlite","--output",str(database),str(report)],stdout=stream,stderr=subprocess.STDOUT,check=True)
            output=LOGS/(name+".analysis.json")
            data=analyze(database)
            # The generic analyzer's substring "gemm" heuristic also matches
            # CUTLASS namespaces on MoE setup/activation kernels. Do not export
            # those legacy operator subtotals as actual MoE GEMM measurements.
            for values in data['phases'].values():
                for key in ('gemm_ms','attention_ms','other_kernel_ms'):
                    values.pop(key,None)
            data['operator_attribution']='Legacy namespace-based operator subtotals omitted. Use trace_kernels.csv categories and per-replay validated target_components.csv for actual expert GEMM, router and BF16 projection attribution.'
            output.write_text(json.dumps(data,indent=2))
            info={"case":name,"method":a["method"],"configured_bs":a["bs"],"width":a["block"] if a["method"]!="ar" else 1}
            for phase,values in data["phases"].items():
                if phase in ("DRAFT","VERIFY","DECODE","decode_control"):
                    phase_rows.append({**info,"phase":phase,**values})
            for values in data["by_NVTX_batch_size"]:
                batch_rows.append({**info,**values})
            details=kernel_details(database)
            kernel_rows.extend({**info,**r} for r in details)
            if data["gpu_counters"]:
                for phase,values in data["gpu_counters"]["phases"].items():
                    counter_rows.append({**info,"phase":phase,"sample_windows":values["overlapping_sample_windows"],**values["metrics_percent"]})
            fp4=[r for r in details if r["category"] in ("FP4_kernel_candidate","native_SM120_MXFP4_GEMM") and r["phase"] in ("DECODE","VERIFY")]
            native=[r for r in fp4 if r['category']=='native_SM120_MXFP4_GEMM']
            evidence.append({**info,"nsys_rep":str(report.relative_to(ROOT)),"nsys_rep_SHA256":hashlib.sha256(report.read_bytes()).hexdigest(),
                "analysis":str(output.relative_to(ROOT)),"correlation_coverage":data["correlation_coverage"],"graph_launches":data["cuda_graph_launches"],
                "graph_captures":data["cuda_graph_captures"],"decode_window":data["decode_window"],
                "FP4_named_target_kernels":fp4,
                "verified_native_FP4_target_GEMMs":native,
                "native_FP4_predicate":"Actual target CUDA kernel is cutlass::device_kernel with MainloopSm120, SM120::BLOCKSCALED TiledMMA, float_e2m1_t and float_e4m3_t operands. Full demangled names retained.",
                "FP4_evidence_limit":"Generic FP4-named setup/quantization kernels alone are insufficient; only the stricter native predicate identifies the Tensor Core GEMM."})
    for filename,rows in (("trace_phases.csv",phase_rows),("trace_by_active_bs.csv",batch_rows),("trace_kernels.csv",kernel_rows),("trace_counters.csv",counter_rows)):
        if rows:
            fields=list(dict.fromkeys(k for r in rows for k in r))
            with (OUT/filename).open("w",newline="") as stream:
                writer=csv.DictWriter(stream,fieldnames=fields,lineterminator="\n");writer.writeheader();writer.writerows(rows)
    (OUT/"trace_evidence.json").write_text(json.dumps({"cases":evidence,"limits":["Profile-only timings; host and GPU overlap.","Kernel sums are work, not critical-path latency.","Categories are name heuristics; all original names retained.","1 kHz counters are not achieved FLOP/s or a strict roofline bound."]},indent=2))
    print(json.dumps({"traces":len(evidence),"kernel_groups":len(kernel_rows),"FP4_target_name_candidates":sum(len(r["FP4_named_target_kernels"]) for r in evidence)}))


if __name__=="__main__":main()
