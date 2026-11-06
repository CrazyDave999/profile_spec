"""Summarize independent MoE benchmark cells; diagnostics never enter gains."""
import _bootstrap
import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import statistics

ROOT=_bootstrap.ROOT
LOGS=ROOT/"logs/moe_sd_20261106"
OUT=ROOT/"reports/2026-11-06_moe_parallel_sd"


def med(values):
    return statistics.median(values) if values else None


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--sequence",default="native01")
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    rows=[];sources=[];correctness=[];states=[];parity_cells=[]
    for campaign_path in sorted(LOGS.glob(args.sequence+"_*.campaign.json")):
        campaign=json.loads(campaign_path.read_text());stage=campaign["stage"]
        states.append({"file":str(campaign_path.relative_to(ROOT)),"stage":stage,"status":campaign["status"],"planned_cells":len(campaign["commands"])})
        if stage not in ("full","width","dense_control"):
            continue
        cells=[]
        for item in campaign["commands"]:
            path=LOGS/(item["name"]+".json");job=LOGS/(item["name"]+".job.json")
            if not path.exists() or not job.exists():continue
            state=json.loads(job.read_text());data=json.loads(path.read_text())
            if state["state"]!="success" or data["status"]!="success":continue
            a=data["arguments"]
            assert not a.get("profile") and not a.get("routing_record")
            cells.append((item["name"],data,path))
        grouped={}
        for name,data,path in cells:
            a=data["arguments"];grouped.setdefault((a["dataset"],a["bs"]),[]).append((name,data,path))
        for (dataset,bs),group in grouped.items():
            baselines=[d for n,d,p in group if d["arguments"]["method"]=="ar"]
            baseline=med([d["median_tokens_per_second"] for d in baselines]) if len(baselines)==2 else None
            drift=abs(baselines[1]["median_tokens_per_second"]/baselines[0]["median_tokens_per_second"]-1) if len(baselines)==2 else None
            for name,data,path in group:
                a=data["arguments"];method=a["method"];rates=[r["tokens_per_second"] for r in data["results"]]
                outputs=[o for r in data["results"] for o in r["outputs"]]
                cycles=[]
                for o in outputs:
                    m=o["meta_info"];steps=m.get("spec_verify_ct",0) if method!="ar" else m["completion_tokens"]-1
                    if steps:cycles.append(1000*(m["e2e_latency"]-m["first_token_latency"])/steps)
                taus=[r["effective_decode_tokens_per_verify_excluding_prefill_seed"] for r in data["results"] if r["effective_decode_tokens_per_verify_excluding_prefill_seed"] is not None]
                row={"case":name,"stage":stage,"target":a["target"],"dataset":dataset,"bs":bs,"method":method,"width":1 if method=="ar" else a["block"],
                    "median_tps":data["median_tokens_per_second"],"gain":data["median_tokens_per_second"]/baseline if baseline else None,
                    "tau":med(taus) if taus else 1.,"request_mean_cycle_median_ms":med(cycles),
                    "peak_physical_GiB":data["physical_device_peak_used_bytes"]/2**30,
                    "repeat_CV":statistics.stdev(rates)/statistics.mean(rates) if len(rates)>1 else None,"AR_bracket_drift":drift,
                    "cached_tokens":sum(o["meta_info"].get("cached_tokens",0) for o in outputs),
                    "retractions":sum(o["meta_info"].get("num_retractions",0) for o in outputs)}
                rows.append(row)
                parity_cells.append({k:row[k] for k in ('case','stage','target','dataset','bs','method','width')}|{'input_sha256':data['selected_prompt_sha256'],'outputs':[o['output_ids'] for o in data['results'][0]['outputs']]})
                sources.append({"case":name,"file":str(path.relative_to(ROOT)),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"worker_sha256":data["worker_source_sha256"]})
                repeats=[r["outputs"] for r in data["results"]]
                assert all(len(r)==len(repeats[0]) for r in repeats)
                repeat_exact=sum(all(r[i]["output_ids"]==repeats[0][i]["output_ids"] for r in repeats[1:]) for i in range(len(repeats[0])))
                comparison={"case":name,"repeat_exact_requests":repeat_exact,"requests":len(repeats[0])}
                if baselines and method!="ar":
                    ar=baselines[0]["results"][0]["outputs"];pairs=[]
                    assert data["selected_prompt_sha256"]==baselines[0]["selected_prompt_sha256"]
                    for i,(left,right) in enumerate(zip(ar,repeats[0])):
                        x,y=left["output_ids"],right["output_ids"]
                        first=next((j for j,(u,v) in enumerate(zip(x,y)) if u!=v),None)
                        pairs.append({"prompt_index":i,"exact":x==y,"first_mismatch":first,"AR_token":x[first] if first is not None else None,"SD_token":y[first] if first is not None else None})
                    comparison.update(AR_exact_requests=sum(p["exact"] for p in pairs),pairs=pairs)
                correctness.append(comparison)
    if rows:
        with (OUT/"benchmarks.csv").open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator="\n");writer.writeheader();writer.writerows(rows)
    (OUT/"source_manifest.json").write_text(json.dumps({"campaigns":states,"cells":sources},indent=2))
    (OUT/"correctness.json").write_text(json.dumps(correctness,indent=2))
    (OUT/"summary.json").write_text(json.dumps({"sequence":args.sequence,"rows":rows,"campaigns":states},indent=2))
    from analyze_moe_tail import analyze as analyze_tails
    analyze_tails(args.sequence)
    from analyze_moe_parity import analyze as analyze_parity
    analyze_parity(parity_cells,OUT)
    print(json.dumps({"successful_benchmark_cells":len(rows),"campaigns":states},indent=2))


if __name__=="__main__":main()
