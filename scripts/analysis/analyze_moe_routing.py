"""CPU analysis of upstream per-pass expert counts, separate from timing arms."""
import _bootstrap
from collections import defaultdict
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT=_bootstrap.ROOT
LOGS=ROOT/"logs/moe_sd_20261106"
OUT=ROOT/"reports/2026-11-06_moe_parallel_sd"


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--sequence')
    args=parser.parse_args()
    import torch
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((ROOT/"models-moe-sd.json").read_text())["checkpoints"]["target"]
    expert_bytes=defaultdict(int)
    for file in manifest["weights"]:
        for name,tensor in file["tensors"].items():
            if ".mlp.experts." in name:
                layer=int(name.split(".")[2]);shape=tensor["shape"]
                assert shape[0]==32
                expert_bytes[layer]+=math.prod(shape)//32*({"U8":1,"BF16":2}[tensor["dtype"]])
    assert len(expert_bytes)==24
    rows=[];sources=[];trajectory_checks=[]
    for job in sorted(LOGS.glob((args.sequence or '*')+"_routing_*.job.json")):
        if json.loads(job.read_text())["state"]!="success":continue
        result=job.with_name(job.name.replace(".job.json",".json"))
        data=json.loads(result.read_text());a=data["arguments"]
        assert a["routing_record"]
        # Recording is not timed as a benchmark. Check separately whether it
        # preserves the output trajectory of the matched warmed trace arm.
        peers=[]
        for peer_job in LOGS.glob((args.sequence or '*')+'_traces_*.job.json'):
            if json.loads(peer_job.read_text())['state']!='success':continue
            peer_path=peer_job.with_name(peer_job.name.replace('.job.json','.json'))
            peer=json.loads(peer_path.read_text());pa=peer['arguments']
            if all(pa[k]==a[k] for k in ('target','method','dataset','bs','block','samples','tokens')):
                assert peer['selected_prompt_sha256']==data['selected_prompt_sha256']
                left,right=peer['results'][0]['outputs'],data['results'][0]['outputs']
                assert len(left)==len(right)
                peers.append({'reference':str(peer_path.relative_to(ROOT)),'reference_SHA256':hashlib.sha256(peer_path.read_bytes()).hexdigest(),
                    'requests':len(left),'exact_requests':sum(x['output_ids']==y['output_ids'] for x,y in zip(left,right))})
        trajectory_checks.append({'case':result.stem,'matched_trace_comparisons':peers,'note':'Width4 has no matched nsys arm in the planned matrix.' if not peers else ''})
        for filename in data["expert_distribution_files"]:
            path=Path(filename);dump=torch.load(path,map_location="cpu",weights_only=True)
            sources.append({"case":result.stem,"file":str(path.relative_to(ROOT)),"SHA256":hashlib.sha256(path.read_bytes()).hexdigest(),"records":len(dump["records"])})
            for record in dump["records"]:
                meta=record["profile_spec_metadata"]
                counts=record.get("global_physical_count")
                if counts is None:continue
                assert list(counts.shape)==[24,32]
                for layer,values in enumerate(counts.tolist()):
                    assignments=sum(values)
                    if not assignments:continue
                    assert assignments%4==0
                    tokens=assignments//4;unique=sum(v>0 for v in values)
                    rows.append({"case":result.stem,"method":a["method"],"configured_bs":a["bs"],"width":a["block"] if a["method"]!="ar" else 1,
                        "forward_pass_id":record["forward_pass_id"],"phase":meta["forward_mode"],"input_token_rows":meta["input_token_rows"],"request_rows":meta["request_rows"],
                        "layer":layer,"counted_routed_token_rows":tokens,"unique_experts":unique,"assignments":assignments,
                        "mean_assignments_per_used_expert":assignments/unique,"max_expert_assignments":max(values),
                        "uniform_independent_expected_unique":32*(1-(1-4/32)**tokens),
                        "checkpoint_unique_expert_bytes_proxy":unique*expert_bytes[layer]})
    groups=defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in ("case","method","configured_bs","width","phase","request_rows","counted_routed_token_rows"))].append(row)
    summary=[]
    for key,values in groups.items():
        case,method,bs,width,phase,requests,tokens=key
        summary.append({"case":case,"method":method,"configured_bs":bs,"width":width,"phase":phase,"request_rows":requests,"routed_token_rows":tokens,"layer_observations":len(values),
            "mean_unique_experts":statistics.mean(r["unique_experts"] for r in values),
            "mean_assignments_per_used_expert":statistics.mean(r["mean_assignments_per_used_expert"] for r in values),
            "mean_max_expert_assignments":statistics.mean(r["max_expert_assignments"] for r in values),
            "mean_checkpoint_expert_bytes_proxy_per_layer":statistics.mean(r["checkpoint_unique_expert_bytes_proxy"] for r in values)})
    for name,records in (("routing_layers.csv",rows),("routing_summary.csv",summary)):
        if records:
            with (OUT/name).open("w",newline="") as stream:
                writer=csv.DictWriter(stream,fieldnames=list(records[0]),lineterminator="\n");writer.writeheader();writer.writerows(records)
    evidence={"sources":sources,"trajectory_checks":trajectory_checks,"expert_bytes_per_layer_per_expert":dict(expert_bytes),
        "diagnostic":"Upstream per-pass recording enabled after warmup, disabled after measurement. Recording timings excluded from headline.",
        "limits":["Counts describe routed assignments; compare with input/query rows to detect graph padding or masked routes.","Distinct checkpoint expert bytes are a logical access-footprint proxy, not measured HBM traffic; runtime padding, cache reuse, repeated GEMM reads, bias handling and kernel tiling differ.","Uniform independent routing is an explanatory reference, not an assumption imposed on measured routes."]}
    (OUT/"routing_evidence.json").write_text(json.dumps(evidence,indent=2))
    print(json.dumps({"diagnostic_dumps":len(sources),"layer_observations":len(rows),"summary_groups":len(summary)}))


if __name__=="__main__":main()
