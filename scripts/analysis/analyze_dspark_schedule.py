"""Verify whether adaptive scheduling changes the replayed target work."""
import _bootstrap
import argparse
from bisect import bisect_left
import csv
import hashlib
import json
from pathlib import Path
import statistics

ROOT=_bootstrap.ROOT
LOGS=ROOT/'logs/parallel_sd_20261104'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',required=True);args=parser.parse_args()
    campaign=json.loads((LOGS/(args.run_id+'.campaign.json')).read_text());assert campaign['stage']=='adaptive'
    rows=[];steps=[];sources={}
    for cell in campaign['commands']:
        job=LOGS/(cell['name']+'.job.json');path=LOGS/(cell['name']+'.json')
        if not job.exists() or json.loads(job.read_text())['state']!='success':continue
        data=json.loads(path.read_text());a=data['arguments']
        if a['method']=='ar':continue
        assert data['status']=='success' and a['sps_table']
        table=Path(a['sps_table']);assert table.is_file()
        state=data['engine_resolved_after']['internal_states'][0];payload=state['dspark_info_record']
        assert payload.get('simulate_acc_len') in [None,0,-1]
        assert payload['mode']==a['ragged_mode']
        records=payload['records'];assert records
        width=data['engine_resolved_after']['speculative_num_draft_tokens'];buckets=data['engine_resolved_after']['cuda_graph_bs_decode']
        nominal=0;graph_rows=0;uniform_padded=0;trimmed=0;by_bs={}
        for r in records:
            bs=r['num_running_reqs'];reported=r['num_verify_tokens']
            if bs<1 or reported<1:continue
            static=width*buckets[bisect_left(buckets,bs)]
            # The upstream observer reports verify_ids_2d.numel() in static
            # mode (raw rows), but layout.graph_num_tokens in compact mode.
            # Despite the field's name, only the latter is a padded graph key.
            graph=static if a['ragged_mode']=='static' else reported
            budget=bs*width if r.get('budget') is None else bs+r['budget']
            nominal+=bs*width;graph_rows+=graph;uniform_padded+=static;trimmed+=int(graph<static)
            key=str(bs);by_bs.setdefault(key,[]).append(graph)
            steps.append(dict(case=cell['name'],mode=a['ragged_mode'],configured_bs=a['bs'],
                forward_ct=r['forward_ct'],observed_bs=bs,executed_budget_with_anchors=budget,
                reported_num_verify_tokens=reported,replayed_graph_rows=graph,uniform_static_padded_rows=static,
                predicted_step_ms=r.get('predicted_step_ms'),recorded_step_cpu_ms=r.get('step_cpu_ms')))
        rows.append(dict(case=cell['name'],mode=a['ragged_mode'],configured_bs=a['bs'],records=len(records),
            replayed_graph_rows_sum=graph_rows,uniform_static_padded_rows_sum=uniform_padded,
            graph_rows_relative_to_uniform_static=graph_rows/uniform_padded,
            graph_trimmed_calls=trimmed,mean_graph_rows_per_call=graph_rows/len(records),
            by_observed_bs={bs:dict(calls=len(values),mean_graph_rows=statistics.mean(values),min_graph_rows=min(values),max_graph_rows=max(values)) for bs,values in by_bs.items()},
            full_BS_predicted_step_ms_median=statistics.median([r['predicted_step_ms'] for r in records if r.get('num_running_reqs')==a['bs'] and r.get('predicted_step_ms') is not None]) if any(r.get('num_running_reqs')==a['bs'] and r.get('predicted_step_ms') is not None for r in records) else None,
            full_BS_recorded_step_cpu_ms_median=statistics.median([r['step_cpu_ms'] for r in records if r.get('num_running_reqs')==a['bs'] and r.get('step_cpu_ms') is not None]) if any(r.get('num_running_reqs')==a['bs'] and r.get('step_cpu_ms') is not None for r in records) else None,
            table_SHA256=hashlib.sha256(table.read_bytes()).hexdigest(),
            graph_rows_source='Inferred from capture buckets for static; reported layout.graph_num_tokens for compact.',
            observability='core + CPU step intervals enabled equally for static and compact; no per-request confidence D2H or GPU event timers.',
            scope='Records include warmup and measured generations. They establish scheduling behavior, not headline throughput; returned request timing excludes warmup.'))
        sources[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    assert rows,'No successful adaptive DSpark cells yet.'
    out=ROOT/'reports/2026-11-04_parallel_sd'
    stem='dspark_schedule_'+args.run_id
    (out/(stem+'.json')).write_text(json.dumps(dict(run_id=args.run_id,campaign_status=campaign['status'],cases=rows,source_sha256=sources,analyzer_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        definition='Compact num_verify_tokens is layout.graph_num_tokens. Static num_verify_tokens is raw verify_ids_2d.numel(), so static graph rows are inferred by ceil-to-capture-BS times width. Recorded budget is the executed verify budget after graph-tier fill, not the optimizer argmax before alignment; it is not a hardware utilization metric.'),indent=2))
    with (out/(stem+'_steps.csv')).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(steps[0]));writer.writeheader();writer.writerows(steps)
    print([{k:v for k,v in r.items() if k in ['case','mode','configured_bs','graph_rows_relative_to_uniform_static','graph_trimmed_calls']} for r in rows])


if __name__=='__main__':main()
