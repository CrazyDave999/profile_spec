"""Export completed trace cells and build CPU/GPU breakdown figures, offline."""
import _bootstrap
import argparse
import csv
import json
from bisect import bisect_left
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from analyze_sglang_trace import analyze

ROOT=_bootstrap.ROOT
LOGS=ROOT/'logs/parallel_sd_20261104'
PHASES={'DECODE':'#687787','DRAFT':'#147d92','VERIFY':'#de8a28','decode_control':'#805ea3'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',required=True);args=parser.parse_args()
    campaign=json.loads((LOGS/(args.run_id+'.campaign.json')).read_text())
    rows=[];conditioned=[];source=[];counters=[];padding=[]
    output=ROOT/'reports/2026-11-04_psd_traces';output.mkdir(exist_ok=True)
    for cell in campaign['commands']:
        name=cell['name'];job=LOGS/(name+'.job.json')
        if not job.exists() or json.loads(job.read_text())['state']!='success':continue
        result=json.loads((LOGS/(name+'.json')).read_text());a=result['arguments']
        assert a['profile'],'Instrumented and headline throughput cohorts must stay separate.'
        prefix=ROOT/'profiles/parallel_sd_20261104'/name
        report=prefix.with_suffix('.nsys-rep');database=prefix.with_suffix('.sqlite')
        if not database.exists():
            with (LOGS/(name+'.export.log')).open('w') as log:
                subprocess.run(['nsys','export','--type','sqlite','--output',str(database),str(report)],check=True,stdout=log,stderr=subprocess.STDOUT)
        data=analyze(database);analysis=LOGS/(name+'.analysis.json');analysis.write_text(json.dumps(data,indent=2))
        window=data['decode_window'];assert window and data['correlation_coverage']>.99
        assert data['cuda_graph_captures']==0,'A warmed trace must not contain graph capture.'
        for phase,stats in data['phases'].items():
            if phase not in PHASES:continue
            rows.append(dict(case=name,method=a['method'],configured_bs=a['bs'],phase=phase,
                calls=stats['calls'],host_ms_per_call=stats['host_ms_per_call'],
                gpu_kernel_ms_per_call=stats['gpu_kernel_ms_per_call'],
                calls_per_scheduler_round=stats['calls']/window['scheduler_decode_rounds'],
                GPU_kernel_ms_per_scheduler_round=stats['kernel_ms']/window['scheduler_decode_rounds'],
                host_ms_per_scheduler_round=stats['host_ms']/window['scheduler_decode_rounds'],
                sync_api_ms=stats['sync_api_ms'],sync_api_count=stats['sync_api_count'],
                gemm_ms=stats['gemm_ms'],attention_ms=stats['attention_ms'],other_kernel_ms=stats['other_kernel_ms'],
                complete_decode_window_mean_round_ms=window['mean_complete_round_ms'],
                segmented_decode_mean_round_ms=window['mean_segmented_round_ms'],
                GPU_timeline_mean_round_ms=window['mean_GPU_timeline_round_ms'],
                GPU_kernel_work_per_round_ms=window['mean_gpu_kernel_work_per_round_ms'],
                kernel_busy_fraction=window['kernel_busy_fraction'],interleaved_prefill_calls=window['interleaved_prefill_calls']))
        for stats in data['by_NVTX_batch_size']:
            conditioned.append(dict(case=name,method=a['method'],configured_bs=a['bs'],**stats))
        if a['bs']==16:
            phase='DECODE' if a['method']=='ar' else 'VERIFY'
            buckets=result['engine_resolved_after']['cuda_graph_bs_decode']
            distribution=[r for r in data['by_NVTX_batch_size'] if r['phase']==phase]
            count=sum(r['calls'] for r in distribution)
            actual=sum(r['batch_size']*r['calls'] for r in distribution)
            padded=sum(buckets[bisect_left(buckets,r['batch_size'])]*r['calls'] for r in distribution)
            padding.append(dict(case=name,method=a['method'],calls=count,
                full_BS16_rounds=sum(r['calls'] for r in distribution if r['batch_size']==16),
                mean_observed_bs=actual/count,actual_request_rows=actual,inferred_graph_request_rows=padded,
                actual_over_inferred_padded_rows=actual/padded,
                note='Graph request rows inferred by ceil to configured capture buckets, matching pinned runner _pad_to_bucket; not a direct graph-key trace or hardware utilization metric.'))
        if data['gpu_counters']:
            for phase,stats in data['gpu_counters']['phases'].items():
                counters.append(dict(case=name,method=a['method'],configured_bs=a['bs'],phase=phase,
                    overlapping_sample_windows=stats['overlapping_sample_windows'],**stats['metrics_percent']))
        source.append(dict(case=name,method=a['method'],bs=a['bs'],report=str(report.relative_to(ROOT)),
            report_bytes=report.stat().st_size,analysis=str(analysis.relative_to(ROOT)),
            graph_launches=data['cuda_graph_launches'],graph_captures=data['cuda_graph_captures'],
            correlation_coverage=data['correlation_coverage'],decode_window=window))
    assert rows,'No successful trace cells yet.'
    for filename,values in [('phase_breakdown.csv',rows),('by_active_batch.csv',conditioned)]:
        with (output/filename).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    if counters:
        # Stable columns even when a very short phase has no overlapping sample.
        fields=list(dict.fromkeys(k for r in counters for k in r))
        with (output/'gpu_counters.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(counters)
    if padding:
        with (output/'graph_padding.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(padding[0]));writer.writeheader();writer.writerows(padding)
    (output/'manifest.json').write_text(json.dumps(dict(campaign=campaign,cells=source,
        caveats=['All timings are instrumented, not headline throughput.',
          'Control means scheduler work outside model forward; not accept-only.',
          'Host and GPU overlap; never add the two charts.',
          'Mean round includes varying active batch and tail; use by_active_batch.csv for fixed observed BS.',
          'High GPU busy is not SM utilization and does not prove compute bound.',
          'GEMM and attention classifications use kernel-name heuristics.']),indent=2))
    plt.rcParams.update({'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    cases=[s['case'] for s in source];labels=[s['method']+' / BS'+str(s['bs']) for s in source]
    fig,axes=plt.subplots(1,2,figsize=(12,max(4,len(cases)*.48)))
    for ax,field,title in zip(axes,['GPU_kernel_ms_per_scheduler_round','host_ms_per_scheduler_round'],['Mean GPU kernel work / scheduler round','Mean host NVTX / scheduler round']):
        left=[0.]*len(cases)
        for phase,color in PHASES.items():
            values=[sum(r[field] or 0 for r in rows if r['case']==case and r['phase']==phase) for case in cases]
            ax.barh(range(len(cases)),values,left=left,color=color,label=phase);left=[a+b for a,b in zip(left,values)]
        ax.set_yticks(range(len(cases)),labels);ax.invert_yaxis();ax.set_xlabel('ms (profile only)');ax.set_title(title);ax.grid(axis='x',alpha=.2)
    axes[0].legend(ncol=2,fontsize=8);fig.tight_layout()
    fig.savefig(output/'phase_breakdown.svg',bbox_inches='tight');fig.savefig(output/'phase_breakdown.png',bbox_inches='tight',dpi=160);plt.close(fig)
    if any(r['configured_bs']==16 and r['phase']=='VERIFY' for r in conditioned):
        colors={'dflash':'#147d92','domino':'#458b50','dspark':'#805ea3','dflash2':'#de8a28'}
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        for ax,phase in zip(axes,['DRAFT','VERIFY']):
            for method,color in colors.items():
                points=sorted((r for r in conditioned if r['method']==method and r['configured_bs']==16 and r['phase']==phase),key=lambda r:r['batch_size'])
                if not points:continue
                ax.plot([r['batch_size'] for r in points],[r['gpu_kernel_ms_per_call'] for r in points],'o-',label=method,color=color)
            ax.set_xticks([1,2,4,8,9,16]);ax.set_xlabel('Observed NVTX batch size');ax.set_ylabel('GPU kernel ms / forward call');ax.set_title(phase+' / configured max BS16');ax.grid(alpha=.2);ax.set_ylim(bottom=0)
        axes[0].legend();fig.tight_layout();fig.savefig(output/'active_batch_cost.svg',bbox_inches='tight');fig.savefig(output/'active_batch_cost.png',bbox_inches='tight',dpi=160);plt.close(fig)
    if counters:
        selected=[r for r in counters if r['phase'] in ['DECODE','DRAFT','VERIFY']]
        fig,axes=plt.subplots(1,3,figsize=(14,max(4,.28*len(selected))))
        for ax,metric in zip(axes,['SMs Active [Throughput %]','Tensor Active [Throughput %]','DRAM Read Bandwidth [Throughput %]']):
            ax.barh(range(len(selected)),[r.get(metric,0) for r in selected],color=[PHASES[r['phase']] for r in selected])
            ax.set_yticks(range(len(selected)),[f"{r['method']} BS{r['configured_bs']} {r['phase']}" for r in selected]);ax.invert_yaxis();ax.set_xlim(0,100);ax.set_title(metric);ax.grid(axis='x',alpha=.2)
        fig.tight_layout();fig.savefig(output/'gpu_counters.svg',bbox_inches='tight');fig.savefig(output/'gpu_counters.png',bbox_inches='tight',dpi=160);plt.close(fig)
    print('Analyzed',len(source),'completed trace cells in',output)


if __name__=='__main__':main()
