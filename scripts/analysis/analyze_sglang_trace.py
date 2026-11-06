"""CPU-only launch-correlation analysis of SGLang step/scheduler NVTX ranges."""
import _bootstrap
import argparse
from bisect import bisect_right
from collections import defaultdict
import json
from pathlib import Path
import re
import sqlite3
import statistics


def union(intervals):
    end=None;total=0
    for a,b in sorted(intervals):
        if end is None or a>end:total+=b-a;end=b
        elif b>end:total+=b-end;end=b
    return total


def analyze(path):
    c=sqlite3.connect(f'file:{path.resolve()}?mode=ro',uri=True)
    assert c.execute('PRAGMA quick_check').fetchone()[0]=='ok'
    strings=dict(c.execute('SELECT id,value FROM StringIds'))
    steps=defaultdict(list);batches=defaultdict(list);step_metadata={}
    for a,b,tid,t,ti in c.execute('SELECT start,end,globalTid,text,textId FROM NVTX_EVENTS WHERE end IS NOT NULL'):
        name=t or strings.get(ti,'')
        if name.startswith('step['):
            phase=name[5:].split()[0].rstrip(']')
            steps[tid].append((a,b,phase))
            match=re.search(r'bs=(\d+)',name)
            step_metadata[tid,a]={'name':name,'phase':phase,'host_ms':(b-a)/1e6,
                'batch_size':int(match.group(1)) if match else None,'gpu_kernel_ms':0.,'kernel_count':0}
        if name=='scheduler.run_batch':batches[tid].append((a,b))
    for rs in steps.values():rs.sort()
    for rs in batches.values():rs.sort()
    step_starts={t:[r[0] for r in rs] for t,rs in steps.items()};batch_starts={t:[r[0] for r in rs] for t,rs in batches.items()}
    def find(tid,t,groups,starts):
        if tid not in starts:return None
        i=bisect_right(starts[tid],t)-1
        return groups[tid][i] if i>=0 and groups[tid][i][0]<=t<=groups[tid][i][1] else None
    phases=defaultdict(lambda:{'calls':0,'host_ms':0.,'kernel_ms':0.,'kernel_count':0,'graph_kernel_count':0,'graph_launch_count':0,'sync_api_ms':0.,'sync_api_count':0,'memcpy_ms':0.,'gemm_ms':0.,'attention_ms':0.,'other_kernel_ms':0.})
    kinds={};decode_batches=[];batch_info={}
    for tid,rs in steps.items():
        for a,b,p in rs:phases[p]['calls']+=1;phases[p]['host_ms']+=(b-a)/1e6
    for tid,rs in batches.items():
        for a,b in rs:
            children=[s for s in steps.get(tid,[]) if a<=s[0]<=s[1]<=b]
            kind='prefill_control' if any(s[2]=='EXTEND' for s in children) else 'decode_control'
            kinds[tid,a]=kind;phases[kind]['calls']+=1
            batch_info[tid,a]={'thread_id':tid,'start_ns':a,'end_ns':b,'kind':kind,
                'first_kernel_start_ns':None,'last_kernel_end_ns':None,'gpu_kernel_ms':0.,
                'NVTX_batch_sizes':sorted({step_metadata[tid,x]['batch_size'] for x,y,p in children if step_metadata[tid,x]['batch_size'] is not None})}
            phases[kind]['host_ms']+=((b-a)-union([(x,y) for x,y,_ in children]))/1e6
            if kind=='decode_control':decode_batches.append((a,b))
    runtime={};runtime_step={};runtime_batch={};graph=0;captures=0
    for corr,a,b,tid,ni in c.execute('SELECT correlationId,start,end,globalTid,nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME'):
        name=strings.get(ni,'');graph+=int('GraphLaunch' in name);captures+=int('StreamBeginCapture' in name)
        step=find(tid,a,steps,step_starts)
        batch=find(tid,a,batches,batch_starts)
        phase=step[2] if step else (kinds[tid,batch[0]] if batch else 'outside_scheduler')
        runtime[corr]=phase
        if step:runtime_step[corr]=(tid,step[0])
        if batch:runtime_batch[corr]=(tid,batch[0])
        phases[phase]['graph_launch_count']+=int('GraphLaunch' in name)
        if 'Synchronize' in name:phases[phase]['sync_api_ms']+=(b-a)/1e6;phases[phase]['sync_api_count']+=1
    intervals=[];phase_intervals=defaultdict(list);decode_intervals=[];matched=0;total=0;names=defaultdict(lambda:defaultdict(float))
    for a,b,corr,ni,gi in c.execute('SELECT start,end,correlationId,demangledName,graphId FROM CUPTI_ACTIVITY_KIND_KERNEL'):
        total+=1;intervals.append((a,b));phase=runtime.get(corr,'unmatched');matched+=int(corr in runtime)
        phase_intervals[phase].append((a,b))
        m=phases[phase];ms=(b-a)/1e6;m['kernel_ms']+=ms;m['kernel_count']+=1;m['graph_kernel_count']+=int(bool(gi))
        if corr in runtime_step:
            meta=step_metadata[runtime_step[corr]];meta['gpu_kernel_ms']+=ms;meta['kernel_count']+=1
        if corr in runtime_batch:
            batch=batch_info[runtime_batch[corr]];batch['gpu_kernel_ms']+=ms
            batch['first_kernel_start_ns']=min(batch['first_kernel_start_ns'] if batch['first_kernel_start_ns'] is not None else a,a)
            batch['last_kernel_end_ns']=max(batch['last_kernel_end_ns'] or 0,b)
        name=strings.get(ni,'');names[phase][name]+=ms;lower=name.lower()
        # FlashInfer also names norm kernels, and DFlash names accept kernels.
        # Neither is an attention core merely because it contains "flash".
        category='gemm_ms' if 'gemm' in lower or 'gemv' in lower else ('attention_ms' if 'flash_fwd' in lower or 'attention' in lower or 'fmha' in lower else 'other_kernel_ms')
        m[category]+=ms
        if phase in ['DRAFT','VERIFY','DECODE','decode_control']:decode_intervals.append((a,b))
    tables={r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if 'CUPTI_ACTIVITY_KIND_MEMCPY' in tables:
        for a,b,corr in c.execute('SELECT start,end,correlationId FROM CUPTI_ACTIVITY_KIND_MEMCPY'):
            phases[runtime.get(corr,'unmatched')]['memcpy_ms']+=(b-a)/1e6
    counters=None
    if 'GPU_METRICS' in tables and 'TARGET_INFO_GPU_METRICS' in tables:
        metric_names={i:n for i,n in c.execute('SELECT metricId,metricName FROM TARGET_INFO_GPU_METRICS') if n in [
            'SMs Active [Throughput %]','SM Issue [Throughput %]','Tensor Active [Throughput %]',
            'Compute Warps in Flight [Throughput %]','DRAM Read Bandwidth [Throughput %]','DRAM Write Bandwidth [Throughput %]']}
        values=defaultdict(dict)
        for timestamp,mid,value in c.execute('SELECT timestamp,metricId,value FROM GPU_METRICS WHERE value IS NOT NULL'):
            if mid in metric_names:values[timestamp][mid]=value
        times=sorted(values)
        if len(times)>10:
            period=statistics.median(b-a for a,b in zip(times,times[1:]));summaries={}
            for phase,raw in phase_intervals.items():
                merged=[]
                for a,b in sorted(raw):
                    if merged and a<=merged[-1][1]:merged[-1]=(merged[-1][0],max(b,merged[-1][1]))
                    else:merged.append((a,b))
                sums=defaultdict(float);weights=defaultdict(float);windows=set()
                for a,b in merged:
                    j=max(1,bisect_right(times,a))
                    while j<len(times) and times[j-1]<b:
                        overlap=max(0,min(b,times[j])-max(a,times[j-1]))
                        if overlap:
                            windows.add(j)
                            for mid,value in values[times[j]].items():
                                sums[mid]+=overlap*value;weights[mid]+=overlap
                        j+=1
                summaries[phase]={'overlapping_sample_windows':len(windows),
                    'metrics_percent':{metric_names[mid]:sums[mid]/weights[mid] for mid in sums}}
            counters={'sampling_hz_estimate':1e9/period,'phases':summaries,
                'definition':'Sample-window metrics weighted by overlap with launch-correlated GPU kernel intervals. Phase boundaries mix at 1 kHz. SM active/Tensor active are not achieved FLOP/s.'}
    for p,m in phases.items():
        m['gpu_kernel_ms_per_call']=m['kernel_ms']/m['calls'] if m['calls'] else None
        m['host_ms_per_call']=m['host_ms']/m['calls'] if m['calls'] else None
        m['top_kernels']=sorted(names[p].items(),key=lambda v:-v[1])[:5]
    window=None
    if decode_batches and decode_intervals:
        lo=min(a for a,b in decode_batches);hi=max(max(b for a,b in decode_batches),max(b for a,b in decode_intervals))
        ns=hi-lo;busy=union([(max(a,lo),min(b,hi)) for a,b in intervals if b>lo and a<hi])
        interleaved=sum(lo<=a<=hi for rs in steps.values() for a,b,p in rs if p=='EXTEND')
        window={'wall_ms':ns/1e6,'scheduler_decode_rounds':len(decode_batches),'mean_complete_round_ms':ns/1e6/len(decode_batches),'kernel_busy_fraction':busy/ns,'interleaved_prefill_calls':interleaved,'definition':'first decode scheduler.run_batch start to latest decode batch host end/correlated kernel end; includes inter-batch gaps; excludes initial prefill and profiler stop/export, but may include interleaved prefill (see count)'}
        runs=[]
        for tid in batches:
            current=[]
            for batch in sorted((v for (t,a),v in batch_info.items() if t==tid),key=lambda v:v['start_ns']):
                if batch['kind']=='decode_control':current.append(batch)
                elif current:runs.append(current);current=[]
            if current:runs.append(current)
        segments=[(run[0]['start_ns'],max(max(v['end_ns'],v['last_kernel_end_ns'] or 0) for v in run)) for run in runs]
        segmented=union(segments)/1e6
        gpu_segments=[(min(v['first_kernel_start_ns'] for v in run if v['first_kernel_start_ns'] is not None),max(v['last_kernel_end_ns'] for v in run if v['last_kernel_end_ns'] is not None)) for run in runs if any(v['first_kernel_start_ns'] is not None for v in run)]
        gpu_timeline=union(gpu_segments)/1e6
        window.update(decode_run_segments=len(segments),segmented_decode_envelopes_ms=segmented,
            mean_segmented_round_ms=segmented/len(decode_batches),
            mean_gpu_kernel_work_per_round_ms=sum(v['gpu_kernel_ms'] for v in batch_info.values() if v['kind']=='decode_control')/len(decode_batches),
            GPU_timeline_decode_envelopes_ms=gpu_timeline,mean_GPU_timeline_round_ms=gpu_timeline/len(decode_batches),
            GPU_timeline_definition='Union of consecutive decode-run envelopes from first correlated decode GPU kernel start to last decode GPU kernel end, broken at prefill dispatch boundaries. Includes inter-round GPU gaps, excludes initial CPU dispatch and GPU queueing before first decode kernel, and excludes final host response work. Request-derived cycles cover end-to-end decode instead.',
            segmented_definition='Union of consecutive decode-run envelopes, broken at every scheduler prefill batch. Each envelope starts at first CPU dispatch and ends at last correlated GPU kernel/host completion. Excludes between-wave API and prefill dispatch gaps; queueing of preceding GPU work can still overlap its beginning. This is a pipeline envelope, not isolated kernel latency.')
    by_bs=defaultdict(lambda:{'calls':0,'gpu_kernel_ms':0.,'host_ms':0.})
    for meta in step_metadata.values():
        group=by_bs[meta['phase'],meta['batch_size']]
        group['calls']+=1;group['gpu_kernel_ms']+=meta['gpu_kernel_ms'];group['host_ms']+=meta['host_ms']
    conditioned=[]
    for (phase,bs),group in sorted(by_bs.items()):
        conditioned.append(dict(phase=phase,batch_size=bs,**group,
            gpu_kernel_ms_per_call=group['gpu_kernel_ms']/group['calls'],
            host_ms_per_call=group['host_ms']/group['calls']))
    c.close()
    return {'database':str(path),'kernel_events':total,'launch_correlated_kernel_events':matched,'correlation_coverage':matched/max(total,1),'cuda_graph_launches':graph,'cuda_graph_captures':captures,'device_busy_union_ms':union(intervals)/1e6,'phases':dict(phases),'decode_window':window,'by_NVTX_batch_size':conditioned,'steps':list(step_metadata.values()),'scheduler_batches':list(batch_info.values()),'gpu_counters':counters,'scope':'profile-only. CPU and GPU overlap; host includes waits; kernel sum is not critical-path latency; control is outside model.forward, not a named accept-only phase. NVTX batch_size is the model forward descriptor, not necessarily the CUDA graph padded tier.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('database',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=analyze(a.database);a.output.write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items() if k!='phases'},indent=2));print(json.dumps({p:{k:v for k,v in m.items() if k in ['calls','host_ms_per_call','gpu_kernel_ms_per_call','graph_launch_count']} for p,m in r['phases'].items()},indent=2))


if __name__=='__main__':main()
