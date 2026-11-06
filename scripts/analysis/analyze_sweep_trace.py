"""Attribute CUDA kernels and copies to leaf NVTX phases through launch correlation.

GPU kernel work is summed device execution time, not CPU NVTX duration. CPU
launch timestamps choose the phase; device execution may finish after NVTX pop.
"""
import _bootstrap
import argparse
from bisect import bisect_right
from collections import defaultdict
import json
from pathlib import Path
import sqlite3

from sweep_worker import save

PHASES = {'AR/prefill','AR/decode_forward','AR/kv_pack','AR/target_prepare','AR/accept','AR/kv_commit',
    'JetSpec/prefill','JetSpec/draft_prepare','JetSpec/draft_forward','JetSpec/tree_build','JetSpec/kv_pack',
    'JetSpec/target_prepare','JetSpec/verify_forward','JetSpec/accept','JetSpec/kv_commit'}


def union_ns(intervals):
    total, left, right = 0, None, None
    for a,b in sorted(intervals):
        if left is None: left,right = a,b
        elif a <= right: right=max(right,b)
        else: total += right-left; left,right=a,b
    return total + (right-left if left is not None else 0)


def analyze(path):
    con = sqlite3.connect(f'file:{Path(path).resolve()}?mode=ro',uri=True)
    strings = dict(con.execute('SELECT id,value FROM StringIds'))
    ranges = defaultdict(list)
    requests = defaultdict(list)
    phases = defaultdict(lambda: {'calls':0,'host_ms':0.,'kernel_ms':0.,'memcpy_ms':0.,'memcpy_bytes':0,
                                  'kernel_count':0,'graph_kernel_count':0,'launch_configs':defaultdict(int)})
    per_interval = defaultdict(list)
    kernel_names = defaultdict(lambda:defaultdict(lambda:[0,0.]))
    for start,end,tid,text,text_id in con.execute('SELECT start,end,globalTid,text,textId FROM NVTX_EVENTS WHERE end IS NOT NULL'):
        name = text or strings.get(text_id)
        if name in PHASES:
            ident=len(ranges[tid])
            ranges[tid].append((start,end,name,ident))
            phases[name]['calls']+=1
            phases[name]['host_ms']+=(end-start)/1e6
        elif name and name.startswith('profile_spec/') and len(name.split('/')) >= 3:
            method = name.split('/')[1]
            if method in ('AR', 'JetSpec'):
                requests[tid].append((start,end,method+'/other',len(requests[tid])))
    for tid in ranges: ranges[tid].sort()
    for tid in requests:
        requests[tid].sort()
        for start,end,phase,ident in requests[tid]:
            phases[phase]['calls'] += 1
            leaf_ns = sum(b-a for a,b,_,_ in ranges[tid] if start <= a <= b <= end)
            phases[phase]['host_ms'] += max(0,end-start-leaf_ns)/1e6
    starts={tid:[r[0] for r in rows] for tid,rows in ranges.items()}
    request_starts={tid:[r[0] for r in rows] for tid,rows in requests.items()}
    runtime={}
    graph_launches=0
    capture_calls=0
    for corr,start,tid,name_id in con.execute('SELECT correlationId,start,globalTid,nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME'):
        name=strings.get(name_id,'')
        graph_launches += 'GraphLaunch' in name
        capture_calls += 'StreamBeginCapture' in name
        attributed = False
        if tid in starts:
            idx=bisect_right(starts[tid],start)-1
            if idx>=0:
                a,b,phase,ident=ranges[tid][idx]
                if a<=start<=b:
                    runtime[corr]=(phase,(tid,ident)); attributed=True
        if not attributed and tid in request_starts:
            idx=bisect_right(request_starts[tid],start)-1
            if idx>=0:
                a,b,phase,ident=requests[tid][idx]
                if a<=start<=b: runtime[corr]=(phase,('request',tid,ident))
    total=matched=0
    gpu_intervals=[]
    for start,end,corr,name_id,gx,gy,gz,bx,by,bz,regs,shared,graph_id in con.execute(
        'SELECT start,end,correlationId,demangledName,gridX,gridY,gridZ,blockX,blockY,blockZ,'
        'registersPerThread,staticSharedMemory,graphId FROM CUPTI_ACTIVITY_KIND_KERNEL'):
        total+=1
        gpu_intervals.append((start,end))
        item=runtime.get(corr)
        if not item: continue
        matched+=1
        phase,ident=item
        ms=(end-start)/1e6
        phases[phase]['kernel_ms']+=ms
        phases[phase]['kernel_count']+=1
        phases[phase]['graph_kernel_count']+=bool(graph_id)
        cfg=f'grid={gx}x{gy}x{gz}; block={bx}x{by}x{bz}; regs={regs}; static_smem={shared}'
        phases[phase]['launch_configs'][cfg]+=1
        per_interval[ident].append((start,end))
        name=strings.get(name_id,str(name_id))
        kernel_names[phase][name][0]+=1
        kernel_names[phase][name][1]+=ms
    tables={r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if 'CUPTI_ACTIVITY_KIND_MEMCPY' in tables:
        for start,end,corr,nbytes in con.execute('SELECT start,end,correlationId,bytes FROM CUPTI_ACTIVITY_KIND_MEMCPY'):
            item=runtime.get(corr)
            if item:
                phase,ident=item
                phases[phase]['memcpy_ms']+=(end-start)/1e6
                phases[phase]['memcpy_bytes']+=nbytes
                per_interval[ident].append((start,end))
    for phase,values in phases.items():
        values['top_launch_configs']=sorted(values.pop('launch_configs').items(),key=lambda x:-x[1])[:6]
        values['top_kernels']=[{'name':k,'count':v[0],'kernel_ms':v[1]}
            for k,v in sorted(kernel_names[phase].items(),key=lambda x:-x[1][1])[:8]]
        values['mean_kernel_ms_per_phase_call']=values['kernel_ms']/max(values['calls'],1)
        spans=[]
        for tid,rows in ranges.items():
            for a,b,name,ident in rows:
                if name!=phase: continue
                intervals=per_interval.get((tid,ident),[])
                if intervals: spans.append((max(x[1] for x in intervals)-min(x[0] for x in intervals))/1e6)
        values['summed_gpu_envelope_ms']=sum(spans)
    lo=min((a for a,b in gpu_intervals),default=0)
    hi=max((b for a,b in gpu_intervals),default=0)
    memory_events=con.execute('SELECT COUNT(*) FROM CUDA_GPU_MEMORY_USAGE_EVENTS').fetchone()[0] if 'CUDA_GPU_MEMORY_USAGE_EVENTS' in tables else 0
    con.close()
    return {'analysis_version':2,'database':str(Path(path).resolve()),'kernel_events':total,'attributed_kernel_events':matched,
        'kernel_attribution_fraction':matched/max(total,1),'graph_launch_calls':graph_launches,
        'graph_capture_calls_in_capture_range':capture_calls,'cuda_memory_usage_events':memory_events,
        'gpu_kernel_busy_union_ms':union_ns(gpu_intervals)/1e6,
        'gpu_kernel_envelope_ms':(hi-lo)/1e6,'phases':dict(phases),
        'methodology':'CPU launch correlation selects leaf NVTX phase; device start/end determine GPU kernel work; summed kernel work may double-count concurrent streams.'}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('database',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    result=analyze(args.database)
    save(args.output,result)
    print(json.dumps({k:v for k,v in result.items() if k!='phases'},indent=2))


if __name__=='__main__': main()
