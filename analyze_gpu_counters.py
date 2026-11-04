"""Attribute 1 kHz GPU metrics to GPU execution phases using overlap weights.

Counters average a sampling window. They are not per-kernel Nsight Compute
measurements; boundary windows mix phases and idle periods. Never use the short
CPU NVTX launch interval as if it were the duration of graph execution.
"""
from bisect import bisect_right
from collections import defaultdict
import json
from pathlib import Path
import sqlite3
import statistics

from sweep_worker import save

ROOT=Path(__file__).resolve().parent
PHASES={'AR/decode_forward','JetSpec/draft_forward','JetSpec/verify_forward'}
METRICS=['SMs Active [Throughput %]','SM Issue [Throughput %]','Tensor Active [Throughput %]',
         'DRAM Read Bandwidth [Throughput %]','DRAM Write Bandwidth [Throughput %]',
         'Compute Warps in Flight [Throughput %]']


def merge(intervals):
    output=[]
    for a,b in sorted(intervals):
        if output and a<=output[-1][1]: output[-1]=(output[-1][0],max(output[-1][1],b))
        else: output.append((a,b))
    return output


def analyze(path):
    c=sqlite3.connect(f'file:{Path(path).resolve()}?mode=ro',uri=True)
    ranges=defaultdict(list)
    for start,end,tid,text,text_id in c.execute('SELECT start,end,globalTid,text,textId FROM NVTX_EVENTS WHERE end IS NOT NULL'):
        if text in PHASES: ranges[tid].append((start,end,text))
    for tid in ranges:ranges[tid].sort()
    starts={tid:[r[0] for r in rows] for tid,rows in ranges.items()}
    correlation={}
    for corr,start,tid in c.execute('SELECT correlationId,start,globalTid FROM CUPTI_ACTIVITY_KIND_RUNTIME'):
        if tid not in starts:continue
        i=bisect_right(starts[tid],start)-1
        if i>=0 and start<=ranges[tid][i][1]:correlation[corr]=ranges[tid][i][2]
    intervals=defaultdict(list)
    for start,end,corr in c.execute('SELECT start,end,correlationId FROM CUPTI_ACTIVITY_KIND_KERNEL'):
        if corr in correlation:intervals[correlation[corr]].append((start,end))
    names={mid:name for mid,name in c.execute('SELECT metricId,metricName FROM TARGET_INFO_GPU_METRICS') if name in METRICS}
    values=defaultdict(dict)
    for timestamp,mid,value in c.execute('SELECT timestamp,metricId,value FROM GPU_METRICS WHERE value IS NOT NULL'):
        if mid in names:
            assert 0<=value<=100,(names[mid],value)
            values[timestamp][mid]=value
    times=sorted(values)
    assert len(times)>10
    period=statistics.median(b-a for a,b in zip(times,times[1:]))
    records={}
    for phase,raw in intervals.items():
        sums=defaultdict(float); weights=defaultdict(float); windows=set()
        for a,b in merge(raw):
            j=bisect_right(times,a)
            while j<len(times):
                left=times[j-1] if j else times[0]-period
                right=times[j]
                if left>=b:break
                overlap=max(0,min(b,right)-max(a,left))
                if overlap:
                    windows.add(j)
                    for mid,value in values[right].items():
                        sums[mid]+=overlap*value; weights[mid]+=overlap
                j+=1
        assert weights,phase
        records[phase]={'GPU_kernel_union_ms':sum(b-a for a,b in merge(raw))/1e6,
                       'overlapping_sample_windows':len(windows),
                       'metrics_percent':{names[mid]:sums[mid]/weights[mid] for mid in sums}}
    info=c.execute("SELECT name,totalMemory,memoryBandwidth,smCount,l2CacheSize FROM TARGET_INFO_GPU WHERE name LIKE '%5090%'").fetchone()
    c.close()
    return {'sampling_hz_estimate':1e9/period,'sampling_method':'sampling-window value weighted by overlap with correlation-attributed GPU kernel intervals',
        'warning':'1 kHz windows include phase-boundary mixing and idle periods; these are coarse phase aggregates, not per-kernel counter results.',
        'GPU':{'name':info[0],'memory_bytes':info[1],'peak_memory_bandwidth_bytes_per_s':info[2],'SM_count':info[3],'L2_bytes':info[4]},
        'phases':records,'database':str(Path(path).resolve())}


def main():
    root=ROOT/'logs/gpu_metrics_20261103'
    for path in root.glob('*.json'):
        if path.name in ['plan.json','status.json'] or path.name.endswith(('.progress.json','.analysis.json','.counters.json')):continue
        row=json.loads(path.read_text())
        if row.get('status')=='success' and row.get('nsys_gpu_metrics_enabled'):
            summary=analyze(row['nsys_sqlite'])
            save(path.with_suffix('.counters.json'),summary)
            print(path.stem,{k:{m:round(v,1) for m,v in r['metrics_percent'].items()} for k,r in summary['phases'].items()},flush=True)


if __name__=='__main__':main()
