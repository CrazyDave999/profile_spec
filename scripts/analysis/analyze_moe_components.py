"""Recover target components with per-replay operator-count validation."""
import _bootstrap
import argparse
from bisect import bisect_right
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import sqlite3

from analyze_moe_traces import classification

ROOT=_bootstrap.ROOT
LOGS=ROOT/'logs/moe_sd_20261106'
PROFILES=ROOT/'profiles/moe_sd_20261106'
OUT=ROOT/'reports/2026-11-06_moe_parallel_sd'
PARTS=['MoE_FP4_GEMM','router_linear','router_select','moe_metadata','moe_dispatch','moe_activation','moe_combine','activation_quantization','QKV_O_projection','LM_head','attention_core','other']


def analyze(path):
    c=sqlite3.connect(f'file:{path.resolve()}?mode=ro',uri=True)
    strings=dict(c.execute('SELECT id,value FROM StringIds'))
    by_thread=defaultdict(list);rounds=[]
    for a,b,tid,text,text_id in c.execute('SELECT start,end,globalTid,text,textId FROM NVTX_EVENTS WHERE end IS NOT NULL ORDER BY start'):
        name=text or strings.get(text_id,'')
        if name.startswith(('step[DECODE ','step[VERIFY ')):
            match=re.search(r'bs=(\d+)',name);assert match,name
            idx=len(rounds);rounds.append({'index':idx,'phase':name[5:].split()[0],'active_bs':int(match.group(1)),'kernels':[]})
            by_thread[tid].append((a,b,idx))
    starts={tid:[r[0] for r in items] for tid,items in by_thread.items()};correlations={}
    for a,corr,tid in c.execute('SELECT start,correlationId,globalTid FROM CUPTI_ACTIVITY_KIND_RUNTIME'):
        if tid not in starts:continue
        i=bisect_right(starts[tid],a)-1
        if i>=0 and a<=by_thread[tid][i][1]:correlations[corr]=by_thread[tid][i][2]
    for a,b,corr,name_id,gx,gy,gz in c.execute('SELECT start,end,correlationId,demangledName,gridX,gridY,gridZ FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start'):
        if corr in correlations:
            rounds[correlations[corr]]['kernels'].append({'name':strings[name_id],'ms':(b-a)/1e6,'grid':[gx,gy,gz]})
    c.close()
    for row in rounds:
        kernels=row['kernels'];parts=[classification(k['name']) for k in kernels]
        linears=[]
        for i,k in enumerate(kernels):
            low=k['name'].lower()
            if parts[i]=='other_GEMM' and ('bf16' in low or 'bfloat16' in low):linears.append(i)
        assert len(linears)==49,(path.name,row['index'],'BF16 QKV/O/head',len(linears))
        assert sum(p=='native_SM120_MXFP4_GEMM' for p in parts)==48,(path.name,row['index'],'native GEMMs')
        assert sum(p=='router_linear' for p in parts)==24,(path.name,row['index'],'router GEMMs')
        assert sum(p=='router_select' for p in parts)==24,(path.name,row['index'],'router selections')
        # Same projection role must have the same launch signature in each of
        # the 24 homogeneous layers; the final 49th linear is the LM head.
        for layer in range(24):
            for role in range(2):
                a,b=kernels[linears[role]],kernels[linears[2*layer+role]]
                assert (a['name'],a['grid'])==(b['name'],b['grid']),(path.name,row['index'],layer,role)
        for i in linears[:-1]:parts[i]='QKV_O_projection'
        parts[linears[-1]]='LM_head'
        mapped=['MoE_FP4_GEMM' if p=='native_SM120_MXFP4_GEMM' else p if p in PARTS else 'other' for p in parts]
        row['components_ms']={p:sum(k['ms'] for k,label in zip(kernels,mapped) if label==p) for p in PARTS}
        row['total_ms']=sum(k['ms'] for k in kernels)
        row['validated_BF16_linears']=len(linears)
        row['validated_FP4_GEMMs']=48
        del row['kernels']
    return rounds


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--sequence',required=True);args=parser.parse_args()
    rows=[];sources=[]
    campaign=LOGS/(args.sequence+'_traces.campaign.json')
    if not campaign.exists():return
    for cell in json.loads(campaign.read_text())['commands']:
        name=cell['name'];job=LOGS/(name+'.job.json');database=PROFILES/(name+'.sqlite')
        if not job.exists() or not database.exists() or json.loads(job.read_text())['state']!='success':continue
        data=json.loads((LOGS/(name+'.json')).read_text());a=data['arguments'];rounds=analyze(database)
        assert rounds,name
        for scope,selected in [('all_observed_rounds',rounds),('full_active_batch',[r for r in rounds if r['active_bs']==a['bs']])]:
            assert selected,(name,scope)
            rows.append({'case':name,'method':a['method'],'configured_bs':a['bs'],'width':a['block'] if a['method']!='ar' else 1,'scope':scope,'calls':len(selected),
                'total_ms':sum(r['total_ms'] for r in selected)/len(selected),**{p+'_ms':sum(r['components_ms'][p] for r in selected)/len(selected) for p in PARTS}})
        sources.append({'case':name,'database':str(database.relative_to(ROOT)),'validated_rounds':len(rounds),'rounds':rounds})
    OUT.mkdir(parents=True,exist_ok=True)
    if rows:
        path=OUT/'target_components.csv';temp=path.with_suffix('.csv.tmp')
        with temp.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
        temp.replace(path)
    evidence={'cases':sources,'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['scripts/analysis/analyze_moe_components.py','scripts/analysis/analyze_moe_traces.py','third_party/SGLang/python/sglang/srt/models/gpt_oss.py','third_party/SGLang/python/sglang/srt/layers/quantization/mxfp4.py']},
        'scope':'Summed target GPU kernel work per forward, with per-replay validation of 49 BF16 projection/head linears, 48 native FP4 MoE GEMMs, 24 router linears and 24 router selections. Kernel sum is not critical-path latency. FFN auxiliary work is listed separately.'}
    path=OUT/'target_components_evidence.json';temp=path.with_suffix('.json.tmp');temp.write_text(json.dumps(evidence,indent=2));temp.replace(path)
    print(json.dumps({'trace_cases':len(sources),'validated_target_rounds':sum(c['validated_rounds'] for c in sources),'summary_rows':len(rows)}))


if __name__=='__main__':main()
