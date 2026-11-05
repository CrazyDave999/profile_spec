"""Recover Qwen3-8B verify components from existing CPU/GPU launch correlations.

These pinned, non-quantized implementations execute a fixed linear-op order.
Validate every replay's GEMM count, layer signatures, attention count and FFN
activation positions before interpreting that order. No inference is launched.
"""
from bisect import bisect_right
from collections import defaultdict
import csv
import hashlib
import html
import json
from pathlib import Path
import re
import sqlite3

ROOT=Path(__file__).resolve().parent
PARTS=['FFN','QKV_O_projection','attention_core','LM_head','other']


def analyze(path, engine, dataset, bs):
    connection=sqlite3.connect(f'file:{path.resolve()}?mode=ro',uri=True)
    strings=dict(connection.execute('SELECT id,value FROM StringIds'))
    by_tid=defaultdict(list);rounds=[]
    for start,end,tid,text,text_id in connection.execute('SELECT start,end,globalTid,text,textId FROM NVTX_EVENTS WHERE end IS NOT NULL ORDER BY start'):
        name=text or strings.get(text_id,'')
        if name=='JetSpec/verify_forward' or name.startswith('step[VERIFY '):
            index=len(rounds);match=re.search(r'bs=(\d+)',name)
            rounds.append(dict(index=index,active_bs=int(match.group(1)) if match else None,kernels=[]))
            by_tid[tid].append((start,end,index))
    assert rounds,path
    if engine=='JetSpec eager':
        records=json.loads((ROOT/'logs/overnight_20261103'/path.with_suffix('.json').name).read_text())
        metadata=[r['metadata'] for r in records['phase_records'] if r['phase']=='JetSpec/verify_forward']
        assert len(metadata)==len(rounds)
        for row,meta in zip(rounds,metadata):row['active_bs']=meta['batch']
    starts={tid:[r[0] for r in rows] for tid,rows in by_tid.items()}
    correlations={}
    for start,corr,tid in connection.execute('SELECT start,correlationId,globalTid FROM CUPTI_ACTIVITY_KIND_RUNTIME'):
        if tid not in starts:continue
        i=bisect_right(starts[tid],start)-1
        if i>=0 and start<=by_tid[tid][i][1]:correlations[corr]=by_tid[tid][i][2]
    for start,end,corr,name_id,gx,gy,gz in connection.execute('SELECT start,end,correlationId,demangledName,gridX,gridY,gridZ FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start'):
        if corr in correlations:
            rounds[correlations[corr]]['kernels'].append(dict(start=start,end=end,name=strings[name_id],grid=[gx,gy,gz],ms=(end-start)/1e6))
    connection.close()
    per_layer=7 if engine=='JetSpec eager' else 4
    first_ffn=4 if per_layer==7 else 2
    for row in rounds:
        kernels=row['kernels']
        gemms=[i for i,k in enumerate(kernels) if ('gemm' in k['name'].lower() or 'gemv' in k['name'].lower()) and ('bf16' in k['name'].lower() or 'bfloat16' in k['name'].lower())]
        assert len(gemms)==36*per_layer+1,(path,row['index'],len(gemms))
        labels=['other']*len(kernels)
        for layer in range(36):
            indices=gemms[layer*per_layer:(layer+1)*per_layer]
            # Identical projection shapes must have identical launch signatures
            # across the 36 homogeneous decoder layers in a replay.
            if layer:
                reference=gemms[:per_layer]
                assert all((kernels[a]['name'],kernels[a]['grid'])==(kernels[b]['name'],kernels[b]['grid']) for a,b in zip(indices,reference)),(path,row['index'],layer)
            for i in indices[:first_ffn]:labels[i]='QKV_O_projection'
            ff_start,ff_end=indices[first_ffn],indices[-1]
            assert any('silu' in kernels[i]['name'].lower() or 'act_and_mul' in kernels[i]['name'].lower() for i in range(ff_start,ff_end+1)),(path,layer,'FFN activation missing')
            for i in range(ff_start,ff_end+1):labels[i]='FFN'
        labels[gemms[-1]]='LM_head'
        cores=[]
        for i,kernel in enumerate(kernels):
            lower=kernel['name'].lower()
            if 'fmha_' in lower or 'flash_fwd' in lower or 'flashattentionforward' in lower:
                assert labels[i]=='other';labels[i]='attention_core';cores.append(i)
        assert len(cores)==36,(path,row['index'],'attention kernels',len(cores))
        values={p:sum(k['ms'] for k,label in zip(kernels,labels) if label==p) for p in PARTS}
        row.update(components=values,total_ms=sum(k['ms'] for k in kernels),gemm_count=len(gemms))
        del row['kernels']
    result=[]
    for scope,selected in [('all_observed_rounds',rounds),('full_active_batch',[r for r in rounds if r['active_bs']==bs])]:
        assert selected
        result.append(dict(engine=engine,dataset=dataset,nominal_bs=bs,scope=scope,calls=len(selected),
            mean_active_bs=sum(r['active_bs'] for r in selected)/len(selected),
            total_ms=sum(r['total_ms'] for r in selected)/len(selected),
            **{p+'_ms':sum(r['components'][p] for r in selected)/len(selected) for p in PARTS}))
    return result,dict(database=str(path.relative_to(ROOT)),database_bytes=path.stat().st_size,
        expected_BF16_GEMMs_per_verify=36*per_layer+1,validated_rounds=len(rounds),
        validated_attention_core_kernels_per_verify=36,rounds=rounds)


def main():
    output=ROOT/'reports/2026-11-05_verify_components';output.mkdir(exist_ok=True)
    rows=[];evidence=[]
    for dataset in ['gsm8k','mt_bench']:
        for bs in [1,2,4,8,16]:
            path=ROOT/f'profiles/overnight_20261103/batch_reference_{dataset}_bs{bs}_trace_both_t128.sqlite'
            if path.exists():
                values,proof=analyze(path,'JetSpec eager',dataset,bs);rows+=values;evidence.append(proof)
    for bs,index in [(1,1),(16,6)]:
        path=ROOT/f'profiles/parallel_sd_20261104/follow01_traces_{index:02d}_dflash_gsm8k_bs{bs}.sqlite'
        if path.exists():
            values,proof=analyze(path,'SGLang DFlash','gsm8k',bs);rows+=values;evidence.append(proof)
    with (output/'components.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    sources=['analyze_verify_components.py','batch_reference.py',
        'third_party/SGLang/python/sglang/srt/models/qwen3.py','third_party/SGLang/python/sglang/srt/models/qwen2.py',
        '.venv/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py']
    (output/'evidence.json').write_text(json.dumps(dict(cases=evidence,
        source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources if (ROOT/p).exists()},
        method='CPU NVTX launch correlation -> GPU kernels; validated linear ordering and homogeneous-layer launch signatures. FFN includes its linears and activation/multiply between them. QKV/O includes linears only. Norm/RoPE/KV/copies remain other.',
        scope='Summed GPU kernel execution time per verify; not CPU duration or isolated critical-path latency. Existing short-context traces; not a long-context claim.'),indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    panels=sorted({(r['engine'],r['dataset']) for r in rows})
    fig,axes=plt.subplots(1,len(panels),figsize=(5*len(panels),4),squeeze=False)
    colors=['#147d92','#458b50','#de8a28','#805ea3','#8997a5']
    for ax,(engine,dataset) in zip(axes[0],panels):
        selected=[r for r in rows if (r['engine'],r['dataset'],r['scope'])==(engine,dataset,'all_observed_rounds')]
        xx=range(len(selected));bottom=[0.]*len(selected)
        for part,color in zip(PARTS,colors):
            values=[r[part+'_ms'] for r in selected];ax.bar(xx,values,bottom=bottom,color=color,label=part);bottom=[a+b for a,b in zip(bottom,values)]
        ax.set_xticks(list(xx),[str(r['nominal_bs']) for r in selected]);ax.set_title(engine+' / '+dataset);ax.set_xlabel('Nominal batch size');ax.set_ylabel('GPU kernel ms / verify')
    handles,labels=axes[0][0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',ncol=5,fontsize=9,frameon=False);fig.tight_layout(rect=(0,0,1,.88));fig.savefig(output/'breakdown.svg',bbox_inches='tight');fig.savefig(output/'breakdown.png',bbox_inches='tight',dpi=160);plt.close(fig)
    table='<table><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in ['Engine / task','BS','scope','calls','total ms']+PARTS)+'</tr>'
    for row in rows:
        values=[row['engine']+' / '+row['dataset'],row['nominal_bs'],row['scope'],row['calls'],f"{row['total_ms']:.3f}"]+[f"{row[p+'_ms']:.3f}" for p in PARTS]
        table+='<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in values)+'</tr>'
    table+='</table>'
    doc=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>2026-11-05 · Verify 算子分解</title><style>body{{font:16px/1.7 system-ui;margin:30px;color:#203247}}main{{max-width:1400px;margin:auto}}img{{max-width:100%}}table{{border-collapse:collapse;font-size:12px}}td,th{{padding:8px;border-bottom:1px solid #ddd;white-space:nowrap}}.scroll{{overflow:auto}}</style><main><a href="index.html">报告索引</a><h1>Verify：FFN、attention 与 LM head</h1><p>2026-11-05（北京时间）· RTX 5090 · Qwen3-8B BF16 · 只重分析已有 trace，没有启动新 GPU 推理。</p><p>JetSpec 的 BS 扩展数据来自 eager batch_reference，tree budget128；SGLang DFlash 是 CUDA Graph，verify width16。两组实现、输入集及 query 宽度不同，分别解释，不能用绝对时间作引擎公平比较。</p><img src="{output.name}/breakdown.svg" alt="verify GPU kernel 时间分解"><p>图中为全部轮次均值，包含 active BS 下降的 tail；full_active_batch 行只选择实际 active BS 等于标称值的轮次。Kernel 时间之和不是端到端 latency。不同 BS 的短输入与轨迹差异保留。</p><div class="scroll">{table}</div><p>FFN 包括 gate/up/down GEMM 及中间激活和乘法；attention core 指 QK/softmax/AV kernel；QKV/O 投影另外列出，其他包含 norm、RoPE、KV 写入、拷贝等。不能把所有 GEMM 都标为 FFN，也不能把含 FlashInfer 名称的 norm kernel 标为 attention。</p><p>归因通过每轮固定执行顺序恢复，并逐轮校验：JetSpec 253 个 BF16 GEMM，SGLang 145 个；均为36层加一个 LM head；每层有一个 attention core，FFN 激活处在相应投影之间，36层对应角色的 kernel/launch grid 一致。</p><p><a href="{output.name}/components.csv">CSV</a> · <a href="{output.name}/evidence.json">逐轮数据、校验与源码 SHA256</a>。这些短上下文结果不外推为长上下文或任意模型结论。</p></main></html>'''
    (output.with_suffix('.html')).write_text(doc)
    from build_report_index import write_index
    write_index()
    for row in rows:
        if row['scope']=='all_observed_rounds':print(row)


if __name__=='__main__':main()
