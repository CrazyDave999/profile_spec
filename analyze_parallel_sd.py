"""Report only terminal, successful cells from a named SGLang campaign."""
import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from build_report_index import write_index
from parallel_sd_scaling_figures import build_scaling, COLORS

ROOT=Path(__file__).resolve().parent
LOGS=ROOT/'logs/parallel_sd_20261104'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',required=True);parser.add_argument('--slug',default='psd_pilot');args=parser.parse_args()
    campaign=json.loads((LOGS/f'{args.run_id}.campaign.json').read_text())
    assert campaign['stage'] not in ['sps','numerics'],'Calibration/diagnostics cannot enter headline throughput reports.'
    stem='2026-11-04_'+args.slug;assets=ROOT/'reports'/stem;assets.mkdir(exist_ok=True)
    cases=[];sources={}
    for cell in campaign['commands']:
        job=LOGS/(cell['name']+'.job.json');p=LOGS/(cell['name']+'.json')
        if not job.exists() or not p.exists():continue
        status=json.loads(job.read_text());j=json.loads(p.read_text())
        if status['state']!='success' or j['status']!='success':continue
        a=j['arguments'];r=j['results'];outputs=r[0]['outputs'];rates=[x['tokens_per_second'] for x in r]
        if a['profile'] or a.get('logprobs',False):continue
        effective=[x.get('effective_decode_tokens_per_verify_excluding_prefill_seed') for x in r];effective=[x for x in effective if x is not None]
        key=(a.get('target','qwen3-8b'),a['dataset'],a['bs'],a['samples'],a['tokens'],a.get('extra_context',0),a.get('eos',False),a.get('fp32_lm_head',False))
        log=(LOGS/(cell['name']+'.log')).read_text(errors='replace')
        feature_lines=[line for line in log.splitlines() if 'server_args=' not in line and any(s in line for s in ['folded into','fused KV materialization enabled','cuda graph: True','Capture target verify CUDA graph end','Capture target decode CUDA graph end','Capture draft verify CUDA graph end','metadata glue graph off','rollout enabled'])]
        row={'case':cell['name'],'target':key[0],'dataset':a['dataset'],'bs':a['bs'],'samples':a['samples'],'output_cap':a['tokens'],'method':a['method'],'verify_width':j['engine_resolved_after'].get('speculative_num_draft_tokens') if a['method']!='ar' else 1,'median_tps':statistics.median(rates),'repeat_relative_std':statistics.stdev(rates)/statistics.mean(rates) if len(rates)>1 else None,'effective_decode_tokens_per_verify':statistics.mean(effective) if effective else 1,'physical_peak_GiB':j['physical_device_peak_used_bytes']/2**30,'repeats':len(r),'profile':a['profile'],'cached_tokens_sum':sum(o['meta_info'].get('cached_tokens',0) for x in r for o in x['outputs']),'retractions_sum':sum(o['meta_info'].get('num_retractions',0) for x in r for o in x['outputs'])}
        row.update(ragged_mode=a.get('ragged_mode','static'),method_variant=a['method']+('/'+a['ragged_mode'] if a.get('ragged_mode','static')!='static' else ''),sps_table=a.get('sps_table'))
        cases.append({'row':row,'group':key,'data':j,'feature_lines':feature_lines})
        sources[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    assert cases,'No successful terminal cells to report.'
    groups={x['group'] for x in cases};comparisons=[];pairwise=[]
    for key in sorted(groups):
        group=[x for x in cases if x['group']==key];ars=[x for x in group if x['row']['method']=='ar'];baseline=ars[0] if ars else None
        ar_rates=[x['row']['median_tps'] for x in ars]
        sd=[x for x in group if x['row']['method']!='ar']
        if sd:
            ref=sd[0];aa=ref['data']['results'][0]['outputs']
            pairwise.append({'reference':ref['row']['case'],'per_method_exact_prompts':{x['row']['method_variant']:sum(a['output_ids']==b['output_ids'] for a,b in zip(aa,x['data']['results'][0]['outputs'])) for x in sd},'samples':ref['row']['samples']})
        for x in group:
            row=x['row'];row['speedup_vs_paired_AR']=row['median_tps']/statistics.median(ar_rates) if ar_rates else None
            row['AR_pair_drift_fraction']=(max(ar_rates)-min(ar_rates))/min(ar_rates) if len(ar_rates)>1 else None
            row['exact_matching_prompts_vs_first_AR']=None
            if baseline:
                assert baseline['data']['selected_prompt_sha256']==x['data']['selected_prompt_sha256']
                assert baseline['data']['package_versions']==x['data']['package_versions']
                assert baseline['data']['toolkit_header_sha256']==x['data']['toolkit_header_sha256']
                assert baseline['data']['worker_source_sha256']==x['data']['worker_source_sha256']
                aa=baseline['data']['results'][0]['outputs'];bb=x['data']['results'][0]['outputs'];matches=[]
                for i,(a,b) in enumerate(zip(aa,bb)):
                    ax,bx=a['output_ids'],b['output_ids'];index=next((n for n,(v,w) in enumerate(zip(ax,bx)) if v!=w),None)
                    if index is None and len(ax)!=len(bx):index=min(len(ax),len(bx))
                    matches.append({'prompt':i,'equal':ax==bx,'first_mismatch':index})
                row['exact_matching_prompts_vs_first_AR']=sum(m['equal'] for m in matches)
                comparisons.append({'case':row['case'],'baseline':baseline['row']['case'],'matches':matches})
    data=[x['row'] for x in cases]
    with (assets/'results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    (assets/'correctness.json').write_text(json.dumps(comparisons,indent=2))
    (assets/'SD_pairwise_outputs.json').write_text(json.dumps(pairwise,indent=2))
    analysis_sources=['analyze_parallel_sd.py','parallel_sd_scaling_figures.py','analyze_sglang_trace.py','third_party/SGLang/python/sglang/srt/observability/req_time_stats.py','models-parallel-sd.json','logs/dflash2_config_conversion.json','logs/parallel_sd_20261104/checkpoint_memory_audit.json']
    (assets/'source_manifest.json').write_text(json.dumps({'campaign':campaign,'source_SHA256':sources,'analysis_and_configuration_SHA256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in analysis_sources if (ROOT/p).exists()},'feature_lines':{x['row']['case']:x['feature_lines'] for x in cases}},indent=2))
    plt.rcParams.update({'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'font.size':10})
    fig,axes=plt.subplots(1,3,figsize=(13,4));xx=range(len(data));labels=[r['method_variant']+f"\nBS{r['bs']}" for r in data];colors=[COLORS[r['method_variant']] for r in data]
    for ax,field,title in zip(axes,['median_tps','effective_decode_tokens_per_verify','physical_peak_GiB'],['Unprofiled output tokens/s','Effective decode tokens / request verify','Physical GPU peak used GiB']):
        ax.bar(xx,[r[field] for r in data],color=colors);ax.set_xticks(list(xx),labels,rotation=20,ha='right');ax.set_title(title)
    fig.tight_layout();fig.savefig(assets/'comparison.svg',bbox_inches='tight');fig.savefig(assets/'comparison.png',bbox_inches='tight',dpi=160);plt.close(fig)
    headers=['Case','Method','BS','Verify width','tok/s','vs paired AR','Repeat CV','Decode tokens/verify','GPU GiB','Exact prompts / AR']
    cells=[]
    for r in data:
        gain='pending' if r['speedup_vs_paired_AR'] is None else f"{r['speedup_vs_paired_AR']:.2f}×"
        cells.append([r['case'],r['method_variant'],r['bs'],r['verify_width'],f"{r['median_tps']:.1f}",gain,f"{100*r['repeat_relative_std']:.2f}%" if r['repeat_relative_std'] is not None else '—',f"{r['effective_decode_tokens_per_verify']:.2f}",f"{r['physical_peak_GiB']:.2f}",str(r['exact_matching_prompts_vs_first_AR'])+'/'+str(r['samples'])])
    table='<table><tr>'+''.join('<th>'+html.escape(str(s))+'</th>' for s in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(s))+'</td>' for s in row)+'</tr>' for row in cells)+'</table>'
    feature=''.join('<details><summary>'+html.escape(x['row']['case'])+'</summary><pre>'+html.escape('\n'.join(x['feature_lines']))+'</pre></details>' for x in cases)
    text=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>2026-11-04 · Parallel SD / SGLang {args.run_id}</title><style>body{{font:16px/1.8 system-ui;color:#203247;background:#f3f5f8;margin:0}}main{{max-width:1300px;margin:30px auto;padding:28px;background:white}}h1{{color:#18334c}}a{{color:#147d92}}img{{max-width:100%}}.scroll{{overflow:auto}}table{{border-collapse:collapse;font-size:13px}}td,th{{padding:9px;border-bottom:1px solid #ddd;white-space:nowrap;text-align:left}}.note{{padding:16px;background:#fff6e6;border-left:4px solid #de8a28}}pre{{white-space:pre-wrap;font-size:12px}}details{{margin:12px 0}}</style><main><a href="index.html">← 报告索引</a><h1>Parallel SD / SGLang：{args.run_id}</h1><p>RTX 5090 · 原始真实输入 token IDs · BF16 target · FA4 · 同一 SGLang Engine</p><p class="note">探索性实验，不是论文 leaderboard。此报告只收录已成功结束的 cell：{len(data)}/{len(campaign['commands'])}；campaign 当前状态：{html.escape(campaign['status'])}。阶段：{html.escape(campaign['stage'])}。未完成项没有填充数据。</p><p>decode CUDA Graph、overlap scheduler 开启；初始 cohort 的 prefill graph 关闭；prefix reuse 关闭。固定输出忽略 EOS，用于隔离资源扩展，不能当作自然停止的应用吞吐。加载、编译、warmup 和 flush-cache 不计入吞吐窗口。</p><p>每个 cell 使用相同输入与输出上限，metadata 保存请求时间、返回 token IDs、实际 spec verify 次数和 device memory 采样。AR 前后两次用于检查漂移；speedup 分母是匹配 AR cell median 的中位数。任何 BF16 token 分歧均保留，不据此声称逐 token 一致或已验证答案质量。</p><a href="{stem}/comparison.svg"><img src="{stem}/comparison.svg" alt="方法吞吐、接受长度和物理显存对比"></a><div class="scroll">{table}</div><p>有效 decode tokens/verify = (实际输出 tokens − 每请求 prefill seed token) / 请求 verify 次数之和，包含被提交的 bonus。AR 为 1。截断最后一轮会影响它，不能把它当成未截断的模型内生 acceptance。</p><p>物理显存来自 NVML 50 ms 采样，不是 PyTorch allocated/reserved。SGLang memory_usage 的 target weight/KV 项与其合并 graph 项不能直接当作完整 draft 权重和 draft KV 的总和。</p><h2>功能路径证据</h2>{feature}<h2>原始数据与后续工作</h2><p><a href="{stem}/results.csv">CSV</a> · <a href="{stem}/correctness.json">逐请求 token 比较</a> · <a href="{stem}/source_manifest.json">版本、原始 cell SHA256 与功能日志</a> · <a href="../logs/parallel_sd_20261104/protocol.json">实验协议</a></p><p>后续仍需完成 BS/上下文/verify 宽度消融、同宽度方法对照、代表性 Nsight 阶段图、DSpark 固定/动态 verify 分离，以及 hybrid 的 recurrent-state commit 分析。本报告不会把 smoke test 或 instrumented profile 当作 headline 吞吐。</p></main></html>'''
    pilot_note='；初始 pilot 的四个 SD 方法在 8/8 个 prompt 上互相一致' if campaign['stage']=='pilot' else ''
    text=text.replace('<h2>功能路径证据</h2>','<p>不同 checkpoint 的训练数据、block size 和推理模块不同，本表是部署组合比较，不是算法单因素消融。<a href="'+stem+'/SD_pairwise_outputs.json">SD 方法之间的输出比较</a>另列'+pilot_note+'。</p><h2>功能路径证据</h2>')
    text=text.replace('后续仍需完成 BS/上下文/verify 宽度消融、同宽度方法对照、代表性 Nsight 阶段图、DSpark 固定/动态 verify 分离，以及 hybrid 的 recurrent-state commit 分析。本报告不会把 smoke test 或 instrumented profile 当作 headline 吞吐。','完整矩阵、同宽度、代表性 Nsight、hybrid 与动态 verify 的结果和当前状态见 <a href="2026-11-04_parallel_sd.html">总报告</a>。本 cohort 未覆盖长上下文与自然 EOS 的应用负载；这些可作为后续扩展。吞吐表仅收录独立、无 profiler 的测量。')
    if campaign['stage'] != 'pilot':
        # The per-cell bar chart is useful for six pilot cells, but obscures the
        # batch/workload axes in the full matrix. Keep its artifact, show scaling.
        chart=f'<a href="{stem}/comparison.svg"><img src="{stem}/comparison.svg" alt="方法吞吐、接受长度和物理显存对比"></a>'
        text=text.replace(chart,build_scaling(cases,assets))
    if campaign['stage']=='hybrid':
        text=text.replace('原始真实输入 token IDs · BF16 target · FA4','真实输入使用 Qwen3.5 tokenizer 重新套模板 · BF16 target · full/linear attention=triton · draft=FA4')
    trace_path=LOGS/'ns_dflash_b1_a01.analysis.json'
    if trace_path.exists() and campaign['stage']=='pilot':
        trace=json.loads(trace_path.read_text());phases=trace['phases'];window=trace['decode_window'];labels=['DRAFT','VERIFY','decode_control']
        fig,axes=plt.subplots(1,2,figsize=(11,3.7))
        for ax,field,title in zip(axes,['gpu_kernel_ms_per_call','host_ms_per_call'],['GPU kernel sum per SD round','Host NVTX per SD round (waits included)']):
            bottom=0
            for label,color in zip(labels,['#147d92','#de8a28','#805ea3']):
                v=phases[label][field];ax.barh([0],[v],left=[bottom],color=color,label=label);bottom+=v
            ax.set_yticks([0],['DFlash BS1']);ax.set_xlabel('ms');ax.set_title(title);ax.legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.2),ncol=3)
        fig.tight_layout();fig.savefig(assets/'dflash_trace_breakdown.svg',bbox_inches='tight');fig.savefig(assets/'dflash_trace_breakdown.png',bbox_inches='tight',dpi=160);plt.close(fig)
        section=f'<h2>第一份 warmed Nsight：DFlash BS1</h2><img src="{stem}/dflash_trace_breakdown.svg" alt="DFlash GPU/host 阶段分开归因"><p>单个 GSM8K prompt、cap64、block16，共 {window["scheduler_decode_rounds"]} 轮。GPU launch correlation 覆盖 {trace["correlation_coverage"]:.0%}，graph launch {trace["cuda_graph_launches"]} 次，capture {trace["cuda_graph_captures"]} 次。完整 decode 窗口每轮约 {window["mean_complete_round_ms"]:.2f} ms，GPU kernel busy fraction 约 {window["kernel_busy_fraction"]:.1%}。</p><p>本例的 GPU 空隙很小；高 busy 不等于 compute bound。control 是 model.forward 以外的 scheduler 工作，不能把它全部标为 accept。CPU 与 GPU 图不可相加；本图是 instrumented profile，不能替代上方独立吞吐。</p><p><a href="../logs/parallel_sd_20261104/ns_dflash_b1_a01.analysis.json">归因 JSON</a>；原始 .nsys-rep 在 ignored profiles/parallel_sd_20261104，约 790 KB。</p>'
        text=text.replace('<h2>原始数据与后续工作</h2>',section+'<h2>原始数据与后续工作</h2>')
    (ROOT/'reports'/f'{stem}.html').write_text(text);write_index();print('Reported',len(data),'terminal successful cells to',stem+'.html')


if __name__=='__main__':main()
