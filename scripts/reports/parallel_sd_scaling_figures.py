"""CPU-only figures and consistency checks for the matched SD campaign."""
import _bootstrap
import csv
import html
import json
import statistics

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

COLORS = {'ar':'#687787','dflash':'#147d92','domino':'#458b50',
          'dspark':'#805ea3','dspark/compact':'#b14286','dflash2':'#de8a28'}


def build_scaling(cases, assets):
    grouped = {}
    correctness = [];request_rows=[]
    for case in cases:
        row, data = case['row'], case['data']
        repeats = data['results']
        reference = repeats[0]['outputs']
        assert len(reference) == row['samples']
        repeat_exact = []
        for repeat in repeats:
            outputs = repeat['outputs']
            assert len(outputs) == len(reference)
            tokens = sum(x['meta_info']['completion_tokens'] for x in outputs)
            assert tokens == repeat['output_tokens']
            assert all(len(x['output_ids']) == x['meta_info']['completion_tokens'] for x in outputs)
            assert abs(tokens / repeat['seconds'] - repeat['tokens_per_second']) < 1e-8
            repeat_exact.append(sum(a['output_ids'] == b['output_ids'] for a, b in zip(reference, outputs)))
            for index,output in enumerate(outputs):
                meta=output['meta_info'];steps=meta.get('spec_verify_ct') if row['method']!='ar' else meta['completion_tokens']-1
                duration=meta['e2e_latency']-meta['first_token_latency']
                assert duration>0 and steps>0
                assert abs((meta['completion_tokens']-1)/duration-meta['decode_throughput'])<1e-5
                request_rows.append(dict(case=row['case'],repeat=repeat['repeat'],prompt=index,
                    method=row.get('method_variant',row['method']),bs=row['bs'],decode_elapsed_ms=duration*1000,
                    decode_steps=steps,decode_tokens_per_step=(meta['completion_tokens']-1)/steps,
                    request_mean_cycle_ms=duration*1000/steps,TTFT_ms=meta['first_token_latency']*1000))
        correctness.append({'case': row['case'], 'samples': len(reference),
                            'repeat_exact_prompts_vs_repeat0': repeat_exact})
        key = (row['target'], row['dataset'], row['bs'], row.get('method_variant',row['method']))
        grouped.setdefault(key, []).append(case)

    aggregated = []
    for (target, dataset, bs, method), group in sorted(grouped.items()):
        # Pool only identical experimental conditions; AR's two bracketing cells
        # are intentionally retained in the raw table and combined in these plots.
        assert len({x['group'] for x in group}) == 1
        rows = [x['row'] for x in group]
        outputs = [o for x in group for r in x['data']['results'] for o in r['outputs']]
        def median_meta(name, scale=1):
            values = [o['meta_info'][name]*scale for o in outputs if o['meta_info'].get(name) is not None]
            return statistics.median(values) if values else None
        aggregated.append(dict(target=target, dataset=dataset, bs=bs, method=method,
            cells=len(group), median_tps=statistics.median(r['median_tps'] for r in rows),
            min_repeat_tps=min(r['tokens_per_second'] for x in group for r in x['data']['results']),
            max_repeat_tps=max(r['tokens_per_second'] for x in group for r in x['data']['results']),
            gain=statistics.median(r['speedup_vs_paired_AR'] for r in rows if r['speedup_vs_paired_AR'] is not None)
                 if any(r['speedup_vs_paired_AR'] is not None for r in rows) else None,
            tau=statistics.mean(r['effective_decode_tokens_per_verify'] for r in rows),
            physical_peak_GiB=max(r['physical_peak_GiB'] for r in rows),
            request_e2e_median_ms=median_meta('e2e_latency', 1000),
            request_ttft_median_ms=median_meta('first_token_latency', 1000),
            request_decode_tps_median=median_meta('decode_throughput'),
            request_mean_cycle_median_ms=statistics.median(r['request_mean_cycle_ms'] for r in request_rows if r['case'] in {c['row']['case'] for c in group}),
            max_repeat_CV=max(r['repeat_relative_std'] or 0 for r in rows),
            AR_drift=max((r['AR_pair_drift_fraction'] or 0 for r in rows), default=0),
            cached_tokens=sum(r['cached_tokens_sum'] for r in rows),
            retractions=sum(r['retractions_sum'] for r in rows)))
    for row in aggregated:
        baseline=next((r for r in aggregated if (r['target'],r['dataset'],r['bs'],r['method'])==(row['target'],row['dataset'],row['bs'],'ar')),None)
        row['cycle_cost_relative_to_AR']=row['request_mean_cycle_median_ms']/baseline['request_mean_cycle_median_ms'] if baseline else None
    with (assets/'scaling.csv').open('w', newline='') as stream:
        writer=csv.DictWriter(stream, fieldnames=list(aggregated[0]));writer.writeheader();writer.writerows(aggregated)
    with (assets/'request_cycle_latency.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(request_rows[0]));writer.writeheader();writer.writerows(request_rows)
    (assets/'repeat_output_consistency.json').write_text(json.dumps(correctness, indent=2))

    # Check batch-size invariance per method separately from AR/SD agreement.
    batch_comparisons=[]
    for target, dataset, method in sorted({(r['target'],r['dataset'],r['method']) for r in aggregated}):
        selected=sorted((c for c in cases if (c['row']['target'],c['row']['dataset'],c['row'].get('method_variant',c['row']['method'])) == (target,dataset,method)), key=lambda c:c['row']['bs'])
        base=selected[0]
        aa=base['data']['results'][0]['outputs']
        for other in selected:
            assert base['data']['selected_prompt_sha256']==other['data']['selected_prompt_sha256']
            bb=other['data']['results'][0]['outputs']
            batch_comparisons.append({'reference':base['row']['case'],'case':other['row']['case'],
                'exact_prompts':sum(a['output_ids']==b['output_ids'] for a,b in zip(aa,bb)), 'samples':len(aa)})
    (assets/'batch_output_consistency.json').write_text(json.dumps(batch_comparisons,indent=2))

    panels=sorted({(r['target'],r['dataset']) for r in aggregated})
    fields=[('gain','Throughput relative to paired AR','gain_scaling'),
            ('median_tps','Output tokens/s, unprofiled','throughput_scaling'),
            ('tau','Committed decode tokens / request verify','acceptance_scaling'),
            ('physical_peak_GiB','Physical GPU peak used (GiB)','memory_scaling'),
            ('request_mean_cycle_median_ms','Median request mean decode cycle (ms)','cycle_latency_scaling'),
            ('cycle_cost_relative_to_AR','SD cycle / AR token step, request medians','cycle_cost_scaling'),
            ('request_e2e_median_ms','Median request end-to-end latency (ms)','request_latency_scaling')]
    pictures=[]
    for field,title,name in fields:
        fig, axes=plt.subplots(1,len(panels),figsize=(max(5,4.5*len(panels)),3.8), squeeze=False)
        for ax,(target,dataset) in zip(axes[0],panels):
            for method,color in COLORS.items():
                points=sorted((r for r in aggregated if (r['target'],r['dataset'],r['method'])==(target,dataset,method) and r[field] is not None),key=lambda r:r['bs'])
                if not points:continue
                xx=[r['bs'] for r in points];yy=[r[field] for r in points]
                ax.plot(xx,yy,'o-',label=method,color=color,lw=2,markersize=5)
                if field=='median_tps':
                    ax.vlines(xx,[r['min_repeat_tps'] for r in points],[r['max_repeat_tps'] for r in points],color=color,alpha=.55)
            ax.set_title(f'{target} / {dataset}');ax.set_xlabel('Configured max batch size')
            ax.set_xscale('log',base=2);ax.set_xticks([1,2,4,8,16],['1','2','4','8','16']);ax.grid(alpha=.18)
            if field=='gain':ax.axhline(1,color='#555',ls=':',lw=1)
            if field not in ['physical_peak_GiB']:ax.set_ylim(bottom=0)
        axes[0][0].set_ylabel(title)
        present={r['method'] for r in aggregated}
        fig.legend(handles=[Line2D([0],[0],color=c,marker='o',label=m) for m,c in COLORS.items() if m in present],loc='upper center',ncol=min(5,len(present)),frameon=False)
        fig.tight_layout(rect=(0,0,1,.90))
        fig.savefig(assets/(name+'.svg'),bbox_inches='tight');fig.savefig(assets/(name+'.png'),bbox_inches='tight',dpi=160);plt.close(fig)
        pictures.append(f'<figure><img src="{assets.name}/{name}.svg" alt="{html.escape(title)}"><figcaption>{html.escape(title)}</figcaption></figure>')

    findings=[]
    for target,dataset in panels:
        for method in COLORS:
            if method=='ar':continue
            points=sorted((r for r in aggregated if (r['target'],r['dataset'],r['method'])==(target,dataset,method) and r['gain'] is not None),key=lambda r:r['bs'])
            if len(points)<2:continue
            a,b=points[0],points[-1]
            findings.append(f"<tr><td>{html.escape(dataset)}</td><td>{method}</td><td>{a['bs']} → {b['bs']}</td><td>{a['gain']:.2f}× → {b['gain']:.2f}×</td><td>{a['tau']:.2f} → {b['tau']:.2f}</td><td>{a['request_e2e_median_ms']:.1f} → {b['request_e2e_median_ms']:.1f}</td></tr>")
    table='<div class="scroll"><table><tr><th>任务</th><th>方法</th><th>BS</th><th>吞吐收益</th><th>有效 tokens/verify</th><th>请求 E2E 中位数 ms</th></tr>'+''.join(findings)+'</table></div>' if findings else ''
    total=sum(r['samples']*r['repeats'] for r in (x['row'] for x in cases))
    unstable=sum(any(n!=x['samples'] for n in x['repeat_exact_prompts_vs_repeat0']) for x in correctness)
    note=f'<p>检查 {total} 次请求返回：输出长度、重复计时和吞吐算式一致；{unstable} 个 cell 的不同重复出现 token 分歧。<a href="{assets.name}/repeat_output_consistency.json">重复一致性</a> · <a href="{assets.name}/batch_output_consistency.json">跨 BS 一致性</a> · <a href="{assets.name}/scaling.csv">聚合 CSV</a>。这些检查不是答案质量评测。</p>'
    changed=sum(x['exact_prompts']!=x['samples'] for x in batch_comparisons)
    note+=f'<p>{changed} 个跨 BS 比较出现 token 分歧；因此不能把不同 BS 的实测 acceptance 当作严格相同输出路径的单因素实验。配置内重复稳定、跨 BS 不一致的现象也出现在 AR。当前证据不能单独确定分歧来自哪一层或哪个 kernel。</p>'
    note+='<p>折线仅连接已有测量点，不代表中间 BS 已测。AR 图点合并前后两次 cell；吞吐竖线表示各次重复的范围，不是置信区间。max batch size 不等于每一轮实际活跃 BS，SD 的异步完成会产生 tail。请求 E2E 包含 prefill、排队与 decode，不能标为单轮 latency。</p>'
    note+=f'<p>请求平均 decode 周期 = (E2E − TTFT) / spec_verify_ct；AR 分母是 completion_tokens−1。图中是各请求平均周期的中位数，覆盖 GPU、CPU 控制与等待，排除了首 token 之前的时间；不是某个独立 NVTX 迭代的直接测量。<a href="{assets.name}/request_cycle_latency.csv">逐请求周期明细</a>。理想近似 gain≈τ/(SD cycle / AR step)，实际 wall throughput 还受 prefill、波次和 tail 影响。</p>'
    audit_path=assets.parent.parent/'logs/parallel_sd_20261104/checkpoint_memory_audit.json'
    if audit_path.exists() and all(row['target']=='qwen3-8b' for row in aggregated):
        audit=json.loads(audit_path.read_text())
        cells=''.join(f"<tr><td>{r['method']}</td><td>{r['file_GiB']:.3f}</td><td>{r['non_vocab_tensor_GiB']:.3f}</td><td>{r['draft_KV_bytes_per_token']/1024:.0f}</td><td>{r['raw_selected_hidden_bytes_per_token']/1024:.0f}</td></tr>" for r in audit['checkpoints'])
        note+='<h2>Checkpoint 文件大小与运行显存</h2><div class="scroll"><table><tr><th>方法</th><th>文件 GiB</th><th>去掉 embedding/head 后 tensor GiB</th><th>Draft KV KiB/token</th><th>原始 hidden taps KiB/token</th></tr>'+cells+'</table></div>'
        note+='<p>DSpark 文件中的 embedding 与 LM head 合计 2.318 GiB；当前 Qwen loader 跳过这两份权重，由 worker 挂接 target 的模块，因此文件大约 4.416 GiB 不代表额外驻留 4.416 GiB。去掉这两项后 checkpoint tensor 约 2.098 GiB，和物理显存中观察到的较小增量相容。</p><p>本 cohort 的 target KV 容量统一 32768 tokens，逻辑大小 4.500 GiB；五层 draft KV 是 0.625 GiB。40 KiB/token 的原始 target hidden taps 是临时特征大小，不能再假定对整个上下文永久保留并重复计入历史 KV。NVML 总显存还包含 CUDA context、graph、allocator 和 workspace，不能用 checkpoint 文件大小直接推导。</p><p><a href="../logs/parallel_sd_20261104/checkpoint_memory_audit.json">权重 header、KV 几何与源码 SHA256</a>。</p>'
    return '<h2>Batch size 扩展规律</h2>'+table+''.join(pictures)+note
