"""Cross-campaign width comparisons with matched inputs and archived code."""
import _bootstrap
import ast
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT=_bootstrap.ROOT
LOGS=ROOT/'logs/parallel_sd_20261104'
METHODS={'dflash':'#147d92','domino':'#458b50','dspark':'#805ea3','dflash2':'#de8a28'}


def read_csv(path):
    return list(csv.DictReader(path.open())) if path.exists() else []


class PruneHybrid(ast.NodeTransformer):
    def visit_If(self,node):
        test=node.test
        if isinstance(test,ast.Compare) and len(test.ops)==1 and isinstance(test.ops[0],ast.Eq) and ast.dump(test.left)==ast.dump(ast.Attribute(value=ast.Name(id='args',ctx=ast.Load()),attr='target',ctx=ast.Load())) and isinstance(test.comparators[0],ast.Constant) and test.comparators[0].value=='qwen3.5-4b' and not node.orelse:
            return None
        return self.generic_visit(node)


def qwen8_source(sha):
    path=LOGS/'worker_sources'/(sha+'.py');source=path.read_bytes()
    assert hashlib.sha256(source).hexdigest()==sha
    tree=PruneHybrid().visit(ast.parse(source))
    return hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest()


def width_comparison(main,output):
    directory=ROOT/'reports/2026-11-04_psd_width8'
    measured=read_csv(directory/'scaling.csv');raw=read_csv(directory/'results.csv')
    base_raw=read_csv(ROOT/'reports/2026-11-04_psd_scaling/results.csv')
    if not measured:return ''
    comparison=[];provenance=[]
    for new in measured:
        method=new['method']
        if method=='ar':continue
        match=lambda r:r['dataset']==new['dataset'] and r['bs']==new['bs']
        if sum(match(r) and r['method']=='ar' for r in raw)!=2:continue
        old=next((r for r in main if match(r) and r['method']==method),None)
        if old is None:continue
        base_cell=next(r for r in base_raw if match(r) and r['method']==method)
        new_cell=next(r for r in raw if match(r) and r['method']==method)
        aa=json.loads((LOGS/(base_cell['case']+'.json')).read_text());bb=json.loads((LOGS/(new_cell['case']+'.json')).read_text())
        for field in ['selected_prompt_sha256','package_versions','toolkit_header_sha256','sglang_revision']:
            assert aa[field]==bb[field],field
        for field in ['target','dataset','bs','samples','tokens','extra_context','eos','fp32_lm_head','logprobs','profile','attention','ragged_mode']:
            assert aa['arguments'][field]==bb['arguments'][field],field
        for field in ['attention_backend','dtype','tp_size','max_total_tokens','enable_fp32_lm_head']:
            assert aa['engine_resolved_after'][field]==bb['engine_resolved_after'][field],field
        assert aa['arguments']['target']==bb['arguments']['target']=='qwen3-8b'
        assert qwen8_source(aa['worker_source_sha256'])==qwen8_source(bb['worker_source_sha256'])
        first_a=aa['results'][0]['outputs'];first_b=bb['results'][0]['outputs']
        row=dict(dataset=new['dataset'],bs=int(new['bs']),method=method,
            old_width=int(base_cell['verify_width']),new_width=int(new_cell['verify_width']),
            old_gain=float(old['gain']),new_gain=float(new['gain']),
            relative_gain=float(new['gain'])/float(old['gain']),
            old_tau=float(old['tau']),new_tau=float(new['tau']),
            old_cycle_ms=float(old['request_mean_cycle_median_ms']),new_cycle_ms=float(new['request_mean_cycle_median_ms']),
            exact_prompts_between_widths=sum(a['output_ids']==b['output_ids'] for a,b in zip(first_a,first_b)),
            samples=len(first_a),old_case=base_cell['case'],new_case=new_cell['case'])
        assert row['new_width']==8
        comparison.append(row)
        provenance.append(dict(old_case=row['old_case'],new_case=row['new_case'],
            old_worker_sha256=aa['worker_source_sha256'],new_worker_sha256=bb['worker_source_sha256'],
            qwen8_worker_AST_sha256=qwen8_source(aa['worker_source_sha256']),
            matched_fields=['prompt IDs','packages','toolkit header','SGLang revision','attention backend','dtype','TP','KV capacity','head precision'],
            source_check='Only hybrid-specific if args.target==qwen3.5-4b branches are pruned; remaining worker AST matches. Actual target here is qwen3-8b.'))
    if not comparison:return '<p>同宽度数据已开始返回；等待前后 AR 都结束后，再发布跨 campaign 对照。</p>'
    with (output/'width_comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(comparison[0]));writer.writeheader();writer.writerows(comparison)
    (output/'width_comparison_provenance.json').write_text(json.dumps(provenance,indent=2))
    groups=sorted({(r['dataset'],r['bs']) for r in comparison})
    plt.rcParams.update({'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,len(groups),figsize=(max(5,4.5*len(groups)),4),squeeze=False)
    for ax,(dataset,bs) in zip(axes[0],groups):
        points=[r for method in METHODS for r in comparison if r['method']==method and (r['dataset'],r['bs'])==(dataset,bs)]
        xx=list(range(len(points)))
        ax.bar([x-.18 for x in xx],[r['old_gain'] for r in points],width=.36,color='#b7c1ca')
        ax.bar([x+.18 for x in xx],[r['new_gain'] for r in points],width=.36,color=[METHODS[r['method']] for r in points])
        ax.set_xticks(xx,[r['method']+f"\n{r['old_width']} -> 8" for r in points],rotation=15,ha='right')
        ax.set_title(f'{dataset} / BS{bs}');ax.axhline(1,color='#555',ls=':',lw=1);ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.2)
    axes[0][0].set_ylabel('Throughput / own paired AR')
    fig.legend(handles=[Patch(color='#b7c1ca',label='Default width'),Patch(color='#147d92',label='Width 8 (method colors)')],loc='upper center',ncol=2,frameon=False)
    fig.tight_layout(rect=(0,0,1,.9));fig.savefig(output/'width_effect.svg',bbox_inches='tight');fig.savefig(output/'width_effect.png',bbox_inches='tight',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,len(groups),figsize=(max(5,4.5*len(groups)),4),squeeze=False)
    for ax,(dataset,bs) in zip(axes[0],groups):
        for row in comparison:
            if (row['dataset'],row['bs'])!=(dataset,bs):continue
            color=METHODS[row['method']]
            ax.plot([row['old_cycle_ms']],[row['old_tau']],'o',mfc='white',mec=color)
            ax.plot([row['new_cycle_ms']],[row['new_tau']],'o',color=color,label=row['method'])
            ax.annotate('',xy=(row['new_cycle_ms'],row['new_tau']),xytext=(row['old_cycle_ms'],row['old_tau']),arrowprops=dict(arrowstyle='->',color=color,lw=1.5))
        ax.set_xlabel('Median request mean cycle (ms)');ax.set_title(f'{dataset} / BS{bs}');ax.grid(alpha=.2)
    axes[0][0].set_ylabel('Effective decode tokens / request verify');axes[0][0].legend(fontsize=8)
    fig.suptitle('Default width (open) -> width 8 (filled); shorter cycles trade off acceptance')
    fig.tight_layout();fig.savefig(output/'width_tradeoff.svg',bbox_inches='tight');fig.savefig(output/'width_tradeoff.png',bbox_inches='tight',dpi=160);plt.close(fig)
    rows=''.join(f"<tr><td>{r['dataset']}</td><td>{r['bs']}</td><td>{r['method']}</td><td>{r['old_width']}→8</td><td>{r['old_gain']:.2f}× → {r['new_gain']:.2f}×</td><td>{r['old_tau']:.2f} → {r['new_tau']:.2f}</td><td>{r['old_cycle_ms']:.2f} → {r['new_cycle_ms']:.2f}</td><td>{r['exact_prompts_between_widths']}/{r['samples']}</td></tr>" for r in comparison)
    findings=''
    small=next((r for r in comparison if (r['dataset'],r['bs'],r['method'])==('gsm8k',1,'dflash')),None)
    large=next((r for r in comparison if (r['dataset'],r['bs'],r['method'])==('gsm8k',16,'dflash')),None)
    if small and large:
        findings+=f'<p class="finding">同一个 DFlash checkpoint，width16→8：GSM8K BS1 周期 {small["old_cycle_ms"]:.2f}→{small["new_cycle_ms"]:.2f} ms，几乎不变，τ 却从 {small["old_tau"]:.2f} 降到 {small["new_tau"]:.2f}，收益下降；BS16 周期从 {large["old_cycle_ms"]:.2f} 降到 {large["new_cycle_ms"]:.2f} ms，收益从 {large["old_gain"]:.2f}× 升到 {large["new_gain"]:.2f}×。最合适的宽度随 BS 改变，较高 acceptance 本身并不足以保证更快。</p>'
    for row in comparison:
        if row['dataset']=='mt_bench' and row['bs']==16 and row['method'] in ['dflash','domino']:
            findings+=f'<p>MT-Bench BS16，{row["method"]}：default width 的 {row["old_gain"]:.2f}× → width8 的 {row["new_gain"]:.2f}×。该结果用于检验主矩阵负收益是否能通过宽度调整改善，输出一致性见表中对应列。</p>'
    return findings+'<div class="scroll"><table><tr><th>任务</th><th>BS</th><th>方法</th><th>宽度</th><th>相对 AR 吞吐</th><th>τ</th><th>请求周期 ms</th><th>跨宽度相同输出</th></tr>'+rows+'</table></div>'+''.join(f'<figure><a href="{output.name}/{name}.svg"><img src="{output.name}/{name}.svg" alt="{caption}"></a><figcaption>{caption}</figcaption></figure>' for name,caption in [('width_effect','统一宽度与默认配置：各自前后 AR 归一化'),('width_tradeoff','同 checkpoint 的周期成本与接受长度折中')])+f'<p>只收录前后 AR 均完成的组。DSpark/DFlash2 的 8→8 是重复对照。跨 campaign 输入、运行时和 8B worker 路径已核对；worker SHA 不同源于 hybrid 分支修正，归档源码去掉该未执行分支后 AST 一致。宽度改变仍可能带来 BF16 token 分歧，不能把它当作严格相同输出轨迹。<a href="{output.name}/width_comparison.csv">CSV</a> · <a href="{output.name}/width_comparison_provenance.json">匹配与源码证据</a>。</p>'
