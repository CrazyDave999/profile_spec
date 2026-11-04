"""Build the local Chinese report, scientific plots, CSVs and timeline index."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import csv
import html
import json
import math
from pathlib import Path
import statistics
import zipfile
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from analyze_sweep_trace import analyze
from sweep_worker import save

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'reports/2026-11-03'
DATASETS=['gsm8k','math500','aime25','humaneval','mbpp','livecodebench','mt_bench']
NAMES={'gsm8k':'GSM8K','math500':'MATH-500','aime25':'AIME25','humaneval':'HumanEval',
       'mbpp':'MBPP','livecodebench':'LiveCodeBench','mt_bench':'MT-Bench'}
BS=[1,2,4,8,16]
COLORS=plt.get_cmap('tab10').colors
PHASE_LABELS={'prefill':'Prefill','decode_forward':'AR decode','draft_forward':'Draft',
 'draft_prepare':'Draft preparation','verify_forward':'Verify','tree_build':'Tree build',
 'kv_pack':'KV packing','target_prepare':'Target inputs','accept':'Accept','kv_commit':'KV commit','other':'Other'}


def link(path,label=None):
    path=Path(path)
    try: target='../../'+str(path.resolve().relative_to(ROOT))
    except ValueError: target=str(path)
    return f'<a href="{html.escape(target,quote=True)}">{html.escape(label or path.name)}</a>'


def number(x,digits=2):
    return '—' if x is None or not math.isfinite(float(x)) else f'{x:,.{digits}f}'


def table(headers,rows):
    return '<table><thead><tr>'+''.join('<th>'+h+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join(
        '<tr>'+''.join('<td>'+str(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table>'


def plot(name):
    plt.savefig(OUT/(name+'.svg'),bbox_inches='tight')
    plt.savefig(OUT/(name+'.png'),dpi=170,bbox_inches='tight')
    plt.close()
    return f'<figure><a href="{name}.svg"><img src="{name}.svg" alt="{name}"></a></figure>'


def read_cases():
    cases=[]
    for directory in [ROOT/'logs/overnight_20261103',ROOT/'logs/gpu_metrics_20261103']:
        if not directory.exists(): continue
        for p in directory.glob('*.json'):
            if p.name in ('plan.json','status.json') or p.name.endswith(('.analysis.json','.progress.json')): continue
            row=json.loads(p.read_text())
            if 'arguments' not in row: continue
            row['_path']=str(p)
            row['_name']=p.stem
            if row.get('status')=='success' and row.get('nsys_sqlite'):
                analysis=p.with_suffix('.analysis.json')
                if not analysis.exists(): save(analysis,analyze(row['nsys_sqlite']))
                row['_analysis']=json.loads(analysis.read_text())
                row['_analysis_path']=str(analysis)
                counter_path=p.with_suffix('.counters.json')
                if row.get('nsys_gpu_metrics_enabled') and not counter_path.exists():
                    from analyze_gpu_counters import analyze as analyze_counters
                    save(counter_path,analyze_counters(row['nsys_sqlite']))
                if counter_path.exists():
                    row['_counters']=json.loads(counter_path.read_text())
            cases.append(row)
    return cases


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--package',action='store_true',help='Package all readable .nsys-rep artifacts in a ZIP')
    args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'svg.fonttype':'none','figure.dpi':120})
    cases=read_cases()
    main_plan=json.loads((ROOT/'logs/overnight_20261103/plan.json').read_text())
    case_by_name={r['_name']:r for r in cases}
    stats=Counter(r['status'] for r in cases)
    traces=[r for r in cases if r.get('_analysis')]
    gpu_capacity_GiB=next((r['_counters']['GPU']['memory_bytes']/2**30 for r in traces if r.get('_counters')),32607/1024)
    primary=[c for c in main_plan if c.get('budget',128)==128 and c['tokens'] in (128,256,512)]
    missing=[c['name'] for c in main_plan if c['name'] not in case_by_name]
    benchmark={}
    for r in cases:
        a=r['arguments']
        if not a['trace'] and a['tokens']==512 and a.get('budget',128)==128:
            benchmark[(a['engine'],a['dataset'],a['bs'],a['arm'])]=r
    def result(engine,dataset,bs,arm):
        r=benchmark.get((engine,dataset,bs,arm))
        return r.get('results',{}).get(arm) if r and r['status']=='success' else None
    def speedup(engine,dataset,bs):
        ar,sd=result(engine,dataset,bs,'AR'),result(engine,dataset,bs,'JetSpec')
        return sd['tokens_per_second']/ar['tokens_per_second'] if ar and sd else None
    def trace(engine,dataset,bs):
        return next((r for r in traces if r['arguments']['engine']==engine and r['arguments']['dataset']==dataset
            and r['arguments']['bs']==bs and r['arguments'].get('budget',128)==128
            and not r.get('nsys_gpu_metrics_enabled')),None)
    figures=[]
    x=np.arange(7); width=.35
    fig,ax=plt.subplots(figsize=(10,4))
    for offset,arm,color in [(-width/2,'AR','#777777'),(width/2,'JetSpec','#2678b2')]:
        vals=[(result('optimized',d,1,arm) or {}).get('tokens_per_second',np.nan) for d in DATASETS]
        ax.bar(x+offset,vals,width,label=arm,color=color)
    ax.set_xticks(x,[NAMES[d] for d in DATASETS],rotation=22,ha='right')
    ax.set_ylabel('Output tokens / second'); ax.set_title('Upstream optimized BS=1 — uninstrumented, cap 512')
    ax.legend(); figures.append(plot('optimized_throughput'))
    fig,ax=plt.subplots(figsize=(9,4.5))
    for i,d in enumerate(DATASETS):
        vals=[speedup('batch_reference',d,b) for b in BS]
        ax.plot(BS,[v if v is not None else np.nan for v in vals],marker='o',color=COLORS[i],label=NAMES[d])
    ax.axhline(1,color='black',lw=.8,ls='--'); ax.set_xscale('log',base=2); ax.set_xticks(BS,BS)
    ax.set_xlim(.8,20)
    ax.set_xlabel('Initial batch size'); ax.set_ylabel('JetSpec / AR throughput'); ax.legend(ncol=2)
    ax.set_title('Same experimental batched implementation — cap 512, 30/32 prompts')
    figures.append(plot('batch_speedup'))
    phase_order=['prefill','draft_prepare','draft_forward','tree_build','kv_pack','target_prepare','verify_forward','accept','kv_commit','other']
    phase_colors={phase:plt.get_cmap('tab20')(i*2) for i,phase in enumerate(phase_order)}
    fig,ax=plt.subplots(figsize=(11,4.8)); bottoms=np.zeros(7)
    for phase in phase_order:
        vals=[]
        for d in DATASETS:
            r=trace('optimized',d,1)
            vals.append((r['_analysis']['phases'].get('JetSpec/'+phase,{}).get('kernel_ms',0)/
                         max(r['results']['JetSpec']['total_tokens'],1)) if r else np.nan)
        ax.bar(x,vals,bottom=bottoms,label=PHASE_LABELS[phase],color=phase_colors[phase])
        bottoms=np.nan_to_num(bottoms)+np.nan_to_num(vals)
    ar_vals=[]
    for d in DATASETS:
        r=trace('optimized',d,1)
        ar_vals.append(sum(v['kernel_ms'] for k,v in r['_analysis']['phases'].items() if k.startswith('AR/'))/
                       r['results']['AR']['total_tokens'] if r else np.nan)
    ax.plot(x,ar_vals,'ko--',label='AR total GPU kernel work')
    ax.set_xticks(x,[NAMES[d] for d in DATASETS],rotation=22,ha='right')
    ax.set_ylabel('GPU kernel work (ms / output token)'); ax.set_title('Native trace cohort: 2 prompts, cap 256')
    ax.legend(ncol=3,fontsize=8); figures.append(plot('native_gpu_phase_work'))
    fig,axes=plt.subplots(2,4,figsize=(14,7),sharex=True,sharey=True)
    for i,d in enumerate(DATASETS):
        ax=axes.flat[i]
        for arm,style in [('AR','o-'),('JetSpec','s--')]:
            vals=[((result('batch_reference',d,b,arm) or {}).get('peak_allocated_bytes',np.nan)/2**30) for b in BS]
            ax.plot(BS,vals,style,label=arm)
        ax.axhline(gpu_capacity_GiB,color='red',ls=':',lw=1,label='CUDA-visible capacity')
        ax.set_title(NAMES[d]); ax.set_xscale('log',base=2); ax.set_xticks(BS,BS)
        ax.set_xlim(.8,20)
        ax.set_ylabel('Peak allocated GiB'); ax.set_xlabel('Initial batch size')
    axes.flat[-1].axis('off'); axes.flat[0].legend(fontsize=8)
    fig.suptitle('Separate AR / JetSpec processes — genuine head-free AR memory baseline, cap 512')
    figures.append(plot('batch_peak_memory'))
    fig,axes=plt.subplots(1,2,figsize=(12,4.2))
    for ax,(engine,d,b) in zip(axes,[('optimized','gsm8k',1),('batch_reference','livecodebench',8)]):
        r=trace(engine,d,b)
        if r:
            records=r.get('phase_records',[])
            ax.plot([v['allocated_after']/2**30 for v in records],label='Allocated after phase')
            ax.plot([v['peak_allocated']/2**30 for v in records],alpha=.65,label='Transient phase peak')
            ax.plot([v['reserved_after']/2**30 for v in records],color='gray',ls='--',label='Allocator reserved')
            ax.set_title(f'{engine}, {NAMES[d]}, BS={b}')
        ax.set_xlabel('Instrumented phase index (AR first, then JetSpec)'); ax.set_ylabel('GiB'); ax.legend(fontsize=7)
    figures.append(plot('memory_phase_evolution'))
    counter_cases=[r for r in traces if r.get('_counters')]
    counter_table=[]
    if counter_cases:
        fig,axes=plt.subplots(1,3,figsize=(13,4),sharey=True)
        counter_metrics=['SMs Active [Throughput %]','Tensor Active [Throughput %]','DRAM Read Bandwidth [Throughput %]']
        counter_phases=['AR/decode_forward','JetSpec/draft_forward','JetSpec/verify_forward']
        for ax,metric in zip(axes,counter_metrics):
            for j,phase in enumerate(counter_phases):
                vals=[]
                for d in DATASETS:
                    r=next((r for r in counter_cases if r['arguments']['dataset']==d),None)
                    vals.append(r['_counters']['phases'][phase]['metrics_percent'][metric] if r else np.nan)
                ax.bar(x+(j-1)*.25,vals,.25,label=phase.split('/')[0]+' '+phase.split('/')[1].split('_')[0])
            ax.set_xticks(x,[NAMES[d] for d in DATASETS],rotation=45,ha='right',fontsize=8)
            ax.set_title(metric.split(' [')[0]); ax.set_ylim(0,100); ax.set_ylabel('Throughput %')
        axes[0].legend(fontsize=7)
        hardware_figure=plot('gpu_hardware_counters')
        for r in counter_cases:
            for phase,entry in r['_counters']['phases'].items():
                m=entry['metrics_percent']
                counter_table.append([NAMES[r['arguments']['dataset']],phase,
                    number(m['SMs Active [Throughput %]'],1),number(m['Tensor Active [Throughput %]'],1),
                    number(m['DRAM Read Bandwidth [Throughput %]'],1),entry['overlapping_sample_windows']])
        raw_counters=[]
        for r in counter_cases:
            for phase,entry in r['_counters']['phases'].items():
                raw_counters.append({'dataset':r['arguments']['dataset'],'phase':phase,
                    'overlapping_sample_windows':entry['overlapping_sample_windows'],**entry['metrics_percent']})
        with open(OUT/'gpu_hardware_counters.csv','w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(raw_counters[0])); w.writeheader(); w.writerows(raw_counters)
    else:
        hardware_figure=''
    kv_path=ROOT/'logs/kv_reservation_analysis.json'
    kv_analysis=json.loads(kv_path.read_text()) if kv_path.exists() else None
    kv_table=[]
    if kv_analysis:
        for r in sorted(kv_analysis['native_cases'],key=lambda r:(DATASETS.index(r['dataset']),r['arm'])):
            kv_table.append([NAMES[r['dataset']],r['arm'],r['session_prompt_capacity'],
                number(r['layer_local_compact_pool_bytes']/2**30,3),
                number(r['upstream_physical_pool_bytes']/2**30,3),
                number(r['measured_peak_allocated_bytes']/2**30,3)])
    native_rows=[]; batch_rows=[]; csv_rows=[]
    for engine,bs_values in [('optimized',[1]),('batch_reference',BS)]:
        for d in DATASETS:
            for b in bs_values:
                ar,sd=result(engine,d,b,'AR'),result(engine,d,b,'JetSpec')
                s=speedup(engine,d,b)
                statuses='/'.join((benchmark.get((engine,d,b,a)) or {}).get('status','pending') for a in ['AR','JetSpec'])
                row=[NAMES[d],b,number(ar['tokens_per_second'] if ar else None),number(sd['tokens_per_second'] if sd else None),
                     number(s),number(sd.get('mean_accept_length') if sd else None),
                     number(ar['peak_allocated_bytes']/2**30 if ar else None),number(sd['peak_allocated_bytes']/2**30 if sd else None),statuses]
                (native_rows if engine=='optimized' else batch_rows).append(row)
                csv_rows.append(dict(engine=engine,dataset=d,bs=b,AR_tps=ar['tokens_per_second'] if ar else None,
                    JetSpec_tps=sd['tokens_per_second'] if sd else None,speedup=s,
                    AR_peak_GiB=ar['peak_allocated_bytes']/2**30 if ar else None,
                    JetSpec_peak_GiB=sd['peak_allocated_bytes']/2**30 if sd else None,
                    AR_output_tokens=ar['total_tokens'] if ar else None,
                    JetSpec_output_tokens=sd['total_tokens'] if sd else None,
                    AR_seconds=ar['seconds'] if ar else None,JetSpec_seconds=sd['seconds'] if sd else None,
                    JetSpec_accept_length=sd['mean_accept_length'] if sd else None,status=statuses))
    with open(OUT/'benchmark_results.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(csv_rows[0])); w.writeheader(); w.writerows(csv_rows)
    source=json.loads((ROOT/'data/jetspec_paper/manifest.json').read_text())
    dataset_rows=[]
    for item in source['sources']:
        rows=json.loads((ROOT/'data/jetspec_paper'/f"{item['name']}.json").read_text())
        lengths=[r['prompt_tokens'] for r in rows]
        dataset_rows.append([NAMES[item['name']],len(rows),min(lengths),number(statistics.mean(lengths),1),max(lengths),
            item['truncated_rows'],html.escape(item['revision'][:12])])
    phase_rows=[]; phase_csv=[]
    for r in traces:
        a=r['arguments']; analysis=r['_analysis']
        for phase,v in analysis['phases'].items():
            method=phase.split('/')[0]; tokens=r['results'][method]['total_tokens']
            entry=dict(case=r['_name'],GPU_metrics_case=r.get('nsys_gpu_metrics_enabled',False),
                dataset=a['dataset'],engine=a['engine'],bs=a['bs'],tokens_cap=a['tokens'],budget=a.get('budget',128),
                phase=phase,calls=v['calls'],host_range_ms=v['host_ms'],gpu_kernel_ms=v['kernel_ms'],
                gpu_kernel_ms_per_output_token=v['kernel_ms']/max(tokens,1),memcpy_ms=v['memcpy_ms'],memcpy_bytes=v['memcpy_bytes'])
            phase_csv.append(entry)
            if a['engine']=='optimized' and a.get('budget',128)==128 and not r.get('nsys_gpu_metrics_enabled'):
                phase_rows.append([NAMES[a['dataset']],phase,v['calls'],number(v['host_ms']),number(v['kernel_ms']),
                    number(v['kernel_ms']/max(tokens,1),4),number(v['memcpy_bytes']/2**20)])
    if phase_csv:
        with open(OUT/'gpu_phase_results.csv','w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(phase_csv[0])); w.writeheader(); w.writerows(phase_csv)
    memory_rows=[]
    for r in traces:
        grouped=defaultdict(list)
        for row in r.get('phase_records',[]):grouped[row['phase']].append(row)
        for phase,records in grouped.items():
            memory_rows.append({'case':r['_name'],'phase':phase,'calls':len(records),
                'max_phase_peak_GiB':max(x['peak_allocated'] for x in records)/2**30,
                'max_transient_above_phase_entry_GiB':max(x['peak_allocated']-x['allocated_before'] for x in records)/2**30,
                'max_reserved_GiB':max(x['reserved_after'] for x in records)/2**30})
    if memory_rows:
        with open(OUT/'memory_phase_summary.csv','w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(memory_rows[0])); w.writeheader(); w.writerows(memory_rows)
    timeline_rows=[]; artifact_manifest=[]
    for r in sorted(traces,key=lambda r:(r['arguments']['engine'],DATASETS.index(r['arguments']['dataset']),r['arguments']['bs'],r['arguments'].get('budget',128))):
        a=r['arguments']; rep=Path(r['nsys_report']); db=Path(r['nsys_sqlite']); an=r['_analysis']
        artifact_manifest.append({'name':r['_name'],'dataset':a['dataset'],'engine':a['engine'],'bs':a['bs'],
            'budget':a.get('budget',128),'GPU_metrics':r.get('nsys_gpu_metrics_enabled',False),
            'nsys_rep':str(rep),'bytes':rep.stat().st_size,'sqlite':str(db),
            'kernel_events':an['kernel_events'],'kernel_attribution_fraction':an['kernel_attribution_fraction']})
        timeline_rows.append([NAMES[a['dataset']],a['engine'],a['bs'],a.get('budget',128),
            'yes' if r.get('nsys_gpu_metrics_enabled') else 'no',link(rep,'.nsys-rep'),
            number(rep.stat().st_size/2**20,1)+' MiB',link(db,'SQLite'),link(r['_analysis_path'],'phase JSON'),
            link(r['_path'],'运行/显存 JSON'),
            number(100*an['kernel_attribution_fraction'],2)+'%',an['cuda_memory_usage_events']])
    save(OUT/'timeline_manifest.json',artifact_manifest)
    quality={'expected_main_cases':len(main_plan),'main_cases_present':sum(c['name'] in case_by_name for c in main_plan),
        'missing_main_cases':missing,'outcomes':dict(stats),'readable_timeline_count':len(traces),
        'primary_trace_matrix_expected':42,'primary_trace_matrix_present':sum(
            1 for r in traces if not r.get('nsys_gpu_metrics_enabled') and r['arguments'].get('budget',128)==128),
        'minimum_kernel_attribution_fraction':min((r['_analysis']['kernel_attribution_fraction'] for r in traces),default=0),
        'captured_GPU_kernel_events':sum(r['_analysis']['kernel_events'] for r in traces),
        'generated_Beijing':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()}
    save(OUT/'coverage.json',quality)
    quality['GPU_counter_case_count']=len(counter_cases)
    initial=ROOT/'reports/2026-11-03_initial_0629/coverage.json'
    quality['initial_report_prepared_Beijing']=json.loads(initial.read_text())['generated_Beijing'] if initial.exists() else None
    save(OUT/'coverage.json',quality)
    complete=not missing and all(case_by_name[c['name']]['status'] in ('success','out_of_memory') for c in main_plan)
    header=['数据集','BS','AR tok/s','JetSpec tok/s','加速比','平均接受长度','AR 峰值 GiB','SD 峰值 GiB','状态 AR/SD']
    native_speedups=[speedup('optimized',d,1) for d in DATASETS if speedup('optimized',d,1) is not None]
    sections=[]
    sections.append('<h1>JetSpec / AR GPU 时间与空间行为</h1><p class="subtitle">RTX 5090 32GB · Qwen3-8B BF16 · '+
        datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M 北京时间')+'</p>')
    sections.append('<p class="notice">'+('实验矩阵已完成。' if complete else '进行中：本页是已落盘证据的阶段报告。')+
        f'主计划 {quality["main_cases_present"]}/{len(main_plan)} 个结果；{len(traces)} 份可读 timeline；'+
        f'GPU kernel 归因最低覆盖率 {quality["minimum_kernel_attribution_fraction"]:.2%}。</p>')
    sections.append('<p>本报告比较 JetSpec 自己引擎的 AR 与 learned tree speculative decoding；不是 SGLang 对照。'+
        '目标是解释 draft、verify、accept、KV 操作的 GPU 开销及批大小敏感性，不是完整复现论文速度或评测回答正确率。</p>')
    if native_speedups:
        sections.append(f'<p><b>当前已完成的原生 BS=1 加速范围：{min(native_speedups):.2f}–{max(native_speedups):.2f}×。</b>'+ 
            '这是独立、无 profiler 的 512-token-cap 测量；下方 GPU 分解来自另外的短 profile 样本。</p>')
    native_trace_cases=[trace('optimized',d,1) for d in DATASETS]
    native_trace_cases=[r for r in native_trace_cases if r]
    if native_trace_cases:
        verify_share=[]; draft_share=[]
        for r in native_trace_cases:
            phases=r['_analysis']['phases']
            total=sum(v['kernel_ms'] for k,v in phases.items() if k.startswith('JetSpec/'))
            verify_share.append(100*phases['JetSpec/verify_forward']['kernel_ms']/total)
            draft_share.append(100*phases['JetSpec/draft_forward']['kernel_ms']/total)
        sections.append('<h2>关键发现与优先方向</h2><ol>'+ 
            f'<li><b>GPU work 主要在 verify：</b>七个原生短 trace 中，verify 占 SD GPU kernel work 的 {min(verify_share):.1f}–{max(verify_share):.1f}%，draft 为 {min(draft_share):.1f}–{max(draft_share):.1f}%。这是设备 kernel 时长之和，不是 wall-clock 百分比，也不含 memcpy。</li>'+ 
            '<li><b>接受长度决定能否摊薄开销：</b>原生 32/30 条、cap=512 的测量中，GSM8K 为 '+number(speedup('optimized','gsm8k',1))+ 
            '×，LiveCodeBench 为 '+number(speedup('optimized','livecodebench',1))+'×，MT-Bench 为 '+number(speedup('optimized','mt_bench',1))+ 
            '×。不要用两条 profile 样本的接受率直接解释完整 32 条测量；完整接受长度在性能表中。</li>'+ 
            '<li><b>BS 增大后 AR 更容易摊薄权重访问，SD 相对收益下降：</b>reference 的 MT-Bench 从 BS=1 的 '+number(speedup('batch_reference','mt_bench',1))+ 
            '× 下降到 BS=16 的 '+number(speedup('batch_reference','mt_bench',16))+'×。这条曲线只对同一 reference 实现成立，不是原生引擎的 production batching 结论。</li>'+ 
            '<li><b>CUDA Graph 已启用仍有 host/KV 成本：</b>target graph 之外还有 tree build、accept 同步、KV commit。原生 KV commit 的 host range 常明显大于其 GPU kernel work；但 host range 也可能包含队列等待，不能全部算作 CPU 计算。timeline 中应同时看 CUDA API、kernel 和 memcpy。</li>'+ 
            '<li><b>优化实验优先级：</b>先处理下方证实的 KV 池预留布局；再针对 verify 的主要 BF16 GEMM 形状与分离的 Q/K/V、gate/up 投影评估融合及实现选择；之后减少 commit/accept 的主机往返。较低接受率的数据集应独立调整 tree budget，不能一概扩树。</li></ol>')
    sections.append('<h2>实验口径与可解释范围</h2><ul>'+ 
        '<li>原生优化路径：BS=1，target/draft/AR 专用 torch.compile forward + CUDA Graph，Triton paged attention，session/pool 复用。GEMM weight fusion、nogather 和额外 accept graph 关闭。</li>'+ 
        '<li>BS sweep：上游 learned drafter 不支持 batched tree API，因此使用本仓库的真实 batched reference。每轮所有活跃序列共享一次 draft forward 和一次 target verify forward；AR 使用相同 dense padded KV / SDPA 实现。它没有 CUDA Graph，不代表成熟 serving 引擎的并发性能。</li>'+ 
        '<li>BS=1/2/4/8/16 是初始 batch size，EOS 后活跃序列会减少。prefill 按序列执行；decode/draft/verify 真正批处理。不能把原生 BS=1 与 reference BS&gt;1 连成一条性能曲线。</li>'+ 
        '<li>每数据集取固定前 32 条，AIME25 全部 30 条；MT-Bench 仅第一轮，LiveCodeBench release_v5。thinking=false、temperature=0、EOS stopping。输入上限 2048 tokens，是否截断逐条记录。</li>'+ 
        '<li>性能实验：AR 和 SD 分进程运行，AR 不加载 draft head；不带 NVTX/事件/内存分段采样，排除模型加载、tokenization 和预热，包含 prefill+decode；原生路径还包含 detokenization。单次测量，样本异质性不作为统计置信区间。</li>'+ 
        '<li>Timeline：原生每数据集 2 条、cap=256；batch reference 至多 2 个 batch、cap=128。预热后才开启 CUDA collection，node 级 graph tracing 和 CUDA memory usage 开启。profiler 会增加开销，trace tok/s 不用于性能结论。</li>'+ 
        '<li>GPU 时间按 CUDA launch correlation 归因到 NVTX 阶段，再求设备 kernel 时长。host range 含等待，不等于 CPU 计算时间；kernel work 不是请求延迟，多 stream 时求和可能重叠。无法细分的采样/控制 kernel 归入 Other。</li>'+ 
        '<li>BF16 AR 与 tree 输出存在已记录的数值分歧，性能结果不证明逐 token losslessness。新增 batch reference 通过小模型 FP32 的 HF greedy、ragged isolation 与 batch/single 一致性检查。未执行生成代码。</li>'+ 
        '<li>共享 trace 进程中 AR 时间段仍驻留 draft/SD graph 对象；真正的 AR 显存基线请看分进程 benchmark 峰值。allocated 是活跃 PyTorch 分配，reserved 是 caching allocator 保留量，二者不能混用。</li></ul>')
    sections.append('<h2>原生优化路径：数据集差异</h2>'+figures[0]+table(header,native_rows)+figures[2])
    sections.append('<p>verify 在一次 forward 中处理整棵树，权重访问被多个候选 token 分摊；其单轮 GPU 成本应与每轮实际接受长度一起看。draft 与 verify 在当前单 GPU 实现中顺序执行，不存在 JetSpec++ 的跨卡重叠。</p>')
    sections.append('<h2>真实 batch sweep</h2>'+figures[1]+table(header,batch_rows)+figures[3])
    sections.append('<p>OOM 是本实现与输入/生成长度下的容量边界，不等于 JetSpec 方法或生产引擎的普遍上限。reference 会反复构建 dense padded KV，KV packing/commit 可能主导时间或临时内存；请结合阶段 CSV 与 timeline 判断。</p>')
    sections.append('<h2>空间行为与阶段开销</h2>'+figures[4]+ 
        '<p>Qwen3-8B 的 target KV 理论增量为每 request-token 2×36×8×128×2 bytes = <b>144 KiB</b>；5 个 target taps 的 BF16 feature 约 <b>40 KiB</b>/context-token。draft head 参数约 1.95 GiB；图缓冲、padding、dense KV 复制和临时 logits 使实际峰值更高。图像的横轴是采样阶段序号，不是等间隔 GPU 时间。</p>'+ 
        table(['数据集','NVTX 阶段','调用数','Host range ms','GPU kernel ms','GPU ms/output token','Memcpy MiB'],phase_rows))
    if kv_analysis:
        sections.append('<h2>KV 池预留放大：实现层面的显存机会</h2>'+ 
            '<p><b>当前原生 layout 的预留 pool，相对 layer-local block 编号的等容量布局有 36 倍放大。</b>'+ 
            '原因是 reserve_capacity 先把 block 数乘以层数，而 _grow_pool/_ensure_layer 又让每一层都分配这个完整全局 ID 范围。每层实际只使用其中自己的 block ID，其他位置仍占物理 tensor 空间。'+ 
            '这是这份上游代码的布局选择，不是 paged KV 或 JetSpec 必须如此。</p>'+ 
            '<p>已用原始 PagedKVCache 在 CPU 上做 3 层运行检查，观察到精确 3 倍放大；再按 Qwen3 的 36 层与各 case 的固定 reservation 推导下表。'+ 
            '“紧凑布局”是等容量 pool 的计算基线，不是已经实现或 benchmark 过的优化。总峰值还含权重、head、graph 和临时分配。'+link(kv_path,'验证证据与公式')+'</p>'+ 
            table(['数据集','Arm','Prompt capacity','等容量 layer-local pool GiB','当前原生 pool GiB（推导）','实测总峰值 GiB'],kv_table)+ 
            '<p>可优先研究 layer-local ID / 跨层共享逻辑 ID 的寻址方式，同时保持 CUDA Graph 的地址稳定。不能只把 reservation 数除以 36 而不修改全局 ID 分配器和 block tables。</p>')
    if counter_cases:
        sections.append('<h2>硬件计数器：AR 与 draft/verify 的 GPU 行为差异</h2>'+hardware_figure+ 
            '<p>这 7 个额外 profile 同时采了 1 kHz GPU metrics。以 GSM8K 为例，AR decode 的 DRAM read throughput 约 80.8%、Tensor active 约 1.0%；verify 约为 44.4% 与 43.3%。'+ 
            '这支持 AR 更受权重读取影响、verify 把更多工作转向 Tensor Core GEMM 的判断。draft 的 SM active 约 60.3%，低于 AR/verify 的约 85–87%，其较小矩阵和混合操作仍有调优空间。</p>'+ 
            '<p>采样窗口按与真实 GPU kernel 区间的重叠时长加权，使用 CUDA correlation 归因；没有把很短的 CPU graph-launch NVTX 区间误当作 graph 的 GPU 执行时间。'+ 
            '窗口会混合相邻阶段和空闲，不能作为单 kernel 的精确 occupancy 或 Nsight Compute roofline 结论。指标均保留 NVIDIA 的 throughput % 定义。</p>'+ 
            table(['数据集','GPU 阶段','SM active %','Tensor active %','DRAM read %','重叠采样窗口数'],counter_table))
    budget_rows=[]; length_rows=[]
    for d in DATASETS:
        ar=result('optimized',d,1,'AR')
        for budget in [64,128,256]:
            candidate=next((r for r in cases if r['status']=='success' and not r['arguments']['trace'] and
                r['arguments']['engine']=='optimized' and r['arguments']['dataset']==d and
                r['arguments']['arm']=='JetSpec' and r['arguments']['tokens']==512 and r['arguments'].get('budget',128)==budget),None)
            if candidate:
                sd=candidate['results']['JetSpec']
                budget_rows.append([NAMES[d],budget,number(sd['tokens_per_second']),
                    number(sd['tokens_per_second']/ar['tokens_per_second']) if ar else '—',
                    number(sd['mean_accept_length']),number(sd['peak_allocated_bytes']/2**30)])
        arms={}
        for r in cases:
            a=r['arguments']
            if r['status']=='success' and a['engine']=='optimized' and a['dataset']==d and a['tokens']==1024 and not a['trace']:
                arms[a['arm']]=r['results'][a['arm']]
        if all(a in arms for a in ['AR','JetSpec']):
            length_rows.append([NAMES[d],number(arms['AR']['tokens_per_second']),number(arms['JetSpec']['tokens_per_second']),
                number(arms['JetSpec']['tokens_per_second']/arms['AR']['tokens_per_second']),
                number(arms['JetSpec']['peak_allocated_bytes']/2**30)])
    sections.append('<h2>补充实验：树预算与更长生成</h2>'+ 
        '<p>预算 sweep 使用相同 32/30 条 prompt、cap=512，AR 复用同数据集的独立 AR baseline。不同 budget 可能通过 BF16 计算路径改变输出 token，表中给实际 tok/s。</p>'+ 
        table(['数据集','Tree budget','JetSpec tok/s','相对 AR','平均接受长度','SD 峰值 GiB'],budget_rows)+ 
        '<p><b>扩树并不总是更快：</b>这组实际测量中 budget=256 在全部七个数据集都慢于 budget=128；MT-Bench 的 budget=64 为 114.2 tok/s，比 128 的 101.3 tok/s 更快，虽然平均接受长度略低。</p>'+ 
        '<p>下表 cap=1024 使用前 16 条、budget=128；其样本集合与主表 32/30 条不同，因此不是只改变输出长度的严格单因素对照。</p>'+ 
        table(['数据集','AR tok/s','SD tok/s','相对 AR','SD 峰值 GiB'],length_rows))
    kernel_rows=[]
    for engine,d,b in [('optimized','gsm8k',1),('batch_reference','gsm8k',1),('batch_reference','gsm8k',16)]:
        r=trace(engine,d,b)
        if not r: continue
        for phase in ['AR/decode_forward','JetSpec/draft_forward','JetSpec/verify_forward']:
            v=r['_analysis']['phases'].get(phase,{})
            for k in v.get('top_kernels',[])[:2]:
                kernel_rows.append([engine,b,phase,html.escape(k['name'][:180]),number(k['kernel_ms']),k['count']])
    sections.append('<h2>Kernel 证据</h2>'+table(['实现','BS','阶段','主要 kernel（完整名在 phase JSON）','GPU ms','调用数'],kernel_rows)+ 
        '<p>phase JSON 同时保留 launch grid/block、registers/thread、静态 shared memory 的常见配置。这些是 launch 配置，不能直接当作实际 SM occupancy。CPU perf sampling 受系统权限限制，未用于本报告。</p>')
    sections.append('<h2>Timeline 文件索引</h2><p>用 Nsight Systems GUI 2026.5.1 或兼容的新版本打开 .nsys-rep。'+
        '优先看 native GSM8K 与 MT-Bench，再对照 reference 同一数据集的 BS=1/8/16。可搜索表格过滤文件。</p>'+ 
        '<input id="filter" placeholder="搜索 dataset / engine / BS / budget" oninput="filterRows()">'+
        '<div id="timelines">'+table(['数据集','实现','BS','Budget','GPU metrics','.nsys-rep','大小','数据库','阶段分析','运行/显存','Kernel 归因','CUDA memory events'],timeline_rows)+'</div>')
    sections.append('<h2>数据与复现</h2>'+table(['数据集','样本数','最短 prompt','平均 prompt','最长 prompt','截断条数','HF revision'],dataset_rows)+ 
        '<p>'+link(ROOT/'data/jetspec_paper/manifest.json','数据 manifest')+' · '+link(ROOT/'logs/batch_reference_validation.json','FP32 correctness checks')+' · '+
        '<a href="benchmark_results.csv">性能 CSV</a> · <a href="gpu_phase_results.csv">GPU 阶段 CSV</a> · '+
        '<a href="memory_phase_summary.csv">显存阶段 CSV</a> · <a href="gpu_hardware_counters.csv">硬件计数器 CSV</a> · '+
        '<a href="timeline_manifest.json">Timeline manifest</a> · <a href="coverage.json">覆盖情况</a></p>'+ 
        '<pre>cd ~/profile_spec\nsource env.sh\npython run_overnight.py  # 已有结果自动续跑；运行中不要重复启动\npython build_overnight_report.py --package</pre>'+ 
        '<p>论文数据集口径参考 <a href="https://arxiv.org/html/2606.18394v2">JetSpec 论文 v2</a>。源码和版本在本地保存；未 push。后台 supervisor 每 2 秒检查计算进程，发现其他 GPU 作业会终止自己的当前 case、释放显存并等待重试。</p>')
    if args.package:
        archive=OUT/'nsys_timelines.zip'
        with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_STORED) as z:
            for r in artifact_manifest: z.write(r['nsys_rep'],arcname=Path(r['nsys_rep']).name)
            z.write(OUT/'timeline_manifest.json',arcname='timeline_manifest.json')
        sections.append('<p><a href="nsys_timelines.zip">下载全部 .nsys-rep ZIP</a>（'+number(archive.stat().st_size/2**20,1)+' MiB）。SQLite 留在本地目录，未打包。</p>')
    css='body{font:16px/1.65 system-ui,sans-serif;max-width:1500px;margin:36px auto;padding:0 24px;color:#18212b}h1{font-size:30px}h2{margin-top:42px}.subtitle{color:#536475}.notice{background:#e9f2fa;padding:14px;border-left:4px solid #2678b2}table{border-collapse:collapse;width:100%;font-size:13px;display:block;overflow:auto;margin:16px 0}th,td{padding:8px 10px;border-bottom:1px solid #d9e1e9;text-align:left;vertical-align:top}th{background:#eef2f6;white-space:nowrap}tr:nth-child(even){background:#fafbfd}a{color:#155e9d}img{max-width:100%;height:auto}figure{margin:22px 0}pre{background:#f2f5f8;padding:16px;overflow:auto}input{padding:10px;min-width:360px}li{margin-bottom:9px}'
    javascript='function filterRows(){let q=document.getElementById("filter").value.toLowerCase();document.querySelectorAll("#timelines tbody tr").forEach(r=>r.style.display=r.textContent.toLowerCase().includes(q)?"":"none")}'
    (OUT/'report.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>JetSpec GPU profile — 2026-11-03</title><style>'+css+'</style><body>'+''.join(sections)+'<script>'+javascript+'</script></body></html>')
    text=['JetSpec GPU profile report — Beijing '+quality['generated_Beijing'],
          'HTML report: '+str(OUT/'report.html'),json.dumps(quality,ensure_ascii=False,indent=2),
          'Main native BS=1 speedups: '+str({NAMES[d]:speedup('optimized',d,1) for d in DATASETS}),
          'Key results: native BS=1 speedup 1.07–2.12x; reference MT-Bench BS=16 is 0.75x. Verify accounts for approximately 78–82% of SD GPU kernel work in the short native traces.',
          'Memory: source and CPU runtime check identify a 36x reserved KV-pool layout amplification relative to layer-local equal-capacity pools on the 36-layer target. This is an implementation-specific reservation finding, not a measured speedup from a proposed fix.',
          'Batch sweep uses true batched experimental dense-KV/SDPA implementation, not native CUDA Graph BS>1.',
          'No SGLang comparison. BF16 token equality is not guaranteed. Trace wall times are instrumented; use separate benchmark CSV.',
          'All .nsys-rep links and stage/memory details are in report.html.']
    (OUT/'report.txt').write_text('\n\n'.join(text)+'\n')
    print(json.dumps(quality,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__': main()
