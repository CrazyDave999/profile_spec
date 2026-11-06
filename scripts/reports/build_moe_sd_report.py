"""Render only measured cells; incomplete campaigns stay explicitly incomplete."""
import _bootstrap
import csv
from datetime import datetime, timedelta, timezone
import html
import json
from pathlib import Path

ROOT=_bootstrap.ROOT
OUT=ROOT/"reports/2026-11-06_moe_parallel_sd"
LOGS=ROOT/"logs/moe_sd_20261106"


def read_csv(name):
    path=OUT/name
    return list(csv.DictReader(path.open())) if path.exists() else []


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"svg.fonttype":"none","axes.spines.top":False,"axes.spines.right":False})
    data=json.loads((OUT/"summary.json").read_text()) if (OUT/"summary.json").exists() else {"rows":[],"campaigns":[]}
    rows=data["rows"];figures=[]

    def figure(fig,name,caption):
        fig.tight_layout();fig.savefig(OUT/(name+".svg"),bbox_inches="tight");fig.savefig(OUT/(name+".png"),bbox_inches="tight",dpi=160);plt.close(fig)
        figures.append((name,caption))

    memory_path=OUT/"checkpoint_memory.json"
    if memory_path.exists():
        memory=json.loads(memory_path.read_text());target=memory['checkpoints']['target']['categories_bytes']
        fig,ax=plt.subplots(figsize=(9,2.6));left=0.
        for key,color,label in [('expert_packed_FP4','#147d92','FP4 expert values'),('expert_E8M0_scales','#458b50','Expert scales'),('expert_BF16_biases','#98a978','Expert biases'),('attention_norm_or_other','#de8a28','Attention / norm / other'),('router','#a98c78','Router'),('embedding_and_LM_head','#805ea3','Embedding / head')]:
            value=target[key]/2**30;ax.barh([0],[value],left=left,color=color,label=label);left+=value
        ax.set_yticks([0],['GPT-OSS-20B']);ax.set_xlabel('Checkpoint tensor GiB; not runtime VRAM');ax.legend(ncol=3,loc='upper center',bbox_to_anchor=(.5,1.65),fontsize=8);ax.grid(axis='x',alpha=.2)
        figure(fig,'checkpoint_storage','来自实际 safetensors tensor 元数据的存储分解；不是运行峰值显存或 HBM 传输量。')

    full=[r for r in rows if r["stage"]=="full" and r["method"]=="dflash" and r["gain"] is not None]
    if full:
        fig,axes=plt.subplots(1,3,figsize=(13,4))
        for dataset,color in [("gsm8k","#147d92"),("humaneval","#458b50"),("mt_bench","#de8a28")]:
            selected=sorted([r for r in full if r["dataset"]==dataset],key=lambda r:r["bs"])
            for ax,field,title in zip(axes,["gain","tau","request_mean_cycle_median_ms"],["Throughput gain vs matched AR","Committed decode tokens / round","Median request-mean cycle (ms)"]):
                ax.plot([r["bs"] for r in selected],[r[field] for r in selected],"o-",label=dataset,color=color);ax.set_title(title);ax.set_xlabel("Batch size");ax.set_xticks([1,4,16]);ax.grid(alpha=.2)
        axes[0].axhline(1,color="gray",linestyle="--");axes[0].legend()
        figure(fig,"scaling","GPT-OSS-20B MXFP4、DFlash width8；吞吐与请求周期来自无 profiler 的独立进程。")
    dense=[r for r in rows if r['stage']=='dense_control' and r['method']=='dflash' and r['gain'] is not None]
    if dense:
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        for ax,dataset in zip(axes,('gsm8k','mt_bench')):
            for label,cohort,color in [('GPT-OSS-20B MXFP4',full,'#147d92'),('Qwen3-8B BF16',dense,'#de8a28')]:
                selected=sorted([r for r in cohort if r['dataset']==dataset],key=lambda r:r['bs'])
                ax.plot([r['bs'] for r in selected],[r['gain'] for r in selected],'o-',label=label,color=color)
            ax.set_title(dataset+' / DFlash width8');ax.set_xlabel('Batch size');ax.set_ylabel('Gain vs own matched AR');ax.set_xticks([1,4,16]);ax.axhline(1,color='gray',linestyle='--');ax.grid(alpha=.2);ax.legend(fontsize=8)
        figure(fig,'dense_comparison','本次重跑的同宽度对照；跨模型还改变了精度、draft 和 tokenizer 等因素，不能将差异全部归因于 MoE。')
    widths=[r for r in rows if r["stage"]=="width" and r["method"]=="dflash" and r["gain"] is not None]
    if widths:
        fig,axes=plt.subplots(2,3,figsize=(13,7))
        for axisrow,dataset in zip(axes,["gsm8k","mt_bench"]):
            for bs,color in [(1,"#147d92"),(4,"#458b50"),(16,"#de8a28")]:
                selected=sorted([r for r in widths if r["dataset"]==dataset and r["bs"]==bs],key=lambda r:r["width"])
                for ax,field,title in zip(axisrow,["gain","tau","request_mean_cycle_median_ms"],["Gain","Acceptance / progress","Cycle ms"]):
                    ax.plot([r["width"] for r in selected],[r[field] for r in selected],"o-",label="BS"+str(bs),color=color);ax.set_title(dataset+" / "+title);ax.set_xticks([2,4,8]);ax.set_xlabel("Verify width");ax.grid(alpha=.2)
            axisrow[0].legend()
        figure(fig,"width_tradeoff","同一个 DFlash checkpoint，仅缩短验证宽度；保留跨宽度的 token 分歧。")
    components=[r for r in read_csv('target_components.csv') if r['scope']=='full_active_batch']
    if components:
        fig,ax=plt.subplots(figsize=(12,max(4,len(components)*.48)));left=[0.]*len(components)
        groups=[('Native FP4 expert GEMM',['MoE_FP4_GEMM_ms'],'#147d92'),('Router',['router_linear_ms','router_select_ms'],'#458b50'),('MoE dispatch / activation / quant / combine',['moe_metadata_ms','moe_dispatch_ms','moe_activation_ms','moe_combine_ms','activation_quantization_ms'],'#de8a28'),('QKV/O',['QKV_O_projection_ms'],'#5b94b7'),('LM head',['LM_head_ms'],'#805ea3'),('Attention core',['attention_core_ms'],'#b8a244'),('Other',['other_ms'],'#8997a5')]
        for label,keys,color in groups:
            values=[sum(float(r[k]) for k in keys) for r in components]
            ax.barh(range(len(components)),values,left=left,label=label,color=color);left=[a+b for a,b in zip(left,values)]
        ax.set_yticks(range(len(components)),[f"{r['method']} BS{r['configured_bs']} W{r['width']}" for r in components]);ax.invert_yaxis();ax.set_xlabel('GPU kernel ms / target forward, full active batch');ax.legend(fontsize=8,ncol=2);ax.grid(axis='x',alpha=.2)
        figure(fig,'target_components','仅选择实际活跃 BS 等于标称值的 target forward。每轮核对49个 BF16投影/head、48个原生FP4专家GEMM、24个router线性算子及24个路由选择；时间为GPU kernel之和。')
    routing=read_csv("routing_summary.csv")
    if routing:
        fig,ax=plt.subplots(figsize=(8,4))
        for phase,color in [("DECODE","#147d92"),("TARGET_VERIFY","#de8a28")]:
            selected=[r for r in routing if r["phase"]==phase and int(r['request_rows'])==int(r['configured_bs']) and int(r['routed_token_rows'])==int(r['configured_bs'])*int(r['width'])]
            ax.scatter([int(r["routed_token_rows"]) for r in selected],[float(r["mean_unique_experts"]) for r in selected],label=phase,color=color,alpha=.65)
        xx=list(range(1,129));ax.plot(xx,[32*(1-(1-4/32)**x) for x in xx],"--",color="gray",label="Uniform independent reference")
        ax.set_xscale('log',base=2);ticks=[1,2,4,8,16,32,64,128];ax.set_xticks(ticks,ticks)
        ax.set_xlabel("Routed target token rows / forward (log2)");ax.set_ylabel("Mean distinct experts / layer");ax.set_ylim(0,33);ax.legend();ax.grid(alpha=.2)
        figure(fig,"expert_coverage","独立的路由记录诊断，仅取完整活跃 batch，不用于性能计时；同样 token rows 可有不同的请求内/请求间相关性，虚线仅为均匀独立路由参照。")
    status=[]
    admissions=[LOGS/(data['sequence']+'.admission.json')] if data.get('sequence') else sorted(LOGS.glob('*.admission.json'))[-1:]
    for path in admissions:
        if not path.exists():continue
        s=json.loads(path.read_text());status.append(path.stem+": "+s["status"])
    table="<table><tr>"+"".join("<th>"+x+"</th>" for x in ["stage","任务","BS","width","tok/s","gain","τ","周期 ms","全进程物理峰值 GiB"])+"</tr>"
    for r in rows:
        if r["method"]=="ar":continue
        values=[r["stage"],r["dataset"],r["bs"],r["width"],f'{r["median_tps"]:.1f}',f'{r["gain"]:.3f}×' if r["gain"] is not None else "等待 AR bracket",f'{r["tau"]:.3f}',f'{r["request_mean_cycle_median_ms"]:.3f}',f'{r["peak_physical_GiB"]:.3f}']
        table+="<tr>"+"".join("<td>"+html.escape(str(x))+"</td>" for x in values)+"</tr>"
    table+="</table>"
    views="".join(f'<figure><img src="{OUT.name}/{name}.svg" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption></figure>' for name,caption in figures)
    links=[]
    for name in ["benchmarks.csv","source_manifest.json","correctness.json","cross_configuration_parity.json","checkpoint_memory.json","target_components.csv","target_components_evidence.json","routing_summary.csv","routing_layers.csv","routing_evidence.json","trace_phases.csv","trace_by_active_bs.csv","trace_kernels.csv","trace_counters.csv","trace_evidence.json","wave_tails.csv","wave_tails_evidence.json","numerics.json","findings.json"]:
        if (OUT/name).exists():links.append(f'<a href="{OUT.name}/{name}">{name}</a>')
    doc='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>2026-11-06 · MoE Parallel SD / GPT-OSS-20B</title><style>body{font:16px/1.8 system-ui;background:#eef2f6;color:#203247;margin:0}main{max-width:1250px;background:white;margin:25px auto;padding:30px}img{max-width:100%}.scroll{overflow:auto}table{border-collapse:collapse}td,th{padding:8px;border-bottom:1px solid #ddd;white-space:nowrap}.status{background:#fff5e6;padding:15px}a{color:#147d92}@media(max-width:600px){main{margin:0;padding:15px}}</style><main><a href="index.html">报告索引</a><h1>MoE Parallel SD：GPT-OSS-20B MXFP4</h1><p>2026-11-06（北京时间）· X6 单 RTX 5090 · TP1 · SGLang offline Engine。</p>'''
    doc+=f'<p class="status">已成功汇总 {len(rows)} 个无 profiler benchmark cell。状态：{html.escape("; ".join(status) or "尚未启动 admission")}。未完成的实验不视为结论。</p>'
    recorded_stages=[json.loads((ROOT/c['file']).read_text()) for c in data.get('campaigns',[])]
    if recorded_stages:
        cst=timezone(timedelta(hours=8))
        begin=min(c['started_epoch'] for c in recorded_stages)
        ended=[c['ended_epoch'] for c in recorded_stages if 'ended_epoch' in c]
        fmt=lambda epoch:datetime.fromtimestamp(epoch,cst).strftime('%Y-%m-%d %H:%M:%S %z')
        doc+=f'<p>报告按实验开始日归档。GPU 主队列开始：{fmt(begin)}；已结束阶段最近完成：{fmt(max(ended)) if ended else "等待阶段完成"}；本页更新：{fmt(datetime.now().timestamp())}。模板日期仍固定为2026-11-06。</p>'
    doc+='<p>本次采用 <a href="https://huggingface.co/z-lab/gpt-oss-20b-DFlash">z-lab/gpt-oss-20b-DFlash</a> 的普通 DFlash checkpoint，未启用需要 selector 权重的 DFlash2。补充核查了公开 DFlash2、Domino、DSpark 模型名列表，未找到匹配 GPT-OSS-20B 的可直接运行 checkpoint；检索范围及其局限保存在支持核查记录中。DFlare 有匹配权重，但当前固定版本没有其 QwenDFlareDraftModel 实现。</p>'
    trace_path=OUT/'trace_evidence.json'
    if trace_path.exists():
        traces=json.loads(trace_path.read_text())['cases'];native=[r for r in traces if r.get('verified_native_FP4_target_GEMMs')]
        doc+=f'<p>已汇总 {len(traces)} 个预热 trace，其中 {len(native)} 个包含已执行的原生 SM120 block-scaled FP4 Tensor Core GEMM。依据是完整kernel类型中明确的 MainloopSm120、SM120::BLOCKSCALED、E2M1/E4M3操作数和E8M0 scale；setup/quantization kernel不会单独作为加速证据。</p>'
    doc+='''<p>沿用先前离线协议：每组使用同样的16条原始任务输入。GPT-OSS 使用自身 Harmony 模板重新 tokenize，medium reasoning；模板日期固定为2026-11-06，输入 IDs 提前保存，避免 GPU 等待跨日改变输入。Qwen3-8B dense 对照沿用前次的 non-thinking 模板，因此跨模型也改变了输出模式。固定输出128 tokens、忽略 EOS、重复3次。加载、编译、warmup 和 cache flush 不计时；每任务/BS组前后各跑 AR。Decode CUDA Graph 与 overlap 开启，prefill graph 和 radix/prefix reuse 关闭。</p><p>仅采用配对且被当前实现支持的 checkpoint。GPT-OSS-20B 的 DFlash 已准备；记录的搜索未找到匹配 Domino/DSpark，DFlare 权重使用另一种架构，当前固定 SGLang 没有相应模型实现。不能把 checkpoint 换名后当作已支持的方法。</p><p>GPT-OSS 的 sliding/full softmax attention 不包含 GDN recurrent state。MoE/dense 跨模型比较同时改变了架构、精度、激活参数量、tokenizer 和 draft，不能把 gain 差值全部归因于 MoE。重点使用同模型的 BS/width 对照、实际专家路由与 kernel 成本解释。</p><p>权重 tensor 存储中，FP4 专家值约8.90 GiB、E8M0 scales约0.56 GiB，其余约3.36 GiB主要为BF16部分。单步 AR 只访问被路由到的专家，因此不能用总权重大小除以显存带宽直接估算 decode 时间；实际 footprint 和重用仍需路由与 profiler 证据。</p>'''
    from moe_sd_findings import render
    doc+=render(data,components,routing,read_csv('wave_tails.csv'),OUT)
    doc+='<h2>各 SD 配置的测量表与图</h2><p>下表汇总 SD 配置；含前后 AR 基线的全部75个 cell 见 <a href="'+OUT.name+'/benchmarks.csv">benchmarks.csv</a>。</p><div class="scroll">'+table+'</div>'+views
    doc+='''<p>路由记录和 nsys 是独立诊断，不进入 headline 排名。Kernel 时间之和是 GPU 工作量，不是端到端关键路径；CPU/GPU 时间不能直接相加。Distinct expert bytes 是依据 checkpoint 得出的访问集合代理，不是实测 HBM 传输字节。Tensor/SM 活跃率不能单独证明 compute bound。</p><p>保留 AR/SD、跨 BS/width 和配置内重复的输出 IDs 与首次分歧；固定长度吞吐不等同任务质量。代表性原始 nsys-rep 留在忽略的 profiles/moe_sd_20261106/。</p>'''
    doc+='<p>表中物理显存峰值覆盖加载、编译、warmup 和测试整个进程生命周期，含 allocator 保留、临时重排及 CUDA Graph 等开销。它不是稳态推理显存；AR 与 SD 的两个生命周期峰值不能直接相减，作为 draft 权重、hidden-state 或 KV cache 的额外大小。</p>'
    bundle=ROOT/'profiles/moe_sd_20261106/representative_traces.zip'
    if bundle.exists():
        doc+=f'<p><a href="../profiles/moe_sd_20261106/representative_traces.zip">本机下载5份代表性 nsys-rep 与分析（ZIP，{bundle.stat().st_size/2**20:.1f} MiB）</a> · <a href="../logs/moe_sd_20261106/representative_traces_manifest.json">原始 trace / ZIP SHA256 清单</a>。profiles/ 中的原始 profiler 文件保留在实验机器，不随 Git 提交。需 Nsight Systems 2026.5.1 或兼容更新版本；单独选择 AR/DF 的 BS1/16 和 BS16 width2，用于比较 target 成本及验证宽度。</p>'
        doc+='<p>建议先对比 AR BS1/16，再对比 DFlash W8 BS1/16，最后看 BS16 W2 与 W8。展开 CUDA Graph 的 node 以查看真实 kernel；target 的 CPU NVTX 为 step[DECODE …] 或 step[VERIFY …]，可搜索 MainloopSm120 定位原生 FP4 专家 GEMM。与 DRAFT、decode_control 对照时，按 CPU launch correlation 归属 GPU 工作，不要仅按 CPU NVTX 的横向时间边界截取；二者异步，可能重叠。</p>'
    doc+='<p>'+' · '.join(links)+'</p><p><a href="../logs/moe_sd_20261106/protocol.json">实验协议</a> · <a href="../logs/moe_sd_20261106/checkpoint_support_survey.json">Checkpoint 支持核查</a> · <a href="../models-moe-sd.json">权重 revision / tensor 元数据</a></p></main></html>'
    OUT.with_suffix(".html").write_text(doc)
    from build_report_index import write_index
    write_index()
    print("report",OUT.with_suffix(".html"),"figures",len(figures),"benchmark_cells",len(rows))


if __name__=="__main__":main()
