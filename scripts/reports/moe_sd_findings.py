"""Render measured conclusions from the current MoE cohort, including partial runs."""
import _bootstrap
import html
import csv
from collections import defaultdict
import json
from pathlib import Path
import statistics


def table(headers, records):
    return '<div class="scroll"><table><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in records)+'</table></div>'


def render(data, components, routing, tails, out):
    rows=data['rows'];parts=['<h2>当前测得的主要结论</h2>'];evidence={}
    full=[r for r in rows if r['stage']=='full' and r['method']=='dflash' and r['gain'] is not None]
    evidence['headline']=full
    if full:
        parts.append('<p>同一套 GPT-OSS-20B MXFP4 target 与 DFlash width8，在不同任务上的加速比并不相同。下面全部使用前后两个 AR 的中位吞吐作为基线；尚缺配对基线的点不填数。</p>')
        records=[]
        for dataset in ('gsm8k','humaneval','mt_bench'):
            selected={r['bs']:r for r in full if r['dataset']==dataset}
            if not selected:continue
            records.append([dataset]+[f"{selected[b]['gain']:.3f}× / τ={selected[b]['tau']:.3f}" if b in selected else '待完成' for b in (1,4,16)])
        parts.append(table(['任务','BS1：gain / τ','BS4：gain / τ','BS16：gain / τ'],records))
        parts.append('<p>τ 是去掉 prefill 首 token 后，每个请求 verify 的平均提交 token 数；包括每轮额外提交的 target token，不等于纯 draft 接受数。固定长度离线 wave 的吞吐还受 prefill、请求完成尾部和 API 开销影响，不能把全局 τ 乘以“请求平均周期的中位数”当作精确的吞吐公式。</p>')
    component_index={(r['method'],int(r['configured_bs']),int(r['width'])):r for r in components}
    required=[('ar',1,1),('dflash',1,8),('ar',16,1),('dflash',16,8)]
    if all(k in component_index for k in required):
        selected=[component_index[k] for k in required];evidence['target_costs']=selected
        ar1,sd1,ar16,sd16=selected
        ar_scale=float(ar16['total_ms'])/float(ar1['total_ms']);sd_scale=float(sd16['total_ms'])/float(sd1['total_ms'])
        ratio1=float(sd1['total_ms'])/float(ar1['total_ms']);ratio16=float(sd16['total_ms'])/float(ar16['total_ms'])
        expert_growth={label:(float(large['MoE_FP4_GEMM_ms'])-float(small['MoE_FP4_GEMM_ms']))/(float(large['total_ms'])-float(small['total_ms'])) for label,small,large in [('AR',ar1,ar16),('verify_W8',sd1,sd16)]}
        evidence['expert_fraction_of_target_cost_growth_BS1_to_16']=expert_growth
        parts.append('<h2>成本增长主要在哪里</h2>')
        parts.append(table(['target forward','GPU kernel 合计 ms','原生 FP4 专家 GEMM ms','QKV/O ms','LM head ms'],[[f"{r['method']} BS{r['configured_bs']} W{r['width']}"]+[f"{float(r[k]):.3f}" for k in ('total_ms','MoE_FP4_GEMM_ms','QKV_O_projection_ms','LM_head_ms')] for r in selected]))
        growth=[]
        for label,small,large,field in [('AR 专家 GEMM',ar1,ar16,'MoE_FP4_GEMM_ms'),('W8 verify 专家 GEMM',sd1,sd16,'MoE_FP4_GEMM_ms'),('AR target 全部 kernel',ar1,ar16,'total_ms'),('W8 verify 全部 kernel',sd1,sd16,'total_ms')]:
            x,y=float(small[field]),float(large[field]);growth.append({'component':label,'BS1_ms':x,'BS16_ms':y,'growth_factor':y/x})
        evidence['BS_growth_GPU_work']=growth
        parts.append(table(['GSM8K / GPU 工作量','BS1 ms','BS16 ms','BS1→16 增长'],[[r['component'],f"{r['BS1_ms']:.2f}",f"{r['BS16_ms']:.2f}",f"{r['growth_factor']:.2f}×"] for r in growth]))
        parts.append(f'<p>这里只取活跃请求数等于标称 BS 的 target forward，时间是各 GPU kernel 的累计执行时间。BS1→16，AR target 成本增加 {ar_scale:.2f}×，width8 verify 增加 {sd_scale:.2f}×；verify/AR 的成本比由 {ratio1:.2f} 降到 {ratio16:.2f}。因此，即使 τ 接近不变，gain 也可能随 BS 上升。AR 自己随 BS 增加的专家访问成本同样重要。该表不是端到端 SD 周期：尚未加入 draft、轮次间空隙和请求尾部。</p>')
        parts.append(f'<p>进一步看增长量，原生 FP4 专家 GEMM 占 AR target 增量的 {expert_growth["AR"]:.1%}、占 width8 verify 增量的 {expert_growth["verify_W8"]:.1%}。本组从 BS1 增大到 BS16 的 target 成本增长主要发生在专家 GEMM 中，router 本身不是主要增量来源。</p>')
        parts.append('<p>MXFP4 仅用于专家权重，专家 GEMM 实际使用 MXFP8 激活 × MXFP4 权重；QKV/O、router、LM head 等仍有 BF16 运算。小 BS 时约固定的 head 和投影成本占比较高；增大 BS/width 后专家 GEMM 的增长更显著。低比特专家与未量化 dense 部分的组合也会改变瓶颈，不能将整个模型当作统一的 FP4 GEMM。</p>')
    phase_path=out/'trace_phases.csv'
    if phase_path.exists():
        phases=[r for r in csv.DictReader(phase_path.open()) if r['method']=='dflash' and int(r['width'])==8 and int(r['configured_bs']) in (1,16) and r['phase'] in ('DRAFT','VERIFY','decode_control')]
        evidence['SD_phase_costs']=phases
        parts.append(table(['DFlash W8','阶段','GPU kernel ms / round','CPU NVTX ms / round'],[[f"BS{r['configured_bs']}",r['phase'],f"{float(r['gpu_kernel_ms_per_call']):.3f}",f"{float(r['host_ms_per_call']):.3f}"] for r in phases]))
        parts.append('<p>阶段表包含实际运行中的请求完成尾部，因而 BS16 verify 与前表“完整活跃 batch”的值略有不同。CPU NVTX 是包围异步提交、等待或控制逻辑的 wall time；其中可能与 GPU 执行重叠。不能把 CPU 和 GPU 两列相加，也不能把整个 control NVTX 都称为 CPU 计算。当前主要增长体现在 target GPU 工作，draft 与 control 的增长小得多。</p>')
        phase_index={(int(r['configured_bs']),r['phase']):r for r in phases}
        if all((b,'DRAFT') in phase_index and ('ar',b,1) in component_index for b in (1,16)):
            ratios={b:float(phase_index[b,'DRAFT']['gpu_kernel_ms_per_call'])/float(component_index['ar',b,1]['total_ms']) for b in (1,16)}
            evidence['draft_GPU_work_relative_to_AR_target']=ratios
            parts.append(f'<p>另一个影响 gain 的因素是 draft 的相对成本：本次 draft 的 GPU 工作在 BS1 约相当于一个 AR target forward 的 {ratios[1]:.0%}，到 BS16 降为 {ratios[16]:.0%}。稀疏、低比特 target 让 AR 本身更快，dense draft 即使绝对耗时不大，也可能成为明显的额外负担。这些是 GPU 工作量的比例，不能当作互不重叠的端到端周期份额。</p>')
    routing_full=[r for r in routing if r['phase'] in ('DECODE','TARGET_VERIFY') and int(r['request_rows'])==int(r['configured_bs']) and int(r['routed_token_rows'])==int(r['configured_bs'])*int(r['width'])]
    evidence['routing_full_active_batch']=routing_full
    if routing_full:
        parts.append('<h2>实际专家覆盖与 MoE 的重用</h2>')
        parts.append(table(['方法','BS','width','target token rows','每层平均命中专家 / 32','每个命中专家平均 token 数','每层最繁忙专家 token 数均值'],[[r['method'],r['configured_bs'],r['width'],r['routed_token_rows'],f"{float(r['mean_unique_experts']):.2f}",f"{float(r['mean_assignments_per_used_expert']):.2f}",f"{float(r['mean_max_expert_assignments']):.2f}"] for r in routing_full]))
        route_index={(r['method'],int(r['configured_bs']),int(r['width'])):r for r in routing_full}
        if all(k in route_index for k in required):
            coverage_ratios={bs:float(route_index['dflash',bs,8]['mean_unique_experts'])/float(route_index['ar',bs,1]['mean_unique_experts']) for bs in (1,16)}
            evidence['verify_to_AR_expert_coverage_ratio']=coverage_ratios
            parts.append(f'<p>实际覆盖数给出了更直接的解释：W8 verify 相对 AR 的专家集合大小，在 BS1 是 {coverage_ratios[1]:.2f} 倍，到 BS16 降为 {coverage_ratios[16]:.2f} 倍。AR 随 BS 增大也逐渐访问更多专家，因此 verify 的增量工作集并不按8倍增长。这是测得的专家集合比例，仍不是 HBM 字节比例；每个专家的 token 负载、tile padding、缓存与权重重读还会影响时间。</p>')
            large=route_index['dflash',16,8]
            parts.append(f'<p>BS16×W8 的全局128个 verify token rows，被分流后，每个命中专家平均只有 {float(large["mean_assignments_per_used_expert"]):.1f} 个 token；每层最繁忙专家平均为 {float(large["mean_max_expert_assignments"]):.1f} 个。Dense FFN 则由同一组权重处理全部128个 token。MoE 的全局大 M 不代表所有专家 GEMM 都有同样大的 M；这也是不能直接套用“prefill 大矩阵已 compute bound”判断的原因。</p>')
        parts.append('<p>MoE 的有用矩阵行数更接近每个专家收到的 token 数 nₑ，而非统一的 B×W。增加 B 或 W 一方面增加已有专家内的重用，另一方面可能让更多专家的权重进入本轮工作集；后者会增加权重流量。专家覆盖逐渐接近上限后，增加 W 的额外专家集合可能小于小 BS 时的增量，所以 verify 相对 AR 的成本可以缩小。实际路由具有相关性，上表来自独立记录，均匀独立路由曲线仅用于参照。专家 checkpoint 字节的集合是逻辑 footprint，不能冒充实测 HBM 传输量或严格的 roofline 结论。</p>')
        parts.append('<p>作为 roofline 近似，单个专家矩阵 Wₑ∈R<sup>K×N</sup> 接收 nₑ 个 token 时，FLOPs≈2nₑKN。只计一次权重读取的理想化强度约为 AIₑ≈2nₑ/s<sub>w</sub> FLOP/byte；实际还要计激活、scale、dispatch/combine、缓存和权重重读。对 MXFP4，专家权重与 scale 的平均 s<sub>w</sub>≈0.53125 byte/值。这个式子解释了为什么应看专家 token 负载，而不能仅凭全局 B×W 判断是否 compute bound；原生低精度算力峰值也随精度变化。</p>')
    counters_path=out/'trace_counters.csv'
    if counters_path.exists():
        counters=[r for r in csv.DictReader(counters_path.open()) if int(r['configured_bs']) in (1,16) and ((r['method']=='ar' and r['phase']=='DECODE') or (r['method']=='dflash' and int(r['width'])==8 and r['phase']=='VERIFY'))]
        evidence['target_activity_metrics']=counters
        parts.append('<h2>活动率能说明什么</h2>')
        parts.append(table(['target 阶段','BS','SM active %','Tensor active %','DRAM read %'],[[r['phase'],r['configured_bs']]+[f"{float(r[k]):.1f}" for k in ('SMs Active [Throughput %]','Tensor Active [Throughput %]','DRAM Read Bandwidth [Throughput %]')] for r in counters]))
        parts.append('<p>以上为 nsys 1 kHz 采样按相关 kernel 区间加权后的活动率，含不同类型算子的混合，不等于各 kernel 的实测 FLOP/s。SM active 表示有工作，不代表 tensor 吞吐已经饱和；Tensor active、DRAM read 可作为方向性证据，但无法单独证明整个 verify 已 compute bound。要严格定位 roofline，还需每个主要 kernel 的运算量、实际内存字节和对应精度下的可达峰值。</p>')
    dense=[r for r in rows if r['stage']=='dense_control' and r['method']=='dflash' and r['gain'] is not None]
    evidence['dense_control']=dense
    if dense:
        parts.append('<h2>与 dense target 的同宽度对照</h2>')
        records=[]
        for dataset in ('gsm8k','mt_bench'):
            for b in (1,4,16):
                m=next((r for r in full if r['dataset']==dataset and r['bs']==b),None)
                d=next((r for r in dense if r['dataset']==dataset and r['bs']==b),None)
                if not m or not d:continue
                records.append([dataset,b,f"{m['gain']:.3f}×",f"{d['gain']:.3f}×",f"{m['tau']:.3f}",f"{d['tau']:.3f}"])
        parts.append(table(['任务','BS','GPT-OSS20B MXFP4 gain','Qwen3-8B BF16 gain','MoE τ','Dense τ'],records))
        parts.append('<p>两边都固定 DFlash width8，并使用本次重跑的 SGLang FA4 离线协议。跨模型还改变了 tokenizer、prompt 模板、draft、权重精度及激活参数量；这组对照展示现象，不是只替换 MoE 这一变量的因果实验。MoE 机制的判断应结合本模型的 BS/width 扫描、路由与 kernel 分解。</p>')
        cycle_growth=[]
        for stage,label in [('full','GPT-OSS20B MXFP4'),('dense_control','Qwen3-8B BF16')]:
            values={}
            for method in ('ar','dflash'):
                for bs in (1,16):
                    selected=[r['request_mean_cycle_median_ms'] for r in rows if r['stage']==stage and r['dataset']=='gsm8k' and r['method']==method and r['bs']==bs]
                    if selected:values[method,bs]=statistics.median(selected)
            if len(values)!=4:continue
            cycle_growth.append({'target':label,'AR_BS1_ms':values['ar',1],'AR_BS16_ms':values['ar',16],'AR_growth':values['ar',16]/values['ar',1]-1,'SD_BS1_ms':values['dflash',1],'SD_BS16_ms':values['dflash',16],'SD_growth':values['dflash',16]/values['dflash',1]-1})
        if cycle_growth:
            evidence['BS_growth_request_cycles']=cycle_growth
            parts.append('<h3>解释 BS 趋势：同时看 AR 与整个 SD 周期</h3>')
            parts.append(table(['GSM8K / target','AR 周期 BS1→16','AR 增长','SD 周期 BS1→16','SD 增长'],[[r['target'],f"{r['AR_BS1_ms']:.2f}→{r['AR_BS16_ms']:.2f} ms",f"{r['AR_growth']:.0%}",f"{r['SD_BS1_ms']:.2f}→{r['SD_BS16_ms']:.2f} ms",f"{r['SD_growth']:.0%}"] for r in cycle_growth]))
            parts.append('<p>MoE 的整个 SD 周期涨幅其实高于 dense 对照，但其 AR 涨得更快，因此 gain 仍可上升。这里是请求平均 decode 周期的中位数，AR 再取前后基线的中位数；SD 含 draft、verify 与控制等待，和上面的 GPU kernel 工作量表口径不同。全局 τ 与这些中位数不能精确还原含 prefill 和完成尾部的 wave 吞吐。不能仅凭 gain 就推断 MoE verify GEMM 比 dense verify GEMM 增长更慢。</p>')
    if tails:
        records=[]
        cases={r['case']:r for r in rows if r['stage']=='full' and r['bs']==16}
        # The first AR bookend and its paired SD arm; summarize all three repeats.
        for dataset in ('gsm8k','humaneval','mt_bench'):
            for method in ('ar','dflash'):
                names=sorted(n for n,r in cases.items() if r['dataset']==dataset and r['method']==method)
                if not names:continue
                selected=[r for r in tails if r['case']==names[0]]
                if not selected:continue
                fields=('wave_wall_ms','critical_TTFT_ms','critical_decode_steps','critical_decode_tokens_per_step','critical_mean_cycle_ms')
                records.append([dataset,method]+[f"{statistics.median(float(r[k]) for r in selected):.3f}" for k in fields])
        evidence['critical_wave_BS16']=records
        if records:
            parts.append('<h2>为什么平均接受长度不能完全预测吞吐</h2>')
            parts.append(table(['任务 BS16','方法','wave wall ms','最后完成请求 TTFT ms','该请求 decode steps','该请求 tokens/step','该请求平均周期 ms'],records))
            parts.append('<p>各列分别取三次重复的中位数。一个 wave 要等最后一个请求完成；该请求的 token/step 可能低于全局 τ，不同方法的最后完成请求也可能不同。这能区分接受长度变化、target 周期增长与完成尾部，不能把全部 gain 变化都归因于 GPU 算力。</p>')
    correctness_path=out/'correctness.json'
    if correctness_path.exists():
        correctness=json.loads(correctness_path.read_text());by_case={r['case']:r for r in rows}
        observed=[(by_case[c['case']],c) for c in correctness if c['case'] in by_case and by_case[c['case']]['stage']=='full' and 'AR_exact_requests' in c]
        if observed:
            parts.append('<h2>输出一致性与适用边界</h2>')
            parts.append(table(['任务','BS','配置内重复完全相同请求','与 AR 完全相同请求'],[[r['dataset'],r['bs'],f"{c['repeat_exact_requests']}/{c['requests']}",f"{c['AR_exact_requests']}/{c['requests']}"] for r,c in observed]))
            parts.append('<p>这些结果保留了 AR/SD token 分歧，不能宣称端到端逐 token 等价。固定输出128 tokens 的实验测速度，不测最终答案正确率；浮点形状、量化执行路径及输出轨迹差异都需要独立核查，不能只凭出现分歧就确定原因。</p>')
    parity_path=out/'cross_configuration_parity.json'
    if parity_path.exists():
        parity=json.loads(parity_path.read_text());groups=defaultdict(list)
        for c in parity['comparisons']:groups[c['comparison']].append(c)
        evidence['cross_configuration_parity_summary']={k:{'comparisons':len(v),'min_exact_requests':min(c['exact_requests'] for c in v),'max_exact_requests':max(c['exact_requests'] for c in v),'requests_per_comparison':v[0]['requests']} for k,v in groups.items()}
        parts.append(table(['同输入输出对照','配置比较数','每组16条中完全相同序列数范围'],[[k,len(v),f"{min(c['exact_requests'] for c in v)}–{max(c['exact_requests'] for c in v)} / 16"] for k,v in groups.items()]))
        parts.append('<p>同配置 AR 前后基线、跨 BS、跨 width，以及重复的 width8 cohort 都另有逐条首次分歧记录。这里的“完全相同”比较完整128-token 序列；跨模型的 token IDs 不作等价比较。</p>')
    numerics_path=out/'numerics.json'
    if numerics_path.exists():
        numerical=json.loads(numerics_path.read_text());evidence['numerical_diagnostic']=numerical
        parts.append('<p>另外对 headline 已出现分歧的两条 GSM8K 输入单独采集 top5 logprobs，分别使用 BF16/FP32 LM head；这些 cell 不参与性能排名。诊断范围仅限这两条输入，FP32 head 不会把量化/BF16 backbone 变成全 FP32。</p>')
        parts.append(table(['LM head FP32','AR/SD 完全相同输入 / 2'],[[c['enable_fp32_lm_head'],f"{c['exact_prompts']}/{c['samples']}"] for c in numerical['comparisons']]))
        first_differences=[]
        for c in numerical['comparisons']:
            for p in c['prompts']:
                if p['equal']:continue
                margins=[f"{p[k]:.5f}" if p[k] is not None else '未返回' for k in ('AR_top2_logprob_margin','SD_top2_logprob_margin')]
                first_differences.append([c['enable_fp32_lm_head'],p['prompt'],p['first_mismatch'],repr(p['AR_token_text']),repr(p['SD_token_text'])]+margins)
        if first_differences:parts.append(table(['LM head FP32','输入','首次分歧 index（0起）','AR token','SD token','AR top1−top2 logprob','SD top1−top2 logprob'],first_differences))
        parts.append('<p>只在首次分歧位置比较 logits/logprobs，此时两边的历史前缀相同；分歧后不能把不同前缀下的 logits 当作同一个条件分布来比较。还需查看 BF16 logprob 路径是否复现了 headline 输出，避免把诊断自身改变路径的结果混入解释。</p>')
    widths=[r for r in rows if r['stage']=='width' and r['method']=='dflash' and r['gain'] is not None]
    if widths:
        parts.append('<h2>验证宽度的取舍</h2>')
        complete=[]
        for dataset in ('gsm8k','mt_bench'):
            for b in (1,4,16):
                group=[r for r in widths if r['dataset']==dataset and r['bs']==b]
                if {r['width'] for r in group}!={2,4,8}:continue
                best=max(group,key=lambda r:r['gain']);complete.append({'dataset':dataset,'bs':b,'best_measured_width':best['width'],'gain':best['gain']})
        evidence['best_measured_width']=complete
        if complete:parts.append(table(['任务','BS','W2/4/8 中本次最快','gain'],[[r['dataset'],r['bs'],r['best_measured_width'],f"{r['gain']:.3f}×"] for r in complete]))
        parts.append('<p>更宽会提高潜在提交长度，也会增加 target 专家工作、draft 成本和未接受 token 的浪费。width 扫描使用同一 checkpoint，不代表针对不同宽度重新训练；最快值仅对本次候选宽度和输入成立，且须同时参考接受长度和输出轨迹。</p>')
    evidence['limits']=['Cross-model comparisons change more than MoE.','Kernel-sum costs are not end-to-end latency.','Routing footprint is not measured HBM traffic.','Fixed-length output speed does not measure task quality.','Output parity is reported, not assumed.']
    path=out/'findings.json';temporary=path.with_suffix('.json.tmp');temporary.write_text(json.dumps(evidence,indent=2,ensure_ascii=False));temporary.replace(path)
    return ''.join(parts)
