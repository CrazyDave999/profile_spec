"""Evidence-based overview; incomplete experiments remain visibly incomplete."""
import _bootstrap
import csv
import hashlib
import html
import json
from bisect import bisect_right,bisect_left
from pathlib import Path

from build_report_index import write_index
from parallel_sd_refinement_figures import width_comparison

ROOT=_bootstrap.ROOT
LOGS=ROOT/'logs/parallel_sd_20261104'
STEM='2026-11-04_parallel_sd'


def read_csv(path):
    return list(csv.DictReader(path.open())) if path.exists() else []


def table(headers,rows):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def image(path,description):
    if not (ROOT/'reports'/path).is_file():return ''
    return f'<figure><a href="{path}"><img src="{path}" alt="{html.escape(description)}"></a><figcaption>{description}</figcaption></figure>'


def main():
    output=ROOT/'reports'/STEM;output.mkdir(exist_ok=True)
    active_path=LOGS/'active_runs.json'
    active=json.loads(active_path.read_text()) if active_path.exists() else {}
    scaling=read_csv(ROOT/'reports/2026-11-04_psd_scaling/scaling.csv')
    raw=read_csv(ROOT/'reports/2026-11-04_psd_scaling/results.csv')
    assert scaling and raw,'Generate the successful-cell scaling report first.'
    campaign=json.loads((LOGS/'full01.campaign.json').read_text())
    live_successes=[]
    for cell in campaign['commands']:
        path=LOGS/(cell['name']+'.job.json')
        if path.exists() and json.loads(path.read_text())['state']=='success':live_successes.append(cell['name'])
    # A cell can finish while matplotlib is rendering the snapshot. Keep the
    # artifact snapshot intact and disclose the current live count separately.
    successes=[r['case'] for r in raw]
    assert set(successes)<=set(live_successes),'A reported cell lacks successful terminal evidence.'
    pending=[cell['name'] for cell in campaign['commands'] if cell['name'] not in successes]
    sources={}
    for cell in successes:
        path=LOGS/(cell+'.json');sources[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    first=json.loads((LOGS/(successes[0]+'.json')).read_text())
    complete=len(successes)==len(campaign['commands']) and campaign['status']=='success'
    state=f"主矩阵当前成功 {len(live_successes)}/{len(campaign['commands'])} 项；此页快照收录 {len(successes)} 项；状态 {campaign['status']}。"
    intro=f'<p class="status">{state}'+('完整主矩阵已收齐。' if complete else '此页为进行中的实验快照；未完成项不填值。')+'</p>'
    intro+='<p>本机 RTX 5090 · Qwen3-8B BF16 · TP1 · FA4 · SGLang offline Engine。四种 parallel draft 使用同一 target 和同一运行时 AR 基线。每个主矩阵 cell 为相同的 16 条真实输入、固定生成 128 tokens、独立进程、重复 3 次；每个任务/BS 组前后各跑一次 AR。</p>'
    intro+='<p>代码、环境与报告位于 <code>~/profile_spec</code>，权重在外部 HF cache；decode CUDA Graph 和 overlap scheduler 开启，初始 cohort 的 prefill graph 关闭，prefix reuse 关闭。计时排除加载、编译、warmup 和 cache flush。固定输出忽略 EOS；本表不是自然停止条件下的应用吞吐或质量 leaderboard。</p>'
    cohort_manifest=[]
    artifact_audit=LOGS/'campaign_artifact_audit.json'
    audited={a['run_id']:a for a in json.loads(artifact_audit.read_text())} if artifact_audit.exists() else {}
    for key in ['full','traces','hybrid','width8','numerics','sps','adaptive','adaptive_graph']:
        run=active.get(key)
        if not run or not (LOGS/(run+'.campaign.json')).exists():continue
        current=json.loads((LOGS/(run+'.campaign.json')).read_text());check=audited.get(run,{})
        category={'traces':'profile','numerics':'diagnostic','sps':'calibration'}.get(key,'benchmark')
        valid=check.get('complete',False) and current['status']=='success' and check.get('checked')==len(current['commands'])
        cohort_manifest.append(dict(role=key,run_id=run,category=category,status=current['status'],cells=len(current['commands']),artifact_audit_complete=valid,worker_SHA256=current['worker_sha256']))
        if valid:
            for item in check['source_artifacts']:
                name='logs/parallel_sd_20261104/'+item['case']+'.json';digest=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                assert digest==item['SHA256'],'A terminal source artifact changed after audit.'
                sources[name]=digest
    if len(cohort_manifest)==8 and all(c['artifact_audit_complete'] for c in cohort_manifest):
        intro+='<p>本轮已完成并核查 104 个无 profiler 的 benchmark cell、10 个 warmed trace、4 个数值诊断 cell 与 5 个 SPS 校准 cell，共 123 项。诊断、校准、profile 均不进入主吞吐排名。</p>'
    methods=[['AR','1','同一个 SGLang target，独立进程，无闲置 draft'],
      ['DFlash','16','五层 block-diffusion draft；五个 target hidden taps；复用 target embedding/head'],
      ['Domino','16','五层 draft + GRU rollout；candidate pool 2048；不是仅换参数的普通 DFlash'],
      ['DSpark','8','五层 draft + Markov rank256 + confidence head；主矩阵为 static verify'],
      ['DFlash2','8','五层 draft + local convolution + candidate selector；Speculators checkpoint 转为 SGLang config']]
    body='<h2>比较范围与 feature</h2>'+table(['方法','默认 verify 宽度','路径'],methods)
    body+='<p>checkpoint 训练和宽度不同，因此主表比较的是可部署组合。DFlash2 权重未改，层号转换有源码依据，已通过 native SGLang graph capture 和生成；尚未证明与原 Speculators 实现完全数值等价。Rust 扩展未构建。FA4 的 draft metadata glue graph 未开启，不能声称整个 SD 周期都在一张 graph 内。</p>'
    body+='<p><a href="2026-11-04_psd_scaling.html">完整逐 cell 数据、功能日志与质量限制</a> · <a href="../logs/dflash2_config_conversion.json">DFlash2 转换记录</a> · <a href="../models-parallel-sd.json">checkpoint revision</a>。</p>'
    gains=[]
    for dataset in ['gsm8k','humaneval','mt_bench']:
        for method in ['dflash','domino','dspark','dflash2']:
            values=[]
            for bs in [1,4,16]:
                row=next((r for r in scaling if r['dataset']==dataset and r['method']==method and int(r['bs'])==bs),None)
                values.append(f"{float(row['gain']):.2f}× / τ {float(row['tau']):.2f}" if row else '待测')
            gains.append([dataset,method,*values])
    body+='<h2>吞吐收益随 BS 和任务变化</h2>'+table(['任务','方法','BS1：gain / τ','BS4：gain / τ','BS16：gain / τ'],gains)
    body+='<p>gain = cell 吞吐中位数 / 匹配组前后 AR 吞吐中位数的中位数。τ = (实际输出 tokens − 每请求 prefill seed) / 请求 verify 次数之和，包含 bonus，受最后一轮截断影响。配置 BS 是最大活跃请求数，SD 的不同完成时间会产生 tail。</p>'
    for filename,description in [('gain_scaling.svg','各任务的吞吐收益；未测点不插值填数'),('acceptance_scaling.svg','接受长度与收益分别观察，不能只用 τ 排名'),('throughput_scaling.svg','输出吞吐；短竖线为重复范围，不是置信区间')]:
        body+=image('2026-11-04_psd_scaling/'+filename,description)
    gsm={ (r['method'],int(r['bs'])):r for r in scaling if r['dataset']=='gsm8k'}
    if all((m,b) in gsm for m in ['dflash','domino','dspark','dflash2'] for b in [1,16]):
        observations=[]
        for method in ['dflash','domino','dspark','dflash2']:
            a,b=gsm[method,1],gsm[method,16]
            observations.append(f"{method}：gain {float(a['gain']):.2f}× → {float(b['gain']):.2f}×，τ {float(a['tau']):.2f} → {float(b['tau']):.2f}")
        body+='<p class="finding">GSM8K：'+'；'.join(observations)+'。接受长度变化较小，吞吐收益下降明显；这是每轮成本相对 AR 增长的线索，单靠吞吐仍不能区分 prefill、GPU 计算、带宽与控制开销。</p>'
        cycles=[]
        for method in ['ar','dflash','domino','dspark','dflash2']:
            a,b=gsm[method,1],gsm[method,16]
            cycles.append([method,f"{float(a['request_mean_cycle_median_ms']):.2f}",f"{float(b['request_mean_cycle_median_ms']):.2f}",f"{float(b['request_mean_cycle_median_ms'])/float(a['request_mean_cycle_median_ms']):.2f}×"])
        body+='<h2>总 SD 周期：请求计时给出的证据</h2>'+table(['方法','BS1 ms/周期','BS16 ms/周期','增长'],cycles)
        body+=image('2026-11-04_psd_scaling/cycle_latency_scaling.svg','请求平均 decode 周期的中位数；覆盖 CPU、GPU 与反馈等待')
        body+='<p>每个请求的平均周期 = (E2E−TTFT)/verify 次数；AR 分母为输出 tokens−1。时间定义来自 SGLang req_time_stats，与其返回的 decode_throughput 算式交叉检查。它排除了首 token 前的时间，但仍包含控制和等待；不是 GPU kernel 的独立时长。近似 gain≈τ×T_AR/T_SD，wall throughput 还受 prefill 与 tail 影响。<a href="2026-11-04_psd_scaling/request_cycle_latency.csv">逐请求计算明细</a>。</p>'
    body+='<h2>显存：文件、逻辑 KV 与物理占用分开</h2>'
    body+=image('2026-11-04_psd_scaling/memory_scaling.svg','NVML 50 ms 采样的物理显存峰值，包含启动和运行；不是 torch allocated')
    audit=json.loads((LOGS/'checkpoint_memory_audit.json').read_text())
    memory=[]
    for r in audit['checkpoints']:
        memory.append([r['method'],f"{r['file_GiB']:.3f}",f"{r['non_vocab_tensor_GiB']:.3f}",f"{r['draft_KV_bytes_per_token']/1024:.0f}",f"{r['raw_selected_hidden_bytes_per_token']/1024:.0f}"])
    body+=table(['方法','文件 GiB','非 vocab tensors GiB','Draft KV KiB/token','原始 taps KiB/token'],memory)
    body+='<p>target KV 每 token 144 KiB，32768-token pool 为 4.500 GiB；所有 draft 均为五层、每 token 20 KiB，pool 约 0.625 GiB。40 KiB/token 的原始 hidden taps 经投影并物化为 draft KV，不应再假定全历史永久保留原始 taps。DSpark 文件中 2.318 GiB 的 embedding/head 在当前 Qwen loader 被跳过，运行时挂接 target 模块，因此 4.416 GiB 文件不等于新增 4.416 GiB 显存。</p>'
    body+='<p>这些是逻辑大小和 header 计数，不能把它们直接相加宣称解释了全部 NVML 峰值；CUDA context、graph、allocator 和 workspace 也占显存。<a href="../logs/parallel_sd_20261104/checkpoint_memory_audit.json">显存核查来源</a>。</p>'
    body+='<h2>Nsight：每轮成本来自哪里</h2>'
    trace_manifest=ROOT/'reports/2026-11-04_psd_traces/manifest.json'
    if trace_manifest.exists():
        traces=json.loads(trace_manifest.read_text());trace_rows=[]
        body+=f'<p>代表 trace 当前成功汇总 {len(traces["cells"])}/{len(traces["campaign"]["commands"])} 项；采集状态 {traces["campaign"]["status"]}。所有图只使用已成功结束的 profile cell。</p>'
        for cell in traces['cells']:
            w=cell['decode_window']
            trace_rows.append([cell['method'],cell['bs'],w['scheduler_decode_rounds'],f"{w['mean_segmented_round_ms']:.2f}",f"{w['mean_GPU_timeline_round_ms']:.2f}",f"{w['mean_gpu_kernel_work_per_round_ms']:.2f}",w['interleaved_prefill_calls'],cell['graph_launches'],f"{cell['report_bytes']/2**20:.2f}"])
        body+=table(['方法','配置 BS','轮数','CPU-dispatch envelope ms/轮','GPU timeline ms/轮','GPU work ms/轮','穿插 prefill','Graph replay','nsys MiB'],trace_rows)
        dflash_one=next((cell for cell in traces['cells'] if cell['method']=='dflash' and cell['bs']==1),None)
        phase_rows=read_csv(ROOT/'reports/2026-11-04_psd_traces/phase_breakdown.csv')
        host_control=next((r for r in phase_rows if r['method']=='dflash' and r['configured_bs']=='1' and r['phase']=='decode_control'),None)
        if dflash_one and host_control:
            w=dflash_one['decode_window']
            body+=f'<p class="finding">DFlash BS1：每轮 GPU kernel work {w["mean_gpu_kernel_work_per_round_ms"]:.2f} ms，decode envelope {w["mean_segmented_round_ms"]:.2f} ms，host 控制 NVTX {float(host_control["host_ms_per_scheduler_round"]):.2f} ms。host 包含等待且与 GPU 重叠，不能整段再加到 GPU 时间上。GPU work 是 kernel 时长之和，envelope 是流水线区间；二者之差也不能直接归因于纯 CPU。</p>'
        body+=image('2026-11-04_psd_traces/phase_breakdown.svg','GPU kernel 工作与 host NVTX 时间分图，不能相加；control 不是 accept-only')
        body+=image('2026-11-04_psd_traces/gpu_counters.svg','1 kHz 采样按 GPU 执行区间加权；存在边界混合，不是每 kernel counter')
        conditioned=read_csv(ROOT/'reports/2026-11-04_psd_traces/by_active_batch.csv')
        costs=[]
        for method in ['ar','dflash','domino','dspark','dflash2']:
            for phase in (['DECODE'] if method=='ar' else ['DRAFT','VERIFY']):
                selected={int(r['configured_bs']):r for r in conditioned if r['method']==method and r['phase']==phase and int(r['batch_size'])==int(r['configured_bs'])}
                if 1 in selected and 16 in selected:
                    a,b=selected[1],selected[16]
                    costs.append([method,phase,f"{float(a['gpu_kernel_ms_per_call']):.3f}",f"{float(b['gpu_kernel_ms_per_call']):.3f}",f"{float(b['gpu_kernel_ms_per_call'])/float(a['gpu_kernel_ms_per_call']):.2f}×",f"{float(a['host_ms_per_call']):.3f} → {float(b['host_ms_per_call']):.3f}"])
        if costs:body+='<h3>限定观测 BS 的 forward 成本</h3>'+table(['方法','阶段','GPU ms，BS1','GPU ms，BS16','GPU 增长','Host NVTX ms，1→16'],costs)
        body+=image('2026-11-04_psd_traces/active_batch_cost.svg','Tail 中的 forward GPU 成本呈 bucket 台阶，实际 BS 与 graph padded BS 分开')
        padding=read_csv(ROOT/'reports/2026-11-04_psd_traces/graph_padding.csv')
        if padding:
            body+=table(['方法','BS16 轮数 / 全部轮数','平均观测 BS','实际请求行 / 推算 padded 行'],[[r['method'],r['full_BS16_rounds']+' / '+r['calls'],f"{float(r['mean_observed_bs']):.2f}",f"{float(r['actual_over_inferred_padded_rows']):.1%}"] for r in padding])
            body+='<p>capture buckets 为 [1,2,4,8,16]；pinned runner 将 raw BS 向上补到最近 bucket。DFlash 的 BS9/11/13/15 verify 因此仍接近 BS16 的成本，图中的台阶与源码规则相符。表中 padded 行是按配置和 NVTX BS 推算，不是直接读取 graph key；该比例不是硬件利用率，也不包含候选 token 被拒绝的浪费。增加 capture buckets、缩小间隔是后续可测的优化方向，此处没有宣称已测得收益。<a href="2026-11-04_psd_traces/graph_padding.csv">推算明细</a>。</p>'
            body+='<p>即使使用 offline Engine，batch 组织、请求完成后的 tail 和 overlap 调度也已经影响 SD 收益。线上 continuous batching 会进一步改变实际 BS 与补齐成本；本实验没有测在线请求到达流量和服务容量。</p>'
        counter_rows=read_csv(ROOT/'reports/2026-11-04_psd_traces/gpu_counters.csv')
        ar_counter=next((r for r in counter_rows if r['method']=='ar' and r['configured_bs']=='1' and r['phase']=='DECODE'),None)
        if ar_counter:
            body+=f'<p class="finding">AR BS1：SM active {float(ar_counter["SMs Active [Throughput %]"]):.1f}%，Tensor active {float(ar_counter["Tensor Active [Throughput %]"]):.1f}%，DRAM read {float(ar_counter["DRAM Read Bandwidth [Throughput %]"]):.1f}%。高 SM active 与低 Tensor activity 同时出现；这些观测更符合带宽侧约束，不能把 SM active 当成计算峰值利用率。</p>'
        ar_sixteen=next((r for r in counter_rows if r['method']=='ar' and r['configured_bs']=='16' and r['phase']=='DECODE'),None)
        df_one=next((r for r in counter_rows if r['method']=='dflash' and r['configured_bs']=='1' and r['phase']=='VERIFY'),None)
        if ar_sixteen and df_one:
            body+=f'<p>AR BS16 decode 与 DFlash BS1、width16 verify 都约有 16 行 target query：Tensor active 分别 {float(ar_sixteen["Tensor Active [Throughput %]"]):.1f}% / {float(df_one["Tensor Active [Throughput %]"]):.1f}%，DRAM read {float(ar_sixteen["DRAM Read Bandwidth [Throughput %]"]):.1f}% / {float(df_one["DRAM Read Bandwidth [Throughput %]"]):.1f}%。attention 的历史布局不同，但这组相近的资源特征支持按有效 query 行数分析；“decode / prefill”名称本身不足以判断瓶颈，短 verify 也可能偏带宽侧。</p>'
        body+='<p>BS1 与 BS16 都使用相同 16 条 GSM8K 输入、cap64；与 cap128 的无 profile 吞吐表分开。CPU-dispatch envelope 按 prefill 边界切分，排除波次间的 API/prefill dispatch 间隔，但其起点可能仍在等待先前的 GPU prefill。GPU timeline 则从首个 decode GPU kernel 到最后一个 decode kernel，包含轮间空隙，排除首次 dispatch 前准备、首次 GPU 排队和最后的 host 返回。端到端周期另看请求计时。整体窗口中穿插的 prefill 数另列，不能把整体 wall/轮标成纯 decode 成本。</p>'
        body+='<p><a href="2026-11-04_psd_traces/phase_breakdown.csv">阶段 CSV</a> · <a href="2026-11-04_psd_traces/by_active_batch.csv">按观测 BS 条件化的 forward 成本</a> · <a href="2026-11-04_psd_traces/manifest.json">trace 清单和原始路径</a>。固定 BS 的 GPU 成本优先看条件化表，避免 tail 混入。</p>'
        bundle_path=LOGS/'representative_traces_manifest.json'
        if bundle_path.exists():
            bundle=json.loads(bundle_path.read_text())
            body+=f'<p>本机代表性 trace 包：<a href="../{bundle["bundle"]}">representative_traces.zip</a>，{bundle["bundle_bytes"]/2**20:.1f} MiB，含 {len(bundle["traces"])} 个 nsys-rep 及归因 JSON。此链接指向服务器本地 ignored profiles/，GitHub 不托管该 ZIP；下载时使用清单中的服务器路径。</p>'
            body+='<p><a href="../logs/parallel_sd_20261104/representative_traces_manifest.json">逐文件大小与 SHA256</a>。</p>'
    else:
        body+='<p>同输入组的 BS1/16 counter trace 尚未汇总；不据此对大 BS 的瓶颈下结论。已有单 prompt DFlash smoke trace 仅用于验证 graph replay 和采集链路，见 pilot 报告。</p>'
    body+='<p>GPU busy fraction 只说明有没有 kernel 占据时间轴；SM active 只说明是否有活跃 warp；Tensor active 和 DRAM 带宽是不同资源的活动指标。高 SM active 不等于达到计算峰值。若 verify 的 GEMM 工作、Tensor activity 和轮延迟共同增长，而带宽不再接近峰值，才更支持向计算侧移动；当前资料不提供逐 kernel achieved FLOP/s，不能声称已严格证明 compute bound。</p>'
    body+='<h2>怎样用 roofline 理解收益下降</h2><p>对一个 BF16 GEMM，令输入行数为 M、权重为 K×N。只计权重读取时，计算量约 2MKN FLOPs、权重流量约 2KN bytes，因此算术强度近似 M FLOP/byte。AR 的 M 通常随活跃 BS 增长；static verify 的 M 近似 BS×verify width。这个推导忽略 KV、激活、缓存重用与其他算子，属于解释方向的近似。</p><p>小 BS 下，SD 把多个候选 token 的 target 计算合并，能够分摊权重读取和 launch。BS 增大后，AR 本身也在更好地分摊权重读取；verify 则处理更多行，较早进入高计算量区域。所以即使 τ 不变，T_SD/T_AR 也可能增加，gain≈τ×T_AR/T_SD 随之下降。这不要求 draft 是唯一瓶颈，也不意味着 decode 永远 memory bound。</p><p>判断本机瓶颈时先看完整周期及分阶段 GPU 工作，再结合带宽、Tensor activity 和实际活跃 BS；最后才考虑单 kernel roofline。MoE 的实际权重流量还取决于专家路由，hybrid 还包含 recurrent-state verify/commit，这个 dense GEMM 近似不能直接给它们预测加速倍数。</p>'
    body+='<h2>同宽度、动态 verify 与 hybrid 消融</h2>'
    for title,run_id,slug in [('统一 verify width=8',active.get('width8','refine01_width8'),'psd_width8'),('DSpark static / compact：原始 probe 表',active.get('adaptive','refine01_adaptive'),'psd_adaptive'),('DSpark static / compact：graph bucket 对齐表',active.get('adaptive_graph','graphaware01_adaptive'),'psd_adaptive_graph'),('Qwen3.5-4B hybrid',active.get('hybrid','follow01_hybrid'),'psd_hybrid')]:
        path=LOGS/(run_id+'.campaign.json');report=ROOT/'reports'/('2026-11-04_'+slug+'.html')
        status=json.loads(path.read_text())['status'] if path.exists() else '尚未运行'
        link=f'<a href="{report.name}">{title}</a>' if report.exists() else title
        body+=f'<p>{link}：{html.escape(status)}。</p>'
        if slug=='psd_width8' and report.exists():
            body+=width_comparison(scaling,output)
        if slug in ['psd_adaptive','psd_adaptive_graph'] and report.exists():
            adaptive=read_csv(report.with_suffix('')/'scaling.csv');adaptive_rows=[]
            for dataset in sorted({r['dataset'] for r in adaptive}):
                for bs in [1,4,16]:
                    a=next((r for r in adaptive if r['dataset']==dataset and r['method']=='dspark' and int(r['bs'])==bs),None)
                    b=next((r for r in adaptive if r['dataset']==dataset and r['method']=='dspark/compact' and int(r['bs'])==bs),None)
                    if a and b:adaptive_rows.append([dataset,bs,f"{float(a['gain']):.2f}×",f"{float(b['gain']):.2f}×",f"{float(b['gain'])/float(a['gain']):.2f}×",f"{float(a['tau']):.2f} → {float(b['tau']):.2f}",f"{float(a['request_mean_cycle_median_ms']):.2f} → {float(b['request_mean_cycle_median_ms']):.2f}"])
            body+=table(['任务','BS','Static / AR','Compact / AR','Compact / Static','τ，static→compact','周期 ms，static→compact'],adaptive_rows)
            body+=image('2026-11-04_'+slug+'/gain_scaling.svg','相同 checkpoint 与 SPS table 的 static/compact 对照')
            body+='<p>这组 static/compact 都启用 core + CPU step interval 记录，以核对实际调度；没有启用 confidence tensor D2H 或 GPU event 计时。它与主矩阵独立，不能将观测开关的成本隐藏进算法收益。</p>'
            schedule_stem='dspark_schedule_'+run_id
            scheduling=output/(schedule_stem+'.json')
            if scheduling.exists():
                schedule=json.loads(scheduling.read_text())
                if schedule['run_id']==run_id:
                    body+=table(['模式','配置 BS','记录轮数','Graph 行 / 同 BS static padded 行','减少 graph 行的轮数','满 BS 预测 / 记录周期 ms'],[[r['mode'],r['configured_bs'],r['records'],f"{r['graph_rows_relative_to_uniform_static']:.1%}",r['graph_trimmed_calls'],f"{r['full_BS_predicted_step_ms_median']:.2f} / {r['full_BS_recorded_step_cpu_ms_median']:.2f}" if r.get('full_BS_predicted_step_ms_median') is not None and r.get('full_BS_recorded_step_cpu_ms_median') is not None else '—'] for r in schedule['cases']])
                    body+='<p>调度记录包含 warmup 与测量阶段，仅用于核对预算与 graph 行数；吞吐仍来自排除 warmup 的请求计时。compact 的 num_verify_tokens 来自 layout.graph_num_tokens；static 的同名字段是 raw 行数，因此 static graph 行数按 capture bucket 推算。不把预算减少直接当成 GPU 计算减少。<a href="'+STEM+'/'+schedule_stem+'.json">调度核查</a> · <a href="'+STEM+'/'+schedule_stem+'_steps.csv">逐轮记录</a>。</p>'
            tail_stem='dspark_tail_'+run_id;tail_path=output/(tail_stem+'.json')
            if tail_path.exists():
                tail=json.loads(tail_path.read_text());tail_rows=[]
                for group in tail['groups']:
                    last_a,last_b=group['slowest_prompt']['static'],group['slowest_prompt']['compact']
                    a=group['requests'][last_a];b=group['requests'][last_b]
                    tail_rows.append([group['dataset'],f'{last_a} / {last_b}',f"{group['wave_median_ms']['static']:.1f} → {group['wave_median_ms']['compact']:.1f}",f"{a['static']['TTFT_ms']:.1f} → {b['compact']['TTFT_ms']:.1f}",f"{a['static']['verify_steps']:.0f} → {b['compact']['verify_steps']:.0f}",str(a['first_mismatch']) if last_a==last_b else '不同最慢请求'])
                body+=table(['任务','最慢 prompt，static / compact','整批 wall ms','最慢请求 TTFT ms','最慢请求 verify 轮数','其首次输出分歧'],tail_rows)
                body+=image(STEM+'/'+tail_stem+'.svg','逐请求 E2E：虚线下方表示 compact 更快；同一批最后完成的请求决定 wall throughput')
                body+='<p class="finding">MT-Bench 的 graph 行数减少约 34%，但最慢请求多跑了 verify 轮次，整批完成时间反而增加；TTFT 基本不变。请求平均周期、总 GPU 工作与固定 batch makespan 是不同指标。输出路径同时存在 BF16 分歧，因此不能把全部变化单独归因于预算策略。此结果不能直接预测在线 continuous batching 的容量收益。</p>'
        if slug=='psd_hybrid' and report.exists():
            hybrid=read_csv(report.with_suffix('')/'scaling.csv');hybrid_rows=[]
            for bs in [1,4]:
                ar=next((r for r in hybrid if r['method']=='ar' and int(r['bs'])==bs),None)
                sd=next((r for r in hybrid if r['method']=='dflash' and int(r['bs'])==bs),None)
                if ar and sd:hybrid_rows.append([bs,f"{float(ar['median_tps']):.1f}",f"{float(sd['median_tps']):.1f}",f"{float(sd['gain']):.2f}×",f"{float(sd['tau']):.2f}",f"{float(ar['physical_peak_GiB']):.2f} / {float(sd['physical_peak_GiB']):.2f}"])
            body+=table(['BS','AR tok/s','DFlash tok/s','gain','τ','AR / SD NVML GiB'],hybrid_rows)
            body+='<p>hybrid 使用自己的 tokenizer、8 条输入、cap128、3 次重复；SM120 wrapper 不接受 FA4 target，因此 full/linear target 都为 triton，draft 为 FA4。它是独立 target/backend cohort，不能与 Qwen3-8B 直接合并绝对吞吐。</p>'
            mem_path=output/'hybrid_memory.json'
            if mem_path.exists():
                mem=json.loads(mem_path.read_text())
                if mem['run_id']==run_id:
                    body+=table(['方法','BS','基础 SSM GiB','额外中间 SSM GiB','Target KV GiB','Draft KV GiB'],[[r['method'],r['bs'],f"{r['base_SSM_state_GiB']:.3f}",f"{r['intermediate_SSM_state_GiB']:.3f}",f"{r['target_KV_GiB']:.3f}",f"{r['draft_KV_GiB']:.3f}"] for r in mem['cases'] if r['method']=='dflash'])
                    body+='<p class="finding">Hybrid 的额外显存不仅是 draft：Qwen3.5-4B 有 24 个 GDN 层，每请求完整 FP32 recurrent state 约 48 MiB。当前实现为 verify 的每个位置缓存中间 state；BS1 额外中间 SSM 为 1.50 GiB，BS4 为 3.75 GiB，已与分配日志核对。形状包含 pool padding slot，所以小 BS 的比例较大，不应把这个布局常数当作算法固有要求。</p><p>本次 ReplaySSM spec 关闭。上游有可选 ring/replay 路径跳过这块中间 SSM buffer，是值得独立验证的显存优化方向；这里没有其性能或质量数据。<a href="'+STEM+'/hybrid_memory.json">状态形状、分配日志与源码 SHA256</a>。</p>'
    body+='<p>同宽度实验只将训练宽度 16 的 DFlash/Domino 缩到 8，不把训练宽度 8 的模型盲目扩到 16。DSpark compact 必须读取本机测得 SPS table，校准使用 simulated accept=1，校准输出不进入吞吐或正确性表。主矩阵 static 结果不能称为完整动态 DSpark 收益。</p>'
    sps_path=LOGS/(active.get('sps','refine03_sps')+'.sps.json')
    mapped_path=sps_path.with_name(sps_path.stem.replace('.sps','')+'.graph_bucket.sps.json')
    if sps_path.exists() and mapped_path.exists():
        raw_sps=json.loads(sps_path.read_text());aligned_sps=json.loads(mapped_path.read_text());costs=[]
        for m in [16,64,65,96,127,128]:
            raw_index=max(0,bisect_right(raw_sps['sample_batch_tokens'],m)-1)
            aligned_index=max(0,bisect_right(aligned_sps['sample_batch_tokens'],m)-1)
            graph=raw_sps['sample_batch_tokens'][bisect_left(raw_sps['sample_batch_tokens'],m)]
            costs.append([m,graph,f"{1000/raw_sps['sample_steps_per_sec'][raw_index]:.2f}",f"{1000/aligned_sps['sample_steps_per_sec'][aligned_index]:.2f}"])
        body+='<h3>成本模型要对应实际 graph，而非只对应请求预算</h3>'+table(['预算总行数 M','补齐后 graph 行数','原始 probe 表预测 ms','Graph-aware 表预测 ms'],costs)
        body+='<p>SpsCostTable 采用 floor lookup，graph runner 采用 ceil bucket。在本次 [8,16,32,64,128] 的 graph 网格下，直接使用 probe 坐标会低估 65–127 行预算的实际 graph 成本。对照表保留全部五个实测 SPS 值，只将 floor-lookup 区间起点改为 [1,9,17,33,65]；对 M=1..128 已逐项验证它等价于查找向上补齐 graph 的实测值。这些起点是查询区间边界，不是新的实测点。</p><p>两种表作为独立实验保留；一维模型仍把请求数与 verify 行数混在一起，不能视作完整二维成本模型。<a href="../logs/parallel_sd_20261104/'+mapped_path.with_suffix('.mapping.json').name+'">映射与源表 SHA256</a>。</p>'
    body+='<p>hybrid 的特征使用与 recurrent-state 提交是两件事：Qwen3.5-4B draft 的 taps [1,5,9,13,17,21,25,29] 全是 linear-attention block 输出；它们不是 GDN 内部 state。能捕获 hidden 不代表 verify/accept 的 state commit 已验证，需看对应实验的实际运行结果。</p>'
    body+='<h2>数值一致性和结论边界</h2>'
    drift=max(float(r['AR_drift']) for r in scaling);cv=max(float(r['max_repeat_CV']) for r in scaling)
    body+=f'<p>当前主矩阵 AR 前后最大漂移 {drift:.2%}，cell 最大重复 CV {cv:.2%}；累计 cached tokens {sum(int(r["cached_tokens"]) for r in scaling)}，retractions {sum(int(r["retractions"]) for r in scaling)}。完整性是运行和算式核查，不是质量评估。</p>'
    body+='<p>AR 与 SD、不同 BS 之间已有 BF16 token 分歧，配置内重复稳定也不能推出跨形状逐 token 一致。保留全部输出 IDs 和 first mismatch；没有把原因直接归结为舍入，也没有以理论拒绝采样精确性代替本机数值验证。FP32-head/logprob 诊断与 headline 吞吐分开。</p>'
    numeric_path=output/'numerics.json'
    if numeric_path.exists():
        numeric=json.loads(numeric_path.read_text())
        if numeric['run_id']==active.get('numerics'):
            body+=table(['enable_fp32_lm_head','AR/SD 相同输出','请求数'],[[r['enable_fp32_lm_head'],r['exact_prompts'],r['samples']] for r in numeric['comparisons']])
            default=next((r for r in numeric['comparisons'] if not r['enable_fp32_lm_head']),None)
            precise=next((r for r in numeric['comparisons'] if r['enable_fp32_lm_head']),None)
            if default and precise:
                body+=f'<p class="finding">两个已知分歧 prompt：默认 head 的 AR/SD 精确匹配 {default["exact_prompts"]}/{default["samples"]}，FP32 head 输出时为 {precise["exact_prompts"]}/{precise["samples"]}。BF16 首次分歧至少一侧存在 top-score tie；这指向 head 数值精度/实现路径的重要影响，不能据此证明全部分歧都只来自最终舍入，也不能推出全数据数值等价。</p>'
            diagnostic=[]
            for row in numeric['comparisons']:
                for prompt in row['prompts']:
                    if prompt['equal']:continue
                    diagnostic.append([row['enable_fp32_lm_head'],prompt['prompt'],prompt['first_mismatch'],html.escape(repr(prompt.get('AR_token_text',prompt['AR_token_id']))),html.escape(repr(prompt.get('SD_token_text',prompt['SD_token_id']))),f"{prompt['AR_top2_logprob_margin']:.4g}" if prompt['AR_top2_logprob_margin'] is not None else '—',f"{prompt['SD_top2_logprob_margin']:.4g}" if prompt['SD_top2_logprob_margin'] is not None else '—'])
            body+=table(['FP32 head','Prompt','首次分歧位置','AR token','SD token','AR top2 margin','SD top2 margin'],diagnostic)
            body+=table(['BF16 logprob 诊断 vs headline','相同输出 / 请求数'],[[r['method'],f"{r['exact_prompts']}/{r['samples']}"] for r in numeric['BF16_logprob_vs_headline']])
            body+='<p>仅两个 GSM8K prompt 的诊断。只在首次分歧处比较 top-logprob，因为此时前缀仍一致；FP32 head 输出仍保留 BF16 backbone 和权重，不能证明整个模型数值等价。返回 logprob 会改变执行路径，其 BF16 输出与 headline 的比较也保留在 <a href="'+STEM+'/numerics.json">诊断 JSON</a> 中，不使用诊断吞吐排名。</p>'
    body+='<p>旧 JetSpec 原生图模式与 eager batch reference 是其他 engine/cohort；本次 SGLang 收益不能与其 baseline 混除。<a href="2026-11-04_jetspec_profile.html">已发布的 JetSpec 可视化报告</a>保留原始测量和 KV 预留布局限制。</p>'
    body+='<h2>由本轮证据得到的优化方向</h2>'+table(['方向','直接证据','仍需独立验证'],[
        ['按 BS / 任务选择 verify 宽度','width8 在大 BS 改善成本，在小 BS 损失 acceptance','长期上下文、自然 EOS 与在线流量下的策略'],
        ['让成本模型对应 replay graph','floor SPS lookup 与 ceil graph bucket 存在档位差；已做映射对照','二维请求数/verify 行数模型和调度目标'],
        ['处理 tail 与 graph padding','BS9–15 接近 padded BS16 的成本；最慢请求决定固定波次完成时间','更细 capture buckets 的显存/启动/吞吐折中'],
        ['降低 hybrid verify 状态缓冲','中间 SSM 在 BS1/4 占 1.50/3.75 GiB','ReplaySSM ring/replay 的净显存、速度及数值行为'],
        ['优先定位真实关键路径','SGLang host NVTX 与 GPU 重叠；大 BS verify GPU 工作增长显著','针对热点 kernel 的细粒度 counter 分析']])
    worker_sha=first['worker_source_sha256']
    worker_link=f'<a href="../logs/parallel_sd_20261104/worker_sources/{worker_sha}.py">{worker_sha}</a>'
    body+='<h2>复现材料</h2>'+table(['项目','版本'],[['SGLang',first['sglang_revision']],*[[html.escape(k),html.escape(v)] for k,v in first['package_versions'].items()],['Worker SHA256',worker_link]])
    body+='<p><a href="../requirements/sglang.txt">环境 lock</a> · <a href="../logs/parallel_sd_20261104/protocol.json">协议</a> · <a href="2026-11-04_psd_scaling/scaling.csv">主矩阵 CSV</a> · <a href="2026-11-04_psd_scaling/source_manifest.json">逐 cell 证据</a> · <a href="2026-11-04_psd_scaling/correctness.json">AR/SD token 比较</a> · <a href="2026-11-04_psd_scaling/batch_output_consistency.json">跨 BS 比较</a> · <a href="../logs/parallel_sd_20261104/campaign_artifact_audit.json">123 项 artifact 核查</a> · <a href="../logs/parallel_sd_20261104/active_runs.json">当前 run 与保留的失败尝试</a>。</p><p>原始 nsys-rep、SQLite 和权重不进入 Git。小型 manifest、CSV 和报告图进入仓库；原始 trace 路径见 profile 清单。</p>'
    intro+='<h2>主要观察</h2><ul><li>GSM8K 的接受长度基本稳定，DFlash 请求平均周期却从 BS1 的 13.19 ms 增至 BS16 的 28.14 ms；AR 从 9.78 增至 11.18 ms。大 BS 的主要增长在 GPU 工作，不能仅凭 host NVTX 长度判断 CPU bubble。</li><li>verify width 需要随 BS 改变：缩到 8 在小 BS 降低收益，却让 MT-Bench BS16 的 DFlash / Domino 从 0.84× / 0.94× 恢复到 1.15× / 1.33×。</li><li>动态预算不是自动加速：graph-aware DSpark 在 MT-Bench 减少约 34% graph 行数，但固定波次吞吐仍回退约 3.6%，需同时分析 acceptance 和尾部请求。</li><li>Hybrid 不只增加 draft KV：Qwen3.5-4B 的中间 recurrent-state 缓冲在 BS1/4 额外占 1.50/3.75 GiB；所选 hidden taps 全来自 linear-attention block。</li><li>两个已知 BF16 分歧 prompt 开启 FP32 head 输出后都与 AR 一致；这是有限的数值诊断，不是全部输入的质量或逐 token 等价证明。</li></ul>'
    text='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>2026-11-04 · Parallel SD 5090</title><style>body{margin:0;background:#eef2f6;color:#203247;font:16px/1.8 system-ui}main{max-width:1300px;margin:28px auto;padding:30px;background:white}h1{line-height:1.4}h2{margin-top:40px;border-bottom:1px solid #dde5ec}a{color:#147d92}img{max-width:100%;height:auto}figure{margin:24px 0}figcaption{color:#627182;font-size:14px}.scroll{overflow:auto}table{border-collapse:collapse;font-size:14px;width:100%}td,th{padding:10px;text-align:left;border-bottom:1px solid #dde5ec;white-space:nowrap}.status,.finding{padding:16px;background:#fff5e6;border-left:4px solid #de8a28}code{overflow-wrap:anywhere}@media(max-width:600px){main{padding:16px;margin:0}h1{font-size:25px}td,th{padding:8px}}</style><main><a href="index.html">← 报告索引</a><h1>Parallel SD：5090 上的多方法扩展性与 GPU 成本</h1>'''+intro+body+'</main></html>'
    path=ROOT/'reports'/(STEM+'.html');path.write_text(text)
    analysis_modules=['scripts/analysis/analyze_parallel_sd.py','scripts/reports/parallel_sd_scaling_figures.py','scripts/reports/parallel_sd_refinement_figures.py','scripts/analysis/analyze_sglang_trace.py','scripts/analysis/summarize_sglang_traces.py','scripts/analysis/package_parallel_sd_traces.py','scripts/analysis/analyze_dspark_schedule.py','scripts/analysis/analyze_adaptive_tail.py','scripts/analysis/analyze_sd_numerics.py','scripts/setup/align_sps_graph_buckets.py','scripts/analysis/audit_hybrid_memory.py','scripts/analysis/audit_parallel_sd_memory.py','scripts/analysis/audit_parallel_sd_campaigns.py']
    (output/'source_manifest.json').write_text(json.dumps(dict(matrix_status=campaign['status'],snapshot_successful_cells=len(successes),live_successful_cells=len(live_successes),not_in_report_snapshot=pending,cohorts=cohort_manifest,source_sha256=sources,analysis_code_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in analysis_modules},report_builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
    write_index();print('Overview',len(successes),'successful cells ->',path)


if __name__=='__main__':main()
