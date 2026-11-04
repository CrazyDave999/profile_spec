"""Build the dated JetSpec research report from existing measurements (CPU only)."""
from __future__ import annotations

import csv
import hashlib
import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
STEM = "2026-11-04_jetspec_profile"
ASSETS = REPORTS / STEM
DATASETS = ["gsm8k", "math500", "aime25", "humaneval", "mbpp", "livecodebench", "mt_bench"]
NAMES = ["GSM8K", "MATH-500", "AIME25", "HumanEval", "MBPP", "LiveCodeBench", "MT-Bench"]
BS = [1, 2, 4, 8, 16]
PHASES = ["draft_prepare", "draft_forward", "tree_build", "kv_pack", "target_prepare", "verify_forward", "accept", "kv_commit"]
COLORS = ["#a6cee3", "#1f78b4", "#b2df8a", "#33a02c", "#fdbf6f", "#e6550d", "#cab2d6", "#6a3d9a"]
SOURCES = ["reports/2026-11-03/benchmark_results.csv", "reports/2026-11-04/round_latency.csv", "reports/2026-11-04/bs_phase_scaling.csv", "reports/2026-11-04/budget_verify_scaling.csv", "reports/2026-11-03/gpu_hardware_counters.csv", "logs/kv_reservation_analysis.json", "logs/feature_audit_5090.json", "logs/divergence_5090.json"]


def rows(path):
    return list(csv.DictReader((ROOT / path).open()))


def savefig(name):
    plt.savefig(ASSETS / f"{name}.svg", bbox_inches="tight")
    plt.savefig(ASSETS / f"{name}.png", bbox_inches="tight", dpi=160)
    plt.close()


def figure(name, caption):
    return f'<figure><a href="{STEM}/{name}.svg"><img loading="lazy" src="{STEM}/{name}.svg" alt="{html.escape(caption)}"></a><figcaption>{caption} · <a href="{STEM}/{name}.png">PNG</a></figcaption></figure>'


def table(headers, data):
    return '<div class="scroll"><table><thead><tr>' + ''.join(f'<th>{html.escape(str(x))}</th>' for x in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f'<td>{html.escape(str(x))}</td>' for x in row) + '</tr>' for row in data) + '</tbody></table></div>'


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none", "figure.dpi": 120})
    bench = rows(SOURCES[0]); rounds = rows(SOURCES[1]); phases = rows(SOURCES[2]); budget = rows(SOURCES[3]); counters = rows(SOURCES[4])
    kv = json.loads((ROOT / SOURCES[5]).read_text())
    native = {r["dataset"]: r for r in bench if r["engine"] == "optimized"}
    batch = {(r["dataset"], int(r["bs"])): r for r in bench if r["engine"] == "batch_reference"}
    rr = {(r["dataset"], int(r["nominal_bs"]), r["arm"]): r for r in rounds}
    pp = {(r["dataset"], int(r["nominal_bs"]), r["phase"]): r for r in phases}
    assert len(native) == 7 and len(batch) == 35 and len(rounds) == 70

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.7, 1]})
    x = np.arange(7)
    ax.bar(x-.19, [float(native[d]["AR_tps"]) for d in DATASETS], .38, label="JetSpec native AR", color="#687787")
    ax.bar(x+.19, [float(native[d]["JetSpec_tps"]) for d in DATASETS], .38, label="JetSpec tree", color="#147d92")
    ax.set_xticks(x, NAMES, rotation=25, ha="right"); ax.set_ylabel("Output tokens/s (unprofiled, request wall time)"); ax.legend()
    accept = [float(native[d]["JetSpec_accept_length"]) for d in DATASETS]
    gain = [float(native[d]["speedup"]) for d in DATASETS]
    ax2.scatter(accept, gain, color="#147d92", s=45)
    for i, name in enumerate(NAMES): ax2.annotate(name, (accept[i], gain[i]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax2.axhline(1, color="#888", ls=":"); ax2.set_xlabel("Mean accepted/committed tokens per round"); ax2.set_ylabel("Speedup over same native AR")
    fig.suptitle("Native BS=1: workload acceptance matters (cap=512, budget=128)"); fig.tight_layout(); savefig("native_workloads")

    fig,axes=plt.subplots(1,2,figsize=(12,4.3))
    length_rows=[]; extra_sources=[]
    for i,d in enumerate(DATASETS):
        xx=[];yy=[];mm=[]
        for cap in [512,1024]:
            arms={}
            for arm in ["AR","JetSpec"]:
                p=ROOT/f"logs/overnight_20261103/optimized_{d}_bs1_bench_{arm}_t{cap}.json"
                if p.exists():
                    j=json.loads(p.read_text())
                    if j.get("status")=="success":
                        arms[arm]=j["results"][arm]
                        extra_sources.append(str(p.relative_to(ROOT)))
            if len(arms)==2:
                gain=float(arms['JetSpec']['tokens_per_second'])/float(arms['AR']['tokens_per_second'])
                xx.append(cap);yy.append(gain);mm.append(float(arms['JetSpec']['peak_allocated_bytes'])/2**30)
                length_rows.append({'dataset':d,'output_cap':cap,'speedup':gain,'JetSpec_peak_GiB':mm[-1]})
        axes[0].plot(xx,yy,marker="o",label=NAMES[i]);axes[1].plot(xx,mm,marker="o",label=NAMES[i])
    for ax in axes:ax.set_xscale('log',base=2);ax.set_xticks([512,1024],['512','1024']);ax.set_xlabel('Maximum new tokens (EOS still enabled)');ax.grid(alpha=.2)
    axes[0].set_ylabel('Speedup over native AR');axes[0].axhline(1,color='#888',ls=':');axes[1].set_ylabel('JetSpec peak allocated GiB');axes[1].legend(fontsize=8,ncols=2)
    fig.suptitle('Native output-cap sweep: request mixture and reserved capacity also change');fig.tight_layout();savefig('output_cap_scaling')
    with (ASSETS/'output_cap_scaling.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(length_rows[0]));w.writeheader();w.writerows(length_rows)

    fig,axes=plt.subplots(1,2,figsize=(13,4.6))
    native_phases=['draft_forward','tree_build','verify_forward','accept','kv_commit']
    native_rows=[]
    native_analyses={}
    for d in DATASETS:
        p=ROOT/f'logs/overnight_20261103/optimized_{d}_bs1_trace_both_t256.analysis.json'
        extra_sources.append(str(p.relative_to(ROOT)))
        native_analyses[d]=json.loads(p.read_text())
    for ax,field,title in zip(axes,['host_ms','kernel_ms'],['Host NVTX duration (waits included)','GPU kernel sum (launch-correlated)']):
        bottom=np.zeros(7)
        for phase in native_phases:
            color=COLORS[PHASES.index(phase)]
            vv=[]
            for d in DATASETS:
                p=native_analyses[d]['phases']['JetSpec/'+phase];v=p[field]/p['calls'];vv.append(v)
                native_rows.append({'dataset':d,'phase':phase,'metric':field,'ms_per_call':v,'graph_kernel_count':p['graph_kernel_count'],'kernel_count':p['kernel_count']})
            ax.bar(x,vv,bottom=bottom,color=color,label=phase);bottom+=vv
        ax.set_xticks(x,NAMES,rotation=25,ha='right');ax.set_ylabel('ms / SD round');ax.set_title(title)
    axes[1].legend(fontsize=8,ncols=2);fig.suptitle('Native BS=1: host accept waits do not equal accept GPU computation');fig.tight_layout();savefig('native_breakdown')
    with (ASSETS/'native_phase_breakdown.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(native_rows[0]));w.writeheader();w.writerows(native_rows)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for i, d in enumerate(DATASETS):
        axes[0].plot(BS, [float(batch[d,b]["speedup"]) for b in BS], marker="o", label=NAMES[i])
        axes[1].plot(BS, [float(batch[d,b]["JetSpec_accept_length"]) for b in BS], marker="o", label=NAMES[i])
    for ax in axes: ax.set_xscale("log", base=2); ax.set_xticks(BS, [str(b) for b in BS]); ax.set_xlabel("Nominal batch size"); ax.grid(alpha=.2)
    axes[0].axhline(1, color="black", ls=":"); axes[0].set_ylabel("Speedup over eager reference AR")
    axes[1].set_ylabel("Mean accepted tokens per round"); axes[1].legend(fontsize=8, ncols=2)
    fig.suptitle("Experimental eager batch reference: gains shrink; acceptance is less sensitive"); fig.tight_layout(); savefig("batch_scaling")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    for i, d in enumerate(["gsm8k", "mt_bench"]):
        for arm, ls in [("AR", "--"), ("JetSpec", "-")]:
            axes[0].plot(BS, [float(rr[d,b,arm]["mean_complete_round_ms"]) for b in BS], marker="o", ls=ls, color=["#147d92", "#d87520"][i], label=f"{d} {arm}")
        axes[1].plot(BS, [float(rr[d,b,"JetSpec"]["trace_effective_tokens_per_active_row_round"]) for b in BS], marker="o", label=d)
        axes[2].plot(BS, [float(rr[d,b,"JetSpec"]["mean_active_batch"])/float(rr[d,b,"AR"]["mean_active_batch"]) for b in BS], marker="o", label=d)
    for ax in axes: ax.set_xscale("log", base=2); ax.set_xticks(BS, [str(b) for b in BS]); ax.set_xlabel("Nominal BS"); ax.grid(alpha=.2); ax.legend(fontsize=8)
    axes[0].set_ylabel("Complete decode round wall time (ms)"); axes[1].set_ylabel("Effective output tokens / active row / round"); axes[2].set_ylabel("Mean active batch: SD / AR")
    fig.suptitle("Profiled decode: round latency grows, active rows and accepted output also matter"); fig.tight_layout(); savefig("round_scaling")

    for d in ["gsm8k", "mt_bench"]:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
        for ax, field, title in zip(axes, ["host_ms_per_call", "gpu_kernel_ms_per_call"], ["Host NVTX duration (includes waits)", "GPU kernel-duration sum (launch-correlated)"]):
            bottom = np.zeros(5)
            for phase, color in zip(PHASES, COLORS):
                vals = np.array([float(pp[d,b,"JetSpec/"+phase][field]) for b in BS])
                ax.bar(np.arange(5), vals, bottom=bottom, label=phase, color=color); bottom += vals
            ax.set_xticks(np.arange(5), [str(b) for b in BS]); ax.set_xlabel("Nominal BS"); ax.set_ylabel("ms / phase call"); ax.set_title(title)
        axes[1].legend(fontsize=8, ncols=2, loc="upper left"); fig.suptitle(f"{d}: eight SD phases, per-round attribution (eager reference)"); fig.tight_layout(); savefig(f"breakdown_{d}")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for i,d in enumerate(DATASETS):
        rs = sorted([r for r in budget if r["dataset"]==d], key=lambda r:int(r["budget_including_root"]))
        xx=[int(r["budget_including_root"]) for r in rs]
        for ax, field in zip(axes, ["verify_gpu_kernel_ms_per_round", "benchmark_accept_length", "benchmark_tps"]): ax.plot(xx,[float(r[field]) for r in rs], marker="o", label=NAMES[i])
    for ax in axes: ax.set_xscale("log",base=2); ax.set_xticks([64,128,256],["64","128","256"]); ax.set_xlabel("Tree budget (including root)"); ax.grid(alpha=.2)
    axes[0].set_ylabel("Profiled verify kernel sum (ms/round)"); axes[1].set_ylabel("Unprofiled mean accepted length"); axes[2].set_ylabel("Unprofiled output tokens/s"); axes[2].legend(fontsize=7,ncols=2)
    fig.suptitle("Native BS=1 budget sweep: more candidates cost more; throughput optimum varies"); fig.tight_layout(); savefig("budget_scaling")

    memory = {(r["dataset"],r["arm"]):r for r in kv["native_cases"]}
    fig,axes=plt.subplots(1,2,figsize=(14,4.8))
    for offset,arm in [(-.2,"AR"),(.2,"JetSpec")]:
        xx=x+offset; bottom=np.zeros(7)
        for field,label,color in [("target_parameter_bytes","Target weights","#687787"),("head_parameter_bytes","Draft head weights","#147d92"),("upstream_physical_pool_bytes","Physical target KV pool reservation","#de8a28")]:
            vv=np.array([memory[d,arm][field]/2**30 for d in DATASETS]);axes[0].bar(xx,vv,.36,bottom=bottom,color=color,label=label if arm=="AR" else None);bottom+=vv
        rest=np.array([memory[d,arm]["measured_peak_allocated_bytes"]/2**30 for d in DATASETS])-bottom
        assert np.all(rest>=0)
        axes[0].bar(xx,rest,.36,bottom=bottom,color="#b5a6c9",label="Unattributed remainder (caches/graphs/workspace)" if arm=="AR" else None)
    axes[0].set_xticks(x,NAMES,rotation=25,ha="right");axes[0].set_ylabel("Peak allocated GiB; AR left, JetSpec right");axes[0].set_title("Independent head-free AR vs native JetSpec")
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,-.09),ncol=2,fontsize=8)
    for d,marker in [("gsm8k","o"),("livecodebench","s")]:
        for arm,ls in [("AR","--"),("JetSpec","-")]: axes[1].plot(BS,[float(batch[d,b][arm+"_peak_GiB"]) for b in BS],marker=marker,ls=ls,label=f"{d} {arm}")
    axes[1].set_xscale("log",base=2);axes[1].set_xticks(BS,[str(b) for b in BS]);axes[1].set_xlabel("Nominal BS");axes[1].set_ylabel("Peak allocated GiB");axes[1].set_title("Eager batch reference (different KV implementation)");axes[1].legend(fontsize=8)
    fig.suptitle("Measured GPU memory: implementation reservation is a major term");fig.tight_layout();savefig("memory_breakdown")

    fig,ax=plt.subplots(figsize=(10,3.6));lengths=np.array([512,1024,2048,4096])
    for bytes_per_token,label in [(147456,"Target BF16 KV: 144 KiB/token"),(20480,"5-layer draft BF16 KV: 20 KiB/token"),(40960,"5 target taps, raw concat: 40 KiB/token")]: ax.plot(lengths,lengths*bytes_per_token/2**20,marker="o",label=label)
    ax.set_xlabel("Context tokens per request");ax.set_ylabel("Logical storage MiB");ax.set_title("Geometry-derived storage, not an additive runtime peak measurement");ax.legend();ax.grid(alpha=.2);savefig("logical_memory")

    fig,ax=plt.subplots(figsize=(11,4));xx=np.arange(7)
    for i,(phase,color) in enumerate([("AR/decode_forward","#687787"),("JetSpec/draft_forward","#147d92"),("JetSpec/verify_forward","#de8a28")]):
        selected={r["dataset"]:r for r in counters if r["phase"]==phase}
        vals=[float(selected[d]["DRAM Read Bandwidth [Throughput %]"]) for d in DATASETS]
        ax.bar(xx+(i-1)*.24,vals,.24,label=phase,color=color)
    ax.set_xticks(xx,NAMES,rotation=20,ha="right");ax.set_ylabel("DRAM read bandwidth (% peak, phase-overlap samples)");ax.legend(fontsize=8);ax.set_title("Native BS=1 counters only: no measured large-BS roofline conclusion");fig.tight_layout();savefig("hardware_counters")

    # Include all dataset/BS phase data for a dependency-free interactive chart.
    interactive={d:{str(b):{p:{"host":float(pp[d,b,"JetSpec/"+p]["host_ms_per_call"]),"gpu":float(pp[d,b,"JetSpec/"+p]["gpu_kernel_ms_per_call"])} for p in PHASES} for b in BS} for d in DATASETS}
    (ASSETS/"interactive_data.json").write_text(json.dumps(interactive,indent=2))
    source_manifest={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCES+extra_sources}
    (ASSETS/"manifest.json").write_text(json.dumps({"report_date":"2026-11-04","measurement_dates":"2026-11-02 to 2026-11-03 UTC; later CPU reanalysis","target":"Qwen3-8B BF16","gpu":"RTX 5090","native_revision":"673267e4faba71a63099c258f0802c2f3aa44f8e","new_GPU_runs":False,"source_sha256":source_manifest,"figures":sorted(p.name for p in ASSETS.glob('*.svg'))},indent=2))
    native_table=table(["Workload","AR tok/s","JetSpec tok/s","Speedup","Accepted/round","AR / SD peak GiB"],[[NAMES[i],f'{float(native[d]["AR_tps"]):.1f}',f'{float(native[d]["JetSpec_tps"]):.1f}',f'{float(native[d]["speedup"]):.2f}×',f'{float(native[d]["JetSpec_accept_length"]):.2f}',f'{float(native[d]["AR_peak_GiB"]):.2f} / {float(native[d]["JetSpec_peak_GiB"]):.2f}'] for i,d in enumerate(DATASETS)])
    phase_rows=[]
    for d in ["gsm8k","mt_bench"]:
        for p in PHASES:
            a,b=pp[d,1,"JetSpec/"+p],pp[d,16,"JetSpec/"+p]
            phase_rows.append([d,p,*[f'{float(r[k]):.2f}' for k in ["host_ms_per_call","gpu_kernel_ms_per_call"] for r in [a,b]]])
    phase_table=table(["Workload","Phase","Host BS1 ms","Host BS16 ms","GPU BS1 ms","GPU BS16 ms"],phase_rows)
    evidence=''.join(f'<li><a href="../{p}">{p}</a></li>' for p in SOURCES)
    content=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>2026-11-04 · JetSpec profile</title>
<style>body{{margin:0;background:#f3f5f8;color:#203247;font:16px/1.8 system-ui,-apple-system,sans-serif}}main{{max-width:1250px;margin:auto;padding:32px}}header{{background:#18334c;color:white;padding:36px;border-radius:16px}}h1{{margin:0;font-size:32px}}h2{{margin-top:40px}}.badge{{display:inline-block;background:#2a526e;padding:3px 10px;border-radius:12px;margin:10px 6px 0 0;font-size:13px}}section{{background:white;padding:22px 28px;margin-top:24px;border-radius:12px}}a{{color:#14778d}}header a{{color:#b9efff}}.callout{{border-left:4px solid #de8a28;background:#fff7e9;padding:12px 18px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:16px}}.card{{background:#edf4f6;padding:16px;border-radius:8px}}figure{{margin:24px 0}}img{{max-width:100%;height:auto}}figcaption{{font-size:13px;color:#526172}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:9px;border-bottom:1px solid #dde3e9;text-align:left;white-space:nowrap}}th{{background:#edf2f7}}.scroll{{overflow:auto}}code{{background:#edf2f7;padding:2px 5px}}select{{font:inherit;padding:5px}}#interactive svg{{width:100%;min-width:650px}}.legend{{display:flex;gap:12px;flex-wrap:wrap;font-size:12px}}.legend i{{display:inline-block;width:12px;height:12px;margin-right:5px}}footer{{font-size:13px;color:#526172;margin-top:30px}}@media print{{section{{break-inside:avoid}}body{{background:white}}}}</style>
<main><header><p><a href="index.html">← Report index</a> · 2026-11-04 / JetSpec profile</p><h1>Parallel SD 从哪里获益，又在哪里失去收益？</h1><p>RTX 5090 · Qwen3-8B BF16 · JetSpec 原生 engine 与实验性 batch reference</p><span class="badge">175 成功实验</span><span class="badge">63 Nsight traces</span><span class="badge">7 workloads</span><span class="badge">仅整理已有测量</span></header>
<section><h2>已经观察到的规律</h2><div class="grid"><div class="card"><b>接受长度决定 workload 差异</b><p>原生 BS=1 的 GSM8K 为 2.12×，MT-Bench 约 1.07×。相同 budget 下，接受更多 token 才能摊薄整轮成本。</p></div><div class="card"><b>大 BS 的 verify 扩展更快</b><p>eager reference 的 BS1→16，GSM8K / MT-Bench 约 90% 的 SD kernel 时长增量来自 verify。它不只是 CPU overhead。</p></div><div class="card"><b>大 budget 不是稳定更快</b><p>64/128/256 改变 graph bucket 和实际 verify 节点；接受长度提高的收益，需要抵消 verify 和整轮 latency 的增长。</p></div><div class="card"><b>显存差异包含实现预留</b><p>draft 权重约 1.95 GiB；target KV pool 另有布局导致的 36×预留放大。它不是 SD 或 paged attention 的必然成本。</p></div></div></section>
<section><h2>1 · 测量边界与功能开关</h2><p>吞吐来自未挂 profiler 的独立 benchmark，计入每次请求的 tokenization、prefill、decode 和 detokenization，排除 load/compile/warmup。原生 AR 是 JetSpec 自己的 compiled、CUDA-graphed、paged-KV 单 token decoder；不是 SGLang/vLLM。AR 单独进程不加载 draft head。</p><div class="callout"><b>两种实现必须分开读：</b>原生 optimized 只测 BS=1；BS=1/2/4/8/16 sweep 是真正 batched 的 eager SDPA / dense padded KV reference，所有 BS 都无 CUDA Graph。它能解释此实现的扩展规律，不能代表生产 SGLang 的 batch 性能。</div><p>原生 AR/draft/verify 使用 graph replay；测量区间零重新 capture。GPU accept 已开启，但 accept graph、nogather、GEMM weight fusion、FA2 关闭；tree attention 是 Triton paged kernel。CUDA Graph 缩小 host launch 成本，不意味着整个 SD cycle 都在 graph 内。</p><p>实验固定 greedy、thinking off；基础 tree depth=15、width=7、budget=128（含 root），基准输出 cap=512。输出长度 sweep 为 128/512/1024；budget sweep 为 64/128/256。profiling 使用短输入子集和 cap128/256，其时间不能与长 benchmark 混算 speedup。</p></section>
<section><h2>2 · 原生 BS=1：吞吐与接受长度</h2>{figure('native_workloads','独立 benchmark。右图只显示关联，不构成接受长度的单变量因果消融。')}{native_table}<p>数学与对话的接受长度不同，因此“每轮能省掉多少 target decode”不同。更长输出 cap 也改变 EOS 截断和请求构成，已有结果不能单凭 token 上限推断线性收益。</p></section>
<section><h2>3 · Batch scaling：整轮 latency、接受长度与活跃 batch</h2>{figure('batch_scaling','实验性 eager batch reference；不同 BS 输入子集和 padding 不完全一致。')}{figure('round_scaling','短 trace 的完整 decode 轮次，包含阶段间空隙与最后 GPU 尾部，排除 prefill。')}<p>MT-Bench 的 SD 完整轮次从 BS1 的 33.66 ms 增至 BS16 的 159.95 ms；AR 从 25.70 ms 增至 45.63 ms。effective accepted output 每活跃行每轮约 3.50→3.74，但 SD/AR 平均活跃 batch 比下降到约 0.68。完成序列退出、EOS 和批次尾部使“accept length 与 BS 无关”只适合作近似。</p><p><code>decode gain = effective output/active-row/SD-round × mean_active_BS_SD/mean_active_BS_AR × AR_round_ms/SD_round_ms</code></p><p>该等式已对同一 trace 的 token 数和 decode 窗口验证。不能用 cap512 benchmark 的接受率配 cap128 trace latency 计算精确 gain。</p></section>
<section><h2>4 · Latency breakdown：CPU 段与 GPU 工作分开看</h2><p>下图每个阶段都通过 CPU launch correlation 归因到 GPU；异步 kernel 即使在 NVTX pop 后才执行，也计入发起它的阶段。host NVTX 包含 API 和等待，不是纯 CPU 计算。host 与 GPU 两幅图不可相加。</p><label>选择 workload：<select id="dataset">{''.join(f'<option value="{d}" '+('selected' if d=='mt_bench' else '')+f'>{n}</option>' for d,n in zip(DATASETS,NAMES))}</select></label> <label>时间口径：<select id="metric"><option value="gpu">GPU kernel duration sum</option><option value="host">Host NVTX duration</option></select></label><div class="scroll" id="interactive"></div><div class="legend" id="legend"></div><noscript>{figure('breakdown_mt_bench','默认 MT-Bench breakdown；启用 JavaScript 可切换其他 workload。')}</noscript>{figure('breakdown_gsm8k','GSM8K：GPU verify 的增加占主要增量；CPU accept 的增长可能包含等待先前 verify。')}{phase_table}<p>tree_build / accept / kv_commit 不在主要 forward graph 中，确有小 kernel、host 决策或同步边界。但长 accept CPU 范围常在等待前面异步发出的 verify；不能将这段等待全部归因成 accept 自身 GPU 算法。</p></section>
<section><h2>5 · Tree budget sweep：更多候选与更贵的 verify</h2>{figure('budget_scaling','原生 graph bucket 对应 64/128/256；verify、accept 与 throughput 图来自不同计时协议，不能交叉组合成精确速度比。')}<p>verify 是对一组树节点做 causal/tree-masked extend：节点之间有 attention 依赖与祖先关系，既不是独立单 token decode，也不是普通全三角 prompt prefill。GEMM 的 token 维随节点数扩大，类似短 extend。固定 bucket 还会对实际节点补齐。</p><p>已测最小 budget=64；没有 budget16/32 的 GPU 结果。小 budget 不自动退化成单链：width、depth 和全局候选选择仍决定树的形状。只有实际选择的所有节点都落在一条路径上才是单链。</p></section>
<section><h2>6 · AR 与 JetSpec 的 GPU memory</h2>{figure('memory_breakdown','左图是 native 独立进程的 peak allocated 分解；remainder 未进一步归因，不能直接标成 hidden-state 或 graph 的独占占用。')}{figure('logical_memory','从模型几何推导的存储单位，表示不同候选 buffer 的规模；不代表这些 buffer 必然同时持久保存。')}<p>BF16 target 为 36 层、8 KV heads、head_dim128：逻辑 KV 为 <b>144 KiB/token/request</b>。5 层 draft 的逻辑 KV 为 <b>20 KiB/token/request</b>；5 个 target hidden taps 原始拼接为 <b>40 KiB/token/request</b>。这些 feature 可分块投影成 draft KV，不意味着必须长期保留全上下文的每层 hidden。</p><p>native cache 在各层使用不同的 global block IDs，同时每层分配覆盖整个 global-ID span 的 pool。36 层因而相对同 token capacity 的 layer-local 紧凑池多预留 36 倍；CPU 小模型布局检查观察到对应的层数倍放大。这里量化的是预留，<b>没有修改布局并实测性能改善</b>。</p><p>GSM8K：AR peak 19.27 GiB，JetSpec peak 22.01 GiB，差约 2.74 GiB。其中 draft 权重约 1.95 GiB，额外 target tree-capacity 的物理池预留约 0.55 GiB，其余来自未完全拆分的 cache、graph 和 workspace。<code>reserved</code>（allocator）与 <code>allocated</code>（活跃张量）不同，不能混用。</p></section>
<section><h2>7 · GPU util 能说明什么？</h2>{figure('hardware_counters','已有硬件计数只覆盖 native BS=1。1 kHz phase-overlap 采样存在边界混合。')}<p>GPU busy 是“有 kernel 在执行的时间比例”；SM Active 是有 warp 活动的 SM 比例；Tensor Active、SM Issue、DRAM bandwidth 各自描述不同资源。高 GPU busy / SM Active 都不等于 compute bound。大 BS 是否 compute bound，仍需 FLOPs、实际 DRAM bytes、有效带宽/算力和 kernel 级证据。</p><p>现有证据支持：batch reference 的 verify kernel 工作增加、SD 的 GPU duty fraction 升高。它还不足以证明大 BS 已经 compute bound。roofline 可解释机制，但不能代替对这条执行路径的测量。MoE 的专家并集与通信、hybrid 的 recurrent-state commit 也会改变结论。</p></section>
<section><h2>8 · 代表性 Nsight trace 与下一步</h2><ul><li>原生 GSM8K BS1：<code>optimized_gsm8k_bs1_trace_both_t256</code>，观察 AR/draft/verify graph 与 tree accept。</li><li>eager MT-Bench BS1 / BS4 / BS16：观察 verify 增长与 accept 同步等待；GSM8K BS1 / BS16 作为不同接受率对照。</li><li>原生 GSM8K budget64 / 128 / 256：观察 graph bucket 增加带来的 verify 成本。</li></ul><p>大 .nsys-rep / SQLite 保存在本地 ignored <code>profiles/</code>，不提交 Git。原先整包约 2.1 GiB；按需取代表文件，不把整个 ZIP 放进报告仓库。<a href="2026-11-03/representative_timelines_manifest.json">代表 trace 清单</a> · <a href="2026-11-03/timeline_manifest.json">全部 trace 索引</a></p><p>后续先用同一个 SGLang Engine 跑 Qwen3-8B AR、DFlash、Domino、DSpark（固定与动态 verify 分开），独立环境保持原 JetSpec 可复现。固定输入和 backend，逐步扩大 BS/上下文/verify 宽度，再通过最有区分度的 nsys trace 分析主导成本。</p><div class="callout"><b>正确性边界：</b>原始 BF16 AR/tree token equality 未通过；已记录 eager 首个差异 token48（目标 logits 接近/打平）和 graph prompt token53。它与不同形状的浮点舍入一致，但不是 engine 正确性证明。本轮没有新增答案质量评测，也不宣称 bitwise 相同。</div></section>
<section><h2>数据与复现</h2><p><code>source env.sh; python build_jetspec_summary.py</code> 只读已完成的数据，使用 CPU 生成图表，不启动 GPU inference。</p><ul>{evidence}</ul><p><a href="{STEM}/manifest.json">图表与 source SHA256 manifest</a> · <a href="2026-11-03/report.html">旧完整报告</a> · <a href="2026-11-04/round_latency.html">轮次分析</a> · <a href="2026-11-04/bs_phase_scaling.html">阶段分析</a> · <a href="2026-11-04/parallel_sd_checkpoints/report.html">后续方法与模型核查</a></p></section><footer>代码与小型证据在仓库；权重留在外部 Hugging Face cache。测量来自 2026-11-02/03，本文为 2026-11-04 汇总。所有图都有独立 SVG / PNG。</footer></main>
<script>const DATA=__DATA__,PHASES=__PHASES__,COLORS=__COLORS__,BATCH=[1,2,4,8,16];
function draw(){{const dataset=document.getElementById('dataset').value,metric=document.getElementById('metric').value,rows=DATA[dataset];let totals=BATCH.map(b=>PHASES.reduce((s,p)=>s+rows[b][p][metric],0)),max=Math.max(...totals)*1.12;const w=1080,h=335,left=80,top=24,bottom=48,pw=w-left-30,ph=h-top-bottom;let s=`<svg viewBox="0 0 ${{w}} ${{h}}" role="img" aria-label="SD phase breakdown"><text x="12" y="18" font-size="13">ms/round</text>`;for(let i=0;i<=5;i++){{let v=max*i/5,y=top+ph*(1-v/max);s+=`<line x1="${{left}}" x2="${{w-30}}" y1="${{y}}" y2="${{y}}" stroke="#e1e6ed"/><text x="${{left-10}}" y="${{y+4}}" text-anchor="end" font-size="12">${{v.toFixed(1)}}</text>`;}}BATCH.forEach((b,i)=>{{let acc=0,x=left+(i+.2)*pw/5,bw=pw/5*.6;PHASES.forEach((p,j)=>{{let v=rows[b][p][metric],hh=v/max*ph,y=top+ph-(acc+v)/max*ph;s+=`<rect x="${{x}}" y="${{y}}" width="${{bw}}" height="${{hh}}" fill="${{COLORS[j]}}"><title>${{dataset}} BS=${{b}} ${{p}}: ${{v.toFixed(3)}} ms (${{metric}})</title></rect>`;acc+=v;}});s+=`<text x="${{x+bw/2}}" y="${{h-22}}" text-anchor="middle">BS ${{b}}</text><text x="${{x+bw/2}}" y="${{top+ph-acc/max*ph-8}}" text-anchor="middle" font-size="12">${{acc.toFixed(2)}}</text>`;}});document.getElementById('interactive').innerHTML=s+'</svg>';}}
document.getElementById('legend').innerHTML=PHASES.map((p,i)=>`<span><i style="background:${{COLORS[i]}}"></i>${{p}}</span>`).join('');document.getElementById('dataset').addEventListener('change',draw);document.getElementById('metric').addEventListener('change',draw);draw();</script></html>'''
    content=content.replace('__DATA__',json.dumps(interactive)).replace('__PHASES__',json.dumps(PHASES)).replace('__COLORS__',json.dumps(COLORS))
    content=content.replace('输出长度 sweep 为 128/512/1024','原生输出长度 benchmark sweep 为 512/1024')
    content=content.replace('<section><h2>3 · Batch scaling', '<section><h2>输出上限 sweep：吞吐与显存预留</h2>'+figure('output_cap_scaling','原生 512/1024 cap；EOS 保留，cap 不是固定输出长度，也改变 KV 预留容量。')+'</section><section><h2>3 · Batch scaling')
    content=content.replace('<label>选择 workload：',figure('native_breakdown','原生 BS=1：主 forward graph 外仍有 tree/accept/KV 操作；accept 的 host 等待与自身 GPU 工作量差异很大。')+'<label>选择 workload：')
    (REPORTS/f"{STEM}.html").write_text(content)
    index='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Parallel SD research reports</title><style>body{font:17px/1.8 system-ui;max-width:950px;margin:50px auto;padding:24px;color:#203247}a{color:#14778d}li{margin:16px 0}small{color:#657384}</style><h1>Parallel SD · 实验报告</h1><p>新报告统一放在 reports/，文件名使用 YYYY-MM-DD_主题缩写.html；图表及小型证据放在同名目录。大 profiler 文件保留在本地 profiles/。</p><ul><li><a href="2026-11-04_jetspec_profile.html">2026-11-04 · JetSpec profile：latency、batch/budget scaling 与 GPU memory</a><br><small>本机实测汇总，交互阶段图及独立 SVG/PNG。</small></li><li><a href="2026-11-04/parallel_sd_checkpoints/report.html">2026-11-04 · Parallel SD：方法、checkpoint 与大型 MoE 证据</a><br><small>源码/公开结果核查，与本机测量分开。</small></li><li><a href="2026-11-04/sglang_sd_research/report.html">2026-11-04 · SGLang SD benchmark 与 merge/PR 调研</a></li><li><a href="2026-11-03/report.html">2026-11-03 · 原始 JetSpec 全量 overnight 报告</a></li></ul><p>打开 HTML 请下载/克隆仓库后在浏览器访问，或使用 GitHub Pages。GitHub 的源码页面不会直接执行 HTML。</p></html>'''
    (REPORTS/'index.html').write_text(index)
    print(f"Built {REPORTS/STEM}.html; {len(list(ASSETS.glob('*.svg')))} standalone figures.")


if __name__ == "__main__":
    main()
