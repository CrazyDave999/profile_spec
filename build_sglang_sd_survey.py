"""Build a read-only SGLang SD survey from source and individually checked PRs."""
import csv
from datetime import datetime
import html
import json
import os
from pathlib import Path
import re
import subprocess
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
SRC = ROOT / 'third_party/SGLang'
OUT = ROOT / 'reports/2026-11-04/sglang_sd_research'


def write_csv(name, rows):
    with (OUT/name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def local_link(path):
    target = SRC/path
    assert target.is_file(), target
    return f'<a href="{html.escape(os.path.relpath(target, OUT))}">{html.escape(path)}</a>'


def table(headers, rows):
    return '<table><thead><tr>' + ''.join(f'<th>{html.escape(h)}</th>' for h in headers) + \
        '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f'<td>{v}</td>' for v in r) + '</tr>'
                                      for r in rows) + '</tbody></table>'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    snap = json.loads((OUT/'pull_request_snapshot.json').read_text())
    prs = snap['pull_requests']
    assert all('error' not in r for r in prs)
    bynum = {p['number']:p for p in prs}
    spec = 'python/sglang/srt/speculative/'
    docs = 'docs/docs/advanced_features/'
    methods = [
        ('EAGLE / EAGLE-2','独立算法','EAGLE','特征自回归 drafting + tree verify',spec+'eagle_worker_v2.py',''),
        ('EAGLE-3','独立算法','EAGLE3','多层 target feature + 自回归 drafting',spec+'eagle_worker_v2.py',''),
        ('MTP / NEXTN','模型内置预测头','EAGLE / NEXTN','NEXTN 是 EAGLE 别名；需要目标模型自带兼容 MTP 头',docs+'speculative_decoding.mdx',''),
        ('Frozen-KV MTP','独立算法','FROZEN_KV_MTP','复用/冻结上下文 KV 的 MTP 路径，支持范围依模型与后端',spec+'frozen_kv_mtp_worker_v2.py',''),
        ('Standalone draft model','独立算法','STANDALONE','独立小语言模型逐步 draft，使用已有 verify 路径',spec+'standalone_worker_v2.py',''),
        ('NGRAM','独立算法','NGRAM','不需要 learned draft head；从 token 历史等检索候选',spec+'ngram_worker.py',''),
        ('DFlash','独立算法','DFLASH','一次 block draft，主线默认验证线性链',spec+'dflash_worker_v2.py','22077,23000'),
        ('DFlash2','DFLASH checkpoint 变体','DFLASH + 对应 checkpoint','分组卷积 + candidate selector；不是新的算法枚举',
         'python/sglang/srt/models/dflash.py','35371'),
        ('Domino','DFLASH checkpoint 变体','DFLASH + Domino checkpoint','GRU logit correction / rollout；已有 TP>1 支持',spec+'domino_utils.py','36899,37069'),
        ('LiLiCorr','DFLASH checkpoint 变体','DFLASH + LiLiCorrDraftModel','候选 lattice 转移打分与路径重排，可折叠到 draft graph',spec+'lilicorr_utils.py','37462'),
        ('DSpark','独立算法','DSPARK','置信度驱动的半自回归 drafting 与 verify 调度',spec+'dspark_components/dspark_worker_v2.py',''),
        ('UNO','独立算法','UNO + --uno-lora-path','同一 target 的 LoRA diffusion 路径 draft、AR 路径 verify；可链/树',spec+'uno_worker_v2.py','37667'),
        ('FR-Spec','可选优化','--speculative-token-map','缩小 drafter lm_head 词表子集，不是独立 worker',docs+'speculative_decoding.mdx',''),
        ('Adaptive Spec','可选策略','--speculative-adaptive','动态选择 steps / draft tokens；当前文档限制 EAGLE/EAGLE3 且 topk=1',docs+'adaptive_speculative_decoding.mdx','23336'),
        ('Spec V2','执行机制','V2 worker / overlap scheduler','调度与 CPU/GPU overlap 机制，不等于 draft/verify 跨轮并行',spec+'spec_info.py','23000'),
        ('自定义算法 registry','扩展机制','SpeculativeAlgorithm.register','插件可在树外注册算法；不能仅靠枚举判断所有扩展',spec+'spec_registry.py','23991'),
    ]
    methods = [dict(zip(['method','kind','entry','description','source','related_prs'],r), status='present_in_local_main')
               for r in methods]
    for r in methods:
        assert (SRC/r['source']).is_file(), r['source']
        for number in r['related_prs'].split(','):
            if number: assert bynum[int(number)]['merged'] is True
    write_csv('merged_methods.csv', methods)
    benchmarks = [
        ('在线 serving','python/sglang/benchmark/serving.py','python -m sglang.benchmark.serving',
         'request rate、max concurrency、输入/输出长度、数据集','output tok/s、TTFT、TPOT/ITL、e2e latency、accept length',
         '连接真实 server，包含 scheduler/HTTP；优先用于 AR/SD 端到端对照'),
        ('固定 batch server','python/sglang/benchmark/one_batch_server.py','python -m sglang.benchmark.one_batch_server',
         'batch-size、input-len、output-len；可启动或复用 server','batch latency、吞吐、stream metrics、stage profile',
         '通过真实 server 测量；它的 batch-size/concurrency 不保证所有内部轮次都保持相同 active BS'),
        ('离线 Engine 吞吐','python/sglang/benchmark/offline_throughput.py','python -m sglang.benchmark.offline_throughput',
         '随机/真实输入、生成长度、max-running-requests','request/input/output throughput；可 profiling',
         '默认 Engine backend，不需要 HTTP；仍包含真实 scheduler，不是仅裸模型 forward'),
        ('稳态 serving','python/sglang/benchmark/steady_state_serving.py','稳态 serving 统计辅助模块',
         'steady-state measurement window','稳态 token 吞吐与请求延迟统计',
         '是 serving harness 的统计辅助，不能当成独立 SD 算法测试入口'),
        ('SD 参数 sweep','scripts/playground/bench_speculative.py','python scripts/playground/bench_speculative.py',
         'batch-size × steps × topk × num_draft_tokens；全0为 AR','accept length、20分位 step time、speed 估计值',
         '会起 server；speed=accept_length/step_time，不等同于直接测得的端到端吞吐。固定 prompts 偏探索用途'),
        ('Adaptive vs static','benchmark/bench_adaptive_speculative.py','python benchmark/bench_adaptive_speculative.py',
         '低接受、高接受、交替流量；concurrency','吞吐、平均请求延迟、接受长度',
         '对已经启动的 server 发 /generate 请求；adaptive/static 需要分别配置 server'),
        ('SD 性能回归','test/registered/basic_perf/test_eagle3_latency.py','python -m unittest ...TestEagle3Latency',
         'ShareGPT 300条、request-rate=8、context上限3072','median e2e latency 上界 + accept length 下界',
         '固定 CI model/GPU 阈值，不能直接当作5090性能目标，也不是跨算法公平排名'),
        ('正确性/质量/功能 CI','test/registered/spec/dflash/test_dflash.py','test/registered/spec/ 与 e2e/speculative/',
         '模型、后端、page size、sampling、overlap、EOS、grammar/logprob','GSM8K/MMLU、accept length、API与输出行为',
         'DFlash / EAGLE / DSpark / UNO / NGRAM 等各有套件；许多测试只约束接受长度，没有速度门槛'),
        ('kernel microbenchmark','test/registered/kernels/benchmark/speculative/bench_spec_topk1.py','python .../bench_spec_topk1.py',
         'batch-size 1..2048 × vocab size','topk1 等 helper kernel 的耗时',
         '只说明单个 kernel 的改进，不能直接当端到端 SD 加速比'),
        ('低层目标模型基线','python/sglang/benchmark/one_batch.py','python -m sglang.benchmark.one_batch',
         'batch/input/output lengths、profile','模型 prefill/decode latency',
         '本地源码固定 spec_algorithm=NONE；不要直接用来声明已经覆盖 SD 端到端 worker'),
        ('DSpark cost fitting','python/sglang/benchmark/dspark_sps_profiler.py','python -m sglang.benchmark.dspark_sps_profiler',
         'batch/context/verify配置与 SPS cost table','阶段成本表与拟合',
         '方法专用调优工具；不替代通用 serving throughput/latency benchmark'),
    ]
    benchmarks = [dict(zip(['kind','source','entry','dimensions','metrics','caveat'],r)) for r in benchmarks]
    for r in benchmarks: assert (SRC/r['source']).is_file(), r['source']
    write_csv('benchmark_inventory.csv',benchmarks)
    metadata = {'local_repository':str(SRC),'commit':subprocess.check_output(['git','-C',str(SRC),'rev-parse','HEAD'],text=True).strip(),
                'worktree_status':subprocess.check_output(['git','-C',str(SRC),'status','--short'],text=True),
                'shallow':subprocess.check_output(['git','-C',str(SRC),'rev-parse','--is-shallow-repository'],text=True).strip(),
                'prepared_Beijing':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                'pr_queried_at_utc':snap['queried_at_utc'],'gpu_benchmarks_run':False,
                'scope':'Major implemented methods and selected algorithm/system PRs, not an exhaustive list of all SD bugfix PRs.',
                'search_limitations':'Some keyword searches hit GitHub search rate limits; saved errors are not evidence of absence. Every listed PR was individually fetched and its merged boolean checked.'}
    (OUT/'survey_metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
    enum = re.search(r'class SpeculativeAlgorithm\(Enum\):(.*?)    @classmethod',
                     (SRC/spec/'spec_info.py').read_text(),re.S).group(1)
    enum_names = re.findall(r'^    ([A-Z_0-9]+) = auto\(\)',enum,re.M)
    def e(x): return html.escape(str(x))
    def pr_link(n):
        p=bynum[n];return f'<a href="{p["html_url"]}">#{n}</a>'
    def pr_rows(items):
        return [[pr_link(p['number']),e(p['title']),e('MERGED' if p['merged'] else p['state'].upper()+' / NOT MERGED'),
                 e(p['draft']),e(p['merged_at'] or ''),e(p['description'] if 'description' in p else '')] for p in items]
    content=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>SGLang SD survey</title>',
             '<style>body{font:16px system-ui;max-width:1400px;margin:32px auto;padding:0 20px;line-height:1.6}'
             'table{border-collapse:collapse;width:100%;margin:16px 0}td,th{border:1px solid #ccd;padding:8px;text-align:left;vertical-align:top}'
             'code{background:#f3f5f7;overflow-wrap:anywhere}pre{white-space:pre-wrap}a{overflow-wrap:anywhere}</style>',
             '<h1>SGLang speculative decoding：基准入口与合并状态</h1>',
             '<p>本地 main commit <code>'+e(metadata['commit'])+'</code>；核验日期 '+e(metadata['prepared_Beijing'])+
             '。仅源码调研，没有安装新环境或运行 GPU benchmark；main 已合并不等于已随某个稳定 release 发布。</p>',
             '<p><a href="merged_methods.csv">方法 CSV</a> · <a href="benchmark_inventory.csv">基准入口 CSV</a> · '
             '<a href="pull_request_status.csv">PR 状态 CSV</a> · <a href="pull_request_snapshot.json">完整 PR 元数据快照</a> · '
             '<a href="survey_metadata.json">调研元数据</a></p>',
             '<h2>本地已实现的方法与机制</h2><p>内置枚举：'+e(', '.join(enum_names))+'. '
             'DFlash2、Domino、LiLiCorr 是 DFLASH checkpoint/runtime 分支，不能只靠枚举找支持情况。</p>',
             table(['名称','类别','入口','行为','源码','已核验合并PR'],[
                 [e(r['method']),e(r['kind']),e(r['entry']),e(r['description']),local_link(r['source']),
                  ' '.join(pr_link(int(n)) for n in r['related_prs'].split(',') if n)] for r in methods]),
             '<h2>基准与测试入口</h2>',table(['类型','入口','扫描维度','指标','解释边界'],[
                 [e(r['kind']),local_link(r['source'])+'<br><code>'+e(r['entry'])+'</code>',e(r['dimensions']),e(r['metrics']),e(r['caveat'])]
                 for r in benchmarks]),'<h2>仍开放：重点算法与系统工作</h2>',
             table(['PR','标题','状态','Draft','合并时间','备注'],pr_rows([p for p in prs if p['state']=='open'])),
             '<h2>已关闭但未合并：不要误判为已实现</h2>',
             table(['PR','标题','状态','Draft','合并时间','备注'],pr_rows([p for p in prs if p['state']=='closed' and not p['merged']])),
             '<h2>核验的合并记录</h2>',table(['PR','标题','状态','Draft','合并时间（UTC）','备注'],pr_rows([p for p in prs if p['merged']])),
             '<h2>对当前项目的含义</h2><ul>'
             '<li>优先用 one_batch_server 做固定长度/并发对照，再用 serving 做 request-rate/concurrency 曲线；保存真实活跃BS、吞吐、TTFT/TPOT、accept length。</li>'
             '<li>匹配 target、输入与输出长度、sampling、dtype、KV dtype、attention backend、graph与overlap设置、warmup和prefix-cache状态；AR通过不启用SD得到。</li>'
             '<li>单次多位置draft、CPU/GPU overlap scheduler、跨轮draft/verify流水线是三种不同机制；Spec V2本身不证明后者。</li>'
             '<li>UNO当前文档要求CUDA/FA3、TP=PP=1；合并状态不保证RTX5090与所有后端可运行。本次未做硬件验证。</li>'
             '<li>JetSpec旧PR与SPECTRE均关闭未合并；DFlash2 tree PR仍开放。P-EAGLE在该commit未发现专门内置路径，SpecForge支持训练不能替代serving验证。</li>'
             '<li>D-PACE属于训练目标选择，不是独立SGLang runtime算法；FR-Spec与Adaptive Spec是已有路径上的优化/策略。</li>'
             '</ul><h2>调研范围</h2><pre>'+e(metadata['scope'])+'\n'+e(metadata['search_limitations'])+'</pre></html>']
    (OUT/'report.html').write_text('\n'.join(content))
    print(json.dumps({'methods':len(methods),'benchmarks':len(benchmarks),'verified_prs':len(prs),
                      'open_prs':sum(p['state']=='open' for p in prs),'report':str(OUT/'report.html')},ensure_ascii=False))


if __name__=='__main__':
    main()
