"""CPU-only analysis of existing batch-reference traces; does not run inference."""
import _bootstrap
import bisect
import collections
import csv
import html
import json
import math
from pathlib import Path
import sqlite3
import zipfile

ROOT = _bootstrap.ROOT
OUT = ROOT / 'reports/2026-11-04'
LOGS = ROOT / 'logs/overnight_20261103'
BS = [1, 2, 4, 8, 16]
DATASETS = ['gsm8k', 'math500', 'aime25', 'humaneval', 'mbpp', 'livecodebench', 'mt_bench']
ORDER = ['draft_prepare', 'draft_forward', 'tree_build', 'kv_pack',
         'target_prepare', 'verify_forward', 'accept', 'kv_commit']
DETAIL_CASES = [('mt_bench', b) for b in [1, 4, 16]] + [('gsm8k', b) for b in [1, 16]]


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def detailed_runtime(database, phases):
    """Use the CPU thread/launch correlation, not GPU timestamp containment."""
    c = sqlite3.connect(f'file:{database}?mode=ro', uri=True)
    strings = dict(c.execute('SELECT id,value FROM StringIds'))
    ranges = collections.defaultdict(list)
    for start, end, tid, text, text_id in c.execute(
            'SELECT start,end,globalTid,text,textId FROM NVTX_EVENTS WHERE end IS NOT NULL'):
        name = text or strings.get(text_id)
        if name in phases and not name.endswith('/other'):
            ranges[tid].append((start, end, name))
    for rows in ranges.values():
        rows.sort()
    starts = {tid: [r[0] for r in rows] for tid, rows in ranges.items()}
    metrics = collections.defaultdict(lambda: collections.defaultdict(float))
    launches = {}
    for start, end, tid, corr, name_id in c.execute(
            'SELECT start,end,globalTid,correlationId,nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME'):
        if tid not in starts:
            continue
        index = bisect.bisect_right(starts[tid], start) - 1
        if index < 0:
            continue
        a, b, phase = ranges[tid][index]
        if start >= b:
            continue
        launches[corr] = phase
        name = strings.get(name_id, '')
        m = metrics[phase]
        if 'Synchronize' in name:
            m['sync_api_ms'] += (end - start) / 1e6
            m['sync_api_count'] += 1
        if 'LaunchKernel' in name:
            m['kernel_launch_count'] += 1
        if 'GraphLaunch' in name:
            m['graph_launch_count'] += 1
        if 'Memcpy' in name:
            m['memcpy_api_count'] += 1
    for start, end, corr, name_id in c.execute(
            'SELECT start,end,correlationId,demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL'):
        phase = launches.get(corr)
        if phase is None:
            continue
        name = strings.get(name_id, '').lower()
        category = ('gemm' if 'gemm' in name or 'gemv' in name else
                    'attention' if any(s in name for s in ['fmha', 'flash', 'attention']) else 'other')
        metrics[phase][f'{category}_kernel_ms'] += (end - start) / 1e6
    c.close()
    for phase, m in metrics.items():
        kernel_sum = sum(m[f'{k}_kernel_ms'] for k in ['gemm', 'attention', 'other'])
        assert math.isclose(kernel_sum, phases[phase]['kernel_ms'], rel_tol=1e-9, abs_tol=1e-5)
    return metrics


def table(headers, rows):
    def cell(x):
        return html.escape(str(x))
    return '<table><thead><tr>' + ''.join(f'<th>{cell(h)}</th>' for h in headers) + \
        '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f'<td>{cell(x)}</td>' for x in row) +
        '</tr>' for row in rows) + '</tbody></table>'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, cases = [], {}
    for dataset in DATASETS:
        for bs in BS:
            name = f'batch_reference_{dataset}_bs{bs}_trace_both_t128'
            run = json.loads((LOGS / f'{name}.json').read_text())
            analysis = json.loads((LOGS / f'{name}.analysis.json').read_text())
            cases[dataset, bs] = (run, analysis)
            shape_info = {}
            for arm in ['AR', 'JetSpec']:
                key = 'AR/decode_forward' if arm == 'AR' else 'JetSpec/verify_forward'
                n = analysis['phases'][key]['calls']
                # The stored forward-shape log also contains the pre-capture warmup.
                # Keep only the last n entries for each arm, which are the captured calls.
                shapes = [s for s in run['actual_forward_batch_shapes'] if s['method'] == arm][-n:]
                assert len(shapes) == n
                shape_info[arm] = {
                    'mean_active_batch': sum(s['batch'] for s in shapes) / n,
                    'mean_padded_prefix': sum(s['max_past'] for s in shapes) / n,
                    'mean_query_nodes': sum(s['query'] for s in shapes) / n,
                }
            details = detailed_runtime(analysis['database'], analysis['phases']) \
                if (dataset, bs) in DETAIL_CASES else {}
            for phase, v in analysis['phases'].items():
                arm = phase.split('/')[0]
                count = v['calls']
                row = {'dataset': dataset, 'nominal_bs': bs, 'phase': phase, 'calls': count,
                       'samples': run['protocol']['samples'], 'tokens_cap': 128,
                       'output_tokens': run['results'][arm]['total_tokens'],
                       'host_ms_per_call': v['host_ms'] / count,
                       'gpu_kernel_ms_per_call': v['kernel_ms'] / count,
                       'gpu_memcpy_ms_per_call': v['memcpy_ms'] / count,
                       'gpu_kernel_ms_per_output_token': v['kernel_ms'] / run['results'][arm]['total_tokens'],
                       **shape_info[arm]}
                for key in ['sync_api_ms', 'sync_api_count', 'kernel_launch_count', 'graph_launch_count',
                            'memcpy_api_count', 'gemm_kernel_ms', 'attention_kernel_ms', 'other_kernel_ms']:
                    row[f'{key}_per_call'] = details[phase][key] / count if phase in details else ''
                rows.append(row)
    write_csv(OUT / 'bs_phase_scaling.csv', rows)
    selected = [r for r in rows if (r['dataset'], r['nominal_bs']) in DETAIL_CASES]
    write_csv(OUT / 'bs_phase_runtime_details.csv', selected)
    bykey = {(r['dataset'], r['nominal_bs'], r['phase']): r for r in rows}

    notes = '''这些数据来自已经完成的 RTX 5090 / Qwen3-8B BF16 batch_reference trace。
所有 BS 使用相同 eager SDPA / dense padded KV 实现，无 CUDA Graph；不是 vLLM 或原生优化 engine。
BS=1/2/4/8/16 分别采集最多 2/4/8/16/32 条输入，即最多两批；AIME25 只有30条。
输出上限128，tree budget=128、depth=15。输入子集和 padding 随 BS 改变，不能当成严格固定输入的 BS 消融。
完成的序列会退出；mean_active_batch 是正式采集中真实的活跃 batch 均值，已剔除 warmup 记录。
host_ms_per_call 是 CPU NVTX 范围长度，包含 CUDA API 和等待；不是纯 CPU 计算时间。
gpu_kernel_ms_per_call 是通过 launch correlation 归因到该 NVTX 的 GPU kernel 时长之和，包含 NVTX pop 后执行的 kernel。
CPU/GPU 时间可能重叠，不能相加；memcpy 单列。sync_api_ms 仅统计显式命名的 CUDA Synchronize API，
不包括其他可能阻塞的 API，也不能将 host-sync 当作纯 CPU 时间。
每轮表排除 prefill；prefill 按每条输入顺序执行，不能拿其总时长比较 BS 扩展。
kernel 类别是按名称划分的 GEMM/GEMV、attention 和其余 kernel，未测 FLOPs 或 DRAM bytes，不能据此确认 compute bound。
profiler 有开销；headline 吞吐应查看 reports/2026-11-03/benchmark_results.csv 的独立运行。
查看 nsys：打开 NVTX、CUDA API、GPU kernel 行，定位 JetSpec/verify_forward 和随后的 accept；
accept 的长 cudaStreamSynchronize 常在等待先前异步提交的 verify，并不代表 accept 自身 GPU 工作很重。
'''
    (OUT / 'README.txt').write_text(notes)
    report = ['<!doctype html><html lang="zh"><meta charset="utf-8"><title>BS phase scaling</title>',
              '<style>body{font:16px system-ui;max-width:1280px;margin:32px auto;padding:0 20px;line-height:1.6}'
              'table{border-collapse:collapse;width:100%;margin:18px 0}td,th{border:1px solid #ccd;padding:7px;text-align:right}'
              'td:first-child,th:first-child{text-align:left}pre{white-space:pre-wrap;background:#f3f5f7;padding:16px}'
              'img{max-width:100%}</style>', '<h1>Batch reference：CPU NVTX 与 GPU kernel 随 BS 的变化</h1>',
              '<p>每个单元格为 CPU host / GPU kernel，单位 ms/阶段调用。解码阶段每次调用对应一个批量轮次；prefill 每次调用处理一条输入。</p>',
              '<p><a href="bs_phase_scaling.csv">全部7数据集 CSV</a> · '
              '<a href="bs_phase_runtime_details.csv">选定案例 CUDA API 与 kernel 类别明细</a> · '
              '<a href="README.txt">口径说明</a></p>',
              '<p>代表性 timelines：<a href="../2026-11-03/nsys_batch_scaling_bs1_bs16.zip">MT-Bench BS=1/16（64 MiB）</a> · '
              '<a href="nsys_mtbench_bs4.zip">MT-Bench BS=4（28 MiB）</a>。每份均包含 AR 和 JetSpec。</p>']
    for dataset in DATASETS:
        report.append(f'<h2>{dataset}</h2>')
        report.append(table(['阶段'] + [f'BS={b}' for b in BS], [
            [phase] + [f"{bykey[dataset,b,phase]['host_ms_per_call']:.3f} / "
                       f"{bykey[dataset,b,phase]['gpu_kernel_ms_per_call']:.3f}" for b in BS]
            for phase in ['AR/prefill', 'AR/decode_forward', 'JetSpec/prefill'] + ['JetSpec/' + p for p in ORDER]]))
        report.append(table(['BS', 'SD轮数', '平均活跃BS', '平均padding前缀长度', 'AR单步GPU ms', 'SD verify GPU ms'], [
            [b, bykey[dataset,b,'JetSpec/verify_forward']['calls'],
             f"{bykey[dataset,b,'JetSpec/verify_forward']['mean_active_batch']:.2f}",
             f"{bykey[dataset,b,'JetSpec/verify_forward']['mean_padded_prefix']:.1f}",
             f"{bykey[dataset,b,'AR/decode_forward']['gpu_kernel_ms_per_call']:.2f}",
             f"{bykey[dataset,b,'JetSpec/verify_forward']['gpu_kernel_ms_per_call']:.2f}"] for b in BS]))
    report.append('<h2>选定案例的 CUDA 同步等待与 kernel 类别</h2>')
    report.append(table(['数据集','BS','阶段','host ms','sync API ms','GPU kernel ms','GEMM ms','attention ms'], [
        [r['dataset'],r['nominal_bs'],r['phase']] + [f'{r[k]:.3f}' for k in [
            'host_ms_per_call','sync_api_ms_per_call','gpu_kernel_ms_per_call','gemm_kernel_ms_per_call',
            'attention_kernel_ms_per_call']] for r in selected if r['phase'].startswith('JetSpec/')
        and r['phase'] != 'JetSpec/other']))
    report.append('<h2>方法与限制</h2><pre>' + html.escape(notes) + '</pre></html>')
    (OUT / 'bs_phase_scaling.html').write_text('\n'.join(report))

    # Reuse the already shared BS=1/16 package. Add only the middle point, to avoid
    # asking the user to download the same traces twice.
    name = 'batch_reference_mt_bench_bs4_trace_both_t128'
    source = ROOT / 'profiles/overnight_20261103' / f'{name}.nsys-rep'
    archive = OUT / 'nsys_mtbench_bs4.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=3) as z:
        z.write(source, source.name)
        z.writestr('README.txt', '中间点 BS=4：配合此前 BS=1/16 对照包查看。每份包含 AR 与 JetSpec。\n'
                   '用 Nsight Systems GUI 2026.5.1 或兼容的新版本打开。\n\n' + notes)
        z.write(OUT / 'bs_phase_scaling.csv', 'bs_phase_scaling.csv')
        z.write(OUT / 'bs_phase_runtime_details.csv', 'bs_phase_runtime_details.csv')
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
    old = ROOT / 'reports/2026-11-03/nsys_batch_scaling_bs1_bs16.zip'
    with zipfile.ZipFile(old) as z:
        assert z.testzip() is None
    summary = {'rows': len(rows), 'detailed_rows': len(selected), 'no_gpu_inference_run': True,
               'archives': [{'path': str(p), 'MiB': p.stat().st_size / 2**20, 'CRC_verified': True}
                            for p in [old, archive]], 'growth': {}}
    for dataset in ['gsm8k', 'mt_bench']:
        sums = [sum(bykey[dataset,b,'JetSpec/'+p]['gpu_kernel_ms_per_call'] for p in ORDER) for b in [1,16]]
        v = [bykey[dataset,b,'JetSpec/verify_forward']['gpu_kernel_ms_per_call'] for b in [1,16]]
        summary['growth'][dataset] = {'mean_sum_SD_kernel_ms_BS1_BS16': sums,
            'verify_fraction_of_kernel_time_growth': (v[1]-v[0])/(sums[1]-sums[0])}
    (OUT / 'bs_phase_scaling_manifest.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    for dataset, bs in DETAIL_CASES:
        print(dataset, bs, [(p, round(bykey[dataset,bs,'JetSpec/'+p]['sync_api_ms_per_call'],3))
                           for p in ['tree_build','verify_forward','accept','kv_pack','kv_commit']])


if __name__ == '__main__':
    main()
