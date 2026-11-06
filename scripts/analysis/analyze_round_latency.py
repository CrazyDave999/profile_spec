"""Measure complete decode-loop wall latency from existing NVTX traces, CPU-only."""
import _bootstrap
import csv
import html
import json
from pathlib import Path
import sqlite3
import statistics

ROOT = _bootstrap.ROOT
OUT = ROOT / 'reports/2026-11-04'
DATASETS = ['gsm8k','math500','aime25','humaneval','mbpp','livecodebench','mt_bench']
BUSY_CASES = {('mt_bench', b) for b in [1,4,16]} | {('gsm8k', b) for b in [1,16]}


def analyze(dataset, bs):
    name = f'batch_reference_{dataset}_bs{bs}_trace_both_t128'
    run = json.loads((ROOT / 'logs/overnight_20261103' / f'{name}.json').read_text())
    c = sqlite3.connect(f'file:{ROOT / "profiles/overnight_20261103" / (name+".sqlite")}?mode=ro', uri=True)
    nvtx = c.execute("SELECT start,end,text,globalTid FROM NVTX_EVENTS WHERE end IS NOT NULL "
                     "AND (text LIKE 'profile_spec/%' OR text LIKE 'AR/%' OR text LIKE 'JetSpec/%') "
                     "ORDER BY start").fetchall()
    syncs = c.execute("SELECT r.start,r.end,r.globalTid FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
                      "JOIN StringIds s ON r.nameId=s.id WHERE s.value LIKE '%DeviceSynchronize%' "
                      "ORDER BY r.start").fetchall()
    requests = [r for r in nvtx if r[2].startswith('profile_spec/')]
    windows, result = [], []
    for arm, first in [('AR','AR/kv_pack'), ('JetSpec','JetSpec/draft_prepare')]:
        periods, elapsed_ns, rounds = [], 0, 0
        for request_index, (a, b, label, tid) in enumerate(requests):
            if label.split('/')[1] != arm:
                continue
            points = [s for s,e,t,th in nvtx if t == first and a <= s < b and th == tid]
            assert points
            # sweep_worker synchronizes immediately after run() returns. Include
            # this tail so the last round's asynchronously queued kernels finish.
            next_request_start = requests[request_index+1][0] if request_index+1 < len(requests) else float('inf')
            sync = next((x for x in syncs if b <= x[0] < next_request_start and x[2] == tid), None)
            assert sync is not None, (name, label)
            end = sync[1]
            windows.append((points[0], end, arm))
            elapsed_ns += end - points[0]
            rounds += len(points)
            periods += [(y-x)/1e6 for x,y in zip(points,points[1:])]
        shapes = [s for s in run['actual_forward_batch_shapes'] if s['method'] == arm][-rounds:]
        assert len(shapes) == rounds
        active_rows = sum(s['batch'] for s in shapes)
        outputs = run['results'][arm]['total_tokens']
        samples = run['protocol']['samples']
        decode_outputs = outputs - samples  # prefill produces the first token per request
        if arm == 'AR':
            assert active_rows == decode_outputs
        bench = json.loads((ROOT / 'logs/overnight_20261103' /
                            f'batch_reference_{dataset}_bs{bs}_bench_{arm}_t512.json').read_text())
        result.append({'dataset':dataset, 'nominal_bs':bs, 'arm':arm, 'samples':samples,
                       'output_cap':128, 'rounds':rounds,
                       'decode_wall_ms_including_final_gpu_sync':elapsed_ns/1e6,
                       'mean_complete_round_ms':elapsed_ns/1e6/rounds,
                       'mean_start_to_next_start_ms_excluding_last_round':statistics.mean(periods),
                       'mean_active_batch':active_rows/rounds,
                       'trace_raw_accept_length':run['results'][arm].get('mean_accept_length'),
                       'trace_effective_tokens_per_active_row_round':decode_outputs/active_rows,
                       'trace_decode_output_tokens':decode_outputs,
                       'trace_decode_tokens_per_second':decode_outputs/(elapsed_ns/1e9),
                       'benchmark_cap512_accept_length':bench['results'][arm].get('mean_accept_length'),
                       'benchmark_cap512_tokens_per_second':bench['results'][arm]['tokens_per_second'],
                       'gpu_kernel_busy_fraction_in_decode_window':None})
    if (dataset,bs) in BUSY_CASES:
        # Merge device kernel intervals inside each decode window. This is a
        # wall-time duty fraction, not SM utilization, occupancy or FLOP efficiency.
        windows.sort()
        index, left, right = 0, None, None
        busy_ns = {'AR':0, 'JetSpec':0}
        for a,b in c.execute('SELECT start,end FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start'):
            while index < len(windows) and a >= windows[index][1]:
                if left is not None:
                    busy_ns[windows[index][2]] += right-left
                    left = right = None
                index += 1
            if index >= len(windows):
                break
            w0,w1,arm = windows[index]
            lo,hi = max(a,w0),min(b,w1)
            if hi <= lo:
                continue
            if left is None:
                left,right = lo,hi
            elif lo <= right:
                right = max(right,hi)
            else:
                busy_ns[arm] += right-left
                left,right = lo,hi
        if left is not None:
            busy_ns[windows[index][2]] += right-left
        for row in result:
            frac = busy_ns[row['arm']] / (row['decode_wall_ms_including_final_gpu_sync']*1e6)
            assert 0 <= frac <= 1
            row['gpu_kernel_busy_fraction_in_decode_window'] = frac
    c.close()
    ar,sd = result
    exact_gain = sd['trace_decode_tokens_per_second']/ar['trace_decode_tokens_per_second']
    factored_gain = sd['trace_effective_tokens_per_active_row_round'] * \
        sd['mean_active_batch']/ar['mean_active_batch'] * ar['mean_complete_round_ms']/sd['mean_complete_round_ms']
    assert abs(exact_gain-factored_gain) < 1e-9
    for row in result:
        row['trace_decode_gain'] = exact_gain
        row['benchmark_cap512_gain'] = sd['benchmark_cap512_tokens_per_second']/ar['benchmark_cap512_tokens_per_second']
    return result


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    rows = [r for dataset in DATASETS for bs in [1,2,4,8,16] for r in analyze(dataset,bs)]
    with (OUT / 'round_latency.csv').open('w',newline='') as f:
        w = csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    notes = '''完整轮次均值 = decode wall time / 正式采集轮数。
每个请求的 decode window 从首个 AR/kv_pack 或 JetSpec/draft_prepare 开始，到该请求返回后紧随的 cudaDeviceSynchronize 完成。
因此包含各阶段、阶段间空隙及最后一轮的 GPU 尾部，不包含 prefill，不把 CPU NVTX 与 GPU kernel 时长相加。
另外保留相邻轮次起点间隔（排除每个请求的最后一轮），供交叉核对。
这些是 instrumented batch_reference trace，cap=128，所有 BS 都没有 CUDA Graph；不是 headline benchmark。
BS=1只采两条输入，BS=16采32条；AIME25只有30条。序列提前结束后退出，padding 和输入子集随 BS 改变。
trace 的 raw accept length 可能含输出截断前的候选 block；effective acceptance 则按实际输出的 decode tokens / 活跃 row-rounds 定义。
benchmark_cap512_* 字段来自另外的无 profiler 运行，不能与 trace latency 混合重建速度比。
gpu_kernel_busy_fraction 是 decode window 内设备 kernel 区间并集 / wall time，不包含纯 memcpy 活跃时间。
它只表示时间上的 kernel duty fraction，不是 SM/Tensor 饱和度，也不能证明 compute bound。
'''
    (OUT / 'round_latency_README.txt').write_text(notes)
    content = ['<!doctype html><html lang="zh"><meta charset="utf-8"><title>SD round latency</title>',
               '<style>body{font:16px system-ui;max-width:1150px;margin:32px auto;padding:0 20px;line-height:1.6}'
               'table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccd;padding:8px;text-align:right}'
               'pre{white-space:pre-wrap;background:#f3f5f7;padding:16px}</style>',
               '<h1>完整生成轮次 latency</h1><p><a href="round_latency.csv">CSV</a> · '
               '<a href="bs_phase_scaling.html">各 NVTX 阶段时间</a> · '
               '<a href="round_latency_README.txt">测量口径</a></p>']
    for dataset in DATASETS:
        content.append(f'<h2>{dataset}</h2><table><tr><th>BS</th><th>AR ms/step</th><th>SD ms/round</th>'
                       '<th>SD/AR周期成本比</th><th>平均活跃BS AR / SD</th><th>benchmark τ</th><th>benchmark gain</th></tr>')
        for bs in [1,2,4,8,16]:
            ar = next(r for r in rows if r['dataset']==dataset and r['nominal_bs']==bs and r['arm']=='AR')
            sd = next(r for r in rows if r['dataset']==dataset and r['nominal_bs']==bs and r['arm']=='JetSpec')
            content.append(f"<tr><td>{bs}</td><td>{ar['mean_complete_round_ms']:.2f}</td>"
                           f"<td>{sd['mean_complete_round_ms']:.2f}</td>"
                           f"<td>{sd['mean_complete_round_ms']/ar['mean_complete_round_ms']:.2f}</td>"
                           f"<td>{ar['mean_active_batch']:.2f} / {sd['mean_active_batch']:.2f}</td>"
                           f"<td>{sd['benchmark_cap512_accept_length']:.2f}</td>"
                           f"<td>{sd['benchmark_cap512_gain']:.2f}×</td></tr>")
        content.append('</table>')
    content.append('<h2>方法与限制</h2><pre>'+html.escape(notes)+'</pre></html>')
    (OUT / 'round_latency.html').write_text('\n'.join(content))
    print('Saved',len(rows),'rows. No new GPU inference.')
    for row in rows:
        if row['dataset'] in ['gsm8k','mt_bench']:
            print(row['dataset'],row['nominal_bs'],row['arm'],
                  round(row['mean_complete_round_ms'],3),'ms',
                  'kernel duty',round(100*row['gpu_kernel_busy_fraction_in_decode_window'],2)
                  if row['gpu_kernel_busy_fraction_in_decode_window'] is not None else None)


if __name__=='__main__':
    main()
