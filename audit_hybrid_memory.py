"""Explain hybrid spec scratch using allocated-shape source and runtime logs."""
import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',required=True);args=parser.parse_args()
    directory=ROOT/'logs/parallel_sd_20261104';campaign=json.loads((directory/(args.run_id+'.campaign.json')).read_text())
    target=json.loads((Path.home()/'models/Qwen3.5-4B/config.json').read_text())['text_config']
    draft=json.loads((Path.home()/'models/qwen3.5-4b-dflash/config.json').read_text())
    linear=target['layer_types'].count('linear_attention');full=target['layer_types'].count('full_attention')
    state_bytes=linear*target['linear_num_value_heads']*target['linear_value_head_dim']*target['linear_key_head_dim']*4
    target_kv=2*full*target['num_key_value_heads']*target['head_dim']*2
    draft_kv=2*draft['num_hidden_layers']*draft['num_key_value_heads']*draft['head_dim']*2
    taps=draft['dflash_config']['target_layer_ids'];assert all(target['layer_types'][i]=='linear_attention' for i in taps)
    rows=[]
    for cell in campaign['commands']:
        job=directory/(cell['name']+'.job.json');path=directory/(cell['name']+'.json')
        if not job.exists() or json.loads(job.read_text())['state']!='success':continue
        data=json.loads(path.read_text());a=data['arguments'];state=data['engine_resolved_after']['internal_states'][0]
        assert a['target']=='qwen3.5-4b'
        log=(directory/(cell['name']+'.log')).read_text();allocation_lines=[s for s in log.splitlines() if 'Mamba Cache is allocated' in s or 'KV Cache is allocated' in s]
        slots=state['max_mamba_cache_size']+1
        width=data['engine_resolved_after']['speculative_num_draft_tokens'] if a['method']!='ar' else 0
        intermediate=state_bytes*slots*width
        assert not state['enable_linear_replayssm_spec'],'ReplaySSM uses a different scratch model.'
        if a['method']!='ar':
            line=next(s for s in allocation_lines if 'intermediate_ssm_state_cache size:' in s)
            recorded=float(re.search(r'intermediate_ssm_state_cache size: ([0-9.]+)GB',line).group(1))
            assert abs(recorded-intermediate/2**30)<.011
        rows.append(dict(case=cell['name'],method=a['method'],bs=a['bs'],physical_peak_GiB=data['physical_device_peak_used_bytes']/2**30,
            runtime_mamba_slots_without_padding=state['max_mamba_cache_size'],slots_with_padding=slots,
            base_SSM_state_GiB=state_bytes*slots/2**30,intermediate_SSM_state_GiB=intermediate/2**30,
            target_KV_GiB=target_kv*32768/2**30,draft_KV_GiB=draft_kv*32768/2**30 if width else 0,
            verify_width=width,allocation_log_lines=allocation_lines))
    sources=['third_party/SGLang/python/sglang/srt/mem_cache/memory_pool.py','third_party/SGLang/python/sglang/srt/speculative/dflash_worker_v2.py']
    output=ROOT/'reports/2026-11-04_parallel_sd/hybrid_memory.json'
    output.write_text(json.dumps(dict(run_id=args.run_id,campaign_status=campaign['status'],cases=rows,target_linear_layers=linear,target_full_layers=full,
        logical_SSM_state_bytes_per_request=state_bytes,draft_layers=draft['num_hidden_layers'],target_layer_ids=taps,
        raw_hidden_tap_bytes_per_token=len(taps)*target['hidden_size']*2,
        formula='intermediate SSM = linear_layers * (spec_state_size+1) * verify_width * value_heads * value_dim * key_dim * sizeof(float32). Runtime max_mamba_cache_size matches spec_state_size in this cohort; allocation logs cross-check the result.',
        source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},
        limits=['Physical peaks are NVML samples; logical tensors do not explain allocator/context/graph/workspace exactly.',
          'The extra slot is the current pool padding/layout, not an intrinsic 2x SD requirement.',
          'ReplaySSM spec is disabled here. Upstream has an opt-in ring/replay path that skips this intermediate SSM buffer; no speedup/quality claim is made without testing it.']),ensure_ascii=False,indent=2))
    print(output);print([{k:r[k] for k in ['method','bs','physical_peak_GiB','base_SSM_state_GiB','intermediate_SSM_state_GiB','draft_KV_GiB']} for r in rows])


if __name__=='__main__':main()
