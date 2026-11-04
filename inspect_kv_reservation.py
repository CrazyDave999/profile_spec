"""CPU-check the upstream pool layout and quantify native reserve amplification."""
import json
import math
from pathlib import Path

import torch
from jetspec.engines.jetspec.serving.paged_kv_cache import PagedKVCache
from sweep_worker import save

ROOT=Path(__file__).resolve().parent


def main():
    torch.set_num_threads(2)
    cache=PagedKVCache(block_size=16,device=torch.device('cpu'),dtype=torch.float32)
    kv=torch.zeros((1,2,10,8))
    for layer in range(3):cache.append(kv,kv,layer)
    cache.reserve_capacity(128)
    actual=sum(x.numel()*x.element_size() for x in list(cache._kpool.values())+list(cache._vpool.values()))
    compact=3*(math.ceil(128/16)+1)*16*2*8*4*2
    assert actual==compact*3,(actual,compact)
    conf=json.loads((Path.home()/'models/Qwen3-8B/config.json').read_text())
    layers,heads,dim=conf['num_hidden_layers'],conf['num_key_value_heads'],conf['head_dim']
    rows=[]
    for path in (ROOT/'logs/overnight_20261103').glob('optimized_*_bs1_bench_*_t512.json'):
        r=json.loads(path.read_text()); a=r['arguments']
        if r['status']!='success' or a.get('budget',128)!=128:continue
        capacity=math.ceil(max(r['protocol']['prompt_token_lengths'])/256)*256
        total=capacity+512+(1 if a['arm']=='AR' else 128)
        blocks=math.ceil(total/16)+1
        compact=layers*blocks*16*heads*dim*2*2
        physical=compact*layers
        rows.append({'dataset':a['dataset'],'arm':a['arm'],'session_prompt_capacity':capacity,
            'reserved_token_capacity':total,'per_layer_logical_blocks':blocks,
            'layer_local_compact_pool_bytes':compact,'upstream_physical_pool_bytes':physical,
            'pool_amplification_vs_layer_local_capacity':layers,
            'measured_peak_allocated_bytes':r['results'][a['arm']]['peak_allocated_bytes'],
            'target_parameter_bytes':r['target_parameter_bytes'],'head_parameter_bytes':r['head_parameter_bytes']})
    save(ROOT/'logs/kv_reservation_analysis.json',{
        'CPU_runtime_check':{'layers':3,'reserved_tokens':128,'physical_pool_bytes':actual,
                             'compact_layer_local_bytes':3*9*16*2*8*4*2,'observed_ratio':3,'status':'passed'},
        'Qwen3_geometry':{'layers':layers,'KV_heads':heads,'head_dimension':dim,'dtype_bytes':2,
                         'logical_KV_bytes_per_token':2*layers*heads*dim*2},
        'finding':'Global block IDs are distinct across layers, while each layer allocates a full global-ID-span K/V tensor. reserve_capacity multiplies block count by layer count, and each layer pool has that full count.',
        'scope':'Amplification applies to this upstream per-layer pool/global-ID layout; not an inherent requirement of paged attention or JetSpec.',
        'source':'third_party/JetSpec/jetspec/engines/jetspec/serving/paged_kv_cache.py:155-235',
        'native_cases':rows})
    print('CPU pool-layout check passed. Qwen3 native reserved pool amplification:',layers)


if __name__=='__main__':main()
