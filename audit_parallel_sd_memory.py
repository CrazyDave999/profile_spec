"""Safetensors header accounting, kept separate from physical GPU measurements."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import struct

ROOT=Path(__file__).resolve().parent


def audit():
    models=json.loads((ROOT/'models-parallel-sd.json').read_text())
    rows=[]
    for method in ['dflash','domino','dspark','dflash2']:
        model=models[method];directory=Path(model['alias'])
        config=json.loads((directory/'config.json').read_text())
        parts=defaultdict(int);tensors=[]
        for entry in model['weights']:
            file=directory/entry['file']
            with file.open('rb') as stream:
                length=struct.unpack('<Q',stream.read(8))[0]
                header=json.loads(stream.read(length))
            for name,value in header.items():
                if name=='__metadata__':continue
                size=value['data_offsets'][1]-value['data_offsets'][0]
                key='embedding_and_lm_head' if name in ['embed_tokens.weight','lm_head.weight'] else ('extra_heads' if any(n in name for n in ['markov_head','confidence_head','candidate_selector','prefix_gru','embed_proj']) else 'backbone_and_projection')
                parts[key]+=size;tensors.append({'name':name,'shape':value['shape'],'dtype':value['dtype'],'bytes':size,'category':key})
        taps=config.get('target_layer_ids') or config.get('dflash_config',{}).get('target_layer_ids')
        dim=config.get('head_dim',config['hidden_size']//config['num_attention_heads'])
        rows.append(dict(method=method,repo=model['repo'],revision=model['revision'],
            config_sha256=hashlib.sha256((directory/'config.json').read_bytes()).hexdigest(),
            file_GiB=sum(x['bytes'] for x in model['weights'])/2**30,
            tensor_GiB=sum(parts.values())/2**30, parts_GiB={k:v/2**30 for k,v in parts.items()},
            non_vocab_tensor_GiB=(sum(parts.values())-parts.get('embedding_and_lm_head',0))/2**30,
            draft_KV_bytes_per_token=2*config['num_hidden_layers']*config['num_key_value_heads']*dim*2,
            raw_selected_hidden_bytes_per_token=len(taps)*config['hidden_size']*2,
            projected_hidden_bytes_per_token=config['hidden_size']*2,
            draft_layer_count=config['num_hidden_layers'],target_layer_ids=taps,tensors=tensors))
    sources=['third_party/SGLang/python/sglang/srt/models/dspark.py',
             'third_party/SGLang/python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py']
    result=dict(checkpoints=rows,source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},
        interpretation=['Checkpoint tensor bytes are not physical GPU peak.',
          'DSpark Qwen loader skips checkpoint embed_tokens/lm_head; worker attaches target modules.',
          'KV and hidden rows are BF16 logical bytes per token, not all additional persistent allocations.',
          'Raw target hidden taps need not be retained for the entire history once materialized as draft KV.',
          'Measured logs show target 32768-token KV=4.5GiB and draft KV=0.625GiB for this cohort.'])
    path=ROOT/'logs/parallel_sd_20261104/checkpoint_memory_audit.json'
    path.write_text(json.dumps(result,indent=2));return result


if __name__=='__main__':
    result=audit()
    for row in result['checkpoints']:
        print(row['method'],{k:row[k] for k in ['file_GiB','non_vocab_tensor_GiB','draft_KV_bytes_per_token','raw_selected_hidden_bytes_per_token']})
