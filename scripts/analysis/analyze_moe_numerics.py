"""First-divergence logits for two known mismatches, not a quality evaluation."""
import _bootstrap
import argparse
import hashlib
import json
from pathlib import Path

from analyze_sd_numerics import top_at, margin

ROOT=_bootstrap.ROOT
LOGS=ROOT/'logs/moe_sd_20261106'
OUT=ROOT/'reports/2026-11-06_moe_parallel_sd'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',required=True);parser.add_argument('--headline-sequence',required=True);args=parser.parse_args()
    campaign=json.loads((LOGS/(args.run_id+'.numerics.json')).read_text());cases={};sources={}
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(str(Path.home()/'models/gpt-oss-20b'))
    for item in campaign['commands']:
        job=LOGS/(item['name']+'.job.json');path=LOGS/(item['name']+'.json')
        if not job.exists() or json.loads(job.read_text())['state']!='success':continue
        data=json.loads(path.read_text());a=data['arguments'];assert a['logprobs'] and not a['profile']
        assert a['fp32_lm_head']==data['engine_resolved_after']['enable_fp32_lm_head']
        cases[a['method'],a['fp32_lm_head']]=data;sources[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    comparisons=[]
    for fp32 in (False,True):
        ar,sd=cases.get(('ar',fp32)),cases.get(('dflash',fp32))
        if not ar or not sd:continue
        assert ar['selected_prompt_sha256']==sd['selected_prompt_sha256'] and ar['package_versions']==sd['package_versions']
        prompts=[]
        for i,(left,right) in enumerate(zip(ar['results'][0]['outputs'],sd['results'][0]['outputs'])):
            x,y=left['output_ids'],right['output_ids'];assert len(x)==len(y)
            first=next((j for j,(a,b) in enumerate(zip(x,y)) if a!=b),None)
            row={'prompt':i,'equal':x==y,'first_mismatch':first}
            if first is not None:
                a,b=top_at(left,first),top_at(right,first)
                row.update(AR_token_id=x[first],SD_token_id=y[first],AR_token_text=tokenizer.decode([x[first]]),SD_token_text=tokenizer.decode([y[first]]),
                    AR_top5=a,SD_top5=b,AR_top2_logprob_margin=margin(a),SD_top2_logprob_margin=margin(b),common_prefix_tail=tokenizer.decode(x[max(0,first-24):first]))
            prompts.append(row)
        comparisons.append({'enable_fp32_lm_head':fp32,'samples':len(prompts),'exact_prompts':sum(p['equal'] for p in prompts),'prompts':prompts})
    checks=[]
    for (method,fp32),data in cases.items():
        if fp32:continue
        refs=sorted(LOGS.glob(args.headline_sequence+f'_full_*_{method}_gsm8k_bs1_w*.json'))
        refs=[p for p in refs if not p.name.endswith(('.job.json','.analysis.json'))]
        reference=json.loads(refs[0].read_text());count=data['arguments']['samples']
        assert reference['frozen_inputs_file_sha256']==data['frozen_inputs_file_sha256']
        checks.append({'method':method,'requests':count,'exact_requests':sum(a['output_ids']==b['output_ids'] for a,b in zip(reference['results'][0]['outputs'][:count],data['results'][0]['outputs'])),'headline_reference':str(refs[0].relative_to(ROOT))})
    evidence={'run_id':args.run_id,'status':campaign['status'],'successful_cells':len(cases),'comparisons':comparisons,'BF16_logprob_vs_headline':checks,'source_sha256':sources,
        'scope':'Two GSM8K prompts deliberately chosen because the headline BF16-head AR/SD results differ. Logprob and FP32-head paths are diagnostics, never timing arms; compare logits only at first divergence while prefixes match. FP32 head does not make the MXFP4/FP8/BF16 backbone fully FP32 and does not prove general quality or distribution equivalence.'}
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'numerics.json').write_text(json.dumps(evidence,indent=2,ensure_ascii=False))
    print(json.dumps([{k:v for k,v in c.items() if k!='prompts'} for c in comparisons]))


if __name__=='__main__':main()
