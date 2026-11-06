"""Inspect first-divergence top-logprobs; diagnostics are not timing results."""
import _bootstrap
import argparse
import hashlib
import json
from pathlib import Path

ROOT=_bootstrap.ROOT


def top_at(output,index):
    rows=output['meta_info'].get('output_top_logprobs')
    if rows is None:return None
    assert len(rows)==len(output['output_ids'])
    return rows[index] if index<len(rows) else None


def margin(rows):
    if not rows or len(rows)<2:return None
    values=sorted((float(r[0]) for r in rows),reverse=True)
    return values[0]-values[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',required=True);parser.add_argument('--decode',action='store_true');args=parser.parse_args()
    tokenizer=None
    if args.decode:
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(str(Path.home()/'models/Qwen3-8B'))
    directory=ROOT/'logs/parallel_sd_20261104';campaign=json.loads((directory/(args.run_id+'.campaign.json')).read_text())
    assert campaign['stage']=='numerics'
    cases={};sources={}
    for cell in campaign['commands']:
        path=directory/(cell['name']+'.json');job=directory/(cell['name']+'.job.json')
        if not job.exists() or json.loads(job.read_text())['state']!='success':continue
        data=json.loads(path.read_text());a=data['arguments'];assert data['status']=='success' and a['logprobs']
        assert a['fp32_lm_head']==data['engine_resolved_after']['enable_fp32_lm_head']
        cases[a['method'],a['fp32_lm_head']]=data;sources[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    rows=[]
    for fp32 in [False,True]:
        ar,sd=cases.get(('ar',fp32)),cases.get(('dflash',fp32))
        if ar is None or sd is None:continue
        assert ar['selected_prompt_sha256']==sd['selected_prompt_sha256']
        assert ar['package_versions']==sd['package_versions']
        aa=ar['results'][0]['outputs'];bb=sd['results'][0]['outputs'];prompts=[]
        for index,(a,b) in enumerate(zip(aa,bb)):
            ax,bx=a['output_ids'],b['output_ids'];assert len(ax)==len(bx)
            first=next((n for n,(v,w) in enumerate(zip(ax,bx)) if v!=w),None)
            item=dict(prompt=index,equal=ax==bx,first_mismatch=first)
            if first is not None:
                ar_top,sd_top=top_at(a,first),top_at(b,first)
                item.update(AR_token_id=ax[first],SD_token_id=bx[first],AR_top5=ar_top,SD_top5=sd_top,
                    AR_top2_logprob_margin=margin(ar_top),SD_top2_logprob_margin=margin(sd_top),
                    common_prefix_last_ids=ax[max(0,first-12):first])
                if tokenizer is not None:
                    item.update(AR_token_text=tokenizer.decode([ax[first]]),SD_token_text=tokenizer.decode([bx[first]]),
                        common_prefix_tail=tokenizer.decode(ax[max(0,first-24):first]))
                    for top in [ar_top,sd_top]:
                        if top:
                            for candidate in top:candidate[2]=tokenizer.decode([candidate[1]])
            prompts.append(item)
        rows.append(dict(enable_fp32_lm_head=fp32,samples=len(prompts),exact_prompts=sum(p['equal'] for p in prompts),prompts=prompts))
    headline_checks=[]
    for (method,fp32),data in cases.items():
        if fp32:continue
        reference_paths=sorted(directory.glob(f'full01_*_{method}_gsm8k_bs1.json'))
        reference=json.loads(reference_paths[0].read_text())
        count=data['arguments']['samples'];aa=reference['results'][0]['outputs'][:count];bb=data['results'][0]['outputs']
        assert reference['prompt_file_sha256']==data['prompt_file_sha256']
        assert reference['input_lengths'][:count]==data['input_lengths']
        headline_checks.append(dict(method=method,samples=count,
            exact_prompts=sum(a['output_ids']==b['output_ids'] for a,b in zip(aa,bb)),
            reference=str(reference_paths[0].relative_to(ROOT))))
    output=ROOT/'reports/2026-11-04_parallel_sd/numerics.json'
    output.write_text(json.dumps(dict(run_id=args.run_id,campaign_status=campaign['status'],successful_cells=len(cases),comparisons=rows,BF16_logprob_vs_headline=headline_checks,source_sha256=sources,
        scope='Diagnostic only. Logprob return and head precision alter execution; do not pool timings with default BF16 benchmarks. Compare distributions only at first divergence, where prefixes still match. FP32 head output leaves BF16 backbone/weights and shape-dependent arithmetic.'),ensure_ascii=False,indent=2))
    print(output);print([{k:v for k,v in row.items() if k!='prompts'} for row in rows])


if __name__=='__main__':main()
