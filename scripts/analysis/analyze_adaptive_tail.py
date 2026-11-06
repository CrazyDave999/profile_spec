"""Fixed-wave makespan and per-request evidence for adaptive verify."""
import _bootstrap
import argparse
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=_bootstrap.ROOT
LOGS=ROOT/'logs/parallel_sd_20261104'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',required=True);args=parser.parse_args()
    campaign=json.loads((LOGS/(args.run_id+'.campaign.json')).read_text());groups={}
    for cell in campaign['commands']:
        job=LOGS/(cell['name']+'.job.json');path=LOGS/(cell['name']+'.json')
        if not job.exists() or json.loads(job.read_text())['state']!='success':continue
        data=json.loads(path.read_text());a=data['arguments']
        if a['method']!='dspark':continue
        groups.setdefault((a['dataset'],a['bs']),{})[a['ragged_mode']]=(cell['name'],data)
    summaries=[]
    for (dataset,bs),modes in sorted(groups.items()):
        if set(modes)!= {'static','compact'}:continue
        static,compact=modes['static'][1],modes['compact'][1]
        assert static['selected_prompt_sha256']==compact['selected_prompt_sha256']
        assert static['arguments']['samples']==compact['arguments']['samples']==bs,'This analysis is for a single fixed wave.'
        rows=[]
        for index in range(bs):
            a=static['results'][0]['outputs'][index]['output_ids'];b=compact['results'][0]['outputs'][index]['output_ids']
            first=next((n for n,(v,w) in enumerate(zip(a,b)) if v!=w),None)
            row=dict(prompt=index,outputs_equal=a==b,first_mismatch=first)
            for mode,data in [('static',static),('compact',compact)]:
                metas=[r['outputs'][index]['meta_info'] for r in data['results']]
                row[mode]=dict(E2E_ms=statistics.median(m['e2e_latency']*1000 for m in metas),
                    TTFT_ms=statistics.median(m['first_token_latency']*1000 for m in metas),
                    verify_steps=statistics.median(m['spec_verify_ct'] for m in metas),
                    request_mean_cycle_ms=statistics.median((m['e2e_latency']-m['first_token_latency'])*1000/m['spec_verify_ct'] for m in metas))
            rows.append(row)
        slowest={mode:max(rows,key=lambda r:r[mode]['E2E_ms'])['prompt'] for mode in ['static','compact']}
        summaries.append(dict(dataset=dataset,bs=bs,requests=rows,slowest_prompt=slowest,
            wave_median_ms={mode:statistics.median(r['seconds']*1000 for r in data['results']) for mode,data in [('static',static),('compact',compact)]},
            source_cases={mode:entry[0] for mode,entry in modes.items()}))
    assert summaries
    out=ROOT/'reports/2026-11-04_parallel_sd';stem='dspark_tail_'+args.run_id
    (out/(stem+'.json')).write_text(json.dumps(dict(run_id=args.run_id,groups=summaries,
        scope='Unprofiled fixed-wave request timing medians. Different token outputs are retained; elapsed/round changes cannot be attributed exclusively to scheduling policy when output paths diverge.'),indent=2))
    plt.rcParams.update({'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,len(summaries),figsize=(5*len(summaries),4.5),squeeze=False)
    for ax,group in zip(axes[0],summaries):
        values=[]
        for row in group['requests']:
            x,y=row['static']['E2E_ms'],row['compact']['E2E_ms'];values.extend([x,y])
            ax.scatter([x],[y],color='#147d92' if row['outputs_equal'] else '#de8a28',s=35)
            ax.annotate(str(row['prompt']),(x,y),xytext=(4,3),textcoords='offset points',fontsize=8)
        lo,hi=min(values)*.95,max(values)*1.03;ax.plot([lo,hi],[lo,hi],':',color='#687787')
        ax.set_xlim(lo,hi);ax.set_ylim(lo,hi);ax.set_aspect('equal',adjustable='box');ax.set_xlabel('Static request E2E (ms)');ax.set_ylabel('Compact request E2E (ms)');ax.set_title(f"{group['dataset']} / BS{group['bs']}");ax.grid(alpha=.15)
    fig.suptitle('Below diagonal: compact faster. Teal: same output; orange: token divergence.')
    fig.tight_layout();fig.savefig(out/(stem+'.svg'),bbox_inches='tight');fig.savefig(out/(stem+'.png'),bbox_inches='tight',dpi=160);plt.close(fig)
    print([(g['dataset'],g['slowest_prompt'],g['wave_median_ms']) for g in summaries])


if __name__=='__main__':main()
