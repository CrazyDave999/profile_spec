"""Measure fixed-wave critical requests without confusing mean tau with tails."""
import _bootstrap
import csv
import hashlib
import json
from pathlib import Path

ROOT=_bootstrap.ROOT
LOGS=ROOT/'logs/moe_sd_20261106'
OUT=ROOT/'reports/2026-11-06_moe_parallel_sd'


def analyze(sequence):
    rows=[];sources={}
    for manifest_path in sorted(LOGS.glob(sequence+'_*.campaign.json')):
        campaign=json.loads(manifest_path.read_text());stage=campaign['stage']
        if stage not in ('full','width','dense_control'):continue
        for item in campaign['commands']:
            name=item['name'];job=LOGS/(name+'.job.json');path=LOGS/(name+'.json')
            if not job.exists() or json.loads(job.read_text())['state']!='success':continue
            data=json.loads(path.read_text());a=data['arguments'];assert not a['profile'] and not a.get('routing_record')
            sources[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
            for result in data['results']:
                outputs=result['outputs'];bs=a['bs'];waves=[outputs[i:i+bs] for i in range(0,len(outputs),bs)]
                assert len(waves)==len(result['wave_seconds'])
                for wave_idx,(wave,seconds) in enumerate(zip(waves,result['wave_seconds'])):
                    critical=max(range(len(wave)),key=lambda i:wave[i]['meta_info']['request_finished_ts'])
                    m=wave[critical]['meta_info'];steps=m.get('spec_verify_ct',0) if a['method']!='ar' else m['completion_tokens']-1
                    assert steps>0
                    e2e=m['e2e_latency'];ttft=m['first_token_latency']
                    rows.append({'case':name,'stage':stage,'target':a['target'],'dataset':a['dataset'],'method':a['method'],'configured_bs':bs,'width':a['block'] if a['method']!='ar' else 1,
                        'repeat':result['repeat'],'wave':wave_idx,'requests':len(wave),'wave_wall_ms':seconds*1000,
                        'critical_prompt_index':wave_idx*bs+critical,'critical_E2E_ms':e2e*1000,'critical_TTFT_ms':ttft*1000,'critical_decode_ms':(e2e-ttft)*1000,
                        'critical_decode_steps':steps,'critical_decode_tokens_per_step':(m['completion_tokens']-1)/steps,'critical_mean_cycle_ms':(e2e-ttft)*1000/steps,
                        'wave_min_request_steps':min(o['meta_info'].get('spec_verify_ct',0) if a['method']!='ar' else o['meta_info']['completion_tokens']-1 for o in wave),
                        'wave_max_request_steps':max(o['meta_info'].get('spec_verify_ct',0) if a['method']!='ar' else o['meta_info']['completion_tokens']-1 for o in wave),
                        'wave_tokens':sum(o['meta_info']['completion_tokens'] for o in wave)})
    OUT.mkdir(parents=True,exist_ok=True)
    if rows:
        target=OUT/'wave_tails.csv';temp=target.with_suffix('.csv.tmp')
        with temp.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
        temp.replace(target)
    evidence={'sequence':sequence,'wave_rows':len(rows),'source_sha256':sources,
        'definition':'Critical request is the last server-finished request in each explicitly timed offline wave. Its E2E-minus-TTFT divided by its own verify count is a request-mean cycle, not an isolated GPU iteration.',
        'limits':['The median of request-mean cycles and global tau cannot be multiplied to exactly reproduce fixed-wave throughput.','The critical request can differ by method, batch size, width and numerical output path.','Output lengths are fixed; different acceptance lengths still cause request-finish tails.','Prefill and client/API time remain in wall throughput.']}
    p=OUT/'wave_tails_evidence.json';temp=p.with_suffix('.json.tmp');temp.write_text(json.dumps(evidence,indent=2));temp.replace(p)
    return rows


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--sequence',required=True);a=parser.parse_args();print('wave rows',len(analyze(a.sequence)))
