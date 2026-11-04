"""Check runtime artifacts, not just supervisor labels, before publishing."""
import argparse
import hashlib
import json
from pathlib import Path

import psutil

from parallel_sd_worker import GPU, ROOT


def inspect(run_id):
    directory=ROOT/'logs/parallel_sd_20261104'
    path=directory/(run_id+'.campaign.json');campaign=json.loads(path.read_text())
    incomplete=[];failures=[];checked=[];cohorts={}
    snapshot=directory/'worker_sources'/(campaign['worker_sha256']+'.py')
    if not snapshot.exists():incomplete.append({'missing':'worker source snapshot','SHA256':campaign['worker_sha256']})
    elif hashlib.sha256(snapshot.read_bytes()).hexdigest()!=campaign['worker_sha256']:
        failures.append({'error':'Worker snapshot SHA256 mismatch.'})
    for cell in campaign['commands']:
        job=directory/(cell['name']+'.job.json');result=directory/(cell['name']+'.json')
        if not job.exists():incomplete.append(cell['name']);continue
        state=json.loads(job.read_text())
        if state['state'] in ['starting','running','yielding_to_foreign_GPU']:
            # A stale running label is not proof of a live job.
            try:
                process=psutil.Process(state['pid'])
                live=process.create_time()==state['process_start_time'] and process.status()!=psutil.STATUS_ZOMBIE
            except psutil.NoSuchProcess:live=False
            incomplete.append({'case':cell['name'],'state':state['state'],'live':live});continue
        if state['state']!='success':failures.append({'case':cell['name'],'state':state['state']});continue
        try:
            assert state['exit_code']==0 and result.is_file()
            data=json.loads(result.read_text());assert data['status']=='success'
            args=data['arguments'];engine=data['engine_resolved_after'];results=data['results']
            assert data['GPU_UUID']==GPU
            assert data['worker_source_sha256']==campaign['worker_sha256']
            assert len(results)==args['repeats']
            assert engine['dtype']=='bfloat16' and engine['tp_size']==1
            assert engine['cuda_graph_backend_decode']=='full' and engine['cuda_graph_backend_prefill']=='disabled'
            assert engine['disable_overlap_schedule'] is False and engine['disable_radix_cache'] is True
            assert engine['max_total_tokens']==32768 and engine['max_running_requests']==args['bs']
            assert args['profile']==(campaign['stage']=='traces')
            assert len(data['input_lengths'])==args['samples'] and max(data['input_lengths'])+args['tokens']<4096
            assert data['physical_device_peak_used_bytes']>0
            for repeat in results:
                outputs=repeat['outputs'];assert len(outputs)==args['samples']
                assert all(len(o['output_ids'])==o['meta_info']['completion_tokens'] for o in outputs)
                assert sum(o['meta_info']['completion_tokens'] for o in outputs)==repeat['output_tokens']
                assert abs(repeat['output_tokens']/repeat['seconds']-repeat['tokens_per_second'])<1e-8
                if not args['eos']:assert all(len(o['output_ids'])==args['tokens'] for o in outputs)
                assert all(o['meta_info'].get('cached_tokens',0)==0 for o in outputs)
                assert all(o['meta_info'].get('num_retractions',0)==0 for o in outputs)
                if args['method']!='ar':assert repeat['sum_request_verify_steps']>0
            signature=(args['target'],args['dataset'],args['samples'],args['tokens'],args['extra_context'])
            frozen=(data['selected_prompt_sha256'],data['package_versions'],data['toolkit_header_sha256'],data['sglang_revision'])
            if signature in cohorts:assert cohorts[signature]==frozen,'Mismatched prompt IDs/runtime in comparison cohort.'
            else:cohorts[signature]=frozen
            log=(directory/(cell['name']+'.log')).read_text(errors='replace')
            marker='Capture target decode CUDA graph end' if args['method']=='ar' else 'Capture target verify CUDA graph end'
            assert marker in log,marker
            if args['method']!='ar':assert engine['speculative_num_draft_tokens']>=2
            if args['block'] is not None:assert engine['speculative_num_draft_tokens']==args['block']
            if campaign['stage']=='traces':
                analysis=directory/(cell['name']+'.analysis.json')
                if not analysis.exists():incomplete.append({'case':cell['name'],'missing':'trace analysis'})
                else:
                    trace=json.loads(analysis.read_text());assert trace['cuda_graph_launches']>0
                    assert trace['cuda_graph_captures']==0 and trace['correlation_coverage']>.99
                    assert trace['gpu_counters'] and trace['gpu_counters']['sampling_hz_estimate']>900
                    assert any(s['batch_size']==args['bs'] for s in trace['by_NVTX_batch_size'])
            checked.append({'case':cell['name'],'SHA256':hashlib.sha256(result.read_bytes()).hexdigest()})
        except (AssertionError,KeyError,TypeError,ValueError) as exc:
            failures.append({'case':cell['name'],'error':str(exc) or type(exc).__name__})
    return dict(run_id=run_id,supervisor_status=campaign['status'],expected=len(campaign['commands']),
        checked=len(checked),incomplete=incomplete,failures=failures,source_artifacts=checked,
        complete=campaign['status']=='success' and not incomplete and not failures and len(checked)==len(campaign['commands']))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('run_ids',nargs='+');parser.add_argument('--require-complete',action='store_true');args=parser.parse_args()
    audits=[inspect(run_id) for run_id in args.run_ids]
    output=ROOT/'logs/parallel_sd_20261104/campaign_artifact_audit.json';output.write_text(json.dumps(audits,indent=2))
    for audit in audits:print({k:v for k,v in audit.items() if k!='source_artifacts'})
    if any(a['failures'] for a in audits) or (args.require_complete and not all(a['complete'] for a in audits)):raise SystemExit(1)


if __name__=='__main__':main()
