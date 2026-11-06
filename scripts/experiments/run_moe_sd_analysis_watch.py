"""CPU-only incremental analysis of a specific live experiment controller."""
import _bootstrap
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import psutil
from parallel_sd_worker import ROOT, save
from run_moe_sd_campaign import LOGS


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--sequence',required=True)
    parser.add_argument('--attempt',default='')
    args=parser.parse_args()
    assert args.sequence.replace('_','').replace('-','').isalnum()
    lock=(LOGS/(args.sequence+'.analysis.lock')).open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    admission_path=LOGS/(args.sequence+'.admission.json')
    admission=json.loads(admission_path.read_text())
    controller_pid,controller_start=admission['pid'],admission['process_start_time']
    suffix=('_'+args.attempt) if args.attempt else ''
    output=LOGS/(args.sequence+'.analysis_watch'+suffix+'.json')
    assert not output.exists(),'Use a fresh analysis attempt name.'
    state={'status':'watching','pid':os.getpid(),'process_start_time':psutil.Process().create_time(),
        'sequence':args.sequence,'controller_pid':controller_pid,'controller_start_time':controller_start,
        'started_epoch':time.time(),'updates':[]}
    save(output,state)
    previous=None
    while True:
        admission=json.loads(admission_path.read_text())
        try:
            controller=psutil.Process(controller_pid)
            live=controller.create_time()==controller_start and controller.status()!=psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            live=False
        successful=[]
        for path in sorted(LOGS.glob(args.sequence+'_*.job.json')):
            if json.loads(path.read_text())['state']=='success':successful.append(path.stem)
        source_names=['scripts/analysis/analyze_moe_sd.py','scripts/analysis/analyze_moe_traces.py','scripts/analysis/analyze_moe_components.py','scripts/analysis/analyze_moe_routing.py','scripts/reports/build_moe_sd_report.py']
        source_hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in source_names}
        fingerprint=(tuple(successful),admission['status'],tuple(source_hashes.items()))
        if fingerprint!=previous:
            commands=[['.venv-sglang/bin/python','scripts/analysis/analyze_moe_sd.py','--sequence',args.sequence]]
            if any('_traces_' in name for name in successful):
                commands.append(['.venv-sglang/bin/python','scripts/analysis/analyze_moe_traces.py','--sequence',args.sequence])
                commands.append(['.venv-sglang/bin/python','scripts/analysis/analyze_moe_components.py','--sequence',args.sequence])
            if any('_routing_' in name for name in successful):
                commands.append(['.venv-sglang/bin/python','scripts/analysis/analyze_moe_routing.py','--sequence',args.sequence])
            commands.append(['.venv/bin/python','scripts/reports/build_moe_sd_report.py'])
            update={'epoch':time.time(),'successful_cells':len(successful),'controller_status':admission['status'],'source_sha256':source_hashes,'commands':[]}
            for command in commands:
                result=subprocess.run(command,cwd=ROOT)
                update['commands'].append({'argv':command,'exit_code':result.returncode})
                if result.returncode:
                    state.update(status='analysis_failed',failed_command=command,ended_epoch=time.time());state['updates'].append(update);save(output,state)
                    raise SystemExit(result.returncode)
            state['updates'].append(update);previous=fingerprint
        state.update(controller_live=live,controller_status=admission['status'],last_checked_epoch=time.time(),successful_cells=len(successful))
        save(output,state)
        if not live or admission['status'] in ('success','smoke_stopped','sequence_stopped'):
            state.update(status='success' if admission['status']=='success' else 'controller_stopped',ended_epoch=time.time());save(output,state)
            break
        time.sleep(10)


if __name__=='__main__':main()
