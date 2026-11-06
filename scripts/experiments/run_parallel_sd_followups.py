"""Wait for the live full campaign, then serialize admitted profile/hybrid cells.

The dependency is checked using both its recorded PID creation time and terminal
campaign state. A timeout never means that the dependency has stopped.
"""
import _bootstrap
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

from parallel_sd_worker import ROOT, save


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--after', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--dependency-type', choices=['campaign','sequence'],default='campaign')
    parser.add_argument('--stages',nargs='+',default=['traces','hybrid'])
    parser.add_argument('--sps-table',type=Path)
    parser.add_argument('--adaptive-dataset',choices=['gsm8k','mt_bench','both'],default='gsm8k')
    parser.add_argument('--adaptive-bs',type=int,choices=[1,4,16])
    args=parser.parse_args()
    directory=ROOT/'logs/parallel_sd_20261104'
    assert args.run_id.replace('_','').replace('-','').isalnum()
    lock=(directory/(args.run_id+'.sequence.lock')).open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    out=directory/(args.run_id+'.sequence.json')
    if out.exists():raise RuntimeError('Sequence already exists; inspect before launching a new attempt.')
    dependency=directory/(args.after+'.'+args.dependency_type+'.json')
    before=json.loads(dependency.read_text())
    dependency_pid=before.get('supervisor_pid',before.get('pid'))
    dependency_start=before.get('supervisor_start_time',before.get('process_start_time'))
    state=dict(status='waiting_for_live_dependency', pid=os.getpid(),
        process_start_time=psutil.Process().create_time(), dependency=str(dependency),
        dependency_pid=dependency_pid,dependency_start_time=dependency_start,
        worker_sha256=hashlib.sha256((ROOT/'scripts/experiments/parallel_sd_worker.py').read_bytes()).hexdigest(),
        started_epoch=time.time(),stages=[])
    save(out,state)
    while True:
        current=json.loads(dependency.read_text())
        try:
            process=psutil.Process(dependency_pid)
            live=process.create_time()==dependency_start and process.status()!=psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:live=False
        state.update(dependency_live=live,last_checked_epoch=time.time(),dependency_status=current['status'])
        save(out,state)
        if current['status'] not in ['running','success','waiting_for_live_dependency']:
            state.update(status='dependency_failed',ended_epoch=time.time());save(out,state)
            raise RuntimeError('Dependency stopped unsuccessfully: '+current['status'])
        if not live:
            if current['status']!='success':
                state.update(status='dependency_gone_without_success',ended_epoch=time.time());save(out,state)
                raise RuntimeError('Dependency process is gone without terminal success.')
            break
        time.sleep(2)
    for stage in args.stages:
        name=args.run_id+'_'+stage
        command=[sys.executable,str(ROOT/'scripts/experiments/run_parallel_sd_campaign.py'),'--stage',stage,'--run-id',name]
        if stage=='adaptive':
            calibration=args.sps_table or directory/(args.run_id+'_sps.sps.json')
            assert calibration.is_file(),'SPS calibration must finish before adaptive evaluation.'
            command+=['--sps-table',str(calibration),'--adaptive-dataset',args.adaptive_dataset]
            if args.adaptive_bs is not None:command+=['--adaptive-bs',str(args.adaptive_bs)]
        entry=dict(stage=stage,run_id=name,argv=command,started_epoch=time.time())
        state['stages'].append(entry);state.update(status='running',current_stage=stage);save(out,state)
        print('START_FOLLOWUP',name,flush=True)
        code=subprocess.call(command,cwd=ROOT)
        entry.update(exit_code=code,ended_epoch=time.time());save(out,state)
        if code:
            state.update(status='stopped_after_stage_failure',ended_epoch=time.time());save(out,state)
            raise SystemExit(code)
    state.update(status='success',ended_epoch=time.time());save(out,state)
    print('FOLLOWUP_SEQUENCE_SUCCESS',args.run_id,flush=True)


if __name__=='__main__':main()
