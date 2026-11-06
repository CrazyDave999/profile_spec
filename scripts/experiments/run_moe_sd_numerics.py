"""Run a small, separate numerical diagnostic after the admitted main queue."""
import _bootstrap
import argparse
import json
import os
import subprocess
import sys
import time

import psutil
from parallel_sd_worker import ROOT, save
from run_moe_sd_campaign import LOGS, PROFILES


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--after',required=True);parser.add_argument('--run-id',required=True);args=parser.parse_args()
    assert args.run_id.replace('_','').replace('-','').isalnum()
    path=LOGS/(args.run_id+'.numerics.json');assert not path.exists(),'Choose a fresh diagnostic attempt.'
    dep_path=LOGS/(args.after+'.admission.json');dep=json.loads(dep_path.read_text());pid,created=dep['pid'],dep['process_start_time']
    state={'status':'waiting_for_main_queue','pid':os.getpid(),'process_start_time':psutil.Process().create_time(),
        'dependency':str(dep_path),'dependency_pid':pid,'dependency_start_time':created,'started_epoch':time.time(),'commands':[]}
    save(path,state)
    while True:
        current=json.loads(dep_path.read_text())
        try:
            process=psutil.Process(pid);live=process.create_time()==created and process.status()!=psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:live=False
        state.update(dependency_live=live,dependency_status=current['status'],last_checked_epoch=time.time());save(path,state)
        if not live:
            if current['status']!='success':
                state.update(status='dependency_stopped',ended_epoch=time.time());save(path,state);return
            break
        time.sleep(3)
    for i,(fp32,method) in enumerate([(False,'ar'),(False,'dflash'),(True,'ar'),(True,'dflash')]):
        name=f'{args.run_id}_{i:02d}_{method}_headfp32_{int(fp32)}'
        cmd=[sys.executable,str(ROOT/'scripts/experiments/run_parallel_sd_cell.py'),'--name',name,'--directory',str(LOGS),'--profile-directory',str(PROFILES),
            '--','--target','gpt-oss-20b','--method',method,'--dataset','gsm8k','--bs','1','--samples','2','--tokens','128','--repeats','1',
            '--attention','fa4','--moe-runner-backend','flashinfer_mxfp4','--logprobs']
        if method=='dflash':cmd+=['--block','8']
        if fp32:cmd+=['--fp32-lm-head']
        state['commands'].append({'name':name,'argv':cmd});state.update(status='running',current_cell=name);save(path,state)
        code=subprocess.call(cmd,cwd=ROOT)
        if code:
            state.update(status='stopped_after_cell_failure',exit_code=code,ended_epoch=time.time());save(path,state);raise SystemExit(code)
    state.update(status='success',ended_epoch=time.time());save(path,state)


if __name__=='__main__':main()
