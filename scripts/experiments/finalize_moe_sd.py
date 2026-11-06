"""CPU-only final analysis after the existing numeric and analysis processes exit."""
import _bootstrap
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import psutil
from parallel_sd_worker import ROOT, save

LOGS=ROOT/'logs/moe_sd_20261106'


def main():
    p=argparse.ArgumentParser();p.add_argument('--sequence',default='native03');p.add_argument('--numerics',default='numerics03');p.add_argument('--watcher',default='native03.analysis_watch_a02.json');p.add_argument('--attempt',required=True);a=p.parse_args()
    path=LOGS/(a.sequence+'.finalization_'+a.attempt+'.json');assert not path.exists(),'Use a fresh attempt.'
    dependencies=[]
    for name in (a.numerics+'.numerics.json',a.watcher):
        f=LOGS/name;d=json.loads(f.read_text());dependencies.append({'file':str(f),'pid':d['pid'],'process_start_time':d['process_start_time']})
    state={'status':'waiting','pid':os.getpid(),'process_start_time':psutil.Process().create_time(),'dependencies':dependencies,'started_epoch':time.time(),'commands':[]};save(path,state)
    while True:
        done=True
        for dep in dependencies:
            d=json.loads(Path(dep['file']).read_text())
            try:
                q=psutil.Process(dep['pid']);live=q.create_time()==dep['process_start_time'] and q.status()!=psutil.STATUS_ZOMBIE
            except psutil.NoSuchProcess:live=False
            dep.update(live=live,status=d['status'])
            if not live and d['status']!='success':
                state.update(status='dependency_stopped',ended_epoch=time.time());save(path,state);raise SystemExit(1)
            done&=not live and d['status']=='success'
        state.update(last_checked_epoch=time.time());save(path,state)
        if done:break
        time.sleep(10)
    commands=[['.venv-sglang/bin/python','scripts/analysis/analyze_moe_sd.py','--sequence',a.sequence],
        ['.venv-sglang/bin/python','scripts/analysis/analyze_moe_traces.py','--sequence',a.sequence],
        ['.venv-sglang/bin/python','scripts/analysis/analyze_moe_components.py','--sequence',a.sequence],
        ['.venv-sglang/bin/python','scripts/analysis/analyze_moe_routing.py','--sequence',a.sequence],
        ['.venv-sglang/bin/python','scripts/analysis/analyze_moe_numerics.py','--run-id',a.numerics,'--headline-sequence',a.sequence],
        ['.venv/bin/python','scripts/analysis/package_moe_sd_traces.py','--sequence',a.sequence],
        ['.venv/bin/python','scripts/reports/build_moe_sd_report.py'],
        ['.venv/bin/python','scripts/analysis/audit_moe_sd_campaign.py','--sequence',a.sequence,'--numerics',a.numerics]]
    for index,command in enumerate(commands):
        log=LOGS/(a.sequence+'.finalization_'+a.attempt+f'.{index:02d}.log')
        item={'argv':command,'log':str(log.relative_to(ROOT)),'started_epoch':time.time()};state['commands'].append(item);state.update(status='running');save(path,state)
        with log.open('w') as f:code=subprocess.call(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        item.update(exit_code=code,ended_epoch=time.time());save(path,state)
        if code:
            state.update(status='analysis_failed',ended_epoch=time.time());save(path,state);raise SystemExit(code)
    state.update(status='success',ended_epoch=time.time());save(path,state)


if __name__=='__main__':main()
