"""Resumable overnight GPU-idle-only experiment supervisor.

All workloads are child process groups. A foreign compute process on either GPU
causes our current case to terminate and release its GPU memory, then retry after
the workstation is idle. No other process is stopped or modified.
"""
import _bootstrap
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from sweep_worker import save

ROOT = _bootstrap.ROOT
DATASETS = ['gsm8k','math500','aime25','humaneval','mbpp','livecodebench','mt_bench']


def gpu_processes():
    r = subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,process_name','--format=csv,noheader,nounits'],
                       text=True,capture_output=True,timeout=10)
    if r.returncode:
        raise RuntimeError('Unable to confirm GPU idleness')
    rows = []
    for line in r.stdout.splitlines():
        fields = [f.strip() for f in line.split(',',2)]
        if len(fields)==3 and fields[0].isdigit():
            rows.append({'pid':int(fields[0]),'gpu':fields[1],'name':fields[2]})
    return rows


def descendants(root):
    parents = {}
    for path in Path('/proc').iterdir():
        if path.name.isdigit():
            try:
                stat = (path/'stat').read_text().split(') ',1)[1].split()
                parents[int(path.name)] = int(stat[1])
            except (OSError,ValueError,IndexError):
                pass
    found = {root}
    while True:
        new = {pid for pid,ppid in parents.items() if ppid in found} | found
        if new == found:
            return found
        found = new


def plan():
    cases = []
    def add(engine,dataset,bs,trace,arm,tokens,samples,**extra):
        case = dict(engine=engine,dataset=dataset,bs=bs,trace=trace,arm=arm,tokens=tokens,samples=samples,**extra)
        case['name'] = f"{engine}_{dataset}_bs{bs}_{'trace' if trace else 'bench'}_{arm}_t{tokens}" + (f"_budget{extra['budget']}" if extra else '')
        cases.append(case)
    # Establish timeline coverage across all paper datasets first.
    for dataset in DATASETS:
        add('optimized',dataset,1,True,'both',256,2)
    for bs in [1,2,4,8,16]:
        for dataset in DATASETS:
            add('batch_reference',dataset,bs,True,'both',128,min(bs*2,32))
    # Separate processes give the AR arm its genuine head-free memory footprint.
    for engine,bs_list in [('optimized',[1]),('batch_reference',[1,2,4,8,16])]:
        for bs in bs_list:
            for dataset in DATASETS:
                for arm in ['AR','JetSpec']:
                    add(engine,dataset,bs,False,arm,512,32)
    # Additional tree-budget/context-growth probes, if time remains after the main sweep.
    for budget in [64,256]:
        for dataset in DATASETS:
            add('optimized',dataset,1,True,'both',256,2,budget=budget)
            add('optimized',dataset,1,False,'JetSpec',512,32,budget=budget)
    for dataset in DATASETS:
        for arm in ['AR','JetSpec']:
            add('optimized',dataset,1,False,arm,1024,16)
    return cases


def terminate(process):
    owned = descendants(process.pid)
    try:
        os.killpg(process.pid,signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        try: os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError: pass
        process.wait(timeout=15)
    # Release CUDA workers that a profiler might have placed in another process group.
    for pid in owned - {process.pid}:
        try:
            if (Path('/proc')/str(pid)).exists(): os.kill(pid,signal.SIGTERM)
        except (ProcessLookupError,PermissionError): pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--deadline',default='2026-11-03T01:35:00+00:00') # 09:35 China; leave report time.
    p.add_argument('--only',default=None)
    p.add_argument('--campaign',default='overnight_20261103')
    p.add_argument('--metrics-only',action='store_true')
    args = p.parse_args()
    dest = ROOT/'logs'/args.campaign
    dest.mkdir(parents=True,exist_ok=True)
    lock = open(dest/'supervisor.lock','w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    deadline = datetime.fromisoformat(args.deadline).timestamp()
    jobs = plan()
    if args.metrics_only:
        jobs = [{'engine':'optimized','dataset':dataset,'bs':1,'trace':True,'arm':'both',
                 'tokens':256,'samples':2,'gpu_metrics':True,
                 'name':f'optimized_{dataset}_bs1_gpu_metrics_t256'} for dataset in DATASETS]
    if args.only: jobs = [c for c in jobs if args.only in c['name']]
    save(dest/'plan.json',jobs)
    status = {'pid':os.getpid(),'started_utc':datetime.now(timezone.utc).isoformat(),
              'deadline_utc':args.deadline,'state':'starting','completed':[],'preemptions':[],
              'total_cases':len(jobs),'current':None}
    def record(): save(dest/'status.json',status)
    for index, case in enumerate(jobs):
        result = dest/(case['name']+'.json')
        if result.exists():
            prior = json.loads(result.read_text())
            if prior.get('status') in ['success','out_of_memory']:
                status['completed'].append({'name':case['name'],'status':prior['status'],'resumed':True})
                continue
        while time.time()<deadline:
            try: foreign=gpu_processes()
            except Exception as e:
                status.update(state='waiting_for_GPU_status',error=str(e)); record(); time.sleep(10); continue
            if foreign:
                status.update(state='waiting_for_idle_GPU',foreign_processes=foreign,current=case['name']); record()
                time.sleep(10); continue
            if case['trace']:
                report = ROOT/'profiles'/args.campaign/case['name']
                report.parent.mkdir(parents=True,exist_ok=True)
                command = ['nsys','profile','--trace=cuda,nvtx,osrt','--sample=none','--cpuctxsw=none',
                    '--cuda-graph-trace=node','--cuda-memory-usage=true',
                    '--capture-range=cudaProfilerApi','--capture-range-end=stop','--force-overwrite=true',
                    '--output='+str(report),sys.executable,'-u','scripts/experiments/sweep_worker.py','--trace']
                if case.get('gpu_metrics'):
                    command[2:2] = ['--gpu-metrics-devices=cuda-visible','--gpu-metrics-frequency=1000',
                                    '--gpu-metrics-set=gb20x']
            else:
                command = [sys.executable,'-u','scripts/experiments/sweep_worker.py']
            for key in ['dataset','engine','bs','tokens','samples','arm','budget']:
                if key in case: command += ['--'+key.replace('_','-'),str(case[key])]
            command += ['--output',str(result)]
            logfile = open(dest/(case['name']+'.log'),'a')
            process = subprocess.Popen(command,cwd=ROOT,stdout=logfile,stderr=subprocess.STDOUT,start_new_session=True)
            status.update(state='running',current=case['name'],worker_pid=process.pid,case_index=index)
            status.pop('foreign_processes',None); record()
            interrupted = False
            while process.poll() is None:
                time.sleep(2)
                try: others=[r for r in gpu_processes() if r['pid'] not in descendants(os.getpid())]
                except Exception: others=[{'name':'GPU-status-unavailable'}]
                if others or time.time()>deadline:
                    terminate(process)
                    interrupted=True
                    status['preemptions'].append({'case':case['name'],'at':time.time(),'foreign':others})
                    status.update(state='yielding_GPU'); record()
                    if result.exists(): result.unlink()
                    break
            logfile.close()
            if interrupted:
                continue
            if not result.exists():
                save(result,{'status':'process_error','returncode':process.returncode,'case':case})
            outcome=json.loads(result.read_text())
            if case['trace'] and outcome.get('status')=='success':
                sqlite = str(report)+'.sqlite'
                with open(dest/(case['name']+'.export.log'),'w') as log:
                    exported=subprocess.run(['nsys','export','--type=sqlite','--force-overwrite=true',
                        '--output='+sqlite,str(report)+'.nsys-rep'],stdout=log,stderr=subprocess.STDOUT)
                outcome['nsys_report']=str(report)+'.nsys-rep'
                outcome['nsys_sqlite']=sqlite if exported.returncode==0 else None
                outcome['nsys_gpu_metrics_enabled']=bool(case.get('gpu_metrics'))
                save(result,outcome)
            status['completed'].append({'name':case['name'],'status':outcome.get('status')})
            record()
            break
        else:
            status.update(state='deadline_reached'); record(); break
    status.update(state='experiments_finished',finished_utc=datetime.now(timezone.utc).isoformat(),current=None)
    record()
    # The agent also inspects all results and the report; this process only creates artifacts.
    if (ROOT/'scripts/reports/build_overnight_report.py').exists():
        with open(dest/'report_build.log','w') as log:
            r=subprocess.run([sys.executable,'scripts/reports/build_overnight_report.py'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        status['report_build_exit']=r.returncode
        record()


if __name__=='__main__':
    main()
