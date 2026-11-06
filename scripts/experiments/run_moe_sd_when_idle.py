"""Admit a fresh smoke/sequence only after the authorized GPU stays idle."""
import _bootstrap
import argparse
import fcntl
import os
import subprocess
import sys
import time

import psutil
import pynvml
from parallel_sd_worker import ROOT, GPU, save
from run_moe_sd_campaign import LOGS


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--smoke",required=True)
    parser.add_argument("--sequence",required=True)
    args=parser.parse_args()
    lock=(LOGS/"admission.lock").open("w")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    path=LOGS/(args.sequence+".admission.json")
    if path.exists():raise RuntimeError("Admission attempt already exists; inspect its recorded PID.")
    for name in [args.smoke+".campaign.json",args.sequence+".sequence.json"]:
        assert not (LOGS/name).exists(),"Choose fresh attempt names."
    pynvml.nvmlInit();handle=pynvml.nvmlDeviceGetHandleByUUID(GPU)
    state={"status":"waiting_for_idle_GPU","pid":os.getpid(),"process_start_time":psutil.Process().create_time(),
        "GPU_UUID":GPU,"started_epoch":time.time(),"smoke_run_id":args.smoke,"sequence_run_id":args.sequence}
    idle_since=None
    while True:
        now=time.time();foreign=[p.pid for p in pynvml.nvmlDeviceGetComputeRunningProcesses(handle)]
        idle_since=None if foreign else (idle_since or now)
        state.update(foreign_GPU_pids=foreign,last_checked_epoch=now,idle_seconds=now-idle_since if idle_since else 0)
        save(path,state)
        if idle_since and now-idle_since>=15:break
        time.sleep(2)
    smoke=[sys.executable,str(ROOT/'scripts/experiments/run_moe_sd_campaign.py'),"--stage","smoke","--run-id",args.smoke]
    state.update(status="running_smoke",smoke_argv=smoke);save(path,state)
    code=subprocess.call(smoke,cwd=ROOT)
    if code:
        state.update(status="smoke_stopped",exit_code=code,ended_epoch=time.time());save(path,state)
        raise SystemExit(code)
    sequence=[sys.executable,str(ROOT/'scripts/experiments/run_moe_sd_sequence.py'),"--after",args.smoke,"--run-id",args.sequence]
    state.update(status="running_sequence",sequence_argv=sequence);save(path,state)
    code=subprocess.call(sequence,cwd=ROOT)
    state.update(status="success" if code==0 else "sequence_stopped",exit_code=code,ended_epoch=time.time());save(path,state)
    raise SystemExit(code)


if __name__=="__main__":main()
