"""Wait for a verified campaign, then run independent stages serially."""
import _bootstrap
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time

import psutil
from parallel_sd_worker import ROOT, save
from run_moe_sd_campaign import LOGS


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--after",required=True)
    parser.add_argument("--run-id",required=True)
    parser.add_argument("--stages",nargs="+",default=["traces","full","width","routing","dense_control"])
    parser.add_argument("--attention",default="fa4")
    parser.add_argument("--moe-backend",default="flashinfer_mxfp4")
    args=parser.parse_args()
    assert args.run_id.replace("_", "").replace("-", "").isalnum()
    lock=(LOGS/(args.run_id+".sequence.lock")).open("w")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    path=LOGS/(args.run_id+".sequence.json")
    if path.exists():
        raise RuntimeError("Inspect the existing sequence before starting another attempt.")
    dependency=LOGS/(args.after+".campaign.json")
    prior=json.loads(dependency.read_text())
    pid,created=prior["supervisor_pid"],prior["supervisor_start_time"]
    state={"status":"waiting_for_live_dependency","pid":os.getpid(),
        "process_start_time":psutil.Process().create_time(),"dependency":str(dependency),
        "dependency_pid":pid,"dependency_start_time":created,"stages":[],"started_epoch":time.time()}
    save(path,state)
    while True:
        current=json.loads(dependency.read_text())
        try:
            process=psutil.Process(pid)
            live=process.create_time()==created and process.status()!=psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            live=False
        state.update(dependency_live=live,dependency_status=current["status"],last_checked_epoch=time.time());save(path,state)
        if current["status"] not in ("running","success"):
            state.update(status="dependency_failed",ended_epoch=time.time());save(path,state)
            raise RuntimeError("Dependency failed: "+current["status"])
        if not live:
            assert current["status"]=="success","Dependency disappeared without success."
            break
        time.sleep(2)
    for stage in args.stages:
        run=args.run_id+"_"+stage
        command=[sys.executable,str(ROOT/'scripts/experiments/run_moe_sd_campaign.py'),"--stage",stage,"--run-id",run,
            "--attention",args.attention,"--moe-backend",args.moe_backend]
        item={"stage":stage,"run_id":run,"argv":command,"started_epoch":time.time()}
        state["stages"].append(item);state.update(status="running",current_stage=stage);save(path,state)
        print("START_STAGE",run,flush=True)
        code=subprocess.call(command,cwd=ROOT)
        item.update(exit_code=code,ended_epoch=time.time());save(path,state)
        if code:
            state.update(status="stopped_after_stage_failure",ended_epoch=time.time());save(path,state)
            raise SystemExit(code)
    state.update(status="success",ended_epoch=time.time());save(path,state)
    print("SEQUENCE_SUCCESS",args.run_id,flush=True)


if __name__=="__main__":
    main()
