"""Serialized, resumable GPT-OSS MXFP4 experiments, with untouched old cohorts."""
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

LOGS = ROOT/"logs/moe_sd_20261106"
PROFILES = ROOT/"profiles/moe_sd_20261106"


def experiments(stage):
    if stage == "smoke":
        return [(m,"gsm8k",1,2,32,1,8,False,False) for m in ("ar","dflash")]
    if stage == "full":
        return [(m,d,b,16,128,3,8,False,False) for d in ("gsm8k","humaneval","mt_bench") for b in (1,4,16) for m in ("ar","dflash","ar")]
    if stage == "width":
        return [(m,d,b,16,128,3,w,False,False) for d in ("gsm8k","mt_bench") for b in (1,4,16) for m,w in (("ar",8),("dflash",2),("dflash",4),("dflash",8),("ar",8))]
    if stage == "traces":
        return [(m,"gsm8k",b,16,64,1,w,True,False) for b in (1,4,16) for m,w in (("ar",8),("dflash",2),("dflash",8))]
    if stage == "routing":
        return [(m,"gsm8k",b,16,64,1,w,False,True) for b in (1,4,16) for m,w in (("ar",8),("dflash",2),("dflash",4),("dflash",8))]
    if stage == "dense_control":
        return [(m,d,b,16,128,3,8,False,False) for d in ("gsm8k","mt_bench") for b in (1,4,16) for m in ("ar","dflash","ar")]
    raise ValueError(stage)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("smoke","full","width","traces","routing","dense_control"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--attention", default="fa4")
    parser.add_argument("--moe-backend", default="flashinfer_mxfp4", choices=("flashinfer_mxfp4","marlin","triton_kernel"))
    parser.add_argument("--cell-limit", type=int)
    args = parser.parse_args()
    assert args.run_id.replace("_", "").replace("-", "").isalnum()
    LOGS.mkdir(parents=True, exist_ok=True)
    lock = (LOGS/"supervisor.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    worker = ROOT/'scripts/experiments/parallel_sd_worker.py'
    sha = hashlib.sha256(worker.read_bytes()).hexdigest()
    archive = LOGS/"worker_sources"/(sha+".py")
    archive.parent.mkdir(exist_ok=True)
    if not archive.exists():
        archive.write_bytes(worker.read_bytes())
    cells=[]
    for i,(method,dataset,bs,samples,tokens,repeats,width,profile,routing) in enumerate(experiments(args.stage)):
        name=f"{args.run_id}_{i:02d}_{method}_{dataset}_bs{bs}_w{width if method != 'ar' else 1}"
        command=[sys.executable,str(ROOT/'scripts/experiments/run_parallel_sd_cell.py'),"--name",name,"--directory",str(LOGS),"--profile-directory",str(PROFILES)]
        if profile:
            command += ["--nsys","--gpu-metrics"]
        target = "qwen3-8b" if args.stage == "dense_control" else "gpt-oss-20b"
        command += ["--","--target",target,"--method",method,"--dataset",dataset,"--bs",str(bs),"--samples",str(samples),"--tokens",str(tokens),"--repeats",str(repeats),"--attention",args.attention]
        if target == "gpt-oss-20b":
            command += ["--moe-runner-backend",args.moe_backend]
        if method != "ar":
            command += ["--block",str(width)]
        if routing:
            command += ["--routing-record"]
        cells.append({"name":name,"argv":command})
    if args.cell_limit:
        cells=cells[:args.cell_limit]
    path=LOGS/(args.run_id+".campaign.json")
    if path.exists():
        prior=json.loads(path.read_text())
        assert prior["commands"] == cells and prior["worker_sha256"] == sha
        pid=prior.get("supervisor_pid")
        if pid != os.getpid() and psutil.pid_exists(pid):
            process=psutil.Process(pid)
            if process.create_time()==prior["supervisor_start_time"] and process.status()!=psutil.STATUS_ZOMBIE:
                raise RuntimeError("Recorded supervisor is still live.")
    state={"status":"running","stage":args.stage,"run_id":args.run_id,"commands":cells,
        "supervisor_pid":os.getpid(),"supervisor_start_time":psutil.Process().create_time(),
        "worker_sha256":sha,"started_epoch":time.time(),"arguments":vars(args)}
    save(path,state)
    for i,cell in enumerate(cells):
        job=LOGS/(cell["name"]+".job.json")
        if job.exists():
            old=json.loads(job.read_text())
            if old["state"]=="success":
                continue
            raise RuntimeError("Inspect retained non-successful attempt: "+str(job))
        assert hashlib.sha256(worker.read_bytes()).hexdigest()==sha,"Worker changed during campaign."
        state.update(current_index=i,current_cell=cell["name"]);save(path,state)
        print("START",cell["name"],flush=True)
        code=subprocess.call(cell["argv"],cwd=ROOT)
        if code:
            state.update(status="stopped_after_cell_failure",exit_code=code,ended_epoch=time.time());save(path,state)
            raise SystemExit(code)
    state.update(status="success",ended_epoch=time.time());save(path,state)
    print("CAMPAIGN_SUCCESS",args.run_id,flush=True)


if __name__ == "__main__":
    main()
