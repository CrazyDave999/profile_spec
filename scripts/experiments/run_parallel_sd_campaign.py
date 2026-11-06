"""Serial, resumable exploratory matrix. A failed cell stops for inspection."""
import _bootstrap
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from parallel_sd_worker import ROOT, save


def plan(stage):
    if stage == "smoke":
        return [(m, "gsm8k", 1, 2, 32, 1, False) for m in ["domino", "dspark", "dflash2"]]
    if stage == "pilot":
        return [(m, "gsm8k", 1, 8, 128, 2, False) for m in ["ar", "dflash", "domino", "dspark", "dflash2", "ar"]]
    if stage == "full":
        return [(m, d, b, 16, 128, 3, False) for d in ["gsm8k", "humaneval", "mt_bench"] for b in [1, 4, 16] for m in ["ar", "dflash", "domino", "dspark", "dflash2", "ar"]]
    if stage == "traces":
        return [(m, "gsm8k", b, 16, 64, 1, True) for b in [1, 16] for m in ["ar", "dflash", "domino", "dspark", "dflash2"]]
    if stage == "hybrid":
        return [(m, "gsm8k", b, 8, 128, 3, False) for b in [1, 4] for m in ["ar", "dflash", "ar"]]
    if stage == "width8":
        return [(m, d, b, 16, 128, 3, False) for d in ['gsm8k','mt_bench'] for b in [1, 16] for m in ["ar", "dflash", "domino", "dspark", "dflash2", "ar"]]
    if stage == "numerics":
        return [(m, "gsm8k", 1, 2, 128, 1, False) for _ in [False, True] for m in ["ar", "dflash"]]
    if stage == "sps":
        return [("dspark", "gsm8k", b, b, 256, 2, False) for b in [1, 2, 4, 8, 16]]
    if stage == "adaptive":
        return [(m, "gsm8k", b, 16, 128, 3, False) for b in [1, 4, 16] for m in ['ar','dspark','dspark','ar']]
    raise ValueError(stage)


def fit_sps(directory, manifest):
    probes=[];evidence=[]
    for cell in manifest['commands']:
        data=json.loads((directory/(cell['name']+'.json')).read_text())
        bs=data['arguments']['bs'];width=data['engine_resolved_after']['speculative_num_draft_tokens']
        states=data['engine_resolved_after']['internal_states']
        assert len(states)==1,'This calibration is TP1 / DP1 only.'
        payload=states[0].get('dspark_info_record') or {}
        assert float(payload.get('simulate_acc_len') or 0)==1.0,payload.keys()
        records=[r for r in payload.get('records',[]) if r.get('num_running_reqs')==bs and r.get('num_verify_tokens')==bs*width and r.get('step_cpu_ms') is not None]
        # The measured 256-token, simulated-accept-one generation contributes
        # >128 full-BS steps. Taking its final 128 excludes the 32-token warmup.
        assert len(records)>=128, (cell['name'],len(records))
        selected=records[-128:];times=[r['step_cpu_ms']/1000 for r in selected]
        median=statistics.median(times);probes.append((bs*width,1/median))
        evidence.append(dict(case=cell['name'],bs=bs,verify_width=width,records=selected,
            median_step_seconds=median,relative_std=statistics.stdev(times)/statistics.mean(times)))
    table=dict(sample_batch_tokens=[x for x,y in probes],sample_steps_per_sec=[y for x,y in probes],max_batch_tokens=max(x for x,y in probes))
    out=directory/(manifest['run_id']+'.sps.json');save(out,table)
    save(out.with_suffix('.evidence.json'),dict(probes=evidence,
        definition='1 / median recorded complete step CPU interval, static verify, simulated accept length 1, final 128 aligned full-BS steps per cell.',
        caveats=['Calibration outputs are simulated and invalid for correctness/quality or headline speedup.',
          '1D static table confounds request count with verify tokens; compact steps with same tokens may involve more requests and longer KV history.',
          'Measured on short GSM8K prompts with generated history up to 256 tokens; not universal across contexts.',
          'Table lookup uses floor probe, matching upstream SpsCostTable.']))
    return str(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["smoke", "pilot", "full", "traces", "hybrid", "width8", "numerics", "sps", "adaptive"], required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument('--sps-table',type=Path)
    parser.add_argument('--adaptive-dataset',choices=['gsm8k','mt_bench','both'],default='gsm8k')
    parser.add_argument('--adaptive-bs',type=int,choices=[1,4,16])
    args = parser.parse_args()
    if args.stage=='adaptive' and (args.sps_table is None or not args.sps_table.is_file()):
        raise ValueError('Adaptive comparison requires a completed local SPS calibration table.')
    if args.stage == 'sps':
        os.environ.update(SGLANG_DSPARK_ENABLE_SPS_RECORD='1',SGLANG_SIMULATE_ACC_LEN='1.0')
    elif float(os.environ.get('SGLANG_SIMULATE_ACC_LEN','-1'))>0:
        raise RuntimeError('Simulated acceptance is forbidden outside the SPS calibration stage.')
    if args.stage=='adaptive':
        # Matched, CPU-only core/step interval records reveal whether compact
        # actually trims work. No confidence-tensor D2H or GPU event timers.
        os.environ['SGLANG_DSPARK_ENABLE_SPS_RECORD']='1'
    directory = ROOT/"logs/parallel_sd_20261104"
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory/"supervisor.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    import psutil
    manifest_path = directory/f"{args.run_id}.campaign.json"
    commands = []
    experiments=plan(args.stage)
    if args.stage=='adaptive':
        datasets=['gsm8k','mt_bench'] if args.adaptive_dataset=='both' else [args.adaptive_dataset]
        experiments=[(m,d,b,n,t,r,p) for d in datasets for m,_,b,n,t,r,p in experiments if args.adaptive_bs is None or b==args.adaptive_bs]
    for i,(m,d,b,n,t,r,profile) in enumerate(experiments):
        name=f"{args.run_id}_{i:02d}_{m}_{d}_bs{b}"
        cmd=[sys.executable,str(ROOT/'scripts/experiments/run_parallel_sd_cell.py'),"--name",name]
        if profile:cmd += ["--nsys","--gpu-metrics"]
        cmd += ["--","--method",m,"--dataset",d,"--bs",str(b),"--samples",str(n),"--tokens",str(t),"--repeats",str(r)]
        if args.stage == "hybrid":cmd += ["--target","qwen3.5-4b"]
        if args.stage == "width8" and m != "ar":cmd += ["--block","8"]
        if args.stage == "numerics":
            cmd += ["--logprobs"]
            if i >= 2:cmd += ["--fp32-lm-head"]
        if args.stage == 'adaptive' and m=='dspark':
            cmd += ['--sps-table',str(args.sps_table.resolve())]
            if i%4==2:cmd += ['--ragged-mode','compact']
        commands.append({"name":name,"argv":cmd})
    if manifest_path.exists():
        previous=json.loads(manifest_path.read_text())
        assert previous['commands']==commands
    manifest={"status":"running","stage":args.stage,"run_id":args.run_id,"commands":commands,"supervisor_pid":os.getpid(),"supervisor_start_time":psutil.Process().create_time(),"worker_sha256":hashlib.sha256((ROOT/'scripts/experiments/parallel_sd_worker.py').read_bytes()).hexdigest(),"started_epoch":time.time(),"calibration_environment":{k:os.environ.get(k) for k in ['SGLANG_DSPARK_ENABLE_SPS_RECORD','SGLANG_SIMULATE_ACC_LEN']},"sps_table_input":str(args.sps_table) if args.sps_table else None}
    save(manifest_path,manifest)
    for i,cell in enumerate(commands):
        job=directory/(cell['name']+'.job.json')
        if job.exists():
            prior=json.loads(job.read_text())
            if prior['state']=='success':continue
            raise RuntimeError(f"Existing non-successful cell needs inspection: {job}")
        manifest.update(current_index=i,current_cell=cell['name']);save(manifest_path,manifest)
        print('START',cell['name'],flush=True)
        code=subprocess.call(cell['argv'],cwd=ROOT)
        if code:
            manifest.update(status='stopped_after_cell_failure',exit_code=code,ended_epoch=time.time());save(manifest_path,manifest)
            raise SystemExit(code)
    if args.stage == 'sps':
        try:manifest['sps_table']=fit_sps(directory,manifest)
        except Exception as exc:
            manifest.update(status='stopped_after_calibration_failure',error=repr(exc),ended_epoch=time.time());save(manifest_path,manifest)
            raise
    manifest.update(status='success',ended_epoch=time.time());save(manifest_path,manifest)
    print('CAMPAIGN_SUCCESS',args.run_id,flush=True)


if __name__=='__main__':
    main()
