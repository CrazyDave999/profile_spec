"""Launch one process group, retain terminal state, and yield to foreign GPU jobs."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from parallel_sd_worker import GPU, ROOT, save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--nsys", action="store_true")
    parser.add_argument("--gpu-metrics", action="store_true", help="Profile-only 1 kHz Blackwell desktop counters.")
    parser.add_argument("worker_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.gpu_metrics and not args.nsys:raise ValueError('--gpu-metrics requires --nsys')
    import pynvml
    import psutil
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByUUID(GPU)
    if pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
        raise RuntimeError("GPU is occupied; no job was launched.")
    assert args.name.replace("_", "").replace("-", "").isalnum()
    directory = ROOT / "logs/parallel_sd_20261104"
    result_path = directory / f"{args.name}.json"
    state_path = directory / f"{args.name}.job.json"
    log_path = directory / f"{args.name}.log"
    if state_path.exists():
        raise RuntimeError("Use a new attempt name; terminal evidence must not be overwritten.")
    cmd = [str(ROOT/".venv-sglang/bin/python"), str(ROOT/"parallel_sd_worker.py")]
    worker_args = args.worker_args[1:] if args.worker_args[:1] == ["--"] else args.worker_args
    cmd += worker_args + ["--output", str(result_path)]
    env = os.environ.copy()
    for key in ["GITHUB_TOKEN", "GH_TOKEN", "GH_ACCESS_TOKEN", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"]:
        env.pop(key, None)
    if args.nsys:
        output = ROOT / "profiles/parallel_sd_20261104" / args.name
        output.parent.mkdir(parents=True, exist_ok=True)
        env.update(SGLANG_ENABLE_NVTX_SCHEDULER="1", SGLANG_ENABLE_NVTX_OPERATIONS="1", PROFILE_SPEC_NVTX="1")
        cmd = ["nsys", "profile", "--trace=cuda,nvtx,osrt", "--sample=none", "--cpuctxsw=none", "--cuda-graph-trace=node", "--capture-range=cudaProfilerApi", "--capture-range-end=stop", "--force-overwrite=true", "--output", str(output)] + cmd + ["--profile"]
        if args.gpu_metrics:
            cmd[2:2]=['--gpu-metrics-devices=cuda-visible','--gpu-metrics-frequency=1000','--gpu-metrics-set=gb20x']
    state = {"state": "starting", "argv": cmd, "result": str(result_path), "log": str(log_path), "started_epoch": time.time()}
    directory.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        child = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        state.update(state="running", pid=child.pid, process_start_time=psutil.Process(child.pid).create_time())
        owned={child.pid:state['process_start_time']}
        save(state_path, state)
        try:
            while child.poll() is None:
                try:
                    descendants = psutil.Process(child.pid).children(recursive=True)
                    for process in descendants:
                        try:owned[process.pid]=process.create_time()
                        except psutil.NoSuchProcess:pass
                except psutil.NoSuchProcess:
                    pass
                foreign=[]
                for gpu_process in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
                    pid=gpu_process.pid
                    try:
                        process=psutil.Process(pid);created=process.create_time()
                        if os.getpgid(pid)==child.pid:
                            owned[pid]=created;continue
                        if owned.get(pid)==created:continue
                    except (psutil.NoSuchProcess,ProcessLookupError,psutil.AccessDenied):
                        if pid in owned:continue
                    foreign.append(pid)
                if foreign:
                    state.update(state="yielding_to_foreign_GPU", foreign_pids=foreign)
                    save(state_path, state)
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                    break
                time.sleep(.5)
        finally:
            code = child.wait()
            result = json.loads(result_path.read_text()) if result_path.exists() else {}
            state.update(exit_code=code, ended_epoch=time.time(), worker_status=result.get("status"), owned_process_start_times=owned, state="yielded" if state.get("foreign_pids") else ("success" if code == 0 and result.get("status") == "success" else "failed"))
            save(state_path, state)
    print(json.dumps(state), flush=True)
    raise SystemExit(0 if state["state"] == "success" else 2)


if __name__ == "__main__":
    main()
