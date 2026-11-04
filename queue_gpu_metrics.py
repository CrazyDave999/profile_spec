"""Wait for the live main campaign to finish, then collect GPU counter timelines.

Checks the authoritative PID and /proc state rather than treating a stale status
file as evidence of liveness. This does not stop or restart the primary worker.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent


def live_supervisor(pid):
    try:
        path=Path('/proc')/str(pid)
        state=(path/'stat').read_text().split(') ',1)[1].split()[0]
        command=(path/'cmdline').read_bytes()
        return state not in ('Z','X') and b'run_overnight.py' in command
    except (OSError,IndexError):
        return False


def main():
    pid=json.loads((ROOT/'logs/overnight_supervisor_pid.json').read_text())['pid']
    status=ROOT/'logs/gpu_metrics_queue.json'
    while live_supervisor(pid):
        status.write_text(json.dumps({'state':'waiting_for_live_main_supervisor','supervisor_pid':pid,
                                     'checked_unix':time.time()},indent=2)+'\n')
        time.sleep(20)
    status.write_text(json.dumps({'state':'starting_metrics_campaign','checked_unix':time.time()},indent=2)+'\n')
    command=[sys.executable,'-u',str(ROOT/'run_overnight.py'),'--metrics-only',
             '--campaign','gpu_metrics_20261103','--deadline','2026-11-03T01:45:00+00:00']
    with open(ROOT/'logs/gpu_metrics_supervisor.log','a') as log:
        child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        status.write_text(json.dumps({'state':'running_metrics_campaign','pid':child.pid},indent=2)+'\n')
        code=child.wait()
    status.write_text(json.dumps({'state':'metrics_campaign_finished','returncode':code,
                                 'finished_unix':time.time()},indent=2)+'\n')


if __name__=='__main__': main()
