"""Audit delivery against the requested dataset/BS/time/memory/timeline scope."""
import _bootstrap
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import sqlite3
from urllib.parse import urlparse,unquote
import zipfile

from PIL import Image
from sweep_worker import save

ROOT=_bootstrap.ROOT
OUT=ROOT/'reports/2026-11-03'
DATASETS=['gsm8k','math500','aime25','humaneval','mbpp','livecodebench','mt_bench']


class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):
        self.links.extend(value for key,value in attrs if key in ('src','href'))


def main():
    result={'status':'running'}
    path=OUT/'completion_audit.json'
    save(path,result)
    cases=[]
    for campaign in ['overnight_20261103','gpu_metrics_20261103']:
        folder=ROOT/'logs'/campaign
        plan=json.loads((folder/'plan.json').read_text())
        for case in plan:
            r=json.loads((folder/(case['name']+'.json')).read_text())
            assert r['status']=='success',(case['name'],r['status'])
            a=r['arguments']
            for key in ['dataset','engine','bs','tokens','samples','arm']:
                assert a[key]==case[key],(case['name'],key)
            assert r['upstream_revision']=='673267e4faba71a63099c258f0802c2f3aa44f8e'
            assert r['gpu']=='NVIDIA GeForce RTX 5090'
            if a['arm']=='AR':assert r['head_parameter_bytes']==0
            else:assert r['head_parameter_bytes']>0
            cases.append((case,r,folder/(case['name']+'.analysis.json')))
    assert len(cases)==175
    for dataset in DATASETS:
        for bs in [1,2,4,8,16]:
            match=[r for c,r,p in cases if r['arguments']['dataset']==dataset and r['arguments']['engine']=='batch_reference'
                and r['arguments']['bs']==bs and r['arguments']['trace']]
            assert len(match)==1,(dataset,bs)
            r=match[0]
            for phase in ['AR/decode_forward','JetSpec/draft_forward','JetSpec/verify_forward']:
                batch_values=[x['metadata']['batch'] for x in r['phase_records'] if x['phase']==phase]
                assert batch_values and max(batch_values)==bs,(dataset,bs,phase,batch_values)
            for arm in ['AR','JetSpec']:
                matches=[r for c,r,p in cases if r['arguments']['dataset']==dataset and r['arguments']['engine']=='batch_reference'
                    and r['arguments']['bs']==bs and not r['arguments']['trace'] and r['arguments']['arm']==arm]
                assert len(matches)==1
                assert len(matches[0]['results'][arm]['outputs'])==(30 if dataset=='aime25' else 32)
        native=[r for c,r,p in cases if r['arguments']['dataset']==dataset and r['arguments']['engine']=='optimized'
                and r['arguments']['trace'] and r['arguments'].get('budget',128)==128 and not r.get('nsys_gpu_metrics_enabled')]
        assert len(native)==1
    trace_cases=[(c,r,p) for c,r,p in cases if r['arguments']['trace']]
    assert len(trace_cases)==63
    def check_trace(item):
        case,r,apath=item
        rep,db=Path(r['nsys_report']),Path(r['nsys_sqlite'])
        assert rep.is_file() and rep.stat().st_size>1024
        assert db.is_file()
        analysis=json.loads(apath.read_text())
        assert analysis['kernel_events']>0 and analysis['kernel_attribution_fraction']==1.0
        assert analysis['graph_capture_calls_in_capture_range']==0
        if r['arguments']['engine']=='optimized':assert analysis['graph_launch_calls']>0
        con=sqlite3.connect(f'file:{db}?mode=ro',uri=True)
        assert con.execute('PRAGMA quick_check').fetchone()[0]=='ok'
        assert con.execute('SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL').fetchone()[0]==analysis['kernel_events']
        assert con.execute('SELECT COUNT(*) FROM NVTX_EVENTS').fetchone()[0]>0
        assert con.execute("SELECT COUNT(*) FROM sqlite_master WHERE name='CUDA_GPU_MEMORY_USAGE_EVENTS'").fetchone()[0]==1
        if r.get('nsys_gpu_metrics_enabled'):
            assert con.execute('SELECT COUNT(*) FROM GPU_METRICS').fetchone()[0]>0
            counters=json.loads(apath.with_name(case['name']+'.counters.json').read_text())
            assert set(counters['phases'])=={'AR/decode_forward','JetSpec/draft_forward','JetSpec/verify_forward'}
            assert abs(counters['sampling_hz_estimate']-1000)<10
        con.close()
        assert r['phase_records']
        assert all(x['peak_allocated']>=x['allocated_before'] for x in r['phase_records'])
        return rep
    with ThreadPoolExecutor(2) as pool:reports=list(pool.map(check_trace,trace_cases))
    parsed=Links();parsed.feed((OUT/'report.html').read_text())
    missing=[]
    for href in parsed.links:
        url=urlparse(href)
        if url.scheme or href.startswith('#'):continue
        target=(OUT/unquote(url.path)).resolve()
        if not target.exists():missing.append(href)
    assert not missing,missing
    for p in OUT.glob('*.png'):
        with Image.open(p) as im:im.verify()
    archive=OUT/'nsys_timelines.zip'
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        names=set(z.namelist())
        assert len(names)==64
        for p in reports:assert p.name in names and z.getinfo(p.name).file_size==p.stat().st_size
    initial=ROOT/'reports/2026-11-03_initial_0629'
    original=json.loads((initial/'coverage.json').read_text())
    prepared=datetime.fromisoformat(original['generated_Beijing'])
    deadline=datetime.fromisoformat('2026-11-03T10:00:00+08:00')
    assert prepared<deadline
    assert original['main_cases_present']==168 and original['missing_main_cases']==[]
    assert original['readable_timeline_count']==63
    old_links=Links();old_links.feed((initial/'report.html').read_text())
    assert len([p for p in old_links.links if p.endswith('.nsys-rep')])==63
    assert datetime.fromtimestamp((initial/'report.html').stat().st_mtime,prepared.tzinfo)<deadline
    validation=json.loads((ROOT/'logs/batch_reference_validation.json').read_text())
    assert validation['status']=='passed'
    reservation=json.loads((ROOT/'logs/kv_reservation_analysis.json').read_text())
    assert reservation['CPU_runtime_check']['status']=='passed'
    result={'status':'passed','successful_experiment_cases':175,'SQLite_integrity_checked_reports':63,
        'dataset_count':7,'batch_sizes':[1,2,4,8,16],
        'true_batched_forward_metadata_verified':True,'independent_head_free_AR_benchmarks_verified':True,
        'GPU_phase_kernel_correlation_coverage':1.0,'GPU_hardware_counter_reports':7,
        'report_local_links_checked':len(parsed.links),'missing_report_links':[],
        'timeline_zip_entries':64,'timeline_zip_CRC_check':'passed','timeline_zip_bytes':archive.stat().st_size,
        'initial_complete_report_prepared_Beijing':prepared.isoformat(),'initial_report_ready_before_10':True,
        'final_audit_time_Beijing':datetime.now(prepared.tzinfo).isoformat(),
        'final_experiment_source_SHA256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (ROOT/'scripts').rglob('*.py') if not p.is_symlink()},
        'limitations':['Batch sweep uses experimental eager SDPA/dense KV, not native optimized BS>1.',
            'BF16 token equality is not guaranteed; no answer-quality evaluation was run.',
            'Hardware counters are 1 kHz phase aggregates with boundary-window mixing.',
            'Initial report was ready at 06:29; hardware-counter interpretation and final packaging were added later.']}
    save(path,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['final_experiment_source_SHA256','limitations']},indent=2))


if __name__=='__main__':main()
