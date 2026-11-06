"""Check experiment completion against the frozen protocol and actual artifacts."""
import _bootstrap
import argparse
import csv
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import struct
import subprocess
import time
from urllib.parse import unquote, urlsplit
import zipfile
from audit_script_layout import canonical_tree
import ast

ROOT=_bootstrap.ROOT
LOGS=ROOT/'logs/moe_sd_20261106'
OUT=ROOT/'reports/2026-11-06_moe_parallel_sd'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(sequence,smoke,numerics):
    checks=[]
    def record(name,ok,details=None):
        checks.append({'requirement':name,'achieved':bool(ok),'evidence':details})
    protocol=json.loads((LOGS/'protocol.json').read_text())
    worker_sha=protocol['source_sha256']['parallel_sd_worker.py']
    measured_source=LOGS/'worker_sources'/(worker_sha+'.py')
    current_source=ROOT/'scripts/experiments/parallel_sd_worker.py'
    same_computation=canonical_tree(current_source.read_text())==ast.dump(ast.parse(measured_source.read_text()),include_attributes=False)
    record('Frozen measured worker preserved across directory relocation',sha(measured_source)==worker_sha and same_computation,{'measured_SHA256':worker_sha,'current_SHA256':sha(current_source),'same_computation_AST':same_computation})
    record('Archived worker source',sha(LOGS/'worker_sources'/(worker_sha+'.py'))==worker_sha)
    record('Pinned SGLang revision',subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT/'third_party/SGLang',text=True).strip()==protocol['sglang_revision'])
    for name in ('SGLang','JetSpec','DFlash'):
        changes=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT/'third_party'/name,text=True)
        record(name+' upstream clean',not changes,changes)
    for role,checkpoint in json.loads((ROOT/'models-moe-sd.json').read_text())['checkpoints'].items():
        alias=Path(checkpoint['alias']);snapshot=Path(checkpoint['snapshot']);valid=alias.resolve()==snapshot.resolve();files=[]
        for weight in checkpoint['weights']:
            path=alias/weight['file'];ok=path.is_file() and path.stat().st_size==weight['bytes']
            if ok:
                with path.open('rb') as f:
                    size=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(size))
                actual={n:{'shape':v['shape'],'dtype':v['dtype']} for n,v in header.items() if n!='__metadata__'}
                ok=actual==weight['tensors']
            valid&=ok;files.append({'path':str(path),'bytes':weight['bytes'],'header_and_size_match':ok})
        record(role+' external pinned checkpoint',valid,{'revision':checkpoint['revision'],'files':files})
    successful={};input_hashes={};packages=set();versions=set();headline=[];routing=[];traces=[]
    for stage,expected in protocol['planned_stages'].items():
        manifest=LOGS/(sequence+'_'+stage+'.campaign.json')
        if not manifest.exists():record(stage+' complete matrix',False,'Manifest absent');continue
        campaign=json.loads(manifest.read_text());valid=campaign['status']=='success' and len(campaign['commands'])==expected
        details=[]
        for cell in campaign['commands']:
            name=cell['name'];job=LOGS/(name+'.job.json');result=LOGS/(name+'.json')
            if not job.exists() or not result.exists():valid=False;details.append({'case':name,'state':'missing'});continue
            j=json.loads(job.read_text());d=json.loads(result.read_text());a=d.get('arguments',{})
            if j['state']!='success' or d.get('status')!='success':valid=False;details.append({'case':name,'state':j['state']});continue
            successful[name]=sha(result)
            expected_target='qwen3-8b' if stage=='dense_control' else 'gpt-oss-20b'
            good=(d['GPU_UUID']==protocol['hardware']['GPU_UUID'] and d['worker_source_sha256']==worker_sha and d['sglang_revision']==protocol['sglang_revision'] and a['target']==expected_target and a['samples']==16 and a['bs'] in (1,4,16) and a['attention']=='fa4')
            instrumented=stage in ('traces','routing')
            good&=a['tokens']==(64 if instrumented else 128) and a['repeats']==(1 if instrumented else 3)
            good&=bool(a['profile'])==(stage=='traces') and bool(a.get('routing_record'))==(stage=='routing') and not a.get('logprobs') and not a.get('fp32_lm_head')
            engine=d['engine_resolved_after'];graph=engine['cuda_graph_config']
            good&=graph['decode']['backend']=='full' and graph['decode']['bs']==[1,2,4,8,16] and graph['prefill']['backend']=='disabled' and not engine['disable_overlap_schedule'] and engine['disable_radix_cache']
            good&=engine['attention_backend']=='fa4' and engine['max_total_tokens']==32768
            if expected_target=='gpt-oss-20b':
                good&=engine['moe_runner_backend']=='flashinfer_mxfp4' and d['template_date']=='2026-11-06'
                frozen=ROOT/d['frozen_inputs_file'] if not Path(d['frozen_inputs_file']).is_absolute() else Path(d['frozen_inputs_file'])
                good&=sha(frozen)==d['frozen_inputs_file_sha256']
            if a['method']=='dflash':good&=engine['speculative_draft_attention_backend']=='fa4'
            input_hashes.setdefault((expected_target,a['dataset']),set()).add(d['selected_prompt_sha256'])
            packages.add(json.dumps(d['package_versions'],sort_keys=True));versions.add(d['sglang_revision'])
            trial_lengths=[]
            for trial in d['results']:
                good&=len(trial['outputs'])==16 and trial['output_tokens']==16*a['tokens'] and trial['seconds']>0
                for output in trial['outputs']:
                    m=output['meta_info'];good&=len(output['output_ids'])==a['tokens'] and m['completion_tokens']==a['tokens'] and m.get('cached_tokens',0)==0 and m.get('num_retractions',0)==0
                trial_lengths.append(len(trial['outputs']))
            good&=len(d['results'])==a['repeats']
            if stage=='traces':traces.append(name)
            elif stage=='routing':routing.append(name)
            else:headline.append(name)
            valid&=good;details.append({'case':name,'state':'success','protocol_matches':good,'result_SHA256':successful[name]})
        record(stage+' complete matrix',valid,{'expected':expected,'campaign_status':campaign['status'],'cells':details})
    record('Inputs identical within target/dataset across methods, BS and widths',all(len(v)==1 for v in input_hashes.values()),{'/'.join(k):sorted(v) for k,v in input_hashes.items()})
    record('Single package/version cohort',len(packages)==1 and len(versions)==1,{'package_variants':len(packages),'revisions':sorted(versions)})
    summary=json.loads((OUT/'summary.json').read_text());rows=summary['rows']
    record('All 75 headline cells summarized with both AR bookends',len(rows)==75 and set(r['case'] for r in rows)==set(headline) and all(r['gain'] is not None and r['AR_bracket_drift'] is not None for r in rows),{'rows':len(rows),'max_AR_drift':max((r['AR_bracket_drift'] or 0) for r in rows) if rows else None,'max_repeat_CV':max((r['repeat_CV'] or 0) for r in rows) if rows else None})
    sources=json.loads((OUT/'source_manifest.json').read_text())['cells']
    record('Summary source hashes match actual results',set(r['case'] for r in sources)==set(headline) and all(successful.get(r['case'])==r['sha256'] for r in sources))
    correctness=json.loads((OUT/'correctness.json').read_text())
    record('AR/SD and repeated-output parity explicitly retained',set(r['case'] for r in correctness)==set(headline) and all('AR_exact_requests' in r for r in correctness if '_dflash_' in r['case']))
    parity_path=OUT/'cross_configuration_parity.json'
    if parity_path.exists():
        parity=json.loads(parity_path.read_text())['comparisons'];counts={k:sum(c['comparison']==k for c in parity) for k in {c['comparison'] for c in parity}}
        record('Bookends, BS, width and repeated-cohort output comparisons retained',counts=={'AR_bookends':21,'batch_size_1_to_4':18,'batch_size_1_to_16':18,'width_2_to_4':6,'width_2_to_8':6,'repeated_width8_cohort':6},counts)
    else:record('Bookends, BS, width and repeated-cohort output comparisons retained',False)
    trace_evidence=json.loads((OUT/'trace_evidence.json').read_text())['cases']
    record('All 9 warmed traces prove native FP4, replay and correlation',len(trace_evidence)==9 and set(t['case'] for t in trace_evidence)==set(traces) and all(t['graph_launches']>0 and t['graph_captures']==0 and t['correlation_coverage']==1 and t['verified_native_FP4_target_GEMMs'] and sha(ROOT/t['nsys_rep'])==t['nsys_rep_SHA256'] for t in trace_evidence))
    components=json.loads((OUT/'target_components_evidence.json').read_text())['cases']
    record('Per-replay operator attribution validated for 9 traces',len(components)==9 and set(c['case'] for c in components)==set(traces) and all(c['validated_rounds']>0 and all(r['validated_BF16_linears']==49 and r['validated_FP4_GEMMs']==48 for r in c['rounds']) for c in components))
    route_path=OUT/'routing_evidence.json'
    if route_path.exists():
        route_evidence=json.loads(route_path.read_text());route_sources=route_evidence['sources']
        record('All 12 routing dumps analyzed with matching file hashes',len(route_sources)==12 and set(r['case'] for r in route_sources)==set(routing) and all(sha(ROOT/r['file'])==r['SHA256'] and r['records']>0 for r in route_sources))
        comparisons=[c for r in route_evidence.get('trajectory_checks',[]) for c in r['matched_trace_comparisons']]
        record('Routing instrumentation trajectory compared against all 9 matching traces',len(comparisons)==9 and all(c['requests']==16 and sha(ROOT/c['reference'])==c['reference_SHA256'] for c in comparisons),comparisons)
    else:record('All 12 routing dumps analyzed with matching file hashes',False)
    for run,kind,expected in [(smoke,'campaign',2),(numerics,'numerics',4)]:
        path=LOGS/(run+'.'+kind+'.json');valid=path.exists();count=0
        if valid:
            m=json.loads(path.read_text());valid=m['status']=='success' and len(m['commands'])==expected
            for c in m['commands']:
                p=LOGS/(c['name']+'.job.json');ok=p.exists() and json.loads(p.read_text())['state']=='success';valid&=ok;count+=bool(ok)
                result=LOGS/(c['name']+'.json')
                if ok:
                    d=json.loads(result.read_text());args=d['arguments']
                    valid&=d['status']=='success' and d['GPU_UUID']==protocol['hardware']['GPU_UUID'] and d['worker_source_sha256']==worker_sha and d['sglang_revision']==protocol['sglang_revision'] and args['target']=='gpt-oss-20b' and args['samples']==2
                    valid&=args['tokens']==(32 if kind=='campaign' else 128) and args['repeats']==1
                    if kind=='numerics':valid&=args['logprobs'] and args['fp32_lm_head']==d['engine_resolved_after']['enable_fp32_lm_head']
        record(run+' completed independent diagnostic cells',valid,{'expected':expected,'success':count})
    numeric=OUT/'numerics.json'
    if numeric.exists():
        n=json.loads(numeric.read_text());mismatches=[p for c in n['comparisons'] for p in c['prompts'] if not p['equal']]
        record('Numerical diagnostics analyzed separately from headline',n['status']=='success' and n['successful_cells']==4 and len(n['comparisons'])==2 and len(n['BF16_logprob_vs_headline'])==2 and all(p['AR_top5'] and p['SD_top5'] and p['AR_top2_logprob_margin'] is not None and p['SD_top2_logprob_margin'] is not None for p in mismatches),n['BF16_logprob_vs_headline'])
    else:record('Numerical diagnostics analyzed separately from headline',False)
    bundle=LOGS/'representative_traces_manifest.json'
    if bundle.exists():
        m=json.loads(bundle.read_text());zpath=ROOT/m['bundle'];ok=sha(zpath)==m['bundle_SHA256'] and len(m['cases'])==5
        with zipfile.ZipFile(zpath) as z:
            ok&=z.testzip() is None
            for case in m['cases']:
                for field in ('raw','analysis'):
                    p=ROOT/case[field+'_file'];expected=case[field+'_SHA256'];ok&=sha(p)==expected and hashlib.sha256(z.read(p.name)).hexdigest()==expected
        record('Five representative traces packaged with verified ZIP and SHA256',ok,{'bundle':m['bundle'],'MiB':m['bundle_bytes']/2**20})
    else:record('Five representative traces packaged with verified ZIP and SHA256',False)
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.refs=[]
        def handle_starttag(self,tag,attrs):
            self.refs.extend(v for k,v in attrs if k in ('href','src'))
    report=OUT.with_suffix('.html');parser=Links();parser.feed(report.read_text());missing=[]
    for ref in parser.refs:
        u=urlsplit(ref)
        if not u.scheme and u.path and not (report.parent/unquote(u.path)).exists():missing.append(ref)
    record('Report assets and local download links exist',not missing,missing)
    index=(ROOT/'reports/index.html').read_text();record('Dated report linked from index',report.name in index)
    required=['scaling.svg','width_tradeoff.svg','dense_comparison.svg','expert_coverage.svg','target_components.svg','checkpoint_storage.svg','findings.json','wave_tails.csv']
    record('All requested comparisons and detail artifacts generated',all((OUT/f).exists() for f in required),required)
    for name in (sequence+'.admission.json',sequence+'.sequence.json'):
        p=LOGS/name;m=json.loads(p.read_text());record(name+' successful controller termination',m['status']=='success',m['status'])
    findings=json.loads((OUT/'findings.json').read_text()) if (OUT/'findings.json').exists() else {}
    record('Report covers headline, dense comparison, routing, width and numerical limits',len(findings.get('headline',[]))==9 and len(findings.get('dense_control',[]))==6 and len(findings.get('routing_full_active_batch',[]))==12 and len(findings.get('best_measured_width',[]))==6 and 'numerical_diagnostic' in findings)
    return {'sequence':sequence,'checked_epoch':time.time(),'achieved':all(c['achieved'] for c in checks),'checks':checks,'successful_main_cells':len(successful),'headline_cells':len(headline),'profile_cells':len(traces),'routing_cells':len(routing)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--sequence',default='native03');p.add_argument('--smoke',default='smoke03');p.add_argument('--numerics',default='numerics03');a=p.parse_args();result=audit(a.sequence,a.smoke,a.numerics)
    path=LOGS/'completion_audit.json';temporary=path.with_suffix('.json.tmp');temporary.write_text(json.dumps(result,indent=2));temporary.replace(path)
    print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2));print(json.dumps([c for c in result['checks'] if not c['achieved']],indent=2));raise SystemExit(0 if result['achieved'] else 1)


if __name__=='__main__':main()
