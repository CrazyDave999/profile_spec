"""Small, explicit local trace bundle; no Git publication of profiler binaries."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',required=True);args=parser.parse_args()
    directory=ROOT/'logs/parallel_sd_20261104'
    campaign=json.loads((directory/(args.run_id+'.campaign.json')).read_text())
    assert campaign['stage']=='traces' and campaign['status']=='success'
    selected={('ar',1),('ar',16),('dflash',1),('dflash',16),('domino',16),('dspark',16),('dflash2',16)}
    records=[];files=[]
    for cell in campaign['commands']:
        name=cell['name'];data=json.loads((directory/(name+'.json')).read_text());a=data['arguments']
        if (a['method'],a['bs']) not in selected:continue
        assert json.loads((directory/(name+'.job.json')).read_text())['state']=='success'
        report=ROOT/'profiles/parallel_sd_20261104'/(name+'.nsys-rep')
        analysis=directory/(name+'.analysis.json');parsed=json.loads(analysis.read_text())
        assert parsed['correlation_coverage']>.99 and parsed['cuda_graph_captures']==0
        records.append(dict(case=name,method=a['method'],bs=a['bs'],samples=a['samples'],cap=a['tokens'],
            bytes=report.stat().st_size,SHA256=hashlib.sha256(report.read_bytes()).hexdigest(),
            analysis_SHA256=hashlib.sha256(analysis.read_bytes()).hexdigest(),
            graph_launches=parsed['cuda_graph_launches'],file=str(report.relative_to(ROOT))))
        files.extend([report,analysis])
    assert len(records)==len(selected)
    manifest=dict(campaign=args.run_id,traces=records,
        purpose='AR low/high BS and representative parallel SD; all warmed, same 16 GSM8K inputs, cap64, profile-only.',
        caveats=['Profiles include instrumentation and GPU metrics; use separate unprofiled CSV for throughput.',
          'CPU NVTX brackets graph dispatch, not GPU completion. Expand CUDA Graph nodes to inspect GPU execution.',
          'decode_control is outside model.forward and includes bookkeeping; not accept-only.',
          'BS1 contains repeated request waves and prefill. Use segmented decode envelopes and by-active-batch costs.',
          'DFlash/Domino default verify width16; DSpark default8, static verify.'])
    output=ROOT/'profiles/parallel_sd_20261104/representative_traces.zip'
    temporary=output.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=5) as archive:
        archive.writestr('MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2))
        for path in files:archive.write(path,arcname=path.name)
    with zipfile.ZipFile(temporary) as archive:
        assert archive.testzip() is None
        for r in records:
            name=Path(r['file']).name
            assert hashlib.sha256(archive.read(name)).hexdigest()==r['SHA256']
            assert hashlib.sha256(archive.read(name.replace('.nsys-rep','.analysis.json'))).hexdigest()==r['analysis_SHA256']
    temporary.replace(output)
    manifest['bundle']=str(output.relative_to(ROOT));manifest['bundle_bytes']=output.stat().st_size;manifest['bundle_SHA256']=hashlib.sha256(output.read_bytes()).hexdigest()
    (directory/'representative_traces_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print('Verified',len(records),'traces, ZIP MiB',round(output.stat().st_size/2**20,2),output)


if __name__=='__main__':main()
