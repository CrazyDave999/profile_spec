"""Bundle representative completed MoE traces with exact hashes and analyses."""
import _bootstrap
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=_bootstrap.ROOT
LOGS=ROOT/'logs/moe_sd_20261106'
PROFILES=ROOT/'profiles/moe_sd_20261106'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--sequence',required=True);args=parser.parse_args()
    campaign=json.loads((LOGS/(args.sequence+'_traces.campaign.json')).read_text());selected=[]
    for cell in campaign['commands']:
        name=cell['name'];result=json.loads((LOGS/(name+'.json')).read_text());a=result['arguments']
        choose=(a['method']=='ar' and a['bs'] in (1,16)) or (a['method']=='dflash' and ((a['block']==8 and a['bs'] in (1,16)) or (a['block']==2 and a['bs']==16)))
        if not choose:continue
        assert json.loads((LOGS/(name+'.job.json')).read_text())['state']=='success'
        raw=PROFILES/(name+'.nsys-rep');analysis=LOGS/(name+'.analysis.json')
        assert raw.exists() and analysis.exists()
        selected.append({'case':name,'method':a['method'],'bs':a['bs'],'width':a['block'] if a['method']!='ar' else 1,
            'raw_file':str(raw.relative_to(ROOT)),'raw_bytes':raw.stat().st_size,'raw_SHA256':hashlib.sha256(raw.read_bytes()).hexdigest(),
            'analysis_file':str(analysis.relative_to(ROOT)),'analysis_SHA256':hashlib.sha256(analysis.read_bytes()).hexdigest()})
    assert len(selected)==5,len(selected)
    manifest={'sequence':args.sequence,'cases':selected,'protocol':'Same 16 GSM8K prompts, fixed cap64, one warmed profiled repeat; instrumented data excluded from headline throughput.',
        'caveats':['CPU NVTX and GPU work overlap; do not add their durations.','Expand CUDA Graph nodes to inspect expert GEMM execution.','Use target_components.csv / trace evidence for validated operator attribution.']}
    archive=PROFILES/'representative_traces.zip';temporary=archive.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for item in selected:
            for field in ('raw_file','analysis_file'):
                path=ROOT/item[field];z.write(path,path.name)
        z.writestr('bundle_manifest.json',json.dumps(manifest,indent=2))
    with zipfile.ZipFile(temporary) as z:
        assert z.testzip() is None
        for item in selected:
            assert hashlib.sha256(z.read(Path(item['raw_file']).name)).hexdigest()==item['raw_SHA256']
            assert hashlib.sha256(z.read(Path(item['analysis_file']).name)).hexdigest()==item['analysis_SHA256']
    temporary.replace(archive)
    manifest.update(bundle=str(archive.relative_to(ROOT)),bundle_bytes=archive.stat().st_size,bundle_SHA256=hashlib.sha256(archive.read_bytes()).hexdigest())
    (LOGS/'representative_traces_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({'traces':len(selected),'bundle_MiB':archive.stat().st_size/2**20,'bundle':str(archive.relative_to(ROOT))}))


if __name__=='__main__':main()
