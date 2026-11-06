"""Compare stored output IDs across bookends, batch sizes, widths and cohorts."""
import _bootstrap
from collections import defaultdict
import json


def analyze(cells,out):
    comparisons=[]
    def compare(left,right,kind):
        assert left['target']==right['target'] and left['dataset']==right['dataset'] and left['input_sha256']==right['input_sha256']
        assert len(left['outputs'])==len(right['outputs'])
        prompts=[]
        for i,(a,b) in enumerate(zip(left['outputs'],right['outputs'])):
            assert len(a)==len(b)
            first=next((j for j,(x,y) in enumerate(zip(a,b)) if x!=y),None)
            prompts.append({'prompt':i,'exact':a==b,'first_mismatch':first,'left_token':a[first] if first is not None else None,'right_token':b[first] if first is not None else None})
        comparisons.append({'comparison':kind,'left':left['case'],'right':right['case'],'target':left['target'],'dataset':left['dataset'],'requests':len(prompts),'exact_requests':sum(p['exact'] for p in prompts),'prompts':prompts})
    groups=defaultdict(list)
    for c in cells:groups[c['stage'],c['dataset'],c['bs'],c['method'],c['width']].append(c)
    unique={}
    for key,items in groups.items():
        items.sort(key=lambda c:c['case']);unique[key]=items[0]
        if key[3]=='ar' and len(items)==2:compare(items[0],items[1],'AR_bookends')
    for key,c in unique.items():
        stage,dataset,bs,method,width=key
        if bs in (4,16) and (stage,dataset,1,method,width) in unique:
            compare(unique[stage,dataset,1,method,width],c,'batch_size_1_to_'+str(bs))
        if stage=='width' and method=='dflash' and width in (4,8) and (stage,dataset,bs,method,2) in unique:
            compare(unique[stage,dataset,bs,method,2],c,'width_2_to_'+str(width))
        if stage=='width' and method=='dflash' and width==8 and ('full',dataset,bs,method,width) in unique:
            compare(unique['full',dataset,bs,method,width],c,'repeated_width8_cohort')
    result={'comparisons':comparisons,'scope':'First-repeat output IDs, exact same input IDs within each comparison; bookends and batch/width comparisons preserve first divergence. Cross-target token IDs are deliberately not compared. No quality or distribution-equivalence claim.'}
    path=out/'cross_configuration_parity.json';tmp=path.with_suffix('.json.tmp');tmp.write_text(json.dumps(result,indent=2));tmp.replace(path)
    return result
