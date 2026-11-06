"""Represent measured ceil-to-graph costs in SpsCostTable's floor-lookup format."""
import _bootstrap
import argparse
from bisect import bisect_left,bisect_right
import hashlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    source=json.loads(args.source.read_text());buckets=source['sample_batch_tokens'];values=source['sample_steps_per_sec']
    assert buckets==[8,16,32,64,128] and len(values)==len(buckets)
    edges=[1]+[b+1 for b in buckets[:-1]]
    aligned=dict(sample_batch_tokens=edges,sample_steps_per_sec=values,max_batch_tokens=source['max_batch_tokens'])
    for count in range(1,source['max_batch_tokens']+1):
        expected=values[bisect_left(buckets,count)]
        actual=values[max(0,bisect_right(edges,count)-1)]
        assert expected==actual
    payload=json.dumps(aligned,indent=2)
    if args.output.exists():assert args.output.read_text()==payload,'Preserve prior table attempts.'
    else:args.output.write_text(payload)
    evidence=dict(source=str(args.source),source_SHA256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
        output_SHA256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
        measured_graph_rows=buckets,lookup_interval_starts=edges,measured_steps_per_sec=values,
        proof='For every integer M=1..128, floor lookup in the transformed table equals the measured SPS at ceil_to_capture_bucket(M). No new hardware samples or cost values are invented.',
        limits=['The new sample_batch_tokens fields are interval starts, NOT measured token counts.',
          'A one-dimensional table still conflates request count and verify rows; compact at the same M can involve more requests/KV history than static calibration.',
          'Source and transformed table are separate experiment arms; never mutate a live campaign table.'])
    args.output.with_suffix('.mapping.json').write_text(json.dumps(evidence,indent=2))
    print(args.output);print('lookup interval starts',edges,'measured graph rows',buckets)


if __name__=='__main__':main()
