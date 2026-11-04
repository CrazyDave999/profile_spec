"""One isolated benchmark/profile case. Large artifacts stay in profiles/."""
import argparse
from collections import Counter
from contextlib import contextmanager, nullcontext
import functools
import importlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
import tempfile

import torch
from jetspec import load_draft_head
from jetspec.engines.jetspec import JetSpecEngine, SamplingParams
from jetspec.methods.jetspec.draft_head_adapter import GraphedDraftHead
from batch_reference import BatchedReference, Recorder

ROOT = Path(__file__).resolve().parent


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w',dir=path.parent,prefix=path.name+'.',suffix='.tmp',delete=False) as f:
        f.write(json.dumps(data, indent=2) + '\n')
        temp=Path(f.name)
    temp.replace(path)


@contextmanager
def optimized_instrumentation(rec, method):
    from jetspec.core.model_runner import ModelRunner
    from jetspec.engines.jetspec.verify.ar_graph import GraphedAR
    from jetspec.engines.jetspec.verify.graph_capture import GraphedVerify
    from jetspec.engines.jetspec.serving.paged_kv_cache import PagedKVCache
    from jetspec.methods.jetspec.tree import get_algorithm
    patches = []

    def wrap(owner, attr, label):
        original = getattr(owner, attr)
        @functools.wraps(original)
        def f(*args, **kwargs):
            name = label() if callable(label) else label
            with rec.span(name):
                return original(*args, **kwargs)
        patches.append((owner, attr, original))
        setattr(owner, attr, f)

    wrap(ModelRunner, 'forward', lambda: method[0] + '/prefill')
    wrap(GraphedAR, 'replay_ar', 'AR/decode_forward')
    wrap(GraphedVerify, 'replay', 'JetSpec/verify_forward')
    wrap(GraphedDraftHead, '__call__', 'JetSpec/draft_forward')
    wrap(PagedKVCache, 'gather', 'JetSpec/kv_commit')
    wrap(importlib.import_module('jetspec.methods.jetspec.tree._core.accept'), 'gpu_tree_accept', 'JetSpec/accept')
    wrap(type(get_algorithm('accum_logp')), 'build', 'JetSpec/tree_build')
    try:
        yield
    finally:
        for owner, attr, original in reversed(patches):
            setattr(owner, attr, original)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--engine', choices=['optimized', 'batch_reference'], required=True)
    p.add_argument('--bs', type=int, default=1)
    p.add_argument('--tokens', type=int, default=512)
    p.add_argument('--samples', type=int, default=32)
    p.add_argument('--arm', choices=['AR', 'JetSpec', 'both'], default='both')
    p.add_argument('--trace', action='store_true')
    p.add_argument('--budget', type=int, default=128)
    p.add_argument('--depth', type=int, default=15)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    meta = {'arguments': vars(args) | {'output': str(args.output)}, 'status': 'running',
            'implementation': 'upstream optimized BS=1' if args.engine == 'optimized' else
                'experimental true batched learned head + batched target; eager SDPA and dense padded KV',
            'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'gpu': None, 'results': {}}
    save(args.output.with_suffix('.progress.json'), meta)
    try:
        assert args.engine != 'optimized' or args.bs == 1
        assert '5090' in torch.cuda.get_device_name(0)
        meta['gpu'] = torch.cuda.get_device_name(0)
        torch.manual_seed(0)
        torch.set_num_threads(4)
        rows = json.loads((ROOT / 'data/jetspec_paper' / f'{args.dataset}.json').read_text())[:args.samples]
        assert rows
        prompts = [torch.tensor([r['token_ids']], device='cuda') for r in rows]
        batches = [prompts[i:i + args.bs] for i in range(0, len(prompts), args.bs)]
        arms = ['AR', 'JetSpec'] if args.arm == 'both' else [args.arm]
        eng = JetSpecEngine(os.environ['JETSPEC_MODEL'], device='cuda', dtype=torch.bfloat16,
            attn_implementation='sdpa', attn_backend='triton_paged_tree_cudagraph' if args.engine == 'optimized' else 'sdpa')
        meta['target_loaded_allocated_bytes'] = torch.cuda.memory_allocated()
        meta['target_parameter_bytes'] = sum(x.numel()*x.element_size() for x in eng.model.parameters())
        head = load_draft_head(os.environ['JETSPEC_DRAFT_HEAD'], device='cuda', dtype=torch.bfloat16,
                              attn_implementation='sdpa') if 'JetSpec' in arms else None
        meta['head_parameter_bytes'] = sum(x.numel()*x.element_size() for x in head.parameters()) if head else 0
        meta['loaded_allocated_bytes'] = torch.cuda.memory_allocated()
        meta['AR_memory_includes_unused_head'] = args.arm == 'both'
        meta['upstream_revision'] = subprocess.check_output(['git','-C',str(ROOT/'third_party/JetSpec'),
                                                            'rev-parse','HEAD'],text=True).strip()
        meta['protocol'] = {'batching': 'fixed initial batch; completed sequences drop out', 'dtype': 'bfloat16',
            'temperature': 0.0, 'thinking': False, 'EOS_stopping': True, 'tokenization_in_timer': False,
            'prefill_in_timer': True, 'compile_warmup_in_timer': False, 'detokenization_in_timer': args.engine == 'optimized',
            'samples': len(rows), 'sample_indices': [r['index'] for r in rows],
            'prompt_token_lengths': [r['prompt_tokens'] for r in rows], 'fuse_gemms': False,
            'CUDA_Graph': args.engine == 'optimized', 'mode': 'instrumented_trace' if args.trace else 'uninstrumented_benchmark'}
        rec = Recorder(False)
        current_method = ['warmup']
        if args.engine == 'batch_reference':
            ref = BatchedReference(eng.model, head, eng.eos_token_ids, rec, args.depth, 7, args.budget)
            def run(method, batch, tokens):
                return ref.generate(batch, tokens, method)
        else:
            drafter = GraphedDraftHead(head, target=eng.model, block_size=args.depth + 1,
                target_layer_ids=head.target_layer_ids, draft_shift=False) if head else None
            capacity = ((max(r['prompt_tokens'] for r in rows) + 255) // 256) * 256
            def run(method, batch, tokens):
                sp = SamplingParams(temperature=0., max_new_tokens=tokens)
                if method == 'AR':
                    out = eng.generate(batch[0], sp, session=True, session_prompt_capacity=capacity)
                else:
                    out = eng.generate_tree(batch[0], drafter, sampling_params=sp,
                        block_size=args.depth + 1, tree_depth=args.depth, tree_width=7, budget=args.budget,
                        target_layer_ids=head.target_layer_ids, return_stats=True,
                        session=True, session_prompt_capacity=capacity)
                return [out]
        print(f"Warmup {args.dataset} {args.engine} BS={args.bs} arm={args.arm}", flush=True)
        for arm in arms:
            for batch in (batches if args.engine == 'optimized' else batches[:1]):
                run(arm, batch, args.tokens if args.engine == 'optimized' else min(16,args.tokens))
        torch.cuda.synchronize()
        rec.enabled = args.trace
        context = optimized_instrumentation(rec, current_method) if args.trace and args.engine == 'optimized' else nullcontext()
        with context:
            if args.trace:
                torch.cuda.profiler.start()
            try:
                for arm in arms:
                    current_method[0] = arm
                    torch.cuda.reset_peak_memory_stats()
                    started = time.perf_counter()
                    outputs = []
                    request_ms = []
                    for i, batch in enumerate(batches):
                        t = time.perf_counter()
                        with torch.cuda.nvtx.range(f'profile_spec/{arm}/batch_{i}') if args.trace else nullcontext():
                            result = run(arm, batch, args.tokens)
                        torch.cuda.synchronize()
                        outputs.extend(result)
                        request_ms.append((time.perf_counter()-t)*1000)
                    elapsed = time.perf_counter()-started
                    ntokens = sum(len(o['token_ids']) for o in outputs)
                    acceptance = [a for o in outputs for a in o.get('accept_lengths',[])]
                    meta['results'][arm] = {'seconds':elapsed,'total_tokens':ntokens,'tokens_per_second':ntokens/elapsed,
                        'batches':len(batches),'batch_latency_ms':request_ms,
                        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                        'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
                        'mean_accept_length':sum(acceptance)/len(acceptance) if acceptance else None,
                        'rounds':len(acceptance),'outputs':outputs}
                    save(args.output.with_suffix('.progress.json'), {'status':'running','completed_arm':arm,
                        'arguments':meta['arguments']})
                    print(f"{arm}: {ntokens} tokens / {elapsed:.3f}s = {ntokens/elapsed:.1f} tok/s",flush=True)
            finally:
                if args.trace:
                    torch.cuda.profiler.stop()
        if args.trace:
            meta['phase_records'] = rec.serialize()
            # Recorder resets per-phase peaks; derive an honest global maximum from every phase.
            meta['profile_peak_allocated_bytes'] = max((r['peak_allocated'] for r in meta['phase_records']),default=0)
        if args.engine == 'batch_reference':
            meta['actual_forward_batch_shapes'] = ref.batch_shapes
        else:
            meta['graphs'] = {'AR':sorted(eng._ar_session['graph'].graphs) if eng._ar_session else [],
                'verify':[sorted(g.graphs) for g in eng._graphed_verify.values()],
                'draft':sorted(drafter.graphs) if drafter else []}
        if len(arms) == 2:
            meta['speedup'] = meta['results']['JetSpec']['tokens_per_second'] / meta['results']['AR']['tokens_per_second']
            meta['equal_token_sequences'] = sum(a['token_ids']==b['token_ids'] for a,b in
                zip(meta['results']['AR']['outputs'],meta['results']['JetSpec']['outputs']))
        meta['status'] = 'success'
    except torch.cuda.OutOfMemoryError:
        meta['status'] = 'out_of_memory'
        meta['error'] = traceback.format_exc()
    except Exception:
        meta['status'] = 'error'
        meta['error'] = traceback.format_exc()
    meta['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    save(args.output, meta)
    print(json.dumps({k:meta[k] for k in ['status','arguments']}),flush=True)
    if meta['status'] == 'error':
        print(meta['error'],flush=True)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
