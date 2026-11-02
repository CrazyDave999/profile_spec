"""Warm AR/tree sessions, audit graph reuse, and expose a CUDA/NVTX capture range.

The instrumentation adds overhead; use bench_5090.sh for throughput measurement.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
import functools
import importlib
import json
import os
from pathlib import Path

import torch
from jetspec import load_draft_head
from jetspec.engines.jetspec import JetSpecEngine, SamplingParams
from jetspec.engines.jetspec.verify.ar_graph import GraphedAR
from jetspec.engines.jetspec.verify.graph_capture import GraphedVerify
from jetspec.engines.jetspec.serving.paged_kv_cache import PagedKVCache
from jetspec.methods.jetspec.draft_head_adapter import GraphedDraftHead

ROOT = Path(__file__).resolve().parent


@contextmanager
def instrumentation(counters):
    patches = []

    def wrap(owner, attribute, label):
        original = getattr(owner, attribute)

        @functools.wraps(original)
        def traced(*args, **kwargs):
            name = label(*args) if callable(label) else label
            counters[name] += 1
            with torch.cuda.nvtx.range(name):
                return original(*args, **kwargs)

        patches.append((owner, attribute, original))
        setattr(owner, attribute, traced)

    wrap(GraphedAR, "replay_ar", "AR/graph_replay")
    wrap(GraphedVerify, "replay", "JetSpec/target_verify_graph_replay")
    wrap(GraphedVerify, "_capture_bucket",
         lambda obj, *rest: "AR/graph_capture" if isinstance(obj, GraphedAR) else "JetSpec/target_graph_capture")
    wrap(GraphedDraftHead, "__call__", "JetSpec/draft_graph_replay")
    wrap(GraphedDraftHead, "_capture_bucket", "JetSpec/draft_graph_capture")
    wrap(PagedKVCache, "gather", "JetSpec/KV_gather")
    accept = importlib.import_module("jetspec.methods.jetspec.tree._core.accept")
    wrap(accept, "gpu_tree_accept", "JetSpec/GPU_tree_accept")
    try:
        yield
    finally:
        for owner, attribute, original in reversed(patches):
            setattr(owner, attribute, original)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--budget", type=int, default=128)
    parser.add_argument("--depth", type=int, default=15)
    parser.add_argument("--capture", action="store_true", help="Use cudaProfilerStart/Stop after warmup")
    parser.add_argument("--output", type=Path, default=ROOT / "logs" / "feature_audit_5090.json")
    args = parser.parse_args()
    if args.tokens < 2 or args.repeats < 1:
        parser.error("--tokens must be >=2 and --repeats must be >=1")
    torch.manual_seed(0)
    assert "5090" in torch.cuda.get_device_name(0), "Source ~/profile_spec/env.sh first"
    engine = JetSpecEngine(os.environ["JETSPEC_MODEL"], device="cuda", dtype=torch.bfloat16,
                          attn_backend=os.environ["JETSPEC_BACKEND"], attn_implementation="sdpa")
    assert engine._use_cudagraph
    head = load_draft_head(os.environ["JETSPEC_DRAFT_HEAD"], device="cuda", dtype=torch.bfloat16,
                           attn_implementation="sdpa")
    drafter = GraphedDraftHead(head, target=engine.model, block_size=args.depth + 1,
                              target_layer_ids=head.target_layer_ids, draft_shift=False)
    prompt = engine.tokenizer.apply_chat_template(
        [{"role": "user", "content": "How many positive whole-number divisors does 196 have? Explain briefly."}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    prompt_len = len(engine.tokenizer.encode(prompt))
    capacity = ((prompt_len + 255) // 256) * 256
    sp = SamplingParams(temperature=0.0, max_new_tokens=args.tokens)
    ar_kw = dict(session=True, session_prompt_capacity=capacity)
    tree_kw = dict(block_size=args.depth + 1, tree_depth=args.depth, tree_width=7,
                   budget=args.budget, target_layer_ids=head.target_layer_ids,
                   sampling_params=sp, return_stats=True, **ar_kw)
    counters = Counter()
    outputs = {}
    with instrumentation(counters):
        print("Warming both reusable sessions before the capture range...", flush=True)
        engine.generate(prompt, sp, **ar_kw)
        engine.generate_tree(prompt, drafter, **tree_kw)
        torch.cuda.synchronize()
        warm_counts = counters.copy()
        if args.capture:
            torch.cuda.profiler.start()
        try:
            for i in range(args.repeats):
                for method, fn in [
                    ("AR", lambda: engine.generate(prompt, sp, **ar_kw)),
                    ("JetSpec", lambda: engine.generate_tree(prompt, drafter, **tree_kw)),
                ]:
                    with torch.cuda.nvtx.range(f"profile_spec/{method}/request_{i}"):
                        outputs[method] = fn()
                    torch.cuda.synchronize()
        finally:
            if args.capture:
                torch.cuda.profiler.stop()
    measured = {k: counters[k] - warm_counts[k] for k in sorted(counters)}
    assert measured.get("AR/graph_replay", 0) > 0
    assert measured.get("JetSpec/target_verify_graph_replay", 0) > 0
    assert measured.get("JetSpec/draft_graph_replay", 0) > 0
    assert all(v == 0 for k, v in measured.items() if k.endswith("graph_capture")), measured
    report = {
        "gpu": torch.cuda.get_device_name(0), "backend": engine.attn_backend,
        "baseline": "JetSpecEngine.generate: single-token compiled CUDA Graph AR with paged KV; no SGLang",
        "dtype": "bfloat16", "HF_attention": engine.resolved_attn_implementation,
        "tree_attention": "custom Triton paged tree attention",
        "target_verify_torch_compile": True, "draft_torch_compile": drafter.compiled.compile,
        "full_HF_model_torch_compile": False, "fuse_gemms": engine.fuse_gemms,
        "nogather": engine.attn_backend.endswith("_nogather"),
        "GPU_tree_accept": measured.get("JetSpec/GPU_tree_accept", 0) > 0,
        "graph_accept": False, "graph_accept_note": "The upstream graph accept path requires the nogather backend",
        "session_reuse": engine._ar_session is not None and engine._tree_session is not None,
        "AR_graph_buckets": sorted(engine._ar_session["graph"].graphs),
        "verify_graph_buckets": [sorted(g.graphs) for g in engine._graphed_verify.values()],
        "draft_graph_buckets": sorted(drafter.graphs),
        "warmup_counts": dict(warm_counts), "measured_counts": measured,
        "no_graph_recapture_in_measured_region": True,
        "tokens_per_request": args.tokens, "repeats": args.repeats,
        "depth": args.depth, "budget": args.budget,
        "equal_AR_tree_tokens": outputs["AR"]["token_ids"] == outputs["JetSpec"]["token_ids"],
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "instrumentation_overhead": "Do not use this script for headline throughput; use bench_5090.sh",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
