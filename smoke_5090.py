"""Run real JetSpec inference and compare the output with greedy AR."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import time

import torch
from jetspec import load_draft_head
from jetspec.engines.jetspec import JetSpecEngine, SamplingParams
from jetspec.methods.jetspec.draft_head_adapter import DraftHeadTreeDrafter, GraphedDraftHead


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default=os.environ.get("JETSPEC_BACKEND", "triton_paged_tree"))
    parser.add_argument("--drafter", choices=["eager", "graphed"], default="graphed")
    parser.add_argument("--budget", type=int, default=128)
    parser.add_argument("--depth", type=int, default=15)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--prompt", default="How many positive whole-number divisors does 196 have? Explain briefly.")
    parser.add_argument("--output", default="/home/qinghao/profile_spec/logs/smoke_5090.json")
    args = parser.parse_args()
    torch.manual_seed(0)
    assert torch.cuda.is_available()
    name = torch.cuda.get_device_name(0)
    assert "5090" in name, f"Expected RTX 5090, got {name}; source ~/profile_spec/env.sh"
    torch.cuda.reset_peak_memory_stats()
    print(f"GPU: {name}; backend={args.backend}; drafter={args.drafter}", flush=True)
    engine = JetSpecEngine(os.environ["JETSPEC_MODEL"], device="cuda", dtype=torch.bfloat16,
                          attn_implementation="sdpa", attn_backend=args.backend)
    head = load_draft_head(os.environ["JETSPEC_DRAFT_HEAD"], device="cuda", dtype=torch.bfloat16,
                           attn_implementation="sdpa")
    cls = GraphedDraftHead if args.drafter == "graphed" else DraftHeadTreeDrafter
    drafter = cls(head, target=engine.model, block_size=args.depth + 1,
                  target_layer_ids=head.target_layer_ids, draft_shift=False)
    prompt = engine.tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    sp = SamplingParams(temperature=0.0, max_new_tokens=args.max_tokens)
    kw = dict(block_size=args.depth + 1, tree_depth=args.depth, tree_width=7,
              budget=args.budget, target_layer_ids=head.target_layer_ids,
              sampling_params=sp, return_stats=True)

    # Exclude first compilation and graph capture from the measured calls.
    print("Warming AR and tree inference...", flush=True)
    engine.generate(prompt, sp)
    engine.generate_tree(prompt, drafter, **kw)
    results = {}
    for method, fn in [
        ("ar", lambda: engine.generate(prompt, sp)),
        ("jetspec", lambda: engine.generate_tree(prompt, drafter, **kw)),
    ]:
        torch.cuda.synchronize()
        started = time.perf_counter()
        output = fn()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        token_ids = output["token_ids"]
        if torch.is_tensor(token_ids):
            token_ids = token_ids.detach().cpu().flatten().tolist()
        results[method] = {"tokens": len(token_ids), "seconds": elapsed,
                           "tokens_per_second": len(token_ids) / elapsed,
                           "token_ids": token_ids, "text": output["text"]}
        for key in ("tpf", "rounds", "accept_lengths"):
            if key in output:
                results[method][key] = output[key]
        print(f"{method}: {len(token_ids)} tokens, {elapsed:.3f}s, {len(token_ids)/elapsed:.1f} tok/s", flush=True)
    ar_ids, tree_ids = results["ar"]["token_ids"], results["jetspec"]["token_ids"]
    mismatch = next((i for i, (a, b) in enumerate(zip(ar_ids, tree_ids)) if a != b), None)
    report = {
        "gpu": name, "torch_cuda": torch.version.cuda,
        "versions": {p: importlib.metadata.version(p) for p in ("jetspec", "torch", "transformers", "triton")},
        "git_revision": subprocess.check_output(["git", "-C", "/home/qinghao/profile_spec/third_party/JetSpec", "rev-parse", "HEAD"], text=True).strip(),
        "arguments": vars(args), "model": os.environ["JETSPEC_MODEL"],
        "draft": os.environ["JETSPEC_DRAFT_HEAD"], "dtype": "bfloat16",
        "enable_thinking": False, "warmup": "one full AR and one full tree generation",
        "session": False, "fuse_gemms": os.environ.get("JETSPEC_FUSE_GEMMS", "0"),
        "equal_token_ids": ar_ids == tree_ids, "first_mismatch_index": mismatch,
        "speedup": results["jetspec"]["tokens_per_second"] / results["ar"]["tokens_per_second"],
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "results": results,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"AR token equality: {report['equal_token_ids']}; peak allocation {report['peak_allocated_gib']:.2f} GiB", flush=True)
    print(results["jetspec"]["text"], flush=True)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
