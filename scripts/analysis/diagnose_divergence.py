"""Capture target logits at the first eager AR/tree divergence on one prompt."""
import _bootstrap
import json
import importlib
import os
from pathlib import Path

import torch
from jetspec import load_draft_head, DraftHeadTreeDrafter
from jetspec.engines.jetspec import JetSpecEngine, SamplingParams
accept_api = importlib.import_module("jetspec.methods.jetspec.tree._core.accept")


def main():
    torch.manual_seed(0)
    engine = JetSpecEngine(os.environ["JETSPEC_MODEL"], device="cuda", dtype=torch.bfloat16,
                          attn_implementation="sdpa", attn_backend="triton_paged_tree")
    head = load_draft_head(os.environ["JETSPEC_DRAFT_HEAD"], device="cuda", dtype=torch.bfloat16)
    drafter = DraftHeadTreeDrafter(head, target=engine.model, block_size=16,
                                  target_layer_ids=head.target_layer_ids, draft_shift=False)
    prompt = engine.tokenizer.apply_chat_template(
        [{"role": "user", "content": "How many positive whole-number divisors does 196 have? Explain briefly."}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    sp = SamplingParams(temperature=0.0, max_new_tokens=64)

    def top(logits):
        vals, ids = logits.float().topk(8)
        return [{"id": i, "text": engine.tokenizer.decode([i]), "logit": v}
                for v, i in zip(vals.tolist(), ids.tolist())]

    ar_logits = []
    hook = engine.model.register_forward_hook(lambda mod, ins, out: ar_logits.append(out.logits[0, -1].detach().float().cpu()))
    ar = engine.generate(prompt, sp)
    hook.remove()
    ar_ids = ar["token_ids"]
    generated = 1
    divergence = None
    original = accept_api.gpu_tree_accept
    latest = []
    hook = engine.model.register_forward_hook(lambda mod, ins, out: latest.__setitem__(slice(None), [out.logits.detach()]))

    def traced(tree_tokens, greedy_targets, parent_indices, depths, max_depth=None):
        nonlocal generated, divergence
        result = original(tree_tokens, greedy_targets, parent_indices, depths, max_depth=max_depth)
        path, accepted, correction = result
        path = path.tolist()
        logits = latest[0]
        round_ids = tree_tokens[path[1:]].tolist() + [int(correction.item())]
        for j, token in enumerate(round_ids):
            idx = generated + j
            if idx >= len(ar_ids):
                break
            if token != ar_ids[idx] and divergence is None:
                a = ar_logits[idx]
                b = logits[0, path[j]].detach().float().cpu()
                divergence = {
                    "token_index_zero_based": idx, "ar_token": ar_ids[idx], "tree_token": token,
                    "ar_top8": top(a), "tree_top8": top(b),
                    "ar_logit_margin_for_diverging_tokens": float(a[ar_ids[idx]] - a[token]),
                    "tree_logit_margin_for_diverging_tokens": float(b[ar_ids[idx]] - b[token]),
                    "max_abs_logit_difference": float((a - b).abs().max()),
                }
        generated += len(round_ids)
        return result

    accept_api.gpu_tree_accept = traced
    try:
        result = engine.generate_tree(prompt, drafter, block_size=16, tree_depth=15,
                                      tree_width=7, budget=128, target_layer_ids=head.target_layer_ids,
                                      sampling_params=sp, return_stats=True)
    finally:
        accept_api.gpu_tree_accept = original
        hook.remove()
    if ar_ids != result["token_ids"]:
        assert divergence is not None, "Divergence tracing did not capture the mismatched prediction"
    report = {"backend": "triton_paged_tree", "dtype": "bfloat16",
              "equal_token_ids": ar_ids == result["token_ids"], "first_divergence": divergence}
    Path('/home/qinghao/profile_spec/logs/divergence_5090.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
