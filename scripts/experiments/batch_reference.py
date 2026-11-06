"""True batched JetSpec learned drafting and verification for profiling.

Experimental dense padded KV implementation, not the optimized upstream BS=1
engine. Every live sequence participates in ONE draft-head forward and ONE
target verify forward per round. No stub drafter or sequential-request BS proxy.
The same target/SDPA/KV-packing implementation is used for the AR comparison.
"""
import _bootstrap
from contextlib import contextmanager
from dataclasses import dataclass
import time

import torch
from transformers import DynamicCache
from jetspec.core.model_runner import ModelRunner
from jetspec.methods.jetspec.tree import get_algorithm, build_ancestor_matrix, gpu_tree_accept


class Recorder:
    def __init__(self, enabled=False):
        self.enabled = enabled
        self.records = []

    @contextmanager
    def span(self, name, **metadata):
        if not self.enabled:
            yield
            return
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        t = time.perf_counter()
        with torch.cuda.nvtx.range(name):
            start.record()
            try:
                yield
            finally:
                end.record()
                self.records.append({"phase": name, "host_ms": (time.perf_counter() - t) * 1000,
                    "allocated_before": before, "allocated_after": torch.cuda.memory_allocated(),
                    "peak_allocated": torch.cuda.max_memory_allocated(),
                    "reserved_after": torch.cuda.memory_reserved(), "metadata": metadata,
                    "events": (start, end)})

    def serialize(self):
        torch.cuda.synchronize()
        result = []
        for item in self.records:
            row = dict(item)
            start, end = row.pop("events")
            row["gpu_event_span_ms"] = start.elapsed_time(end)
            result.append(row)
        return result


@dataclass
class State:
    prompt_length: int
    tokens: list
    kv: list
    hidden: torch.Tensor | None
    acceptance: list


class BatchedReference:
    def __init__(self, model, head, eos_ids, recorder=None, depth=15, width=7, budget=128):
        self.model, self.head = model, head
        self.eos = set(eos_ids)
        self.runner = ModelRunner(model)
        self.rec = recorder or Recorder()
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        self.depth, self.width, self.budget = depth, width, budget
        self.layer_ids = head.target_layer_ids if head is not None else []
        self.algo = get_algorithm("accum_logp")
        self.batch_shapes = []

    @torch.inference_mode()
    def prefill(self, prompts, method):
        states = []
        for ids in prompts:
            ids = ids.to(self.device)
            with self.rec.span(f"{method}/prefill", prompt_tokens=ids.numel()):
                logits, cache, hidden = self.runner.forward(ids, DynamicCache(),
                    torch.arange(ids.shape[1], device=self.device)[None],
                    output_hidden_states=method == "JetSpec", target_layer_ids=self.layer_ids if method == "JetSpec" else None,
                    last_position_logits_only=True)
                token = int(logits[0, -1].argmax().item())
                kv = [(layer.keys, layer.values) for layer in cache.layers]
                states.append(State(ids.numel(), [token], kv, hidden, []))
        return states

    def _active(self, states, max_tokens):
        return [i for i, s in enumerate(states) if len(s.tokens) < max_tokens and s.tokens[-1] not in self.eos]

    def _pack(self, states, active, method):
        past = [states[i].kv[0][0].shape[2] for i in active]
        maximum = max(past)
        template = states[active[0]].kv[0][0]
        b, h, d = len(active), template.shape[1], template.shape[-1]
        cache = DynamicCache()
        with self.rec.span(f"{method}/kv_pack", batch=b, max_past=maximum):
            for layer in range(len(states[active[0]].kv)):
                keys = torch.zeros((b, h, maximum, d), device=self.device, dtype=self.dtype)
                values = torch.zeros_like(keys)
                for row, sid in enumerate(active):
                    keys[row, :, :past[row]] = states[sid].kv[layer][0][0]
                    values[row, :, :past[row]] = states[sid].kv[layer][1][0]
                cache.update(keys, values, layer)
        return cache, past, maximum

    def draft_logits(self, states, active):
        b, block = len(active), self.depth + 1
        lens = [states[sid].hidden.shape[1] for sid in active]
        cap = max(lens)
        features = states[active[0]].hidden.shape[-1]
        with self.rec.span("JetSpec/draft_prepare", batch=b, context=cap):
            hidden = torch.zeros((b, cap, features), dtype=self.dtype, device=self.device)
            positions = torch.zeros((b, cap + block), dtype=torch.long, device=self.device)
            noise = torch.full((b, block), self.head.mask_token_id, dtype=torch.long, device=self.device)
            mask = torch.full((b, 1, block, cap + block), torch.finfo(self.dtype).min,
                              device=self.device, dtype=self.dtype)
            causal = torch.tril(torch.ones((block, block), device=self.device, dtype=torch.bool))
            for row, sid in enumerate(active):
                n = lens[row]
                hidden[row, :n] = states[sid].hidden[0]
                positions[row, :n] = torch.arange(n, device=self.device)
                positions[row, n:cap] = max(0, n - 1)
                positions[row, cap:] = torch.arange(n, n + block, device=self.device)
                noise[row, 0] = states[sid].tokens[-1]
                mask[row, :, :, :n] = 0
            mask[:, :, :, cap:] = torch.where(causal, 0., torch.finfo(self.dtype).min)
            embeddings = self.model.model.embed_tokens(noise)
        with self.rec.span("JetSpec/draft_forward", batch=b, context=cap):
            out = self.head(target_hidden=hidden, noise_embedding=embeddings,
                position_ids=positions, attention_mask=mask, past_key_values=None,
                use_cache=False, is_causal=False)
            logits = self.model.lm_head(out[:, 1:, :])
        return logits

    @torch.inference_mode()
    def generate(self, prompts, max_tokens=128, method="JetSpec"):
        assert method in ("AR", "JetSpec")
        states = self.prefill(prompts, method)
        while active := self._active(states, max_tokens):
            b = len(active)
            if method == "JetSpec":
                logits = self.draft_logits(states, active)
                with self.rec.span("JetSpec/tree_build", batch=b):
                    trees = [self.algo.build(states[sid].tokens[-1], logits[row:row+1],
                        self.depth + 1, self.width, self.budget, self.device)
                        for row, sid in enumerate(active)]
                    ancestors = [build_ancestor_matrix(t).bool() for t in trees]
                del logits
                n = max(t.num_nodes for t in trees)
            else:
                trees, n = None, 1
            cache, past, maximum = self._pack(states, active, method)
            with self.rec.span(f"{method}/target_prepare", batch=b, query=n):
                ids = torch.zeros((b, n), dtype=torch.long, device=self.device)
                positions = torch.zeros_like(ids)
                mask = torch.full((b, 1, n, maximum + n), torch.finfo(self.dtype).min,
                                  dtype=self.dtype, device=self.device)
                for row, sid in enumerate(active):
                    mask[row, :, :, :past[row]] = 0
                    if trees is None:
                        ids[row, 0] = states[sid].tokens[-1]
                        positions[row, 0] = past[row]
                        mask[row, :, :, maximum:] = 0
                    else:
                        tree = trees[row]
                        ids[row, :tree.num_nodes] = tree.token_ids
                        positions[row, :tree.num_nodes] = past[row] + tree.depth
                        mask[row, 0, :tree.num_nodes, maximum:maximum + tree.num_nodes] = torch.where(
                            ancestors[row], 0., torch.finfo(self.dtype).min)
                cache_positions = torch.arange(maximum, maximum + n, device=self.device)
            phase = "JetSpec/verify_forward" if trees is not None else "AR/decode_forward"
            with self.rec.span(phase, batch=b, query=n, max_past=maximum):
                logits, cache, hidden = self.runner.forward(ids, cache, positions,
                    attention_mask=mask, cache_position=cache_positions,
                    output_hidden_states=trees is not None,
                    target_layer_ids=self.layer_ids if trees is not None else None)
            self.batch_shapes.append({"method": method, "batch": b, "query": n, "max_past": maximum,
                                      "past_lengths": list(past),
                                      "query_lengths": [t.num_nodes for t in trees] if trees else [1] * b})
            with self.rec.span(f"{method}/accept", batch=b):
                decisions = []
                for row, sid in enumerate(active):
                    if trees is None:
                        path = torch.tensor([0], device=self.device)
                        block = [int(logits[row, -1].argmax().item())]
                    else:
                        t = trees[row]
                        path, accepted, correction = gpu_tree_accept(t.token_ids,
                            logits[row, :t.num_nodes].argmax(-1), t.parent_indices, t.depth,
                            max_depth=self.depth)
                        block = t.token_ids[path[1:]].tolist() + [int(correction.item())]
                        states[sid].acceptance.append(len(block))
                    decisions.append((path, block))
            with self.rec.span(f"{method}/kv_commit", batch=b):
                for row, sid in enumerate(active):
                    path, block = decisions[row]
                    columns = torch.cat([torch.arange(past[row], device=self.device), maximum + path])
                    states[sid].kv = [(layer.keys[row:row+1, :, columns, :],
                                       layer.values[row:row+1, :, columns, :]) for layer in cache.layers]
                    if trees is not None:
                        states[sid].hidden = torch.cat([states[sid].hidden, hidden[row:row+1, path]], dim=1)
                    for token in block:
                        states[sid].tokens.append(token)
                        if token in self.eos:
                            break
            del cache, logits, hidden, mask
        return [{"token_ids": s.tokens[:max_tokens], "accept_lengths": s.acceptance,
                 "prompt_tokens": s.prompt_length} for s in states]
