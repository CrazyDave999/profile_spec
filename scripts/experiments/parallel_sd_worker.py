"""One isolated SGLang method/configuration; no HTTP server or upstream edits."""
from __future__ import annotations
import _bootstrap

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import subprocess
import threading
import time
import traceback

ROOT = _bootstrap.ROOT
GPU = "GPU-eed4707e-7a00-58e0-4123-1ccbe4326b7a"

if os.environ.get("PROFILE_SPEC_NVTX") == "1":
    from parallel_sd_nvtx import enable
    enable()

if os.environ.get("PROFILE_SPEC_ROUTING") == "1":
    from moe_sd_routing import enable as enable_routing_labels
    enable_routing_labels()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str))
    temporary.replace(path)


class DeviceSampler:
    """Physical device usage, not torch's live/reserved tensor accounting."""
    def __init__(self):
        import pynvml
        self.nvml = pynvml
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByUUID(GPU)
        self.samples = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        while not self.stop.is_set():
            m = self.nvml.nvmlDeviceGetMemoryInfo(self.handle)
            self.samples.append({"monotonic": time.monotonic(), "used_bytes": m.used})
            self.stop.wait(.05)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["ar", "dflash", "domino", "dspark", "dflash2"], required=True)
    parser.add_argument("--target", choices=["qwen3-8b", "qwen3.5-4b", "gpt-oss-20b"], default="qwen3-8b")
    parser.add_argument("--dataset", choices=["gsm8k", "humaneval", "mt_bench"], default="gsm8k")
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--block", type=int)
    parser.add_argument("--extra-context", type=int, default=0, help="Synthetic token-prefix length, inserted into the existing user turn.")
    parser.add_argument("--fp32-lm-head", action="store_true", help="Separate numerical diagnostic; never pool with default BF16-head timings.")
    parser.add_argument("--logprobs", action="store_true", help="Diagnostic output; not headline timing.")
    parser.add_argument("--ragged-mode", choices=["static", "compact"], default="static")
    parser.add_argument("--sps-table", type=Path)
    parser.add_argument("--attention", default="fa4")
    parser.add_argument("--moe-runner-backend", choices=["flashinfer_mxfp4", "marlin", "triton_kernel"])
    parser.add_argument("--routing-record", action="store_true", help="Separate diagnostic: upstream per-pass expert records; never headline timing.")
    parser.add_argument("--eos", action="store_true", help="Default fixed-output isolates resource scaling; this arm enables EOS.")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ['SGLANG_RAGGED_VERIFY_MODE'] = args.ragged_mode
    if args.ragged_mode != 'static' and (args.method != 'dspark' or args.sps_table is None):
        raise ValueError("Scheduled DSpark requires an explicit locally profiled SPS table.")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != GPU:
        raise RuntimeError("Bind CUDA_VISIBLE_DEVICES to the authorized RTX 5090 UUID.")
    assert 1 <= args.bs <= 16 and 1 <= args.samples <= 32
    import pynvml
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByUUID(GPU)
    active = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
    if active:
        raise RuntimeError(f"GPU admission rejected: live compute PIDs {[p.pid for p in active]}")
    import sglang

    prompt_file = ROOT / f"data/jetspec_paper/{args.dataset}.json"
    prompts = json.loads(prompt_file.read_text())[:args.samples]
    target_path = Path.home()/"models"/{"qwen3-8b":"Qwen3-8B", "qwen3.5-4b":"Qwen3.5-4B", "gpt-oss-20b":"gpt-oss-20b"}[args.target]
    inputs = [p["token_ids"] for p in prompts]
    frozen_inputs_path = None
    if args.target == "gpt-oss-20b":
        frozen_inputs_path = ROOT/f"data/moe_sd_20261106/{args.dataset}.json"
        frozen_inputs = json.loads(frozen_inputs_path.read_text())
        assert frozen_inputs['source_sha256'] == hashlib.sha256(prompt_file.read_bytes()).hexdigest()
        selected = frozen_inputs['prompts'][:args.samples]
        assert len(selected) == len(prompts)
        assert all(record['raw_prompt_sha256'] == hashlib.sha256(prompt['raw_prompt'].encode()).hexdigest() for record,prompt in zip(selected,prompts))
        inputs = [record['input_ids'] for record in selected]
        if args.method not in ["ar", "dflash"]:
            raise ValueError("Only a matched DFlash checkpoint has been prepared for this target.")
    if args.target == "qwen3.5-4b":
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(target_path))
        inputs = [tokenizer.apply_chat_template([{"role":"user","content":p['raw_prompt']}], tokenize=True, add_generation_prompt=True, return_dict=False, enable_thinking=False) for p in prompts]
        assert all(isinstance(ids,list) and all(isinstance(token,int) for token in ids) for ids in inputs)
        if args.method not in ["ar", "dflash"]:
            raise ValueError("Only a matched DFlash checkpoint has been prepared for this target.")
    if args.extra_context:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(target_path))
        unit = tokenizer.encode("Background reference text. This material is unrelated to the final question.\n", add_special_tokens=False)
        extra = (unit * ((args.extra_context+len(unit)-1)//len(unit)))[:args.extra_context]
        assert args.target == "qwen3-8b" and all(p[:3] == [151644, 872, 198] for p in inputs)
        inputs = [p[:3] + extra + p[3:] for p in inputs]
    assert max(map(len, inputs)) + args.tokens < 4096
    engine_args = dict(
        model_path=str(target_path), dtype="bfloat16", tp_size=1,
        attention_backend=args.attention, mem_fraction_static=.80,
        max_total_tokens=32768, context_length=4096, max_running_requests=args.bs,
        chunked_prefill_size=1024, disable_radix_cache=True,
        cuda_graph_max_bs_decode=16, cuda_graph_bs_decode=[1, 2, 4, 8, 16],
        cuda_graph_backend_decode="full", cuda_graph_backend_prefill="disabled",
        random_seed=42, log_level="info", decode_log_interval=50,
        enable_metrics=True, enable_layerwise_nvtx_marker=args.profile,
        enable_fp32_lm_head=args.fp32_lm_head,
    )
    if args.target == "qwen3.5-4b":
        # SM120 hybrid GDN restricts the full-attention backend as well as the
        # linear backend. Keep FA4 for the separate draft runner only.
        engine_args.update(attention_backend="triton", linear_attn_prefill_backend="triton", linear_attn_decode_backend="triton")
    if args.moe_runner_backend:
        assert args.target == "gpt-oss-20b"
        engine_args["moe_runner_backend"] = args.moe_runner_backend
    if args.routing_record:
        assert args.target == "gpt-oss-20b" and not args.profile
        engine_args.update(expert_distribution_recorder_mode="per_pass", expert_distribution_recorder_buffer_size=-1)
        routing_directory = ROOT/"profiles/moe_sd_20261106"/(args.output.stem+"_routing")
        routing_directory.mkdir(parents=True, exist_ok=True)
        os.environ["SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR"] = str(routing_directory)
    if args.method != "ar":
        if args.target == "gpt-oss-20b":
            model = json.loads((ROOT/"models-moe-sd.json").read_text())["checkpoints"][args.method]
        else:
            manifest = json.loads((ROOT/"models-parallel-sd.json").read_text())
            model = manifest["qwen35_dflash" if args.target == "qwen3.5-4b" else args.method]
        if model["format"] not in ("native", "native DFlash"):
            raise RuntimeError("Convert and validate this checkpoint before claiming native support.")
        engine_args.update(speculative_algorithm="DSPARK" if args.method == "dspark" else "DFLASH",
            speculative_draft_model_path=model["alias"], speculative_draft_attention_backend=args.attention)
        if args.block is not None:
            engine_args["speculative_num_draft_tokens"] = args.block
        if args.sps_table is not None:
            engine_args["speculative_dspark_sps_table_path"] = str(args.sps_table.resolve())
    sampler = DeviceSampler(); sampler.thread.start()
    versions = {name: importlib.metadata.version(name) for name in ["torch", "transformers", "triton", "sglang", "flashinfer-python", "nvidia-cuda-nvcc", "nvidia-cuda-runtime"]}
    data = {"status": "starting", "arguments": vars(args), "engine_requested": engine_args, "package_versions": versions,
        "worker_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "toolkit_header_sha256": hashlib.sha256((Path(os.environ['CUDA_HOME'])/'include/crt/math_functions.h').read_bytes()).hexdigest(),
        "feature_environment": {key:os.environ.get(key) for key in ["SGLANG_CACHE_DIR", "SGLANG_RAGGED_VERIFY_MODE", "PROFILE_SPEC_NVTX"]},
        "GPU_UUID": GPU, "sglang_revision": subprocess.check_output(["git", "-C", str(ROOT/"third_party/SGLang"), "rev-parse", "HEAD"], text=True).strip(),
        "prompt_file_sha256": hashlib.sha256(prompt_file.read_bytes()).hexdigest(),
        "selected_prompt_sha256": hashlib.sha256(json.dumps(inputs).encode()).hexdigest(),
        "input_lengths": list(map(len, inputs)), "protocol": "offline Engine; fixed input IDs; no prefix reuse; load/compile/warmup excluded; request wall throughput; profiling separate; fixed output unless --eos", "results": []}
    if frozen_inputs_path is not None:
        data['frozen_inputs_file'] = str(frozen_inputs_path)
        data['frozen_inputs_file_sha256'] = hashlib.sha256(frozen_inputs_path.read_bytes()).hexdigest()
        data['template_date'] = frozen_inputs['template_date']
    save(args.output, data)
    engine = None
    try:
        started = time.perf_counter()
        engine = sglang.Engine(**engine_args)
        data["startup_seconds"] = time.perf_counter()-started
        data["engine_resolved_before"] = engine.get_server_info()
        sampling = {"temperature": 0, "max_new_tokens": args.tokens, "ignore_eos": not args.eos}
        diagnostics = {"return_logprob":True,"top_logprobs_num":5} if args.logprobs else {}
        engine.generate(input_ids=inputs[:args.bs], sampling_params={**sampling, "max_new_tokens": min(32, args.tokens)}, **diagnostics)
        engine.flush_cache()
        if args.routing_record:
            engine.start_expert_distribution_record()
        data["status"] = "running"; save(args.output, data)
        if args.profile:
            engine.start_profile(activities=["CUDA_PROFILER"], profile_id=args.output.stem)
        for repeat in range(args.repeats):
            outputs = []; wave_seconds = []; start = time.perf_counter()
            for offset in range(0, len(inputs), args.bs):
                tick = time.perf_counter()
                out = engine.generate(input_ids=inputs[offset:offset+args.bs], sampling_params=sampling, **diagnostics)
                wave_seconds.append(time.perf_counter()-tick)
                outputs.extend(out if isinstance(out, list) else [out])
            seconds = time.perf_counter()-start
            tokens = sum(o["meta_info"]["completion_tokens"] for o in outputs)
            verify = sum(o["meta_info"].get("spec_verify_ct", 0) for o in outputs)
            data["results"].append({"repeat": repeat, "seconds": seconds, "output_tokens": tokens,
                "tokens_per_second": tokens/seconds, "sum_request_verify_steps": verify,
                "committed_tokens_per_request_verify": tokens/verify if verify else None,
                "effective_decode_tokens_per_verify_excluding_prefill_seed": (tokens-len(outputs))/verify if verify else None,
                "wave_seconds": wave_seconds, "outputs": outputs})
            save(args.output, data)
            engine.flush_cache()
        if args.profile:
            engine.stop_profile()
        if args.routing_record:
            engine.stop_expert_distribution_record()
            engine.dump_expert_distribution_record()
            data["expert_distribution_files"] = [str(p) for p in sorted(routing_directory.glob("*.pt"))]
            data["protocol"] += "; expert recording diagnostic: timing excluded from headline results"
        data["engine_resolved_after"] = engine.get_server_info()
        rates = [r["tokens_per_second"] for r in data["results"]]
        data["median_tokens_per_second"] = statistics.median(rates)
        data["status"] = "success"
    except BaseException:
        data["status"] = "failed"; data["error"] = traceback.format_exc()
        raise
    finally:
        if engine is not None:
            engine.shutdown()
        sampler.stop.set(); sampler.thread.join(timeout=2)
        data["physical_device_peak_used_bytes"] = max((s["used_bytes"] for s in sampler.samples), default=None)
        data["device_memory_samples"] = sampler.samples
        save(args.output, data)
    print(json.dumps({k: data.get(k) for k in ["status", "median_tokens_per_second", "startup_seconds", "physical_device_peak_used_bytes"]}), flush=True)


if __name__ == "__main__":
    main()
