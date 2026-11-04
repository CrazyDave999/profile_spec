from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Optional
from urllib.request import urlretrieve


CACHE_DIR = Path(__file__).resolve().parent / "cache"
SUPPORTED_WORKLOADS = ("gsm8k", "math500", "humaneval", "mbpp", "mt-bench")
DEFAULT_WORKLOADS = "gsm8k"
DEFAULT_TIMEOUT_S = 3600
DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH = 600
SERVER_SHUTDOWN_DRAIN_TIMEOUT_S = 30.0
SERVER_SHUTDOWN_TIMEOUT_S = 120.0
GSM8K_TEST_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
MT_BENCH_QUESTION_URL = (
    "https://raw.githubusercontent.com/lm-sys/FastChat/main/"
    "fastchat/llm_judge/data/mt_bench/question.jsonl"
)


@dataclass(frozen=True)
class SharedServerConfig:
    tp_size: int = 8
    attention_backend: str = "trtllm_mha"
    dtype: str = "bfloat16"
    max_running_requests: int = 32
    cuda_graph_max_bs: int = 32
    mem_fraction_static: Optional[float] = 0.8
    page_size: Optional[int] = None
    mamba_scheduler_strategy: str = "extra_buffer"
    mamba_ssm_dtype: str = "bfloat16"
    linear_attn_backend: str = "flashinfer"
    enable_piecewise_cuda_graph: bool = True
    enable_flashinfer_allreduce_fusion: bool = True

    def to_args(self) -> list[str]:
        args = [
            "--trust-remote-code",
            "--attention-backend",
            self.attention_backend,
            "--tp-size",
            str(self.tp_size),
            "--dtype",
            self.dtype,
            "--max-running-requests",
            str(self.max_running_requests),
            "--cuda-graph-max-bs-decode",
            str(self.cuda_graph_max_bs),
            "--mamba-scheduler-strategy",
            self.mamba_scheduler_strategy,
            "--mamba-ssm-dtype",
            self.mamba_ssm_dtype,
            "--linear-attn-backend",
            self.linear_attn_backend,
            "--cuda-graph-backend-prefill",
            "tc_piecewise" if self.enable_piecewise_cuda_graph else "disabled",
        ]
        if self.enable_flashinfer_allreduce_fusion:
            args.append("--enable-flashinfer-allreduce-fusion")
        if self.mem_fraction_static is not None:
            args.extend(["--mem-fraction-static", str(self.mem_fraction_static)])
        if self.page_size is not None:
            args.extend(["--page-size", str(int(self.page_size))])
        return args

    def summary_label(self) -> str:
        return (
            f"tp:{self.tp_size},attention:{self.attention_backend},"
            f"dtype:{self.dtype},max_running:{self.max_running_requests},"
            f"cuda_graph_max_bs_decode:{self.cuda_graph_max_bs},"
            f"mem_fraction:{self.mem_fraction_static},page_size:{self.page_size}"
        )


BASE_SHARED_SERVER_CONFIG = SharedServerConfig()


@dataclass(frozen=True)
class MTPConfig:
    num_steps: int
    eagle_topk: int = 1

    @property
    def mode_key(self) -> str:
        return f"mtp_s{self.num_steps}"

    @property
    def display_name(self) -> str:
        return f"MTP steps={self.num_steps}"

    @property
    def expect_spec(self) -> bool:
        return True

    @property
    def num_draft_tokens(self) -> int:
        return self.num_steps + 1

    def to_args(self) -> list[str]:
        return [
            "--speculative-algorithm",
            "EAGLE",
            "--speculative-num-steps",
            str(self.num_steps),
            "--speculative-eagle-topk",
            str(self.eagle_topk),
            "--speculative-num-draft-tokens",
            str(self.num_draft_tokens),
        ]


@dataclass(frozen=True)
class DFlashConfig:
    draft_model: str
    block_size: Optional[int] = None
    draft_attention_backend: str = "fa4"

    @property
    def mode_key(self) -> str:
        if self.block_size is None:
            return "dflash"
        return f"dflash_b{self.block_size}"

    @property
    def display_name(self) -> str:
        if self.block_size is None:
            return "DFLASH"
        return f"DFLASH block={self.block_size}"

    @property
    def expect_spec(self) -> bool:
        return True

    def to_args(self) -> list[str]:
        args = [
            "--speculative-algorithm",
            "DFLASH",
            "--speculative-draft-model-path",
            self.draft_model,
            "--speculative-draft-attention-backend",
            self.draft_attention_backend,
        ]
        if self.block_size is not None:
            args.extend(["--speculative-dflash-block-size", str(int(self.block_size))])
        return args


@dataclass(frozen=True)
class BaselineConfig:
    @property
    def mode_key(self) -> str:
        return "baseline"

    @property
    def display_name(self) -> str:
        return "Baseline"

    @property
    def expect_spec(self) -> bool:
        return False

    def to_args(self) -> list[str]:
        return []


@dataclass(frozen=True)
class ServerDeployment:
    shared_config: SharedServerConfig
    mode_config: BaselineConfig | MTPConfig | DFlashConfig

    @property
    def mode_key(self) -> str:
        return self.mode_config.mode_key

    @property
    def display_name(self) -> str:
        return self.mode_config.display_name

    @property
    def expect_spec(self) -> bool:
        return self.mode_config.expect_spec

    @property
    def server_args(self) -> list[str]:
        return [*self.shared_config.to_args(), *self.mode_config.to_args()]

    @property
    def mtp_num_steps(self) -> Optional[int]:
        if isinstance(self.mode_config, MTPConfig):
            return self.mode_config.num_steps
        return None

    @property
    def dflash_block_size(self) -> Optional[int]:
        if isinstance(self.mode_config, DFlashConfig):
            return self.mode_config.block_size
        return None

    @property
    def enable_overlap_plan_stream(self) -> bool:
        # MTP on Qwen3.5 uses CUDA HybridLinearAttnBackend, which does not
        # implement update_verify_buffers_to_fill_after_draft for overlap plan
        # streams. DFlash has its own compatible planning path.
        return isinstance(self.mode_config, DFlashConfig)


@dataclass(frozen=True)
class DeploymentSweep:
    include_baseline: bool
    spec_modes: tuple[str, ...]
    mtp_num_steps: tuple[int, ...]
    dflash_draft_model: Optional[str]
    dflash_block_sizes: tuple[Optional[int], ...]

    @property
    def mode_keys(self) -> list[str]:
        mode_keys: list[str] = []
        for spec_mode in self.spec_modes:
            if spec_mode == "mtp":
                mode_keys.extend(f"mtp_s{int(steps)}" for steps in self.mtp_num_steps)
            elif spec_mode == "dflash":
                for block_size in self.dflash_block_sizes:
                    mode_keys.append(
                        DFlashConfig("", block_size=block_size).mode_key
                    )
            else:
                mode_keys.append(spec_mode)
        return mode_keys


@dataclass(frozen=True)
class SamplingConfig:
    enable_thinking: bool
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int


@dataclass(frozen=True)
class BenchmarkMethodologyConfig:
    num_samples: Optional[int]
    min_generation_turns_per_config: int
    min_warmup_generation_turns: int
    runs_per_config: int
    timeout_s: int = DEFAULT_TIMEOUT_S
    server_shutdown_drain_timeout_s: float = SERVER_SHUTDOWN_DRAIN_TIMEOUT_S
    server_shutdown_timeout_s: float = SERVER_SHUTDOWN_TIMEOUT_S


@dataclass(frozen=True)
class SweepConfig:
    target_model: str
    dflash_draft_model: Optional[str]
    workloads: tuple[str, ...]
    concurrencies: tuple[int, ...]
    sampling: SamplingConfig
    methodology: BenchmarkMethodologyConfig
    deployment_sweep: DeploymentSweep
    csv_output: Optional[str]


@dataclass(frozen=True)
class RunKey:
    workload: str
    backend: str
    tp: int
    concurrency: int
    mode: str

    def metric_key(self) -> tuple[str, int, int, str]:
        return (self.backend, self.tp, self.concurrency, self.mode)


@dataclass(frozen=True)
class BenchmarkPlan:
    measured_samples: list[list[str]]
    warmup_samples: list[list[str]]
    warmdown_samples: list[list[str]]

    @property
    def measured_sample_count(self) -> int:
        return len(self.measured_samples)

    @property
    def measured_generation_turn_count(self) -> int:
        return _generation_turn_count(self.measured_samples)

    @property
    def warmup_generation_turn_count(self) -> int:
        return _generation_turn_count(self.warmup_samples)

    @property
    def warmdown_generation_turn_count(self) -> int:
        return _generation_turn_count(self.warmdown_samples)


@dataclass(frozen=True)
class BenchmarkJob:
    target_model: str
    workload: str
    deployment: ServerDeployment
    concurrency: int
    run_index: int
    sampling: SamplingConfig
    methodology: BenchmarkMethodologyConfig

    @property
    def key(self) -> RunKey:
        shared_config = self.deployment.shared_config
        return RunKey(
            workload=self.workload,
            backend=shared_config.attention_backend,
            tp=shared_config.tp_size,
            concurrency=self.concurrency,
            mode=self.deployment.mode_key,
        )

    @property
    def label(self) -> str:
        key = self.key
        return (
            f"workload={key.workload} backend={key.backend} tp={key.tp} "
            f"conc={key.concurrency} ({self.deployment.display_name})"
        )

    @property
    def run_label(self) -> str:
        return f"run={self.run_index + 1}/{self.methodology.runs_per_config}"


def _parse_int_csv(value: str) -> list[int]:
    return [int(x) for x in value.split(",") if x.strip()]


def _parse_optional_int_csv(value: str) -> list[Optional[int]]:
    values: list[Optional[int]] = []
    for raw in value.split(","):
        item = raw.strip().lower()
        if not item:
            continue
        if item in ("default", "none"):
            values.append(None)
        else:
            values.append(int(item))
    return values or [None]


def _parse_str_csv(value: str) -> list[str]:
    return [x.strip().lower() for x in value.split(",") if x.strip()]


def _duplicate_values(values: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return duplicates


def _shared_server_config_to_payload(config: SharedServerConfig) -> dict[str, Any]:
    return {
        "tp_size": config.tp_size,
        "attention_backend": config.attention_backend,
        "dtype": config.dtype,
        "max_running_requests": config.max_running_requests,
        "cuda_graph_max_bs": config.cuda_graph_max_bs,
        "mem_fraction_static": config.mem_fraction_static,
        "page_size": config.page_size,
        "mamba_scheduler_strategy": config.mamba_scheduler_strategy,
        "mamba_ssm_dtype": config.mamba_ssm_dtype,
        "linear_attn_backend": config.linear_attn_backend,
        "enable_piecewise_cuda_graph": config.enable_piecewise_cuda_graph,
        "enable_flashinfer_allreduce_fusion": (
            config.enable_flashinfer_allreduce_fusion
        ),
    }


def _shared_server_config_from_payload(payload: dict[str, Any]) -> SharedServerConfig:
    return SharedServerConfig(
        tp_size=int(payload["tp_size"]),
        attention_backend=str(payload["attention_backend"]),
        dtype=str(payload["dtype"]),
        max_running_requests=int(payload["max_running_requests"]),
        cuda_graph_max_bs=int(payload["cuda_graph_max_bs"]),
        mem_fraction_static=payload.get("mem_fraction_static"),
        page_size=payload.get("page_size"),
        mamba_scheduler_strategy=str(payload["mamba_scheduler_strategy"]),
        mamba_ssm_dtype=str(payload["mamba_ssm_dtype"]),
        linear_attn_backend=str(payload["linear_attn_backend"]),
        enable_piecewise_cuda_graph=bool(payload["enable_piecewise_cuda_graph"]),
        enable_flashinfer_allreduce_fusion=bool(
            payload["enable_flashinfer_allreduce_fusion"]
        ),
    )


def _mode_config_to_payload(
    config: BaselineConfig | MTPConfig | DFlashConfig,
) -> dict[str, Any]:
    if isinstance(config, BaselineConfig):
        return {"kind": "baseline"}
    if isinstance(config, MTPConfig):
        return {
            "kind": "mtp",
            "num_steps": config.num_steps,
            "eagle_topk": config.eagle_topk,
        }
    if isinstance(config, DFlashConfig):
        return {
            "kind": "dflash",
            "draft_model": config.draft_model,
            "block_size": config.block_size,
            "draft_attention_backend": config.draft_attention_backend,
        }
    raise TypeError(f"Unsupported mode config type: {type(config).__name__}")


def _mode_config_from_payload(
    payload: dict[str, Any],
) -> BaselineConfig | MTPConfig | DFlashConfig:
    kind = payload["kind"]
    if kind == "baseline":
        return BaselineConfig()
    if kind == "mtp":
        return MTPConfig(
            num_steps=int(payload["num_steps"]),
            eagle_topk=int(payload.get("eagle_topk", 1)),
        )
    if kind == "dflash":
        return DFlashConfig(
            draft_model=str(payload["draft_model"]),
            block_size=payload.get("block_size"),
            draft_attention_backend=str(
                payload.get("draft_attention_backend", "fa4")
            ),
        )
    raise ValueError(f"Unsupported mode config kind: {kind}")


def _deployment_to_payload(deployment: ServerDeployment) -> dict[str, Any]:
    return {
        "shared_config": _shared_server_config_to_payload(deployment.shared_config),
        "mode_config": _mode_config_to_payload(deployment.mode_config),
    }


def _deployment_from_payload(payload: dict[str, Any]) -> ServerDeployment:
    return ServerDeployment(
        shared_config=_shared_server_config_from_payload(payload["shared_config"]),
        mode_config=_mode_config_from_payload(payload["mode_config"]),
    )


def _sampling_config_to_payload(config: SamplingConfig) -> dict[str, Any]:
    return {
        "enable_thinking": config.enable_thinking,
        "max_new_tokens": config.max_new_tokens,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "top_k": config.top_k,
    }


def _sampling_config_from_payload(payload: dict[str, Any]) -> SamplingConfig:
    return SamplingConfig(
        enable_thinking=bool(payload["enable_thinking"]),
        max_new_tokens=int(payload["max_new_tokens"]),
        temperature=float(payload["temperature"]),
        top_p=float(payload["top_p"]),
        top_k=int(payload["top_k"]),
    )


def _methodology_to_payload(config: BenchmarkMethodologyConfig) -> dict[str, Any]:
    return {
        "num_samples": config.num_samples,
        "min_generation_turns_per_config": (
            config.min_generation_turns_per_config
        ),
        "min_warmup_generation_turns": config.min_warmup_generation_turns,
        "runs_per_config": config.runs_per_config,
        "timeout_s": config.timeout_s,
        "server_shutdown_drain_timeout_s": (
            config.server_shutdown_drain_timeout_s
        ),
        "server_shutdown_timeout_s": config.server_shutdown_timeout_s,
    }


def _methodology_from_payload(
    payload: dict[str, Any],
) -> BenchmarkMethodologyConfig:
    return BenchmarkMethodologyConfig(
        num_samples=payload.get("num_samples"),
        min_generation_turns_per_config=int(
            payload["min_generation_turns_per_config"]
        ),
        min_warmup_generation_turns=int(payload["min_warmup_generation_turns"]),
        runs_per_config=int(payload["runs_per_config"]),
        timeout_s=int(payload["timeout_s"]),
        server_shutdown_drain_timeout_s=float(
            payload["server_shutdown_drain_timeout_s"]
        ),
        server_shutdown_timeout_s=float(payload["server_shutdown_timeout_s"]),
    )


def benchmark_job_to_payload(job: BenchmarkJob) -> dict[str, Any]:
    return {
        "target_model": job.target_model,
        "workload": job.workload,
        "deployment": _deployment_to_payload(job.deployment),
        "concurrency": job.concurrency,
        "run_index": job.run_index,
        "sampling": _sampling_config_to_payload(job.sampling),
        "methodology": _methodology_to_payload(job.methodology),
    }


def benchmark_job_from_payload(payload: dict[str, Any]) -> BenchmarkJob:
    return BenchmarkJob(
        target_model=str(payload["target_model"]),
        workload=str(payload["workload"]),
        deployment=_deployment_from_payload(payload["deployment"]),
        concurrency=int(payload["concurrency"]),
        run_index=int(payload["run_index"]),
        sampling=_sampling_config_from_payload(payload["sampling"]),
        methodology=_methodology_from_payload(payload["methodology"]),
    )


def _parse_workload_selection(value: str) -> list[str]:
    values = _parse_str_csv(value)
    if values == ["all"]:
        return list(SUPPORTED_WORKLOADS)
    unknown = sorted(set(values) - set(SUPPORTED_WORKLOADS))
    if unknown:
        raise ValueError(
            f"Unknown workloads: {','.join(unknown)}. Supported: "
            f"{','.join(SUPPORTED_WORKLOADS)} or all."
        )
    if not values:
        raise ValueError("--workloads must include at least one workload.")
    duplicates = _duplicate_values(values)
    if duplicates:
        raise ValueError(f"Duplicate workloads: {','.join(duplicates)}.")
    return values


def _filter_attention_backends(backends: list[str], *, device_sm: int) -> list[str]:
    if not (80 <= device_sm <= 90):
        backends = [b for b in backends if b != "fa3"]
    if device_sm < 100:
        backends = [b for b in backends if b not in ("fa4", "trtllm_mha")]
    return backends or ["flashinfer"]


def _read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def _download_to_cache(url: str, filename: str) -> Path:
    CACHE_DIR.mkdir(exist_ok=True)
    out_path = CACHE_DIR / filename
    if out_path.exists():
        return out_path

    tmp_path = out_path.with_name(f"{out_path.name}.{os.getpid()}.tmp")
    print(f"[download] {url}")
    urlretrieve(url, tmp_path)
    os.replace(tmp_path, out_path)
    return out_path


def _load_gsm8k_user_prompts() -> list[str]:
    path = _download_to_cache(GSM8K_TEST_URL, "gsm8k_test.jsonl")
    if not path.is_file():
        raise RuntimeError(f"GSM8K data file does not exist: {path}")

    prompts: list[str] = []
    for row in _read_jsonl(path):
        prompts.append(
            row["question"]
            + "\nPlease reason step by step, and put your final answer within \\boxed{}."
        )
    return prompts


def _load_math500_user_prompts() -> list[str]:
    rows = _load_hf_dataset_rows("HuggingFaceH4/MATH-500", split="test")

    prompts: list[str] = []
    for row in rows:
        prompts.append(
            row["problem"]
            + "\nPlease reason step by step, and put your final answer within \\boxed{}."
        )
    return prompts


def _load_hf_dataset_rows(*load_args, **load_kwargs) -> list[dict]:
    from datasets import load_dataset

    return list(load_dataset(*load_args, **load_kwargs))


def _load_humaneval_user_prompts() -> list[str]:
    rows = _load_hf_dataset_rows("openai/openai_humaneval", split="test")

    return [row["prompt"] for row in rows]


def _load_mbpp_user_prompts() -> list[str]:
    rows = _load_hf_dataset_rows(
        "google-research-datasets/mbpp", "sanitized", split="test"
    )

    return [row["prompt"] for row in rows]


def _load_mt_bench_user_turns() -> list[list[str]]:
    path = _download_to_cache(MT_BENCH_QUESTION_URL, "mt_bench_question.jsonl")
    if not path.is_file():
        raise RuntimeError(f"MT-bench data file does not exist: {path}")
    rows = _read_jsonl(path)

    prompts: list[list[str]] = []
    for row in rows:
        turns = row.get("turns", row.get("prompt"))
        if not isinstance(turns, list):
            raise RuntimeError(
                "MT-bench rows must contain a list-valued `turns` or `prompt` field."
            )
        turns = [str(turn) for turn in turns[:2]]
        if len(turns) != 2:
            raise RuntimeError(
                f"MT-bench rows must contain exactly two turns; got {len(turns)}."
            )
        prompts.append(turns)
    return prompts


def _load_user_turns(workload: str) -> list[list[str]]:
    if workload == "gsm8k":
        return [[prompt] for prompt in _load_gsm8k_user_prompts()]
    if workload == "math500":
        return [[prompt] for prompt in _load_math500_user_prompts()]
    if workload == "humaneval":
        return [[prompt] for prompt in _load_humaneval_user_prompts()]
    if workload == "mbpp":
        return [[prompt] for prompt in _load_mbpp_user_prompts()]
    if workload == "mt-bench":
        return _load_mt_bench_user_turns()
    raise ValueError(f"Unknown workload: {workload}")


def _flush_cache(
    base_url: str, timeout_s: float = SERVER_SHUTDOWN_DRAIN_TIMEOUT_S
) -> None:
    import requests

    try:
        requests.get(
            base_url + "/flush_cache",
            params={"timeout": float(timeout_s)},
            timeout=max(float(timeout_s) + 5.0, 10.0),
        ).raise_for_status()
    except Exception as exc:
        raise RuntimeError(
            "Failed to flush cache before the next benchmark phase; "
            "SGLang still had pending requests after waiting for drain."
        ) from exc


def _flush_cache_best_effort(base_url: str, timeout_s: float) -> None:
    import requests

    try:
        requests.get(
            base_url + "/flush_cache",
            params={"timeout": float(timeout_s)},
            timeout=max(float(timeout_s) + 5.0, 10.0),
        ).raise_for_status()
    except Exception as exc:
        print(f"[shutdown] /flush_cache failed before server shutdown: {exc}")


def _shutdown_server(proc, base_url: str, *, drain_timeout_s: float, kill_timeout_s: float) -> None:
    from sglang.srt.utils import kill_process_tree

    if proc.poll() is not None:
        return

    _flush_cache_best_effort(base_url, drain_timeout_s)

    if proc.poll() is not None:
        return

    print(f"[shutdown] sending SIGTERM to server pid={proc.pid}")
    proc.terminate()
    try:
        proc.wait(timeout=float(kill_timeout_s))
        return
    except Exception:
        print(
            f"[shutdown] server pid={proc.pid} did not exit within "
            f"{kill_timeout_s}s; falling back to kill_process_tree."
        )

    kill_process_tree(proc.pid, wait_timeout=30)


def _send_generate(
    base_url: str,
    text: str,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    timeout_s: int,
) -> dict:
    import requests

    sampling_params: dict = {
        "temperature": float(temperature),
        "top_p": float(top_p),
        "top_k": int(top_k),
        "max_new_tokens": int(max_new_tokens),
    }
    resp = requests.post(
        base_url + "/generate",
        json={
            "text": text,
            "sampling_params": sampling_params,
        },
        timeout=int(timeout_s),
    )
    resp.raise_for_status()
    out = resp.json()
    if isinstance(out, list):
        raise RuntimeError(
            "Expected an object response for single /generate, but got "
            f"type={type(out).__name__}."
        )
    return out


@dataclass(frozen=True)
class BenchMetrics:
    sample_count: int
    generation_turn_count: int
    latency_s: float
    output_tokens: int
    output_toks_per_s: float
    spec_accept_length: Optional[float]
    spec_verify_ct_sum: int


@dataclass(frozen=True)
class JobResult:
    key: RunKey
    deployment: ServerDeployment
    source_sample_count: int
    source_generation_turn_count: int
    warmup_generation_turn_count: int
    warmdown_generation_turn_count: int
    run_index: int
    metrics: BenchMetrics


@dataclass(frozen=True)
class JobFailure:
    key: RunKey
    deployment: ServerDeployment
    run_index: int
    error_type: str
    error_message: str


@dataclass(frozen=True)
class ConfigResult:
    key: RunKey
    deployment: ServerDeployment
    source_sample_count: Optional[int]
    source_generation_turn_count: Optional[int]
    warmup_generation_turn_count: Optional[int]
    warmdown_generation_turn_count: Optional[int]
    metrics: Optional[BenchMetrics]
    repeat_metrics: tuple[BenchMetrics, ...]
    successful_run_indices: tuple[int, ...]
    failures: tuple[JobFailure, ...]

    @property
    def run_count(self) -> int:
        return self.successful_run_count + self.failed_run_count

    @property
    def successful_run_count(self) -> int:
        return len(self.repeat_metrics)

    @property
    def failed_run_count(self) -> int:
        return len(self.failures)

    @property
    def status(self) -> str:
        if self.failed_run_count == 0:
            return "ok"
        if self.successful_run_count == 0:
            return "failed"
        return "partial_failed"


def _run_key_to_payload(key: RunKey) -> dict[str, Any]:
    return {
        "workload": key.workload,
        "backend": key.backend,
        "tp": key.tp,
        "concurrency": key.concurrency,
        "mode": key.mode,
    }


def _run_key_from_payload(payload: dict[str, Any]) -> RunKey:
    return RunKey(
        workload=str(payload["workload"]),
        backend=str(payload["backend"]),
        tp=int(payload["tp"]),
        concurrency=int(payload["concurrency"]),
        mode=str(payload["mode"]),
    )


def _bench_metrics_to_payload(metrics: BenchMetrics) -> dict[str, Any]:
    return {
        "sample_count": metrics.sample_count,
        "generation_turn_count": metrics.generation_turn_count,
        "latency_s": metrics.latency_s,
        "output_tokens": metrics.output_tokens,
        "output_toks_per_s": metrics.output_toks_per_s,
        "spec_accept_length": metrics.spec_accept_length,
        "spec_verify_ct_sum": metrics.spec_verify_ct_sum,
    }


def _bench_metrics_from_payload(payload: dict[str, Any]) -> BenchMetrics:
    return BenchMetrics(
        sample_count=int(payload["sample_count"]),
        generation_turn_count=int(payload["generation_turn_count"]),
        latency_s=float(payload["latency_s"]),
        output_tokens=int(payload["output_tokens"]),
        output_toks_per_s=float(payload["output_toks_per_s"]),
        spec_accept_length=payload.get("spec_accept_length"),
        spec_verify_ct_sum=int(payload["spec_verify_ct_sum"]),
    )


def job_outcome_to_payload(result: JobResult | JobFailure) -> dict[str, Any]:
    if isinstance(result, JobFailure):
        return {
            "kind": "failure",
            "key": _run_key_to_payload(result.key),
            "deployment": _deployment_to_payload(result.deployment),
            "run_index": result.run_index,
            "error_type": result.error_type,
            "error_message": result.error_message,
        }
    return {
        "kind": "result",
        "key": _run_key_to_payload(result.key),
        "deployment": _deployment_to_payload(result.deployment),
        "source_sample_count": result.source_sample_count,
        "source_generation_turn_count": result.source_generation_turn_count,
        "warmup_generation_turn_count": result.warmup_generation_turn_count,
        "warmdown_generation_turn_count": result.warmdown_generation_turn_count,
        "run_index": result.run_index,
        "metrics": _bench_metrics_to_payload(result.metrics),
    }


def job_outcome_from_payload(payload: dict[str, Any]) -> JobResult | JobFailure:
    kind = payload["kind"]
    if kind == "failure":
        return JobFailure(
            key=_run_key_from_payload(payload["key"]),
            deployment=_deployment_from_payload(payload["deployment"]),
            run_index=int(payload["run_index"]),
            error_type=str(payload["error_type"]),
            error_message=str(payload["error_message"]),
        )
    if kind == "result":
        return JobResult(
            key=_run_key_from_payload(payload["key"]),
            deployment=_deployment_from_payload(payload["deployment"]),
            source_sample_count=int(payload["source_sample_count"]),
            source_generation_turn_count=int(
                payload["source_generation_turn_count"]
            ),
            warmup_generation_turn_count=int(
                payload["warmup_generation_turn_count"]
            ),
            warmdown_generation_turn_count=int(
                payload["warmdown_generation_turn_count"]
            ),
            run_index=int(payload["run_index"]),
            metrics=_bench_metrics_from_payload(payload["metrics"]),
        )
    raise ValueError(f"Unsupported job outcome kind: {kind}")


@dataclass(frozen=True)
class SampleMetrics:
    generation_turn_count: int
    output_tokens: int
    spec_verify_ct_sum: int
    spec_accept_lengths: tuple[float, ...]


def _extract_generated_text(out: dict) -> str:
    text = out.get("text")
    if isinstance(text, str):
        return text
    if isinstance(text, list) and len(text) == 1 and isinstance(text[0], str):
        return text[0]
    raise RuntimeError(
        "Expected /generate response to include generated text in `text`; "
        f"got keys={sorted(out.keys())}."
    )


def _extract_generate_stats(out: dict) -> tuple[int, int, Optional[float]]:
    meta = out.get("meta_info", {}) or {}
    output_tokens = int(meta.get("completion_tokens", 0))
    spec_verify_ct = int(meta.get("spec_verify_ct", 0))
    spec_accept_length = None
    if "spec_accept_length" in meta:
        try:
            spec_accept_length = float(meta["spec_accept_length"])
        except (TypeError, ValueError):
            pass
    return output_tokens, spec_verify_ct, spec_accept_length


def _run_sample(
    base_url: str,
    *,
    turns: list[str],
    tokenizer,
    sampling: SamplingConfig,
    timeout_s: int,
) -> SampleMetrics:
    messages: list[dict[str, str]] = []
    total_tokens = 0
    spec_verify_ct_sum = 0
    turn_accept_lengths: list[float] = []

    for turn_idx, user_content in enumerate(turns):
        messages.append({"role": "user", "content": user_content})
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=bool(sampling.enable_thinking),
        )
        out = _send_generate(
            base_url=base_url,
            text=prompt,
            max_new_tokens=sampling.max_new_tokens,
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            top_k=sampling.top_k,
            timeout_s=timeout_s,
        )
        output_tokens, spec_verify_ct, turn_accept_length = _extract_generate_stats(out)
        total_tokens += output_tokens
        spec_verify_ct_sum += spec_verify_ct
        if spec_verify_ct > 0:
            turn_accept_lengths.append(float(output_tokens) / float(spec_verify_ct))
        elif turn_accept_length is not None:
            turn_accept_lengths.append(turn_accept_length)

        if turn_idx + 1 < len(turns):
            messages.append({"role": "assistant", "content": _extract_generated_text(out)})

    return SampleMetrics(
        generation_turn_count=len(turns),
        output_tokens=int(total_tokens),
        spec_verify_ct_sum=int(spec_verify_ct_sum),
        spec_accept_lengths=tuple(turn_accept_lengths),
    )


def _run_unmeasured_requests(
    base_url: str,
    *,
    samples: list[list[str]],
    tokenizer,
    sampling: SamplingConfig,
    concurrency: int,
    timeout_s: int,
) -> None:
    if not samples:
        return

    with ThreadPoolExecutor(max_workers=int(concurrency)) as pool:
        futures = [
            pool.submit(
                _run_sample,
                base_url=base_url,
                turns=turns,
                tokenizer=tokenizer,
                sampling=sampling,
                timeout_s=timeout_s,
            )
            for turns in samples
        ]
        for fut in as_completed(futures):
            fut.result()


def _take_samples(samples: list[list[str]], *, start: int, count: int) -> list[list[str]]:
    if count <= 0:
        return []
    if not samples:
        raise RuntimeError("Cannot take benchmark samples from an empty workload.")
    return [samples[(start + i) % len(samples)] for i in range(count)]


def _generation_turn_count(samples: list[list[str]]) -> int:
    return sum(len(turns) for turns in samples)


def _take_samples_for_min_generation_turns(
    samples: list[list[str]], *, start: int, min_generation_turns: int
) -> list[list[str]]:
    if min_generation_turns <= 0:
        return []
    if not samples:
        raise RuntimeError("Cannot take benchmark samples from an empty workload.")

    out: list[list[str]] = []
    generation_turns = 0
    idx = 0
    while generation_turns < int(min_generation_turns):
        sample = samples[(start + idx) % len(samples)]
        out.append(sample)
        generation_turns += len(sample)
        idx += 1
    return out


def _build_measured_samples(
    samples: list[list[str]], *, num_samples: Optional[int], min_generation_turns: int
) -> list[list[str]]:
    if not samples:
        raise RuntimeError("Cannot build measured samples from an empty workload.")
    if num_samples is not None:
        if num_samples <= 0:
            raise RuntimeError(f"--num-samples must be > 0, got {num_samples}.")
        return _take_samples(samples, start=0, count=int(num_samples))
    if min_generation_turns < 0:
        raise RuntimeError(
            "--min-generation-turns-per-config must be >= 0, "
            f"got {min_generation_turns}."
        )
    source_generation_turns = _generation_turn_count(samples)
    if source_generation_turns <= 0:
        raise RuntimeError("Cannot build measured samples with zero generation turns.")
    repeats = max(1, math.ceil(int(min_generation_turns) / source_generation_turns))
    return samples * repeats


def _build_measured_samples_for_concurrency(
    samples: list[list[str]],
    *,
    num_samples: Optional[int],
    min_generation_turns: int,
    concurrency: int,
) -> list[list[str]]:
    # Concurrency 1 is the stable accept-length pass; use one full workload by
    # default instead of cache-favorable repeated copies.
    if num_samples is None and int(concurrency) == 1:
        return samples
    return _build_measured_samples(
        samples,
        num_samples=num_samples,
        min_generation_turns=min_generation_turns,
    )


def _build_benchmark_plan(
    samples: list[list[str]],
    *,
    concurrency: int,
    methodology: BenchmarkMethodologyConfig,
) -> BenchmarkPlan:
    measured_samples = _build_measured_samples_for_concurrency(
        samples,
        num_samples=methodology.num_samples,
        min_generation_turns=int(methodology.min_generation_turns_per_config),
        concurrency=int(concurrency),
    )
    warmup_min_generation_turns = max(
        int(methodology.min_warmup_generation_turns), 2 * int(concurrency)
    )
    warmup_samples = _take_samples_for_min_generation_turns(
        measured_samples,
        start=0,
        min_generation_turns=warmup_min_generation_turns,
    )
    warmdown_samples = _take_samples(
        measured_samples,
        start=len(warmup_samples),
        count=int(concurrency),
    )
    return BenchmarkPlan(
        measured_samples=measured_samples,
        warmup_samples=warmup_samples,
        warmdown_samples=warmdown_samples,
    )


def _run_requests(
    base_url: str,
    *,
    samples: list[list[str]],
    warmdown_samples: list[list[str]],
    tokenizer,
    sampling: SamplingConfig,
    concurrency: int,
    timeout_s: int,
    expect_spec: bool,
) -> BenchMetrics:
    start = time.perf_counter()
    total_tokens = 0
    spec_verify_ct_sum = 0
    generation_turn_count = 0
    turn_accept_lengths: list[float] = []
    measured_completed = 0
    latency: Optional[float] = None

    with ThreadPoolExecutor(max_workers=int(concurrency)) as pool:
        measured_futures = [
            pool.submit(
                _run_sample,
                base_url=base_url,
                turns=turns,
                tokenizer=tokenizer,
                sampling=sampling,
                timeout_s=timeout_s,
            )
            for turns in samples
        ]
        measured_future_set = set(measured_futures)
        # Queue warmdown behind measured work so the server does not immediately
        # drain to idle as the measured tail completes. These futures are waited
        # on for correctness, but excluded from timing and metrics.
        warmdown_futures = [
            pool.submit(
                _run_sample,
                base_url=base_url,
                turns=turns,
                tokenizer=tokenizer,
                sampling=sampling,
                timeout_s=timeout_s,
            )
            for turns in warmdown_samples
        ]
        consumed_warmdown_futures = set()

        for fut in as_completed([*measured_futures, *warmdown_futures]):
            if fut in measured_future_set:
                sample_metrics = fut.result()
                total_tokens += sample_metrics.output_tokens
                spec_verify_ct_sum += sample_metrics.spec_verify_ct_sum
                generation_turn_count += sample_metrics.generation_turn_count
                turn_accept_lengths.extend(sample_metrics.spec_accept_lengths)
                measured_completed += 1
                if measured_completed == len(measured_futures):
                    latency = time.perf_counter() - start
                    break
            else:
                consumed_warmdown_futures.add(fut)
                fut.result()

        for fut in warmdown_futures:
            if fut not in consumed_warmdown_futures:
                fut.result()

    if latency is None:
        latency = time.perf_counter() - start
    toks_per_s = total_tokens / max(latency, 1e-6)

    if expect_spec and spec_verify_ct_sum <= 0:
        raise RuntimeError(
            "Speculative decoding sanity check failed: did not observe any "
            "`spec_verify_ct` in responses (speculative decoding may not have been enabled)."
        )

    spec_accept_length = (
        float(statistics.mean(turn_accept_lengths))
        if turn_accept_lengths
        else None
    )

    return BenchMetrics(
        sample_count=len(samples),
        generation_turn_count=int(generation_turn_count),
        latency_s=float(latency),
        output_tokens=int(total_tokens),
        output_toks_per_s=float(toks_per_s),
        spec_accept_length=spec_accept_length,
        spec_verify_ct_sum=int(spec_verify_ct_sum),
    )


def _format_table(
    *,
    tp_sizes: list[int],
    concurrencies: list[int],
    values: dict[tuple[int, int], Optional[float]],
    float_fmt: str,
) -> str:
    header = ["tp\\conc"] + [str(c) for c in concurrencies]
    rows: list[list[str]] = [header]
    for tp in tp_sizes:
        row = [str(tp)]
        for c in concurrencies:
            v = values.get((tp, c), None)
            row.append("N/A" if v is None else format(v, float_fmt))
        rows.append(row)

    col_widths = [
        max(len(row[col_idx]) for row in rows) for col_idx in range(len(rows[0]))
    ]

    lines: list[str] = []
    lines.append("  ".join(cell.rjust(col_widths[i]) for i, cell in enumerate(rows[0])))
    lines.append("  ".join("-" * w for w in col_widths))
    for row in rows[1:]:
        lines.append("  ".join(cell.rjust(col_widths[i]) for i, cell in enumerate(row)))
    return "\n".join(lines)


def _build_shared_server_configs(
    *, device_sm: int, visible_gpus: int, max_concurrency: int
) -> list[SharedServerConfig]:
    attention_backends = _filter_attention_backends(
        [BASE_SHARED_SERVER_CONFIG.attention_backend], device_sm=device_sm
    )
    scheduler_capacity = max(
        BASE_SHARED_SERVER_CONFIG.max_running_requests,
        int(max_concurrency),
    )
    configs = [
        replace(
            BASE_SHARED_SERVER_CONFIG,
            attention_backend=backend,
            max_running_requests=scheduler_capacity,
            cuda_graph_max_bs=max(
                BASE_SHARED_SERVER_CONFIG.cuda_graph_max_bs,
                scheduler_capacity,
            ),
        )
        for backend in attention_backends
    ]
    runnable_configs = [
        config for config in configs if 1 <= config.tp_size <= visible_gpus
    ]
    if not runnable_configs:
        raise RuntimeError(
            f"No shared server configs are runnable with visible_gpus={visible_gpus}. "
            "Set CUDA_VISIBLE_DEVICES accordingly."
        )
    return runnable_configs


def _build_deployments(
    shared_config: SharedServerConfig, sweep: DeploymentSweep
) -> list[ServerDeployment]:
    deployments: list[ServerDeployment] = []
    if sweep.include_baseline:
        deployments.append(
            ServerDeployment(
                shared_config=shared_config,
                mode_config=BaselineConfig(),
            )
        )

    for spec_mode in sweep.spec_modes:
        if spec_mode == "mtp":
            for mtp_num_steps in sweep.mtp_num_steps:
                mtp_config = MTPConfig(num_steps=int(mtp_num_steps))
                deployments.append(
                    ServerDeployment(
                        shared_config=shared_config,
                        mode_config=mtp_config,
                    )
                )
        elif spec_mode == "dflash":
            if sweep.dflash_draft_model is None:
                raise RuntimeError("DFlash deployment requires a draft model.")
            for block_size in sweep.dflash_block_sizes:
                dflash_config = DFlashConfig(
                    draft_model=sweep.dflash_draft_model,
                    block_size=block_size,
                )
                deployments.append(
                    ServerDeployment(
                        shared_config=shared_config,
                        mode_config=dflash_config,
                    )
                )
        else:
            raise ValueError(f"Unknown speculative mode: {spec_mode}")
    return deployments


def _build_benchmark_jobs(
    config: SweepConfig, shared_configs: list[SharedServerConfig]
) -> list[BenchmarkJob]:
    jobs: list[BenchmarkJob] = []
    for shared_config in shared_configs:
        deployments = _build_deployments(shared_config, config.deployment_sweep)
        for deployment in deployments:
            for workload in config.workloads:
                for concurrency in config.concurrencies:
                    concurrency_deployment = replace(
                        deployment,
                        shared_config=replace(
                            deployment.shared_config,
                            max_running_requests=int(concurrency),
                            cuda_graph_max_bs=int(concurrency),
                        ),
                    )
                    for run_index in range(config.methodology.runs_per_config):
                        jobs.append(
                            BenchmarkJob(
                                target_model=config.target_model,
                                workload=workload,
                                deployment=concurrency_deployment,
                                concurrency=concurrency,
                                run_index=run_index,
                                sampling=config.sampling,
                                methodology=config.methodology,
                            )
                        )
    return jobs


def _mode_display_name(mode: str) -> str:
    if mode.startswith("mtp_s"):
        return f"MTP steps={mode.removeprefix('mtp_s')}"
    if mode.startswith("dflash_b"):
        return f"DFLASH block={mode.removeprefix('dflash_b')}"
    return {
        "baseline": "Baseline",
        "dflash": "DFLASH",
    }.get(mode, mode)


def _collect_metric(
    *,
    results: dict[tuple[str, int, int, str], BenchMetrics],
    backend: str,
    tp_sizes: list[int],
    concurrencies: list[int],
    mode: str,
    field: str,
) -> dict[tuple[int, int], Optional[float]]:
    out: dict[tuple[int, int], Optional[float]] = {}
    for tp in tp_sizes:
        for conc in concurrencies:
            metrics = results.get((backend, tp, conc, mode), None)
            out[(tp, conc)] = None if metrics is None else getattr(metrics, field)
    return out


def _compute_speedup(
    baseline: dict[tuple[int, int], Optional[float]],
    speculative: dict[tuple[int, int], Optional[float]],
) -> dict[tuple[int, int], Optional[float]]:
    return {
        key: None if (b is None or d is None or b <= 0) else (d / b)
        for key, b in baseline.items()
        for d in [speculative.get(key, None)]
    }


def _metric_map_from_config_results(
    config_results: list[ConfigResult],
) -> dict[tuple[str, int, int, str], BenchMetrics]:
    return {
        result.key.metric_key(): result.metrics
        for result in config_results
        if result.metrics is not None
    }


def _print_kv_lines(items: list[tuple[str, object]]) -> None:
    for key, value in items:
        print(f"{key}={value}")


def _print_failure_summary(config_results: list[ConfigResult]) -> None:
    failed_results = [
        result for result in config_results if result.failed_run_count > 0
    ]
    if not failed_results:
        return

    print("\n=== Failed/Partial Runs ===")
    for result in failed_results:
        key = result.key
        print(
            f"workload={key.workload} backend={key.backend} tp={key.tp} "
            f"mode={key.mode} conc={key.concurrency} status={result.status} "
            f"successful_runs={result.successful_run_count} "
            f"failed_runs={result.failed_run_count} "
            f"successful_run_numbers={_format_successful_run_numbers(result)} "
            f"failed_run_numbers={_format_failed_run_numbers(result.failures)} "
            f"errors={_format_failure_messages(result.failures)}"
        )


def _server_env_for_job(job: BenchmarkJob) -> dict[str, str]:
    return {
        "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": (
            "1" if job.deployment.enable_overlap_plan_stream else "0"
        ),
        "SGLANG_PYSPY_DUMP_BEFORE_CRASH": "0",
        "SGLANG_CUDA_COREDUMP_BEFORE_CRASH": "0",
    }


def _run_benchmark_job(job: BenchmarkJob) -> JobResult:
    from sglang.test.test_utils import (
        DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH as SGLANG_DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        find_available_port,
        popen_launch_server,
    )
    from transformers import AutoTokenizer

    key = job.key
    print(f"\n=== {job.label} {job.run_label} ===")
    samples = _load_user_turns(job.workload)
    if not samples:
        raise RuntimeError(f"Workload '{job.workload}' did not produce any prompts.")

    source_sample_count = len(samples)
    source_generation_turn_count = _generation_turn_count(samples)
    plan = _build_benchmark_plan(
        samples,
        concurrency=job.concurrency,
        methodology=job.methodology,
    )
    if plan.measured_sample_count > source_sample_count:
        print(
            "[config] measured sample count exceeds workload size; "
            "repeating whole workload copies with radix cache enabled."
        )

    base_url = f"http://127.0.0.1:{find_available_port(20000)}"
    tokenizer = AutoTokenizer.from_pretrained(job.target_model)
    server_start_timeout_s = int(
        max(SGLANG_DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH, job.methodology.timeout_s)
    )
    server_env = _server_env_for_job(job)
    print(
        "server_env="
        + ",".join(f"{key}:{value}" for key, value in sorted(server_env.items()))
    )
    proc = popen_launch_server(
        job.target_model,
        base_url,
        timeout=server_start_timeout_s,
        other_args=job.deployment.server_args,
        env=server_env,
    )
    try:
        _send_generate(
            base_url,
            "Hello",
            max_new_tokens=8,
            temperature=job.sampling.temperature,
            top_p=job.sampling.top_p,
            top_k=job.sampling.top_k,
            timeout_s=min(job.methodology.timeout_s, 300),
        )

        _flush_cache(base_url)
        print(
            f"[warmup {job.run_label}] run {len(plan.warmup_samples)} samples / "
            f"{plan.warmup_generation_turn_count} generation turns after "
            "/flush_cache; excluded from metrics."
        )
        _run_unmeasured_requests(
            base_url,
            samples=plan.warmup_samples,
            tokenizer=tokenizer,
            sampling=job.sampling,
            concurrency=job.concurrency,
            timeout_s=job.methodology.timeout_s,
        )
        _flush_cache(base_url)
        print(
            f"[warmup {job.run_label}] flushed cache after warmup; "
            "starting measured workload."
        )
        metrics = _run_requests(
            base_url,
            samples=plan.measured_samples,
            warmdown_samples=plan.warmdown_samples,
            tokenizer=tokenizer,
            sampling=job.sampling,
            concurrency=job.concurrency,
            timeout_s=job.methodology.timeout_s,
            expect_spec=job.deployment.expect_spec,
        )
        line = (
            f"[{job.label} {job.run_label}] samples={plan.measured_sample_count:<4} "
            f"turns={plan.measured_generation_turn_count:<4} "
            f"toks/s={metrics.output_toks_per_s:,.2f} "
            f"latency={metrics.latency_s:.1f}s "
            f"warmup_turns={plan.warmup_generation_turn_count} "
            f"warmdown_turns={plan.warmdown_generation_turn_count}"
        )
        if job.deployment.expect_spec:
            accept_len = (
                "N/A"
                if metrics.spec_accept_length is None
                else f"{metrics.spec_accept_length:.3f}"
            )
            line += (
                f" accept_len_mean={accept_len} "
                f"spec_verify_ct_sum={metrics.spec_verify_ct_sum}"
            )
        print(line)
        return JobResult(
            key=key,
            deployment=job.deployment,
            source_sample_count=source_sample_count,
            source_generation_turn_count=source_generation_turn_count,
            warmup_generation_turn_count=plan.warmup_generation_turn_count,
            warmdown_generation_turn_count=plan.warmdown_generation_turn_count,
            run_index=job.run_index,
            metrics=metrics,
        )
    finally:
        _shutdown_server(
            proc,
            base_url,
            drain_timeout_s=job.methodology.server_shutdown_drain_timeout_s,
            kill_timeout_s=job.methodology.server_shutdown_timeout_s,
        )


def _run_benchmark_job_gracefully(job: BenchmarkJob) -> JobResult | JobFailure:
    try:
        return _run_benchmark_job(job)
    except Exception as exc:
        error_type = type(exc).__name__
        error_message = _one_line(str(exc) or repr(exc))
        print(
            f"[failed {job.label} {job.run_label}] "
            f"{error_type}: {error_message}"
        )
        return JobFailure(
            key=job.key,
            deployment=job.deployment,
            run_index=job.run_index,
            error_type=error_type,
            error_message=error_message,
        )


def _print_summary(
    *,
    config: SweepConfig,
    workload: str,
    config_results: list[ConfigResult],
    shared_configs: list[SharedServerConfig],
    attention_backends: list[str],
    tp_sizes: list[int],
    concurrencies: list[int],
    device_sm: int,
    mode_keys: list[str],
    source_sample_count: Optional[int],
    source_generation_turn_count: Optional[int],
    results: dict[tuple[str, int, int, str], BenchMetrics],
) -> None:
    print("\n=== Speculative Benchmark Sweep Summary ===")
    _print_kv_lines(
        [
            ("workload", workload),
            ("source_sample_count", source_sample_count),
            ("source_generation_turn_count", source_generation_turn_count),
            ("target_model", config.target_model),
            ("dflash_draft_model", config.dflash_draft_model),
            ("spec_modes", ",".join(mode_keys)),
            (
                "mtp_num_steps",
                ",".join(str(x) for x in config.deployment_sweep.mtp_num_steps),
            ),
            (
                "mtp_num_draft_tokens",
                ",".join(
                    str(int(x) + 1)
                    for x in config.deployment_sweep.mtp_num_steps
                ),
            ),
            ("mtp_eagle_topk", 1),
            ("max_new_tokens", config.sampling.max_new_tokens),
            ("enable_thinking", bool(config.sampling.enable_thinking)),
            ("timeout_s", config.methodology.timeout_s),
            (
                "server_shutdown_drain_timeout_s",
                config.methodology.server_shutdown_drain_timeout_s,
            ),
            (
                "server_shutdown_timeout_s",
                config.methodology.server_shutdown_timeout_s,
            ),
            (
                "shared_server_configs",
                ";".join(
                    server_config.summary_label()
                    for server_config in shared_configs
                ),
            ),
            (
                "sampling",
                f"temperature:{config.sampling.temperature}, "
                f"top_p:{config.sampling.top_p}, top_k:{config.sampling.top_k}",
            ),
            ("attention_backends", ",".join(attention_backends)),
            (
                "dflash_block_sizes",
                ",".join(
                    "default" if x is None else str(x)
                    for x in config.deployment_sweep.dflash_block_sizes
                ),
            ),
            ("tp_sizes", ",".join(str(x) for x in tp_sizes)),
            ("concurrencies", ",".join(str(x) for x in concurrencies)),
            ("num_samples", config.methodology.num_samples),
            ("runs_per_config", config.methodology.runs_per_config),
            (
                "min_generation_turns_per_config",
                config.methodology.min_generation_turns_per_config,
            ),
            (
                "min_warmup_generation_turns",
                config.methodology.min_warmup_generation_turns,
            ),
            ("disable_radix_cache", False),
            ("device_sm", device_sm),
            ("skip_baseline", not config.deployment_sweep.include_baseline),
        ]
    )
    _print_failure_summary(config_results)

    for backend in attention_backends:
        print(f"\n=== Backend: {backend} ===")
        baseline_output_tps = _collect_metric(
            results=results,
            backend=backend,
            tp_sizes=tp_sizes,
            concurrencies=concurrencies,
            mode="baseline",
            field="output_toks_per_s",
        )
        sections: list[tuple[str, dict[tuple[int, int], Optional[float]], str]] = [
            ("Baseline output tok/s", baseline_output_tps, ",.2f")
        ]

        for spec_mode in mode_keys:
            display_name = _mode_display_name(spec_mode)
            spec_output_tps = _collect_metric(
                results=results,
                backend=backend,
                tp_sizes=tp_sizes,
                concurrencies=concurrencies,
                mode=spec_mode,
                field="output_toks_per_s",
            )
            spec_accept_length = _collect_metric(
                results=results,
                backend=backend,
                tp_sizes=tp_sizes,
                concurrencies=concurrencies,
                mode=spec_mode,
                field="spec_accept_length",
            )
            sections.extend(
                [
                    (f"{display_name} output tok/s", spec_output_tps, ",.2f"),
                    (
                        f"Speedup ({display_name} / baseline)",
                        _compute_speedup(baseline_output_tps, spec_output_tps),
                        ".3f",
                    ),
                    (
                        f"{display_name} acceptance length (mean per generation turn)",
                        spec_accept_length,
                        ".3f",
                    ),
                ]
            )

        for title, values, fmt in sections:
            print(f"\n{title}")
            print(
                _format_table(
                    tp_sizes=tp_sizes,
                    concurrencies=concurrencies,
                    values=values,
                    float_fmt=fmt,
                )
            )


CSV_FIELDS = [
    "workload",
    "backend",
    "tp",
    "mode",
    "mtp_num_steps",
    "dflash_block_size",
    "concurrency",
    "source_sample_count",
    "source_generation_turn_count",
    "runs_per_config",
    "successful_runs",
    "failed_runs",
    "status",
    "successful_run_numbers",
    "failed_run_numbers",
    "failure_messages",
    "measured_sample_count",
    "measured_generation_turn_count",
    "output_toks_per_s",
    "output_toks_per_s_std",
    "latency_s",
    "latency_s_std",
    "output_tokens",
    "speedup_vs_baseline",
    "accept_length_mean_from_conc1",
    "accept_length_mean_this_conc",
    "accept_length_mean_this_conc_std",
    "spec_verify_ct_sum",
]


def _one_line(value: str) -> str:
    return " ".join(str(value).split())


def _fmt_optional_int(value: Optional[int]) -> str:
    if value is None:
        return ""
    return str(value)


def _fmt_csv_value(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def _format_failed_run_numbers(failures: tuple[JobFailure, ...]) -> str:
    return ",".join(str(failure.run_index + 1) for failure in failures)


def _format_successful_run_numbers(result: ConfigResult) -> str:
    return ",".join(
        str(run_index + 1) for run_index in result.successful_run_indices
    )


def _format_failure_messages(failures: tuple[JobFailure, ...]) -> str:
    return " | ".join(
        (
            f"run={failure.run_index + 1} "
            f"{failure.error_type}: {failure.error_message}"
        )
        for failure in failures
    )


def _mean_optional(values: list[Optional[float]]) -> Optional[float]:
    present_values = [value for value in values if value is not None]
    if not present_values:
        return None
    return float(statistics.mean(present_values))


def _stdev_optional(values: list[Optional[float]]) -> Optional[float]:
    present_values = [value for value in values if value is not None]
    if len(present_values) < 2:
        return None
    return float(statistics.stdev(present_values))


def _metric_stdev(
    metrics: tuple[BenchMetrics, ...], field: str
) -> Optional[float]:
    return _stdev_optional([getattr(metric, field) for metric in metrics])


def _aggregate_bench_metrics(metrics: list[BenchMetrics]) -> BenchMetrics:
    if not metrics:
        raise RuntimeError("Cannot aggregate an empty metrics list.")
    first = metrics[0]
    return BenchMetrics(
        sample_count=first.sample_count,
        generation_turn_count=first.generation_turn_count,
        latency_s=float(statistics.mean(metric.latency_s for metric in metrics)),
        output_tokens=int(
            round(statistics.mean(metric.output_tokens for metric in metrics))
        ),
        output_toks_per_s=float(
            statistics.mean(metric.output_toks_per_s for metric in metrics)
        ),
        spec_accept_length=_mean_optional(
            [metric.spec_accept_length for metric in metrics]
        ),
        spec_verify_ct_sum=int(
            round(statistics.mean(metric.spec_verify_ct_sum for metric in metrics))
        ),
    )


def _aggregate_job_results(
    job_results: list[JobResult | JobFailure],
) -> list[ConfigResult]:
    grouped_results: dict[RunKey, list[JobResult | JobFailure]] = {}
    ordered_keys: list[RunKey] = []
    for result in job_results:
        if result.key not in grouped_results:
            grouped_results[result.key] = []
            ordered_keys.append(result.key)
        grouped_results[result.key].append(result)

    config_results: list[ConfigResult] = []
    for key in ordered_keys:
        results = sorted(grouped_results[key], key=lambda result: result.run_index)
        successful_results = [
            result for result in results if isinstance(result, JobResult)
        ]
        failures = tuple(
            result for result in results if isinstance(result, JobFailure)
        )
        first = results[0]
        first_success = successful_results[0] if successful_results else None
        repeat_metrics = tuple(result.metrics for result in successful_results)
        successful_run_indices = tuple(
            result.run_index for result in successful_results
        )
        metrics = (
            _aggregate_bench_metrics(list(repeat_metrics))
            if repeat_metrics
            else None
        )
        config_results.append(
            ConfigResult(
                key=key,
                deployment=first.deployment,
                source_sample_count=(
                    None if first_success is None else first_success.source_sample_count
                ),
                source_generation_turn_count=(
                    None
                    if first_success is None
                    else first_success.source_generation_turn_count
                ),
                warmup_generation_turn_count=(
                    None
                    if first_success is None
                    else first_success.warmup_generation_turn_count
                ),
                warmdown_generation_turn_count=(
                    None
                    if first_success is None
                    else first_success.warmdown_generation_turn_count
                ),
                metrics=metrics,
                repeat_metrics=repeat_metrics,
                successful_run_indices=successful_run_indices,
                failures=failures,
            )
        )
    return config_results


def _build_csv_rows(
    *,
    config_results: list[ConfigResult],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    results_by_key = {result.key: result for result in config_results}

    for result in config_results:
        key = result.key
        metrics = result.metrics
        baseline_result = results_by_key.get(
            RunKey(
                workload=key.workload,
                backend=key.backend,
                tp=key.tp,
                concurrency=key.concurrency,
                mode="baseline",
            )
        )
        speedup = None
        if (
            metrics is not None
            and key.mode != "baseline"
            and baseline_result is not None
            and baseline_result.metrics is not None
            and baseline_result.metrics.output_toks_per_s > 0
        ):
            speedup = (
                metrics.output_toks_per_s
                / baseline_result.metrics.output_toks_per_s
            )

        accept_source_result = results_by_key.get(
            RunKey(
                workload=key.workload,
                backend=key.backend,
                tp=key.tp,
                concurrency=1,
                mode=key.mode,
            )
            )
        accept_length_from_conc1 = (
            None
            if accept_source_result is None or accept_source_result.metrics is None
            else accept_source_result.metrics.spec_accept_length
        )

        rows.append(
            {
                "workload": key.workload,
                "backend": key.backend,
                "tp": key.tp,
                "mode": key.mode,
                "mtp_num_steps": result.deployment.mtp_num_steps or "",
                "dflash_block_size": result.deployment.dflash_block_size or "",
                "concurrency": key.concurrency,
                "source_sample_count": _fmt_optional_int(result.source_sample_count),
                "source_generation_turn_count": _fmt_optional_int(
                    result.source_generation_turn_count
                ),
                "runs_per_config": result.run_count,
                "successful_runs": result.successful_run_count,
                "failed_runs": result.failed_run_count,
                "status": result.status,
                "successful_run_numbers": _format_successful_run_numbers(result),
                "failed_run_numbers": _format_failed_run_numbers(result.failures),
                "failure_messages": _format_failure_messages(result.failures),
                "measured_sample_count": (
                    "" if metrics is None else metrics.sample_count
                ),
                "measured_generation_turn_count": (
                    "" if metrics is None else metrics.generation_turn_count
                ),
                "output_toks_per_s": _fmt_csv_value(
                    None if metrics is None else metrics.output_toks_per_s
                ),
                "output_toks_per_s_std": _fmt_csv_value(
                    _metric_stdev(result.repeat_metrics, "output_toks_per_s")
                ),
                "latency_s": _fmt_csv_value(
                    None if metrics is None else metrics.latency_s
                ),
                "latency_s_std": _fmt_csv_value(
                    _metric_stdev(result.repeat_metrics, "latency_s")
                ),
                "output_tokens": "" if metrics is None else metrics.output_tokens,
                "speedup_vs_baseline": _fmt_csv_value(speedup),
                "accept_length_mean_from_conc1": _fmt_csv_value(
                    accept_length_from_conc1
                ),
                "accept_length_mean_this_conc": _fmt_csv_value(
                    None if metrics is None else metrics.spec_accept_length
                ),
                "accept_length_mean_this_conc_std": _fmt_csv_value(
                    _metric_stdev(result.repeat_metrics, "spec_accept_length")
                ),
                "spec_verify_ct_sum": (
                    "" if metrics is None else metrics.spec_verify_ct_sum
                ),
            }
        )
    return rows


def _print_csv_summary(rows: list[dict[str, object]]) -> None:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CSV_FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    print("\n=== CSV Summary ===")
    print(buffer.getvalue(), end="", flush=True)


def _write_csv_summary(path: str, rows: list[dict[str, object]]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[csv] wrote {len(rows)} rows to {out_path}", flush=True)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workloads",
        dest="workloads",
        default=DEFAULT_WORKLOADS,
        help=(
            "Comma-separated workloads to run, or `all`."
        ),
    )
    parser.add_argument(
        "--csv-output",
        default=None,
        help="Optional path to write the final CSV summary.",
    )
    parser.add_argument("--target-model", default="Qwen/Qwen3.5-397B-A17B")
    parser.add_argument(
        "--dflash-draft-model",
        dest="dflash_draft_model",
        default=None,
        help="Required when --spec-modes includes dflash.",
    )
    parser.add_argument(
        "--spec-modes",
        default="mtp",
        help="Comma-separated speculative modes to benchmark. Supported: mtp,dflash.",
    )
    parser.add_argument(
        "--mtp-num-steps",
        default="3",
        help=(
            "Comma-separated MTP/EAGLE speculative num steps. num draft tokens "
            "is always num_steps + 1."
        ),
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip running the baseline (target-only) sweep; only run speculative modes and report N/A for baseline/speedup.",
    )
    thinking_group = parser.add_mutually_exclusive_group()
    thinking_group.add_argument(
        "--enable-thinking",
        dest="enable_thinking",
        action="store_true",
        default=True,
        help="Pass enable_thinking=True when applying the model chat template (default).",
    )
    thinking_group.add_argument(
        "--disable-thinking",
        dest="enable_thinking",
        action="store_false",
        help="Pass enable_thinking=False when applying the model chat template.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--concurrencies", default="1,32")
    parser.add_argument(
        "--num-samples",
        dest="num_samples",
        type=int,
        default=None,
        help=(
            "Exact number of measured samples per config. Repeats the selected "
            "workload if this exceeds the workload size. Default: unset."
        ),
    )
    parser.add_argument(
        "--runs-per-config",
        type=int,
        default=1,
        help=(
            "Number of repeated measured runs per benchmark configuration. "
            "The final reported metrics are averaged across these runs."
        ),
    )
    parser.add_argument(
        "--min-generation-turns-per-config",
        dest="min_generation_turns_per_config",
        type=int,
        default=1024,
        help=(
            "When --num-samples is unset and concurrency > 1, repeat whole workload "
            "copies until each config measures at least this many generation turns. "
            "Use 0 for one full workload copy."
        ),
    )
    parser.add_argument(
        "--min-warmup-generation-turns",
        type=int,
        default=8,
        help=(
            "Minimum generation turns to run after /flush_cache before measured "
            "timing. Effective warmup is max(this value, 2 * concurrency)."
        ),
    )
    parser.add_argument(
        "--dflash-block-sizes",
        default="default",
        help=(
            "Comma-separated DFlash block-size sweep. Use `default` to omit "
            "--speculative-dflash-block-size and let the server choose."
        ),
    )
    args = parser.parse_args(argv)
    try:
        workloads = _parse_workload_selection(args.workloads)
    except ValueError as exc:
        parser.error(str(exc))

    spec_modes = _parse_str_csv(args.spec_modes)
    supported_spec_modes = {"mtp", "dflash"}
    unknown_spec_modes = sorted(set(spec_modes) - supported_spec_modes)
    if unknown_spec_modes:
        parser.error(
            "--spec-modes contains unsupported values: "
            + ",".join(unknown_spec_modes)
        )
    if not spec_modes:
        parser.error("--spec-modes must include at least one mode: mtp or dflash")
    if "dflash" in spec_modes and not args.dflash_draft_model:
        parser.error(
            "--dflash-draft-model is required when --spec-modes includes dflash"
        )
    try:
        dflash_block_sizes = _parse_optional_int_csv(str(args.dflash_block_sizes))
    except ValueError as exc:
        parser.error(f"--dflash-block-sizes must be integers/default: {exc}")
    if any(x is not None and x <= 0 for x in dflash_block_sizes):
        parser.error(
            "--dflash-block-sizes values must be > 0, "
            f"got {args.dflash_block_sizes}"
        )
    mtp_num_steps = _parse_int_csv(str(args.mtp_num_steps))
    if not mtp_num_steps:
        parser.error("--mtp-num-steps must include at least one positive integer")
    if any(x <= 0 for x in mtp_num_steps):
        parser.error(f"--mtp-num-steps values must be > 0, got {args.mtp_num_steps}")
    mode_keys = DeploymentSweep(
        include_baseline=not args.skip_baseline,
        spec_modes=tuple(spec_modes),
        mtp_num_steps=tuple(mtp_num_steps),
        dflash_draft_model=args.dflash_draft_model,
        dflash_block_sizes=tuple(dflash_block_sizes),
    ).mode_keys
    duplicate_mode_keys = _duplicate_values(mode_keys)
    if duplicate_mode_keys:
        parser.error(
            "Duplicate deployment modes from sweep flags: "
            + ",".join(duplicate_mode_keys)
        )
    args.workloads = workloads
    args.spec_modes = spec_modes
    args.mtp_num_steps = mtp_num_steps
    args.dflash_block_sizes = dflash_block_sizes
    return args


def build_sweep_config_from_args(args: argparse.Namespace) -> SweepConfig:
    sampling = SamplingConfig(
        enable_thinking=bool(args.enable_thinking),
        max_new_tokens=int(args.max_new_tokens),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        top_k=int(args.top_k),
    )
    methodology = BenchmarkMethodologyConfig(
        num_samples=args.num_samples,
        min_generation_turns_per_config=int(args.min_generation_turns_per_config),
        min_warmup_generation_turns=int(args.min_warmup_generation_turns),
        runs_per_config=int(args.runs_per_config),
    )
    deployment_sweep = DeploymentSweep(
        include_baseline=not args.skip_baseline,
        spec_modes=tuple(args.spec_modes),
        mtp_num_steps=tuple(args.mtp_num_steps),
        dflash_draft_model=args.dflash_draft_model,
        dflash_block_sizes=tuple(args.dflash_block_sizes),
    )

    if sampling.temperature < 0.0:
        raise RuntimeError(f"--temperature must be >= 0, got {sampling.temperature}.")
    if not (0.0 < sampling.top_p <= 1.0):
        raise RuntimeError(f"--top-p must be in (0, 1], got {sampling.top_p}.")
    if sampling.top_k == 0 or sampling.top_k < -1:
        raise RuntimeError(
            f"--top-k must be -1 (all vocab) or >= 1, got {sampling.top_k}."
        )
    if methodology.num_samples is not None and methodology.num_samples <= 0:
        raise RuntimeError(f"--num-samples must be > 0, got {methodology.num_samples}.")
    if methodology.runs_per_config <= 0:
        raise RuntimeError(
            f"--runs-per-config must be > 0, got {methodology.runs_per_config}."
        )
    if (
        methodology.min_generation_turns_per_config < 0
        or methodology.min_warmup_generation_turns < 0
    ):
        raise RuntimeError(
            "--min-generation-turns-per-config and "
            "--min-warmup-generation-turns must be >= 0."
        )

    try:
        concurrencies = _parse_int_csv(args.concurrencies)
    except ValueError as exc:
        raise RuntimeError("--concurrencies must be comma-separated integers.") from exc
    if not concurrencies:
        raise RuntimeError("No concurrencies specified.")
    if any(c < 1 for c in concurrencies):
        raise RuntimeError(
            f"--concurrencies values must be >= 1, got {concurrencies}."
        )
    duplicate_concurrencies = _duplicate_values([str(c) for c in concurrencies])
    if duplicate_concurrencies:
        raise RuntimeError(
            "Duplicate concurrencies: " + ",".join(duplicate_concurrencies)
        )

    return SweepConfig(
        target_model=args.target_model,
        dflash_draft_model=args.dflash_draft_model,
        workloads=tuple(args.workloads),
        concurrencies=tuple(concurrencies),
        sampling=sampling,
        methodology=methodology,
        deployment_sweep=deployment_sweep,
        csv_output=args.csv_output,
    )


def _get_current_cuda_runtime() -> tuple[int, int]:
    import torch
    from sglang.srt.utils import get_device_sm

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this sweep.")
    return int(torch.cuda.device_count()), int(get_device_sm())


def build_shared_configs_for_runtime(
    sweep_config: SweepConfig,
) -> tuple[list[SharedServerConfig], int]:
    visible_gpus, device_sm = _get_current_cuda_runtime()
    shared_configs = _build_shared_server_configs(
        device_sm=device_sm,
        visible_gpus=visible_gpus,
        max_concurrency=max(sweep_config.concurrencies),
    )
    return shared_configs, device_sm


def build_shared_configs_for_modal(
    sweep_config: SweepConfig,
    *,
    device_sm: int,
    visible_gpus: int,
) -> list[SharedServerConfig]:
    return _build_shared_server_configs(
        device_sm=int(device_sm),
        visible_gpus=int(visible_gpus),
        max_concurrency=max(sweep_config.concurrencies),
    )


def build_benchmark_jobs(
    sweep_config: SweepConfig, shared_configs: list[SharedServerConfig]
) -> list[BenchmarkJob]:
    return _build_benchmark_jobs(sweep_config, shared_configs)


def run_benchmark_job_payload(payload: dict[str, Any]) -> dict[str, Any]:
    job = benchmark_job_from_payload(payload)
    return job_outcome_to_payload(_run_benchmark_job_gracefully(job))


def aggregate_job_outcomes(
    outcomes: list[JobResult | JobFailure],
) -> list[ConfigResult]:
    return _aggregate_job_results(outcomes)


def render_results(
    *,
    sweep_config: SweepConfig,
    shared_configs: list[SharedServerConfig],
    device_sm: int,
    config_results: list[ConfigResult],
) -> list[dict[str, object]]:
    mode_keys = sweep_config.deployment_sweep.mode_keys
    for workload in sweep_config.workloads:
        workload_results = [
            result for result in config_results if result.key.workload == workload
        ]
        if not workload_results:
            continue
        workload_shared_configs: list[SharedServerConfig] = []
        seen_shared_configs: set[SharedServerConfig] = set()
        for result in workload_results:
            shared_config = result.deployment.shared_config
            if shared_config not in seen_shared_configs:
                seen_shared_configs.add(shared_config)
                workload_shared_configs.append(shared_config)
        attention_backends = sorted(
            {config.attention_backend for config in workload_shared_configs}
        )
        tp_sizes = sorted({config.tp_size for config in workload_shared_configs})
        source_sample_count = next(
            (
                result.source_sample_count
                for result in workload_results
                if result.source_sample_count is not None
            ),
            None,
        )
        source_generation_turn_count = next(
            (
                result.source_generation_turn_count
                for result in workload_results
                if result.source_generation_turn_count is not None
            ),
            None,
        )
        print(f"\n\n##### Workload Summary: {workload} #####")
        _print_summary(
            config=sweep_config,
            workload=workload,
            config_results=workload_results,
            shared_configs=workload_shared_configs,
            attention_backends=attention_backends,
            tp_sizes=tp_sizes,
            concurrencies=list(sweep_config.concurrencies),
            device_sm=device_sm,
            mode_keys=mode_keys,
            source_sample_count=source_sample_count,
            source_generation_turn_count=source_generation_turn_count,
            results=_metric_map_from_config_results(workload_results),
        )

    csv_rows = _build_csv_rows(config_results=config_results)
    _print_csv_summary(csv_rows)
    if sweep_config.csv_output is not None:
        _write_csv_summary(sweep_config.csv_output, csv_rows)
    return csv_rows


def run_local_sweep(sweep_config: SweepConfig) -> list[ConfigResult]:
    shared_configs, device_sm = build_shared_configs_for_runtime(sweep_config)
    jobs = build_benchmark_jobs(sweep_config, shared_configs)
    job_results = [_run_benchmark_job_gracefully(job) for job in jobs]
    config_results = aggregate_job_outcomes(job_results)
    render_results(
        sweep_config=sweep_config,
        shared_configs=shared_configs,
        device_sm=device_sm,
        config_results=config_results,
    )
    return config_results


def main() -> None:
    args = parse_args()
    sweep_config = build_sweep_config_from_args(args)
    run_local_sweep(sweep_config)


if __name__ == "__main__":
    main()
