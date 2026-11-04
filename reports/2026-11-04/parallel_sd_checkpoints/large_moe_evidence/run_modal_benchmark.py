"""Modal entrypoint for running benchmark jobs in parallel."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

import modal


MINUTES = 60
HOURS = 60 * MINUTES

APP_NAME = "qwen35-397b-a17b-dflash-eval"
SGLANG_IMAGE = "lmsysorg/sglang:v0.5.13-cu130"
SGLANG_GIT_SHA = "ec36dde58083aca8f26c3740332498a11a06debf"

GPU_TYPE = "B200"
N_GPUS = 8
GPU = f"{GPU_TYPE}:{N_GPUS}"
CLOUD: str | None = "csc"
MAX_CONTAINERS = 32
JOB_TIMEOUT_S = 24 * HOURS
STARTUP_TIMEOUT_S = 30 * MINUTES

MODAL_VISIBLE_GPUS = N_GPUS
MODAL_DEVICE_SM = 100

HF_CACHE_PATH = "/root/.cache/huggingface"
FLASHINFER_CACHE_PATH = "/root/.cache/flashinfer"
EVAL_CACHE_PATH = "/cache/eval"
REMOTE_BENCHMARK_PATH = "/root/run_benchmark.py"
REMOTE_PATCH_DIR = "/root/patches"
REMOTE_PATCH_SCRIPT_PATH = "/root/apply_modal_patches.py"

LOCAL_BENCHMARK_PATH = Path(__file__).resolve().parent / "run_benchmark.py"
LOCAL_PATCH_DIR = Path(__file__).resolve().parent / "patches"
LOCAL_PATCH_SCRIPT_PATH = LOCAL_PATCH_DIR / "apply_patches.py"
FLASHINFER_PATCH_PATH = LOCAL_PATCH_DIR / "flashinfer-pr-3312.patch"

HF_CACHE_VOL = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
FLASHINFER_CACHE_VOL = modal.Volume.from_name(
    "flashinfer-cache", create_if_missing=True
)
EVAL_CACHE_VOL = modal.Volume.from_name(
    "qwen35-dflash-eval-cache", create_if_missing=True
)
HF_SECRET = modal.Secret.from_name("huggingface-secret")

RUNTIME_ENV = {
    "HF_HOME": HF_CACHE_PATH,
    "HF_HUB_CACHE": HF_CACHE_PATH,
    "HF_DATASETS_CACHE": f"{HF_CACHE_PATH}/datasets",
    "TRANSFORMERS_CACHE": HF_CACHE_PATH,
    "XDG_CACHE_HOME": f"{HF_CACHE_PATH}/xdg",
    "HF_XET_HIGH_PERFORMANCE": "1",
    "PYTHONPATH": "/root",
    "SGLANG_PYSPY_DUMP_BEFORE_CRASH": "0",
    "SGLANG_CUDA_COREDUMP_BEFORE_CRASH": "0",
}

sglang_image = (
    modal.Image.from_registry(SGLANG_IMAGE)
    .entrypoint([])
    .add_local_file(
        FLASHINFER_PATCH_PATH,
        remote_path=f"{REMOTE_PATCH_DIR}/flashinfer-pr-3312.patch",
        copy=True,
    )
    .add_local_file(
        LOCAL_PATCH_SCRIPT_PATH,
        remote_path=REMOTE_PATCH_SCRIPT_PATH,
        copy=True,
    )
    .env(
        {
            "PYTHONPATH": "/root",
        }
    )
    .uv_pip_install(
        f"git+https://github.com/sgl-project/sglang.git@{SGLANG_GIT_SHA}#subdirectory=python",
        "datasets",
        "hf_xet",
        "requests",
        "transformers",
    )
    .run_commands(f"python {REMOTE_PATCH_SCRIPT_PATH}")
    .run_commands(f"rm -rf {HF_CACHE_PATH}")
    .add_local_file(LOCAL_BENCHMARK_PATH, remote_path=REMOTE_BENCHMARK_PATH)
)

app = modal.App(name=APP_NAME)


def _load_benchmark_module():
    return importlib.import_module("run_benchmark")


@app.function(
    image=sglang_image,
    gpu=GPU,
    env=RUNTIME_ENV,
    secrets=[HF_SECRET],
    volumes={
        HF_CACHE_PATH: HF_CACHE_VOL,
        FLASHINFER_CACHE_PATH: FLASHINFER_CACHE_VOL,
        EVAL_CACHE_PATH: EVAL_CACHE_VOL,
    },
    cloud=CLOUD,
    retries=0,
    timeout=JOB_TIMEOUT_S,
    startup_timeout=STARTUP_TIMEOUT_S,
    max_containers=MAX_CONTAINERS,
    single_use_containers=True,
)
def run_benchmark_job_modal(job_payload: dict[str, Any]) -> dict[str, Any]:
    sys.path.insert(0, str(Path(REMOTE_BENCHMARK_PATH).parent))
    benchmark = _load_benchmark_module()
    benchmark.CACHE_DIR = Path(EVAL_CACHE_PATH)
    return benchmark.run_benchmark_job_payload(job_payload)


def _platform_failure_payload(
    job_payload: dict[str, Any],
    exc: BaseException | str,
) -> dict[str, Any]:
    benchmark = _load_benchmark_module()
    job = benchmark.benchmark_job_from_payload(job_payload)
    if isinstance(exc, BaseException):
        error_type = type(exc).__name__
        error_message = f"Modal platform failure: {exc!r}"
    else:
        error_type = "UnexpectedModalResult"
        error_message = exc
    return benchmark.job_outcome_to_payload(
        benchmark.JobFailure(
            key=job.key,
            deployment=job.deployment,
            run_index=job.run_index,
            error_type=error_type,
            error_message=error_message,
        )
    )


def _write_raw_output(path: str, outcome_payloads: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for payload in outcome_payloads:
            f.write(json.dumps(payload, sort_keys=True) + "\n")
    print(
        f"[modal] wrote {len(outcome_payloads)} raw job outcomes to {output_path}",
        flush=True,
    )


def _build_eval_argv(
    *,
    workloads: str,
    csv_output: str,
    target_model: str,
    dflash_draft_model: str,
    spec_modes: str,
    mtp_num_steps: str,
    skip_baseline: bool,
    enable_thinking: bool,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    concurrencies: str,
    num_samples: int,
    runs_per_config: int,
    min_generation_turns_per_config: int,
    min_warmup_generation_turns: int,
    dflash_block_sizes: str,
) -> list[str]:
    argv = [
        "--workloads",
        workloads,
        "--target-model",
        target_model,
        "--spec-modes",
        spec_modes,
        "--mtp-num-steps",
        mtp_num_steps,
        "--max-new-tokens",
        str(int(max_new_tokens)),
        "--temperature",
        str(float(temperature)),
        "--top-p",
        str(float(top_p)),
        "--top-k",
        str(int(top_k)),
        "--concurrencies",
        concurrencies,
        "--runs-per-config",
        str(int(runs_per_config)),
        "--min-generation-turns-per-config",
        str(int(min_generation_turns_per_config)),
        "--min-warmup-generation-turns",
        str(int(min_warmup_generation_turns)),
        "--dflash-block-sizes",
        dflash_block_sizes,
    ]
    if csv_output:
        argv.extend(["--csv-output", csv_output])
    if dflash_draft_model:
        argv.extend(["--dflash-draft-model", dflash_draft_model])
    if skip_baseline:
        argv.append("--skip-baseline")
    if enable_thinking:
        argv.append("--enable-thinking")
    else:
        argv.append("--disable-thinking")
    if num_samples > 0:
        argv.extend(["--num-samples", str(int(num_samples))])
    return argv


@app.local_entrypoint()
def main(
    workloads: str = "gsm8k",
    csv_output: str = "",
    target_model: str = "Qwen/Qwen3.5-397B-A17B",
    dflash_draft_model: str = "",
    spec_modes: str = "mtp",
    mtp_num_steps: str = "3",
    skip_baseline: bool = False,
    enable_thinking: bool = True,
    max_new_tokens: int = 4096,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 1,
    concurrencies: str = "1,32",
    num_samples: int = 0,
    runs_per_config: int = 1,
    min_generation_turns_per_config: int = 1024,
    min_warmup_generation_turns: int = 8,
    dflash_block_sizes: str = "default",
    raw_output: str = "",
    device_sm: int = MODAL_DEVICE_SM,
    visible_gpus: int = MODAL_VISIBLE_GPUS,
) -> None:
    benchmark = _load_benchmark_module()
    args = benchmark.parse_args(
        _build_eval_argv(
            workloads=workloads,
            csv_output=csv_output,
            target_model=target_model,
            dflash_draft_model=dflash_draft_model,
            spec_modes=spec_modes,
            mtp_num_steps=mtp_num_steps,
            skip_baseline=skip_baseline,
            enable_thinking=enable_thinking,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            concurrencies=concurrencies,
            num_samples=num_samples,
            runs_per_config=runs_per_config,
            min_generation_turns_per_config=min_generation_turns_per_config,
            min_warmup_generation_turns=min_warmup_generation_turns,
            dflash_block_sizes=dflash_block_sizes,
        )
    )
    sweep_config = benchmark.build_sweep_config_from_args(args)
    shared_configs = benchmark.build_shared_configs_for_modal(
        sweep_config,
        device_sm=device_sm,
        visible_gpus=visible_gpus,
    )
    jobs = benchmark.build_benchmark_jobs(sweep_config, shared_configs)
    job_payloads = [benchmark.benchmark_job_to_payload(job) for job in jobs]

    print(
        f"[modal] dispatching {len(job_payloads)} benchmark jobs "
        f"across up to {MAX_CONTAINERS} containers"
    )
    mapped_outputs = list(
        run_benchmark_job_modal.map(
            job_payloads,
            order_outputs=True,
            return_exceptions=True,
        )
    )

    outcome_payloads: list[dict[str, Any]] = []
    for job_payload, output in zip(job_payloads, mapped_outputs):
        if isinstance(output, BaseException):
            outcome_payloads.append(_platform_failure_payload(job_payload, output))
        elif not isinstance(output, dict):
            outcome_payloads.append(
                _platform_failure_payload(
                    job_payload,
                    f"Modal returned {type(output).__name__}, expected dict payload.",
                )
            )
        else:
            outcome_payloads.append(output)

    if raw_output:
        _write_raw_output(raw_output, outcome_payloads)

    outcomes = [
        benchmark.job_outcome_from_payload(payload) for payload in outcome_payloads
    ]
    config_results = benchmark.aggregate_job_outcomes(outcomes)
    benchmark.render_results(
        sweep_config=sweep_config,
        shared_configs=shared_configs,
        device_sm=device_sm,
        config_results=config_results,
    )
    sys.stdout.flush()
    sys.stderr.flush()
