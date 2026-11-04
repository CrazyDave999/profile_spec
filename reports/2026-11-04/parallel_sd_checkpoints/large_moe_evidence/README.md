# Benchmark Reproduction

This folder contains the benchmark harness, Modal runner, runtime patch, and raw outputs used for the model card tables.

These scripts are intended to reproduce the release benchmark. They are not a general-purpose benchmark framework, and they should not be treated as a reference for Modal application structure or production Modal best practices.

## Full Sweep

The full sweep runs baseline, MTP, and DFlash across all workloads at concurrency 1 and 32, with 5 independent runs per configuration. It uses up to 32 Modal containers, each with 8 B200 GPUs.

This is a large benchmark and can take a long time to complete.

```bash
modal run run_modal_benchmark.py \
  --workloads all \
  --concurrencies 1,32 \
  --spec-modes mtp,dflash \
  --mtp-num-steps 3,7,15 \
  --dflash-draft-model modal-labs/Qwen3.5-397B-A17B-DFlash \
  --dflash-block-sizes 4,8,16 \
  --runs-per-config 5 \
  --raw-output results/full-sweep-r5.raw.jsonl \
  --csv-output results/full-sweep-r5.csv
```

The generated CSV and raw JSONL are written under `results/`. The raw JSONL contains per-run records; the CSV contains aggregated rows used by the README tables.

## Files

- `run_benchmark.py`: local benchmark harness and result aggregation.
- `run_modal_benchmark.py`: Modal entrypoint that dispatches one benchmark job per configuration.
- `patches/`: runtime patch helper and FlashInfer patch used by the reproduced environment.
- `results/full-sweep-r5.csv`: aggregate benchmark results.
- `results/full-sweep-r5.raw.jsonl`: raw per-run benchmark records.
