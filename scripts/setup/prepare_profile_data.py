"""Pin and export the seven JetSpec evaluation prompt sets without executing dataset code."""
import _bootstrap
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
import pyarrow.parquet as pq
from transformers import AutoTokenizer

ROOT = _bootstrap.ROOT
SOURCES = {
    "gsm8k": ("openai/gsm8k", ["main/test-00000-of-00001.parquet"]),
    "math500": ("HuggingFaceH4/MATH-500", ["test.jsonl"]),
    "aime25": ("yentinglin/aime_2025", ["data/train-00000-of-00001-243207c6c994e1bd.parquet"]),
    "humaneval": ("openai/openai_humaneval", ["openai_humaneval/test-00000-of-00001.parquet"]),
    "mbpp": ("google-research-datasets/mbpp", ["full/test-00000-of-00001.parquet"]),
    "livecodebench": ("livecodebench/code_generation_lite", ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl"]),
    "mt_bench": ("HuggingFaceH4/mt_bench_prompts", ["data/train-00000-of-00001-67c6c9fef07685a3.parquet"]),
}


def prompt(name, row):
    if name == "gsm8k":
        return row["question"] + "\nPlease reason step by step, and put your final answer within \\boxed{}.", row.get("answer")
    if name in ("math500", "aime25"):
        return row["problem"] + "\nPlease reason step by step, and put your final answer within \\boxed{}.", row.get("answer")
    if name == "humaneval":
        return "Write a solution to the following problem and make sure that it passes the tests:\n```python\n" + row["prompt"] + "\n```", None
    if name == "mbpp":
        return ("You are an expert Python programmer. Write a solution to the following task and make sure it passes the tests:\n"
                + row["text"] + "\nYour code should pass these tests:\n" + "\n".join(row["test_list"])), None
    if name == "livecodebench":
        return "Write a Python solution to the following competitive-programming problem:\n```\n" + row["question_content"] + "\n```", None
    return row["prompt"][0], None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--max-prompt-tokens", type=int, default=2048)
    args = parser.parse_args()
    dest = ROOT / "data" / "jetspec_paper"
    dest.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(str(Path.home() / "models/Qwen3-8B"))

    def fetch(name):
        repo, files = SOURCES[name]
        info = HfApi().dataset_info(repo)
        rows = []
        for file in files:
            path = hf_hub_download(repo, file, repo_type="dataset", revision=info.sha)
            if file.endswith("parquet"):
                rows.extend(pq.read_table(path).to_pylist())
            else:
                rows.extend(json.loads(line) for line in Path(path).read_text().splitlines() if line.strip())
        if name == "livecodebench":
            unique = {}
            for row in rows:
                unique[str(row.get("question_id", row.get("question_title")))] = row
            rows = list(unique.values())
        output = []
        for index, row in enumerate(rows[:args.samples]):
            text, answer = prompt(name, row)
            ids = tokenizer.apply_chat_template([{"role": "user", "content": text}],
                tokenize=True, add_generation_prompt=True, enable_thinking=False)
            original = len(ids)
            if original > args.max_prompt_tokens:
                # Keep system/chat prefix and latest task instructions, record truncation explicitly.
                ids = ids[:64] + ids[-(args.max_prompt_tokens - 64):]
            output.append({"index": index, "token_ids": ids, "original_prompt_tokens": original,
                           "prompt_tokens": len(ids), "truncated": original != len(ids),
                           "raw_prompt": text, "answer": answer})
        path = dest / f"{name}.json"
        path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
        record = {"name": name, "repo": repo, "revision": info.sha, "files": files,
                  "available_rows": len(rows), "sampled_rows": len(output),
                  "truncated_rows": sum(r["truncated"] for r in output),
                  "export_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        print(json.dumps(record), flush=True)
        return record

    records = list(ThreadPoolExecutor(4).map(fetch, SOURCES))
    (dest / "manifest.json").write_text(json.dumps({"paper": "https://arxiv.org/html/2606.18394v2",
        "selection": "first min(32, dataset size) records in upstream order; MT-Bench first turn; LCB release_v5 cumulative files",
        "thinking": False, "max_prompt_tokens": args.max_prompt_tokens,
        "sources": records}, indent=2) + "\n")


if __name__ == "__main__":
    main()
