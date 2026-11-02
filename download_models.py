"""Fetch pinned model snapshots into the existing Hugging Face cache."""
import json
from pathlib import Path
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent


def main():
    records = json.loads((ROOT / "models.json").read_text())
    for record in records:
        path = snapshot_download(record["repo"], revision=record["revision"], max_workers=4,
                                 ignore_patterns=["*.bin", "*.pt", "*.onnx", "*.gguf"])
        link = Path(record["alias"])
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.exists() and not link.is_symlink():
            link.symlink_to(path, target_is_directory=True)
        print(record["repo"], path, flush=True)


if __name__ == "__main__":
    main()
