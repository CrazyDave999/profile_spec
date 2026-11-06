"""Fetch pinned draft checkpoints into the external HF cache; never fetch target duplicates."""
import _bootstrap
import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from safetensors import safe_open

ROOT = _bootstrap.ROOT
CHECKPOINTS = {
    "dflash": ("z-lab/Qwen3-8B-DFlash-b16", "9b41424b7109f9c5413454f481b09a82b85333f4"),
    "domino": ("Huang2020/Qwen3-8B-Domino-b16", "9a2c3213cbe7b0718805aa92ab1e1404ee028a90"),
    "dspark": ("deepseek-ai/dspark_qwen3_8b_block7", "03326e5043815da1f81b109078b2889737c26017"),
    "dflash2": ("RedHatAI/Qwen3-8B-speculator.dflash2", "e1c737a13c34ba1e1f0f739e02af0bf082fe8080"),
    "qwen35_target": ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"),
    "qwen35_dflash": ("z-lab/Qwen3.5-4B-DFlash", "9a1996ccf887b79ab3af4fcbf8c1d1f4b5658bcf"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", choices=CHECKPOINTS, default=["dflash", "domino", "dspark"])
    args = parser.parse_args()
    destination = ROOT / "models-parallel-sd.json"
    manifest = json.loads(destination.read_text()) if destination.exists() else {}
    for method in args.methods:
        repo, revision = CHECKPOINTS[method]
        info = HfApi().model_info(repo, revision=revision, files_metadata=True)
        snapshot = Path(snapshot_download(repo, revision=revision, allow_patterns=["*.json", "*.safetensors", "README.md", "tokenizer.*", "merges.txt", "chat_template*"], max_workers=4))
        weights = []
        for item in info.siblings:
            if not item.rfilename.endswith(".safetensors"):
                continue
            p = snapshot / item.rfilename
            assert p.stat().st_size == item.size, p
            with safe_open(str(p), framework="pt", device="cpu") as f:
                tensor_count = len(list(f.keys()))
            weights.append({"file": item.rfilename, "bytes": item.size, "tensor_count": tensor_count})
        alias_name = {"qwen35_target": "Qwen3.5-4B", "qwen35_dflash": "qwen3.5-4b-dflash"}.get(method, "qwen3-8b-"+method)
        alias = Path.home() / "models" / alias_name
        if alias.is_symlink():
            assert alias.resolve() == snapshot.resolve(), f"Existing alias points at another revision: {alias}"
        elif alias.exists():
            raise RuntimeError(f"Will not replace existing model directory: {alias}")
        else:
            alias.symlink_to(snapshot, target_is_directory=True)
        manifest[method] = {"repo": repo, "revision": info.sha, "snapshot": str(snapshot), "alias": str(alias), "weights": weights, "format": "speculators; needs conversion" if method == "dflash2" else "native"}
        destination.write_text(json.dumps(manifest, indent=2))
        print(json.dumps({"method": method, "revision": info.sha, "weight_GiB": round(sum(w['bytes'] for w in weights)/2**30, 3), "alias": str(alias)}), flush=True)


if __name__ == "__main__":
    main()
