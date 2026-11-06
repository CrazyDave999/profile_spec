"""Download pinned GPT-OSS target/drafts outside the repository, with provenance."""
import _bootstrap
import json
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from safetensors import safe_open

ROOT = _bootstrap.ROOT
CHECKPOINTS = {
    "target": ("openai/gpt-oss-20b", "6cee5e81ee83917806bbde320786a8fb61efebee", "gpt-oss-20b"),
    "dflash": ("z-lab/gpt-oss-20b-DFlash", "d53f6551543204c859e8bbaaddbd15d11b447af9", "gpt-oss-20b-dflash"),
}


def main():
    api = HfApi()
    manifest = {"checked_at_UTC": datetime.now(timezone.utc).isoformat(), "checkpoints": {}}
    destination = ROOT / "models-moe-sd.json"
    if destination.exists():
        manifest = json.loads(destination.read_text())
    for key, (repo, revision, alias_name) in CHECKPOINTS.items():
        info = api.model_info(repo, revision=revision, files_metadata=True)
        files = [f for f in info.siblings if "/" not in f.rfilename and
                 (f.rfilename.endswith((".safetensors", ".json", ".txt", ".jinja", ".tiktoken")) or f.rfilename == "README.md")]
        print("DOWNLOAD", key, repo, revision, flush=True)
        snapshot = Path(snapshot_download(repo, revision=revision,
                        allow_patterns=[f.rfilename for f in files], max_workers=4))
        weights = []
        for item in files:
            if not item.rfilename.endswith(".safetensors"):
                continue
            path = snapshot / item.rfilename
            assert path.stat().st_size == item.size
            tensors = {}
            with safe_open(str(path), framework="pt", device="cpu") as stream:
                for name in stream.keys():
                    tensor = stream.get_slice(name)
                    tensors[name] = {"shape": tensor.get_shape(), "dtype": tensor.get_dtype()}
            weights.append({"file": item.rfilename, "bytes": item.size, "tensors": tensors})
        alias = Path.home() / "models" / alias_name
        alias.parent.mkdir(exist_ok=True)
        if alias.is_symlink():
            assert alias.resolve() == snapshot.resolve(), alias
        elif alias.exists():
            raise RuntimeError(f"Refusing to replace existing directory: {alias}")
        else:
            alias.symlink_to(snapshot, target_is_directory=True)
        manifest["checkpoints"][key] = {"repo": repo, "revision": info.sha,
            "alias": str(alias), "snapshot": str(snapshot), "weights": weights,
            "format": "native MXFP4" if key == "target" else "native DFlash"}
        destination.write_text(json.dumps(manifest, indent=2))
        print("READY", key, sum(w["bytes"] for w in weights), str(alias), flush=True)


if __name__ == "__main__":
    main()
