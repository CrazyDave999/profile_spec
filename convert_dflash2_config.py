"""Adapt this pinned Speculators DFlash2 config; share unchanged tensor files."""
import hashlib
import json
from pathlib import Path

from safetensors import safe_open

ROOT = Path(__file__).resolve().parent


def main():
    manifest_path = ROOT / "models-parallel-sd.json"
    manifest = json.loads(manifest_path.read_text())
    source = Path(manifest["dflash2"]["snapshot"])
    original = json.loads((source/"config.json").read_text())
    assert original["architectures"] == ["DFlash2DraftModel"]
    assert original["sample_from_anchor"] is False
    assert original["speculators_config"]["verifier"]["name_or_path"] == "Qwen/Qwen3-8B"
    config = dict(original["transformer_layer_config"])
    assert set(config["layer_types"]) == {"sliding_attention"}, "Mixed causal/full layouts need separate conversion."
    target = json.loads((Path.home()/"models/Qwen3-8B/config.json").read_text())
    # Speculators converter explicitly adds 1 to z-lab's post-block IDs.
    # Inverting it is essential: these [2,10,...] are hidden_states indices.
    taps = [i-1 for i in original["aux_hidden_state_layer_ids"]]
    assert min(taps) >= 0 and max(taps) < target["num_hidden_layers"]-1
    config.update(architectures=["DFlash2DraftModel"], torch_dtype="bfloat16",
        num_target_layers=target["num_hidden_layers"], block_size=original["block_size"],
        is_causal=not original["sliding_window_non_causal"],
        dflash_config={"block_size": original["block_size"], "target_layer_ids": taps,
            **{k: original[k] for k in ["conv_kernel_size", "conv_group_size", "selector_rank", "selector_top_k", "mask_token_id"]}})
    shapes = {}
    with safe_open(str(source/"model.safetensors"), framework="pt", device="cpu") as f:
        shapes = {k: f.get_slice(k).get_shape() for k in f.keys()}
    assert shapes["fc.weight"] == [config["hidden_size"], len(taps)*target["hidden_size"]]
    assert len([k for k in shapes if "_conv." in k]) == config["num_hidden_layers"]*4
    assert shapes["candidate_selector.predecessor_codebook"] == [config["vocab_size"], original["selector_rank"]]
    assert shapes["candidate_selector.successor_codebook"] == [config["vocab_size"], original["selector_rank"]]
    assert shapes["candidate_selector.hidden_projection.weight"] == [original["selector_rank"], config["hidden_size"]]
    destination = Path.home()/"models/qwen3-8b-dflash2-sglang"
    destination.mkdir(exist_ok=True)
    (destination/"config.json").write_text(json.dumps(config, indent=2))
    weights = destination/"model.safetensors"
    if not weights.exists():
        weights.symlink_to(source/"model.safetensors")
    assert weights.resolve() == (source/"model.safetensors").resolve()
    conversion = {"source_revision": manifest["dflash2"]["revision"], "source_config_sha256": hashlib.sha256((source/"config.json").read_bytes()).hexdigest(),
        "target_layer_ids": taps, "weights_unchanged_symlink": True, "tensor_shapes": shapes,
        "indexing_reference": "https://github.com/vllm-project/speculators/blob/b964834c3fb3e1f202f93302204dea4c8cf8cc2d/src/speculators/convert/dflash/converter.py#L117",
        "limitations": "Only this pinned Qwen3-8B all-sliding-window checkpoint. Full runtime and output validation remain separate from config/header checks."}
    (ROOT/"logs/dflash2_config_conversion.json").write_text(json.dumps(conversion, indent=2))
    manifest["dflash2"].update(format="native", alias=str(destination), original_alias=str(Path.home()/"models/qwen3-8b-dflash2"), config_conversion="logs/dflash2_config_conversion.json", runtime_validation="pending")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"destination": str(destination), "target_layer_ids": taps, "tensor_count": len(shapes)}))


if __name__ == "__main__":
    main()
