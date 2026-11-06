"""Check genuine batching, ragged isolation and greedy semantics in fp32."""
import _bootstrap
import json
from pathlib import Path
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from jetspec.methods.jetspec.draft_head import DFlashDraftModel
from batch_reference import BatchedReference


def main():
    torch.set_num_threads(2)
    torch.manual_seed(17)
    config = Qwen3Config(vocab_size=256, hidden_size=64, intermediate_size=128,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, pad_token_id=0, bos_token_id=1, eos_token_id=255)
    model = Qwen3ForCausalLM(config).float().eval()
    model.config._attn_implementation = "sdpa"
    hc = Qwen3Config(vocab_size=256, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=16)
    hc.num_target_layers = 4
    hc.block_size = 4
    hc.dflash_config = {"causal_head": True, "mask_token_id": 0, "target_layer_ids": [0, 2]}
    head = DFlashDraftModel(hc).float().eval()
    head.config._attn_implementation = "sdpa"
    prompts = [torch.tensor([[1, 13, 17, 5, 8]]), torch.tensor([[1, 4, 3, 8, 11, 25, 20, 9]])]
    ref = BatchedReference(model, head, {255}, depth=3, width=4, budget=15)
    outputs = {method: ref.generate(prompts, 16, method) for method in ["AR", "JetSpec"]}
    assert outputs["AR"][0]["token_ids"] == outputs["JetSpec"][0]["token_ids"]
    assert outputs["AR"][1]["token_ids"] == outputs["JetSpec"][1]["token_ids"]
    for i, prompt in enumerate(prompts):
        with torch.inference_mode():
            baseline = model.generate(prompt, do_sample=False, max_new_tokens=16)[0, prompt.numel():].tolist()
        for method in ["AR", "JetSpec"]:
            solo = ref.generate([prompt], 16, method)[0]["token_ids"]
            assert solo == outputs[method][i]["token_ids"], (method, i, solo, outputs[method][i])
            assert solo == baseline, (method, i, solo, baseline)
    assert any(s["method"] == "JetSpec" and s["batch"] == 2 and s["query"] > 1 for s in ref.batch_shapes)
    report = {"status": "passed", "dtype": "float32", "device": "cpu", "ragged_batch_size": 2,
              "generated_tokens_per_sequence": 16, "checks": ["AR equals HF greedy", "JetSpec equals HF greedy",
              "batched equals independent single-sequence decoding", "one target verify forward carries both sequences"]}
    Path('logs/batch_reference_validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
