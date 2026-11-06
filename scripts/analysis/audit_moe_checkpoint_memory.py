"""CPU-only storage ledger and routing-footprint reference, never HBM claims."""
import _bootstrap
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

ROOT=_bootstrap.ROOT
OUT=ROOT/"reports/2026-11-06_moe_parallel_sd"
ELEMENT_BYTES={"BF16":2,"F16":2,"F32":4,"U8":1,"I8":1,"I32":4,"I64":8}


def main():
    manifest_path=ROOT/"models-moe-sd.json"
    manifest=json.loads(manifest_path.read_text())
    result={"manifest_SHA256":hashlib.sha256(manifest_path.read_bytes()).hexdigest(),"checkpoints":{}}
    for key,checkpoint in manifest["checkpoints"].items():
        categories=defaultdict(int);per_layer=defaultdict(lambda:defaultdict(int))
        stored_bytes=0
        for file in checkpoint["weights"]:
            path=Path(checkpoint["snapshot"])/file["file"]
            assert path.stat().st_size==file["bytes"]
            stored_bytes+=file["bytes"]
            for name,tensor in file["tensors"].items():
                size=math.prod(tensor["shape"])*ELEMENT_BYTES[tensor["dtype"]]
                if ".experts." in name:
                    category="expert_packed_FP4" if name.endswith("_blocks") else "expert_E8M0_scales" if name.endswith("_scales") else "expert_BF16_biases"
                    layer=int(name.split(".")[2]);per_layer[layer][category]+=size
                elif "embed_tokens" in name or "lm_head" in name:
                    category="embedding_and_LM_head"
                elif ".router." in name:
                    category="router"
                else:
                    category="attention_norm_or_other"
                categories[category]+=size
        config_path=Path(checkpoint["snapshot"])/"config.json"
        config=json.loads(config_path.read_text())
        result["checkpoints"][key]={"repo":checkpoint["repo"],"revision":checkpoint["revision"],
            "file_bytes":stored_bytes,"tensor_bytes":sum(categories.values()),
            "safetensors_headers_and_metadata_bytes":stored_bytes-sum(categories.values()),
            "categories_bytes":dict(categories),"expert_layer_bytes":dict(per_layer),
            "config_SHA256":hashlib.sha256(config_path.read_bytes()).hexdigest()}
        assert stored_bytes>=sum(categories.values())
        if key=="target":
            experts,topk=config["num_local_experts"],config["num_experts_per_tok"]
            assert experts==32 and topk==4 and len(per_layer)==24
            payload=sum(categories[x] for x in ["expert_packed_FP4","expert_E8M0_scales","expert_BF16_biases"])
            rows=[]
            for tokens in [1,2,4,8,16,32,64,128]:
                unique=experts*(1-(1-topk/experts)**tokens)
                rows.append({"token_rows":tokens,"uniform_independent_expected_unique_experts":unique,
                    "uniform_independent_expected_expert_bytes_proxy":payload*unique/experts,
                    "assignments_divided_by_expected_unique":tokens*topk/unique})
            result["uniform_routing_reference"]={"experts":experts,"topk":topk,"total_expert_payload_bytes":payload,"rows":rows}
    result["limits"]=["Checkpoint bytes are not runtime allocated/reserved/physical VRAM; padding, repacking, caches and startup transients differ.",
        "The uniform-independent table is a mathematical reference, not measured routing or predicted runtime.",
        "Distinct expert checkpoint payload is an access-footprint proxy, not measured DRAM/HBM traffic; caches and GEMM tiling can change reads.",
        "GPU performance remains unproven until admitted smoke, benchmarks and warmed profiling succeed."]
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"checkpoint_memory.json").write_text(json.dumps(result,indent=2))
    print(json.dumps({k:{"file_GiB":v["file_bytes"]/2**30,"tensor_GiB":v["tensor_bytes"]/2**30,"categories_GiB":{c:n/2**30 for c,n in v["categories_bytes"].items()}} for k,v in result["checkpoints"].items()},indent=2))


if __name__=="__main__":main()
