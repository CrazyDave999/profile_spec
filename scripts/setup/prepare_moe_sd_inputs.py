"""Freeze Harmony inputs, including its date, before a potentially delayed run."""
import _bootstrap
import hashlib
import json
from pathlib import Path

ROOT=_bootstrap.ROOT
BENCHMARK_DATE="2026-11-06"


def main():
    from transformers import AutoTokenizer
    target=Path.home()/"models/gpt-oss-20b"
    tokenizer=AutoTokenizer.from_pretrained(str(target))
    manifest=json.loads((ROOT/"models-moe-sd.json").read_text())
    destination=ROOT/"data/moe_sd_20261106";destination.mkdir(parents=True,exist_ok=True)
    for dataset in ("gsm8k","humaneval","mt_bench"):
        source=ROOT/f"data/jetspec_paper/{dataset}.json"
        prompts=json.loads(source.read_text())[:32]
        records=[]
        for i,prompt in enumerate(prompts):
            ids=tokenizer.apply_chat_template([{"role":"user","content":prompt["raw_prompt"]}],
                tokenize=True,add_generation_prompt=True,return_dict=False,reasoning_effort="medium",
                strftime_now=lambda fmt: BENCHMARK_DATE)
            text=tokenizer.decode(ids)
            assert f"Current date: {BENCHMARK_DATE}" in text and "Reasoning: medium" in text
            assert all(isinstance(x,int) for x in ids)
            records.append({"prompt_index":i,"raw_prompt_sha256":hashlib.sha256(prompt["raw_prompt"].encode()).hexdigest(),"input_ids":ids})
        output={"source":str(source.relative_to(ROOT)),"source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
            "target_revision":manifest["checkpoints"]["target"]["revision"],"template_date":BENCHMARK_DATE,
            "reasoning_effort":"medium","prompts":records}
        path=destination/(dataset+".json")
        if path.exists():
            assert json.loads(path.read_text())==output,"Refusing to replace different frozen inputs."
        else:path.write_text(json.dumps(output,indent=2))
        print(dataset,len(records),"input lengths",min(len(r["input_ids"]) for r in records),max(len(r["input_ids"]) for r in records))


if __name__=="__main__":main()
