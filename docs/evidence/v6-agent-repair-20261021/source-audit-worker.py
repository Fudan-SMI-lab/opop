"""Developer validation harness. Explicit case mappings, never optimizer routing."""
from pathlib import Path
import json
import sys


def inference_reference(context):
    import torch
    from kernel_optimizer.project.evaluation import compare
    p = context["parameters"]
    request = p["request"]
    source = Path(p["source"])
    sys.path.insert(0, str(source))
    workload = request["workload"]
    if p["case"] == "graph":
        import infer as original
        config = json.loads((source/"workloads.json").read_text())
        model, inputs = original.prepare(config, workload, context["device"])
        expected = {"output": original.infer(model, inputs).cpu()}
    elif p["case"] == "resnet":
        import numpy as np
        import resnet_inference as original
        from transformers import ResNetForImageClassification
        original.configure()
        assets = Path(p["assets"])
        model = ResNetForImageClassification.from_pretrained(assets, local_files_only=True,
            dtype=torch.float32).to(context["device"]).eval()
        images = torch.from_numpy(np.load(assets/"pixels.npy")).to(context["device"])
        pixels = images[torch.arange(workload["batch"], device=context["device"]) % len(images)].contiguous()
        logits = original.infer(model, pixels).cpu()
        expected = {"logits": logits, p["top1_key"]: logits.argmax(-1)}
        if p.get("top1_representation") == "list":
            expected[p["top1_key"]] = expected[p["top1_key"]].tolist()
    else:
        import qwen_inference as original
        from transformers import AutoTokenizer, Qwen3ForCausalLM
        original.configure()
        config = json.loads((source/"workloads.json").read_text())["qwen_inference"]
        tokenizer = AutoTokenizer.from_pretrained(p["assets"], local_files_only=True)
        model = Qwen3ForCausalLM.from_pretrained(p["assets"], local_files_only=True,
            dtype=torch.bfloat16, attn_implementation="sdpa").to(context["device"]).eval()
        text = tokenizer.apply_chat_template([{"role":"user","content":config["prompt"]}],
            tokenize=False,add_generation_prompt=True,enable_thinking=False)
        base=tokenizer.encode(text,add_special_tokens=False)
        ids=(base*(workload["prompt_tokens"]//len(base)+1))[:workload["prompt_tokens"]]
        inputs=torch.tensor([ids]*workload["batch"],device=context["device"])
        tokens,logits,_=original.infer(model,inputs,workload["new_tokens"],capture=True)
        expected={"tokens":tokens.cpu(),"logits":logits.cpu()}
    actual=torch.load(Path(p["controls"])/"reference-rank-0.pt",map_location="cpu",weights_only=True)
    checks=[compare(x,expected,request["quality"]) for x in actual]
    return {"valid":all(x["passed"] for x in checks),"checks":checks,
            "method":"Original source function and input/model construction; full generated reference comparison",
            "workload":workload,"observation_keys":list(expected),
            "expected_shapes":{k:list(v.shape) if isinstance(v,torch.Tensor) else [len(v)] for k,v in expected.items()}}


def graph_heldout_probe(context):
    import torch
    from kernel_optimizer.project.evaluation import load_adapter,compare,snapshot
    p=context["parameters"]
    source=Path(p["source"])
    sys.path.insert(0,str(source))
    import infer as original
    config=json.loads((source/"workloads.json").read_text())
    request=p["request"]
    model,inputs=original.prepare(config,request["workload"],context["device"])
    expected=original.infer(model,inputs).cpu()
    adapter=load_adapter(Path(p["adapter"]))
    state=adapter.prepare(context,request)
    result=adapter.run(state)
    actual=snapshot(adapter.observe(state,result))["output"]
    check=compare(actual,expected,request["quality"])
    return {"valid":check["passed"],"check":check,"requested_workload":request["workload"],
        "actual_shape":list(actual.shape),"expected_shape":list(expected.shape),
        "method":"Frozen generated adapter under held-out request versus unchanged original source"}


def ddp_reference(context):
    from examples.v6_validation.source_fidelity import ddp_serial_reference
    result=ddp_serial_reference(context)
    import torch
    from kernel_optimizer.project.evaluation import compare
    p=context["parameters"];request=p["request"]
    ranks=[torch.load(Path(p["controls"])/f"reference-rank-{rank}.pt",map_location="cpu",
        weights_only=True,mmap=True) for rank in range(request["workload"]["world_size"])]
    exact=[compare(ranks[0][step]["gradients"],ranks[1][step]["gradients"],{"atol":0,"rtol":0})
           for step in range(request["continuous_steps"])]
    result["replica_gradient_checks"]=exact
    result["valid"] = result["valid"] and all(x["passed"] for x in exact)
    return result
