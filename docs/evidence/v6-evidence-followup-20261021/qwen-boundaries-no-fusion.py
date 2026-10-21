"""Developer diagnostic: observe unchanged parent/candidate intermediate tensors."""
from pathlib import Path
import hashlib,json
from kernel_optimizer.config import load_config
from kernel_optimizer.models.project import CandidateBundle
from kernel_optimizer.project.branches import fork_run
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.runtime.execution import write_json

r=Path('/root/autodl-tmp/dev/v6-work/evidence-followup-20261021')
old=Path('/root/autodl-tmp/dev/v6-work/agent-repair-20261021')
source=old/'fresh-final/qwen-ttft'
d=json.loads((source/'trials/trial-e262267559/parameter-search.json').read_text())
bundle=CandidateBundle.model_validate_json(json.dumps(d['observations'][0]['bundle']))
out=r/'qwen-boundaries-no-fusion';fork_run(source,out,revision_id=bundle.parent_revision)
search=ProjectSearch(load_config(old/'qwen-ttft-fresh-final-private.yaml',[]),out)
helper='''
    audit = {"reference": {}, "handles": [], "sdpa": None, "sdpa_calls": 0}

    def _audit_store(name, value):
        if isinstance(value, (list, tuple)):
            value = value[0]
        audit["reference"][name] = value.detach().cpu().clone()

    def _audit_check(name, value):
        ref = audit["reference"].get(name)
        if ref is None:
            print("BOUNDARY_OBSERVATION", {"name": name, "status": "reference_missing"}, flush=True)
            return
        actual = value.detach().cpu()
        if name == "final_norm":
            ref = ref[:, -1:, :]
        if actual.numel() != ref.numel():
            print("BOUNDARY_OBSERVATION", {"name": name, "status": "shape_mismatch", "shape": list(actual.shape), "reference_shape": list(ref.shape)}, flush=True)
            return
        actual = actual.reshape(ref.shape).float()
        ref = ref.float()
        diff = (actual - ref).abs()
        print("BOUNDARY_OBSERVATION", {"name": name, "shape": list(ref.shape), "max_abs_error": float(diff.max()), "unequal": int((diff != 0).sum()), "violations_002": int((diff > (0.02 + 0.02 * ref.abs())).sum())}, flush=True)

    def _audit_begin():
        for i, layer in enumerate(layer_modules):
            for label, child in [("input_norm", layer.input_layernorm), ("post_norm", layer.post_attention_layernorm), ("q_proj", layer.self_attn.q_proj), ("k_proj", layer.self_attn.k_proj), ("v_proj", layer.self_attn.v_proj), ("o_proj", layer.self_attn.o_proj), ("down_proj", layer.mlp.down_proj), ("layer", layer)]:
                name = f"{i}:{label}"
                audit["handles"].append(child.register_forward_hook(lambda m, a, o, key=name: _audit_store(key, o)))
            name = f"{i}:activation"
            audit["handles"].append(layer.mlp.down_proj.register_forward_pre_hook(lambda m, a, key=name: _audit_store(key, a[0])))
        audit["handles"].append(final_norm.register_forward_hook(lambda m, a, o: _audit_store("final_norm", o)))
        audit["handles"].append(lm_head.register_forward_hook(lambda m, a, o: _audit_store("logits", o)))
        original = F.scaled_dot_product_attention
        audit["sdpa"] = original
        def observed_sdpa(q, k, v, *args, **kwargs):
            i = audit["sdpa_calls"]
            audit["sdpa_calls"] += 1
            for key, value in [("q_rotary", q), ("k_rotary", k), ("v_cache", v)]:
                _audit_store(f"{i}:{key}", value)
            print("SDPA_OBSERVATION", {"layer": i, "q_stride": list(q.stride()), "k_stride": list(k.stride()), "v_stride": list(v.stride()), "args": [str(x) if not isinstance(x, torch.Tensor) else {"shape": list(x.shape), "dtype": str(x.dtype)} for x in args], "kwargs": {key: str(value) if not isinstance(value, torch.Tensor) else {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in kwargs.items()}}, flush=True)
            result = original(q, k, v, *args, **kwargs)
            _audit_store(f"{i}:attention", result)
            return result
        F.scaled_dot_product_attention = observed_sdpa

    def _audit_end():
        F.scaled_dot_product_attention = audit["sdpa"]
        for handle in audit["handles"]:
            handle.remove()
        audit["handles"].clear()

'''

edits=[]
for edit in bundle.edits:
 if edit.path==bundle.installer:
  raw=search.store.get_artifact(edit.content_ref);code=raw.decode()
  def replace(a,b):
   global code
   assert code.count(a)==1,(a,code.count(a))
   code=code.replace(a,b)
  replace('    # ------------------------------------------------------------------\n    def _fused(input_ids):',helper+'    # ------------------------------------------------------------------\n    def _fused(input_ids):')
  replace('                ref_out = parent(\n                    input_ids=input_ids, use_cache=True, logits_to_keep=1\n                )',
          '                _audit_begin()\n                try:\n                    ref_out = parent(input_ids=input_ids, use_cache=True, logits_to_keep=1)\n                finally:\n                    _audit_end()')
  replace('            na3 = na.view(b, s, hidden)','            _audit_check(f"{i}:input_norm", na)\n            na3 = na.view(b, s, hidden)')
  replace('            v = F.linear(na3, L["v_w"])','            v = F.linear(na3, L["v_w"])\n            for label, value in [("q_proj", q), ("k_proj", k), ("v_proj", v)]:\n                _audit_check(f"{i}:{label}", value)')
  replace('            attn = F.scaled_dot_product_attention(', '            for label, value in [("q_rotary", qrot), ("k_rotary", ck), ("v_cache", cv)]:\n                _audit_check(f"{i}:{label}", value)\n            attn = F.scaled_dot_product_attention(')
  replace('            attn2 = attn.transpose(1, 2).contiguous().view(b, s, n_q_heads * head_dim)','            _audit_check(f"{i}:attention", attn)\n            attn2 = attn.transpose(1, 2).contiguous().view(b, s, n_q_heads * head_dim)')
  replace('            o = F.linear(attn2, L["o_w"])','            o = F.linear(attn2, L["o_w"])\n            _audit_check(f"{i}:o_proj", o)')
  replace('            g = F.linear(nb.view(b, s, hidden), L["gw"])','            _audit_check(f"{i}:post_norm", nb)\n            g = F.linear(nb.view(b, s, hidden), L["gw"])')
  replace('            d = F.linear(act.view(b, s, inter), L["dw"])','            _audit_check(f"{i}:activation", act)\n            d = F.linear(act.view(b, s, inter), L["dw"])\n            _audit_check(f"{i}:down_proj", d)')
  replace('            cur = rb','            cur = rb\n            _audit_check(f"{i}:layer", cur)')
  replace('        logits = F.linear(out_last.view(1, 1, hidden), lm_head.weight)','        _audit_check("final_norm", out_last)\n        logits = F.linear(out_last.view(1, 1, hidden), lm_head.weight)\n        _audit_check("logits", logits)')
  assert code.count('HEAD_DIM=head_dim,') == 2
  code=code.replace('HEAD_DIM=head_dim,','HEAD_DIM=head_dim, enable_fp_fusion=False,')
  code=code.replace('return bool(torch.all(diff <= tol).item())', 'print("GUARD_READOUT", {"shape": list(a.shape), "max_abs_error": float(diff.max().item()), "violations_internal": int((diff > tol).sum().item()), "violations_formal": int((diff > (0.02 + 0.02 * rf.abs())).sum().item())}, flush=True)\n        return bool(torch.all(diff <= tol).item())')
  edit=edit.model_copy(update={'content_ref':search.store.put_artifact(code)})
  (out/'instrumented-candidate.py').write_text(code)
  write_json(out/'protocol.json',{'scope':'Observe original parent and candidate intermediate tensors. Math, tolerance, returns and fallback remain unchanged. CPU copies and hooks perturb execution; no performance claim. Intermediate 0.02 counts are diagnostic comparisons, not separate acceptance thresholds.', 'original_sha256':hashlib.sha256(raw).hexdigest(),'instrumented_sha256':hashlib.sha256(code.encode()).hexdigest()})
 edits.append(edit)
bundle=bundle.model_copy(update={'id':'qwen-boundary-diagnostic','edits':tuple(edits)})
revision=search.revisions.stage(bundle)
run=search.execute(revision,'boundaries','candidate_diagnostics')
import ast
rows=[];sdpa=[]
for p in (out/'jobs/boundaries/run').glob('rank-*.log'):
 for line in p.read_text().splitlines():
  if line.startswith('BOUNDARY_OBSERVATION '):rows.append(ast.literal_eval(line.removeprefix('BOUNDARY_OBSERVATION ')))
  if line.startswith('SDPA_OBSERVATION '):sdpa.append(ast.literal_eval(line.removeprefix('SDPA_OBSERVATION ')))
write_json(out/'result.json',{'scope':'Readout only; no code repair or performance claim','run':run,'boundaries':rows,'sdpa':sdpa})
print(json.dumps({'status':run['status'],'boundaries':len(rows),'sdpa_calls':len(sdpa)}),flush=True)
