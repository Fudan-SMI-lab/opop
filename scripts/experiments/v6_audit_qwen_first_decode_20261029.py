"""Recompute first-decode raw numerical controls without any model rerun."""
import argparse
import hashlib
from pathlib import Path
import torch
from kernel_optimizer.project.evaluation import compare
from kernel_optimizer.runtime.execution import write_json
from scripts.experiments.v6_audit_fsdp_20261028 import read, sha


def main(root):
    protocol = read(root/'protocol.json'); report = read(root/'run/first-decode.json'); completion = read(root/'followup-completed.json')
    assert completion['phase'] == 'completed'
    assert all(sha(p) == h for p, h in protocol['frozen_identities'].items())
    source = torch.load(protocol['source_reference']['path'], map_location='cpu', weights_only=True)
    quality = {'atol': .02, 'rtol': .02}
    def load(row):
        assert sha(row['path']) == row['sha256']
        return torch.load(row['path'], map_location='cpu', weights_only=True)
    base = load(report['full_baseline']['raw'])
    assert compare(base, {k:source[k] for k in base}, quality) == report['full_baseline']['comparison']
    assert all(torch.equal(base[k], source[k]) for k in base)
    prefix = load(report['prefix']['raw']); h = hashlib.sha256()
    for row in prefix:
        for key in ['keys','values']:
            t=row[key].contiguous();h.update(str((key,tuple(t.shape),t.dtype)).encode());h.update(t.view(torch.uint8).numpy().tobytes())
    assert h.hexdigest() == report['prefix']['value_sha256']
    outputs={}; caches={}
    old=load({'path': str(root.parent/'qwen-feedback-fixed-validation-20261029/fixed-validation/jobs/2-control-observed/run/observation-00-greedy.pt'),
              'sha256':'42e299a26ee221df37d065b45de9aac7031816da6a598a38a9f4fe9b87ae15bd'})
    for name,row in report['arms'].items():
        outputs[name]=load(row['output_raw']);caches[name]=load(row['updated_kv_raw'])
        expected={k:source[k][:,1:2] for k in outputs[name]}
        assert compare(outputs[name],expected,quality)==row['source_first_decode']
        assert all(torch.equal(outputs[name][k],old[k][:,1:2]) for k in outputs[name]) == row['equals_prior_failed_first_decode']
        assert len(caches[name])==len(prefix)==36
        assert all(tuple(r[k].shape)==(1,8,257,128) and torch.equal(r[k][:,:,:256],p[k]) for r,p in zip(caches[name],prefix) for k in r)
        for actual,ref,saved in zip(caches[name],caches['dynamic-eager'],row['kv_vs_dynamic_eager_descriptive']):
            assert all(float((actual[k].float()-ref[k].float()).abs().max())==saved[k+'_max_abs_error'] for k in actual)
    for i,a in enumerate(outputs):
        for b in list(outputs)[i+1:]:
            saved=report['cross_arm'][a+'__'+b]
            assert torch.equal(outputs[a]['logits'],outputs[b]['logits'])==saved['logits_exact']
            assert float((outputs[a]['logits'].float()-outputs[b]['logits'].float()).abs().max())==saved['max_abs_logit_error']
    audit={'core':report['core'],'finished_at':completion['finished_at'],'report_sha256':sha(root/'run/first-decode.json'),
        'all_raw_tensors_and_comparisons_recomputed':True,'frozen_identities_unchanged':True,
        'full_baseline':report['full_baseline'],'prefix':report['prefix'],'arms':report['arms'],'cross_arm':report['cross_arm'],
        'binding_signature_predicate':report['binding_signature_predicate'],
        'finding':'Dynamic eager zero error. Dynamic Graph and both static modes have identical first-decode logits matching the old failed first-decode. First-layer KV is equal; later KV differs. Binding and Graph replay are not necessary conditions for this particular observed mismatch. A unique numerical mechanism has not been isolated.',
        'limitations':['One first-decode input, fixed sequence and shared process; not full32-token candidate quality or performance acceptance.',
            'Observed cache equality covers full valid prefix and updated range. Intermediate KV errors descriptive, no new threshold.',
            'No native attention backend attribution yet. Actual input stride/predicate checks support code-path reasoning, not retrospective event capture.']}
    write_json(root/'terminal-audit.json',audit)
    print({'audit':str(root/'terminal-audit.json'),'sha256':sha(root/'terminal-audit.json'),'bytes':(root/'terminal-audit.json').stat().st_size})


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
