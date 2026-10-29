"""Small real-framework CPU probes for diagnostic failure persistence; no model/GPU."""
import argparse
from datetime import datetime,timezone
import importlib.util
import json
from pathlib import Path
import traceback
import torch
import torch.distributed as dist
from kernel_optimizer.project import evaluation
from kernel_optimizer.runtime.execution import write_json
from scripts.experiments.v6_audit_fsdp_20261028 import sha


ENTRY = '''import torch
from kernel_optimizer.project import evaluation
def run(context):
    value = context['parameters']['cpu_fixture_value']
    expected = {'loss': torch.tensor(1.), 'logits': torch.zeros(2), 'gradients': {'x': torch.zeros(2)}}
    actual = {'loss': torch.tensor(1.), 'logits': torch.tensor([0., 1e-4]), 'gradients': {'x': torch.tensor([0., value])}}
    check = evaluation.compare(actual, expected, {'atol':1e-6,'rtol':1e-4})
    return {'valid':check['passed'], 'checks':[{'continuous_step':0, **check}], 'samples':[], 'metrics':{}}
'''


def run(probe,output):
    assert not output.exists();output.mkdir(parents=True)
    expected_sha=sha(probe)
    spec=importlib.util.spec_from_file_location('cpu_diagnostic_probe',probe);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    entry=output/'original_entry.py';entry.write_text(ENTRY)
    es=importlib.util.spec_from_file_location('cpu_original_entry',entry);em=importlib.util.module_from_spec(es);es.loader.exec_module(em)
    rows=[];old_compare=evaluation.compare;old_collective=dist.all_gather_into_tensor
    for label,value in [('finite',2e-7),('nan',float('nan')),('inf',float('inf'))]:
        path=output/label;path.mkdir()
        context={'rank':0,'world_size':1,'device':'cpu','output':str(path),'parameters':{'action':'evaluate','cpu_fixture_value':value,
            'request':{'quality':{'atol':1e-6,'rtol':1e-4}},'bundles':[],
            'diagnostic_context':{'original_entry':str(entry),'original_function':'run','arm':label}}}
        original=em.run(context);assert original['valid']is False and original['checks'][0]['observation_path']=='output.logits'
        record={'case':label,'original_result':original,'original_entry_sha256':sha(entry)}
        try:
            observed=module.run(context);record.update(observed_status='returned',observed_result=observed,
                result_preserved={k:v for k,v in observed.items()if k!='diagnostic_index'}==original)
        except Exception as exc:record.update(observed_status='raised',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
        record['evidence_files']=[{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}for p in path.iterdir()if p.is_file()]
        record['compare_restored']=evaluation.compare is old_compare;record['collective_restored']=dist.all_gather_into_tensor is old_collective
        assert record['compare_restored'] and record['collective_restored']
        rows.append(record)
    special=module._full_traversal(torch.tensor([100000.5,2e-6]),torch.tensor([100000.,0.]),{'atol':1e-6,'rtol':1e-4})
    assert sha(probe)==expected_sha and not torch.cuda.is_initialized()
    result={'at':datetime.now(timezone.utc).isoformat(),'probe':str(probe),'probe_sha256':expected_sha,'cases':rows,
        'top_mismatch_label_fixture':special,'scope':'CPU synthetic comparator inputs with real framework compare, not FSDP numerical observations or CUDA quality. Original probe unmodified; no model/control/reference run/copy.',
        'cuda_initialized':torch.cuda.is_initialized()}
    write_json(output/'result.json',result)
    print(json.dumps({'probe_sha256':expected_sha,'cases':[{k:r.get(k)for k in ['case','observed_status','result_preserved','error_type','error','evidence_files']}for r in rows],
        'top_records':special['paths'][0].get('top_mismatches')}))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--probe',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.probe,a.output)
