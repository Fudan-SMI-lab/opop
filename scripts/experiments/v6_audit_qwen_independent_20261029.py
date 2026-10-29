"""Read-only audit of the fact-assisted Qwen complete-version validation."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess

from kernel_optimizer.config import load_config
from kernel_optimizer.project.measurement import paired_decision,valid_run
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.runtime.execution import write_json
from scripts.experiments.v6_audit_fsdp_20261028 import read,sha
from scripts.experiments.v6_audit_qwen_end_to_end_20261029 import endpoint


def native(rank):
    db=sqlite3.connect('file:'+rank['evidence']+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
    rows=[dict(r)for r in db.execute('select k.*,s.value as name from CUPTI_ACTIVITY_KIND_KERNEL k left join StringIds s on k.demangledName=s.id')]
    assert rows==rank['kernels'] and len(rows)==rank['cuda_kernels']
    result={'sqlite':rank['evidence'],'sha256':sha(rank['evidence']),'kernel_count':len(rows),
        'graph_kernel_count':sum(bool(r.get('graphNodeId'))for r in rows),
        'graph_runtime_calls':[list(r)for r in db.execute("select s.value,count(*) from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on r.nameId=s.id where s.value like '%Graph%' group by s.value")],
        'graph_nodes_per_correlation':dict(Counter(r['correlationId']for r in rows if r.get('graphNodeId'))),
        'flash_graph':sum('flash_fwd'in r['name']and bool(r.get('graphNodeId'))for r in rows),
        'flash_outside_graph':sum('flash_fwd'in r['name']and not r.get('graphNodeId')for r in rows)}
    db.close();return result


def audit(root):
    final=read(root/'followup-completed.json');protocol=read(root/'protocol.json');core=protocol['core']
    assert final['phase']=='completed' and all(r['status']=='completed'for r in final['results'].values())
    config=load_config(root/'private-config.yaml',[]);s=ProjectSearch(config,root/'qwen-independent')
    parent_root=Path(protocol['parent']);prior=read(parent_root/'terminal-audit.json')
    assert sha(parent_root/'terminal-audit.json')==protocol['parent_audit_sha256']
    original=ProjectSearch(config,parent_root/'qwen-continuation');current=original.revisions.current()
    pretrial=ProjectSearch(config,Path(prior['reuse']['source'])).revisions.current()
    assert pretrial.id==prior['pointers']['source'] and not pretrial.bundles
    assert current.id==protocol['source_revision']==prior['pointers']['current']
    assert s.revisions.current().id==s.prepared['revision']==protocol['original_baseline']
    assert s.prepared==original.prepared and s.request==original.request
    assert all(sha(p)==h for p,h in prior['frozen_identities_verified'].items())
    baseline=s.revisions.current();files={f.path:f.sha256 for f in baseline.files}
    for revision in [baseline,current]:
        assert revision.contract_ref==baseline.contract_ref
        assert all(hashlib.sha256(original.store.get_artifact(f.content_ref)).hexdigest()==f.sha256 for f in revision.files)
    source=read(root/'source-check.json');assert valid_run(source)
    sr=source['ranks'][0]['result'];assert all(c['passed']and c['max_abs_error']==0 and c['exact_tokens']and c['finite_logits']for c in sr['checks'])
    assert len(sr['checks'])==2 and all(not c['passed']for c in sr['negative_checks'].values())
    p=source['ranks'][0]['parameters']if 'parameters'in source['ranks'][0]else read(root/'source-check-run/request.json')['parameters']
    assert sha(p['reference'])==sr['source_reference_sha256'] and sha(p['controls_reference'])==sr['controls_reference_sha256']
    trials=list((s.output/'version-trials').glob('*/result.json'));assert len(trials)==1;t=read(trials[0])
    assert t['publication']=='exploration'and t['decision']['accepted']and t['quality']=='pass'and not t['coverage_errors']
    assert t['source_revision']==current.id and t['bundles']==list(current.bundles)
    assert paired_decision(t['pairs'],s.request.objective,s.prepared['evaluation'])==t['measurement']
    pairs=[{k:endpoint(v,core)for k,v in pair.items()}for pair in t['pairs']]
    assert len(pairs)==2
    for pair in pairs:
        for e in pair.values():
            for sample in e['samples']:
                assert sample['generated_tokens']==32 and sample['decode_tokens']==31
                assert abs(sample['end_to_end_ms']-sample['ttft_ms']-sample['decode_ms'])<1e-6
                assert abs(sample['decode_tokens_per_second']-31000/sample['decode_ms'])<1e-8
                assert abs(sample['generated_tokens_per_second']-32000/sample['end_to_end_ms'])<1e-8
    native_b=read(root/'native-baseline.json');assert native_b['status']=='observed'and native_b['run']['status']=='passed'
    reused=prior['native'];assert sha(reused['normalized_path'])==reused['normalized_sha256']and sha(reused['sqlite'])==reused['sqlite_sha256']
    natives={'baseline':native(native_b['ranks'][0]),'candidate_reused':native(read(reused['normalized_path'])[0])}
    jobs=[]
    for path in sorted((s.output/'jobs').glob('*/run/rank-0-started.json')):
        started=read(path);request=read(path.parent/'request.json');worker=read(path.parent/'rank-0.json')
        assert worker['status']=='passed'and worker['framework_source']==core
        assert request['parameters']['request']==s.request.portable()
        assert started['parameters_ref']['request_sha256']==sha(path.parent/'request.json')
        actual=started['revision']['files'];assert all(actual.get(k)==v for k,v in files.items())
        assert all(sha(path.parents[1]/'source'/k)==v for k,v in actual.items())
        jobs.append({'path':str(path),'request_sha256':sha(path.parent/'request.json'),'revision':started['revision']['revision'],
            'action':request['parameters']['action'],'entry_sha256':sha(request['entry']),'source_files':actual})
    assert len(jobs)==5
    m=t['measurement'];report={'at':datetime.now(timezone.utc).isoformat(),'finished_at':final['finished_at'],'core':core,
        'terminal_sha256':sha(root/'followup-completed.json'),'source_check':sr,'source_check_sha256':sha(root/'source-check.json'),
        'source_reference_scope':'Unchanged real CUDA controls compared by reviewed manual external mapping; candidate full comparisons to those controls all zero. Arbitrary extra owners not certified.',
        'frozen_files_verified':len(prior['frozen_identities_verified']),'baseline_source_files':files,
        'trial':{'path':str(trials[0]),'sha256':sha(trials[0]),'id':t['trial'],'decision':t['decision'],'publication':t['publication'],
            'source_revision':current.id,'revision':t['revision'],'bundles':t['bundles'],'quality':t['quality'],'coverage_errors':t['coverage_errors'],
            'measurement':m,'pairs':pairs,'gain_percent':100*(1-m['objective']/m['reference'])},
        'native':natives,'jobs':jobs,'counts':{'clean_endpoints':4,'clean_samples':20,'quality_checks':8,'source_checks':2},
        'pointers':{'source_current':current.id,'independent_current':s.revisions.current().id,'original_pretrial_current':pretrial.id},
        'scope':'Independent pass of fact-assisted fresh-adapter continuation, not unaided original-entry success, new agent, additional search, prompt causal superiority, multiple active bundle proof or unique root-cause isolation. Search43.9895% remains separately frozen. Profiles are separate perturbed periods; no score backfill or host/critical-path subtraction.',
        'df':subprocess.check_output(['df','-B1','/root/autodl-tmp'],text=True)}
    write_json(root/'terminal-audit.json',report)
    print(json.dumps({'audit':str(root/'terminal-audit.json'),'sha256':sha(root/'terminal-audit.json'),'bytes':(root/'terminal-audit.json').stat().st_size,
        'gain_percent':report['trial']['gain_percent'],'native':natives,'counts':report['counts']}))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True);audit(p.parse_args().root)
