"""Read-only audit of fixed same-core goal contrast, retaining every sample."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import subprocess

from kernel_optimizer.config import load_config
from kernel_optimizer.project.measurement import paired_decision, valid_run
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.runtime.execution import write_json
from scripts.experiments.v6_audit_fsdp_20261028 import read, sha
from scripts.experiments.v6_ddp_semantics_20261027 import disk_observation
from scripts.experiments.v6_qwen_goal_contrast_20261028 import ORDER, preflight
from types import SimpleNamespace


def audit(root):
    protocol=read(root/'protocol.json'); terminal=read(root/'followup-completed.json')
    assert terminal['phase']=='completed' and protocol['order']==ORDER
    config=load_config(root/'private-config.yaml',[])
    args=SimpleNamespace(ttft=Path(protocol['identities']['T']['source']),decode=Path(protocol['identities']['D']['source']))
    fresh=preflight(args,config)
    for k in ['identities','baseline_files','reference_tensor_equality','contract_differences']:
        assert fresh[k]==protocol[k],k
    assert fresh['core']==protocol['core']
    report={'at':datetime.now(timezone.utc).isoformat(),'completed_at':terminal['finished_at'],
        'terminal_sha256':sha(root/'followup-completed.json'),'core':protocol['core'],
        'source_identity_rechecked':fresh['identities'],'reference_tensor_equality':fresh['reference_tensor_equality'],
        'order':ORDER,'endpoints':{},'disk':disk_observation(Path('/root/autodl-tmp'))}
    endpoints={};pids=[]
    log=[json.loads(line) for line in (root/'controller.log').read_text().splitlines() if line.startswith('{')]
    assert [x['endpoint']for x in log]==ORDER
    assert all(x['status']=='completed' for x in log)
    for name in ORDER:
        branch=name.split('-')[0]; run=read(root/(name+'.json')); endpoints[name]=run
        assert valid_run(run) and len(run['ranks'])==1
        rank=run['ranks'][0];pids.append(rank['pid']); result=rank['result']
        assert rank['exit_code']==0 and rank['framework_source']==protocol['core']
        assert result['math_policy_unchanged'] and len(result['samples'])==5
        assert len(result['checks'])==2 and all(c['passed']and c['max_abs_error']==0 for c in result['checks'])
        search=ProjectSearch(config,root/branch);revision=search.revisions.current()
        identity=protocol['identities']['T' if branch in ['T','BT'] else 'D']
        expected=identity['baseline' if branch.startswith('B') else 'current']
        assert revision.id==expected and revision.contract_ref==identity['contract_ref']
        job=root/branch/'jobs'/name;started=read(job/'run/rank-0-started.json'); execution=read(job/'run/request.json')
        assert started['revision']['revision']==expected
        assert started['parameters_ref']['request_sha256']==sha(job/'run/request.json')
        assert started['revision']['bundle_hashes']==result['bundle_hashes']
        files={f.path:f.sha256 for f in revision.files}
        assert started['revision']['files']==files
        assert all(sha(job/'source'/path)==value for path,value in files.items())
        actual_request=execution['parameters']['request']; expected_request=search.request.model_copy(update={'devices':['0']}).portable()
        assert actual_request==expected_request
        assert execution['parameters']['action']=='evaluate' and execution['parameters']['reference_dir']==identity['controls_path']
        assert execution['env']=={} and execution['launcher']==[]
        assert set(result['coverage'])==set(revision.bundles)
        for coverage in result['coverage'].values():
            assert not coverage['fallbacks'] and all(coverage['counts'].get(k,0)>0 for k in coverage['required'])
        for field,value in result['metrics'].items():
            assert abs(statistics.median(s[field]for s in result['samples'])-value)<1e-7,field
        report['endpoints'][name]={'pid':rank['pid'],'run_sha256':sha(root/(name+'.json')),
            'revision':started['revision'],'request_sha256':sha(job/'run/request.json'),
            **{k:result[k]for k in ['checks','samples','metrics','coverage','setup','bundle_hashes','diagnostics']}}
    assert len(set(pids))==8
    scores=read(root/'diagnostic-scores.json')['scores']; recalculated={}
    for name,parent,candidate in [('T_vs_D','T','D'),('BT_vs_T','BT','T'),('BD_vs_D','BD','D')]:
        pairs=[{'parent':endpoints[parent+'-'+i],'candidate':endpoints[candidate+'-'+i]}for i in ['1','2']]
        recalculated[name]={}
        for goal,key in [('ttft','T'),('decode','D')]:
            search=ProjectSearch(config,root/key)
            recalculated[name][goal]=paired_decision(pairs,search.request.objective,search.prepared['evaluation'])
    assert scores==recalculated
    report['diagnostic_scores']=scores
    report['counts']={'fresh_workers':len(pids),'clean_samples':40,'full_quality_checks':16,
        'compared_elements_per_quality_check':4861988}
    report['endpoint_medians']={key:{field:statistics.median(report['endpoints'][key+'-'+i]['metrics'][field]for i in ['1','2'])
        for field in ['ttft_ms','decode_tokens_per_second','end_to_end_ms','peak_allocated_bytes',
            'framework.allocated_bytes','framework.lifecycle_peak_allocated_bytes']}for key in ['BT','T','D','BD']}
    # Hash the actual frozen evaluation implementation as well as the worker core.
    freeze=Path(read(root/'launch.json')['argv'][3]).parents[2]
    report['evaluation_source']={str(p.relative_to(freeze)):sha(p)for p in (freeze/'src/kernel_optimizer/project').rglob('*.py')
        if 'evaluation' in p.parts or p.name in ['evaluation.py','inner_loop.py']}
    report['limits']=['Fixed previously generated versions, not a new autonomous cohort or formal acceptance.',
        'Two baseline contracts have identical executable source/quality but different time positions; observed baseline timing variation remains.',
        'Same measurement core/window; historical generation processes were not a goal-only randomized snapshot A/B.',
        'Coverage routes reflect quality-phase telemetry. No new native trace was collected; do not reuse old graph counts as present measurements.',
        'All 40 samples, previous independent decisions and old Qwen 80-sample rejection remain unchanged.']
    report['next_decision']='Close fixed contrast; next same-adapter end_to_end_ms variant adds an explicit original-baseline TTFT<=1.05 constraint, with real contract adaptation and fresh autonomous search from baseline only. No old bundles or developer contrast diagnoses supplied.'
    write_json(root/'terminal-audit.json',report)
    print(json.dumps({'audit':str(root/'terminal-audit.json'),'counts':report['counts'],
        'endpoint_medians':report['endpoint_medians'],'evaluation_source_files':len(report['evaluation_source'])}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--root',type=Path,required=True)
    audit(parser.parse_args().root)
