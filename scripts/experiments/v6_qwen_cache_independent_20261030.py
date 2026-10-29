"""One fixed complete-version AB/BA closure, then return priority to communication."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import traceback

from kernel_optimizer.config import load_config
from kernel_optimizer.project.branches import fork_run
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.project.versioning import promote_version
from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.runtime.provenance import framework_source
from scripts.experiments.v6_audit_fsdp_20261028 import read, sha
from scripts.experiments.v6_ddp_semantics_20261027 import disk_observation
from scripts.experiments.v6_audit_qwen_cache_constraint_20261030 import CODE, ACCEPTED


def now():
    return datetime.now(timezone.utc).isoformat()


def identities(args, config):
    audit = read(args.parent/'terminal-audit.json')
    search = ProjectSearch(config,args.parent/'qwen-continuation')
    assert framework_source() == audit['core']
    assert audit['trial']['decision']['accepted'] and audit['trial']['candidate_sha256'] == CODE
    assert search.revisions.current().id == ACCEPTED == audit['pointers']['current']
    assert list(search.revisions.current().bundles) == audit['pointers']['bundles']
    assert sha(audit['trial']['candidate_path']) == CODE
    frozen = audit['frozen_identities_verified']
    assert all(sha(p) == h for p,h in frozen.items())
    native = audit['native']
    assert sha(native['sqlite']) == native['sqlite_sha256'] and sha(native['normalized_path']) == native['normalized_sha256']
    return search,audit


def source_reuse(args,search):
    prior = args.parent.parent/'qwen-device-refresh-fixed-20261030'
    path = prior/'source-check.json'; record = read(path); sr = record['ranks'][0]['result']
    reqpath = prior/'source-run/request.json'; req = read(reqpath)['parameters']
    assert record['status'] == 'passed' and sr['valid'] and len(sr['checks']) == 2
    assert all(c['passed'] and c['max_abs_error'] == 0 and c['exact_tokens'] and c['finite_logits'] for c in sr['checks'])
    assert all(not c['passed'] for c in sr['negative_checks'].values())
    assert sha(req['adapter']) == sr['adapter_sha256'] == search.prepared['adapter_sha256']
    assert sha(req['reference']) == sr['source_reference_sha256']
    assert sha(req['controls_reference']) == sr['controls_reference_sha256']
    assert Path(req['controls_reference']).parent == Path(search.prepared['controls_path'])
    return {'at':now(),'status':'completed','prior_check':str(path),'prior_check_sha256':sha(path),
        'prior_request_sha256':sha(reqpath),'files':{p:sha(p) for p in [req['adapter'],req['reference'],req['controls_reference']]},
        'prior_full_checks':sr['checks'],'prior_negative_checks':sr['negative_checks'],
        'scope':'Exact-hash reuse only; no fresh CPU tensor comparison/model/controls run. Prior reviewed external generated_tokens/step_logits mapping and additional-owner limits remain.'}


def main(args):
    os.umask(0o077); args.output.mkdir(parents=True,exist_ok=True)
    assert not (args.output/'protocol.json').exists()
    config = load_config(args.config,[]); config.usage.mode = 'observe_only'
    search,audit = identities(args,config); reused = source_reuse(args,search)
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    # Only four small evaluate outputs: no discovery/profile/new controls or model copies.
    free = int(subprocess.check_output(['df','-B1','--output=avail','/root/autodl-tmp'],text=True).splitlines()[-1])
    assert free > 8*1024**3 + 32*1024**2
    frozen = dict(audit['frozen_identities_verified'])
    for p in [args.parent/'terminal-audit.json',args.parent/'followup-completed.json',
              Path(audit['trial']['candidate_path']),Path(audit['native']['sqlite']),Path(audit['native']['normalized_path']),
              search.revisions.pointer,Path(reused['prior_check'])]: frozen[str(p)] = sha(p)
    frozen.update(reused['files'])
    write_json(args.output/'source-evidence-reuse.json',reused)
    write_json(args.output/'native-candidate-reuse.json',{'status':'reused','native':audit['native'],
        'scope':'Exact accepted-code current-device SQLite already independently recomputed. No new native invocation. Separate perturbed capture, not acceptance score or critical-path attribution.'})
    protocol = {'at':now(),'core':framework_source(),'parent':str(args.parent),'parent_audit_sha256':sha(args.parent/'terminal-audit.json'),
        'source_revision':ACCEPTED,'candidate_sha256':CODE,'original_baseline':search.prepared['revision'],
        'original_ttft_anchor':23.285755,'original_ttft_limit_ms':24.45004275,'devices':search.request.devices,
        'frozen_identities':frozen,'sequence':['hash-only prior source evidence reuse','one B/C/C/B full-version validation'],
        'publish':False,'new_agents':0,'new_searches':0,'new_native_invocations':0,
        'space_budget':{'free_before':free,'additional_evaluate_output_budget_bytes':32*1024**2,'warning_floor_bytes':8*1024**3,
                        'store_artifacts':'hardlinks on same filesystem; controls shared read-only; four small worker result sets'},
        'scope':'Necessary Qwen closure after real search acceptance; original full work/quality/constraints/noise unchanged. Preserve any rejection, no retry. Then prioritize current-device FSDP baseline and one ordinary communication trial; DDP next. No automatic Qwen continuation or production adoption.',
        'disk':disk_observation(Path('/root/autodl-tmp'))}
    write_json(args.output/'protocol.json',protocol)
    results = {'source_reuse':{'status':'completed','sha256':sha(args.output/'source-evidence-reuse.json')}}
    write_json(args.output/'state.json',{'at':now(),'phase':'running','stage':'independent'})
    try:
        target = args.output/'qwen-independent'; base = fork_run(search.output,target,revision_id=search.prepared['revision'])
        check = ProjectSearch(config,target)
        assert not base.bundles and check.prepared == search.prepared and check.request == search.request
        write_json(args.output/'preparation-reuse.json',{'source':str(search.output),'source_revision':ACCEPTED,
            'baseline':base.id,'contract_ref':base.contract_ref,'controls_path':search.prepared['controls_path'],
            'request_sha256':sha(target/'request.json'),'prepared_sha256':sha(target/'prepared.json'),
            'adapter_sha256':search.prepared['adapter_sha256'],'sequence':'B/C/C/B fresh workers; two continuous checks, two warmups, all five samples',
            'publish':False})
        result = promote_version(check,search.output,revision_id=ACCEPTED,publish=False)
        assert check.revisions.current().id == base.id
        results['independent'] = {'status':'completed',**{k:result[k] for k in ['trial','revision','parent','quality','coverage_errors','measurement','decision','publication']}}
    except Exception:
        results['independent'] = {'status':'error','traceback':traceback.format_exc()}
    try:
        identities(args,config)
        assert all(sha(p) == h for p,h in frozen.items())
        results['preservation'] = {'status':'completed','frozen_files':len(frozen),'source_current':search.revisions.current().id}
    except Exception:
        results['preservation'] = {'status':'error','traceback':traceback.format_exc()}
    terminal = {'phase':'completed' if all(v['status'] == 'completed' for v in results.values()) else 'completed_with_errors',
        'finished_at':now(),'results':results,'disk':disk_observation(Path('/root/autodl-tmp'))}
    write_json(args.output/'followup-results.json',results)
    write_json(args.output/'state.json',terminal); write_json(args.output/'followup-completed.json',terminal)
    print(json.dumps(terminal),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    for name in ['config','output','parent']: parser.add_argument('--'+name,type=Path,required=True)
    main(parser.parse_args())
