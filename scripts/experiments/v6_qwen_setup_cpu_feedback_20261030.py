"""One ordinary trial with original rejection feedback and an unchanged probe-artifact index."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

from kernel_optimizer.config import load_config
from kernel_optimizer.models.project import ExecutionPlan, GroupRegistry
from kernel_optimizer.project.branches import fork_run
from kernel_optimizer.project.feedback import candidate_artifacts, route
from kernel_optimizer.project.observation_artifacts import feedback_artifacts
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.runtime.execution import resolve_devices, write_json
from kernel_optimizer.runtime.provenance import framework_source
from scripts.experiments.v6_audit_fsdp_20261028 import read, sha
from scripts.experiments.v6_ddp_semantics_20261027 import disk_observation


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def prepare(args, config):
    parent_protocol = read(args.parent/'protocol.json')
    completed = read(args.parent/'followup-completed.json')
    assert completed['phase'] == 'completed' and completed['results']['preservation']['unchanged']
    assert sha(args.parent/'terminal-audit.json') == '34724d510e99df73d6cbfcc82ebbc83b639ec55c0aec2abea3bc1600b944273d'
    assert framework_source() == parent_protocol['core']
    source = ProjectSearch(config, args.parent/'qwen-continuation')
    current = source.revisions.current()
    assert current.id == parent_protocol['parent_revision'] and not current.bundles
    assert resolve_devices(source.request.devices) == source.request.devices
    raw = (source.output/'latest-feedback.json').read_bytes()
    original = json.loads(raw)
    assert original['trial'] == 'trial-bf84d454d6' and original['decision'] == {'accepted':False,'reasons':['no_valid_parameter_candidate']}
    assert 'supplemental_evidence' not in original
    cpu_path = args.observation/'audit/cpu-control-flow/result.json'
    cpu = read(cpu_path)
    assert sha(cpu_path) == 'dffa6459d1a0772999041e5704e652a9424e79d6bb60f70513f3393b8979f7d5'
    assert cpu['candidate_sha256'] == '4693d1472b012cfa786209e8ce2ede85a84ae443d7cc1c2212d2680ff42dcd38' and not cpu['cuda_initialized']
    fixture = Path(__file__).resolve().parent/'v6_qwen_setup_feedback_cpu_20261030.py'
    assert sha(fixture) == '9fcbc9ff90a84120393877b69c9f8b2b654a19f5b9121707e1ef2267dc9d314b'
    paths = [('cpu_result',cpu_path),('cpu_fixture',fixture)]
    for path in sorted(source.output.glob('jobs/candidate-diagnostics-*/run/candidate-diagnostics-rank-0.json')):
        paths.append(('original_run_only_sidecar',path))
    assert len(paths) == 4
    index = {'schema_version':'cpu-interface-evidence.v1',
        'scope':'Previously completed CPU controls and original GPU run-only sidecars for the exact prior candidate. No rerun or implementation supplied.',
        'artifacts':[{'kind':label,'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size}for label,path in paths]}
    index_raw = (json.dumps(index,indent=2)+'\n').encode()
    supplemental = {'cpu_interface_observations':{
        'scope':'Developer-created bounded CPU controls of the exact previous generated code. This is explicit assistance, not an unaided result. Setup/verify are stubbed only in the separate build-success control. These facts do not certify full GPU quality, historical per-call GPU cache state, performance or a unique numerical cause. Original GPU failures and all thresholds remain authoritative.',
        'result':cpu,
        'index':{'path':str(args.output/'evidence-index.json'),'sha256':digest(index_raw),'bytes':len(index_raw)}}}
    end = raw.rfind(b'}')
    augmented = raw[:end] + b',\n  "supplemental_evidence": ' + json.dumps(supplemental,indent=2).encode() + b'\n' + raw[end:]
    parsed = json.loads(augmented)
    assert {k:v for k,v in parsed.items()if k!='supplemental_evidence'} == original
    old_files, old_compact = feedback_artifacts(raw.decode())
    new_files, new_compact = feedback_artifacts(augmented.decode())
    assert old_files == new_files
    registry = source.revisions.load(current.registry_ref,GroupRegistry)
    plan = source.revisions.load(current.plan_ref,ExecutionPlan)
    routing=[]
    for label,kwargs in [('selector',{}),('historical_selected_candidate',{'selected_groups':original['selection']['groups'],'plan':plan})]:
        before=json.loads(route(old_compact,current_revision=current.id,registry=registry,**kwargs))
        after=json.loads(route(new_compact,current_revision=current.id,registry=registry,**kwargs))
        assert after.pop('supplemental_evidence') == supplemental
        assert before.pop('supplemental_evidence') is None and before == after
        routing.append({'role':label,'only_supplemental_evidence_differs':True})
    historical=candidate_artifacts(augmented.decode(),source.store)
    code=historical['previous-candidate/candidate.py']
    assert digest(code if isinstance(code,bytes)else code.encode()) == cpu['candidate_sha256']
    frozen=parent_protocol['frozen_identities'].copy()
    extra=[p for _,p in paths]+[args.parent/name for name in ['protocol.json','terminal-audit.json','followup-completed.json']]
    extra += [source.output/name for name in ['latest-feedback.json','prepared.json','request.json','discovery.json','project/current.json','trials/trial-bf84d454d6/result.json']]
    extra += list(source.output.glob('generation-snapshots/*.json'))
    for job in (source.output/'jobs').iterdir():
        extra += [job/'run'/name for name in ['request.json','receipt.json','rank-0.json']]
    for path in extra:frozen[str(path)]=sha(path)
    assert all(sha(path)==value for path,value in frozen.items())
    transport={'original_bytes':len(raw),'original_sha256':digest(raw),'new_bytes':len(augmented),'new_sha256':digest(augmented),
        'bytes_added':len(augmented)-len(raw),'original_prefix_before_closing_brace_preserved':True,'all_original_fields_equal':True,
        'packaged_original_artifacts_equal':True,'route_checks':routing,'historical_candidate_sha256':cpu['candidate_sha256'],
        'scope':'CPU route check only. New developer factual assistance is explicitly declared; real initial snapshots and consumption remain to audit.'}
    return source,index_raw,augmented,frozen,transport


def main(args):
    os.umask(0o077)
    config = load_config(args.config, [])
    config.usage.mode = 'observe_only'
    source, index_raw, feedback_raw, frozen, transport = prepare(args, config)
    if args.preflight:
        print(json.dumps({'status': 'passed', 'core': framework_source(), 'frozen_files': len(frozen),
            'parent': source.revisions.current().id, 'devices': source.request.devices, 'transport': transport,
            'index_bytes': len(index_raw), 'rounds': 1, 'discovery': 'reuse completed current-device empty-parent discovery'}))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    assert not (args.output / 'protocol.json').exists()
    (args.output / 'evidence-index.json').write_bytes(index_raw)
    (args.output / 'diagnostic-feedback.json').write_bytes(feedback_raw)
    write_json(args.output / 'transport-preflight.json', transport)
    write_json(args.output / 'protocol.json', {'at': now(), 'core': framework_source(), 'parent': str(args.parent),
        'cpu_observation': str(args.observation), 'source': str(source.output), 'parent_revision': source.revisions.current().id,
        'baseline': source.prepared['revision'], 'devices': source.request.devices, 'rounds': 1, 'frozen_identities': frozen,
        'index_sha256': digest(index_raw), 'feedback_sha256': digest(feedback_raw),
        'current_device_ttft_anchor': 23.285755, 'constraint_limit': 24.45004275,
        'scope': 'One ordinary trial with original failure feedback plus declared developer CPU interface observations. CPU stubs are not model/GPU quality or a historical GPU cache trace. No manual candidate/implementation, prompt change, source/reference rerun or initial discovery. Not unaided original-entry or same-snapshot A/B. No independent/native stage scheduled; planner may request existing collectors.',
        'disk': disk_observation(Path('/root/autodl-tmp'))})
    results = {}
    try:
        branch = args.output / 'qwen-continuation'
        parent = fork_run(source.output, branch)
        search = ProjectSearch(config, branch)
        assert not parent.bundles and parent.id == source.revisions.current().id
        assert search.prepared == source.prepared and search.request == source.request
        assert sha(branch / 'discovery.json') == sha(source.output / 'discovery.json')
        write_json(args.output / 'preparation-reuse.json', {'source': str(source.output), 'parent': parent.id,
            'baseline': search.prepared['revision'], 'controls_path': search.prepared['controls_path'], 'initial_bundles': [],
            'request_sha256': sha(branch / 'request.json'), 'discovery_sha256': sha(branch / 'discovery.json'),
            'discovery': 'Completed current-device original empty-parent discovery reused, no initial GPU rediscovery or new controls.'})
        argv = [sys.executable, '-B', '-m', 'kernel_optimizer.cli', '--config', str(args.config), 'optimize-project',
            '--request', str(branch / 'request.json'), '--output', str(branch), '--resume', '--rounds', '1',
            '--feedback', str(args.output / 'diagnostic-feedback.json')]
        write_json(args.output / 'search-protocol.json', {'at': now(), 'argv': argv, 'feedback_sha256': digest(feedback_raw),
            'scope': 'Existing default selector/candidate/scopes/parameter exploration and ordinary feedback routing. API observe-only, no new cap. Preserve every failure, no retry-until-pass or automatically installed probe.'})
        write_json(args.output / 'state.json', {'at': now(), 'phase': 'running', 'stage': 'one-ordinary-feedback-trial'})
        with (args.output / 'search-controller.log').open('wb') as log:
            process = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT)
        current = ProjectSearch(config, branch).revisions.current()
        results['search'] = {'status': 'completed' if process.returncode == 0 else 'error', 'returncode': process.returncode,
            'current': current.id, 'bundles': list(current.bundles),
            'trials': [{'path': str(p), 'sha256': sha(p), 'decision': read(p).get('decision')} for p in sorted((branch / 'trials').glob('*/result.json'))],
            'independent_acceptance': 'not_run_in_this_batch'}
    except Exception:
        results['controller'] = {'status': 'error', 'traceback': traceback.format_exc()}
    same = all(sha(path) == value for path, value in frozen.items())
    results['preservation'] = {'status': 'passed' if same else 'failed', 'files': len(frozen), 'unchanged': same}
    terminal = {'phase': 'completed' if same and results.get('search', {}).get('status') == 'completed' else 'completed_with_errors',
        'finished_at': now(), 'results': results, 'disk': disk_observation(Path('/root/autodl-tmp'))}
    write_json(args.output / 'followup-results.json', results)
    write_json(args.output / 'state.json', terminal)
    write_json(args.output / 'followup-completed.json', terminal)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    for name in ['config', 'parent', 'observation', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--preflight', action='store_true')
    main(parser.parse_args())
