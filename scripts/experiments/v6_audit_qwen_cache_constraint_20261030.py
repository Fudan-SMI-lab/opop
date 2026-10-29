"""Read-only terminal audit; no model invocation or candidate modification."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import subprocess

from kernel_optimizer.config import load_config
from kernel_optimizer.models.project import ExecutionPlan, GroupRegistry
from kernel_optimizer.project.feedback import route
from kernel_optimizer.project.generation_snapshot import load
from kernel_optimizer.project.measurement import paired_decision
from kernel_optimizer.project.observation_artifacts import feedback_artifacts
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.runtime.provenance import framework_source
from scripts.experiments.v6_audit_fsdp_20261028 import read, sha
from scripts.experiments.v6_audit_qwen_end_to_end_20261029 import endpoint
from scripts.experiments.v6_audit_qwen_independent_20261029 import native

CODE = 'dbec45af8e995ca52c812806a856b4b85e303e7fc9f808f49086362c45dbdad8'
OLD = '7e78b82dd21391e3db3d6292e3798feff1908f53259aa732a3c011660c28fb92'
ACCEPTED = '2a42a6addc9fecb6808cad53156da916214937bb361077b79cec4f7210372b3b'


def digest(x):
    return hashlib.sha256(x if isinstance(x, bytes) else x.encode()).hexdigest()


def main(root):
    protocol = read(root/'protocol.json'); terminal = read(root/'followup-completed.json')
    assert terminal['phase'] == 'completed' and framework_source() == protocol['core']
    frozen = protocol['frozen_identities']
    assert len(frozen) == 536 and all(sha(p) == h for p,h in frozen.items())
    q = root/'qwen-continuation'; config = load_config(root/'private-config.yaml', [])
    search = ProjectSearch(config,q); current = search.revisions.current()
    parent = search.revisions.revision(protocol['parent'])
    original = ProjectSearch(config,Path(protocol['source']))
    assert current.id == ACCEPTED and original.revisions.current().id == parent.id
    assert not parent.bundles and len(current.bundles) == 1
    for name in ['request.json','prepared.json','discovery.json']:
        assert (q/name).read_bytes() == (original.output/name).read_bytes()
    assert current.contract_ref == parent.contract_ref and current.registry_ref == parent.registry_ref
    trialpath = q/'trials/trial-22524f5ab8/result.json'; trial = read(trialpath)
    assert trial['decision']['accepted'] and trial['decision']['reasons'] == []
    assert trial['decision']['revision'] == current.id and trial['decision']['parent'] == parent.id
    assert trial['publication'] == 'committed' and not trial['missing_coverage']
    assert trial['parameter_search']['space'] == {} and len(trial['parameter_search']['observations']) == 1
    assert paired_decision(trial['pairs'],search.request.objective,search.prepared['evaluation']) == trial['measurement']
    m = trial['measurement']; assert m['constraints'][0]['limit'] == 24.45004275
    pairs = [{k:endpoint(v,protocol['core']) for k,v in pair.items()} for pair in trial['pairs']]
    manual_pairs = []
    for i,pair in enumerate(pairs):
        values = {k:[s['end_to_end_ms'] for s in v['samples']] for k,v in pair.items()}
        med = {k:statistics.median(v) for k,v in values.items()}
        mad = {k:statistics.median(abs(x-med[k]) for x in v) for k,v in values.items()}
        gain = med['parent']-med['candidate']; noise = 2*max(mad.values())
        assert abs(gain-m['paired_gains'][i]) < 1e-8 and abs(noise-m['noise_bounds'][i]) < 1e-8
        manual_pairs.append({'values':values,'medians':med,'MAD':mad,'gain':gain,'noise':noise})
    events = [json.loads(x) for x in (q/'events.jsonl').read_text().splitlines()]
    agents = [e['payload'] for e in events if e['type'] == 'PROJECT_AGENT_FINISHED']
    assert len(agents) == 3 and all(a['contract_repairs'] == 0 for a in agents)
    private = root/'terminal-private-history'; inventory = []; roles = []
    packet, compact = feedback_artifacts((root/'diagnostic-feedback.json').read_text())
    registry = search.revisions.load(parent.registry_ref,GroupRegistry)
    plan = search.revisions.load(parent.plan_ref,ExecutionPlan)
    for agent in agents:
        snap = Path(agent['generation_snapshots'][0]); meta,prompt,files = load(snap)
        assert meta['framework_source'] == protocol['core'] and meta['fresh_session']
        sandbox = Path(meta['sandbox']); assert all((sandbox/n).read_bytes() == b for n,b in files.items())
        delivery = None
        if 'inputs/previous-feedback.txt' in files:
            is_candidate = meta['role'] == 'candidate'
            actual = json.loads(files['inputs/previous-feedback.txt'])
            routed = json.loads(route(compact,current_revision=parent.id,registry=registry,
                selected_groups=trial['selection']['groups'] if is_candidate else None,plan=plan if is_candidate else None))
            assert actual == routed and digest(files['inputs/previous-candidate/candidate.py']) == OLD
            for n,b in packet.items(): assert files['inputs/'+n] == (b.encode() if isinstance(b,str) else b)
            delivery = {'feedback_bytes':len(files['inputs/previous-feedback.txt']),
                'feedback_sha256':digest(files['inputs/previous-feedback.txt']), 'historical_code_sha256':OLD,
                'actual_registry_route_equal':True,'all_packaged_artifacts_equal':True}
        history = read(private/(agent['session']+'.json')); tools = read(private/(agent['session']+'-tools.json'))
        rows = []
        for i,t in enumerate(tools,1):
            row = {'number':i,'part':t['part'],'tool':t['tool'],'status':t['status'],'input':t['input'],
                'exit':t.get('metadata',{}).get('exit'),'output_sha256':digest(t.get('output','')),
                'output_bytes':len(t.get('output','').encode()),'traceback':'Traceback (most recent call last)' in t.get('output','')}
            rows.append(row)
            output_path = t.get('metadata',{}).get('outputPath')
            if output_path: assert sha(private/Path(output_path).name) == sha(output_path)
        inventory.append({'session':agent['session'],'tools':rows})
        roles.append({'role':agent['role'],'session':agent['session'],'snapshot_sha256':sha(snap),
            'inputs':len(files),'input_bytes':sum(map(len,files.values())), 'prompt_sha256':digest(prompt),
            'settings_sha256':digest(json.dumps(meta['tool_model_settings'],sort_keys=True)),
            'input_manifest':meta['files'],'delivery':delivery,'repairs':agent['contract_repairs'],
            'assistant_messages':sum(x['data'].get('role') == 'assistant' for x in history['message']),
            'tools':dict(Counter(t['tool'] for t in tools)),'usage':agent['usage'],
            'errors':[{k:t[k] for k in ['number','part','tool','exit','traceback','output_sha256']} for t in rows if t['traceback'] or t['exit'] not in [0,None]],
            'private_history_sha256':sha(private/(agent['session']+'.json'))})
    write_json(private/'review-inventory.json',inventory)
    ca = next(a for a in agents if a['role'] == 'candidate'); sandbox = Path(ca['sandbox'])
    candidate = sandbox/'candidate.py'; proposal = read(sandbox/'proposal.json')
    assert sha(candidate) == CODE and proposal['parameter_space'] == {} and not proposal['eliminated_calls']
    tools = read(private/(ca['session']+'-tools.json')); initial = tools[35]['input']['content']; rebuilt = initial
    for n in [37,38]:
        edit = tools[n-1]['input']; assert rebuilt.count(edit['oldString']) == 1
        rebuilt = rebuilt.replace(edit['oldString'],edit['newString'],1)
    assert rebuilt.encode() == candidate.read_bytes()
    write_json(private/'generated-before.json',{'tool':36,'sha256':digest(initial),'content':initial,'edits':[37,38],'final_equal':True})
    jobs = {}; clean = 0; checks = 0
    for job in sorted((q/'jobs').iterdir()):
        folder = job/'run'; rec = read(folder/'receipt.json'); req = read(folder/'request.json')
        rank = rec['ranks'][0]; result = rank['result']; started = read(folder/'rank-0-started.json')
        assert rec['status'] == rank['status'] == 'passed' and rank['framework_source'] == protocol['core']
        assert rank['binding']['torch_uuid'] == protocol['devices'][0] and result['valid']
        assert sha(req['entry']) == rec['entry_sha256']
        assert req['parameters']['request'] == search.request.portable()
        assert sha(req['parameters']['adapter']) == search.prepared['adapter_sha256']
        assert started['parameters_ref']['request_sha256'] == sha(folder/'request.json')
        manifest = read(Path(req['cwd'])/'.opop-revision.json')
        assert all(sha(Path(req['cwd'])/p) == h for p,h in manifest['files'].items())
        assert all(started['revision']['files'][p] == h for p,h in manifest['files'].items())
        bundles = [{'id':b['id'],'sha256':sha(b['file']),'parameters':b['parameters']} for b in req['parameters']['bundles']]
        assert all(b['sha256'] == CODE and b['parameters'] == {} for b in bundles)
        ep = endpoint(rec,protocol['core']) if 'samples' in result else None
        if ep:
            clean += len(ep['samples']); checks += len(ep['checks'])
            assert all(c['compared_elements'] == 4861988 for c in ep['checks'])
            for sample in ep['samples']:
                assert sample['generated_tokens'] == 32 and sample['decode_tokens'] == 31
                assert abs(sample['end_to_end_ms']-sample['ttft_ms']-sample['decode_ms']) < 1e-6
                assert abs(sample['decode_tokens_per_second']-31000/sample['decode_ms']) < 1e-8
        logs = {p.name:{'sha256':sha(p),'bytes':p.stat().st_size,
            'traceback':'Traceback (most recent call last)' in p.read_text(errors='replace')} for p in folder.glob('*.log')}
        jobs[job.name] = {'request_sha256':sha(folder/'request.json'),'receipt_sha256':sha(folder/'receipt.json'),
            'worker_sha256':sha(folder/'rank-0.json'),'entry_sha256':rec['entry_sha256'],'adapter_sha256':sha(req['parameters']['adapter']),
            'source_manifest_sha256':sha(Path(req['cwd'])/'.opop-revision.json'),'source_revision':started['revision']['revision'],
            'action':req['parameters']['action'],'bundles':bundles,'status':rank['status'],'valid':result['valid'],
            'endpoint':ep,'logs':logs}
    assert len(jobs) == 13 and clean == 45 and checks == 18
    np = q/'jobs/nsys-8deb650bbd/run/nsys-normalized.json'; nr = read(np)[0]; nat = native(nr)
    nat.update(normalized_path=str(np),normalized_sha256=sha(np),sqlite_sha256=nat['sha256'])
    assert nat['kernel_count'] == 66089 and nat['graph_kernel_count'] == 66056
    assert sorted(nat['graph_nodes_per_correlation'].values()) == [2062]*31+[2134]
    profile_folder = q/'jobs/trial-22524f5ab8-evaluation-2d8ee801-profile/run'
    dp = profile_folder/'discovery-rank-0.json'; disc = read(dp); raw = read(Path(disc['trace']))
    assert sha(disc['trace']) == disc['trace_sha256']
    rawkernels = [e for e in raw['traceEvents'] if e.get('cat') == 'kernel']
    assert len(rawkernels) == len(disc['kernels']) == disc['coverage']['cuda_kernels']
    assert Counter((e['name'],e['ts'],e['dur']) for e in rawkernels) == Counter((e['name'],e['start_us'],e['duration_us']) for e in disc['kernels'])
    profile = {'sha256':sha(dp),'trace_sha256':disc['trace_sha256'],'calls':len(disc['calls']),'groups':len(disc['groups']),
        'kernel_count':len(rawkernels),'graph_kernel_count':sum(bool(k['launch_resources'].get('graph node id')) for k in disc['kernels']),
        'phase_counts':dict(Counter(k['phase'] for k in disc['kernels'])),'coverage':disc['coverage'],'timing_scope':disc['timing_scope']}
    tp = q/'jobs/telemetry-0b2693b18c/run/telemetry-rank-0.json'; tele = read(tp)
    assert tele == read(tp.parent/'rank-0.json')['result']['telemetry']
    telemetry = {'sha256':sha(tp),'kind':tele['kind'],'clock_domain':tele['clock_domain'],'interpretation':tele['interpretation'],'windows':[]}
    for w in tele['windows']:
        assert abs((w['end_ns']-w['start_ns'])/1e6-w['wall_ms']) < 1e-8
        readings = [x for sample in w['samples'] for x in sample]
        telemetry['windows'].append({'repeat':w['repeat'],'wall_ms':w['wall_ms'],'samples':len(w['samples']),
            'readings':len(readings),'statuses':dict(Counter(x['status'] for x in readings)),
            'sampling_failures':w['sampling_failures'],'nvlink_before':w['nvlink_before']['status'],'nvlink_after':w['nvlink_after']['status']})
    cp = q/'jobs/candidate-diagnostics-f4a4578ad7/run/candidate-diagnostics-rank-0.json'; diagnostic = read(cp)
    assert all(sha(e['file']) == CODE for e in diagnostic['events'])
    report = {'at':datetime.now(timezone.utc).isoformat(),'finished_at':terminal['finished_at'],'terminal_reason':'batch_finished',
        'core':protocol['core'],'frozen_identities_verified':frozen,'frozen_files':len(frozen),'all_frozen_unchanged':True,
        'pointers':{'current':current.id,'source_current':parent.id,'bundles':list(current.bundles),'prepared':search.prepared['revision']},
        'trial':{'id':trial['trial'],'sha256':sha(trialpath),'decision':trial['decision'],'publication':trial['publication'],
            'selection':trial['selection'],'candidate_path':str(candidate),'candidate_sha256':CODE,'candidate_bytes':candidate.stat().st_size,
            'proposal_sha256':sha(sandbox/'proposal.json'),'proposal_bytes':(sandbox/'proposal.json').stat().st_size,
            'parameter_space':{},'parameter_observations':1,'measurement':m,'manual_pairs':manual_pairs,
            'gain_percent':100*(1-m['objective']/m['reference'])},
        'roles':roles,'jobs':jobs,'native':nat,'profile':profile,'telemetry':telemetry,
        'run_only_diagnostic':{'sha256':sha(cp),'events':diagnostic['events'],'scope':'run only; guard failures in negative setup controls are not outer quality failures'},
        'counts':{'formal_clean_samples':25,'formal_full_checks':10,'binding_clean_samples':20,'binding_full_checks':8,'workers':13},
        'event_counts':dict(Counter(e['type'] for e in events)),
        'findings':{
            'acceptance':'One empty parameter joint prefill/decode bundle accepted by existing search rules; independent full-version validation not yet run. Baseline second paired TTFT25.417243 is retained, not reanchored.',
            'delivery':'Actual selector/candidate registry routes, full historical7e78 code and all initial files match. New selection changes candidate inputs; not single-variable A/B. No new developer repair or manual group supplied.',
            'access':'Complete recorded tools/output/probes reviewed: packaged feedback/code, source, library, own code/tests. Selector11/12 and candidate9/13 read quality/TTFT constraint facts; no new index/external B/C files opened. No observed later audit/accepted/reviewer answer access; filesystem is not hermetic.',
            'tests':'Candidate40 CPU guard/delta test failed because expected2 for delta1;41 fixes test only. Import/schema/CPUguards do not prove CUDA/binding correctness. Initial tool36 plus edits37/38 exactly reconstruct final bytes. Own __pycache__ removed; no original GPU model probe via role tools.',
            'code':'Manual stock-op prefill/decode, shared masterKV, prebuilt reused DynamicCache/DynamicLayer views, prefill-first+31decode Graphs. No Triton/global capture patch. Stage verification failure now clears flags. Runtime replay exceptions can switch to eager while retaining graph-labelled routes; native rows independently corroborate this capture only.',
            'limits':'Layer0 key/object lineage does not establish all-layer/value mixed aliases or arbitrary owners/concurrency. Direct setup chains do not exercise original framework signature dispatch; teacher stride mismatch persists. Final KV gate covers initial oracle chain, not all chains. Partial installer mapping can still fail declared-site validation.',
            'claims_not_adopted':'Profile inclusive-kernel floor/clean wall subtraction/unique host bottleneck, universal TTFT or quality/lifetime guarantees, baseline-pass observer bound, and memory gain. Runtime/lifecycle peaks increase. Native and Torch captures are distinct perturbed periods, not clean scores/isolated critical path.',
            'next':'User communication priority: close only independent validation of this exact accepted version; no Qwen chain extension. Then current-device FSDP preparation and one ordinary communication trial, DDP next. Original source-mapping/developer assistance boundaries remain.'},
        'audit_tooling_error':'One local PowerShell nested quoting command raised SyntaxError before SSH. First audit incorrectly expected only accepted/reasons in decision; actual decision also records revision/parent/report_ref. Reader corrected, original attempt retained; no model/candidate/fixture rerun.',
        'df':subprocess.check_output(['df','-B1','--output=size,used,avail','/root/autodl-tmp'],text=True),
        'gpu_processes':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],text=True)}
    assert all(sha(p) == h for p,h in frozen.items())
    write_json(root/'terminal-audit.json',report)
    public = {k:v for k,v in report.items() if k not in ['frozen_identities_verified','roles','jobs','run_only_diagnostic']}
    public['trial'] = {k:v for k,v in report['trial'].items() if k != 'selection'}
    public['roles'] = [{k:a[k] for k in ['role','snapshot_sha256','inputs','input_bytes','prompt_sha256','settings_sha256','delivery','repairs','assistant_messages','tools','errors','private_history_sha256']} |
        {'provider_estimated_cost':a['usage']['reported_cost']} for a in roles]
    public['jobs'] = {n:{k:v for k,v in j.items() if k != 'endpoint'} |
        ({'metrics':j['endpoint']['metrics'],'samples':j['endpoint']['samples'],'checks':j['endpoint']['checks'],'setup':j['endpoint']['setup'],'coverage':j['endpoint']['coverage']} if j['endpoint'] else {}) for n,j in jobs.items()}
    public['run_only_diagnostic'] = {'sha256':sha(cp),'event_count':len(diagnostic['events']),'scope':report['run_only_diagnostic']['scope']}
    public['terminal_audit_sha256'] = sha(root/'terminal-audit.json')
    write_json(root/'public-terminal-audit.json',public)
    print(json.dumps({'audit_sha256':public['terminal_audit_sha256'],'public_sha256':sha(root/'public-terminal-audit.json'),
        'public_bytes':(root/'public-terminal-audit.json').stat().st_size,'gain_percent':report['trial']['gain_percent'],
        'roles':[(a['role'],a['inputs'],a['assistant_messages'],sum(a['tools'].values()),a['usage']['reported_cost']) for a in roles],
        'native':nat,'counts':report['counts']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__); parser.add_argument('--root',type=Path,required=True)
    main(parser.parse_args().root)
