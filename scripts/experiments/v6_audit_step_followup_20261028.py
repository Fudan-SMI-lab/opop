"""Read-only audit of frozen native paths, diagnostic samples and original-entry DDP."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3
import statistics

from kernel_optimizer.config import load_config
from kernel_optimizer.models.project import ProjectContract
from kernel_optimizer.project.communication import trace_window
from kernel_optimizer.project.generation_snapshot import load
from kernel_optimizer.project.measurement import paired_decision
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.runtime.execution import write_json
from scripts.experiments.v6_audit_step_goal_20261028 import read, sha, endpoints
from scripts.experiments.v6_audit_ddp_tp_20261028 import normalize_parameters


def aligned(rows):
    assert len({len(r) for r in rows}) == 1
    return [max(v) for v in zip(*rows)]


def audit(root, output):
    terminal = read(root/'followup-completed.json')
    report = {'terminal': terminal['phase'], 'finished_at': terminal['finished_at'],
              'disk': terminal['disk'], 'fixed_paths': [], 'communication': [], 'trials': []}
    for endpoint in read(root/'fixed-path/result.json')['endpoints']:
        for discovery in endpoint['discoveries']:
            path = Path(discovery['discovery']); data = read(path)
            assert discovery['valid'] and sha(data['trace']) == data['trace_sha256']
            models = [c for c in data['calls'] if c['semantics'].endswith('.GPT2LMHeadModel')]
            assert len(models) == (4 if endpoint['arm']=='B' else 1)
            comm = data['communication']
            assert comm['communication_kernel_count'] == 13 and comm['logical_input_bytes'] == 497759232
            report['fixed_paths'].append({'arm': endpoint['arm'], 'rank': data['rank'],
                'path': str(path), 'sha256': sha(path), 'trace_sha256': data['trace_sha256'],
                'models': models, 'call_count': len(data['calls']), 'kernel_count': len(data['kernels']),
                'cat_kernels': sum('CatArrayBatchedCopy' in k['name'] for k in data['kernels']),
                'replacement_kernel_count': sum(bool(k['replacement_sites']) for k in data['kernels']),
                'communication': {k:comm[k] for k in ['communication_kernel_count','logical_input_bytes','payload_coverage']}})
    parameters = {}
    source_hashes = {}
    for endpoint in read(root/'fixed-communication/result.json')['endpoints']:
        receipt_path = Path(endpoint['receipt']); receipt = read(receipt_path)
        checked = endpoints([{'diagnostic': receipt}])[0]
        protocol = read(receipt_path.parent.parent/'protocol.json')
        normalized = normalize_parameters(protocol['parameters'])
        arm = endpoint['arm']
        if arm in parameters: assert parameters[arm] == normalized
        parameters[arm] = normalized
        source_root = receipt_path.parent.parent/'source'
        hashes = {p.relative_to(source_root).as_posix():sha(p) for p in source_root.rglob('*')
                  if p.is_file() and '__pycache__' not in p.parts}
        if arm in source_hashes: assert source_hashes[arm] == hashes
        source_hashes[arm] = hashes
        trace_rows, comm_values = [], []
        for rank in receipt['ranks']:
            value = rank['result']; samples = value['metric_samples']['communication_kernel_sum_ms']
            assert len(samples) == 3
            for p, expected in zip(value['diagnostics']['evidence'], samples):
                trace, window = trace_window(read(p), 'opop.communication_metric_workload')
                assert window is not None
                kernels = [e for e in trace['traceEvents'] if e.get('cat')=='kernel' and e.get('ph')=='X']
                actual = sum(e['dur'] for e in kernels if 'Collective name' in e.get('args',{}) or 'nccl' in e.get('name','').lower())/1000
                assert abs(actual-expected) < 1e-8
                trace_rows.append({'path':p,'sha256':sha(p),'rank':rank['rank'],'value_ms':actual,'window':window})
            comm_values.append(samples)
        clean = aligned([r['step_samples_ms'] for r in checked['ranks']]); comm = aligned(comm_values)
        assert comm == endpoint['max_rank_communication_samples_ms']
        report['communication'].append({'index':endpoint['index'],'arm':arm,'receipt_sha256':sha(receipt_path),
            'endpoint':checked,'all_aligned_clean_samples_ms':clean,'clean_median_ms':statistics.median(clean),
            'all_aligned_communication_samples_ms':comm,'communication_median_ms':statistics.median(comm),
            'raw_traces':trace_rows})
    assert parameters['B']['request'] == parameters['C']['request']
    for name, digest in source_hashes['B'].items():
        if name != '.opop-revision.json': assert source_hashes['C'][name] == digest
    report['fixed_source_hashes'] = source_hashes
    run = root/'ddp-step-original'; request = read(run/'request.json')
    search = ProjectSearch(load_config(root/'private-config.yaml', []),run)
    current = search.revisions.current(); baseline = search.revisions.revision(search.prepared['revision'])
    assert not baseline.bundles and len(current.bundles)==1
    contract = search.revisions.load(current.contract_ref, ProjectContract)
    intake = read(root/'ddp-step-original.intake/result.json')
    m,p,files = load(Path(intake['agent']['generation_snapshots'][0]))
    assert set(files)=={'inputs/deployment.json','inputs/intake.schema.json','inputs/intent.json','inputs/repository.json','inputs/worker-environment.json'}
    report['intake'] = {'agent':intake['agent'],'initial_files':{k:hashlib.sha256(v).hexdigest() for k,v in files.items()},
        'intent':read(root/'ddp-step-original.intake/intent.json'), 'resolved_request_sha256':sha(root/'ddp-step-original.intake/resolved-request.json')}
    assert sha(Path(search.prepared['controls_path']).parent/'reference-source/opop_adapter.py') == search.prepared['adapter_sha256']
    report['contract'] = {'reference':current.contract_ref,'invariants':contract.invariants,'objective':request['objective'],
        'request_sha256':sha(run/'request.json'),'adapter_sha256':search.prepared['adapter_sha256'],
        'baseline_revision':baseline.id,'required_observations':request['required_observations']}
    report['controls'] = [{'rank':r['rank'],'valid':r['result']['valid'],'controls':r['result']['controls']} for r in search.prepared['evaluation']['ranks']]
    events = [json.loads(line) for line in (run/'events.jsonl').read_text().splitlines()]
    report['events'] = {'counts':dict(Counter(e['type'] for e in events)),
        'discoveries':[{k:e['payload'][k] for k in ['parent','revision']} for e in events if e['type']=='PROJECT_DISCOVERED']}
    role_order = [e['payload']['session'] for e in events if e['type']=='PROJECT_AGENT_FINISHED' and e['payload']['role']=='candidate']
    trials = sorted([read(p) for p in (run/'trials').glob('*/result.json')],key=lambda t:role_order.index(t['candidate_agent']['session']))
    conn = sqlite3.connect('file:/root/autodl-tmp/dev/v6-work/private-runtime/data/opencode/opencode.db?mode=ro',uri=True)
    for t in trials:
        a=t['candidate_agent']; candidate=Path(a['sandbox'])/'candidate.py'
        feedback={}
        for role in ['selection_agent','candidate_agent']:
            sm,sp,sf=load(Path(t[role]['generation_snapshots'][0]))
            feedback[role]={k:{'bytes':len(v),'sha256':hashlib.sha256(v).hexdigest()} for k,v in sf.items() if 'feedback' in k}
        tools=[]
        for ident,data in conn.execute('select id,data from part where session_id=? order by time_created,id',(a['session'],)):
            part=json.loads(data)
            if part.get('type')!='tool':continue
            state=part.get('state',{}); value=state.get('input',{})
            tools.append({'part':ident,'tool':part['tool'],'status':state.get('status'),
                'read_path':value.get('filePath'),'input_sha256':hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest(),
                'mentions_feedback': 'feedback' in json.dumps(value)})
        collectors=[]
        for phase,cs in t['resources'].get('extra_collectors',{}).items():
            for name,c in cs.items():
                row={'phase':phase,'name':name,'status':c['status'],'ranks':[]}
                if name=='nsys':
                    for z in c['ranks']:
                        kernels=z['kernels']
                        row['ranks'].append({'evidence':z['evidence'],'sha256':sha(z['evidence']),
                            'cuda_kernels':len(kernels),'graph_kernels':sum(bool(k.get('graphNodeId')) for k in kernels),
                            'nccl_kernels':z['nccl_kernels'],'nccl_graph_kernels':sum(bool(k.get('graphNodeId')) for k in kernels if 'nccl' in k['name'].lower())})
                if name=='candidate_diagnostics':
                    for z in c['run']['ranks']:
                        row['ranks'].append({'rank':z['rank'],'coverage':z['result']['coverage'],
                            'exceptions':[{k:e[k] for k in ['type','message','line']} for e in z['result']['candidate_diagnostics']['events'] if e.get('event')=='exception']})
                collectors.append(row)
        report['trials'].append({'trial':t['trial'],'decision':t['decision'],'measurement':t.get('measurement'),
            'scope':t['selection']['scope'],'rationale':t['selection']['rationale'],'candidate':str(candidate),'candidate_sha256':sha(candidate),
            'agent':a,'feedback_inputs':feedback,'tools':tools,'collectors':collectors,
            'paired_endpoints':endpoints(t['pairs']) if t.get('pairs') else []})
    conn.close()
    independent_path=next((root/'ddp-step-original-independent/version-trials').glob('*/result.json'))
    independent=read(independent_path)
    assert paired_decision(independent['pairs'],request['objective'],search.prepared['evaluation']) == independent['measurement']
    extra=read(root/'followup-results.json')['ddp-independent']
    report['independent']={'path':str(independent_path),'sha256':sha(independent_path),'decision':independent['decision'],
        'measurement':independent['measurement'],'endpoints':endpoints(independent['pairs'])}
    report['source_fidelity']={'valid':extra['source_fidelity_valid'],'ranks':[r['result'] for r in extra['source_fidelity']['ranks']],
        'scope':'Independent original serial two-step trajectory vs baseline references; each candidate endpoint separately compares full observations to those references.'}
    report['negative_probe']={k:v for k,v in extra['negative_probe'].items() if k!='run'}
    report['negative_probe']['ranks']=[r['result'] for r in extra['negative_probe']['run']['ranks']]
    report['current']={'revision':current.id,'bundles':list(current.bundles),'post_final_rediscovery':False}
    report['limits']=['Fixed communication diagnostics are not acceptance, wire time or exposed critical path.',
        'First accepted search candidate fell back to eager. Empty framework fallback counters alone do not establish path.',
        'Final native graph evidence is a separate diagnostic; formal step endpoints have no communication metric.',
        'No historical failure or samples overwritten; no production change in this batch.']
    write_json(output,report)
    print(json.dumps({'audit':str(output),'bytes':output.stat().st_size,'independent_accepted':independent['decision']['accepted'],
        'gain_percent':100*(1-independent['measurement']['objective']/independent['measurement']['reference'])}))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();audit(a.root,a.output)
