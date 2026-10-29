"""One isolated author for internally caught setup failures; no GPU execution."""
import argparse
import json
import os
from pathlib import Path
import traceback

from kernel_optimizer.config import load_config
from kernel_optimizer.project.agents import ProjectAgents
from kernel_optimizer.project.runtime_config import project_agent_config
from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.runtime.provenance import framework_source
from kernel_optimizer.store.run_store import RunStore
from kernel_optimizer.wiring import Runtime
from scripts.experiments.v6_audit_fsdp_20261028 import read, sha
from scripts.experiments.v6_ddp_semantics_20261027 import disk_observation
from scripts.experiments.v6_diagnostic_author_20261029 import INSTRUCTIONS, INTERFACE, now, validate

EXTRA = """
The supplied case includes an internally caught installation/self-check failure;
the outer evaluator can return valid on a fallback path. Preserve both that return
and the original internal catch/fallback behavior. Do not turn a caught exception
into a worker failure, force-enable a path, change thresholds, or rerun the model.
Determine from actual code which comparisons/calls are reached and what is missing.
Choose bounded observations of comparison operands and input/state provenance
needed to assess whether the compared executions are comparable; distinguish those
observations from causal conclusions. Do not dump model weights, all caches or
multi-GB state. Do not reconstruct a different computation to manufacture evidence.
Explain a size bound, retained owners, CPU copies/synchronization and error behavior.
Truthful nonfinite/absent evidence must remain serializable while preserving the
original return or exception. No fixed error-message routing or predetermined cause.
"""


def build_inputs(work):
    parent = work/'qwen-device-feedback-20261030'
    candidate = parent/'qwen-continuation/jobs/trial-0db61e9938-param-4be0257f-screen/run/request.json'
    baseline = work/'qwen-device-refresh-fixed-20261030/qwen-fixed/jobs/B/run/request.json'
    sidecar = parent/'qwen-continuation/jobs/candidate-diagnostics-27cef0befd/run/candidate-diagnostics-rank-0.json'
    diagnostic_request = sidecar.parent/'request.json'
    requests = {k: read(p) for k,p in [('baseline', baseline), ('candidate', candidate)]}
    cp = requests['candidate']['parameters']
    assert requests['baseline']['parameters']['request'] == cp['request']
    assert not requests['baseline']['parameters']['bundles'] and len(cp['bundles']) == 1
    code = Path(cp['bundles'][0]['file']); adapter = Path(cp['adapter'])
    assert sha(code) == '04dfc6c2c9165560ae7412bbdbd00be6e79044e7f52276d096dd48728c6583f8'
    assert cp['bundles'][0]['parameters'] == {}
    assert sha(adapter) == '498c90f3f6f91b993ac8a586ac3230ffe2e6901752165b26948142d978f7cff7'
    assert read(candidate.parent/'rank-0.json')['status'] == 'passed'
    assert read(candidate.parent/'rank-0.json')['result']['valid']
    expected = read(parent/'protocol.json')['frozen_identities'].copy()
    def item(p):
        p = Path(p); h = sha(p); expected[str(p)] = h
        return {'path': str(p), 'sha256': h, 'bytes': p.stat().st_size}
    index = {'original_execution_requests': {k:item(p) for k,p in [('baseline',baseline),('candidate',candidate)]},
        'original_worker_results': {k:item(p.parent/'rank-0.json') for k,p in [('baseline',baseline),('candidate',candidate)]},
        'run_only_diagnostic': {'request': item(diagnostic_request), 'worker': item(sidecar.parent/'rank-0.json'),
            'observations': item(sidecar), 'scope': 'Separate existing collector; original raw events, no developer interpretation. Not an outer-quality evaluation.'},
        'original_entries': {k:{**item(r['entry']), 'function': r['function']} for k,r in requests.items()},
        'candidate': item(code), 'adapter': item(adapter),
        'controls_reference': item(Path(cp['reference_dir'])/'reference-rank-0.pt'),
        'independent_reference': None, 'source_files': {},
        'scope': 'Original current-device requests/code/results and original run-only sidecar only. No developer audit, repair solution, later implementation or other-case diagnostic supplied. Baseline is an existing observed original endpoint; timings are not an acceptance pair.'}
    for arm, request in requests.items():
        manifest = Path(request['cwd'])/'.opop-revision.json'; item(manifest)
        index['source_files'][arm] = {n:item(Path(request['cwd'])/n) for n in read(manifest)['files']}
    # Preserve dependencies and pointers, but do not supply developer audit answers.
    for p in [parent/'protocol.json', parent/'followup-completed.json',
              parent/'qwen-continuation/project/current.json',
              parent/'qwen-continuation/prepared.json', parent/'qwen-continuation/request.json']:
        item(p)
    for folder in (parent/'qwen-continuation/jobs').glob('*/run'):
        for name in ['request.json', 'rank-0.json', 'receipt.json']:
            item(folder/name)
    inputs = {'task.json': json.dumps(cp['request'], indent=2), 'failure.json': sidecar.read_bytes(),
        'adapter.py': adapter.read_bytes(), 'candidate.py': code.read_bytes(),
        'evidence-index.json': json.dumps(index, indent=2), 'interface.md': INTERFACE}
    return inputs, expected, index


def main(args):
    os.umask(0o077); inputs, frozen, index = build_inputs(args.work)
    assert all(sha(p) == h for p,h in frozen.items())
    if args.preflight:
        print(json.dumps({'status':'passed','core':framework_source(), 'frozen_files':len(frozen),
            'input_sizes':{k:len(v.encode() if isinstance(v,str) else v) for k,v in inputs.items()}})); return
    args.output.mkdir(parents=True, exist_ok=True)
    assert not (args.output/'protocol.json').exists()
    write_json(args.output/'protocol.json', {'at':now(), 'core':framework_source(),
        'scope':'One new diagnostic_author for internally caught setup assertions. No GPU/model/collective execution, candidate generation/search/acceptance or production adoption. Original code and original sidecar only, no developer cause or handwritten repair. Filesystem accessible; complete actual access audit required.',
        'input_sizes':{k:len(v.encode() if isinstance(v,str) else v) for k,v in inputs.items()},
        'frozen_identities':frozen, 'input_index':index,
        'sequence':['one real diagnostic_author','syntax/schema only','preservation'],
        'disk':disk_observation(Path('/root/autodl-tmp'))})
    config = load_config(args.config, []); config.usage.mode = 'observe_only'; results = {}
    write_json(args.output/'state.json', {'at':now(),'phase':'running','stage':'diagnostic-author'})
    try:
        directory = args.output/'diagnostic-author'
        store = RunStore.create(directory.parent, directory.name, {'kind':'isolated_setup_diagnostic_author'})
        with Runtime(project_agent_config(config), directory) as runtime:
            output, receipt = ProjectAgents(runtime, store).generate('diagnostic_author', inputs,
                INSTRUCTIONS+EXTRA, ('diagnostic.py','diagnostic-plan.json'), validator=validate)
        results['diagnostic-author'] = {'status':'completed','agent':receipt,
            'probe':str(output/'diagnostic.py'),'probe_sha256':sha(output/'diagnostic.py'),
            'plan':str(output/'diagnostic-plan.json'),'plan_sha256':sha(output/'diagnostic-plan.json'),
            'validation':'Syntax/schema only. Audit actual tools/code/size and preserve failures before any separate declared worker execution.'}
    except Exception:
        results['diagnostic-author'] = {'status':'error','traceback':traceback.format_exc()}
    same = all(sha(p) == h for p,h in frozen.items())
    results['preservation'] = {'status':'completed' if same else 'error','files':len(frozen),'unchanged':same}
    terminal = {'phase':'completed' if all(x['status']=='completed' for x in results.values()) else 'completed_with_errors',
        'finished_at':now(),'results':results,'disk':disk_observation(Path('/root/autodl-tmp'))}
    write_json(args.output/'followup-results.json', results)
    write_json(args.output/'state.json', terminal); write_json(args.output/'followup-completed.json', terminal)
    print(json.dumps({'phase':terminal['phase'],'finished_at':terminal['finished_at']}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    for n in ['config','output','work']: p.add_argument('--'+n, type=Path, required=True)
    p.add_argument('--preflight', action='store_true'); main(p.parse_args())
