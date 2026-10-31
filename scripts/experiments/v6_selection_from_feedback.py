"""One ordinary project selection from an existing revision and measured feedback."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import textwrap
import traceback

from kernel_optimizer.config import load_config
from kernel_optimizer.models.project import GroupRegistry
from kernel_optimizer.project.agents import ProjectAgents
from kernel_optimizer.project.experiment_policy import check_selection
from kernel_optimizer.project.feedback import route
from kernel_optimizer.project.generation_snapshot import load
from kernel_optimizer.project.runtime_config import project_agent_config, worker_environment
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.runtime.provenance import framework_source
from kernel_optimizer.store.run_store import RunStore
from kernel_optimizer.wiring import Runtime


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def ordinary_inputs(search, feedback):
    """Reuse the ordinary read-only assembly, stopping before runtime creation."""
    from kernel_optimizer.project import search as module
    source = textwrap.dedent(inspect.getsource(ProjectSearch._trial))
    method = ast.parse(source).body[0]
    stop = next(i for i, node in enumerate(method.body) if isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'prompt_root' for t in node.targets))
    statements = method.body[:stop + 1]
    namespace = dict(module.__dict__)
    namespace.update(self=search, feedback=feedback,
                     allowed_scopes=('group', 'joint', 'region', 'execution_plan'),
                     leaf_groups_only=False, selection_strategy='agent', feedback_view='full')
    exec(compile(ast.Module(body=statements, type_ignores=[]), '<ordinary-input-assembly>', 'exec'), namespace)
    return namespace, digest('\n'.join(source.splitlines()[:statements[-1].end_lineno]))


def prepare(args):
    config = project_agent_config(load_config(args.config, []))
    config.usage.mode = 'observe_only'
    search = ProjectSearch(config, args.source)
    current = search.revisions.current()
    assert search.current_discovery()['revision'] == current.id
    raw = args.feedback.read_bytes()
    original = json.loads(raw)
    augmented = dict(original)
    constraints = args.user_constraints.read_text()
    augmented['supplemental_evidence'] = {
        'prior': original.get('supplemental_evidence'),
        'user_constraints': constraints, 'latest_user_message': args.user_message}
    augmented_text = json.dumps(augmented, indent=2)
    data, assembly_sha = ordinary_inputs(search, augmented_text)
    data['common']['user-constraints.txt'] = constraints
    feedback = route(data['feedback'], current_revision=current.id,
                     registry=search.revisions.load(current.registry_ref, GroupRegistry))
    assert json.loads(feedback)['decision'] == original['decision']
    expected = {'inputs/' + n: v.encode() if isinstance(v, str) else v
                for n, v in data['common'].items()}
    from kernel_optimizer.project import agents
    expected['inputs/binding-contract.md'] = Path(agents.__file__).with_name('binding_contract.md').read_bytes()
    expected['inputs/worker-environment.json'] = json.dumps(worker_environment(), indent=2).encode()
    expected['inputs/previous-feedback.txt'] = feedback.encode()
    paths = [args.feedback, args.user_constraints]
    paths += [search.output / name for name in
              ['request.json', 'prepared.json', 'discovery.json', 'project/current.json']]
    if (search.output / 'latest-feedback.json').exists():
        paths.append(search.output / 'latest-feedback.json')
    frozen = {str(path): sha(path) for path in paths}
    free = int(subprocess.check_output(['df', '-B1', '--output=avail', str(args.disk_path)], text=True).splitlines()[-1])
    assert free >= args.min_free_bytes + args.forecast_bytes
    assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    details = {'at': now(), 'source': str(search.output), 'current': current.id,
               'source_bundles': current.bundles, 'core': framework_source(),
               'feedback_path': str(args.feedback), 'raw_feedback_sha256': digest(raw),
               'raw_feedback_bytes': len(raw), 'routed_feedback_sha256': digest(feedback),
               'routed_feedback_bytes': len(feedback.encode()),
               'user_constraints_sha256': digest(constraints), 'latest_user_message': args.user_message,
               'ordinary_assembly_sha256': assembly_sha, 'default_policy': data['policy'],
               'inputs': len(expected), 'input_bytes': sum(map(len, expected.values())),
               'input_hashes': {n: digest(v) for n, v in expected.items()},
               'critical_files': len(frozen), 'frozen_identities': frozen,
               'free_bytes': free, 'forecast_bytes': args.forecast_bytes,
               'min_free_bytes': args.min_free_bytes,
               'scope': 'One ordinary project-level selection from current accepted source/discovery and exact measured feedback plus explicit user constraints. No manual optimization/group/algorithm/corrected causal answers. No candidate, parameter, GPU, collector, source-control, reference-copy, publication or acceptance stage. Preserve rejected implementation and original rules; no retry-until-pass.'}
    return config, search, data, feedback, expected, details, augmented_text, raw


def main(args):
    os.umask(0o077)
    config, search, data, feedback, expected, details, augmented, raw = prepare(args)
    if args.preflight:
        print(json.dumps({'status': 'passed', **details}))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    assert not (args.output / 'protocol.json').exists()
    write_json(args.output / 'protocol.json', details)
    (args.output / 'original-feedback.json').write_bytes(raw)
    (args.output / 'augmented-feedback.json').write_text(augmented)
    (args.output / 'routed-feedback.json').write_text(feedback)
    write_json(args.output / 'preparation-reuse.json', {
        'source': details['source'], 'current': details['current'],
        'shared_prepared': search.prepared['revision'], 'no_source_or_reference_rerun': True})
    write_json(args.output / 'state.json', {'phase': 'running', 'stage': 'selection-only', 'at': now()})
    results = {}
    try:
        store = RunStore.create(args.output, 'selection', {'kind': 'selection-from-measured-feedback'})
        body = (data['prompt_root'] / 'selection_prompt.md').read_text()
        with Runtime(config, log_dir=store.run_dir) as runtime:
            directory, receipt = ProjectAgents(runtime, store).generate(
                'selection', data['common'], body, ('selection.json',), feedback,
                validator=lambda path: check_selection(path, data['discovery'],
                    data['policy']['allowed_scopes'], data['eligible']))
        snapshots = []
        for path in (store.run_dir / 'generation-snapshots').glob('*.json'):
            meta, prompt, actual = load(path)
            if meta['fresh_session']:
                assert actual == expected and meta['framework_source'] == details['core']
                snapshots.append(str(path))
        assert len(snapshots) == 1
        results['selection'] = {'status': 'completed', 'agent': receipt,
            'initial_snapshot': snapshots[0], 'selection_sha256': sha(directory / 'selection.json'),
            'inputs_verified': True, 'scope': 'Decision only; no automatic implementation or GPU stage.'}
    except Exception:
        results['selection'] = {'status': 'error', 'traceback': traceback.format_exc()}
    same = all(sha(n) == h for n, h in details['frozen_identities'].items())
    same = same and search.revisions.current().id == details['current']
    results['preservation'] = {'status': 'passed' if same else 'failed',
                               'critical_files': details['critical_files'], 'unchanged': same}
    terminal = {'phase': 'completed' if same and results['selection']['status'] == 'completed'
                else 'completed_with_errors', 'finished_at': now(), 'results': results}
    for name, value in [('followup-results.json', results), ('state.json', terminal),
                        ('followup-completed.json', terminal)]:
        write_json(args.output / name, value)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    for name in ['config', 'source', 'feedback', 'user-constraints', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--user-message', required=True)
    parser.add_argument('--disk-path', type=Path, default=Path('/root/autodl-tmp'))
    parser.add_argument('--min-free-bytes', type=int, default=8 * 1024**3)
    parser.add_argument('--forecast-bytes', type=int, default=384 * 1024**2)
    parser.add_argument('--preflight', action='store_true')
    main(parser.parse_args())
