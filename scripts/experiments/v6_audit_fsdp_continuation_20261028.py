"""Audit continuation identity, inherited feedback and immutable source mappings."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path

from kernel_optimizer.config import load_config
from kernel_optimizer.models.project import ExecutionPlan, GroupRegistry
from kernel_optimizer.project.generation_snapshot import load
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.runtime.execution import write_json
from scripts.experiments.v6_audit_fsdp_20261028 import read, sha
from scripts.experiments.v6_audit_fsdp_search_20261028 import audit


def main(root, reuse_endpoint_audit=False):
    limits = [
        'Source-verified developer-assisted evaluator; subsequent search autonomous, not a fresh unaided cohort.',
        'New bundle IDs alone do not establish a new implementation or an incremental optimization.',
        'Four communication fields use distinct profiled invocations, not same-trace decompositions.',
        'NCCL durations do not establish wire bandwidth or exposed critical path.',
        'Every historical decision and original sample is preserved.']
    if reuse_endpoint_audit:
        existing = read(root / 'terminal-audit.json')
        assert existing['terminal_sha256'] == sha(root / 'followup-completed.json')
        assert existing['limits'] == limits
    else:
        audit(root, root / 'terminal-audit.json', limits=limits)
    config = load_config(root / 'private-config.yaml', [])
    search = ProjectSearch(config, root / 'fsdp-search')
    parent = ProjectSearch(config, root.parent / 'fsdp-search-20261028/fsdp-search')
    baseline = ProjectSearch(config, root.parent / 'fsdp-owner-repair-20261028/fsdp-repaired')
    protocol = read(root / 'search-protocol.json')
    reuse = read(root / 'discovery-reuse.json')
    failed = ProjectSearch(config, Path(reuse['source']))
    assert protocol['feedback_sha256'] == sha(parent.output / 'latest-feedback.json')
    assert reuse['source_discovery_sha256'] == sha(failed.output / 'discovery.json')
    for path in reuse['raw_discovery']:
        raw = read(path)
        assert sha(raw['trace']) == raw['trace_sha256']
    inherited = failed.revisions.current()
    assert inherited.files == parent.revisions.current().files
    assert inherited.contract_ref == parent.revisions.current().contract_ref
    assert inherited.bundles == parent.revisions.current().bundles
    assert search.prepared['controls_path'] == baseline.prepared['controls_path']
    audit_report = read(root / 'terminal-audit.json')
    base_files = {f.path: f.sha256 for f in baseline.revisions.current().files}
    old_bundle = parent.revisions.bundle(list(parent.revisions.current().bundles)[0])
    old_source = {f.path: f for f in parent.revisions.current().files}[old_bundle.installer]
    old_code = parent.store.get_artifact(old_source.content_ref).decode()
    records = []
    for row in audit_report['trials']:
        trial = read(search.output / 'trials' / row['trial'] / 'result.json')
        bundle = search.revisions.bundle(trial['decision']['bundle'])
        source = Path(trial['candidate_agent']['sandbox']) / 'candidate.py'
        original_parent = search.revisions.revision(trial['decision']['parent'])
        registry = search.revisions.load(original_parent.registry_ref, GroupRegistry)
        plan = search.revisions.load(original_parent.plan_ref, ExecutionPlan)
        group_ids = {g.id for g in registry.groups}; call_ids = {c.id for c in plan.calls}
        assert set(trial['selection']['groups']) <= group_ids
        assert set(trial['selection']['affected_calls']) <= call_ids
        revisions = []
        for observation in trial['parameter_search']['observations']:
            revision = search.revisions.revision(observation['revision'])
            files = {f.path: f.sha256 for f in revision.files}
            assert revision.contract_ref == baseline.revisions.current().contract_ref
            assert all(files.get(name) == value for name, value in base_files.items())
            assert files[bundle.installer] == sha(source)
            revisions.append({'id': revision.id, 'bundles': list(revision.bundles),
                              'retired_bundles': list(revision.retired_bundles),
                              'parameters': observation['bundle']['parameters']})
        metadata, prompt, inputs = load(Path(trial['candidate_agent']['generation_snapshots'][0]))
        selection = json.loads(inputs['inputs/selection.json'])
        assert selection == trial['selection']
        history = {k: {'bytes': len(v), 'sha256': hashlib.sha256(v).hexdigest()}
                   for k, v in inputs.items() if 'previous-candidate' in k or 'previous-feedback' in k}
        prior_bytes = inputs.get('inputs/previous-candidate/candidate.py')
        records.append({'trial': row['trial'], 'parent': original_parent.id,
            'source_sha256': sha(source), 'same_as_inherited_source': sha(source) == old_source.sha256,
            'source_diff_from_inherited': ''.join(difflib.unified_diff(old_code.splitlines(keepends=True), source.read_text().splitlines(keepends=True))) if not records else None,
            'same_as_previous_candidate_bytes': prior_bytes == source.read_bytes() if prior_bytes else None,
            'same_parameters_as_inherited': bundle.parameters == old_bundle.parameters,
            'same_calls_as_inherited': bundle.calls == old_bundle.calls,
            'same_groups_as_inherited': bundle.groups == old_bundle.groups,
            'calls': list(bundle.calls), 'groups': list(bundle.groups),
            'affected_call_count': len(bundle.affected_calls),
            'added_affected_calls': sorted(set(bundle.affected_calls)-set(old_bundle.affected_calls)),
            'removed_affected_calls': sorted(set(old_bundle.affected_calls)-set(bundle.affected_calls)),
            'replaces': list(bundle.replaces), 'requires': list(bundle.requires),
            'history_inputs': history, 'snapshot': trial['candidate_agent']['generation_snapshots'][0],
            'selection_input_verified': True,
            'proposal_rationale': trial['proposal']['rationale'],
            'parameter_space': trial['proposal']['parameter_space'],
            'experiment_policy': trial['experiment_policy'],
            'revisions': revisions})
    result = {'source_root': str(root), 'feedback_sha256': protocol['feedback_sha256'],
        'inherited_revision': parent.revisions.current().id,
        'inherited_bundle': old_bundle.id, 'inherited_source_sha256': old_source.sha256,
        'source_files': base_files, 'contract_ref': baseline.revisions.current().contract_ref,
        'controls_path': search.prepared['controls_path'], 'discovery_reuse_verified': True,
        'trials': records, 'source_and_contract_unchanged': True,
        'interpretation': 'Code review: trial 1 removes a module docstring and changes DEFAULT_KEEP from 2 to 3. Both selected bundles explicitly supply keep_unsharded_blocks=3, bypassing that default; the executed policy/call is unchanged. The candidate proposed extending the domain to 0..5, but actual screening only visited 1,2,3. Frozen acceptance is retained but is not evidence of incremental optimization. Later prefetch and keep-4/5 trials are distinct and rejected; no multiple active bundles coexist.'}
    write_json(root / 'continuation-identity-audit.json', result)
    print(json.dumps({'identity_audit': str(root / 'continuation-identity-audit.json'),
                      'trials': [{k: r[k] for k in ['trial', 'same_as_inherited_source', 'same_parameters_as_inherited', 'same_calls_as_inherited']} for r in records]}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--reuse-endpoint-audit', action='store_true')
    args = parser.parse_args()
    main(args.root, args.reuse_endpoint_audit)
