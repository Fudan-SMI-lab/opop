"""Compare one measured-evidence addition in fresh real project-agent sessions.

This is a developer-started experiment, not an autonomous diagnosis claim.
Both arms retain the failed candidate, ordinary feedback and frozen task.
Candidates are evaluated unchanged by the existing parameter/paired GPU path.
"""
import argparse
import hashlib
import json
from pathlib import Path

from kernel_optimizer.config import load_config
from kernel_optimizer.project.agents import ProjectAgents
from kernel_optimizer.project.branches import fork_run
from kernel_optimizer.project.feedback import candidate_artifacts, route
from kernel_optimizer.project.generated import validate_proposal
from kernel_optimizer.project.generation_snapshot import load
from kernel_optimizer.project.runtime_config import project_agent_config
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.runtime.provenance import framework_source
from kernel_optimizer.wiring import Runtime


def main():
    parser = argparse.ArgumentParser(__doc__)
    for name in ('config', 'source-run', 'output', 'evidence'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--trial', required=True)
    parser.add_argument('--role', choices=('selection', 'candidate'), required=True)
    parser.add_argument('--repeats', type=int, default=1,
                        help='Experimental replicates, not an API usage quota')
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('repeats must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.source_run
    saved = json.loads((source/'trials'/args.trial/'generated.json').read_text())
    ordinary = json.loads((source/'trials'/args.trial/'result.json').read_text())
    evidence_bytes = args.evidence.read_bytes()
    evidence = json.loads(evidence_bytes)
    snapshot = Path(saved[args.role + '_agent']['generation_snapshots'][0])
    manifest, _, files = load(snapshot)
    config = project_agent_config(load_config(args.config, []))
    config.usage.mode = 'observe_only'
    protocol = {'source_run': str(source), 'trial': args.trial, 'role': args.role,
                'snapshot': str(snapshot), 'original_framework': manifest['framework_source'],
                'framework': framework_source(), 'repeats': args.repeats,
                'evidence_sha256': hashlib.sha256(evidence_bytes).hexdigest(),
                'treatment': 'Only supplemental_evidence is added to ordinary feedback',
                'scope': 'Fresh sessions; developer-supplied diagnosis; unchanged GPU validation; isolated publication'}
    write_json(args.output/'protocol.json', protocol)
    (args.output/'evidence.json').write_bytes(evidence_bytes)
    rows = []
    for repeat in range(args.repeats):
        for arm in (('control', 'treatment') if repeat % 2 == 0 else ('treatment', 'control')):
            branch = args.output/f'{repeat}-{arm}'
            parent = fork_run(source, branch, revision_id=saved['bundle']['parent_revision'])
            search = ProjectSearch(config, branch)
            feedback = dict(ordinary)
            if arm == 'treatment':
                feedback['supplemental_evidence'] = evidence
            write_json(branch/'experimental-feedback.json', feedback)
            try:
                if args.role == 'selection':
                    result = search.trial(feedback=json.dumps(feedback))
                else:
                    inputs = {name.removeprefix('inputs/'): raw for name, raw in files.items()
                              if name.startswith('inputs/') and not name.startswith('inputs/previous-candidate/')
                              and name != 'inputs/previous-feedback.txt'}
                    inputs.update(candidate_artifacts(json.dumps(ordinary), search.store))
                    prompt = Path(__file__).resolve().parents[1]/'src/kernel_optimizer/project/candidate_prompt.md'
                    instructions = prompt.read_text() + ('\nRepair the actual rejected implementation in '
                        'inputs/previous-candidate/ using the measured feedback. Preserve the selected '
                        'boundary and frozen contract. Further GPU checks belong to the bound worker.')
                    with Runtime(config, log_dir=branch) as runtime:
                        generated, receipt = ProjectAgents(runtime, search.store).generate(
                            'candidate', inputs, instructions, ('candidate.py', 'proposal.json'),
                            route(json.dumps(feedback), current_revision=parent.id), validator=validate_proposal)
                    discovery = search.current_discovery()
                    selection = saved['selection']
                    trial_id = 'repair-' + arm
                    trial_root = branch/'trials'/trial_id
                    trial_root.mkdir(parents=True)
                    bundle, proposal = search.bundle_from_outputs(generated, selection, parent, discovery, trial_id)
                    metadata = {'trial': trial_id, 'selection': selection, 'proposal': proposal,
                                'candidate_agent': receipt, 'experiment': protocol, 'arm': arm, 'repeat': repeat}
                    write_json(trial_root/'generated.json', {**metadata, 'bundle': bundle.model_dump(mode='json')})
                    result = search._evaluate_generated(bundle, metadata, discovery)
                row = {'repeat': repeat, 'arm': arm, 'output': str(branch),
                       'trial': result['trial'], 'decision': result['decision']}
            except Exception as exc:
                import traceback
                row = {'repeat': repeat, 'arm': arm, 'output': str(branch),
                       'error': f'{type(exc).__name__}: {exc}', 'traceback': traceback.format_exc()}
            rows.append(row)
            write_json(args.output/'result.json', {'protocol': protocol, 'runs': rows})
            print(json.dumps(row), flush=True)
    write_json(args.output/'completed.json', {'completed': True, 'runs': len(rows)})


if __name__ == '__main__':
    main()
