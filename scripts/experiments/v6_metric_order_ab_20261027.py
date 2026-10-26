"""One real intake replay per arm; only the supplied schema changes between arms."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import traceback

from kernel_optimizer.config import load_config
from kernel_optimizer.project.agents import ProjectAgents
from kernel_optimizer.project.generation_snapshot import load
from kernel_optimizer.project.intake import IntakeProposal, resolve_request
from kernel_optimizer.project.runtime_config import project_agent_config
from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.runtime.provenance import framework_source
from kernel_optimizer.store.run_store import RunStore
from kernel_optimizer.wiring import Runtime


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    args.output.mkdir(parents=True, exist_ok=False)
    saved, prompt, inputs = load(args.snapshot)
    old = inputs['inputs/intake.schema.json'].decode()
    new = json.dumps(IntakeProposal.model_json_schema(), indent=2)
    environment = json.loads(inputs['inputs/deployment.json'])
    source = Path(json.loads(inputs['inputs/repository.json'])['root'])
    goal = json.loads(inputs['inputs/intent.json'])['goal']
    config = project_agent_config(load_config(args.config, []))
    config.usage.mode = 'observe_only'
    write_json(args.output/'protocol.json', {'original_snapshot': str(args.snapshot),
        'original_framework': saved['framework_source'], 'replay_framework': framework_source(),
        'order': ['control', 'treatment'], 'calls_per_arm': 1,
        'scope': 'Independent fresh-session role replay, not an autonomous project cohort. Automatic contract repair remains enabled; no API budget.',
        'difference': 'Only inputs/intake.schema.json; prompt and other captured inputs unchanged.',
        'schema_sha256': {name: hashlib.sha256(value.encode()).hexdigest() for name, value in [('control', old), ('treatment', new)]}})

    def validate(directory):
        proposal = IntakeProposal.model_validate_json((directory/'intake.json').read_bytes())
        resolve_request(proposal, source, goal, environment)

    results = []
    for arm in ('control', 'treatment'):
        store = RunStore.create(args.output, arm, {'kind': 'metric-aggregation-intake-replay', 'arm': arm})
        edit = None if arm == 'control' else {'path': 'inputs/intake.schema.json', 'old': old, 'new': new}
        try:
            with Runtime(config, log_dir=store.run_dir) as runtime:
                directory, receipt = ProjectAgents(runtime, store).replay_snapshot(args.snapshot,
                    edit=edit, validator=validate, allow_source_change=True)
            proposal = IntakeProposal.model_validate_json((directory/'intake.json').read_bytes())
            request = resolve_request(proposal, source, goal, environment)
            result = {'arm': arm, 'status': proposal.status, 'agent': receipt,
                      'proposal': proposal.model_dump(mode='json'),
                      'request': request.portable() if request else None}
        except Exception:
            result = {'arm': arm, 'status': 'error', 'traceback': traceback.format_exc()}
        results.append(result)
        write_json(args.output/'result.json', {'runs': results})
        print(json.dumps({'arm': arm, 'status': result['status'],
                          'objective': (result.get('request') or {}).get('objective')}), flush=True)
    write_json(args.output/'completed.json', {'runs': results})


if __name__ == '__main__':
    main()
