"""Audit fresh role snapshots and real tool receipts without exporting private settings."""
import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import sqlite3

from kernel_optimizer.config import load_config
from kernel_optimizer.control.direct_task import TaskSpace
from kernel_optimizer.models.core import ParameterSpace
from kernel_optimizer.paramspace.guard import eval_constraint
from kernel_optimizer.project.generated import validate_proposal
from kernel_optimizer.project.generation_snapshot import load
from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.tuning.objective import Objective
from kernel_optimizer.tuning.tpe import OptunaTPETuner
from scripts.experiments.v6_accepted_feedback_replay_20261028 import OLD, NEW


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main(root):
    protocol = json.loads((root / 'protocol.json').read_text())
    terminal = json.loads((root / 'followup-completed.json').read_text())
    assert terminal['phase'] == 'completed'
    original, original_prompt, original_files = load(Path(protocol['snapshot']))
    config = load_config(root / 'private-config.yaml', [])
    db = sqlite3.connect('file:/root/autodl-tmp/dev/v6-work/private-runtime/data/opencode/opencode.db?mode=ro', uri=True)
    report = {'at': datetime.now(timezone.utc).isoformat(), 'completed_at': terminal['finished_at'],
              'snapshot': protocol['snapshot'], 'core': original['framework_source'],
              'change': {'old': OLD, 'new': NEW}, 'arms': {}}
    for arm in ['control', 'treatment']:
        row = terminal['results'][arm]; agent = row['agent']; sandbox = Path(agent['sandbox'])
        saved, prompt, files = load(Path(agent['generation_snapshots'][0]))
        assert files == original_files
        assert prompt == (original_prompt if arm == 'control' else original_prompt.replace(OLD, NEW))
        assert saved['tool_model_settings'] == original['tool_model_settings']
        assert saved['model'] == original['model'] and saved['framework_source'] == original['framework_source']
        assert saved['fresh_session']
        validate_proposal(sandbox)
        tools, copied_probes = [], []
        for ident, raw in db.execute('select id,data from part where session_id=? order by time_created,id', (agent['session'],)):
            part = json.loads(raw)
            if part.get('type') != 'tool':
                continue
            state = part.get('state', {}); inp = state.get('input', {}); metadata = state.get('metadata', {})
            command = inp.get('command', ''); path = inp.get('filePath', '')
            text = str(state.get('output', '')); serialized = json.dumps(inp, sort_keys=True)
            tool = {'part': ident, 'tool': part['tool'], 'status': state.get('status'),
                'input_sha256': sha(serialized.encode()), 'output_sha256': sha(text.encode()),
                'output_bytes': len(text.encode()), 'exit': metadata.get('exit'), 'path': path or None,
                'reads_feedback': 'previous-feedback' in command or ('previous-feedback' in path and part['tool'] == 'read'),
                'reads_selection': 'selection.json' in command or ('selection.json' in path and part['tool'] == 'read')}
            relevant = any(word in command for word in ['previous-feedback', 'cpu_probe.py', 'build_probe', 'rollback_probe'])
            if relevant:
                tool.update(command=command, output_excerpt=text[:1800] if 'previous-feedback' in command else text[-2600:])
            tools.append(tool)
            if part['tool'] == 'write' and path.startswith('/tmp/opencode/candtest/') and path.endswith('.py'):
                # Preserve the original write, including subsequently repaired toy probes.
                content = inp.get('content')
                if isinstance(content, str):
                    target = root / 'audit-cpu-probes' / arm / (ident + '-' + Path(path).name)
                    target.parent.mkdir(parents=True, exist_ok=True); target.write_text(content)
                    copied_probes.append({'path': str(target), 'sha256': sha(target.read_bytes()), 'original': path})
        space = TaskSpace.model_validate(row['proposal']['parameter_space'])
        feasible = []
        for values in itertools.product(*(d.choices for d in space.params)):
            parameters = dict(zip((d.name for d in space.params), values))
            if all(eval_constraint(c.expr, {**config.device.as_env(), **parameters}) for c in space.constraints):
                feasible.append(parameters)
        # The first asks are still in Optuna's random startup phase. No outcomes
        # are told here, and no candidate/GPU code is executed or scored.
        tuner = OptunaTPETuner(ParameterSpace(space_id='audit-'+arm, candidate_id=arm, source_sha=row['candidate_sha256'],
            domains=space.params, constraints=space.constraints),
            lambda p: all(eval_constraint(c.expr, {**config.device.as_env(), **p.values}) for c in space.constraints),
            budget=3, seed=config.run.seed, objective=Objective(direction='minimize', label='step_ms', unit='ms'))
        preview = []
        for _ in range(3):
            item = tuner.ask()
            if item is not None: preview.append(dict(item[1].values))
        audit = {'session': agent['session'], 'snapshot': agent['generation_snapshots'][0],
            'schema_repairs': agent['contract_repairs'], 'usage': agent.get('usage'),
            'all_input_files_unchanged': True, 'input_count': len(files),
            'prompt_sha256': sha(prompt.encode()),
            'settings_sha256': sha(json.dumps(saved['tool_model_settings'], sort_keys=True).encode()),
            'candidate': str(sandbox/'candidate.py'), 'candidate_sha256': sha((sandbox/'candidate.py').read_bytes()),
            'proposal': row['proposal'], 'feasible_parameter_combinations': len(feasible),
            'startup_ask_preview': preview,
            'preview_scope': 'CPU sampler ask preview only, no told scores, candidate execution, quality or performance inference.',
            'tools': tools, 'preserved_toy_probe_sources': copied_probes}
        assert audit['candidate_sha256'] == row['candidate_sha256']
        report['arms'][arm] = audit
    db.close()
    report['review'] = {
        'finding': 'Both fresh arms explicitly recognize accepted feedback and implement actual forward prefetch plus keep-tail. No demonstrated treatment superiority; the earlier unchanged-policy outcome is not causally isolated to the original sentence.',
        'budget_evidence': 'The current request omits parameter_trials, but previous-feedback contains the actual integer 3. Treatment reads and prints that integer after two inspection type errors; do not claim this information was unavailable or wholly unread.',
        'toy_scope': 'Treatment probes use a small CPU world_size=1 FSDP model; initial placeholder-call and mixed Tensor/DTensor failures are retained in tool receipts. Pipeline shell exit 0 did not make those probes successful. Later partial toy tests are not full two-rank GPT2Medium quality or performance.',
        'unverified_claims': ['Full bitwise equivalence of generated candidates before GPU execution',
            'Elimination of all old numerical failure modes', 'Predicted 10-18ms gain or exact memory safety from a few profiled gaps'],
        'production_adoption': 'Not adopted from this role-only result.',
        'historical_results': 'All accepted/rejected experiments and samples retained unchanged.'}
    write_json(root / 'terminal-audit.json', report)
    print(json.dumps({'audit': str(root/'terminal-audit.json'), 'arms': {k: {f:r[f] for f in ['session','schema_repairs','input_count','feasible_parameter_combinations','startup_ask_preview']} for k,r in report['arms'].items()}}))


if __name__ == '__main__':
    p=argparse.ArgumentParser(__doc__); p.add_argument('--root',type=Path,required=True)
    main(p.parse_args().root)
