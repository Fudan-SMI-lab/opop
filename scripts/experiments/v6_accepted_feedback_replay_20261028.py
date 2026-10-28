"""Two fresh candidate role replays; one neutral feedback sentence is the only treatment."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import traceback

from kernel_optimizer.config import load_config
from kernel_optimizer.control.direct_task import TaskSearch, TaskSpace
from kernel_optimizer.evaluation.task_eval import TaskEvaluation
from kernel_optimizer.project.agents import ProjectAgents
from kernel_optimizer.project.generated import validate_proposal
from kernel_optimizer.project.generation_snapshot import load
from kernel_optimizer.project.runtime_config import project_agent_config
from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.runtime.provenance import framework_source
from kernel_optimizer.store.run_store import RunStore
from kernel_optimizer.tuning.objective import Objective
from kernel_optimizer.wiring import Runtime

OLD = 'Read inputs/previous-feedback.txt and repair the concrete failure.'
NEW = ('Read inputs/previous-feedback.txt and determine the previous outcome from its evidence. '
       'Continue the current goal from the installed version; explain the concrete change in this attempt '
       'and how it follows from the feedback and selected optimization. Do not presume a failure when the previous attempt was accepted.')


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def tuner_replay(config, trial):
    """Use recorded observations only; never execute code or invent new scores."""
    expected = trial['parameter_search']['observations']
    actual = []
    class RecordedEvaluator:
        def evaluate(self, candidate_path, params, context):
            observation = expected[len(actual)]
            assert dict(params) == observation['bundle']['parameters']
            actual.append(dict(params))
            return TaskEvaluation.model_validate(observation['result'])
    budget = config.budgets.model_copy(update={'trials_per_space': trial['parameter_search']['parameter_trials'],
                                              'wall_clock_hours': float('inf')})
    search = TaskSearch(RecordedEvaluator(), Objective(direction='minimize', label='step_ms', unit='ms'), budget, device=config.device)
    winner = search.evaluate_candidate(Path(trial['candidate_agent']['sandbox']) / 'candidate.py',
                                      TaskSpace.model_validate(trial['proposal']['parameter_space']), {}, seed=config.run.seed)
    assert len(actual) == len(expected)
    return {'status': 'passed', 'seed': config.run.seed, 'parameter_trials': budget.trials_per_space,
            'observed_sequence': actual, 'selected': winner.params.values if winner else None,
            'declared_space': trial['proposal']['parameter_space'],
            'scope': 'CPU-only exact sampler replay with the original recorded scores; no new timing or GPU validation.'}


def main(args):
    os.umask(0o077)
    args.output.mkdir(parents=True, exist_ok=True)
    assert not (args.output / 'protocol.json').exists()
    config = project_agent_config(load_config(args.config, []))
    config.usage.mode = 'observe_only'
    saved, prompt, files = load(args.snapshot)
    assert saved['fresh_session'] and saved['role'] == 'candidate'
    assert saved['framework_source'] == framework_source()
    assert prompt.count(OLD) == 1
    feedback = json.loads(files['inputs/previous-feedback.txt'])
    assert feedback['decision']['accepted']
    write_json(args.output / 'protocol.json', {'at': now(), 'snapshot': str(args.snapshot),
        'core': framework_source(), 'change': {'old': OLD, 'new': NEW},
        'original_prompt_sha256': digest(prompt.encode()), 'original_model': saved['model'],
        'original_inputs': {k: {'bytes': len(v), 'sha256': digest(v)} for k, v in files.items()},
        'sequence': ['original sampler replay on CPU', 'fresh control candidate', 'fresh treatment candidate'],
        'scope': 'One fresh role per arm with normal schema repair; only one prompt sentence replaced. No new input artifacts, developer optimization diagnosis, GPU search, performance acceptance or production change. API use is observe-only.'})
    results = {}
    try:
        results['tuner-replay'] = tuner_replay(config, json.loads(args.trial.read_text()))
        write_json(args.output / 'tuner-replay.json', results['tuner-replay'])
        for arm in ['control', 'treatment']:
            write_json(args.output / 'state.json', {'at': now(), 'phase': 'running', 'stage': arm})
            store = RunStore.create(args.output, arm, {'kind': 'accepted-feedback-candidate-replay'})
            edit = None if arm == 'control' else {'path': 'prompt', 'old': OLD, 'new': NEW}
            try:
                with Runtime(config, log_dir=store.run_dir) as runtime:
                    directory, receipt = ProjectAgents(runtime, store).replay_snapshot(args.snapshot, edit=edit, validator=validate_proposal)
                results[arm] = {'status': 'completed', 'agent': receipt,
                    'candidate_sha256': digest((directory / 'candidate.py').read_bytes()),
                    'proposal': json.loads((directory / 'proposal.json').read_text()),
                    'scope': 'Generated output only; schema validity does not establish quality, execution or performance.'}
            except Exception:
                results[arm] = {'status': 'error', 'traceback': traceback.format_exc()}
            write_json(args.output / 'followup-results.json', results)
            print(json.dumps({'arm': arm, 'status': results[arm]['status'], 'at': now()}), flush=True)
        if all(results[a]['status'] == 'completed' for a in ['control', 'treatment']):
            left, right = [load(Path(results[a]['agent']['generation_snapshots'][0])) for a in ['control', 'treatment']]
            assert left[2] == right[2] == files
            assert left[1] == prompt and right[1] == prompt.replace(OLD, NEW)
            assert left[0]['tool_model_settings'] == right[0]['tool_model_settings'] == saved['tool_model_settings']
            assert left[0]['framework_source'] == right[0]['framework_source'] == saved['framework_source']
            results['initial-input-audit'] = {'status': 'passed', 'all_inputs_equal': True,
                'prompt_only_change': {'old': OLD, 'new': NEW},
                'model_tools_core_equal': True,
                'settings_sha256': digest(json.dumps(saved['tool_model_settings'], sort_keys=True).encode())}
    except Exception:
        results['controller'] = {'status': 'error', 'traceback': traceback.format_exc()}
    terminal = {'phase': 'completed' if all(r['status'] in {'completed','passed'} for r in results.values()) else 'completed_with_errors',
                'finished_at': now(), 'results': results}
    write_json(args.output / 'followup-results.json', results)
    write_json(args.output / 'state.json', terminal)
    write_json(args.output / 'followup-completed.json', terminal)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    for name in ['config', 'snapshot', 'trial', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    main(parser.parse_args())
