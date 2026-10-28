"""Two fresh real candidate roles; only one existing feedback file gains observed facts."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import traceback

from kernel_optimizer.config import load_config
from kernel_optimizer.project.agents import ProjectAgents
from kernel_optimizer.project.generated import validate_proposal
from kernel_optimizer.project.generation_snapshot import load, edit_one
from kernel_optimizer.project.runtime_config import project_agent_config
from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.runtime.provenance import framework_source
from kernel_optimizer.store.run_store import RunStore
from kernel_optimizer.wiring import Runtime


FIELD = 'fixed_full_output_relation_diagnostic'
INPUT = 'inputs/previous-feedback.txt'


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def prepare(snapshot, audit_path):
    saved, prompt, files = load(snapshot)
    assert saved['fresh_session'] and saved['role'] == 'candidate'
    assert saved['framework_source'] == framework_source()
    audit = json.loads(audit_path.read_text())
    assert audit['C1_C2_full_greedy_exactly_equal']
    assert audit['endpoints']['0-B-observed']['status'] == 'passed'
    assert all(v['status'] == 'failed' for k, v in audit['endpoints'].items() if k != '0-B-observed')
    for name in ['2-C1-observed', '4-C2-observed']:
        rows = audit['observations'][name]['observations']
        assert rows[0]['max_abs_error_by_position'][0][:2] == [0.0, .1875]
        assert max(rows[0]['max_abs_error_by_position'][0]) == 1
        assert max(rows[1]['max_abs_error_by_position'][0]) == 0
        assert audit['observations'][name]['full_value_relations_recomputed']
    facts = {
        'scope': 'External fixed diagnostic evidence, not candidate acceptance or an optimization prescription. Original failed runs remain failed.',
        'candidates': ['trial-d5c20fcae5', 'trial-e74f5ba6cf'],
        'fixed_execution': 'Each exact original candidate/request/parameters was run once unobserved and once observed. All four workers reproduced the original teacher-versus-greedy assertion. Observed baseline passed.',
        'full_output_comparison': 'For BOTH observed candidates, greedy logits already differ from the independent full source before observe: max abs 1.0; position 0 exactly equal, first decode position 1 max abs 0.1875. All 32 tokens match and all logits are finite. Teacher tokens/logits equal source exactly. Full greedy tokens/logits are bitwise unchanged before/after observe; C1 and C2 greedy values are bitwise identical.',
        'comparison_scope': 'Full 1x32x151936 BF16 logits, original atol=rtol=0.02. On the SAME saved tensors, teacher(actual)-versus-greedy(expected) assert_close reports 1,130,457 mismatches on CPU and 1,132,088 on CUDA, the latter reproducing the original assertion; max abs 1.0 on both. Direction/device-specific counts must not be conflated.',
        'limits': 'CPU snapshots add synchronization, host work and lifetimes. Reproducing the unobserved error corroborates this fixed diagnostic, not a unique historical mechanism. Unchanged output during observe does not exclude corruption during generation. No alias, shared-pool, cache, arithmetic or other unique cause is established. Route labels do not alone establish native paths.',
        'diagnostic_api_scope': 'Existing candidate_diagnostics valid=true is execution/coverage only: that action calls run, not observe/compare. It does not certify quality or performance.',
        'evidence': {'path': str(audit_path), 'sha256': digest(audit_path.read_bytes())},
    }
    old = files[INPUT].decode()
    feedback = json.loads(old)
    assert FIELD not in feedback and feedback['decision']['accepted'] is False
    end = old.rfind('}')
    assert old[end:].strip() == '}'
    # Preserve all original field bytes; insert only one top-level member.
    new = old[:end] + ',\n  ' + json.dumps(FIELD) + ': ' + json.dumps(facts, ensure_ascii=False) + '\n' + old[end:]
    changed = json.loads(new)
    assert changed.pop(FIELD) == facts and changed == feedback
    edit = {'path': INPUT, 'old': old, 'new': new}
    revised_prompt, revised_files = edit_one(prompt, files, edit)
    assert revised_prompt == prompt
    assert [k for k in files if files[k] != revised_files[k]] == [INPUT]
    return saved, prompt, files, revised_files, edit, facts


def main(args):
    os.umask(0o077)
    saved, prompt, files, revised, edit, facts = prepare(args.snapshot, args.audit)
    config = project_agent_config(load_config(args.config, [])); config.usage.mode = 'observe_only'
    assert saved['model'] == config.agents.default_model and saved['agent'] == config.opencode.agent
    protocol = {'at': now(), 'snapshot': str(args.snapshot), 'core': framework_source(),
        'prompt_sha256': digest(prompt.encode()), 'input_count': len(files),
        'original_inputs': {k: {'bytes': len(v), 'sha256': digest(v)} for k, v in files.items()},
        'change': {'path': INPUT, 'added_field': FIELD, 'old_bytes': len(files[INPUT]),
                   'new_bytes': len(revised[INPUT]), 'new_sha256': digest(revised[INPUT]), 'facts': facts},
        'settings_sha256': digest(json.dumps(saved['tool_model_settings'], sort_keys=True).encode()),
        'sequence': ['fresh control candidate', 'fresh treatment candidate'],
        'scope': 'One role each with normal schema repair. Prompt/model/tools/core/other input files unchanged. External filesystem is not a hermetic snapshot. Original selection and feedback fields preserved. No developer candidate/causal answer, new autonomous search or formal acceptance. Agent-initiated tool probes are recorded, not independent validation. Production unchanged; API usage observe-only.'}
    if args.preflight:
        print(json.dumps({'status': 'passed', 'core': framework_source(), 'input_count': len(files),
                          'change_bytes': len(revised[INPUT])-len(files[INPUT]), 'path': INPUT,
                          'original_bytes': len(files[INPUT]), 'original_fields_preserved': True}))
        return
    assert not (args.output / 'protocol.json').exists(), 'Do not restart an existing batch'
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / 'protocol.json', protocol)
    results = {}
    stores = {}
    try:
        for arm in ['control', 'treatment']:
            write_json(args.output / 'state.json', {'at': now(), 'phase': 'running', 'stage': arm})
            store = RunStore.create(args.output, arm, {'kind': 'qwen-full-output-feedback-replay'})
            stores[arm] = store
            try:
                with Runtime(config, log_dir=store.run_dir) as runtime:
                    directory, receipt = ProjectAgents(runtime, store).replay_snapshot(
                        args.snapshot, edit=None if arm == 'control' else edit, validator=validate_proposal)
                results[arm] = {'status': 'completed', 'agent': receipt,
                    'candidate_sha256': digest((directory / 'candidate.py').read_bytes()),
                    'proposal': json.loads((directory / 'proposal.json').read_text()),
                    'scope': 'Generation/schema only; no independent GPU quality or performance acceptance.'}
            except Exception:
                results[arm] = {'status': 'error', 'traceback': traceback.format_exc()}
            write_json(args.output / 'followup-results.json', results)
            print(json.dumps({'at': now(), 'arm': arm, 'status': results[arm]['status']}), flush=True)
        audit = {}
        for arm, store in stores.items():
            candidates = [p for p in (store.run_dir / 'generation-snapshots').glob('*.json') if json.loads(p.read_text())['fresh_session']]
            assert len(candidates) == 1, (arm, candidates)
            actual, actual_prompt, actual_files = load(candidates[0])
            assert actual_prompt == prompt
            assert actual_files == (files if arm == 'control' else revised)
            assert actual['framework_source'] == saved['framework_source']
            assert actual['tool_model_settings'] == saved['tool_model_settings']
            audit[arm] = {'snapshot': str(candidates[0]), 'all_initial_inputs_match_declared_arm': True,
                          'prompt_model_tools_core_equal': True}
        results['initial-input-audit'] = {'status': 'passed', 'arms': audit, 'changed_file_only': INPUT,
                                         'original_feedback_fields_preserved': True}
    except Exception:
        results['controller'] = {'status': 'error', 'traceback': traceback.format_exc()}
    terminal = {'phase': 'completed' if all(r['status'] in ['passed', 'completed'] for r in results.values()) else 'completed_with_errors',
                'finished_at': now(), 'results': results}
    write_json(args.output / 'followup-results.json', results)
    write_json(args.output / 'state.json', terminal)
    write_json(args.output / 'followup-completed.json', terminal)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    for name in ['config', 'snapshot', 'audit', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--preflight', action='store_true')
    main(parser.parse_args())
