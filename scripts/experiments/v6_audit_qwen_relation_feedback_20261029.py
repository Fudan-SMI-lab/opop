"""Audit actual role inputs, tool evidence and generated code; no quality claims."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3

from kernel_optimizer.project.generation_snapshot import load
from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.runtime.provenance import framework_source
from scripts.experiments.v6_audit_fsdp_20261028 import read, sha
from scripts.experiments.v6_qwen_relation_feedback_replay_20261029 import prepare, INPUT, FIELD


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def main(root):
    protocol = read(root / 'protocol.json'); terminal = read(root / 'followup-completed.json')
    assert terminal['phase'] == 'completed'
    saved, prompt, original, treatment, _, facts = prepare(Path(protocol['snapshot']), Path(protocol['change']['facts']['evidence']['path']))
    assert protocol['core'] == framework_source() == saved['framework_source']
    assert sha(protocol['change']['facts']['evidence']['path']) == protocol['change']['facts']['evidence']['sha256']
    audit_dir = root / 'audit'; audit_dir.mkdir(exist_ok=True)
    report = {'at': datetime.now(timezone.utc).isoformat(), 'finished_at': terminal['finished_at'],
        'core': framework_source(), 'terminal_sha256': sha(root / 'followup-completed.json'),
        'protocol_sha256': sha(root / 'protocol.json'), 'initial_input_audit': {}, 'roles': {},
        'scope': 'Real generation/input/tool audit only. CPU probes and schema completion do not certify full GPU quality, TTFT or optimization acceptance.'}
    db = sqlite3.connect('file:/root/autodl-tmp/dev/v6-work/private-runtime/data/opencode/opencode.db?mode=ro', uri=True)
    for arm, expected in [('control', original), ('treatment', treatment)]:
        result = terminal['results'][arm]; agent = result['agent']
        assert result['status'] == 'completed'
        metadata, actual_prompt, files = load(Path(agent['generation_snapshots'][0]))
        assert files == expected and actual_prompt == prompt
        assert metadata['tool_model_settings'] == saved['tool_model_settings']
        assert metadata['framework_source'] == saved['framework_source']
        sandbox = Path(agent['sandbox'])
        assert all((sandbox / n).read_bytes() == v for n, v in files.items())
        assert sha(sandbox / 'candidate.py') == result['candidate_sha256']
        proposal = read(sandbox / 'proposal.json'); assert proposal == result['proposal']
        assert proposal['parameter_space'] == {} and proposal['requires'] == proposal['replaces'] == []
        report['initial_input_audit'][arm] = {'snapshot': agent['generation_snapshots'][0],
            'files': {n: {'bytes': len(v), 'sha256': hashlib.sha256(v).hexdigest()} for n, v in files.items()},
            'changed_inputs_vs_original': [n for n in files if files[n] != original[n]],
            'prompt_sha256': digest(prompt), 'settings_sha256': digest(json.dumps(metadata['tool_model_settings'], sort_keys=True)),
            'input_files_unchanged_after_role': True}
        parts = [{'part_id': ident, 'data': json.loads(data)} for ident, data in db.execute(
            'select id,data from part where session_id=? order by time_created,id', (agent['session'],))]
        raw_path = audit_dir / (arm + '-session-parts.json')
        if raw_path.exists():
            assert read(raw_path) == parts
        else:
            write_json(raw_path, parts)
        tools = []; selected = []; probe_paths = set(); texts = []
        for row in parts:
            part = row['data']
            if part.get('type') == 'text':
                text = part.get('text', '')
                texts.append({'part': row['part_id'], 'bytes': len(text.encode()), 'sha256': digest(text)})
            if part.get('type') != 'tool':
                continue
            state = part.get('state', {}); inp = state.get('input', {}); command = inp.get('command', '')
            output = str(state.get('output', ''))
            encoded = json.dumps(inp, sort_keys=True)
            item = {'part': row['part_id'], 'tool': part['tool'], 'status': state.get('status'),
                'input_sha256': digest(encoded), 'input_bytes': len(encoded.encode()),
                'path': inp.get('filePath'), 'command_preview': command[:400],
                'output_sha256': digest(output), 'output_bytes': len(output.encode()),
                'exit': state.get('metadata', {}).get('exit'),
                'output_has_traceback': 'Traceback (most recent call last)' in output,
                'new_fact_field_in_input': FIELD in encoded, 'new_fact_field_in_output': FIELD in output}
            tools.append(item)
            path = inp.get('filePath')
            if path and path.startswith('/tmp/opencode/') and part['tool'] in ['write', 'edit']:
                probe_paths.add(path)
            if ((FIELD in command or 'previous-feedback' in command) and len(command) < 2000) or (
                ('/tmp/opencode' in command or 'candidate_mock' in command) and len(output) < 1600):
                selected.append({**item, 'command': command, 'output': output[:4000]})
        preserved = []
        for p in sorted(probe_paths):
            path = Path(p)
            if path.is_file():
                target = audit_dir / ('probe-' + arm + '-' + path.name)
                if target.exists(): assert target.read_bytes() == path.read_bytes()
                else: shutil.copyfile(path, target)
                preserved.append({'original_path': p, 'saved_path': str(target), 'sha256': sha(target),
                                  'scope': 'Final surviving probe file. Earlier writes/edits and inline probes retained in session-parts JSON.'})
        compact_proposal = {k: v for k, v in proposal.items() if k != 'eliminated_calls'}
        compact_proposal['eliminated_calls'] = {'count': len(proposal['eliminated_calls']),
            'sha256': digest(json.dumps(proposal['eliminated_calls'])), 'scope': 'Declaration only, not native execution proof.'}
        report['roles'][arm] = {'agent': {k: agent[k] for k in ['role', 'session', 'model', 'usage', 'sandbox', 'contract_repairs', 'generation_snapshots']},
            'candidate_sha256': sha(sandbox / 'candidate.py'), 'proposal_sha256': sha(sandbox / 'proposal.json'),
            'proposal': compact_proposal, 'tools': tools, 'selected_tool_evidence': selected, 'text_index': texts,
            'tool_counts': dict(Counter((t['tool'] + ':' + str(t['status'])) for t in tools)),
            'probe_files': preserved, 'raw_session_parts': {'path': str(raw_path), 'sha256': sha(raw_path)}}
    db.close()
    diff = json.loads(treatment[INPUT]); assert diff.pop(FIELD) == facts and diff == json.loads(original[INPUT])
    report['single_change'] = {'path': INPUT, 'old_bytes': len(original[INPUT]), 'new_bytes': len(treatment[INPUT]),
                              'original_fields_preserved': True, 'other_files_prompt_settings_core_equal': True}
    prior_audit = read(Path(protocol['change']['facts']['evidence']['path']))
    identities = {**prior_audit['all_frozen_input_hashes_verified'], **prior_audit['controls_hashes']}
    assert all(sha(p) == h for p, h in identities.items())
    report['unchanged_frozen_source_controls_requests'] = identities
    report['review'] = {
        'treatment_read': 'Tool prt_184415d8d0017jxOmX5QpyntIw printed all factual relation/limits/diagnostic API text. Rationale incorporates greedy-before-observe divergence and exact teacher. Audit path/raw tensors were not separately read in the recorded tools.',
        'control_read': 'Read original previous-feedback and previous candidate; no new fact field supplied. Final rationale describes greedy/teacher divergence, without independently establishing which side matches source.',
        'both': 'One StaticCache graph using original parent, current invocation KV refresh, logits clones; empty parameter space. No successful old candidate imported. Existing quality/TTFT constraints not edited.',
        'differences': 'Control explicit mutable 2D mask, static-cache refresh lazy in first decode, caller DynamicCache returned without normal-step KV rebinding; only fallback materializes KV. Treatment no explicit 2D mask, StaticCache source computes mask, normal-step caller layers rebound to static narrow views, shape/identity step checks. Treatment allocates static_input in build but StaticCache is actually lazy in _setup (proposal incorrectly says StaticCache allocated at build). Control eliminated_calls empty; treatment declares 23225. Neither declaration proves native paths.',
        'probe_limits': 'CPU tiny random models, some FP32 and some BF16, several patched mock runner methods replacing capture/replay. Initial token/shape/type/probe failures retained. shell pipeline status completed/exit0 may only report tail. Real CUDA Graph, full pretrained model, original complete32-token quality, TTFT and lifetimes remain unverified.',
        'unsupported_claims': 'Both overstate a unique shared-pool explanation and unchanged-prefill guarantee of TTFT. Profile timing differences do not isolate host causal cost or validate new native paths. Treatment previous view alias stability does not extend across refresh of a later invocation. No auto adoption or causal superiority conclusion.'}
    write_json(root / 'terminal-audit.json', report)
    print(json.dumps({'audit': str(root / 'terminal-audit.json'), 'bytes': (root / 'terminal-audit.json').stat().st_size,
        'sha256': sha(root / 'terminal-audit.json'), 'single_change': report['single_change'],
        'roles': {a: {'candidate_sha256': v['candidate_sha256'], 'tools': v['tool_counts'], 'probe_files': len(v['probe_files']),
                      'tracebacks': sum(t['output_has_traceback'] for t in v['tools'])} for a, v in report['roles'].items()}}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__); parser.add_argument('--root', type=Path, required=True)
    main(parser.parse_args().root)
