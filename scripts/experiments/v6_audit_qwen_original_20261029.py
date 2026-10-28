"""Read-only preparation audit; source naming failure is not a numerical verdict."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3

from kernel_optimizer.config import load_config
from kernel_optimizer.models.project import ProjectContract
from kernel_optimizer.project.generation_snapshot import load
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.runtime.execution import write_json
from scripts.experiments.v6_audit_fsdp_20261028 import read, sha
from scripts.experiments.v6_audit_fsdp_search_20261028 import agent_record
from scripts.experiments.v6_audit_qwen_end_to_end_20261029 import endpoint
from scripts.experiments.v6_ddp_semantics_20261027 import disk_observation
from scripts.experiments.v6_qwen_original_followup_20261029 import ADAPTER, MAPPING


def audit(root):
    terminal = read(root / 'followup-completed.json')
    assert terminal['phase'] == 'completed_with_errors'
    assert all(terminal['results'][k]['status'] == 'not_run' for k in ['search', 'independent', 'native'])
    s = ProjectSearch(load_config(root / 'private-config.yaml', []), root / 'qwen-original')
    protocol = read(root / 'protocol.json')
    source = Path(s.prepared['controls_path']).parent / 'reference-source'
    assert sha(source / 'opop_adapter.py') == s.prepared['adapter_sha256'] == ADAPTER
    current = s.revisions.current()
    assert not current.bundles
    identities = {f.path: f.sha256 for f in current.files}
    for name, value in identities.items():
        assert sha(source / name) == value
    for row in read(root / 'qwen-original.intake/source-manifest.json')['files']:
        assert sha(row['source']) == row['sha256'] == identities[row['path']]
        assert protocol['source_files'][row['path']] == row['sha256']
    source_reference = protocol['independent_source_reference']
    assert sha(source_reference['path']) == source_reference['sha256']
    assert sha(Path(s.prepared['controls_path']) / 'reference-rank-0.pt') == terminal['results']['source']['controls_reference_sha256']
    controls = endpoint(s.prepared['evaluation'], protocol['core'], controls=True)
    request = read(s.output / 'request.json')
    assert request['objective'] == {'name': 'end_to_end_ms', 'unit': 'ms', 'direction': 'minimize',
                                    'constraints': [{'metric': 'ttft_ms', 'op': 'le', 'baseline_ratio': 1.05}]}
    assert request['quality'] == {'atol': 0.02, 'rtol': 0.02}
    assert request['measurement_lifetime'] == 'invocation'
    assert request['workload']['new_tokens'] == 32 and request['continuous_steps'] == 2
    database = sqlite3.connect('file:/root/autodl-tmp/dev/v6-work/private-runtime/data/opencode/opencode.db?mode=ro', uri=True)
    agents = []
    for agent in [read(root / 'qwen-original.intake/result.json')['agent'], s.prepared['agent']]:
        row = agent_record(agent, database)
        meta, prompt, files = load(Path(agent['generation_snapshots'][0]))
        assert meta['fresh_session'] and meta['framework_source'] == protocol['core']
        row['fresh_session'] = meta['fresh_session']
        row['tools_actual_inputs'] = []
        row['old_result_tool_mentions'] = []
        for ident, raw in database.execute('select id,data from part where session_id=? order by time_created,id', (agent['session'],)):
            part = json.loads(raw)
            if part.get('type') != 'tool':
                continue
            state = part.get('state', {}); inp = state.get('input', {})
            serialized = json.dumps(inp, sort_keys=True)
            row['tools_actual_inputs'].append({'part': ident, 'tool': part['tool'], 'status': state.get('status'),
                'path': inp.get('filePath'), 'command_sha256': hashlib.sha256(str(inp.get('command', '')).encode()).hexdigest()})
            words = ['qwen-end-to-end-20261028', 'qwen-goal-contrast-20261028', 'metric-mapping-decode-20261028',
                     'trial-f71b5c1677', '24ebefd7', 'autonomous-cohort-20261027/qwen-ttft']
            hits = [word for word in words if word in serialized]
            if hits:
                row['old_result_tool_mentions'].append({'part': ident, 'matched': hits})
        agents.append(row)
    database.close()
    result = {
        'at': datetime.now(timezone.utc).isoformat(), 'terminal': terminal, 'terminal_sha256': sha(root / 'followup-completed.json'),
        'core': protocol['core'], 'intent': read(root / 'qwen-original.intake/intent.json'),
        'request': request, 'contract': s.revisions.load(current.contract_ref, ProjectContract).model_dump(mode='json'),
        'source_files': identities, 'baseline_revision': current.id, 'active_bundles': [],
        'controls': controls, 'fresh_baseline_ttft_limit_ms': controls['metrics']['ttft_ms'] * 1.05,
        'failed_source_check': read(root / 'source-check.json'), 'independent_reference': source_reference,
        'reviewed_mapping': MAPPING, 'controls_reference_sha256': terminal['results']['source']['controls_reference_sha256'],
        'agents': agents,
        'code_review': {
            'mapping': 'observe generated_tokens is run tokens.detach, and step_logits is run full stacked logits.detach. Both come from the full original infer mirror, not a reduced/renormalized output.',
            'complete_work': 'One prefill plus31cached single-token forwards, fixed32token greedy output, logits_to_keep1, no EOS early exit. Teacher-forced32step replay is in observe outside measurement, as source correctness work.',
            'timers': 'Original sync before start, first-token sync, final sync retained. The framework also brackets run; primary and TTFT use the adapter internal source clocks. Phase annotations add instrumentation and are not assumed zero-cost.',
            'owners': 'prepare owns model, tokenizer, inputs and CPU dimensions/metadata; no prepared CUDA reference logits or cache owner in this adapter. _infer locals own output/cache until return; quality result owns full logits/tokens, clean result tokens/CPUmetrics. Invocation lifetime clears last quality result before warmup and releases each clean result after metrics before next invocation. No model/cache global owner introduced in reviewed code.',
            'memory_limits': 'Source clean assignment discards returned tokens before peak query; framework keeps current tokens (32int64 values, allocator rounding not separately measured) through metrics then releases. Both computation peaks include tokens. Framework allocated/lifecycle fields are not numerically identical source metrics, and lifecycle includes quality/prepare history; no cross-adapter memory gain inferred.',
            'math': 'Source BF16/SDPA, TF32off/highest, seed and prompt construction preserved. requires_gradFalse differs from original parameter flags but run is inference_mode in both; no source gradient workload exists here.',
            'contract': 'New completion text correctly calls first boundary ttft_ms and complete generation end_to_end_ms; old variant stale prose is not inherited. Source one-device workload preserved.',
            'next': 'Only external reviewed mapping is needed. No production prompt/adapter/request edits. Compare full existing controls to independent source in a recorded CPU worker, then continue unstarted search with shared read-only controls.',
            'limits': 'This is manual source review plus controls, not independent numerical source pass yet. Tool input path audit is not exhaustive external filesystem isolation.'},
        'disk': disk_observation(Path('/root/autodl-tmp'))}
    write_json(root / 'preparation-audit.json', result)
    print(json.dumps({'status': 'audited', 'file': str(root / 'preparation-audit.json'),
        'bytes': (root / 'preparation-audit.json').stat().st_size, 'roles': len(agents),
        'schema_repairs': [a['contract_repairs'] for a in agents],
        'old_result_tool_mentions': [len(a['old_result_tool_mentions']) for a in agents],
        'ttft_limit_ms': result['fresh_baseline_ttft_limit_ms'], 'control_samples': len(controls['samples'])}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__); p.add_argument('--root', type=Path, required=True)
    audit(p.parse_args().root)
