"""Fixed GC diagnosis plus a one-input-file real reviewer presentation replay."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import traceback

from kernel_optimizer.config import load_config
from kernel_optimizer.project.agents import ProjectAgents
from kernel_optimizer.project.generation_snapshot import load
from kernel_optimizer.project.runtime_config import project_agent_config
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.runtime.execution import LocalExecutor, write_json
from kernel_optimizer.runtime.provenance import framework_source
from kernel_optimizer.store.run_store import RunStore
from kernel_optimizer.wiring import Runtime
from scripts.experiments.v6_box43_next_stage_20261027 import environment, now
from scripts.experiments.v6_ddp_semantics_20261027 import disk_observation
from scripts.experiments.v6_ddp_tp_diagnostics_20261028 import ddp_request, endpoint_summary


def gc_summary(path, warmup):
    data = json.loads(path.read_text())
    starts, intervals = {}, []
    for event in data['events']:
        key = (event['thread'], event['generation'])
        if event['phase'] == 'start':
            starts[key] = event
        elif key in starts:
            first = starts.pop(key)
            intervals.append({'start_ns': first['at_ns'], 'end_ns': event['at_ns'],
                'generation': event['generation'], 'collected': event.get('collected'),
                'duration_ms': (event['at_ns']-first['at_ns'])/1e6,
                'trigger_stack': first.get('trigger_stack', [])})
    rows = []
    for item in data['invocations']:
        if item['mode'] != 'measure' or item['index'] < warmup or 'end_ns' not in item:
            continue
        overlaps = [{**event, 'bracket_overlap_ms':
                     max(0, min(item['end_ns'], event['end_ns'])-max(item['start_ns'], event['start_ns']))/1e6}
                    for event in intervals if event['start_ns'] < item['end_ns'] and event['end_ns'] > item['start_ns']]
        def cpu_stats(value):
            return {line.split()[0]: int(line.split()[1]) for line in value.splitlines()}
        before, after = (cpu_stats(item[k]['cgroup_cpu_stat']) for k in ['before', 'after'])
        rows.append({'sample_index': item['index']-warmup, 'elapsed_ms': item['evaluator_elapsed_ms'],
            'gc_enabled': item['gc_enabled'], 'gc_overlaps': overlaps,
            'cgroup_cpu_delta': {k: after[k]-before[k] for k in before},
            'process_cpu_s': sum(item['after'][k]-item['before'][k]
                                 for k in ['process_user_s', 'process_system_s']),
            'minor_fault_delta': item['after']['minor_faults']-item['before']['minor_faults'],
            'major_fault_delta': item['after']['major_faults']-item['before']['major_faults']})
    return {'rank': data['rank'], 'samples': rows, 'scope': data['scope'],
            'artifact': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'unclosed_gc_intervals': len(starts)}


def gc_experiment(args, config):
    search = ProjectSearch(config, args.source/'ddp-communication')
    candidate = search.revisions.current()
    assert candidate.id == '75e28d3d2ccd5fab06175ef1b827a325ed2159d011758c7b2deccda6173b9e2d'
    order = ['U', 'O', 'D', 'D', 'O', 'U']
    modes = {'U': 'unchanged', 'O': 'observe', 'D': 'disable_during_measure_phase'}
    root = args.output/'ddp-gc'
    write_json(root/'protocol.json', {'at': now(), 'candidate': candidate.id, 'order': order,
        'modes': modes, 'warmup': search.request.warmup, 'repeats': 20,
        'scope': 'Same fixed C version and parameters throughout. O adds passive GC callbacks and '
                 'outside-timer CPU counters; D differs from O only by disabling cyclic GC for the measure '
                 'phase. U checks observer effects. Full quality and all samples retained. No publication '
                 'or formal acceptance, no forced GC, no until-pass repeat, no CPU quota modification.'})
    rows = []
    for index, arm in enumerate(order):
        job = root/f'{index}-{arm}'
        request = ddp_request(search, candidate, job, arm='C', action='evaluate')
        request = request.model_copy(update={
            'entry': Path(__file__).with_name('v6_ddp_gc_probe_20261028.py'), 'function': 'run',
            'parameters': {**request.parameters, 'gc_probe': modes[arm]}})
        write_json(job/'protocol.json', request.model_dump(mode='json'))
        before = environment(search.request.devices)
        execution = LocalExecutor().run(request)
        row = {'index': index, 'arm': arm, 'before': before, 'after': environment(search.request.devices),
               'receipt': str(job/'run/receipt.json'), **endpoint_summary(execution)}
        row['gc'] = [gc_summary(path, search.request.warmup) for path in sorted((job/'run').glob('gc-rank-*.json'))]
        rows.append(row)
        write_json(root/'result.json', {'endpoints': rows})
        print(json.dumps({'stage': 'ddp-gc', 'index': index, 'arm': arm, 'valid': row['valid']}), flush=True)
    return {'status': 'completed' if all(r['valid'] for r in rows) else 'completed_with_errors',
            'result': str(root/'result.json')}


def compact_tp(raw):
    """Same comparison values, named references, raw paths; no causal answer added."""
    source = json.loads(raw)
    result = {'status': source['status'], 'ranks': [],
              'table_columns': ['variant', 'status', 'exact_equal', 'elements', 'max_abs_error', 'rms_error']}
    for rank in source['ranks']:
        value = rank['result']
        record = {'rank': rank['rank'], 'error': rank.get('error'), 'scope': value['scope'],
            'source_observations': value['source_observations'], 'artifact': value['artifact'], 'projections': []}
        for projection in value['projections']:
            item = {k: projection[k] for k in ['name', 'status', 'input', 'prior_tp_input', 'weight',
                'checkpoint_sha256', 'checkpoint_row_slice', 'same_input', 'bias_present']}
            item['comparisons'] = {ref: [[variant, v['status'], v['exact_equal'], v['elements'],
                                         v['max_abs_error'], v['rms_error']] for variant, v in projection[ref].items()]
                for ref in ['against_recorded_single_slice', 'against_recorded_tp', 'against_official']}
            record['projections'].append(item)
        result['ranks'].append(record)
    # Keep each comparison row on one line, without a giant minified input line.
    markers, serial = {}, copy.deepcopy(result)
    for rank in serial['ranks']:
        for item in rank['projections']:
            for ref, rows in item['comparisons'].items():
                for i, row in enumerate(rows):
                    key = f'__COMPARISON_ROW_{len(markers)}__'
                    markers[json.dumps(key)] = json.dumps(row)
                    rows[i] = key
    text = json.dumps(serial, indent=2)
    for marker, row in markers.items():
        text = text.replace(marker, row)
    assert json.loads(text) == result
    return text


def review_ab(args, config):
    prior = json.loads((args.previous/'diagnostic-review/result.json').read_text())
    snapshot = Path(prior['agent']['generation_snapshots'][0])
    saved, prompt, inputs = load(snapshot)
    filename = 'inputs/tp-summary.json'; old = inputs[filename].decode(); new = compact_tp(old)
    root = args.output/'review-ab'; root.mkdir(parents=True)
    write_json(root/'protocol.json', {'at': now(), 'snapshot': str(snapshot), 'order': ['control', 'treatment'],
        'only_changed_file': filename, 'old_bytes': len(old.encode()), 'new_bytes': len(new.encode()),
        'old_sha256': hashlib.sha256(old.encode()).hexdigest(), 'new_sha256': hashlib.sha256(new.encode()).hexdigest(),
        'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
        'scope': 'Two fresh real sessions. Same model, framework, prompt and other inputs. Only TP JSON '
                 'presentation changes; comparison values retained, output-metadata duplication indexed '
                 'by the original raw files. No answer/correction or new GC evidence is supplied. '
                 'External filesystem/provider state is not frozen. No production prompt adoption.'})
    (root/'treatment-tp-summary.json').write_text(new)

    def validate(directory):
        data = json.loads((directory/'review.json').read_text())
        if not isinstance(data.get('findings'), list) or not isinstance(data.get('unresolved'), list) \
                or not isinstance(data.get('next_minimal_experiment'), dict):
            raise ValueError('review schema requires findings, unresolved and next_minimal_experiment')

    rows = []
    for arm in ['control', 'treatment']:
        store = RunStore.create(root, arm, {'kind': 'tp-evidence-presentation-replay', 'arm': arm})
        try:
            edit = None if arm == 'control' else {'path': filename, 'old': old, 'new': new}
            with Runtime(project_agent_config(config), store.run_dir) as runtime:
                directory, receipt = ProjectAgents(runtime, store).replay_snapshot(snapshot, edit=edit, validator=validate)
            manifest, actual_prompt, actual_inputs = load(Path(receipt['generation_snapshots'][0]))
            changed = [k for k in sorted(set(inputs)|set(actual_inputs)) if inputs.get(k) != actual_inputs.get(k)]
            assert actual_prompt == prompt and changed == ([] if arm == 'control' else [filename])
            assert manifest['framework_source'] == saved['framework_source']
            rows.append({'arm': arm, 'status': 'completed', 'agent': receipt, 'changed_inputs': changed,
                         'review': str(directory/'review.json'), 'same_prompt': True})
        except Exception:
            rows.append({'arm': arm, 'status': 'error', 'traceback': traceback.format_exc()})
        write_json(root/'result.json', {'runs': rows})
        print(json.dumps({'stage': 'review-ab', 'arm': arm, 'status': rows[-1]['status']}), flush=True)
    return {'status': 'completed' if all(r['status'] == 'completed' for r in rows) else 'completed_with_errors',
            'result': str(root/'result.json')}


def main():
    parser = argparse.ArgumentParser(__doc__)
    for name in ['config', 'output', 'source', 'previous']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args(); os.umask(0o077)
    args.output.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config, [])
    write_json(args.output/'protocol.json', {'at': now(), 'framework_source': framework_source(),
        'sequence': ['fixed U/O/D/D/O/U DDP GC diagnostics', 'real review control/treatment'],
        'source': str(args.source), 'previous': str(args.previous),
        'disk': disk_observation(Path('/root/autodl-tmp')),
        'scope': 'All samples preserved; no production change or formal acceptance. '
                 'No bulk local transfer, Goal, API caps or large TP run.'})
    results = {}
    for name, operation in [('ddp-gc', lambda: gc_experiment(args, config)),
                            ('review-ab', lambda: review_ab(args, config))]:
        write_json(args.output/'state.json', {'at': now(), 'phase': 'running', 'stage': name})
        try:
            results[name] = operation()
        except Exception:
            results[name] = {'status': 'error', 'traceback': traceback.format_exc()}
        write_json(args.output/'followup-results.json', results)
    terminal = {'finished_at': now(), 'phase': 'completed' if all(v['status'] == 'completed' for v in results.values())
                else 'completed_with_errors', 'results': results, 'disk': disk_observation(Path('/root/autodl-tmp'))}
    write_json(args.output/'state.json', terminal)
    write_json(args.output/'followup-completed.json', terminal)


if __name__ == '__main__':
    main()
