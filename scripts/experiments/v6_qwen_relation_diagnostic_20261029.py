"""Fixed full-output relation diagnostics; no search or acceptance."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import traceback

from kernel_optimizer.runtime.execution import ExecutionRequest, LocalExecutor, write_json
from kernel_optimizer.runtime.provenance import framework_source
from scripts.experiments.v6_audit_fsdp_20261028 import read, sha
from scripts.experiments.v6_ddp_semantics_20261027 import disk_observation


def now():
    return datetime.now(timezone.utc).isoformat()


def observed_worker(context):
    import torch
    from kernel_optimizer.project import evaluation
    from kernel_optimizer.project.binding import BindingSession
    params = context['parameters']; output = Path(context['output'])
    reference = torch.load(params['diagnostic_source_reference'], map_location='cpu', weights_only=True, mmap=True)
    expected = {k: reference[k] for k in ['tokens', 'logits']}
    quality = params['request']['quality']; records = []; relations = []; routes = Counter()
    phase = 'setup'; current_run = None; last_teacher = None
    original_load = evaluation.load_adapter; original_route = BindingSession.record_route

    def capture(tokens, logits, label):
        # CPU copies preserve full raw values. They add host work/synchronization;
        # observed runs cannot supply timing or unobserved-lifetime conclusions.
        values = {'tokens': tokens.detach().cpu().clone(), 'logits': logits.detach().cpu().clone()}
        path = output / f'observation-{len(records):02d}-{label}.pt'
        torch.save(values, path)
        delta = (values['logits'].float() - expected['logits'].float()).abs()
        tolerance = quality['atol'] + quality['rtol'] * expected['logits'].float().abs()
        row = {'phase': label, 'path': str(path), 'sha256': sha(path),
            'source_comparison': evaluation.compare(values, expected, quality),
            'exact_source_tokens': bool(torch.equal(values['tokens'], expected['tokens'])),
            'finite_logits': bool(torch.isfinite(values['logits']).all()),
            'max_abs_error_by_position': [[v if math.isfinite(v) else None for v in batch]
                                        for batch in delta.flatten(2).amax(-1).tolist()],
            'outside_tolerance_by_position': (delta > tolerance).flatten(2).sum(-1).tolist()}
        records.append(row)
        return values, row

    def load_adapter(path):
        adapter = original_load(path)
        original_infer = adapter._infer; original_run = adapter.run; original_observe = adapter.observe

        def infer(*args, **kwargs):
            nonlocal current_run, last_teacher
            result = original_infer(*args, **kwargs)
            if kwargs.get('capture', False) and result[1] is not None:
                saved = capture(result[0], result[1], 'teacher' if phase == 'observe' else 'greedy')
                if phase == 'observe': last_teacher = saved
                else: current_run = saved
            return result

        def run(state):
            nonlocal phase
            previous = phase; phase = 'run'
            try: return original_run(state)
            finally: phase = previous

        def observe(state, result):
            nonlocal phase, last_teacher
            previous = phase; phase = 'observe'; last_teacher = None
            error = None
            try:
                return original_observe(state, result)
            except Exception as exc:
                error = {'type': type(exc).__name__, 'message': str(exc)}
                raise
            finally:
                after = capture(result['tokens'], result['logits'], 'greedy-after-observe')
                row = {'error': error, 'after': after[1]['path']}
                if current_run is not None:
                    row['greedy_before_after'] = evaluation.compare(after[0], current_run[0], {'atol': 0, 'rtol': 0})
                if current_run is not None and last_teacher is not None:
                    row['teacher_vs_greedy_before'] = evaluation.compare(last_teacher[0], current_run[0], quality)
                    row['greedy_path'] = current_run[1]['path']; row['teacher_path'] = last_teacher[1]['path']
                relations.append(row); phase = previous
        adapter._infer = infer; adapter.run = run; adapter.observe = observe
        return adapter

    def record_route(binding, call_id, route):
        routes[(phase, call_id, route)] += 1
        return original_route(binding, call_id, route)

    evaluation.load_adapter = load_adapter; BindingSession.record_route = record_route
    error = None
    try:
        return evaluation.worker_entry(context)
    except Exception as exc:
        error = {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}
        raise
    finally:
        evaluation.load_adapter = original_load; BindingSession.record_route = original_route
        write_json(output / 'relation-observations.json', {'observations': records, 'relations': relations,
            'candidate_routes': [{'phase': p, 'call': c, 'route': r, 'count': n} for (p, c, r), n in sorted(routes.items())],
            'worker_error': error, 'source_reference_sha256': sha(params['diagnostic_source_reference']),
            'observer_entry_sha256': sha(__file__),
            'scope': 'Diagnostic CPU snapshots between full invocations and phase-tagged candidate route callbacks. Candidate/source bytes unchanged; runtime observer changes timing/host work/lifetimes. No formal acceptance, no proof from labels or missing routes alone.'})


def main(args):
    os.umask(0o077); args.output.mkdir(parents=True, exist_ok=True)
    assert not (args.output / 'protocol.json').exists()
    terminal = read(args.parent / 'followup-completed.json'); assert terminal['phase'] == 'completed'
    assert terminal['results']['search']['bundles'] == []
    run = args.parent / 'qwen-original'; origin = Path(read(args.parent / 'protocol.json')['parent'])
    reference = read(origin / 'protocol.json')['independent_source_reference']
    assert sha(reference['path']) == reference['sha256']
    baseline = list((run / 'jobs').glob('*evaluation*pair-0-parent/run/request.json')); assert len(baseline) == 1
    requests = {'B': baseline[0]}
    for label, trial in [('C1', 'trial-d5c20fcae5'), ('C2', 'trial-e74f5ba6cf')]:
        result = read(run / 'trials' / trial / 'result.json')
        assert len(result['parameter_search']['observations']) == 1
        ident = result['parameter_search']['observations'][0]['bundle']['id']
        requests[label] = run / 'jobs' / (ident + '-screen') / 'run/request.json'
    sequence = [('B', True), ('C1', False), ('C1', True), ('C2', False), ('C2', True)]
    initial = {str(p): sha(p) for p in requests.values()}
    source_files = {}
    for path in requests.values():
        req = read(path); started = read(path.parent / 'rank-0-started.json')
        for name, digest in started['revision']['files'].items():
            source = Path(req['cwd']) / name; assert sha(source) == digest
            source_files[str(source)] = digest
    write_json(args.output / 'protocol.json', {'at': now(), 'core': framework_source(), 'parent': str(args.parent),
        'sequence': [{'arm': arm, 'observer': observer} for arm, observer in sequence],
        'source_requests': initial, 'source_files': source_files, 'source_reference': reference,
        'disk': disk_observation(Path('/root/autodl-tmp')),
        'scope': 'Fixed five endpoints, once each. B observed validates fixture; C1/C2 unobserved original evaluate then observed. Only observer entry/extra diagnostic reference and output directory differ. Original workload, samples, controls, candidates/parameters and tolerances retained. Failures expected and preserved; no retry until pass, agent search, native trace or acceptance.'})
    results = {}
    for index, (arm, observed) in enumerate(sequence):
        name = f'{index}-{arm}-' + ('observed' if observed else 'unobserved')
        write_json(args.output / 'state.json', {'phase': 'running', 'stage': name, 'at': now()})
        request = ExecutionRequest.model_validate(read(requests[arm]))
        values = {'output': args.output / name, 'wait_for_devices': True}
        if observed:
            values.update(entry=Path(__file__).resolve(), function='observed_worker',
                parameters={**request.parameters, 'diagnostic_source_reference': reference['path']})
        try:
            result = LocalExecutor().run(request.model_copy(update=values))
            write_json(args.output / (name + '-result.json'), result)
            sidecar = args.output / name / 'relation-observations.json'
            results[name] = {'status': 'completed' if result.get('worker_pids_reaped') and (not observed or sidecar.exists()) else 'execution_error',
                'worker_status': result['status'], 'failure': result.get('failure'),
                'result': str(args.output / (name + '-result.json')),
                'observer_summary': str(sidecar) if observed else None,
                'scope': 'Completed diagnostic attempt is not quality or performance acceptance; actual worker errors retained.'}
        except Exception:
            results[name] = {'status': 'execution_error', 'traceback': traceback.format_exc()}
        write_json(args.output / 'followup-results.json', results)
        print(json.dumps({'stage': name, **results[name], 'at': now()}), flush=True)
    unchanged = all(sha(p) == digest for p, digest in {**initial, **source_files}.items())
    unchanged = unchanged and sha(reference['path']) == reference['sha256']
    terminal = {'phase': 'completed' if unchanged and all(r['status'] == 'completed' for r in results.values()) else 'completed_with_errors',
        'finished_at': now(), 'results': results, 'frozen_inputs_unchanged': unchanged,
        'disk': disk_observation(Path('/root/autodl-tmp'))}
    write_json(args.output / 'state.json', terminal); write_json(args.output / 'followup-completed.json', terminal)


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    for name in ['output', 'parent']: p.add_argument('--' + name, type=Path, required=True)
    main(p.parse_args())
