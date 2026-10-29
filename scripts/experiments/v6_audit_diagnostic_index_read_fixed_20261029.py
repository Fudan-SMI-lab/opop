"""Read-only audit of exact newly generated fixed endpoints; no performance acceptance or model rerun."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import subprocess

from kernel_optimizer.project.communication import normalize, trace_window
from kernel_optimizer.project.measurement import valid_run
from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.runtime.provenance import framework_source
from scripts.experiments.v6_audit_fsdp_20261028 import read, sha


def audit(root):
    p = read(root / 'protocol.json'); end = read(root / 'followup-completed.json')
    staged = read(root / 'staged-identities.json')
    assert end['phase'] == 'completed' and end['frozen_files_unchanged']
    assert framework_source() == p['core']
    assert all(sha(f) == h for f, h in p['frozen_identities'].items())
    report = {'at': datetime.now(timezone.utc).isoformat(), 'terminal_sha256': sha(root/'followup-completed.json'),
              'core': p['core'], 'frozen_files_verified': len(p['frozen_identities']), 'endpoints': [],
              'scope': 'Fixed once-each B/C diagnostics, no formal acceptance, source rerun or agent generation.'}
    samples = checks = trace_count = 0
    for case, d in p['cases'].items():
        branch = root / (case+'-fixed'); identity = staged[case]
        original = read(Path(d['source'])/'request.json')
        assert read(branch/'request.json') == original
        assert read(branch/'project/current.json')['revision'] == d['parent']
        assert read(Path(d['source'])/'project/current.json')['revision'] == d['source_current']
        assert read(branch/'prepared.json')['controls_path'] == d['controls_path']
        for arm in ['B', 'C']:
            name = case+'-'+arm; path = root/(name+'-result.json'); run = read(path)
            out = branch/'jobs'/name/'run'; req = read(out/'request.json')
            files = identity['original_files' if arm == 'B' else 'candidate_files']
            assert valid_run(run) and run['worker_pids_reaped']
            assert run == read(out/'receipt.json')
            assert all(sha(Path(req['cwd'])/f) == h for f,h in files.items())
            assert sha(req['parameters']['adapter']) == d['adapter_sha256']
            assert req['entry'] == str(Path(req['cwd'])/'opop_evaluate.py')
            assert run['entry_sha256'] == sha(req['entry']) == files['opop_evaluate.py']
            assert req['parameters']['request'] == original
            assert req['parameters']['reference_dir'] == d['controls_path']
            assert req['parameters']['action'] == 'evaluate'
            assert (original['continuous_steps'],original['warmup'],original['repeats']) == (2,2,5)
            bundles = req['parameters']['bundles']
            assert len(bundles) == (arm == 'C')
            if bundles:
                assert sha(bundles[0]['file']) == d['candidate_sha256']
                assert bundles[0]['parameters'] == d['parameters']
            rows = []
            for rank in run['ranks']:
                ri = rank['rank']; v = rank['result']; worker = read(out/f'rank-{ri}.json')
                assert worker['result'] == v and worker['framework_source'] == p['core']
                assert worker['status'] == 'passed' and v['math_policy_unchanged']
                assert read(out/f'rank-{ri}-started.json')['revision']['files'] == files
                assert len(v['checks']) == 2 and len(v['samples']) == 5
                assert all(c['passed'] and c['max_abs_error'] == 0 for c in v['checks'])
                assert [c['continuous_step'] for c in v['checks']] == [0,1]
                expected_elements = 4861988 if case == 'qwen' else (361257090 if ri == 0 else 361255042)
                assert all(c['compared_elements'] == expected_elements for c in v['checks'])
                samples += len(v['samples']); checks += len(v['checks'])
                for metric in v['samples'][0]:
                    assert v['metrics'][metric] == statistics.median(s[metric] for s in v['samples'])
                if case == 'qwen':
                    for s in v['samples']:
                        assert s['generated_tokens'] == 32 and s['decode_tokens'] == 31
                        assert abs(s['ttft_ms']+s['decode_ms']-s['end_to_end_ms']) < 1e-9
                        assert abs(s['decode_tokens_per_second']-31000/s['decode_ms']) < 1e-9
                traces = []
                assert set(v.get('diagnostic_metrics',{})) == set(original.get('diagnostic_metrics',{}))
                for metric, diagnostic in v.get('diagnostic_metrics',{}).items():
                    assert len(diagnostic['values']) == len(diagnostic['evidence']) == 3
                    assert diagnostic['specification'] == original['diagnostic_metrics'][metric]
                    assert v['metric_samples'][metric] == diagnostic['values']
                    assert v['metrics'][metric] == statistics.median(diagnostic['values'])
                    field = diagnostic['specification']['source']['field']
                    for f,value in zip(diagnostic['evidence'],diagnostic['values'],strict=True):
                        trace,window = trace_window(read(f),'opop.communication_metric_workload'); assert window
                        recalc = normalize(trace,{'kernels':[]},ri)
                        assert abs(recalc[field]-value) < 1e-8
                        native = Counter(e['name'] for e in trace['traceEvents']
                                         if e.get('cat') == 'kernel' and 'nccl' in e.get('name','').lower())
                        assert sum(native.values()) == recalc['communication_kernel_count']
                        traces.append({'metric':metric,'path':f,'sha256':sha(f),'value':value,'nccl_kernels':dict(native)})
                        trace_count += 1
                rows.append({'rank':ri,'worker_sha256':sha(out/f'rank-{ri}.json'),
                             **{k:v[k] for k in ['checks','samples','coverage','setup','bundle_hashes','metrics']},'traces':traces})
            names = list(original.get('diagnostic_metrics',{}))
            primary = 'end_to_end_ms' if case == 'qwen' else 'step_ms'
            constraint = 'ttft_ms' if case == 'qwen' else 'peak_allocated_bytes'
            aligned = {}
            for metric in [primary,constraint,'framework.lifecycle_peak_allocated_bytes',*names]:
                series = [r['result']['metric_samples'][metric] if metric in names else
                          [s[metric] for s in r['result']['samples']] for r in run['ranks']]
                values = [max(x) for x in zip(*series,strict=True)]; median = statistics.median(values)
                aligned[metric] = {'samples':values,'median':median,'mad':statistics.median(abs(x-median)for x in values)}
            upper = 25.2536046 if case == 'qwen' else 2351207577.6
            assert aligned[constraint]['median'] <= upper
            report['endpoints'].append({'case':case,'arm':arm,'result_sha256':sha(path),'request_sha256':sha(out/'request.json'),
                'entry_sha256':run['entry_sha256'],'candidate_sha256':d['candidate_sha256'] if arm=='C' else None,
                'adapter_sha256':d['adapter_sha256'],'parameters':d['parameters'] if arm=='C' else {},
                'contract_ref':identity['contract_ref'],'ranks':rows,'max_rank_then_median':aligned,
                'constraint':{'metric':constraint,'upper':upper,'passed':True},'source_and_branch_current_unchanged':True})
            print(json.dumps({'audited':name,'primary':aligned[primary],'traces':sum(len(x['traces'])for x in rows)}),flush=True)
    assert (samples, checks, trace_count) == (30,12,48)
    source_check = root.parent/'fsdp-owner-repair-20261028/source-check.json'
    prior = read(root.parent/'fsdp-search-20261028/terminal-audit.json')['source_fidelity']
    assert sha(source_check) == prior['sha256']
    assert read(source_check)['status'] == 'passed'
    for reference in prior['unchanged_references']:
        assert sha(reference['path']) == reference['sha256']
    report['reused_fsdp_source_check'] = prior
    report['source_scope'] = 'Prior independent source mapping and six unchanged FSDP references reverified by hash, not a new numerical/model rerun. Qwen source/control references are among frozen identities.'
    assert all(sha(f) == h for f,h in p['frozen_identities'].items())
    report.update(clean_rank_samples=samples,full_zero_error_checks=checks,recomputed_traces=trace_count,
        limits=['No independent source rerun; prior reviewed mapping/owner repair boundaries retained.',
                'Qwen routes are telemetry; no native trace for this exact candidate in this batch.',
                'Different diagnostic metrics use separate perturbed invocations; no cross-metric subtraction, wire or critical-path attribution.',
                'Prior roles read indexes and partial summaries but not producing diagnostic code; GPU validity does not validate causal claims or stable prompt superiority.',
                'No independent AB/BA or new optimization acceptance. All historical outcomes unchanged.'],
        disk=subprocess.check_output(['df','-B1','/root/autodl-tmp'],text=True))
    write_json(root/'terminal-audit.json',report)
    public = {k:v for k,v in report.items() if k!='endpoints'}
    public['audit_sha256'] = sha(root/'terminal-audit.json')
    public['endpoints'] = [{k:v for k,v in x.items() if k!='ranks'} | {'ranks':[{
        'rank':r['rank'],'checks':r['checks'],'coverage':r['coverage'],'setup':r['setup'],
        'trace_count':len(r['traces']),'unique_nccl_counts':[dict(v)for v in {tuple(sorted(t['nccl_kernels'].items()))for t in r['traces']}]}
        for r in x['ranks']]} for x in report['endpoints']]
    write_json(root/'public-audit.json',public)
    print(json.dumps({'audit_sha256':public['audit_sha256'],'samples':samples,'checks':checks,'traces':trace_count}))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--root',type=Path,required=True)
    audit(parser.parse_args().root)
