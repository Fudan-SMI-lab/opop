"""Audit exact fixed endpoints and original constraints; no execution or retiming."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import subprocess

from kernel_optimizer.config import load_config
from kernel_optimizer.project.constraints import constraints_for_runs
from kernel_optimizer.project.measurement import valid_run
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.runtime.provenance import framework_source
from scripts.experiments.v6_audit_fsdp_20261028 import read, sha
from scripts.experiments.v6_audit_qwen_cache_interface_candidate_20261030 import CODE


def main(root):
    protocol = read(root/'protocol.json'); done = read(root/'followup-completed.json')
    assert done['phase'] == 'completed' and protocol['core'] == framework_source()
    frozen = protocol['frozen_identities']; assert len(frozen) == 500
    assert all(sha(p) == h for p,h in frozen.items())
    source = Path(protocol['source']); branch = root/'qwen-fixed'
    config = load_config(source.parent/'private-config.yaml',[])
    search = ProjectSearch(config,branch); original = ProjectSearch(config,source)
    assert search.revisions.current().id == original.revisions.current().id == protocol['parent']
    assert not search.revisions.current().bundles and search.prepared == original.prepared
    for name in ['request.json','prepared.json','discovery.json']:
        assert (branch/name).read_bytes() == (source/name).read_bytes()
    assert search.request.continuous_steps == 2 and search.request.warmup == 2 and search.request.repeats == 5
    assert search.request.diagnostic_metrics == {}
    assert search.prepared['evaluation']['ranks'][0]['result']['metrics']['ttft_ms'] == 23.285755
    stage = read(root/'staged-identities.json'); records = {}
    template_path = root.parent/'qwen-device-refresh-fixed-20261030/qwen-fixed/jobs/B/run/request.json'
    template = read(template_path)
    for arm in ['B','C']:
        runpath = root/(arm+'-result.json'); run = read(runpath)
        folder = branch/'jobs'/arm/'run'; request = read(folder/'request.json')
        assert run == read(folder/'receipt.json') and valid_run(run)
        assert run['worker_pids_reaped'] and run['status'] == 'passed' and len(run['ranks']) == 1
        rank = run['ranks'][0]; result = rank['result']; started = read(folder/'rank-0-started.json')
        assert rank['status'] == 'passed' and rank['exit_code'] == 0
        assert rank['framework_source'] == protocol['core']
        assert rank['binding']['torch_uuid'] == protocol['devices'][0]
        assert started['binding']['torch_uuid'] == rank['binding']['torch_uuid']
        assert request['devices'] == protocol['devices'] and request['function'] == 'run'
        assert request['wait_for_devices'] and not request['launcher'] and not request['env']
        changed = [k for k in request if request[k] != template[k]]
        assert set(changed) == {'entry','cwd','output','parameters'}
        params = request['parameters']; assert params['request'] == search.request.portable()
        assert params['action'] == 'evaluate' and params['collector_options'] == {}
        assert params['reference_dir'] == search.prepared['controls_path'] == template['parameters']['reference_dir']
        assert {k:v for k,v in params.items() if k not in ['adapter','bundles']} == {k:v for k,v in template['parameters'].items() if k not in ['adapter','bundles']}
        assert sha(request['entry']) == run['entry_sha256'] == sha(template['entry'])
        assert sha(params['adapter']) == search.prepared['adapter_sha256'] == '498c90f3f6f91b993ac8a586ac3230ffe2e6901752165b26948142d978f7cff7'
        materialized = Path(request['cwd']); manifest = read(materialized/'.opop-revision.json')
        assert all(sha(materialized/k) == v for k,v in manifest['files'].items())
        assert manifest['revision'] == stage['baseline' if arm=='B' else 'candidate']
        assert manifest['files'] == stage['original_files' if arm=='B' else 'candidate_files']
        assert started['revision'] == manifest
        if arm == 'B':
            assert params['bundles'] == [] and result['coverage'] == {} and result['setup'] == []
        else:
            assert len(params['bundles']) == 1
            bundle = params['bundles'][0]; assert sha(bundle['file']) == CODE and bundle['parameters'] == {}
            assert bundle['expected_calls'] == ['323bcbe006b343648b346af9']
            coverage = result['coverage']['fixed-cache-interface']
            assert coverage['counts'] == {'323bcbe006b343648b346af9':62}
            assert coverage['fallbacks'] == {} and coverage['reference_invocations'] == {}
            routes = {x['route']:x['count'] for x in coverage['candidate_declared_routes']}
            assert routes == {'fast_graph_replay':60,'fast_graph_replay_absorb':2}
        assert result['math_policy_unchanged'] and result['valid']
        assert result['checks'] == [{'continuous_step':i,'passed':True,'compared_elements':4861988,'max_abs_error':0.0} for i in range(2)]
        samples = result['samples']; assert len(samples) == 5
        recomputed = {k:statistics.median([s[k] for s in samples]) for k in result['metrics']}
        assert recomputed == result['metrics']
        assert all(s['generated_tokens'] == 32 and s['decode_tokens'] == 31 for s in samples)
        assert result['metric_samples'] == {} and result['diagnostics'] == {}
        constraints = constraints_for_runs([run],search.request.objective,search.prepared['evaluation'])
        assert constraints == done['results'][arm]['constraints']
        assert len(constraints) == 1 and constraints[0]['limit'] == 24.45004275 and not constraints[0]['passed']
        assert not list(folder.glob('*.sqlite')) and not list(folder.glob('*.nsys-rep'))
        log = (folder/'rank-0.log').read_text(); assert 'Traceback' not in log
        records[arm] = {'result_sha256':sha(runpath),'request_sha256':sha(folder/'request.json'),
            'entry_sha256':sha(request['entry']),'adapter_sha256':sha(params['adapter']),
            'manifest_sha256':sha(materialized/'.opop-revision.json'),'revision':manifest['revision'],
            'worker_status':rank['status'],'valid':True,'checks':result['checks'],'samples':samples,
            'metrics_recomputed':recomputed,'constraints':constraints,'coverage':result['coverage'],'setup':result['setup'],
            'occupancy':rank['binding']['occupancy'],'request_top_level_changes_vs_original_B':changed,
            'worker_log_sha256':sha(folder/'rank-0.log'),'worker_log_scope':'Loading weights only; no traceback.',
            'artifacts':{p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in folder.iterdir() if p.is_file()}}
    differences = {k:records['C']['metrics_recomputed'][k]-records['B']['metrics_recomputed'][k]
        for k in ['framework.peak_allocated_bytes','framework.lifecycle_peak_allocated_bytes']}
    assert all(sha(p) == h for p,h in frozen.items())
    report = {'at':datetime.now(timezone.utc).isoformat(),'terminal_at':done['finished_at'],
        'terminal_reason':'batch_finished','core':framework_source(),'frozen_files':500,
        'all_frozen_unchanged':True,'source_and_branch_current':protocol['parent'],'candidate_sha256':CODE,
        'candidate_revision':stage['candidate'],'parameter_space':{},'endpoints':records,
        'clean_samples':10,'full_zero_error_checks':4,'original_ttft_ms':23.285755,'ttft_limit_ms':24.45004275,
        'memory_delta_bytes':differences,'source_validation':'Exact shared current-device controls/source identities reused; no new independent source comparison. Manual source mapping assistance remains.',
        'quality':'Both endpoints pass original full two-check comparison and coverage. Only tested workload/invocations; no arbitrary owner/concurrency claim.',
        'decision':'Both fixed TTFT constraints fail. No search/paired-noise/independent acceptance; no optimized pointer published. No retiming/reanchor/outlier removal.',
        'routes':'C2absorb+60fast_graph labels/reference counters0. Code can emit fast_graph after eager interior fallback; labels are not native proof. Verify catch can leave fast_enabled true; returned mapping/full quality does not establish every setup selfcheck passed. No setup exception sidecar or native trace in this batch.',
        'limits':'Full candidate raw tensors are not saved by ordinary evaluator, only full comparison summaries. No new source fixture, observer or native collector. B910.177790 and896.865075ms tails retained. Setup5596.321494ms outside warm score; lifecycle/runtime peaks increase, not memory gain.',
        'next':'Freeze this constraint-failing fixed outcome; do not repeat it or arrange acceptance under a changed anchor. Any later ordinary continuation must use actual quality/TTFT and path limitations, not unsupported cause or hand repair. No further GPU stage in this audit.',
        'audit_error_retained':'attempt1 assumed diagnostic_metrics empty list rather than defined empty dict; attempt2 excluded files from expected started revision, but worker stores full manifest. Both reader assumptions corrected against actual schemas; logs retained, stronger full-manifest equality checked. No candidate/model/worker rerun.',
        'df':subprocess.check_output(['df','-B1','--output=size,used,avail','/root/autodl-tmp'],text=True),
        'live_gpu_workers':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)}
    write_json(root/'terminal-audit.json',report)
    public = {k:v for k,v in report.items() if k != 'endpoints'}
    public['endpoints'] = {a:{k:d[k] for k in ['result_sha256','request_sha256','entry_sha256','adapter_sha256','manifest_sha256','revision','worker_status','valid','checks','metrics_recomputed','constraints','coverage','setup','worker_log_scope']} for a,d in records.items()}
    public['samples'] = {a:[{k:s[k] for k in ['ttft_ms','end_to_end_ms','decode_ms','generated_tokens']} for s in d['samples']] for a,d in records.items()}
    write_json(root/'public-terminal-audit.json',public)
    print(json.dumps({'status':'passed','audit_sha256':sha(root/'terminal-audit.json'),
        'public_sha256':sha(root/'public-terminal-audit.json'),'memory_delta_bytes':differences}))


if __name__ == '__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
