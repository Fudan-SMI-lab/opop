"""Recompute fresh baseline/control trace evidence without any execution rerun."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path
import statistics
import subprocess
from kernel_optimizer.config import load_config
from kernel_optimizer.project.communication import trace_window,normalize
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.runtime.provenance import framework_source
from scripts.experiments.v6_audit_fsdp_20261028 import read,sha


def main(root):
    p=read(root/'protocol.json');terminal=read(root/'followup-completed.json');assert terminal['phase']=='completed'
    assert framework_source()==p['core'] and all(sha(n)==h for n,h in p['frozen_identities'].items())
    config=load_config(root/'private-config.yaml',[]);s=ProjectSearch(config,root/'fsdp-current');old=ProjectSearch(config,Path(p['parent']))
    assert old.revisions.current().id==p['source_current'] and s.revisions.current().id==s.prepared['revision']==p['original_baseline']
    assert not s.revisions.current().bundles and s.prepared['reuse']['contract_adaptation'] is None
    before,after=old.request.portable(),s.request.portable();assert {k for k in before if before[k]!=after[k]}=={'devices'}
    folder=Path(s.prepared['controls_path']);req=read(folder/'request.json');receipt=read(folder/'receipt.json')
    assert req['parameters']['request']==after and req['parameters']['action']=='controls' and not req['parameters'].get('bundles')
    assert sha(req['parameters']['adapter'])==s.prepared['adapter_sha256']==p['adapter_sha256']
    assert sha(req['entry'])==receipt['entry_sha256'] and receipt['status']=='passed' and len(receipt['ranks'])==2
    manifest=read(Path(req['cwd'])/'.opop-revision.json');assert all(sha(Path(req['cwd'])/n)==h for n,h in manifest['files'].items())
    rows=[];traces=[];controls=receipt['ranks']
    for r in controls:
        rank=r['rank'];result=r['result'];started=read(folder/f'rank-{rank}-started.json')
        assert started['parameters_ref']['request_sha256']==sha(folder/'request.json')
        assert all(started['revision']['files'][n]==h for n,h in manifest['files'].items())
        assert r['framework_source']==p['core'] and r['binding']['torch_uuid']==p['new_devices'][rank]
        assert result['valid'] and len(result['samples'])==5
        c=result['controls'];assert len(c['reset'])==2 and c['identity']['passed'] and all(x['passed'] and x['max_abs_error']==0 for x in c['reset'])
        assert c['identity_covered'] and c['wrong_output_rejected'] and c['missing_coverage_rejected']
        for name,value in result['metrics'].items():
            values=result['metric_samples'].get(name) or [x[name] for x in result['samples']]
            assert abs(statistics.median(values)-value)<1e-7,name
        for name,rec in result['diagnostic_metrics'].items():
            assert rec['status']=='observed' and len(rec['values'])==len(rec['evidence'])==3
            for path,value in zip(rec['evidence'],rec['values'],strict=True):
                raw=read(path);windowed,window=trace_window(raw,'opop.communication_metric_workload');assert window
                calc=normalize(windowed,{'kernels':[]},rank);field=rec['specification']['source']['field'];assert abs(calc[field]-value)<1e-8
                kernels=[e for e in windowed['traceEvents'] if e.get('cat')=='kernel' and 'nccl' in e.get('name','').lower()]
                traces.append({'rank':rank,'metric':name,'path':path,'sha256':sha(path),'window':window,'value':value,
                    'communication_kernel_count':calc['communication_kernel_count'],'actual_kernel_names':dict(Counter(e['name'] for e in kernels)),
                    'scope':rec['scope'],'protocol':rec['protocol']})
        rows.append({'rank':rank,'worker_sha256':sha(folder/f'rank-{rank}.json'),'started_sha256':sha(folder/f'rank-{rank}-started.json'),
            'device':r['binding']['torch_uuid'],'controls':c,'samples':result['samples'],'metrics':result['metrics'],
            'metric_samples':result['metric_samples'],'reference_sha256':sha(folder/f'reference-rank-{rank}.pt'),
            'log_sha256':sha(folder/f'rank-{rank}.log'),'log_contains_traceback':'Traceback (most recent call last)' in (folder/f'rank-{rank}.log').read_text()})
    assert len(traces)==24
    aggregate={}
    for name in controls[0]['result']['metrics']:
        values=[max(v)for v in zip(*(r['result']['metric_samples'].get(name) or [x[name] for x in r['result']['samples']] for r in controls),strict=True)]
        aggregate[name]={'aligned_rank_max':values,'median':statistics.median(values)}
    source=read(root/'source-check.json');assert source['status']=='passed' and len(source['ranks'])==2
    for r in source['ranks']:
        assert sha(r['adapter_reference'])==r['adapter_reference_sha256'] and len(r['checks'])==2
        for c in r['checks']:
            assert sha(c['source'])==c['source_sha256'] and c['comparison']['passed'] and c['comparison']['max_abs_error']==0
            assert not c['full_logits_plus_one_rejected']['passed'] and not c['false_tied_observation_rejected']['passed']
    assert all(sha(n)==h for n,h in p['frozen_identities'].items())
    report={'at':datetime.now(timezone.utc).isoformat(),'finished_at':terminal['finished_at'],'core':p['core'],
        'frozen_files':len(p['frozen_identities']),'all_frozen_unchanged':True,'request_changed_fields':['devices'],
        'request_sha256':sha(s.output/'request.json'),'worker_request_sha256':sha(folder/'request.json'),
        'entry_sha256':sha(req['entry']),'adapter_sha256':p['adapter_sha256'],'source_manifest_sha256':sha(Path(req['cwd'])/'.opop-revision.json'),
        'source_current':old.revisions.current().id,'current':s.revisions.current().id,'bundles':[],
        'ranks':rows,'aggregates':aggregate,'traces':traces,'source_check_sha256':sha(root/'source-check.json'),'source_check':source,
        'constraints':terminal['results']['baseline']['constraints'],'counts':{'ranks':2,'clean_rank_samples':10,'reset_checks':4,'identity_checks':2,'source_full_comparisons':4,'raw_metric_traces':24},
        'scope':'Fresh current-device original baseline/controls plus full existing independent-source CPU comparison. Not candidate/search/performance acceptance. Developer-assisted source owner and mapping remain. No old timing reuse; different shards not replicas.',
        'limits':['Four metrics belong to distinct three-invocation sets per rank. No cross-metric subtraction or NCCL sum as wire/critical-path time.',
            'Kernel entry names establish observed collective algorithm/protocol names only; NODE/P2P-CNS capability is not actual transport evidence. Logs do not establish a unique physical transport.',
            'Baseline memory anchor is current actual max-rank data, not imported historical number even if equal. Old K3 current/accepted/rejected history remains unchanged.'],
        'startup_reader_error':'Read-only startup helper used rank-N.json[rank], but raw worker files omit rank; receipt supplies rank and filenames identify it. Started records DO contain rank. Initial error note incorrectly blamed started schema; both preserved, audit now uses receipt. First terminal auditor also assumed clean fields lived in metric_samples; actual clean fields are in samples and diagnostic metrics in metric_samples. Reader corrected without model/source/fixture rerun.',
        'df':subprocess.check_output(['df','-B1','--output=size,used,avail','/root/autodl-tmp'],text=True),
        'gpu_processes':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],text=True)}
    write_json(root/'terminal-audit.json',report);public=dict(report);public['terminal_audit_sha256']=sha(root/'terminal-audit.json')
    write_json(root/'public-terminal-audit.json',public)
    print(json.dumps({'audit_sha256':public['terminal_audit_sha256'],'public_sha256':sha(root/'public-terminal-audit.json'),
        'public_bytes':(root/'public-terminal-audit.json').stat().st_size,'step':aggregate['step_ms'],'constraints':report['constraints'],
        'trace_kernel_name_counts':[x['actual_kernel_names']for x in traces[:1]],'counts':report['counts']}))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
