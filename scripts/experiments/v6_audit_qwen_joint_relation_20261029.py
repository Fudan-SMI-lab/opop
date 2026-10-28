"""Audit the fixed joint candidate's complete continuous output relations."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import subprocess
import torch
from kernel_optimizer.project.evaluation import compare
from kernel_optimizer.project.transactions import RevisionStore
from kernel_optimizer.store.run_store import RunStore
from kernel_optimizer.runtime.execution import write_json
from scripts.experiments.v6_audit_fsdp_20261028 import read, sha


def main(root):
    protocol=read(root/'protocol.json'); terminal=read(root/'followup-completed.json')
    assert terminal['phase']=='completed' and terminal['all_frozen_hashes_unchanged']
    assert all(sha(p)==h for p,h in protocol['frozen_identities'].items())
    assert sha(protocol['observer'])==protocol['observer_sha256']
    assert sha(protocol['source_reference']['path'])==protocol['source_reference']['sha256']
    source=torch.load(protocol['source_reference']['path'],map_location='cpu',weights_only=True)
    source={k:source[k]for k in ['tokens','logits']};quality={'atol':.02,'rtol':.02}
    report={'at':datetime.now(timezone.utc).isoformat(),'finished_at':terminal['finished_at'],'core':protocol['core'],
        'protocol_sha256':sha(root/'protocol.json'),'terminal_sha256':sha(root/'followup-completed.json'),
        'frozen_identities_verified':protocol['frozen_identities'],'endpoints':{},'raw_tensor_count':0}
    for index,(arm,observed) in enumerate(protocol['sequence']):
        name=f'{index}-{arm}-'+('observed'if observed else'unobserved')
        row=terminal['results'][name];result=read(row['result']);req=read(row['request']);run=Path(row['request']).parent
        assert sha(row['result'])==row['result_sha256'] and result['worker_pids_reaped']
        worker=read(run/'rank-0.json');started=read(run/'rank-0-started.json')
        original=Path(protocol['source_requests'][arm]);expected=read(original)
        expected.update(output=str(run),wait_for_devices=True)
        if observed:
            expected.update(entry=protocol['observer'],function='observed_worker',parameters={**expected['parameters'],'diagnostic_source_reference':protocol['source_reference']['path']})
        assert req==expected and started['revision']==read(original.parent/'rank-0-started.json')['revision']
        assert all(sha(Path(req['cwd'])/p)==h for p,h in started['revision']['files'].items())
        assert worker['framework_source']==protocol['core']
        assert result['entry_sha256']==sha(req['entry'])
        if arm=='C':
            assert len(req['parameters']['bundles'])==1 and req['parameters']['bundles'][0]['parameters']=={}
            assert sha(req['parameters']['bundles'][0]['file'])==protocol['candidate_sha256']
        value=result['ranks'][0].get('result')or{}
        record={'worker_status':worker['status'],'error':worker.get('error'),'error_type':worker.get('error_type'),
            'traceback':worker.get('traceback'),'request_sha256':sha(row['request']),'request_diff_exact':True,
            'entry_sha256':result['entry_sha256'],'revision':started['revision'],'core':worker['framework_source'],
            'checks':value.get('checks'),'samples':value.get('samples'),'coverage':value.get('coverage'),'metrics':value.get('metrics')}
        if arm=='B':
            assert worker['status']=='passed' and len(value['samples'])==5 and len(value['checks'])==2
            assert all(c['passed']and c['max_abs_error']==0 for c in value['checks']) and value['math_policy_unchanged']
            assert all(number==statistics.median(s[metric]for s in value['samples'])for metric,number in value['metrics'].items())
            for s in value['samples']:
                assert s['generated_tokens']==32 and s['decode_tokens']==31
                assert abs(s['end_to_end_ms']-s['ttft_ms']-s['decode_ms'])<1e-9
        else:
            assert worker['status']=='failed' and worker['error']=='teacher-forced replay token sequence differs from greedy run'
            assert not value.get('samples')
        if observed:
            side=read(row['observer_summary']);assert side['observer_entry_sha256']==protocol['observer_sha256']
            assert side['source_reference_sha256']==protocol['source_reference']['sha256']
            assert len(side['observations'])==6 and len(side['relations'])==2
            tensors=[];rows=[]
            for pos,item in enumerate(side['observations']):
                assert sha(item['path'])==item['sha256']
                raw=torch.load(item['path'],map_location='cpu',weights_only=True);assert set(raw)=={'tokens','logits'}
                assert raw['tokens'].shape==source['tokens'].shape and raw['logits'].shape==source['logits'].shape
                assert compare(raw,source,quality)==item['source_comparison']
                assert bool(torch.equal(raw['tokens'],source['tokens']))==item['exact_source_tokens']
                assert bool(torch.isfinite(raw['logits']).all())==item['finite_logits']
                delta=(raw['logits'].float()-source['logits'].float()).abs()
                assert delta.flatten(2).amax(-1).tolist()==item['max_abs_error_by_position']
                assert (delta>.02+.02*source['logits'].float().abs()).flatten(2).sum(-1).tolist()==item['outside_tolerance_by_position']
                wrong=torch.nonzero(raw['tokens']!=source['tokens']).tolist()
                extra={'token_mismatch_positions':wrong,'actual_tokens':raw['tokens'].tolist(),
                    'expected_tokens':source['tokens'].tolist(),'logits_max_abs_error_full':float(delta.max()),
                    'outside_tolerance_elements_full':int((delta>.02+.02*source['logits'].float().abs()).sum()),
                    'full_logits_comparison_separately':compare(raw['logits'],source['logits'],quality),
                    'exact_full_source':all(torch.equal(raw[k],source[k])for k in raw)}
                assert extra['exact_full_source']==(arm=='B'or pos!=4)
                rows.append({**item,**extra});tensors.append(raw)
            for step,relation in enumerate(side['relations']):
                greedy,teacher,after=tensors[step*3:step*3+3]
                assert compare(after,greedy,{'atol':0,'rtol':0})==relation['greedy_before_after']
                assert compare(teacher,greedy,quality)==relation['teacher_vs_greedy_before']
                assert all(torch.equal(after[k],greedy[k])for k in greedy)
                assert bool(relation['error'])==(arm=='C'and step==1)
            record['observed']={'summary_sha256':sha(row['observer_summary']),'observations':rows,
                'relations':side['relations'],'routes':side['candidate_routes'],'all_raw_values_recomputed':True}
            report['raw_tensor_count']+=len(tensors)
        report['endpoints'][name]=record
    parent=Path(protocol['parent']);previous=read(parent/'preparation-reuse.json')
    report['current_pointers']={p:RevisionStore(RunStore.open(Path(p))).current().id for p in [previous['source'],previous['branch']]}
    assert set(report['current_pointers'].values())=={previous['revision']}
    report['conclusion']={'fixed_baseline':'2 full checks/5 clean samples passed, all6 full snapshots exact source.',
        'candidate':'U and O failed original token relation before clean measurement. O first greedy/teacher/after exactsource; second greedy/after exactsource, second teacher differs starting firstdecode. All12 full raw outputs retained and independently recomputed.',
        'limits':'CPU snapshots perturb host synchronization and lifetimes. These observations do not identify a unique pool/cache/alias root cause or establish historical U per-call internals. No new quality/performance acceptance or production changes.',
        'next':'One normal feedback-driven trial from unchanged empty-bundle baseline, append only factual observed relation summary/index to original framework latest feedback. Existing selector/candidate can inspect original failed candidate; no manual implementation or causal answer.'}
    report['df']=subprocess.check_output(['df','-B1','/root/autodl-tmp'],text=True)
    write_json(root/'terminal-audit.json',report)
    bad=report['endpoints']['2-C-observed']['observed']['observations'][4]
    print(json.dumps({'path':str(root/'terminal-audit.json'),'sha256':sha(root/'terminal-audit.json'),
        'bytes':(root/'terminal-audit.json').stat().st_size,'raw_tensor_count':report['raw_tensor_count'],
        'second_teacher':{k:bad[k]for k in ['token_mismatch_positions','logits_max_abs_error_full','outside_tolerance_elements_full']},'df':report['df']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--root',type=Path,required=True);main(parser.parse_args().root)
