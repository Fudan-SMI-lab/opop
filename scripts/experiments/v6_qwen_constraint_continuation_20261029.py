"""Fixed native corroboration, then one normal trial with factual diagnostic feedback."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from kernel_optimizer.config import load_config
from kernel_optimizer.project.branches import fork_run
from kernel_optimizer.project.collectors import Nsys
from kernel_optimizer.project.feedback import candidate_artifacts
from kernel_optimizer.project.generation_snapshot import load
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.runtime.provenance import framework_source
from scripts.experiments.v6_audit_fsdp_20261028 import read,sha
from scripts.experiments.v6_ddp_semantics_20261027 import disk_observation


def now():return datetime.now(timezone.utc).isoformat()


def inputs(args,config):
    protocol=read(args.parent/'protocol.json');audit=read(args.parent/'terminal-audit.json')
    terminal=read(args.parent/'followup-completed.json');staged=read(args.parent/'staged-identities.json')
    assert terminal['phase']=='completed' and protocol['core']==framework_source()==audit['core']
    assert audit['full_tensor_relations']['all_12_full_observed_outputs_exact_source']
    assert all(sha(p)==h for p,h in protocol['frozen_identities'].items())
    source=ProjectSearch(config,args.parent/'fixed-validation');parent=source.revisions.current()
    assert parent.id==protocol['source_revision'] and not parent.bundles
    revision=source.revisions.revision(staged['candidate-continuation']['revision'])
    assert revision.parent_id==parent.id
    unobserved=terminal['results']['1-candidate-continuation-unobserved']
    run=read(unobserved['result']);assert sha(unobserved['result'])==unobserved['result_sha256']
    assert unobserved['valid_run'] and not unobserved['constraints'][0]['passed']
    role=read(Path(protocol['replay'])/'followup-completed.json')['results']['candidate-continuation']
    assert role['candidate_sha256']=='f12e2f50cfd5f960dddd51537556a73ff8b0793df0786bd78fe6dfcff849e013'
    assert sha(Path(role['agent']['sandbox'])/'candidate.py')==role['candidate_sha256']
    assert len(revision.bundles)==1
    return protocol,audit,terminal,source,parent,revision,unobserved,run,role


def main(args):
    os.umask(0o077);config=load_config(args.config,[]);config.usage.mode='observe_only'
    protocol,audit,terminal,source,parent,revision,row,run,role=inputs(args,config)
    if args.preflight:
        print(json.dumps({'status':'passed','core':framework_source(),'parent':parent.id,'fixed_revision':revision.id,
            'adapter_sha256':source.prepared['adapter_sha256'],'fixed_candidate_sha256':role['candidate_sha256'],
            'constraint':row['constraints'],'controls_shared_read_only':source.prepared['controls_path']}));return
    args.output.mkdir(parents=True,exist_ok=True);assert not(args.output/'protocol.json').exists()
    write_json(args.output/'protocol.json',{'at':now(),'core':framework_source(),'parent_batch':str(args.parent),
        'parent_revision':parent.id,'fixed_revision':revision.id,'fixed_candidate_sha256':role['candidate_sha256'],
        'parent_audit_sha256':sha(args.parent/'terminal-audit.json'),'frozen_identities':protocol['frozen_identities'],
        'sequence':['fork unchanged empty-bundle baseline with shared completed discovery/references',
                    'one separately perturbed fixed-candidate native profile','one normal autonomous selection/candidate trial with explicit diagnostic feedback'],
        'disk':disk_observation(Path('/root/autodl-tmp')),
        'scope':'Continuation of reviewed fresh-adapter task, not a fresh unaided original-entry cohort. Developer-supplied factual diagnostic feedback explicitly recorded. Failed candidate available as historical bytes only, not installed. No retiming/ABBA acceptance of that fixed candidate, no manual group/candidate answer, no production changes. One new ordinary trial; independent acceptance of any newly accepted version remains a later stage.'})
    results={}
    try:
        branch=args.output/'qwen-continuation';forked=fork_run(source.output,branch);search=ProjectSearch(config,branch)
        assert forked.id==parent.id and not forked.bundles
        assert (branch/'request.json').read_bytes()==(source.output/'request.json').read_bytes()
        assert (branch/'discovery.json').read_bytes()==(source.output/'discovery.json').read_bytes()
        write_json(args.output/'preparation-reuse.json',{'source':str(source.output),'branch':str(branch),'revision':parent.id,
            'request_sha256':sha(branch/'request.json'),'discovery_sha256':sha(branch/'discovery.json'),
            'controls_path':search.prepared['controls_path'],'adapter_sha256':search.prepared['adapter_sha256'],
            'installed_bundles':[],'scope':'No fresh discovery/control/model reference generation. Staged failed candidate artifacts copied as history, not active.'})
        write_json(args.output/'state.json',{'phase':'running','stage':'fixed-native','at':now()})
        native=Nsys().collect(search,search.revisions.revision(revision.id));write_json(args.output/'fixed-native.json',native)
        results['fixed-native']={'status':'completed'if native['status']=='observed'else'error','collector_status':native['status'],
            'path':str(args.output/'fixed-native.json'),'sha256':sha(args.output/'fixed-native.json'),
            'scope':'Single perturbed native path diagnostic. Not clean timing, a quality gate, or performance acceptance.'}
        if native['status']=='observed':
            bundle=search.revisions.bundle(revision.bundles[0]);_,_,files=load(Path(role['agent']['generation_snapshots'][0]))
            quality=run['ranks'][0]['result']
            feedback={'trial':'external-fixed-attention-diagnostic','bundle':bundle.model_dump(mode='json'),
                'selection':json.loads(files['inputs/selection.json']),'proposal':role['proposal'],'candidate':run,
                'decision':{'accepted':False,'reasons':['fixed_diagnostic_original_TTFT_constraint_failed'],
                    'scope':'Explicit diagnostic continuation marker; no formal paired-noise decision was executed for this candidate.'},
                'evaluation':{'quality':'pass','original_constraints':row['constraints'],'formal_acceptance_performed':False},
                'measurement':{'scope':'Single unobserved fixed endpoint, diagnostic only. No fresh paired baseline or acceptance.',
                    'candidate_metrics':quality['metrics'],'all_candidate_samples':quality['samples'],'constraints':row['constraints']},
                'resources':{'status':'No fresh per-group resources in this diagnostic feedback','groups':[]},
                'supplemental_evidence':{
                    'scope':'Developer-supplied measured facts and evidence indexes, not an optimization prescription. This is a continuation, not an input-controlled role A/B.',
                    'full_quality':'Original full32 tokens/logits and two continuous checks passed. Observed full greedy/teacher/after snapshots exactly equal independent source; original request/tolerances unchanged.',
                    'constraint':'Unobserved TTFT26.510787 exceeds fixed original25.2536046. Observed25.917823 also exceeds, but observer changes host/synchronization/lifetimes. Prefill unchanged is not evidence that measured TTFT is unchanged.',
                    'limits':'Do not subtract observer/profile costs or claim unique historical cause. Fixed diagnostic improvement is not acceptance. Prior failed and accepted records remain frozen.',
                    'fixed_audit':{'path':str(args.parent/'terminal-audit.json'),'sha256':sha(args.parent/'terminal-audit.json')},
                    'native':{'path':str(args.output/'fixed-native.json'),'sha256':sha(args.output/'fixed-native.json'),'status':native['status']}}}
            feedback_path=args.output/'diagnostic-feedback.json';write_json(feedback_path,feedback)
            raw=feedback_path.read_text();history=candidate_artifacts(raw,search.store)
            assert history['previous-candidate/candidate.py']==Path(role['agent']['sandbox'],'candidate.py').read_bytes()
            assert all(r['status']=='available'for r in json.loads(history['previous-candidate/manifest.json'])['files'])
            argv=[sys.executable,'-B','-m','kernel_optimizer.cli','--config',str(args.config),'optimize-project',
                '--request',str(branch/'request.json'),'--output',str(branch),'--resume','--rounds','1','--feedback',str(feedback_path)]
            write_json(args.output/'search-protocol.json',{'at':now(),'argv':argv,'feedback_sha256':sha(feedback_path),
                'historical_candidate_sha256':role['candidate_sha256'],'initial_bundles':[],
                'scope':'One ordinary trial with fresh real selector/candidate, original allowed scopes and default tuning. No old optimized bundle/import, manual selected group or candidate solution. API observed only. Search acceptance distinct from independent acceptance.'})
            write_json(args.output/'state.json',{'phase':'running','stage':'one-autonomous-trial','at':now()})
            with(args.output/'search-controller.log').open('wb')as log:process=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT)
            current=ProjectSearch(config,branch).revisions.current()
            results['search']={'status':'completed'if process.returncode==0 else'error','returncode':process.returncode,
                'current':current.id,'bundles':list(current.bundles),'independent_acceptance':'not_run_in_this_batch'}
        else:results['search']={'status':'not_run','reason':'Fixed native collection unavailable; preserve error for review.'}
        assert source.revisions.current().id==parent.id
        assert all(sha(p)==h for p,h in protocol['frozen_identities'].items())
    except Exception:results['controller']={'status':'error','traceback':traceback.format_exc()}
    terminal={'phase':'completed'if all(r['status']=='completed'for r in results.values())else'completed_with_errors',
        'finished_at':now(),'results':results,'disk':disk_observation(Path('/root/autodl-tmp'))}
    write_json(args.output/'followup-results.json',results);write_json(args.output/'state.json',terminal);write_json(args.output/'followup-completed.json',terminal)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for name in ['config','parent','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--preflight',action='store_true');main(p.parse_args())
