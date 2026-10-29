"""Two real, isolated evidence interpreters; no new model/GPU experiment or search."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import traceback
from kernel_optimizer.config import load_config
from kernel_optimizer.project.agents import ProjectAgents
from kernel_optimizer.project.feedback import route
from kernel_optimizer.project.runtime_config import project_agent_config
from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.runtime.provenance import framework_source
from kernel_optimizer.store.run_store import RunStore
from kernel_optimizer.wiring import Runtime
from scripts.experiments.v6_audit_fsdp_20261028 import read, sha
from scripts.experiments.v6_ddp_semantics_20261027 import disk_observation


INSTRUCTIONS = """Read the completed diagnostic evidence for the supplied project failure.
Your task is to produce a compact, reliable factual feedback packet that another
project role could consume, not to optimize the candidate or invent a causal answer.
Start from inputs/evidence-index.json, task.json and original-failure.json. Inspect
actual worker results, diagnostic records and the instrument/source implementation.
Use selective reads or small CPU scripts when records are large. If an assertion
requires saved raw values, inspect those values when available; summaries/hashes
alone do not independently reproduce missing raw data. Do not copy reference files.

Write supplemental-evidence.json using inputs/report-schema.json. It will be treated
only as a proposed packet under the existing feedback supplemental_evidence field;
this session does not send it to a selector/candidate or adopt it in production.
Each fact needs exact artifact path, SHA256 and a field/line/invocation locator.
Keep directly supported observations separate from code hypotheses, and preserve
phase, rank, invocation, source/reference and candidate identities. State missing
evidence and instrumentation/measurement limitations. Original errors, returned
quality decisions, diagnostic execution and optimization acceptance are different.
Do not infer a unique cause or performance benefit from labels, successful execution,
run-only probes, counters, separate profile windows or missing events. No tolerance,
workload, result, candidate, measurement or production-code changes are permitted.

Focus on what the added evidence establishes beyond the original failure, what it
contradicts or leaves unresolved, and what downstream reasoning must not assume.
Prefer a small set of useful facts and evidence indexes to copying full records or
all tensors into the packet. There is no API usage cap; fees are observed only.
Do not prescribe a hand-written optimized implementation. Optional hypotheses must
be conditional and list what evidence is missing; they are not repair instructions.

Use only the supplied inputs, their indexed original source/worker/reference files,
the completed observational artifacts they index and relevant framework/library
source. Do not inspect developer terminal-audit/public-audit files, later accepted
implementations, later diagnostic answers, another task's outputs or project-wide
experiment documentation. The external filesystem is accessible, not hermetic;
actual accesses and outputs will be audited. There is no developer factual summary
or repair answer in these inputs. Do not launch models, GPU work, collectives or new
experiments. CPU checks on existing data are allowed; retain their scripts, outputs
and failures under this sandbox. No raw downloads or external modifications.
Schema completion is not factual correctness; the packet will be independently
audited before any downstream use. If evidence is insufficient, say so explicitly.
"""


SCHEMA = {
    'type':'object','required':['diagnostic_evidence_report'],
    'diagnostic_evidence_report':{
        'schema_version':'diagnostic_evidence.v1','status':'observed | insufficient | error',
        'summary':'brief string, no acceptance claim',
        'facts':[{'statement':'directly supported observation','scope':'phase/rank/invocation/identity',
            'evidence':[{'path':'absolute indexed artifact path','sha256':'64 hexadecimal characters',
                'locator':'JSON field, source line, tensor key or event index'}]}],
        'limitations':['nonempty strings'],
        'unresolved':['questions the evidence does not settle'],
        'hypotheses':[{'statement':'conditional code hypothesis','missing_evidence':'what is not established',
            'evidence':[{'path':'absolute indexed artifact path','sha256':'64 hexadecimal characters','locator':'location'}]}]},
    'transport':'This object is the value of existing supplemental_evidence. Not a decision, quality override or new production schema.'}


def validate(directory):
    value=read(directory/'supplemental-evidence.json')
    if set(value)!={'diagnostic_evidence_report'}:raise ValueError('Only diagnostic_evidence_report is allowed at top level')
    data=value['diagnostic_evidence_report']
    if data.get('schema_version')!='diagnostic_evidence.v1' or data.get('status')not in ['observed','insufficient','error']:
        raise ValueError('schema_version/status required')
    if not isinstance(data.get('summary'),str):raise ValueError('summary must be a string')
    for name in ['facts','limitations','unresolved','hypotheses']:
        if not isinstance(data.get(name),list):raise ValueError(name+' must be an array')
    for name in ['limitations','unresolved']:
        if not all(isinstance(v,str)and v for v in data[name]):raise ValueError(name+' must contain nonempty strings')
    for row in data['facts']+data['hypotheses']:
        if not isinstance(row.get('statement'),str)or not isinstance(row.get('evidence'),list)or not row['evidence']:
            raise ValueError('Each fact/hypothesis needs statement and evidence citations')
        for evidence in row['evidence']:
            if not all(isinstance(evidence.get(k),str)and evidence[k]for k in ['path','sha256','locator']):raise ValueError('Citation requires path/sha256/locator')
            if not re.fullmatch('[a-fA-F0-9]{64}',evidence['sha256']):raise ValueError('Citation SHA256 format')
    if not all(isinstance(r.get('scope'),str)for r in data['facts']):raise ValueError('Fact scope required')
    if not all(isinstance(r.get('missing_evidence'),str)for r in data['hypotheses']):raise ValueError('Hypothesis missing_evidence required')


def now():return datetime.now(timezone.utc).isoformat()


def build_cases(work):
    cases={};expected={}
    descriptions={
        'qwen':('diagnostic-author-20261029','diagnostic-author-validation-20261029',['baseline-observed','candidate-observed'],
            work/'qwen-constraint-continuation-20261029/qwen-continuation/latest-feedback.json'),
        'fsdp':('fsdp-diagnostic-repair-20261029','fsdp-diagnostic-validation-20261029',['derived-baseline-observed','candidate-observed'],
            work/'fsdp-search-20261028/fsdp-search/trials/trial-86e14487d0/result.json')}
    for case,(author_name,validation_name,arms,feedback)in descriptions.items():
        author=work/author_name;validation=work/validation_name
        a=read(author/'followup-results.json')['diagnostic-author'];sandbox=Path(a['agent']['sandbox'])
        protocol=read(validation/'protocol.json');terminal=read(validation/'followup-completed.json')
        assert terminal['phase']=='completed'
        expected.update(protocol['frozen_identities'])
        files=[]
        for arm in arms:
            for p in sorted((validation/arm).rglob('*')):
                if not p.is_file():continue
                h=sha(p);expected[str(p)]=h
                files.append({'arm':arm,'path':str(p),'sha256':h,'bytes':p.stat().st_size})
        for key in ['probe','plan']:
            expected[a[key]]=a[key+'_sha256'];assert sha(a[key])==a[key+'_sha256']
        original_index=read(sandbox/'inputs/evidence-index.json')
        index={'original_evidence_index':original_index,
            'diagnostic_implementation':{k:a[k]for k in ['probe','probe_sha256','plan','plan_sha256']},
            'completed_observation_files':files,
            'scope':'Completed fixed observations from an agent-authored probe. Files are evidence, not author/developer interpretations or optimization acceptance. Read actual outcome.',
            'execution_definition':'Baseline and candidate each once; original requests retained. FSDP baseline explicitly derived by clearing bundles; original staged source manifest retained.' if case=='fsdp' else 'Baseline and candidate original requests each once; fixed observation, no new search.'}
        inputs={'task.json':(sandbox/'inputs/task.json').read_bytes(),
            'original-failure.json':(sandbox/'inputs/failure.json').read_bytes(),
            'evidence-index.json':json.dumps(index,indent=2),'report-schema.json':json.dumps(SCHEMA,indent=2)}
        for p in [feedback,validation/'protocol.json',validation/'followup-completed.json']:
            expected[str(p)]=sha(p)
        source_request=original_index.get('original_execution_request',original_index.get('original_execution_requests',{}).get('candidate'))
        request=read(source_request['path']);revision=read(Path(request['cwd'])/'.opop-revision.json')['revision']
        cases[case]={'inputs':inputs,'feedback':feedback,'current_revision':revision,'author':str(author),'validation':str(validation)}
    assert all(sha(p)==h for p,h in expected.items())
    return cases,expected


def main(args):
    os.umask(0o077);cases,expected=build_cases(args.work)
    sizes={case:{n:len(v.encode()if isinstance(v,str)else v)for n,v in data['inputs'].items()}for case,data in cases.items()}
    if args.preflight:
        print(json.dumps({'status':'passed','input_sizes':sizes,'frozen_files':len(expected),'core':framework_source()}));return
    args.output.mkdir(parents=True,exist_ok=True);assert not(args.output/'protocol.json').exists()
    write_json(args.output/'protocol.json',{'at':now(),'core':framework_source(),'input_sizes':sizes,'frozen_identities':expected,
        'cases':{k:{x:str(v[x])for x in ['author','validation','feedback','current_revision']}for k,v in cases.items()},
        'scope':'Two fresh real diagnostic_interpretation roles, same generic instructions, distinct tasks. No candidate/search/GPU/acceptance/adoption; not a single-variable A/B. Actual tool reads and factual correctness require later audit.',
        'disk':disk_observation(Path('/root/autodl-tmp'))})
    cfg=load_config(args.config,[]);cfg.usage.mode='observe_only';results={}
    for case,data in cases.items():
        write_json(args.output/'state.json',{'phase':'running','stage':case,'at':now()})
        try:
            root=args.output/case;store=RunStore.create(root.parent,root.name,{'kind':'isolated_diagnostic_interpretation'})
            with Runtime(project_agent_config(cfg),root)as runtime:
                directory,receipt=ProjectAgents(runtime,store).generate('diagnostic_interpretation',data['inputs'],INSTRUCTIONS,
                    ('supplemental-evidence.json',),validator=validate)
            output=directory/'supplemental-evidence.json';packet=read(output);original=read(data['feedback'])
            before=json.loads(route(json.dumps(original),current_revision=data['current_revision']))
            after=json.loads(route(json.dumps({**original,'supplemental_evidence':packet}),current_revision=data['current_revision']))
            assert after['supplemental_evidence']==packet
            assert {k:v for k,v in before.items()if k!='supplemental_evidence'}=={k:v for k,v in after.items()if k!='supplemental_evidence'}
            write_json(root/'transport-replay.json',{'feedback':str(data['feedback']),'feedback_sha256':sha(data['feedback']),
                'packet_sha256':sha(output),'supplemental_preserved':True,'other_routed_fields_equal':True,
                'scope':'CPU existing route on original feedback; no new decision, real selector input or downstream use. Not an actual candidate delivery test.'})
            results[case]={'status':'completed','agent':receipt,'packet':str(output),'packet_sha256':sha(output),
                'bytes':output.stat().st_size,'transport':str(root/'transport-replay.json'),
                'validation':'Schema and CPU route only; factual validity, citations and actual accesses still require audit.'}
        except Exception:results[case]={'status':'error','traceback':traceback.format_exc()}
        write_json(args.output/'followup-results.json',results)
    same=all(sha(p)==h for p,h in expected.items())
    terminal={'phase':'completed'if same and all(r['status']=='completed'for r in results.values())else'completed_with_errors',
        'finished_at':now(),'results':results,'frozen_files_unchanged':same,'disk':disk_observation(Path('/root/autodl-tmp'))}
    write_json(args.output/'state.json',terminal);write_json(args.output/'followup-completed.json',terminal)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for name in ['config','output','work']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--preflight',action='store_true');main(p.parse_args())
