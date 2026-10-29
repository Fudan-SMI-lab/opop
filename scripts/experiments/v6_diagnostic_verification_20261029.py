"""One evidence-verifier per existing packet; no production or optimization action."""
import argparse
import json
import os
from pathlib import Path
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
from scripts.experiments.v6_diagnostic_interpretation_20261029 import build_cases, now, validate


INSTRUCTIONS = """Independently verify the proposed diagnostic evidence packet against the
supplied original project evidence. This is a verification task, not candidate
optimization, another diagnostic experiment, or automatic approval of a report.
The proposed packet is an untrusted interpretation: neither its presence nor its
citations establish factual correctness. No developer verdict or corrected answer
is provided. Inspect inputs/proposed-packet.json and the original task, failure,
evidence index and schema. Inspect actual indexed artifacts and relevant instrument,
framework/library/source code as needed; use selective CPU checks on saved data.

Write verification.json, covering every summary/fact/limitation/unresolved/hypothesis
entry in the proposed packet. For each, record supported, mixed, unsupported,
contradicted or unverified, with exact evidence paths, SHA256 and locators and a
brief explanation. For compound claims separate supported observations from any
unsupported inference. Check what cited code/fields actually measure, what ran and
what was merely proposed, scope and identity, and whether raw data supports the
assertion. A matching file hash is not entailment. Absence of a returned field is
not necessarily absence of execution. Distinguish a measured result, an inference,
a hypothesis and a general guarantee. Unavailable raw data must remain unavailable;
do not present a summary's hash as independent recomputation of unsaved values.

Also write supplemental-evidence.json using inputs/report-schema.json, retaining
only the small set of defensible observations useful for understanding the original
failure. Prefer a concise facts-only packet and precise limitations/unresolved gaps;
hypotheses may be empty. Do not copy all measurements or your full verification into
the downstream packet. Cite evidence rather than adding broad context. Preserve
the original quality/work/decisions; do not prescribe a candidate implementation.
If reliable observations are insufficient, report that explicitly. Neither output
is automatically sent to an optimization role or installed in production.

Only use the supplied original inputs/indexed source/worker/reference/observational
files, the proposed packet and its cited artifacts (including the previous role's
own CPU checks), and relevant framework/library source. Do not inspect developer
audits, project-wide experiment docs, later successful implementations, other task
outputs or private provider state. External filesystem access is not hermetic;
actual commands/outputs will be audited. No developer factual table or error list
is supplied. Do not rerun models, GPU work, collectives, diagnostic workers or the
original role. Do not change external artifacts or copy large reference files.
CPU checks on existing artifacts are allowed; retain scripts, outputs and failures
in this sandbox. No manual answer editing, retries until pass, performance or causal
claims without evidence. API usage is observed only, with no usage cap.
"""

REVIEW_SCHEMA={'schema_version':'diagnostic_verification.v1','overall':'usable | partial | insufficient',
    'claims':[{'id':'summary or facts[i], limitations[i], unresolved[i], hypotheses[i] (zero-based)',
        'verdict':'supported | mixed | unsupported | contradicted | unverified',
        'reason':'brief explanation separating observation and inference',
        'evidence':[{'path':'absolute evidence path','sha256':'file SHA256','locator':'field/line/step/rank'}]}],
    'limitations':['verification boundaries and missing evidence']}


def claim_ids(packet):
    r=packet['diagnostic_evidence_report']
    return ['summary']+[f'{s}[{i}]' for s in ['facts','limitations','unresolved','hypotheses']for i in range(len(r[s]))]


def validate_review(directory, original):
    validate(directory)
    value=read(directory/'verification.json')
    if value.get('schema_version')!='diagnostic_verification.v1' or value.get('overall')not in ['usable','partial','insufficient']:
        raise ValueError('verification schema_version/overall required')
    rows=value.get('claims',[])
    if sorted(r.get('id','')for r in rows)!=sorted(claim_ids(original)):
        raise ValueError('Cover each original packet entry exactly once using zero-based IDs')
    for r in rows:
        if r.get('verdict')not in ['supported','mixed','unsupported','contradicted','unverified']or not isinstance(r.get('reason'),str):
            raise ValueError('Claim verdict/reason required')
        if not isinstance(r.get('evidence'),list):raise ValueError('Evidence must be an array, empty for unavailable support')
    if not isinstance(value.get('limitations'),list):raise ValueError('verification limitations required')


def main(args):
    os.umask(0o077);cases,expected=build_cases(args.work)
    old=args.work/'diagnostic-interpretation-20261029';terminal=read(old/'followup-completed.json')
    assert terminal['phase']=='completed'
    for case,data in cases.items():
        result=terminal['results'][case];p=Path(result['packet']);assert sha(p)==result['packet_sha256']
        data['inputs']['proposed-packet.json']=p.read_bytes()
        data['inputs']['verification-schema.json']=json.dumps(REVIEW_SCHEMA,indent=2)
        data['packet']=read(p);data['prior_packet']=str(p);expected[str(p)]=sha(p)
        for r in data['packet']['diagnostic_evidence_report']['facts']+data['packet']['diagnostic_evidence_report']['hypotheses']:
            for e in r['evidence']:
                if e['path']not in expected:expected[e['path']]=sha(e['path'])
                assert expected[e['path']]==e['sha256']
    sizes={k:{n:len(v.encode()if isinstance(v,str)else v)for n,v in d['inputs'].items()}for k,d in cases.items()}
    if args.preflight:
        print(json.dumps({'status':'passed','input_sizes':sizes,'frozen_files':len(expected),'core':framework_source()}));return
    args.output.mkdir(parents=True,exist_ok=True);assert not(args.output/'protocol.json').exists()
    write_json(args.output/'protocol.json',{'at':now(),'core':framework_source(),'input_sizes':sizes,'frozen_identities':expected,
        'cases':{k:{'prior_packet':d['prior_packet'],'claim_ids':claim_ids(d['packet']),'feedback':str(d['feedback']),
            'current_revision':d['current_revision']}for k,d in cases.items()},
        'scope':'Two new independent evidence verification roles, same generic instructions, different cases. Exact original proposed packets plus original inputs, no developer findings. One role each, no GPU/search/adoption or automatic downstream use.',
        'disk':disk_observation(Path('/root/autodl-tmp'))})
    cfg=load_config(args.config,[]);cfg.usage.mode='observe_only';results={}
    for case,data in cases.items():
        write_json(args.output/'state.json',{'at':now(),'phase':'running','stage':case})
        try:
            root=args.output/case;store=RunStore.create(root.parent,root.name,{'kind':'isolated_diagnostic_verification'})
            with Runtime(project_agent_config(cfg),root)as runtime:
                directory,receipt=ProjectAgents(runtime,store).generate('diagnostic_verification',data['inputs'],INSTRUCTIONS,
                    ('verification.json','supplemental-evidence.json'),validator=lambda p:validate_review(p,data['packet']))
            packet=read(directory/'supplemental-evidence.json');feedback=read(data['feedback'])
            before=json.loads(route(json.dumps(feedback),current_revision=data['current_revision']))
            after=json.loads(route(json.dumps({**feedback,'supplemental_evidence':packet}),current_revision=data['current_revision']))
            assert after['supplemental_evidence']==packet
            assert {k:v for k,v in before.items()if k!='supplemental_evidence'}=={k:v for k,v in after.items()if k!='supplemental_evidence'}
            write_json(root/'transport-replay.json',{'supplemental_preserved':True,'other_routed_fields_equal':True,
                'feedback_sha256':sha(data['feedback']),'packet_sha256':sha(directory/'supplemental-evidence.json'),
                'scope':'CPU transport only; no selector/candidate consumption or optimization decision.'})
            results[case]={'status':'completed','agent':receipt,
                'files':{n:{'path':str(directory/n),'sha256':sha(directory/n),'bytes':(directory/n).stat().st_size}for n in ['verification.json','supplemental-evidence.json']},
                'scope':'Schema and transport passed, not factual correctness or adoption.'}
        except Exception:results[case]={'status':'error','traceback':traceback.format_exc()}
        write_json(args.output/'followup-results.json',results)
    same=all(sha(p)==h for p,h in expected.items())
    done={'phase':'completed'if same and all(r['status']=='completed'for r in results.values())else'completed_with_errors',
        'finished_at':now(),'results':results,'frozen_files_unchanged':same,'disk':disk_observation(Path('/root/autodl-tmp'))}
    write_json(args.output/'state.json',done);write_json(args.output/'followup-completed.json',done)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for name in ['config','output','work']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--preflight',action='store_true');main(p.parse_args())
