"""Two candidate replays with metadata-only access to completed agent diagnostics."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import traceback
from kernel_optimizer.config import load_config
from kernel_optimizer.project.agents import ProjectAgents
from kernel_optimizer.project.generated import validate_proposal
from kernel_optimizer.project.generation_snapshot import load, edit_one
from kernel_optimizer.project.runtime_config import project_agent_config
from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.runtime.provenance import framework_source
from kernel_optimizer.store.run_store import RunStore
from kernel_optimizer.wiring import Runtime
from scripts.experiments.v6_audit_fsdp_20261028 import read, sha
from scripts.experiments.v6_ddp_semantics_20261027 import disk_observation


def now():return datetime.now(timezone.utc).isoformat()
def digest(value):return hashlib.sha256(value).hexdigest()


def prepare(work):
    # These are original failure-input candidate generations, not successful outputs.
    locations={
        'qwen':('qwen-joint-feedback-20261029/qwen-continuation/generation-snapshots/generation-62e1890de41e.json',
            'trial-a254520daa','a86694e792df3baac1ea6d93e5de574fc43fa1818ebee4ce452d4f4557295178'),
        'fsdp':('fsdp-search-20261028/fsdp-search/generation-snapshots/generation-c519752a5f2d.json',
            'trial-86e14487d0','df3d62aed4bb5f4ca15bc68dbc05266034547ed55e13f6c2fbd71db673e14881')}
    observed=read(work/'diagnostic-interpretation-20261029/followup-results.json')
    preserved=read(work/'diagnostic-interpretation-20261029/protocol.json')['frozen_identities'].copy()
    cases={}
    for case,(rel,trial,code_sha)in locations.items():
        snapshot=work/rel;m,p,files=load(snapshot)
        assert m['fresh_session']and m['role']=='candidate'and m['framework_source']==framework_source()
        original=files['inputs/previous-feedback.txt'];f=json.loads(original)
        assert f['previous_trial']==trial and f['decision']['accepted']is False and f['supplemental_evidence']is None
        assert digest(files['inputs/previous-candidate/candidate.py'])==code_sha
        # Reuse metadata prepared before either interpretation; no report/audit supplied.
        index=Path(observed[case]['agent']['sandbox'])/'inputs/evidence-index.json'
        evidence={'diagnostic_artifacts':{
            'scope':'Completed fixed observational artifacts for the previous failed candidate, including a baseline control. Not an acceptance decision or a unique cause. Instrumentation can perturb execution. Read the index and relevant raw/implementation evidence as needed; missing saved values remain unobserved.',
            'index':{'path':str(index),'sha256':sha(index),'bytes':index.stat().st_size}}}
        marker=b'"supplemental_evidence": null';assert original.count(marker)==1
        replacement=b'"supplemental_evidence": '+json.dumps(evidence,ensure_ascii=False).encode()
        modified=original.replace(marker,replacement,1)
        revised=json.loads(modified);assert revised.pop('supplemental_evidence')==evidence
        assert revised=={k:v for k,v in f.items()if k!='supplemental_evidence'}
        edit={'path':'inputs/previous-feedback.txt','old':original.decode(),'new':modified.decode()}
        new_prompt,new_files=edit_one(p,files,edit)
        assert new_prompt==p and [k for k in files if files[k]!=new_files[k]]==['inputs/previous-feedback.txt']
        for path in [snapshot,index]:preserved[str(path)]=sha(path)
        cases[case]={'snapshot':snapshot,'saved':m,'prompt':p,'files':files,'changed_files':new_files,'edit':edit,
            'evidence':evidence,'previous_trial':trial,'previous_candidate_sha256':code_sha}
    assert all(sha(p)==h for p,h in preserved.items())
    return cases,preserved


def main(args):
    os.umask(0o077);cases,frozen=prepare(args.work)
    config=project_agent_config(load_config(args.config,[]));config.usage.mode='observe_only'
    definitions={}
    for case,d in cases.items():
        assert d['saved']['model']==config.agents.default_model and d['saved']['agent']==config.opencode.agent
        definitions[case]={'snapshot':str(d['snapshot']),'snapshot_sha256':sha(d['snapshot']),
            'input_count':len(d['files']),'original_feedback_bytes':len(d['files']['inputs/previous-feedback.txt']),
            'modified_feedback_bytes':len(d['changed_files']['inputs/previous-feedback.txt']),
            'added_bytes':len(d['changed_files']['inputs/previous-feedback.txt'])-len(d['files']['inputs/previous-feedback.txt']),
            'previous_trial':d['previous_trial'],'previous_candidate_sha256':d['previous_candidate_sha256'],
            'only_change':'inputs/previous-feedback.txt: supplemental_evidence null replaced by observational artifact index',
            'index':d['evidence']['diagnostic_artifacts']['index'],'prompt_sha256':digest(d['prompt'].encode()),
            'settings_sha256':digest(json.dumps(d['saved']['tool_model_settings'],sort_keys=True).encode())}
    if args.preflight:
        print(json.dumps({'status':'passed','core':framework_source(),'cases':definitions,'frozen_files':len(frozen)}));return
    args.output.mkdir(parents=True,exist_ok=True);assert not(args.output/'protocol.json').exists()
    write_json(args.output/'protocol.json',{'at':now(),'core':framework_source(),'cases':definitions,'frozen_identities':frozen,
        'scope':'One fresh candidate replay per case. Original failure/selection/source/prompt/model/tools preserved; only existing supplemental_evidence adds metadata index of completed agent-authored observations. No interpreter/verifier packet or developer factual/causal answer. Not an original-entry cohort, new selection/search, GPU validation or production adoption. Filesystem accessible; actual reads must be audited.',
        'disk':disk_observation(Path('/root/autodl-tmp'))})
    results={}
    for case,d in cases.items():
        write_json(args.output/'state.json',{'phase':'running','stage':case,'at':now()})
        try:
            store=RunStore.create(args.output,case,{'kind':'diagnostic-index-candidate-replay'})
            with Runtime(config,log_dir=store.run_dir)as runtime:
                directory,receipt=ProjectAgents(runtime,store).replay_snapshot(d['snapshot'],edit=d['edit'],validator=validate_proposal)
            actual=[]
            for p in(store.run_dir/'generation-snapshots').glob('*.json'):
                m,prompt,files=load(p)
                if not m['fresh_session']:continue
                assert prompt==d['prompt']and files==d['changed_files']
                assert m['tool_model_settings']==d['saved']['tool_model_settings']and m['framework_source']==d['saved']['framework_source']
                actual.append(str(p))
            assert len(actual)==1
            results[case]={'status':'completed','agent':receipt,'initial_snapshot':actual[0],
                'initial_inputs_verified':True,'prompt_model_tools_core_unchanged':True,
                'candidate_sha256':sha(directory/'candidate.py'),'proposal_sha256':sha(directory/'proposal.json'),
                'parameter_space':read(directory/'proposal.json').get('parameter_space'),
                'scope':'Generation/schema only; actual artifact reads, implementation, probes and later GPU quality remain unaudited. No pointer publication.'}
        except Exception:results[case]={'status':'error','traceback':traceback.format_exc()}
        write_json(args.output/'followup-results.json',results)
    same=all(sha(p)==h for p,h in frozen.items())
    terminal={'phase':'completed'if same and all(v['status']=='completed'for v in results.values())else'completed_with_errors',
        'finished_at':now(),'results':results,'frozen_files_unchanged':same,'disk':disk_observation(Path('/root/autodl-tmp'))}
    write_json(args.output/'state.json',terminal);write_json(args.output/'followup-completed.json',terminal)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for name in ['config','output','work']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--preflight',action='store_true');main(p.parse_args())
