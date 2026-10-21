from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib,json,traceback
from kernel_optimizer.config import load_config
from kernel_optimizer.project.agents import ProjectAgents
from kernel_optimizer.project.branches import fork_run
from kernel_optimizer.project.feedback import candidate_artifacts,route
from kernel_optimizer.project.generated import validate_proposal
from kernel_optimizer.project.generation_snapshot import load
from kernel_optimizer.project.observation_artifacts import feedback_artifacts
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.wiring import Runtime

old=Path('/root/autodl-tmp/dev/v6-work/evidence-followup-20261021')
root=Path('/root/autodl-tmp/dev/v6-work/targeted-isolation-20261021')
source=old/'fresh/qwen-ttft';trial='trial-recovered-19ccfe875f'
saved=json.loads((source/'trials'/trial/'generated.json').read_text())
ordinary=json.loads((source/'trials'/trial/'result.json').read_text())
snapshot=Path(saved['candidate_agent']['generation_snapshots'][0])
manifest,_,files=load(snapshot)
shadow_path=root/'qwen-norm-shadow/result.json';shadow=json.loads(shadow_path.read_text())
probe_path=root/'qwen-rounding-intervention/result.json';probe=json.loads(probe_path.read_text())
summary={name:{'exact_sites':sum(r['comparisons_to_parent'][name]['unequal']==0 for r in shadow['rows']),
               'unequal_elements':sum(r['comparisons_to_parent'][name]['unequal'] for r in shadow['rows']),
               'max_abs_error':max(r['comparisons_to_parent'][name]['max_abs_error'] for r in shadow['rows'])}
         for name in shadow['rows'][0]['comparisons_to_parent']}
evidence={'scope':'Developer-measured diagnostic on this exact rejected candidate; not an autonomous finding. No required repair strategy.',
 'shadow':{'source_sha256':hashlib.sha256(shadow_path.read_bytes()).hexdigest(),
           'method':'At each original first live gate, extra expressions evaluated on the same input; original returned values and gates unchanged. Original complete-model error reproduced exactly. Later norm inputs belong to the candidate trajectory, not the separate original reference trajectory.',
           'first_site':shadow['rows'][0],'sites':len(shadow['rows']),'all_site_summary':summary},
 'intervention':{'source_sha256':hashlib.sha256(probe_path.read_bytes()).hexdigest(),
           'change':'Only at gate and steady norm invocation: weighted native fused RMSNorm -> unweighted native fused BF16 output, then BF16 weight multiplication. Original per-site gate/tolerances/parent bundles unchanged.',
           'checks':[r['result']['checks'] for r in probe['run']['ranks']],
           'scope':'Developer counterfactual still fails full quality; no timing/optimization claim.'}}
write_json(root/'qwen-repair-ab-evidence.json',evidence)
protocol={'source':str(source),'trial':trial,'snapshot':str(snapshot),
 'scope':'Two fresh candidate sessions on the same failed implementation, ordinary feedback, original selected boundary and contract. Treatment only adds compact measured evidence. Outputs enter unchanged normal GPU screening/ABBA. Developer-assisted comparison; no production policy change.',
 'evidence_sha256':hashlib.sha256(json.dumps(evidence,sort_keys=True).encode()).hexdigest()}
def run(arm,index):
    out=root/'qwen-repair-ab'/arm
    parent=fork_run(source,out,revision_id=saved['bundle']['parent_revision'])
    config=load_config(old/'qwen-ttft-fresh-private.yaml',[]);config.opencode.port=16830+index
    config.usage.mode='observe_only';search=ProjectSearch(config,out)
    feedback=dict(ordinary)
    if arm=='treatment':feedback['supplemental_evidence']=evidence
    write_json(out/'experimental-feedback.json',feedback);write_json(out/'protocol.json',protocol)
    evidence_files,indexed=feedback_artifacts(json.dumps(feedback))
    inputs={name.removeprefix('inputs/'):raw for name,raw in files.items() if name.startswith('inputs/')
            and not name.startswith('inputs/previous-candidate/') and name!='inputs/previous-feedback.txt'}
    inputs.update(candidate_artifacts(json.dumps(ordinary),search.store));inputs.update(evidence_files)
    instructions=(Path('/root/autodl-tmp/dev/opop-v6-freezes/a6f2b6e88ec3/src/kernel_optimizer/project/candidate_prompt.md').read_text()+
        '\nRepair the actual rejected implementation in inputs/previous-candidate/ using the measured feedback. Preserve the selected boundary and frozen contract. Further GPU checks belong to the bound worker.')
    try:
        with Runtime(search.config,log_dir=out) as runtime:
            directory,receipt=ProjectAgents(runtime,search.store).generate('candidate',inputs,instructions,
              ('candidate.py','proposal.json'),route(indexed,current_revision=parent.id,selected_groups=saved['selection']['groups']),validator=validate_proposal)
        discovery=search.current_discovery();ident='repair-'+arm
        bundle,proposal=search.bundle_from_outputs(directory,saved['selection'],parent,discovery,ident)
        metadata={'trial':ident,'selection':saved['selection'],'proposal':proposal,'candidate_agent':receipt,'experiment':protocol,'arm':arm}
        write_json(out/'trials'/ident/'generated.json',{**metadata,'bundle':bundle.model_dump(mode='json')})
        result=search._evaluate_generated(bundle,metadata,discovery)
        row={'arm':arm,'status':'evaluated','output':str(out),'trial':ident,'decision':result['decision'],'measurement':result.get('measurement')}
    except Exception as e:row={'arm':arm,'status':'error','error':str(e),'traceback':traceback.format_exc()}
    print(json.dumps(row),flush=True)
    return row
rows=[]
with ThreadPoolExecutor(max_workers=2) as pool:
    tasks=[pool.submit(run,arm,i) for i,arm in enumerate(['control','treatment'])]
    for task in as_completed(tasks):
        rows.append(task.result());write_json(root/'qwen-repair-ab-result.json',{'protocol':protocol,'runs':rows})
write_json(root/'qwen-repair-ab-completed.json',{'completed':True,'runs':len(rows)})
