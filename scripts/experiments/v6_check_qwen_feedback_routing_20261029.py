"""Read-only replay of the experimentally misplaced feedback field; no live repair."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
from kernel_optimizer.models.project import GroupRegistry
from kernel_optimizer.project.feedback import route
from kernel_optimizer.project.generation_snapshot import load
from kernel_optimizer.project.observation_artifacts import feedback_artifacts
from kernel_optimizer.project.transactions import RevisionStore
from kernel_optimizer.store.run_store import RunStore
from scripts.experiments.v6_audit_fsdp_20261028 import read,sha


def main(root):
    protocol=read(root/'protocol.json');old=read(protocol['source_feedback']);aug=read(root/'diagnostic-feedback.json')
    key='fixed_continuous_full_output_diagnostic';fact=aug[key]
    revisions=RevisionStore(RunStore.open(root/'qwen-continuation'))
    parent=revisions.revision(protocol['parent_revision']);registry=revisions.load(parent.registry_ref,GroupRegistry)
    def routed(value):
        _,raw=feedback_artifacts(json.dumps(value))
        return route(raw,current_revision=parent.id,registry=registry)
    original=routed(old);actual=routed(aug)
    assert original==actual and json.loads(actual)['supplemental_evidence']is None
    corrected={**old,'supplemental_evidence':{key:fact}}
    future=json.loads(routed(corrected));assert future['supplemental_evidence'][key]==fact
    future['supplemental_evidence']=None;assert future==json.loads(original)
    observations=[]
    for path in(root/'qwen-continuation/generation-snapshots').glob('*.json'):
        m,p,files=load(path)
        if m['fresh_session']and m['role']=='selection':
            raw=files['inputs/previous-feedback.txt'];assert raw==actual.encode()
            assert key.encode()not in raw
            observations.append({'path':str(path),'sha256':sha(path),'session':m['session'],'role':m['role'],
                'input_bytes':len(raw),'input_sha256':hashlib.sha256(raw).hexdigest(),'added_fact_received':False,
                'full_artifact_routing_matches_actual_snapshot':True})
    assert observations
    report={'at':datetime.now(timezone.utc).isoformat(),'source_feedback_sha256':protocol['source_feedback_sha256'],
        'supplied_feedback_sha256':sha(root/'diagnostic-feedback.json'),'actual_initial_snapshots':observations,
        'issue':'Experimental controller appended an unsupported top-level field. Existing route forwards supplemental_evidence instead; actual initial selector received original failure/history, not added observations.',
        'real_cpu_replay':{'original_and_top_level_augmented_route_identical':True,
            'existing_supplemental_evidence_preserves_facts':True,'other_routed_fields_equal':True,
            'full_observation_artifacts_and_registry_route_matches_snapshot':True},
        'action':'Preserve running frozen trial. No in-flight input/prompt/core edits, restart or parallel repeat. Future correction should use existing supplemental_evidence if still needed after actual result.',
        'scope':'CPU route replay only. Corrected payload was not submitted to any role or GPU. Later external tool reads need independent audit; absence in initial inputs does not establish permanent lack of access.'}
    path=root/'routing-interface-audit.json';path.write_text(json.dumps(report,indent=2));print(json.dumps(report))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
