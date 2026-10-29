"""Bounded candidate prompt experiment addressing observed unread evidence indexes."""
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

PARAGRAPH = ('When prior feedback supplies an evidence index, open that index and inspect the relevant '
    'observation artifacts and their producing code before choosing a repair or asserting a cause. '
    'In the rationale, cite the inspected artifact and location, distinguish directly observed facts from '
    'hypotheses and instrumentation limits, and state any remaining evidence gap; an index pointer or '
    'a diagnostic status alone is not a quality result.')


def now(): return datetime.now(timezone.utc).isoformat()
def digest(value): return hashlib.sha256(value).hexdigest()


def prepare(work):
    prior = work/'diagnostic-index-candidate-20261029'
    terminal = read(prior/'followup-completed.json'); assert terminal['phase']=='completed'
    assert read(prior/'public-audit.json')['terminal_audit_sha256'] == 'f72977c7ee1ec5d796fa4b6791c39d93da437678a5bc6062a1091b1a12531052'
    frozen = read(prior/'protocol.json')['frozen_identities'].copy()
    cases = {}
    for case, result in read(prior/'followup-results.json').items():
        snapshot = Path(result['initial_snapshot']); meta,prompt,files = load(snapshot)
        assert meta['fresh_session'] and meta['role']=='candidate' and meta['framework_source']==framework_source()
        assert PARAGRAPH not in prompt
        index = json.loads(files['inputs/previous-feedback.txt'])['supplemental_evidence']['diagnostic_artifacts']['index']
        assert sha(index['path']) == index['sha256'] and Path(index['path']).stat().st_size == index['bytes']
        edit = {'path':'prompt','old':prompt,'new':prompt+'\n\n'+PARAGRAPH}
        changed,newfiles = edit_one(prompt,files,edit); assert newfiles == files
        frozen[str(snapshot)] = sha(snapshot)
        cases[case]={'snapshot':snapshot,'meta':meta,'files':files,'prompt':prompt,'changed_prompt':changed,'edit':edit,'index':index}
    assert all(sha(f)==h for f,h in frozen.items())
    return cases,frozen


def main(args):
    os.umask(0o077); cases,frozen=prepare(args.work)
    config=project_agent_config(load_config(args.config,[]));config.usage.mode='observe_only'
    definitions={}
    for case,d in cases.items():
        assert d['meta']['model']==config.agents.default_model and d['meta']['agent']==config.opencode.agent
        definitions[case]={'snapshot':str(d['snapshot']),'snapshot_sha256':sha(d['snapshot']),
            'input_count':len(d['files']),'input_bytes':sum(len(v)for v in d['files'].values()),
            'input_identities':{k:{'sha256':digest(v),'bytes':len(v)}for k,v in d['files'].items()},
            'prompt_before_sha256':digest(d['prompt'].encode()),'prompt_after_sha256':digest(d['changed_prompt'].encode()),
            'added_prompt_bytes':len(d['changed_prompt'].encode())-len(d['prompt'].encode()),
            'settings_sha256':digest(json.dumps(d['meta']['tool_model_settings'],sort_keys=True).encode()),'index':d['index']}
    if args.preflight:
        print(json.dumps({'status':'passed','core':framework_source(),'cases':definitions,'frozen_files':len(frozen)}));return
    args.output.mkdir(parents=True,exist_ok=True);assert not(args.output/'protocol.json').exists()
    write_json(args.output/'protocol.json',{'at':now(),'core':framework_source(),'cases':definitions,
        'added_paragraph':PARAGRAPH,'frozen_identities':frozen,'disk':disk_observation(Path('/root/autodl-tmp')),
        'scope':'One replay per original index-supplied initial snapshot; only one identical generic prompt paragraph added. No input file, selection, model, tool, core or feedback change. No fixed-GPU results, reviewer packet, manual facts/cause/repair supplied. Existing candidate tools may probe; audit complete accesses. No controller GPU/search/adoption. No causal superiority from one sample.'})
    results={}
    for case,d in cases.items():
        write_json(args.output/'state.json',{'phase':'running','stage':case,'at':now()})
        try:
            store=RunStore.create(args.output,case,{'kind':'diagnostic-index-read-prompt-experiment'})
            with Runtime(config,log_dir=store.run_dir)as runtime:
                directory,receipt=ProjectAgents(runtime,store).replay_snapshot(d['snapshot'],edit=d['edit'],validator=validate_proposal)
            actual=[]
            for f in(store.run_dir/'generation-snapshots').glob('*.json'):
                m,prompt,files=load(f)
                if not m['fresh_session']:continue
                assert files==d['files'] and prompt==d['changed_prompt']
                assert m['tool_model_settings']==d['meta']['tool_model_settings'] and m['framework_source']==d['meta']['framework_source']
                actual.append(str(f))
            assert len(actual)==1
            results[case]={'status':'completed','agent':receipt,'initial_snapshot':actual[0],
                'all_inputs_unchanged':True,'only_prompt_paragraph_changed':True,
                'candidate_sha256':sha(directory/'candidate.py'),'proposal_sha256':sha(directory/'proposal.json'),
                'parameter_space':read(directory/'proposal.json').get('parameter_space'),
                'scope':'Generation/schema only; evidence reading/correctness/implementation/probes require terminal audit. No GPU quality or acceptance inference.'}
        except Exception:results[case]={'status':'error','traceback':traceback.format_exc()}
        write_json(args.output/'followup-results.json',results)
    same=all(sha(f)==h for f,h in frozen.items())
    end={'phase':'completed'if same and all(v['status']=='completed'for v in results.values())else'completed_with_errors',
        'finished_at':now(),'results':results,'frozen_files_unchanged':same,'disk':disk_observation(Path('/root/autodl-tmp'))}
    write_json(args.output/'state.json',end);write_json(args.output/'followup-completed.json',end)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for name in ['config','output','work']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--preflight',action='store_true');main(p.parse_args())
