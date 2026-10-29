"""Audit bounded index-reading replays and preserve all recorded tool histories."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from kernel_optimizer.project.generation_snapshot import load
from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.runtime.provenance import framework_source
from scripts.experiments.v6_audit_fsdp_20261028 import read, sha
from scripts.experiments.v6_diagnostic_index_read_20261029 import prepare


def digest(b): return hashlib.sha256(b).hexdigest()


def audit(root):
    cases,frozen=prepare(root.parent);done=read(root/'followup-completed.json');protocol=read(root/'protocol.json')
    assert done['phase']=='completed' and done['frozen_files_unchanged']
    assert frozen==protocol['frozen_identities'] and protocol['core']==framework_source()
    assert all(sha(p)==h for p,h in frozen.items())
    rows={};a=root/'audit'
    for case,data in cases.items():
        result=done['results'][case];receipt=result['agent'];sandbox=Path(receipt['sandbox'])
        meta,prompt,files=load(Path(result['initial_snapshot']))
        assert files==data['files'] and prompt==data['changed_prompt']
        assert meta['tool_model_settings']==data['meta']['tool_model_settings'] and meta['framework_source']==framework_source()
        assert all((sandbox/n).read_bytes()==v for n,v in files.items())
        assert sha(sandbox/'candidate.py')==result['candidate_sha256'] and sha(sandbox/'proposal.json')==result['proposal_sha256']
        tools=read(a/(case+'-tools.json'));parts=read(a/(case+'-session-parts.json'))
        assert tools==[p for p in parts if p['data'].get('type')=='tool']
        commands=[];errors=[];text_mentions=[];index_reads=[];observations=[];drafts=[]
        for i,part in enumerate(tools):
            d=part['data'];s=d['state'];inp=s.get('input',{});out=str(s.get('output',''))
            cmd=inp.get('command','');path=inp.get('filePath','')
            if d['tool']=='bash':
                saved=a/f'{case}-command-{i:03d}.txt';assert saved.read_text()==cmd
                commands.append({'index':i,'part':part['id'],'command':cmd,'output_sha256':digest(out.encode()),'output_bytes':len(out.encode())})
            if s.get('status')=='error' or s.get('metadata',{}).get('exit')not in [None,0]:
                errors.append({'index':i,'part':part['id'],'exit':s.get('metadata',{}).get('exit')})
            if 'Traceback (most recent call last)'in out or 'AssertionError'in out:
                text_mentions.append({'index':i,'part':part['id'],'exit':s.get('metadata',{}).get('exit'),
                    'scope':'Text only, may be source text or quoted historical failure; specific new errors reviewed separately.'})
            if 'evidence-index.json'in cmd or path.endswith('evidence-index.json'):
                index_reads.append({'index':i,'part':part['id'],'output_sha256':digest(out.encode()),'output_bytes':len(out.encode())})
            if any(x in cmd+path for x in ['diagnostic-author-validation-20261029','fsdp-diagnostic-validation-20261029']):
                observations.append({'index':i,'part':part['id'],'output_sha256':digest(out.encode())})
            if d['tool']=='write' and path.endswith('candidate.py'):
                f=a/f'{case}-generated-before-edits-{i}.py';f.write_text(inp['content']);drafts.append({'path':str(f),'sha256':sha(f)})
            if d['tool']=='edit' and path.endswith('candidate.py'):
                f=a/f'{case}-candidate-edit-{i}.json';write_json(f,inp);drafts.append({'path':str(f),'sha256':sha(f)})
        assert index_reads and observations
        write_json(a/(case+'-command-index.json'),commands)
        rows[case]={'session':meta['session'],'initial_snapshot':result['initial_snapshot'],'input_count':len(files),
            'input_files_unchanged':True,'only_prompt_added_bytes':len(prompt.encode())-len(data['prompt'].encode()),
            'prompt_sha256':digest(prompt.encode()),'settings_sha256':digest(json.dumps(meta['tool_model_settings'],sort_keys=True).encode()),
            'candidate_sha256':result['candidate_sha256'],'candidate_bytes':(sandbox/'candidate.py').stat().st_size,
            'proposal_sha256':result['proposal_sha256'],'parameter_space':result['parameter_space'],
            'contract_repairs':receipt['contract_repairs'],'usage':receipt['usage'],'tools':len(tools),
            'tool_counts':dict(Counter(t['data']['tool']for t in tools)),'index_reads':index_reads,'observation_access':observations,
            'nonzero_tool_exit_or_state':errors,'exception_text_mentions':text_mentions,
            'history_sha256':sha(a/(case+'-session-parts.json')),'retained_generated_before_and_edits':drafts}
    qt=read(a/'qwen-tools.json');ft=read(a/'fsdp-tools.json')
    assert len(qt)==52 and len(ft)==75
    assert 'evidence-index.json'in qt[16]['data']['state']['input']['command']
    assert 'diagnostic-index-rank-0.json'in qt[18]['data']['state']['input']['command']
    assert 'evidence-index.json'in ft[38]['data']['state']['input']['command']
    assert 'candidate-observed/diagnostic-rank-0.json'in ft[45]['data']['state']['input']['command']
    assert 'dist.init_process_group("gloo", rank=0, world_size=1)'in ft[72]['data']['state']['input']['command']
    assert '    reset()'in ft[72]['data']['state']['input']['command']
    review={
      'qwen':{
        'read_scope':'Opened index, candidate diagnostic JSON, baseline summary metrics, original evaluator/binding/source, installed DynamicCache code. No saved full tensor load, diagnostic.py/plan read or independent recalculation of full-output comparisons was recorded.',
        'useful_observations':'Summary correctly distinguishes both greedy outputs/frozen controls, first teacher exact and second teacher divergent (12 tokens, max31.125, first decode). Independent-source summary itself covers only first greedy. Probe summaries/limitations are not authoritative proofs.',
        'errors_and_limits':['Tool19 assigns route counters after completed-infer ordinal events and transiently attributes the following greedy routes to teacher ordinal1; tool20 then prints full sequence. Missing teacher labels/timing do not independently prove native execution.',
          'Unique mutable-cache cause and exclusion of graph/pool/kernel mechanisms are not established. No per-call cache identity/length record or unique-factor control exists. Stride mismatch is code-supported inference, not per-call native trace.',
          'Proposal treats original run-only candidate_diagnostics as healthy path evidence too strongly; it never observes/compares quality.',
          'Instrumentation timings are diagnostic, not a clean TTFT feasibility bound. Build self-checks do not guarantee all runtime quality or arbitrary concurrent ownership.',
          'Tool30 first missing transformers.cache_messaging import failed although compound shell exit0; tool37 invalid import syntax then missing class layers attribute failed. Tool38 compares an object attribute to itself after mutation (False), not a retained old view identity.'],
        'code':'Fresh prefill cache/layer objects only; decode returns prebuilt mirrors[step], not fresh cache on every return. Shared master tensors remain mutable. First-layer data_ptr plus all-layer shapes guide refresh; arbitrary mixed layer aliases/dtypes not fully certified. 31 per-length decode graphs plus separate prefill pool, capture-scoped global is_tracing patch, rolled-prefill and non-greedy checks. Setup direct replacement/parent chains do not exercise framework dispatch or the exact two run/observe transitions. Final K/V comparison is on first greedy/eager chain; no all-chain proof. Exceptions trigger fallback, numerical runtime drift is not automatically checked.',
        'probe_scope':'CPU DynamicCache identity/view tests plus two-layer prompt8/four-step guard and refresh prints. No real CUDA model quality or timing. Import print claiming no CUDA init was not an explicit is_initialized measurement.'},
      'fsdp':{
        'read_scope':'Opened index and original-failure input, rank0 original result, rank0 derived-baseline and candidate diagnostic summaries. Did not inspect rank1 diagnostics, full path arrays, diagnostic.py/plan or actual raw output tensors (the latter were not saved).',
        'errors_and_limits':['Unique shard-GEMM cause, transport repair success and general bitwise determinism are not established by this observed comparison. First failed comparison is not first execution divergence.',
          'Proposal diagnostic path has dev-v6-work typo; actual accessed path was dev/v6-work. Actual fixed diagnostic covers K6, not every original parameter.',
          'Removing one kernel does not prove its duration fully exposed on critical path. Separate profile periods cannot be subtracted or turned into wire cost.',
          'K2 unconditional peak bound and K>4 impossibility are unsupported allocation-lifetime generalizations; original measured gate still governs.',
          'Tool16/52 list.get errors, nonexistent installed framework path, wrong library grep targets, mock clamp assertion, UnboundLocalError on rollback, and two incomplete rollback asserts retained. Agent fixed its candidate rollback, final11 mock checks passed.'],
        'code':'Exact parent-forward delegation with DEFAULT_KEEP_UNSHARDED_BLOCKS=2. Trailing block public policy setter, private post_forward_mesh_info validation; records previous group values before setter and restores on setup exception. Success policy persists for worker lifetime. Rollback suppresses restoration exceptions, not universal restoration proof. No sharded-head arithmetic replacement. Choices0/1/2/3/4; fixed defaultK2 only justified for next execution, no tuning.',
        'probe_scope':'Tool72 actually creates Gloo world_size1 and a four-block hidden32 CPU FSDP model. It copies install logic inline rather than importing exact candidate; two updates each followed by reset and fresh random inputs, not original continuous two-update quality. No independent reference. Project agent config hides CUDA devices; successful CPU-parameter/default-mesh run is consistent with CPU execution. No original dual-rank model/GPU worker was scheduled or seen in tool history.'}}
    access={'scope':'Complete recorded command/path history and generated edits/probes reviewed; source/summary outputs checked within recorded access boundaries, filesystem not hermetic.',
        'later_answers':'No recorded read of interpreter/verifier output, developer terminal/public factual audits or later accepted Qwen/FSDP implementation. FSDP lists diagnostic directory names, which can expose audit filenames but is not a read of their answers.',
        'producing_code':'Both read original model/framework/library code but neither read Agent diagnostic.py producing new observations. The new paragraph was only partially followed.',
        'session_preservation':'All message/part records, exact bash commands, first generated candidate and edits remain server-side. FSDP deletes own __pycache__; inline probes had no full persistent tensor evidence. No hidden manual candidate edits.'}
    code_review={'fixed_execution_justified':True,'qwen_parameters':{},'fsdp_parameters':{'keep_unsharded_blocks':2},
        'decision':'Index reading improved in both single replays, factual discipline incomplete. Close this bounded prompt experiment without production adoption or another role retry/reviewer. Validate exact new code once per task with fixed B/C after source/parameter review; no automatic search or formal acceptance.'}
    result={'at':datetime.now(timezone.utc).isoformat(),'core':framework_source(),'finished_at':done['finished_at'],
        'frozen_files_verified':len(frozen),'frozen_identities':frozen,'roles':rows,'review':review,'access_review':access,'code_review':code_review,
        'local_reboot':'Remote batch continued and finished05:37:35UTC; local watcher last checked05:22:28, ended at reboot, terminal handled directly without duplicate queue/restart.',
        'df':subprocess.check_output(['df','-B1','/root/autodl-tmp'],text=True)}
    write_json(root/'terminal-audit.json',result)
    public={k:v for k,v in result.items()if k!='frozen_identities'}
    public['roles']={k:{f:v for f,v in row.items()if f not in ['parameter_space','retained_generated_before_and_edits']}for k,row in rows.items()}
    public['terminal_audit_sha256']=sha(root/'terminal-audit.json');write_json(root/'public-audit.json',public)
    print(json.dumps({'sha256':public['terminal_audit_sha256'],'roles':{k:{'tools':v['tools'],'index_reads':len(v['index_reads'])}for k,v in rows.items()}}))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True);audit(p.parse_args().root)
