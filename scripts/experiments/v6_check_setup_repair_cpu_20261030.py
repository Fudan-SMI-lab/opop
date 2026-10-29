"""Once-only retained CPU fixture reuse plus new retention boundary checks."""
import argparse
import difflib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import torch
from scripts.experiments.v6_audit_fsdp_20261028 import read, sha

NEW_SHA = 'bc4751ca9b77f9edaf5269c10dd053298e287d50452f51d113f1555a33265ff5'
OLD_SHA = '8ab9078e71390b240d8a471cc765084a58b9681d3170f6a8a4050a7221fb2ad4'


def main(root, output):
    output.mkdir(parents=True, exist_ok=False)
    probe = Path(read(root/'followup-completed.json')['results']['diagnostic-author']['probe'])
    assert sha(probe) == NEW_SHA
    original = root.parent/'qwen-setup-author-20261030/audit/independent-cpu-cap/fixture.py'
    assert sha(original) == '88c4be346306845accb5c802c8b886fd55107deb6a1ad4b74a40989f22d6e4f2'
    before = original.read_text(); source = before
    changes = [
        (OLD_SHA, NEW_SHA),
        ("window['assert_close_total'] == 129 and window['assert_close_overflow'] == 1",
         "window['assert_close_total'] == 129 and window['assert_close_overflow'] == 0"),
        ("assert all(x['outcome'] == {'passed': True} for x in window['assert_close_sites'])",
         "assert sum(x['outcome'] == {'passed': True} for x in window['assert_close_sites']) == 127\n"
         "    failure = window['assert_close_sites'][-1]\n"
         "    assert failure['seq'] == 129 and 'raised' in failure['outcome']\n"
         "    assert failure['comparison']['max_abs_error'] == 1 and failure['comparison']['violates_call_tolerance_count'] == 2\n"
         "    assert window['evicted_success_records'] == 1 and window['evicted_success_stubs'][0]['seq'] == 1"),
        ("'saved_successes': 128, 'overflow': 1", "'saved_successes': 127, 'overflow': 0, 'evicted_successes': 1"),
        ("'failure_operands_or_outcome_saved': False", "'failure_operands_or_outcome_saved': True"),
        ('Concrete evidence-retention defect: successful assertions exhaust the cap and a later failing comparison has no saved operands/outcome.',
         'Repaired late-failure retention: original comparison sequence retains the failing operands summary/outcome with one declared success eviction.'),
    ]
    for old,new in changes:
        assert source.count(old) == 1, old
        source = source.replace(old,new)
    adapted = output/'adapted-original-fixture.py'; adapted.write_text(source)
    (output/'fixture-adaptation.diff').write_text(''.join(difflib.unified_diff(before.splitlines(True),source.splitlines(True))))
    write = lambda name,value: (output/name).write_text(json.dumps(value,indent=2))
    write('fixture-reuse.json', {'original':str(original),'original_sha256':sha(original),
        'adapted_sha256':sha(adapted),'changes':'Only new probe hash and late-failure assertions/report expectations; original execution sequence unchanged. Generated diagnostic bytes unchanged.'})
    execution = subprocess.run([sys.executable,'-B',str(adapted),'--probe',str(probe),'--output',str(output/'original-sequence')],capture_output=True,text=True)
    (output/'original-sequence.log').write_text(execution.stdout+execution.stderr)
    assert execution.returncode == 0, execution.stderr
    base = read(output/'original-sequence/result.json')
    assert base['late_failure']['failure_operands_or_outcome_saved'] and not base['cuda_initialized']

    spec = importlib.util.spec_from_file_location('retention_boundary_probe',probe)
    diag = importlib.util.module_from_spec(spec); spec.loader.exec_module(diag)
    original_assert = torch.testing.assert_close
    recorder = diag._Recorder({'atol':.02,'rtol':.02},{},{})
    recorder.open_window('mixed-overflow',__file__); restore = recorder.patch_assert_close()
    caught = 0
    try:
        for _ in range(128): torch.testing.assert_close(torch.zeros(2),torch.zeros(2),atol=.012,rtol=.012)
        for _ in range(36):
            try: torch.testing.assert_close(torch.ones(2),torch.zeros(2),atol=.012,rtol=.012)
            except AssertionError: caught += 1
        for _ in range(4): torch.testing.assert_close(torch.zeros(2),torch.zeros(2),atol=.012,rtol=.012)
    finally:
        restore(); recorder.close_window()
    w=recorder.windows[0]
    assert caught == 36 and torch.testing.assert_close is original_assert
    assert (w['assert_close_total'],w['assert_close_passed'],w['assert_close_failed']) == (168,132,36)
    assert len(w['assert_close_sites']) == 128 and w['assert_close_retained_failures'] == 32
    assert w['assert_close_failure_overflow'] == 4 and w['assert_close_overflow'] == 4
    assert w['evicted_success_records'] == 32 and len(w['evicted_success_stubs']) == 32
    assert sum(w['assert_close_unretained_sites'].values()) == 8
    diag._write_evidence_json(output/'mixed-overflow.json',recorder.report({},{}))

    # Exercise actual original entry -> BindingSession.install -> caught failure.
    installer=output/'cpu-installer.py'
    installer.write_text("import torch\ndef build(context):\n"
        "    for _ in range(130): torch.testing.assert_close(torch.zeros(2),torch.zeros(2),atol=.012,rtol=.012)\n"
        "    try: torch.testing.assert_close(torch.ones(2),torch.zeros(2),atol=.012,rtol=.012)\n"
        "    except AssertionError as error: context['state']['caught'].append(error)\n"
        "    return {'s1': context['references']['s1']}\n")
    entry=output/'caught-entry.py'
    entry.write_text("from pathlib import Path\nfrom kernel_optimizer.project.binding import BindingSession\n"
        "class Module:\n    def forward(self,*a,**k): return 'parent'\n"
        "class Adapter:\n    def __init__(self,m): self.m=m\n    def models(self,state): return {'m':self.m}\n"
        "def run(context):\n    context['calls'].append('original')\n    module=Module()\n    old=module.forward\n"
        "    state={'caught':context['caught']}\n"
        "    specs=[{'id':'s1','kind':'module','target':'m','phase':'decode','ranks':[0],'signatures':[]}]\n"
        "    session=BindingSession(Adapter(module),state,specs,rank=0,world_size=1,bundle_id='cpu',backend='legacy')\n"
        "    try:\n        assert session.install(Path(context['installer']),{})==['s1']\n"
        "        assert module.forward()=='parent'\n    finally: session.close()\n"
        "    assert module.forward.__func__ is old.__func__\n    return context['original_result']\n")
    import kernel_optimizer.project.binding as binding
    original_binding=binding.BindingSession; answer={'valid':True,'scope':'CPU mock only'}
    ctx={'rank':0,'world_size':1,'device':'cpu','output':str(output/'caught-install'),
        'parameters':{'request':{'quality':{'atol':.02,'rtol':.02}},'bundles':[],
            'diagnostic_context':{'original_entry':str(entry),'original_function':'run','arm':'cpu-caught'}},
        'calls':[],'caught':[],'installer':str(installer),'original_result':answer}
    assert diag.run(ctx) is answer and ctx['calls']==['original'] and len(ctx['caught'])==1
    assert binding.BindingSession is original_binding and torch.testing.assert_close is original_assert
    saved=read(output/'caught-install/diagnostic-evidence-rank-0.json'); window=saved['install_windows'][0]
    assert len(saved['install_windows']) == 1 and window['assert_close_total']==131
    assert window['assert_close_failed']==1 and window['assert_close_sites'][-1]['seq']==131
    assert window['assert_close_sites'][-1]['comparison']['max_abs_error']==1

    # The new BaseException branch must remain an original interruption, not a failure record.
    sentinel=KeyboardInterrupt('synthetic original interruption'); calls=[]
    def interrupted(*a,**k): calls.append('original'); raise sentinel
    torch.testing.assert_close=interrupted
    rec=diag._Recorder({'atol':.02,'rtol':.02},{},{});rec.open_window('interruption',__file__)
    restore=rec.patch_assert_close()
    try:
        try: torch.testing.assert_close(torch.zeros(1),torch.zeros(1))
        except KeyboardInterrupt as error: assert error is sentinel
        else: raise AssertionError('interruption lost')
    finally:
        restore();rec.close_window();assert torch.testing.assert_close is interrupted
        torch.testing.assert_close=original_assert
    assert calls==['original'] and rec.windows[0]['assert_close_unfinished']==1 and not rec.windows[0]['assert_close_sites']
    diag._write_evidence_json(output/'interruption.json',rec.report({},{}))
    assert sha(probe)==NEW_SHA and not torch.cuda.is_initialized()
    report={'status':'passed','probe_sha256':sha(probe),'cuda_initialized':False,
        'original_fixture_reused_once':base,'mixed_overflow':{k:w[k] for k in ['assert_close_total','assert_close_passed','assert_close_failed','assert_close_overflow','assert_close_failure_overflow','assert_close_retained_failures','evicted_success_records']},
        'caught_install':{'entry_once':True,'same_return_object':True,'internal_failure_remains_caught':True,'late_failure_saved':True,'all_hooks_restored':True},
        'base_exception':{'same_object':True,'once':True,'unfinished':1,'records':0},
        'scope':'Independent CPU fixture mechanics only; no original candidate/model/GPU/collective run. Original fixture assertion adaptation explicit; no generated probe changes.'}
    write('result.json',report);print(json.dumps(report))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();main(args.root,args.output)
