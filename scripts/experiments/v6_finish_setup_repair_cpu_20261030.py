"""Complete only the CPU integration boundary missed by an eager-import fixture."""
import argparse
import importlib.util
import json
from pathlib import Path
import torch
import kernel_optimizer.project.binding as binding
from scripts.experiments.v6_audit_fsdp_20261028 import read, sha


def main(root):
    output=root/'audit/independent-cpu';assert not(output/'result.json').exists()
    base=read(output/'original-sequence/result.json');assert base['late_failure']['failure_operands_or_outcome_saved']
    mixed=read(output/'mixed-overflow.json');w=mixed['install_windows'][0]
    assert (w['assert_close_total'],w['assert_close_passed'],w['assert_close_failed'])==(168,132,36)
    assert len(w['assert_close_sites'])==128 and w['assert_close_retained_failures']==32
    assert w['assert_close_failure_overflow']==4 and w['assert_close_overflow']==4
    original_evidence=read(output/'caught-install/diagnostic-evidence-rank-0.json')
    assert original_evidence['install_windows']==[] and original_evidence['original_outcome']['original_returned']
    entry_source=(output/'caught-entry.py').read_text()
    import_line='from kernel_optimizer.project.binding import BindingSession\n'
    assert entry_source.count(import_line)==1
    corrected=entry_source.replace(import_line,'').replace('def run(context):\n','def run(context):\n    '+import_line)
    entry=output/'caught-entry-corrected.py';assert not entry.exists();entry.write_text(corrected)
    probe=Path(read(root/'followup-completed.json')['results']['diagnostic-author']['probe'])
    expected='bc4751ca9b77f9edaf5269c10dd053298e287d50452f51d113f1555a33265ff5';assert sha(probe)==expected
    spec=importlib.util.spec_from_file_location('corrected_cpu_integration_probe',probe)
    diag=importlib.util.module_from_spec(spec);spec.loader.exec_module(diag)
    original_binding=binding.BindingSession;original_assert=torch.testing.assert_close;answer={'valid':True,'scope':'CPU mock only'}
    ctx={'rank':0,'world_size':1,'device':'cpu','output':str(output/'caught-install-corrected'),
        'parameters':{'request':{'quality':{'atol':.02,'rtol':.02}},'bundles':[],
            'diagnostic_context':{'original_entry':str(entry),'original_function':'run','arm':'cpu-caught'}},
        'calls':[],'caught':[],'installer':str(output/'cpu-installer.py'),'original_result':answer}
    assert diag.run(ctx) is answer and ctx['calls']==['original'] and len(ctx['caught'])==1
    assert binding.BindingSession is original_binding and torch.testing.assert_close is original_assert
    saved=read(output/'caught-install-corrected/diagnostic-evidence-rank-0.json')
    assert len(saved['install_windows'])==1;window=saved['install_windows'][0]
    assert window['assert_close_total']==131 and window['assert_close_failed']==1
    assert window['assert_close_sites'][-1]['seq']==131 and window['assert_close_sites'][-1]['comparison']['max_abs_error']==1
    sentinel=KeyboardInterrupt('synthetic original interruption');calls=[]
    def interrupted(*a,**k):calls.append('original');raise sentinel
    torch.testing.assert_close=interrupted
    rec=diag._Recorder({'atol':.02,'rtol':.02},{},{});rec.open_window('interruption',__file__);restore=rec.patch_assert_close()
    try:
        try:torch.testing.assert_close(torch.zeros(1),torch.zeros(1))
        except KeyboardInterrupt as error:assert error is sentinel
        else:raise AssertionError('interruption lost')
    finally:
        restore();rec.close_window();assert torch.testing.assert_close is interrupted
        torch.testing.assert_close=original_assert
    assert calls==['original'] and rec.windows[0]['assert_close_unfinished']==1 and not rec.windows[0]['assert_close_sites']
    diag._write_evidence_json(output/'interruption.json',rec.report({},{}))
    assert sha(probe)==expected and not torch.cuda.is_initialized()
    report={'status':'passed','probe_sha256':sha(probe),'cuda_initialized':False,
        'original_fixture_reused_once':base,
        'mixed_overflow':{k:w[k] for k in ['assert_close_total','assert_close_passed','assert_close_failed','assert_close_overflow','assert_close_failure_overflow','assert_close_retained_failures','evicted_success_records']},
        'caught_install':{'entry_once':True,'same_return_object':True,'internal_failure_remains_caught':True,'late_failure_saved':True,'all_hooks_restored':True},
        'base_exception':{'same_object':True,'once':True,'unfinished':1,'records':0},
        'fixture_error':'First additional CPU entry imported BindingSession at module load, before run(context) installs class hook; zero windows caused audit IndexError. Original actual evaluator imports inside its evaluate function. Failed fixture/evidence/log preserved; only import placement corrected in a separate entry and new output. Original sequence and mixed-overflow tests were not rerun; no generated-probe/model/quality change.',
        'scope':'Independent CPU mechanics only, not candidate/model/GPU/collective quality. Original fixture expected-hash/retention assertions explicitly adapted, generated bytes unchanged.'}
    (output/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
