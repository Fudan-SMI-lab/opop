"""What does the boundary fix COST, and does the fallback recover it?

An expansion delivers TWO separable things:
  1. a WIDENED RANGE -- new values the tuner could not reach before;
  2. a FRESH TUNING BUDGET -- another `trials_per_space` on the candidate.

`fix-boundary-direction-follows-the-winning-trial.md` improves the aim of (1). But a
winner-anchored flag can withdraw EVERY request, and an empty request list cancels the
expansion outright -- forfeiting (2) as a side effect nobody chose.

That is not hypothetical. Splitting the historical expansions by what the new rule does:

    CANCELLED (new=[])    the candidate loses the re-tune as well
    RE-AIMED  (different) the expansion still happens, just aimed differently
    UNCHANGED             identical request set

Only cancellations lose the budget, and they were expensive: 8 of 43, 6 improving,
including cand-0d0dcd49 9.14 -> 8.13 ms (11.1%, the run's best candidate) and
cand-913f73c9 24.00 -> 21.40 (10.8%). In both, the winning configuration used NO added
value -- it was already reachable pre-expansion, so the fresh budget is what found it, and
more than half of all improving expansions have that shape.

The fix is therefore a floor, not a veto: when the anchored pass asks for nothing, the
median's aim is used instead. Re-running this script after that change shows cancellations
falling 8 -> 1 while the re-aiming is preserved.

Reads events.jsonl only; no GPU.
"""
import json, sys, statistics
sys.path.insert(0,'src'); sys.stdout.reconfigure(encoding='utf-8')
from collections import defaultdict
from pathlib import Path
from kernel_optimizer.config import load_config
from kernel_optimizer.control.orchestrator import boundary_knobs_to_expand
from kernel_optimizer.models.core import ParameterSpace, TrialRecord
from kernel_optimizer.models.reports import TuningStats
from kernel_optimizer.tuning.stats import TuningStatsAnalyzer

cfg = load_config(Path('configs/experiments_l3.yaml'))
an = TuningStatsAnalyzer(cfg.device)
cancel, reaim, inert = [], [], []

for run in sorted(Path('runs').glob('run-l3-*')):
    f = run/'events.jsonl'
    if not f.exists(): continue
    spaces, trials, old, best = [], defaultdict(list), {}, {}
    for ln in f.read_text(encoding='utf-8').splitlines():
        if not ln.strip(): continue
        try: e = json.loads(ln)
        except json.JSONDecodeError: continue
        p = e.get('payload', {})
        if e['type']=='SPACE_PUBLISHED':
            try: spaces.append(ParameterSpace.model_validate(p['space']))
            except Exception: pass
        elif e['type']=='TRIAL_DONE':
            try:
                t=TrialRecord.model_validate(p['trial']); trials[t.space_id].append(t)
            except Exception: pass
        elif e['type']=='STATS_DONE': old[p['stats']['space_id']]=p['stats']
        elif e['type']=='TUNING_DONE': best[p['space_id']]=p['best_ms']
    bc = defaultdict(list)
    for s in spaces: bc[s.candidate_id].append(s)
    for cid, vs in bc.items():
        vs.sort(key=lambda s: s.version)
        for a, b in zip(vs, vs[1:]):
            if a.space_id not in old or not trials.get(a.space_id): continue
            if a.space_id not in best or b.space_id not in best: continue
            b0, b1 = best[a.space_id], best[b.space_id]
            if b0 is None or b1 is None: continue
            o = TuningStats.model_validate(old[a.space_id])
            n = an.analyze(a, trials[a.space_id])
            ol = [k['name'] for k in boundary_knobs_to_expand(o,0.8,a,min_effect_pct=cfg.budgets.min_improvement_pct)]
            nw = [k['name'] for k in boundary_knobs_to_expand(n,0.8,a,min_effect_pct=cfg.budgets.min_improvement_pct)]
            gain = (b0-b1)/b0*100 if b0 else 0.0
            rec = dict(run=run.name[8:], cand=cid, b0=b0, b1=b1, gain=gain, old=ol, new=nw)
            if ol and not nw: cancel.append(rec)
            elif ol != nw: reaim.append(rec)
            else: inert.append(rec)

def summarise(label, group):
    if not group:
        print(f'  {label}: none'); return
    g = [r['gain'] for r in group]
    imp = [r for r in group if r['gain'] > 1e-9]
    print(f'  {label}: n={len(group)}  improved={len(imp)}  '
          f'median gain {statistics.median(g):.2f}%  total gain forgone if cancelled '
          f'{sum(r["gain"] for r in imp):.1f} points')

print('expansions by what the NEW rule would do:')
summarise('CANCELLED (new=[])   ', cancel)
summarise('RE-AIMED (different) ', reaim)
summarise('UNCHANGED            ', inert)
print()
print('CANCELLED cases in detail -- these lose the re-tune entirely:')
for r in sorted(cancel, key=lambda r: -r['gain']):
    print(f"  {r['run']} {r['cand']}  {r['b0']:.2f} -> {r['b1']:.2f}  ({r['gain']:+.2f}%)  old={r['old']}")
