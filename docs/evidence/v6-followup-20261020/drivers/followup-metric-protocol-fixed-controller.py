from pathlib import Path
import subprocess,sys,json
from kernel_optimizer.config import load_config
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.runtime.provenance import framework_source
out=Path(__file__).parent;root=out.parent
cmd=[sys.executable,'-m','pytest','-q','tests/test_v6_timeline.py','tests/test_v6_functional_binding.py','tests/test_v6_project_quality.py','tests/test_v6_replacement_trace.py','tests/test_v6_project_measurement.py','tests/test_v6_candidate_diagnostics.py']
r=subprocess.run(cmd,capture_output=True,text=True);write_json(out/'tests.json',{'command':cmd,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'source':framework_source()});print('tests',r.returncode,flush=True)
if r.returncode:raise SystemExit(r.returncode)
s=ProjectSearch(load_config(root/'configs/box42-v6.yaml',[]),root/'p5-ddp-communication-a');parent=s.revisions.current().id
run=s.execute(s.revisions.current(),out.name+'-baseline','evaluate');write_json(out/'evaluation.json',run)
print('evaluation',run['status'],[(r.get('result',{}).get('valid'),r.get('result',{}).get('diagnostics',{}).get('protocol')) for r in run['ranks']],flush=True)
assert s.revisions.current().id==parent
