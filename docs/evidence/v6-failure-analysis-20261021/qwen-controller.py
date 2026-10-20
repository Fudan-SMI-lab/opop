import subprocess,json
from pathlib import Path
root=Path('/root/autodl-tmp/dev/v6-work/failure-analysis-20261021')
for case,trial in [('qwen-ttft','trial-9a65eeeca9'),('qwen-ttft','trial-af78dd3283'),('qwen-throughput','trial-3c2ed50501'),('qwen-throughput','trial-87dde5b43d')]:
 out=root/trial;out.mkdir(exist_ok=True)
 cmd=['/root/autodl-tmp/dev-venv/bin/python','-B','-u','/root/autodl-tmp/dev/v6-work/qwen_failure_probe.py',case,trial,str(out)]
 with (out/'run.log').open('w') as log:
  r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
 (out/'exit.json').write_text(json.dumps({'returncode':r.returncode}))
