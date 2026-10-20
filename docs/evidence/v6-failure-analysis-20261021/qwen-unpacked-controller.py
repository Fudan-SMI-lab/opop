import subprocess,json
from pathlib import Path
root=Path('/root/autodl-tmp/dev/v6-work/failure-analysis-20261021')
for variant in ['ttft_unpacked_torch','ttft_unpacked_triton']:
 out=root/variant;out.mkdir(exist_ok=True)
 cmd=['/root/autodl-tmp/dev-venv/bin/python','-B','-u','/root/autodl-tmp/dev/v6-work/qwen_failure_probe.py','qwen-ttft','trial-9a65eeeca9',str(out),'--variant',variant]
 with (out/'run.log').open('w') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
 (out/'exit.json').write_text(json.dumps({'returncode':r.returncode}))
