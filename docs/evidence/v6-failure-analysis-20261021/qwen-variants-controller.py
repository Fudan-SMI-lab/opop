import subprocess,time,json
from pathlib import Path
root=Path('/root/autodl-tmp/dev/v6-work/failure-analysis-20261021')
while not (root/'ddp-cpu/completed.json').exists(): time.sleep(2)
for case,trial,variant in [('qwen-ttft','trial-9a65eeeca9','ttft_eager_torch'),('qwen-ttft','trial-9a65eeeca9','ttft_eager_triton'),('qwen-throughput','trial-3c2ed50501','decode_no_graph')]:
 out=root/variant;out.mkdir(exist_ok=True)
 cmd=['/root/autodl-tmp/dev-venv/bin/python','-B','-u','/root/autodl-tmp/dev/v6-work/qwen_failure_probe.py',case,trial,str(out),'--variant',variant]
 with (out/'run.log').open('w') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
 (out/'exit.json').write_text(json.dumps({'returncode':r.returncode}))
