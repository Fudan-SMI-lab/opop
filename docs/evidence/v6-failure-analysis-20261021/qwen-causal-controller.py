import subprocess,time,json
from pathlib import Path
root=Path('/root/autodl-tmp/dev/v6-work/failure-analysis-20261021')
while not (root/'decode_no_graph/exit.json').exists():time.sleep(2)
for case,trial,variant in [('qwen-ttft','trial-af78dd3283','graph_preserve_prefill_mask'),('qwen-throughput','trial-3c2ed50501','decode_live_window_no_graph'),('qwen-throughput','trial-87dde5b43d','static_live_attention_window')]:
 out=root/variant;out.mkdir(exist_ok=True)
 cmd=['/root/autodl-tmp/dev-venv/bin/python','-B','-u','/root/autodl-tmp/dev/v6-work/qwen_failure_probe.py',case,trial,str(out),'--variant',variant]
 with (out/'run.log').open('w') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
 (out/'exit.json').write_text(json.dumps({'returncode':r.returncode}))
