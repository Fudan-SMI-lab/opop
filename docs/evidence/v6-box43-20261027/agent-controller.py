import json
import os
from pathlib import Path
import httpx

from kernel_optimizer.config import load_config
from kernel_optimizer.project.agents import ProjectAgents
from kernel_optimizer.project.runtime_config import project_agent_config
from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.runtime.provenance import framework_source
from kernel_optimizer.store.run_store import RunStore
from kernel_optimizer.wiring import Runtime

os.umask(0o077)
root=Path('/root/autodl-tmp/dev/box43-preflight-20261027/agent-webfetch')
cfg=project_agent_config(load_config(Path('/root/autodl-tmp/dev/v6-work/configs/box43-v6.yaml'),[]))
store=RunStore.create(root.parent,root.name,{'kind':'real-agent-tool-availability-check'})
url='https://docs.pytorch.org/docs/stable/generated/torch.cuda.can_device_access_peer.html'
with Runtime(cfg,log_dir=root) as runtime:
    directory,receipt=ProjectAgents(runtime,store).generate('environment-inspection',{},
        'Use the webfetch tool to read '+url+'. This checks that the project agent can read external API documentation. '
        'Write probe.json containing the URL, the API name, and a brief description of what it reports. '
        'Use webfetch itself, not bash/curl or prior knowledge. Do not perform GPU work.',('probe.json',),
        validator=lambda p: json.loads((p/'probe.json').read_text()))
    with httpx.Client(base_url=runtime.client.base_url,timeout=30) as client:
        response=client.get(f'/session/{receipt["session"]}/message',params={'directory':str(directory)})
        response.raise_for_status(); messages=response.json()
    tools=[]
    for message in messages:
        for part in message.get('parts',[]):
            if part.get('type')=='tool':
                state=part.get('state',{})
                tools.append({'tool':part.get('tool'),'status':state.get('status'),
                              'url':state.get('input',{}).get('url'),'error':state.get('error')})
    result={'passed':any(t['tool']=='webfetch' and t['status']=='completed' and t['url']==url for t in tools),
        'framework_source':framework_source(),'agent':receipt,'tools':tools,
        'output':json.loads((directory/'probe.json').read_text()),
        'permission':json.loads((directory/'opencode.json').read_text())['permission']}
    write_json(root/'public-receipt.json',result)
    print(json.dumps({'passed':result['passed'],'tools':tools,'usage':receipt['usage']}),flush=True)
