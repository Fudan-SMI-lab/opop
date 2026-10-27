from pathlib import Path
import os,json,torch
import torch.distributed as dist
rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
dist.init_process_group('nccl')
x=torch.tensor([rank+1.0],device='cuda');dist.all_reduce(x);torch.cuda.synchronize()
assert x.item()==3.0
print(json.dumps({'rank':rank,'sum':x.item(),'torch':torch.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(rank)}),flush=True)
dist.destroy_process_group()
