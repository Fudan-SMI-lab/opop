from pathlib import Path
import os


def run(context):
    import torch
    output = Path(context['output'])
    action = context['parameters']['action']
    if action == 'extension':
        from torch.utils.cpp_extension import load_inline
        major, minor = torch.cuda.get_device_capability()
        os.environ['TORCH_CUDA_ARCH_LIST'] = f'{major}.{minor}'
        os.environ['MAX_JOBS'] = '2'
        build = output/'build'; build.mkdir()
        module = load_inline(name='box43_probe_add_one',
            cpp_sources='torch::Tensor add_one(torch::Tensor x);',
            cuda_sources='''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
__global__ void add1(const float* x, float* y, int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) y[i]=x[i]+1.0f;
}
torch::Tensor add_one(torch::Tensor x) {
    auto y=torch::empty_like(x);
    add1<<<(x.numel()+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(x.data_ptr<float>(),y.data_ptr<float>(),x.numel());
    return y;
}
''', functions=['add_one'], with_cuda=True, build_directory=str(build), verbose=True)
        x=torch.arange(4096,device='cuda',dtype=torch.float32)
        y=module.add_one(x); torch.cuda.synchronize()
        return {'valid': torch.equal(y,x+1), 'elements': x.numel(), 'build':str(build),
                'loaded':module.__file__, 'architecture':[major,minor]}
    x=torch.ones(1024*1024,device='cuda')
    def compute():
        if context['world_size'] > 1:
            x.fill_(context['rank']+1)
            torch.distributed.all_reduce(x)
        else:
            x.add_(1)
        torch.cuda.synchronize()
    compute()
    if action == 'nsys':
        torch.cuda.cudart().cudaProfilerStart()
        try:
            with torch.cuda.nvtx.range('box43-preflight'):
                compute()
        finally:
            torch.cuda.cudart().cudaProfilerStop()
        return {'valid':True,'value':x[0].item()}
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as p:
        for _ in range(3): compute()
    trace=output/f'trace-rank-{context["rank"]}.json'
    p.export_chrome_trace(str(trace))
    from kernel_optimizer.project.communication import probe, transports
    manifest=probe(context).model_dump(mode='json')
    return {'valid':bool(torch.all(x==3)), 'value':x[0].item(), 'trace':str(trace),
            'capabilities':manifest, 'transport':transports(output.glob('nccl.*.log'))}
