# FSDP2 full training workload

Run `torchrun --standalone --nproc-per-node=2 train.py --output source-run --profile`.
This is seeded GPT-2 Medium training (24 layers, hidden 1024), with synthetic
token inputs, FP32, eager attention, TF32 off and dropout off. It tests execution
and distributed training fidelity, not pretrained language quality. No model
download or private assets are required.

One invocation performs a complete rank-local batch, global-mean gradient and
one SGD update. The fixed workload is in `workload.json`. Complete logits,
all parameter gradients and all post-update parameters must agree with the
independent unsharded serial global-batch reference over two continuous updates
at the declared tolerance; +1 corruption must fail. FSDP shards are different
parts of a parameter, not replicas that should be equal across ranks. Tied input
embedding/output weights must remain tied. The helper compares actual DTensor
shard layouts. Statistical repetitions reset the entire model outside timing.

The source uses official bottom-up FSDP2 defaults with no manually serialized
prefetch or per-layer barriers. The distributed strategy, prefetch scheduling,
optimizer implementation and call grouping are implementation choices, not
semantic requirements. Any alternative must preserve the entire work and quality.
The source task minimizes complete `step_ms` subject to allocated peak memory
at most 1.05 times the source baseline, retaining the sharded task's memory budget.
Rank aggregation is max per aligned invocation, then median across samples.
Reset, quality, statistics and profiling are outside clean timing; both CUDA
completion boundaries remain. Returned logits live through the memory read.

Communication profiles are separate diagnostic invocations (3); sums of NCCL
kernel durations are neither wire time nor exposed critical-path time. The
framework may choose available diagnostic collectors using actual environment
capabilities. `--api-fixture` is a smaller developer API check and does not
validate or change the full task's acceptance.
