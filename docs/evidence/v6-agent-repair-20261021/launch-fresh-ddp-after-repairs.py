import pathlib,time,subprocess,sys
root=pathlib.Path('/root/autodl-tmp/dev/v6-work/agent-repair-20261021')
while not all((root/c/"completed.json").exists() for c in ['qwen-ttft', 'qwen-throughput', 'ddp-step', 'ddp-communication']):
 time.sleep(5)
raise SystemExit(subprocess.call(['/root/autodl-tmp/dev-venv/bin/python', '-B', '-u', '-m', 'kernel_optimizer.cli', '--config', '/root/autodl-tmp/dev/v6-work/agent-repair-20261021/ddp-communication-fresh-final-private.yaml', 'optimize-project', '--project', '/root/autodl-tmp/dev/opop-v6-freezes/88b965174bc2/examples/v6_validation', '--goal', "Reduce the source's diagnostic maximum-rank summed NCCL kernel duration for the accumulated training step, retaining its step-time constraint and full correctness checks.", '--command', 'torchrun --standalone --nproc_per_node=2 gpt2_ddp_training.py --output /tmp/original-ddp --profile', '--output', '/root/autodl-tmp/dev/v6-work/agent-repair-20261021/fresh-final/ddp-communication', '--rounds', '2']))
