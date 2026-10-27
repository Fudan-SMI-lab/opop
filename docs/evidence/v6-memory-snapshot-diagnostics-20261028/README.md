# 显存统计快照实验

协议见[主文](../../v6-memory-snapshot-diagnostics-20261028.md)。服务器根目录`/root/autodl-tmp/dev/v6-work/memory-snapshot-diagnostics-20261028`。

`preflight.json`核对源AST与局部变更SHA，不等同GPU语义控制通过。各worker的`evaluation_source`与实验副本文件记录实际执行字节；生产核心SHA保持不变。原始端点/GC事件/角色快照留服务器，只保存小型公开记录。

无前缀的`launch/startup/wakeup-setup.json`属于首次目录布局错误的尝试；`validation-launch/validation-startup/validation-wakeup-setup.json`属于修正后的新目录。不要混为同一次GPU运行。`validation-startup.json`记录真实两种生命周期CUDA控制和8个DDP端点完成，`review-start.json`记录后续真实角色及16,725-byte输入快照。完整结果与正式判断须等待终态审计。
