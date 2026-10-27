# 固定 DDP / TP 诊断证据

协议见[本批文档](../../v6-ddp-tp-diagnostics-20261028.md)。服务器根目录为 `/root/autodl-tmp/dev/v6-work/ddp-tp-diagnostics-20261028`。

`preflight.json` 记录真实服务器上的编译、依赖导入、源文件/参考路径和请求对比：C/N 唯一差异是现有 `per_collective_align` 参数，任务请求仅将诊断采样从5改为20。adapter SHA 与旧冻结版一致。这里只是启动前检查，不是质量或性能结果。

原始张量、trace、完整 receipts 和 agent 输入/输出全部保留服务器；仅将小型公开记录保存本目录。后续启动记录包含冻结提交、控制器 PID、普通完成监听与初始执行状态。

- `launch.json`：远端真实启动记录、冻结提交和核心源码 SHA。
- `startup.json`：首个 B 端点 valid、双卡真实 worker、启动期 df 和日志尾部。
- `wakeup-setup.json`：唯一新完成监听及启动前真实进程探查；不是新的消息投递测试。

上述记录仅证明启动与首端点状态。最终结果在服务器 `followup-results.json`、`followup-completed.json`、`ddp-fixed/result.json`、`tp-result.json`、`diagnostic-review/result.json`；须区分执行错误、质量失败、诊断发现和原有正式接受/拒绝。
