# P0：会话统计与 GPU 执行设施

box4-2 真实验收通过，结果见 [receipt.json](receipt.json)，相关检查为 [52 passed](tests.log)。驱动为 [validate_v6_p0.py](../../../scripts/validate_v6_p0.py)。原始私有目录：`/root/autodl-tmp/dev/v6-work/p0-implementation/real-a`。

新增 SQLite 用量账本按 session/message ID 更新，保留缺失状态并去重统计中间轮次、缓存和 reasoning 字段。开发配置启用 `usage.mode=observe_only`，不提供额度控制；旧配置默认 disabled，旧 AgentOutcome 的 cost/tokens 返回语义不变。真实 GLM-5.3 请求有 22 条 assistant 消息，报告成本 $0.16883044；重复采集账本摘要完全一致。provider 报告值不等于账单。

GPU helper 在导入项目代码前绑定物理 UUID，通过 torch 实际分配和 nvidia-smi 的 PID→UUID 查询验证。真实 Agent 写出计算代码并通过 helper 执行；框架随后独立在两张卡重跑，结果均为 1240。双 rank NCCL all-reduce 成功，注入 rank 1 失败后 rank 0 被终止，随后双 rank 重跑成功。全部成功 rank 的实际 UUID 与请求一致，PID 查询均有记录。

首次真实调用发现 torch UUID 不带 `GPU-` 前缀，导致绑定核对失败；已修正为同一 UUID 的格式规范化后重跑，未删除核对。Agent 调试时也尝试修改 helper，最终服务器代码已与本地修复逐项比较一致。P2 起优化 Agent 的请求须明确候选输出与固定运行设施的边界；项目版本回装使用经过检查的候选制品，不把任意现场修改当成候选。

本次删除了一个通过 `inspect.getsource(OpencodeClient.prompt)` 搜索字符串的旧测试：提取内部方法后，行为未变，该测试仍会失败。保留超时配置行为与运行检查，没有为方法位置编写替代测试。

这是运行设施验收，不代表自动分组、选组或项目优化已完成。GPU lease 约束使用该 helper 的协作进程；不能声称它限制了任意外部 GPU 程序。worker 时限用于识别卡死任务，与 Agent/API 用量无关。
