# Qwen新CPU接口证据的一轮普通续行

在[上一轮终态审计](v6-qwen-setup-feedback-20261030.md)后声明一次新普通trial，根为`qwen-setup-cpu-feedback-20261030`。它不是重启旧试验、同快照A/B或无辅助原入口证明。依据是对4693原码新增的一次CPU接口控制，而非重复旧优化直到通过。没有手写候选/实现修复、人工选组、新提示或接受规则变更；关闭的提示/解释/核验研究保持关闭。

起点复用上一轮空parent2a42b9b1、prepared009b、当前设备已完成discovery/controls，既有fork_run及CLI resume rounds1使用默认作用域/参数探索。原latest-feedback、旧4693候选和两个原run-only sidecar经现有历史产物机制进入；只在现有supplemental_evidence加入紧凑CPU结果、范围界限和原fixture/结果/sidecar索引。保留所有原反馈字段与字节前缀，先核对真实registry/plan路由，再核对实际新快照。开发者辅助事实边界明确，不声称这次端到端自动发现了CPU问题。

新CPU结果仅说明：singleton切片contiguous保持stride[32,1]，原replacement guard送父路径，CPU同cache示例258→259→260；另一个明确stub setup/verify成功的控制最终仍disabled/not_attempted。前者不是历史GPU逐调用cache轨迹，后者是潜在成功分支缺陷而非本次先发生的E断言原因。fixture无CUDA初始化/模型/Graph执行，既不是完整数值质量，也不给修复代码。原4693 E失败、父回退质量通过但TTFT26.232951超限的拒绝及所有样本保持。

工作量、原始完整质量、两同步/状态生命周期、BF16/SDPA/noTF32、B1/prompt256/full32 greedy、两连续检查/两warmup/五clean不改。当前20e8 GPU原TTFT23.285755、上限24.45004275，禁止旧设备anchor或重新锚定。源映射/此前辅助边界保留，原current及旧接受版本冻结。没有本批独立完整版本/专门native阶段；若新接受，先审核再声明后续独立验证。生成、run-only valid、父回退质量、搜索接受和独立接受分别报告。

API仅观察费用，不加上限。角色实际工具可探测，终态需审核全部命令/输出和初始生成/编辑；外部FS非隔离。没有大raw本地转存或本轮清理；启动时两卡空闲、df约8.83GiB。新长工作只设一个完成/8GiB磁盘监听，低磁盘不是完成，不停止实验或盲删。生产core仍ed24af7a…；raw/private配置/完整会话仅服务器，GitHub仅代码/哈希/小摘要。
