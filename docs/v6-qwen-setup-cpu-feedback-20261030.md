# Qwen新CPU接口证据的一轮普通续行

**已启动：2026-10-29 21:04:45 UTC（香港10月30日05:04），冻结426f21982299c6a8130b0aee4a8f1139ecee66bb，controller43572。** [启动记录](evidence/v6-qwen-setup-cpu-feedback-20261030/launch.json)、[422身份预检](evidence/v6-qwen-setup-cpu-feedback-20261030/frozen-preflight.json)、[真实本地服务健康](evidence/v6-qwen-setup-cpu-feedback-20261030/session-health.json)通过。首次部署快进因已上传audit文件CRLF与提交LF字节不同而中止；核验归一化内容相同、部署精确git blob后才启动，[错误记录](evidence/v6-qwen-setup-cpu-feedback-20261030/launch-preflight-attempt-1.json)保留。错误发生在controller/session/model/GPU/监听创建之前，不是重启实验或改生产。

21:05:39 UTC[实际启动快照](evidence/v6-qwen-setup-cpu-feedback-20261030/startup-verification.json)：selector generation52eb49771097/session ses_e7684d8d0ffeOcUPWcQBSlHLes，30初始文件/94,796,452B逐字匹配；反馈5,894,790B/SHA30459adc210d4848abd1b0c7754ab9ea3bf70b19c2b37266d0f31433641f8462与真实registry route相等，新CPU事实送达、历史4693和原打包产物保持。与上轮selector提示/模型/工具相同，8输入改变、1旧Python观察输入移除，没有新增输入文件，非整个角色单变量A/B。422冻结身份及共享discovery/prepared/request核验；当时7 assistant/14工具，尚无新GPU jobs。早期读取原request/source/adapter/contract/search等，不能由实际送达推断事实已被阅读或正确使用；后续完整审查仍必要。

唯一[完成/8GiB监听](evidence/v6-qwen-setup-cpu-feedback-20261030/wakeup-setup.json)40560及[存活核验](evidence/v6-qwen-setup-cpu-feedback-20261030/wakeup-verification.json)运行，旧7324已处理。启动核验两卡空闲，实际df余9,260,822,528B（约8.63GiB）。无新质量/接受结论，无本轮清理。

在[上一轮终态审计](v6-qwen-setup-feedback-20261030.md)后声明一次新普通trial，根为`qwen-setup-cpu-feedback-20261030`。它不是重启旧试验、同快照A/B或无辅助原入口证明。依据是对4693原码新增的一次CPU接口控制，而非重复旧优化直到通过。没有手写候选/实现修复、人工选组、新提示或接受规则变更；关闭的提示/解释/核验研究保持关闭。

起点复用上一轮空parent2a42b9b1、prepared009b、当前设备已完成discovery/controls，既有fork_run及CLI resume rounds1使用默认作用域/参数探索。原latest-feedback、旧4693候选和两个原run-only sidecar经现有历史产物机制进入；只在现有supplemental_evidence加入紧凑CPU结果、范围界限和原fixture/结果/sidecar索引。保留所有原反馈字段与字节前缀，先核对真实registry/plan路由，再核对实际新快照。开发者辅助事实边界明确，不声称这次端到端自动发现了CPU问题。

新CPU结果仅说明：singleton切片contiguous保持stride[32,1]，原replacement guard送父路径，CPU同cache示例258→259→260；另一个明确stub setup/verify成功的控制最终仍disabled/not_attempted。前者不是历史GPU逐调用cache轨迹，后者是潜在成功分支缺陷而非本次先发生的E断言原因。fixture无CUDA初始化/模型/Graph执行，既不是完整数值质量，也不给修复代码。原4693 E失败、父回退质量通过但TTFT26.232951超限的拒绝及所有样本保持。

工作量、原始完整质量、两同步/状态生命周期、BF16/SDPA/noTF32、B1/prompt256/full32 greedy、两连续检查/两warmup/五clean不改。当前20e8 GPU原TTFT23.285755、上限24.45004275，禁止旧设备anchor或重新锚定。源映射/此前辅助边界保留，原current及旧接受版本冻结。没有本批独立完整版本/专门native阶段；若新接受，先审核再声明后续独立验证。生成、run-only valid、父回退质量、搜索接受和独立接受分别报告。

API仅观察费用，不加上限。角色实际工具可探测，终态需审核全部命令/输出和初始生成/编辑；外部FS非隔离。没有大raw本地转存或本轮清理；启动时两卡空闲、df约8.83GiB。新长工作只设一个完成/8GiB磁盘监听，低磁盘不是完成，不停止实验或盲删。生产core仍ed24af7a…；raw/private配置/完整会话仅服务器，GitHub仅代码/哈希/小摘要。
