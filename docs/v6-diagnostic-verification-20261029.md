# 独立证据核验角色实验

已于 2026-10-29 03:02:08 UTC（香港11:02）启动，冻结 `9c564293739babd1c40a7232807b23d66370e96e`，controller949392，真实 OpenCode 本地服务健康检查通过。预检 Qwen/FSDP 六显式输入分别 **57,776 / 76,543 bytes**，共 123 个冻结文件；生产 core 不变。见[启动回执](evidence/v6-diagnostic-verification-20261029/launch.json)、[预检](evidence/v6-diagnostic-verification-20261029/preflight.json)。

03:04:36 核验 Qwen `generation-8dcfb0a5babb` / session `ses_e7a6445ebffeREPiWgMj9dZ3cX`：7 个初始文件逐字符合协议，17 条真实 assistant 消息、23 次工具调用；真实 read 已读取 `evidence-index.json` 和完整 `proposed-packet.json`。初始工具访问未见开发者审计/后续答案，完整终态仍需核查；FSDP 后续按序执行。见[初始快照核查](evidence/v6-diagnostic-verification-20261029/startup-verification.json)。

唯一普通完成/磁盘监听 PID50544 已核实实际进程命令，原监听47044标记已处理。当前实际 df 余 10,218,143,744 bytes（约9.52GiB），阈值8GiB；没有 GPU worker、清理或 raw 本地转存。见[监听回执](evidence/v6-diagnostic-verification-20261029/wakeup-setup.json)。完成后按真实结果自动续行，等待期间不调用模型。

当前诊断解读已经能找到关键观测，但两份真实报告同时含有具体事实错误、范围错误和过度因果结论。此实验针对这一已展示的缺口，用相同通用 `diagnostic_verification` 指令，按 Qwen、FSDP 各进行一次新的真实请求。不是重复 interpreter 直到通过，也不直接新增生产审批步骤。

每个角色得到上一阶段原始四个输入的精确字节，以及原样 `proposed-packet.json` 和通用 verification schema；框架另加 worker 环境。不给开发者错误清单、修正事实表、后来成功实现或另一案例答案。允许按索引和原 packet 引用读取已有原始文件、前一角色 CPU 检查及相关源码；实际访问仍须审计，外部文件系统不是隔离环境。

输出 `verification.json`，逐项检查原 summary/facts/limitations/unresolved/hypotheses，区分 supported/mixed/unsupported/contradicted/unverified，附引用和理由。另生成 `supplemental-evidence.json`，只保留有支持且对原失败有用的少量观察、边界和未解决问题；不要求优化实现，hypotheses 可以为空。验证器只检查接口与逐项覆盖，不会用硬编码“正确答案”替代事实审计。

本批无新模型/GPU/collective、诊断 worker、候选搜索、性能接受、reference 复制或 production 改动。允许 CPU 检查已有文件，保留错误和所有工具历史。API 仅统计，不限额。两案例任务与证据不同，不是同快照 A/B，也不能据一次结果宣称稳定泛化。若 verifier 仍产生错误，记录不足，不自动迭代直到通过或继续堆叠角色。

生成后仅在 CPU 经既有 `feedback.route` 核对原样 packet 在 `supplemental_evidence` 中保留、其他路由字段不变。不会自动交给 selector/candidate；后续审计必须分别确认生成完成、事实支持、实际输入投递和下游使用，不能互相替代。

控制器为 `scripts/experiments/v6_diagnostic_verification_20261029.py`，目录 `/root/autodl-tmp/dev/v6-work/diagnostic-verification-20261029`。冻结生产 core `ed24af7a…` 不变；保留上一批 116 个冻结文件，并增加原样 packet/引用文件哈希。启动前检查真实容器 df 和 worker，预计仅小型 role/CPU 产物；raw/private state 留服务器，GitHub 仅代码、哈希和小型摘要。新长工作配置唯一普通完成/磁盘 watcher，不启用 Goal、模型等待轮询或投递测试。
