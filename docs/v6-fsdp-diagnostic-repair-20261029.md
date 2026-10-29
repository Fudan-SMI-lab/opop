# FSDP诊断证据保存的最小真实修复请求

## 终态审计与独立 CPU 复核

本批于 2026-10-29 01:49:30 UTC 完成。真实角色 `generation-ccd70156dce4`、session `ses_e7ab05083ffenk5GPPcmK3rVqf` 的 10 份初始文件逐字核实；原六份任务输入不变，新增的是原 probe/plan 和明确的 CPU 保存失败记录。44 条 assistant 消息、50 次工具调用（30 bash、11 read、4 edit、3 write、2 todo），0 次 schema 修复；provider 估计费用 0.9636843991，仅统计，不是账单或限额。完整实际工具输入/输出留在服务器；未见后续 Qwen/FSDP 成功答案访问，但文件系统不是封闭环境。

新 probe SHA `9495eeaddcb50a0db6375923e37243bff0ded8d5bf97d75e66f83d821330fe8b`，plan SHA `8cc8aa131441bbb1da4d98f71e404aa818c8e7cf3e48fbe4adfcd7a1b2440a9c`。AST 核验：原函数/类只有 `run` 与 `_atomic_write_json` 改变，新增 `_json_safe` / `_count_nonfinite`；比较、遍历、通信包装、原质量判据不变。保存时将非有限 float 显式编码为 `NaN` / `Infinity` / `-Infinity` 字符串，schema 升到 2，记录编码说明与数量。完整保存失败时尝试 minimal 文件；两次写入均失败则输出 stderr，并保留原返回/异常，不能声称证据保存成功。

Agent 测试的错误全部保留在工具历史中：非有限值数量误写 3 而实际为 4；首次 PYTHONPATH 错误；随后预建输出目录触发 fixture 断言；误把有限 2e-7 差值当作精确相等；补充测试括号语法错误。这些均由 Agent 修正测试后继续，未修改模型容差。Agent 自己删除并重复了已通过的 fixture 目录，故保留的是最终文件和完整命令/输出/编辑历史，不能声称所有中间临时文件仍在。补充测试原先只核验异常类型/消息，其“相同异常对象”描述由本次独立检查补足。

独立使用原 fixture 在**新目录仅运行一次**，finite/NaN/Inf 三项均保留原 `valid=false` 且写出严格 JSON。NaN 的计数仍可能为 0，但 finite 标记为 false、max/mean 为明确的非有限标记；不将其解释成通过。有限案例所有数值、结构及 digest 字段与旧记录一致。审计最初误将不同调用的 `observer_wall_s` 也要求相等，已保存该审计断言错误；仅排除这一计时字段后比较，没有重跑 fixture 或修改生成代码。

另在新目录做五项独立 CPU 边界检查：比较后负 Inf 加原异常、比较前原异常、完整写入失败但 fallback 成功、两次写入失败且原返回、两次写入失败且原异常。均确认原入口仅一次、异常对象身份保持、所有已安装通信属性与 compare hook 恢复，未初始化 CUDA。双写失败明确无证据文件，不冒充成功采集。这八项只是 CPU 接口证据，不证明实际 FSDP 非有限数据、数值原因或模型质量。

审计 SHA `eca14311e92b61c5084e715fe0428c3f781b3456093c981147ac17a20f6e511c`，见[小型公开回执](evidence/v6-fsdp-diagnostic-repair-20261029/public-audit.json)；完整审计及 session parts 在服务器批次根目录和 `audit/`。16 个原冻结文件/引用再次哈希一致。生产 core/prompt/allocator/sampler/接受规则不变。

决定：原样修复脚本适合下一次[固定 FSDP 双卡观测](v6-fsdp-diagnostic-validation-20261029.md)，不安装为生产 collector。不采用计划提出的重复候选或 K=0 控制。旧局限仍在：只保存摘要/digest 而非完整张量；`top_mismatches` 是全元素绝对误差排序，可能包含容差内值；CPU 遍历改变生命周期与步间隔；通信 API 计数不是物理 collective/字节/带宽；无独立源一致性证明；恢复代码不是对部分安装失败等所有路径的 finally 保证。原 FSDP 拒绝及全部历史接受/拒绝、样本均保留。

已于2026-10-29 01:38:58 UTC启动，冻结`542dfba7126d09ff28d3e54538cbc36df3141012`、controller937152；真实OpenCode健康检查通过。01:41:33核验初始`generation-ccd70156dce4`／session `ses_e7ab05083ffenk5GPPcmK3rVqf`：10份初始文件符合协议，9份显式输入110,168 bytes；原六份任务输入逐字不变，附加原probe/plan及9,204-byte失败记录。16个冻结文件/参考哈希保持，10条真实assistant消息、13工具，初始未见后来答案访问，完整终态仍待审计。见[启动核查](evidence/v6-fsdp-diagnostic-repair-20261029/startup-verification.json)和[回执](evidence/v6-fsdp-diagnostic-repair-20261029/launch.json)。

唯一普通监听PID11988确认运行，旧作者监听已标记处理完成；无重复监听/投递测试。实际df余10,526,539,776B（约9.80GiB），无GPU worker、本轮清理或raw本地转存。见[监听回执](evidence/v6-fsdp-diagnostic-repair-20261029/wakeup-setup.json)。当前只确认真实请求和输入，不预判修复正确或批准GPU运行。

前置[诊断作者审计](v6-fsdp-diagnostic-author-20261029.md)已用真实框架CPU比较复现：原入口返回valid=false，随后诊断摘要包含NaN/Inf时，原样脚本JSON保存失败并抛新ValueError，没有证据文件。有限值对照正常。这是明确的脚本接口问题，不是对FSDP数值根因的推断，也不是新增安全审批流程。

本批只发一次新的真实diagnostic_author请求：原六份task/failure/adapter/candidate/index/interface逐字保留，加前次原样probe、plan和这份CPU失败记录；框架另加worker环境。没有提供Qwen诊断实现、FSDP接受候选、源数值原因或手写修复代码。不投喂无关的top_mismatches等新分析。相同通用原说明后追加明确修复范围，不称单变量A/B或稳定因果实验。

要求仅修复非有限诊断数据的可信持久化与原结果/异常传播：不能掩盖NaN/Inf、伪称有限或容差内；不能放宽质量、改变候选/模型/collective/执行顺序、重新设计采集或实施plan中的其他重跑/参数方案。有限值路径的原观测语义必须保留。Agent可查看提供的CPU重现脚本并自行做有限/NaN/Inf及异常/hook恢复检查；失败与测试产物应留在sandbox。API只统计，无限额。不手工修改生成代码。

当前仍只生成和语法/schema检查，**不自动执行GPU**。完成后审查实际初始输入、全部工具访问、代码差异和CPU测试；再独立检查同一缺陷及原结果保持。如果适合，另声明原baseline/candidate固定双卡观测；原13源/参考及旧probe/plan/失败证据保持，旧失败不会覆盖。新控制仍需显式由原请求清空bundles派生，不能说已有历史baseline测量。

目录`/root/autodl-tmp/dev/v6-work/fsdp-diagnostic-repair-20261029`，控制器`scripts/experiments/v6_fsdp_diagnostic_repair_20261029.py`。生产core ed24af7a/prompt/接受规则均不变；无GPU/大型trace/参考副本、本地raw下载或清理。实际约9.81GiB可用，单普通完成/磁盘监听；无Goal、模型等待轮询或额外投递测试。所有历史Qwen/DDP/FSDP/TP/ResNet结果与样本继续保持。
