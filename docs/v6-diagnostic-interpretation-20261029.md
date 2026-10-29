# 两任务诊断解读与既有反馈接口实验

Qwen 和 FSDP 的原样 Agent 诊断脚本已分别执行，但采集后的事实解读、证据边界判断和反馈整理仍由开发者承担。下一阶段用同一套通用 `diagnostic_interpretation` 指令，按 Qwen、FSDP 顺序各发一次新的真实角色请求，验证这一缺口。不同任务与产物，不是单变量 A/B，不自动采纳新角色或扩大生产上下文。

每个角色只获得原 task、原失败记录、证据文件索引、通用反馈 schema，框架自动增加 worker 环境。索引指向原 adapter/candidate/source/reference、Agent 自己生成的 probe/plan 和已完成的固定 B/C worker/诊断产物；不内嵌所有梯度摘要或大张量，不提供开发者 terminal-audit、上轮事实表、数值原因、其他任务输出或后续成功实现。Agent 可选择读取原始 JSON/CPU 张量和相关源码，实际工具访问与计算必须留档。外部文件系统并非隔离，不能仅凭输入文件名就宣称没有额外访问。

输出 `supplemental-evidence.json`：简短摘要、事实及其 path/SHA/具体位置/phase/rank/调用范围、局限、未解决问题和明确分开的条件性假设。目标是把真实观察交给后续角色，不是要求写修复实现。可以明确报告不足或错误；schema 通过不等于事实正确，不能把 worker 完成、run-only 诊断或摘要 hash 当成质量/因果/性能接受。

此批只做真实文件解读和小型 CPU 核查，无新模型/GPU/collective、候选搜索、源参考复制、性能接受或生产改动。Agent 不能重新运行实验；不得修改原任务/容差/冻结数据。API 费用仅统计，无调用限额。原测试错误与所有记录保留，不重复角色直到通过。

生成后控制器仅在 CPU 将**原样 packet**放入现有 `supplemental_evidence`，对原历史反馈运行既有 `feedback.route`，检查该字段保持且其他路由字段不变。该检查的 current_revision 来自原物化请求，仅验证接口传输；不是新的正式 decision、真实 selector/candidate 输入或下游使用。不会手工补正角色答案。最终须审计真实初始快照、所有读取、生成文件/引用/计算与局限，才能决定后续用途。

Qwen 产物来自 `diagnostic-author-validation-20261029`，FSDP 来自 `fsdp-diagnostic-validation-20261029`；后一脚本经过一次明确的 CPU 持久化缺陷修复。两者的观测扰动、原样数据保留程度、source mapping 与开发者协助边界仍存在，不能合并成无人干预原入口成功证明。

批次目录 `/root/autodl-tmp/dev/v6-work/diagnostic-interpretation-20261029`，控制器 `scripts/experiments/v6_diagnostic_interpretation_20261029.py`。保持生产 core `ed24af7a…`。启动前核实实际 df 与无 GPU worker，预计仅增加角色/session/小型 CPU 产物。大文件与 private provider 信息留服务器；只发布 code、hash、索引及小型脱敏摘要。设置一个普通完成/磁盘监听，无 Goal、模型等待轮询或投递测试。
