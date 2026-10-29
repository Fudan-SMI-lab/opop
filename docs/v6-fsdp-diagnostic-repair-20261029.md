# FSDP诊断证据保存的最小真实修复请求

前置[诊断作者审计](v6-fsdp-diagnostic-author-20261029.md)已用真实框架CPU比较复现：原入口返回valid=false，随后诊断摘要包含NaN/Inf时，原样脚本JSON保存失败并抛新ValueError，没有证据文件。有限值对照正常。这是明确的脚本接口问题，不是对FSDP数值根因的推断，也不是新增安全审批流程。

本批只发一次新的真实diagnostic_author请求：原六份task/failure/adapter/candidate/index/interface逐字保留，加前次原样probe、plan和这份CPU失败记录；框架另加worker环境。没有提供Qwen诊断实现、FSDP接受候选、源数值原因或手写修复代码。不投喂无关的top_mismatches等新分析。相同通用原说明后追加明确修复范围，不称单变量A/B或稳定因果实验。

要求仅修复非有限诊断数据的可信持久化与原结果/异常传播：不能掩盖NaN/Inf、伪称有限或容差内；不能放宽质量、改变候选/模型/collective/执行顺序、重新设计采集或实施plan中的其他重跑/参数方案。有限值路径的原观测语义必须保留。Agent可查看提供的CPU重现脚本并自行做有限/NaN/Inf及异常/hook恢复检查；失败与测试产物应留在sandbox。API只统计，无限额。不手工修改生成代码。

当前仍只生成和语法/schema检查，**不自动执行GPU**。完成后审查实际初始输入、全部工具访问、代码差异和CPU测试；再独立检查同一缺陷及原结果保持。如果适合，另声明原baseline/candidate固定双卡观测；原13源/参考及旧probe/plan/失败证据保持，旧失败不会覆盖。新控制仍需显式由原请求清空bundles派生，不能说已有历史baseline测量。

目录`/root/autodl-tmp/dev/v6-work/fsdp-diagnostic-repair-20261029`，控制器`scripts/experiments/v6_fsdp_diagnostic_repair_20261029.py`。生产core ed24af7a/prompt/接受规则均不变；无GPU/大型trace/参考副本、本地raw下载或清理。实际约9.81GiB可用，单普通完成/磁盘监听；无Goal、模型等待轮询或额外投递测试。所有历史Qwen/DDP/FSDP/TP/ResNet结果与样本继续保持。
