# 从原失败现场生成诊断采集脚本

2026-10-29 00:32 UTC已启动，冻结`5a5a9f181f4fb5ed63f2301dddc6727bec9c0a88`，controller PID928001，实际OpenCode健康检查通过。00:34:26核查初始snapshot `generation-6e81a3220da0`（session `ses_e7aedd58bffetSzUIuSb2IzLpd`）：六份显式输入与协议逐字相同，另有worker-environment共7份，真实provider已有10条assistant记录/18工具调用；目前未见后续诊断或成功实现的路径/字段命中，但不能由此替代终态完整访问审核或宣称环境隔离。21个源/参考/manifest哈希保持，无GPU worker。唯一普通监听PID14176正在等待，实际df10,639,929,344B（9.91GiB）。见[启动核查](evidence/v6-diagnostic-author-20261029/startup-verification.json)、[监听回执](evidence/v6-diagnostic-author-20261029/wakeup-setup.json)。尚无生成物执行结论。

Qwen事实辅助候选已经通过独立整版验收45.17%，但关键观测仍由开发者手工设计。这是当前真实缺口：已有candidate_diagnostics只运行run，不运行完整observe/compare；错误摘要不能告诉下一角色哪一次连续调用、哪一边输出首先不符合原关系。直接把本案例的诊断脚本硬编码到生产不能保证泛化。

本阶段先做一个隔离的真实`diagnostic_author`角色实验，使用已有ProjectAgents/Runtime/snapshot机制，不增加生产路由或失败分类。输入仅来自历史a866失败现场：原task、原失败worker错误/traceback、原adapter和候选代码、baseline/candidate执行请求与原source/controls索引。六份显式输入共约54KB，另有框架自动注入的worker环境；不加载5MB反馈或155MB资源全集。没有提供后来第二次teacher的实测事实、手工observer脚本、ce85/d5bd成功/失败结果或修复答案。

通用任务是让Agent自行检查代码，指出缺失证据，写`diagnostic.py:run(context)`及`diagnostic-plan.json`。未来执行沿用原ExecutionRequest、完整连续质量顺序、warmup/repeats/精度/容差/完成边界；只替换诊断入口、输出路径并附加通用diagnostic_context（原入口和可选源参考）。原始异常在证据持久化后仍必须传播，不能把失败变valid、修改候选或伪造缺失输出。对象/字段映射及观测点由Agent依据源代码生成，不在框架匹配Qwen错误字符串。

这批**只有一次真实角色生成与语法/schema检查，不自动运行其GPU脚本**。允许CPU/API检查，保留所有错误；费用只统计。外部文件系统可访问，提示要求只用原始索引及相关库源码，实际工具访问须核对，不能仅凭初始输入没答案就宣称完全无信息泄漏。该原现场再实验不是同快照A/B，不证明通用能力或因果优势；产生代码也不等于诊断正确。

终态先审核实际输入、完整工具记录、脚本是否保留run/observe连续顺序、字段映射、绑定/所有者/异常/计时边界，以及观测对分配、同步和生命周期的影响。若生成物可执行且没有改变工作，另开明确固定baseline/candidate对照验证；原未观察失败保留，诊断时序不计入接受。原样代码不可用则记录具体问题，再按证据决定最小修正，不重复直到通过。至少还需其他任务验证后才考虑通用生产接口；不会凭单个Qwen正例自动采用。

控制器`scripts/experiments/v6_diagnostic_author_20261029.py`，目录`/root/autodl-tmp/dev/v6-work/diagnostic-author-20261029`。生产core ed24af7a保持；原source/controls及21个文件身份前后校验，raw/private会话只在服务器。当前df约9.92GiB，本批无计划GPU/大型trace/reference复制；无需额外清理。结束后由唯一普通完成/磁盘监听唤醒审查，不启用Goal/模型等待轮询。
