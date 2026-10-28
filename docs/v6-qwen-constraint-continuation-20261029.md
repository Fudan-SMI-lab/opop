# Qwen完整质量通过后的约束反馈续行

终态审计（2026-10-28 22:39:51 UTC）：控制器完成，固定native采集成功；新一轮`trial-a254520daa`没有有效参数候选，current仍b61996e1空bundles，没有搜索接受或独立接受。原request/discovery、baseline文件、adapter498c90f3、source/controls和来源current全部核验不变。没有新discovery或接受后rediscovery。

固定native属于上一份decode-only代码f12e2f50、revision72112d72，不能套给本轮新代码。SQLite重读核对全部68,745条kernel记录与collector导出一致，其中66,495条有Graph node ID；31次cudaGraphLaunch，每次关联2,145条Graph kernel；Graph中2,232条Flash kernel，其他72条Flash。此单独侵入式profile证明旧候选真实Graph执行，仍不改变其26.510787ms TTFT超25.2536046的失败，也不是clean分数或质量验证。

本轮三个真实角色selection/candidate/collection-planner均0 schema修复。实际快照和工具记录确认selector/candidate收到并读取完整目标、质量通过/TTFT失败、旧代码f12e2f50；selector自主选joint范围的prefill与decode两个组（26,868 affected calls），不是手工group。Candidate生成`a86694e792df3baac1ea6d93e5de574fc43fa1818ebee4ce452d4f4557295178`，空参数空间只执行一个参数实例。它共享masters与pool，先捕获31张decode Graph、后捕获一张prefill Graph，用prefill mirror连接两阶段。实际screen在observe内报`teacher-forced replay token sequence differs from greedy run`，早于外层source compare和任何clean样本；没有性能/TTFT测量，不能判断哪条完整输出先偏离源。

Planner仅选candidate_diagnostics并注明无法定位数值首分歧。该instrumented worker的两类事件分别是_is_tracing正常返回False32次和context __exit__正常返回False32次，没有记录异常；它不执行observe/compare，valid不能覆盖真实质量失败，也不能据此推断整个新Graph正常。原Torch的baseline attribution不能替代新候选native路径证明。

候选两次CPU fake-Graph探针也没有完成所宣称的teacher验证：生成32tokens后，探针错误地做了32次decode调用（真实任务为31），触发range fallback；第一次随后NoneType失败，第二次捕获traceback退出0。完整命令/输出已保存。这是探针循环错误，未被证明是当前真实GPU失败原因。Agent仍有跨时段profile减法推断host成本、“decode边界无法修TTFT”、Graph pool只计reserved等过度解释，均不采纳。只确认它依据真实目标反馈改变了选择，不将一次变化解释为稳定泛化成功。

完整[审计](evidence/v6-qwen-constraint-continuation-20261029/terminal-audit.json)SHA`38d564e745e1a5b7f23c3600f955d0f1ad85a78eff37f9edb76d194bceeb0195`；原始SQLite/会话/全部失败留服务器。下一步按[联合候选完整关系诊断](v6-qwen-joint-relation-20261029.md)仅补真实greedy/teacher/source差异及phase routes，不立即追加上下文或改生产规则。GPU已退出，df余11,662,520,320B（10.86GiB），无需清理。

已启动（2026-10-28 22:16:07 UTC／香港10月29日06:16）：冻结`89595894`，控制器PID899959，生产core不变。native worker已完成采集，实际staged revision72112d72；collector observed允许进入一轮普通trial，但原始native/API/kernel仍须终态复核，不预报Graph数量。22:16:50启动核查时真实selection会话`generation-05e1b6eeb267`已有provider Assistant回复，28输入，收到的历史代码SHA确为f12e2f50，质量通过/TTFT失败及完整end_to_end目标均核验。实际routed feedback文件3,053,797字节，不代表全量内容进入模型context；未新增大量资源数据。

空bundle父版本、原request/discovery和adapter498c90f3、共享controls路径均核验相同，未导入已接受版本或复制大参考。唯一普通完成/8GiB监听PID23140 waiting；实际df余11,964,661,760字节（11.14GiB）。仅取小型[启动核查](evidence/v6-qwen-constraint-continuation-20261029/startup-verification.json)、[复用身份](evidence/v6-qwen-constraint-continuation-20261029/preparation-reuse.json)和[监听回执](evidence/v6-qwen-constraint-continuation-20261029/wakeup-setup.json)，没有raw本地转存或清理。新trial独立接受仍未执行。

[固定候选](v6-qwen-attention-fixed-20261029.md)已通过完整质量，但两个端点的TTFT均超过原上限。下一步先为该固定候选采集一次单独Nsight native trace，核查实际Graph/API/kernel；它是侵入式路径证据，不重测clean分数、不作该候选的AB/BA接受。保留其约束失败。

随后从原b61996e1空bundle基线fork，复用未修改request、完成的discovery和只读controls，不重新做intake/evaluator/大参考生成。将一次明确标注为developer-supplied diagnostic的反馈经既有`--feedback`接口给普通流程：原候选完整质量/coverage、无观察器全部5样本、原24.051052 anchor及25.2536046上限、26.510787失败值、原始候选bundle/proposal和完整实现。`decision`中的false是诊断“不接受”标记，明确没有对这份代码执行正式paired-noise判定。基线观察器timing不被伪装成clean配对baseline，旧测量不回写。

旧候选仅作为`previous-candidate`历史修复材料，不安装、不导入旧accepted bundle。新增补充信息只含真实质量/约束事实、测量限制和证据索引，无手工优化方案或指定group。现有route/candidate_artifacts应正常处理；保存实际初始selection/candidate快照、反馈SHA和读取记录，验证它们收到完整目标而非仅decode。允许原有全部优化scope；执行一轮正常真实selection/candidate、默认参数探索和完整评测。API统计observe-only，不加调用限制。失败原样保留，不自动重试；若正式搜索接受新版本，独立完整版本AB/BA与新版本native待下一阶段审计后安排，本批不能宣称独立接受。

这是带有开发者事实诊断输入的原任务续行，不是新一轮无辅助原始入口cohort，也不是同快照单变量A/B。生产core仍ed24af7a，prompt/allocator/sampler/接受规则不变；原same-adapter与fresh-adapter结果以及全部历史失败分别保留。控制器`scripts/experiments/v6_qwen_constraint_continuation_20261029.py`，服务器目录`/root/autodl-tmp/dev/v6-work/qwen-constraint-continuation-20261029`。

启动前真实df约11.50GiB，GPU空闲；仅一个native候选profile和一轮搜索，避免多个大Qwen任务并发，无新controls副本。真实空间由单个8GiB监听关注，完成或低空间仅提交一次续行；无Goal、模型进度轮询、投递测试。本地只保存小摘要/元数据，raw和私有配置留服务器；冻结实验不重启，按既有server-only范围仅在GPU结束后需要时做有校验的无损冷压缩。
