# Qwen完整质量通过后的约束反馈续行

[固定候选](v6-qwen-attention-fixed-20261029.md)已通过完整质量，但两个端点的TTFT均超过原上限。下一步先为该固定候选采集一次单独Nsight native trace，核查实际Graph/API/kernel；它是侵入式路径证据，不重测clean分数、不作该候选的AB/BA接受。保留其约束失败。

随后从原b61996e1空bundle基线fork，复用未修改request、完成的discovery和只读controls，不重新做intake/evaluator/大参考生成。将一次明确标注为developer-supplied diagnostic的反馈经既有`--feedback`接口给普通流程：原候选完整质量/coverage、无观察器全部5样本、原24.051052 anchor及25.2536046上限、26.510787失败值、原始候选bundle/proposal和完整实现。`decision`中的false是诊断“不接受”标记，明确没有对这份代码执行正式paired-noise判定。基线观察器timing不被伪装成clean配对baseline，旧测量不回写。

旧候选仅作为`previous-candidate`历史修复材料，不安装、不导入旧accepted bundle。新增补充信息只含真实质量/约束事实、测量限制和证据索引，无手工优化方案或指定group。现有route/candidate_artifacts应正常处理；保存实际初始selection/candidate快照、反馈SHA和读取记录，验证它们收到完整目标而非仅decode。允许原有全部优化scope；执行一轮正常真实selection/candidate、默认参数探索和完整评测。API统计observe-only，不加调用限制。失败原样保留，不自动重试；若正式搜索接受新版本，独立完整版本AB/BA与新版本native待下一阶段审计后安排，本批不能宣称独立接受。

这是带有开发者事实诊断输入的原任务续行，不是新一轮无辅助原始入口cohort，也不是同快照单变量A/B。生产core仍ed24af7a，prompt/allocator/sampler/接受规则不变；原same-adapter与fresh-adapter结果以及全部历史失败分别保留。控制器`scripts/experiments/v6_qwen_constraint_continuation_20261029.py`，服务器目录`/root/autodl-tmp/dev/v6-work/qwen-constraint-continuation-20261029`。

启动前真实df约11.50GiB，GPU空闲；仅一个native候选profile和一轮搜索，避免多个大Qwen任务并发，无新controls副本。真实空间由单个8GiB监听关注，完成或低空间仅提交一次续行；无Goal、模型进度轮询、投递测试。本地只保存小摘要/元数据，raw和私有配置留服务器；冻结实验不重启，按既有server-only范围仅在GPU结束后需要时做有校验的无损冷压缩。
