# Qwen 独立接受版本的接受后续行

从[独立接受的精确f065完整版本](v6-diagnostic-index-independent-20261029.md) e5ad9a5b3f131dd5a83e238f9b9fc71baf3f91bc6eabea061c103f1823de4939 fork新分支，复用原source/adapter/contract/controls。原independent current仍009b4178；fork显式选择已完整评估且合格的exploration revision，不发布到旧分支。只使用该次原样framework latest-feedback.json，不加入developer数值表、因果判断、手工分组/候选或其他目标旧bundle。

目的为此前Qwen尚缺的接受后重发现、反馈消费和在已安装joint bundle基础上的替换/组合证据。通过既有CLI resume执行**两轮**普通selector/candidate/default tuning；先fresh discovery当前已接受版本，因为此前发现不对应它。两轮是本批预先声明边界，非失败后加轮/重试到通过；API observe-only，无新调用上限。可能没有新接受、可能替换单bundle、可能组合，均按实际记录，不预设多bundle或进一步加速。

原完整B1/prompt256/32greedy BF16/SDPA/noTF32、full logits.02/.02/exacttokens、两同步/state lifetime、2continuous/2warmup/5samples、原TTFT anchor24.051052=>25.2536046、接受噪声规则不变。prepared baseline009b4178和adapter498c90f3、独立源/controls哈希复用，手工外部映射边界保持，不声称新原入口无辅助cohort。生产core/prompt/sampler/allocator保持；433B读取提示研究已关闭，不进入本批生产角色。

仅当搜索接受了**不同的活动bundle集合**，才对最终完整版本相对原prepared baseline再做一次publish=False AB/BA；无变化则not_run，不重测旧f065直到通过。若新ID只是相同有效策略，审计必须区分ID变化与实际机制变化，独立相对原baseline收益不得当作在47.3251794%之上的增量。当前批不另行安排大Nsight，真实planner可能采集，需记录真实选择与路径；若新版本缺原生路径证据，则审计后决定是否有必要，不能沿用旧32Graph数量。

FSDP K2独立拒绝不在本批重试；旧K3接受/后续噪声拒绝、Qwen各原目标接受和旧拒绝80样本、DDP/TP/ResNet历史全部保持。不新增mandatory reviewer/collector、不继续提示或解释角色到通过、不扩大context或硬编码失败路由。新feedback与实际初始snapshot必须核验原样传递；外部FS非隔离，终态审查实际工具使用。

启动前先服务器内无损冷压缩已结束旧FSDP搜索/独立验证的明确worker RUN JSON；source/store/controls/当前接受版本/模型/provider和本批149冻结依赖排除，逐一gzip/解压SHA和其余文件/保留版本复核后才启动。原数据可按独立journal恢复，禁止raw本地转存。单个普通完成/8GiB监听，低磁盘不等同完成；不终止活跃实验或盲删。无Goal、模型等待轮询或额外投递测试。
