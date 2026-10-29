# Qwen 独立接受版本的接受后续行

2026-10-30香港时间用户确认欠费停机后已重启。实际检查原控制器消失、无completion，第一轮45c44c1619已因安装位置与bundle声明不一致失败；第二轮selector原会话27消息仍在但尚无selection.json。271既有文件/33初始输入核验，保留原日志和中断状态；按[断电恢复协议](v6-qwen-accepted-recovery-20261030.md)只恢复原第二轮会话，不重复第一轮、controls或初始discovery。下面“运行”描述均为停机前历史状态。

已于 **2026-10-29 08:51:03 UTC（香港16:51）** 启动，冻结 `b18cef24ebfe3d50223de67c55f43dbe9e2bc4ce`，控制器986775；真实本地 OpenCode server 健康预检通过，`server_url=null`，生产核心仍为ed24af7a。见[启动回执](evidence/v6-qwen-accepted-continuation-20261029/launch.json)、[预检](evidence/v6-qwen-accepted-continuation-20261029/preflight.json)、[准备复用](evidence/v6-qwen-accepted-continuation-20261029/preparation-reuse.json)与[两轮协议](evidence/v6-qwen-accepted-continuation-20261029/search-protocol.json)。

08:52:38 的[实际启动核验](evidence/v6-qwen-accepted-continuation-20261029/startup-verification.json)确认155冻结文件保持，原分支current仍009b4178，新分支已对接受版本执行 `discover-04002e90dc`，实际worker代码为f065、entry为aed604ee，request/adapter/共享controls一致。发现后的revision eb9dfd94保留同一套源码、合同和单bundle；这只是重新发现产生的revision，尚非新优化。真实collection-planner generation-b3e06c43befb已创建，初始6文件；此角色没有previous-feedback输入。新分支保存的40,824B原始framework反馈字节相等，但此时尚不能声称后续selector/candidate已收到或读过，须在终态审计实际快照和工具记录。

唯一普通完成/8GiB磁盘监听36120已运行，[预检与启动](evidence/v6-qwen-accepted-continuation-20261029/wakeup-setup.json)和[等待状态](evidence/v6-qwen-accepted-continuation-20261029/wakeup-verification.json)留档；旧42372已处理并退出，不重启、不额外测试投递。启动核验时处于Agent阶段，GPU进程为空，数据盘可用12,899,360,768B（约12.01GiB）。后续由完成或磁盘事件接续，不轮询模型。以下保留启动前协议与维护记录。

启动前维护已于08:32:00–08:33:17 UTC完成。仅旧FSDP终态28个worker RUN目录，397 payload 4,275,658,397→386,632,092B，逐一gzip/解压SHA与342其余文件核验；149冻结依赖、10当前版本、6完整源参考及新e5ad探索版本源码保持。净释放3,888,533,504B（3.62GiB），余12.50GiB。[后验回执](evidence/v6-qwen-accepted-continuation-20261029/storage-postcheck.json)，恢复入口服务器 `storage-audit-20261029/qwen-accepted-continuation/run/RESTORE.md`/journal。旧未压缩路径缺失不是证据丢失；本批输入与参考不在压缩范围。

CPU启动预检通过：155冻结文件、e5ad完整版本/单joint bundle、原prepared baseline009b、40,824B原样feedback SHA `49c8c518e8f64920adf62c1c1d1bb6a44179e48584f27919fd33ac3aa8718142`。控制器 `scripts/experiments/v6_qwen_accepted_continuation_20261029.py`；two-round默认角色/参数探索不另设API上限。反馈即既有完整版本result字节，不把开发审计/此次维护/手工事实送入Agent。

从[独立接受的精确f065完整版本](v6-diagnostic-index-independent-20261029.md) e5ad9a5b3f131dd5a83e238f9b9fc71baf3f91bc6eabea061c103f1823de4939 fork新分支，复用原source/adapter/contract/controls。原independent current仍009b4178；fork显式选择已完整评估且合格的exploration revision，不发布到旧分支。只使用该次原样framework latest-feedback.json，不加入developer数值表、因果判断、手工分组/候选或其他目标旧bundle。

目的为此前Qwen尚缺的接受后重发现、反馈消费和在已安装joint bundle基础上的替换/组合证据。通过既有CLI resume执行**两轮**普通selector/candidate/default tuning；先fresh discovery当前已接受版本，因为此前发现不对应它。两轮是本批预先声明边界，非失败后加轮/重试到通过；API observe-only，无新调用上限。可能没有新接受、可能替换单bundle、可能组合，均按实际记录，不预设多bundle或进一步加速。

原完整B1/prompt256/32greedy BF16/SDPA/noTF32、full logits.02/.02/exacttokens、两同步/state lifetime、2continuous/2warmup/5samples、原TTFT anchor24.051052=>25.2536046、接受噪声规则不变。prepared baseline009b4178和adapter498c90f3、独立源/controls哈希复用，手工外部映射边界保持，不声称新原入口无辅助cohort。生产core/prompt/sampler/allocator保持；433B读取提示研究已关闭，不进入本批生产角色。

仅当搜索接受了**不同的活动bundle集合**，才对最终完整版本相对原prepared baseline再做一次publish=False AB/BA；无变化则not_run，不重测旧f065直到通过。若新ID只是相同有效策略，审计必须区分ID变化与实际机制变化，独立相对原baseline收益不得当作在47.3251794%之上的增量。当前批不另行安排大Nsight，真实planner可能采集，需记录真实选择与路径；若新版本缺原生路径证据，则审计后决定是否有必要，不能沿用旧32Graph数量。

FSDP K2独立拒绝不在本批重试；旧K3接受/后续噪声拒绝、Qwen各原目标接受和旧拒绝80样本、DDP/TP/ResNet历史全部保持。不新增mandatory reviewer/collector、不继续提示或解释角色到通过、不扩大context或硬编码失败路由。新feedback与实际初始snapshot必须核验原样传递；外部FS非隔离，终态审查实际工具使用。

启动前先服务器内无损冷压缩已结束旧FSDP搜索/独立验证的明确worker RUN JSON；source/store/controls/当前接受版本/模型/provider和本批149冻结依赖排除，逐一gzip/解压SHA和其余文件/保留版本复核后才启动。原数据可按独立journal恢复，禁止raw本地转存。单个普通完成/8GiB监听，低磁盘不等同完成；不终止活跃实验或盲删。无Goal、模型等待轮询或额外投递测试。
