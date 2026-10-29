# Qwen setup失败的一次原样固定观察

后续[普通反馈试验](v6-qwen-setup-feedback-20261030.md)已完成：两个优化角色均未打开本批新观察索引/原文件，新候选在父回退路径通过质量、TTFT超限拒绝。该结果不证明这些观察已被优化器消费；本批所有原样观察、失败及审计SHA保持。新增CPU接口控制针对后续4693代码，与这里04df的GPU观察分开。

后续同一完成通知已核对并去重，不再执行B/C。基于本次新增原样观察产物，准备[一轮普通反馈续行](v6-qwen-setup-feedback-20261030.md)：只追加元数据索引、共享已完成discovery/controls；不向Agent提供本页开发者因果判断，不自动采用probe或修改生产提示。以下终态结论保持。

**终态已审计：2026-10-29 19:55:09–19:56:24 UTC（香港10月30日03:55–03:56），固定B/C各一次完成。** 冻结`f46dab3249065fea699446bed1e1f1efbff4c10f`，controller31776。事件为`batch_finished`，不是磁盘告警。两worker正常返回，10个clean样本、4个完整零误差检查（每次4,861,988元素）全部保留；370份冻结文件、实际source/adapter/request/entry/core和原指针核验不变。没有新Agent、搜索、参数选择、Nsight或性能接受。

[公开审计摘要](evidence/v6-qwen-setup-observation-20261030/public-terminal-audit.json)、[请求身份](evidence/v6-qwen-setup-observation-20261030/request-identities.json)、[完成记录](evidence/v6-qwen-setup-observation-20261030/followup-completed.json)及[审计脚本](../scripts/experiments/v6_audit_qwen_setup_observation_20261030.py)已保存。服务器terminal-audit SHA为`c45883919600b7bdfddb6683a670ff1835a502f1f4cd04f463a3fd4a866b509f`。原样Agent probe为`bc4751ca9b77f9edaf5269c10dd053298e287d50452f51d113f1555a33265ff5`，plan为`33fdb9239bc1420293fd89fe74a139a081276e2c955aca411db4b9c1bdef63a6`，没有手改。实际请求仅改变entry、output及新增diagnostic_context；function/wait原已run/true。共享原controls/reference，independent_reference=null，本批没有重做源比较，人工字段映射边界仍保留。

| 观察端点 | 完整质量 | full generation中位数 | TTFT中位数 | 实际路径 |
|---|---|---:|---:|---|
| 原baseline B | 两次零误差 | 711.711024 ms | 23.741141 ms | 原实现，无安装窗口 |
| 原04df候选 C | 两次零误差 | 708.749832 ms | 24.589549 ms | 62次setup失败后父decode回退、2次父prefill |

原当前设备TTFT上限仍为23.285755×1.05＝**24.45004275 ms**；C超限。framework fallbacks为空不抵消内部回退；这些质量通过只覆盖最终父路径，不能认证候选快路径。所有计时为观察诊断，不作接受或observer开销扣除，原screen拒绝不变。B的761.488258 ms长样本保留。C setup8840.142951 ms在warm-model评分之外；runtime/lifecycle峰值相对B增加103,039,488/231,608,320 B，不是显存收益，也不能把与旧setup时间的差解释为观察器成本。

修复在真实安装中保留了晚发失败：198次原assert调用，196成功、2失败、0unfinished；保存128条详细记录＝126成功+2失败。69条后续成功详情省略，已保存的成功seq2被seq198失败挤出并留下stub，两个失败均保留且failure overflow为0。保留的126条成功统计均为零误差；不能由成功结果推断另外70条未保存详情的原始误差。B证据3,081 B/SHA`6715ecb9ed4663cd134641d7261143c4953e92eb95fb35b128e3008a67ebfea8`，C证据739,927 B/SHA`08e074257fa28546d6e112fb2a5bc3135c1deb793a9d40e1cbc78653bf375d38`，完整JSON留服务器。

| 安装比较 | seq | 报告max abs | 内部 .012/.012超差数 | 请求 .02/.02超差数 |
|---|---:|---:|---:|---:|
| fused_eager_vs_parent_step_1 | 1 | 0.28125 | 31,892 | 10,716 |
| exact_chain_E_absorb | 198 | 22.03125 | 151,854 | 151,790 |

两次均为有限FP32 CUDA操作数，形状1×151936。完整操作数未保存，表中全量计数/max/mean是probe报告，不能声称独立raw重算；审计独立复算了两个失败、两套容差、各64项，共**256个已保存超差值的算术/索引/排序**。例如首项比较index4577为actual−7.46875、expected−7.75，差0.28125大于请求允许0.1749999970。因此至少这些setup操作数的差异并非仅因内部容差更严，但也不等于完整32-token快路径质量结果，更不能确定Triton或某种算术的唯一原因。两次比较argmax各自相同，分别1246、6366；计划预期E的argmax不同未被实际支持。

记录167次parent reference调用（7 prefill、160 decode），无overflow。最后两次分别seq166/167，输入同一token264、同一live cache对象140421529011888，输入长度却分别258/259，返回同对象并推进到259/260；36层计数及首层keys形状一致。结合候选625–629行先parent decode、再把同cache交给replacement，以及fast_enabled在633行检查之后才开启、664–666行未开启时调用parent，支持**E比较的两个输出来自不同decode位置**这一代码/调用顺序推断。reference和assert使用独立计数器，没有统一跨hook事件ID，不能将上述连接说成直接跨事件关联；首层metadata也不是所有层的alias图，不能推为全部数值差异的唯一原因。

原内部异常仍由候选捕获，原entry各执行一次并返回；没有把内部失败抛给外层、放宽阈值或强制快路径。安装内部实现确实执行过，外层回退不等于从未执行内部。GPU现场未另记hook对象恢复身份；恢复结论的范围是代码审查及前置独立CPU边界，不能称为新增GPU身份对照。CPU取值、同步、存活闭包/异常帧和owner仍扰动执行，B通过不是量化无扰动证明；没有新native Graph轨迹。

一次只读摘要审计最初假定零误差记录也有top数组，触发KeyError；producer在计数零时省略数组。原脚本输出保留为audit-attempt-1，只修审计读取为缺数组且计数零时接受，未重跑GPU/模型/原CPU fixture或修改probe。实际[启动核验](evidence/v6-qwen-setup-observation-20261030/startup-verification.json)于19:56:27取到的已是终态，不能当成运行中的早期快照。

这次完成了有界观察器的本案执行验证，保留新的数值/输入来源证据；没有新优化接受或通用诊断闭环证明。后续若继续优化，应以原framework反馈和原样观察产物作为真实输入，另行核验实际角色读取，不能手填修复答案、自动安装collector或重开已关闭的提示/审查研究。本轮不再追加试验。当前两GPU空闲，实际150GiB余10,250,383,360 B（约9.55GiB），未清理、未转存raw/private资料。唯一监听31864已正常发出真实终态通知并退出，本轮已审计并[标记处理](evidence/v6-qwen-setup-observation-20261030/wakeup-handled.json)，不重启、不做投递测试。生产core和全部历史接受/拒绝不变。

## 原定协议（保留）

在[修复作者审计](v6-qwen-setup-diagnostic-repair-20261030.md)和独立CPU边界通过后，声明一次固定baseline B、原04df候选C观察。exact Agent probe `bc4751ca9b77f9edaf5269c10dd053298e287d50452f51d113f1555a33265ff5`、plan `33fdb9239bc1420293fd89fe74a139a081276e2c955aca411db4b9c1bdef63a6`，无手改。使用既有LocalExecutor/设备lease，每worker原entry一次；本批没有Agent、搜索、参数选择、性能接受、Nsight或生产采用。

B请求来自qwen-device-refresh-fixed-20261030的既有原baseline endpoint；C请求来自qwen-device-feedback-20261030 trial0db61e9938原screen。均保留原materialized source、adapter498c90f3、合同/request、shared reference、设备UUID20e8及原bundles，C仍原样04df、空参数。不复制source/controls/reference，不发布指针。仅entry/function/output/wait及diagnostic_context可改变；function/wait原来已是run/true，实际不变。independent_reference=null，不新增源比较；人工源字段映射边界保留。

顺序B-observed、C-observed各一次。B须原worker passed/valid、两次完整检查/math不变、原entry返回且install_windows为空；不以B的TTFT作接受门或重设anchor。若B不满足门，C记not_run。C即使worker passed仍须分别审核原valid、内部catch/fallback、setup比较、质量与覆盖，不按controller完成认定诊断正确。保留所有错误/意外通过，不重测到通过。

完整B1/prompt256/32greedy tokens、BF16/SDPA/noTF32、full logits .02/.02/exact tokens、两同步边界、原state生命周期、2continuous/2warmup/5clean若达到均不变。当前原anchor23.285755→TTFT limit24.45004275；旧25.2536046/25.49943795不适用。观察计时仅诊断，不做接受/差值扣除或跨设备收益。原04df父回退质量通过但TTFT24.679412拒绝保持，内部.012/.012失败不预判为原质量超差。

预期获得安装期比较摘要/有限超差项、原断言结果和部分reference输入/cache来源。新策略最多128详细记录/32失败、reference前2048，后续失败优先淘汰成功项且明确省略；超过范围不补造。失败后CPU取值改变同步/生命周期，存续reference闭包仍有额外调用，不承诺无扰动/任意并发安全。摘要不是完整raw operands，不能重算所有聚合；setup比较与全任务source质量分开，object-id/首层metadata不证明所有KV别名或唯一根因。计划中的未测overhead/内存/因果保证不采纳。

启动前核验真实150GiB df、GPU空闲、实际UUID与370冻结身份；当前约9.56GiB，预计小摘要/原评价结果，无大型trace/参考副本。本轮无清理/raw下载。server保存raw/private历史，GitHub仅代码、SHA、小摘要；如任务仍运行只设一个普通完成/8GiB监听，旧43664处理后不重启。历史接受/拒绝、关闭的研究和生产core全部保持。
