# DDP 新候选原正式配对验证（2026-10-31）

**正式结果：拒绝，原因是第二对收益未超过原噪声界。** 终态2026-10-31 00:35:32.384514UTC / 香港08:35:32，`batch_finished`；不是低磁盘事件。源与隔离branch CURRENT仍为0bc64a5a/已接受02b933。没有独立晋升或候选重试。

| 配对 | B中位数 ms | C中位数 ms | 收益 ms | 原噪声界 ms | 判定 |
| --- | ---: | ---: | ---: | ---: | --- |
| AB | 48.592321 | 47.514165 | 1.078156 | 0.210590 | 通过 |
| BA | 48.134444 | 47.780430 | 0.354014 | 0.380574 | 未通过 |

聚合48.3633825→47.6472975ms只是本次观测（1.4806%），不称接受收益，不与前固定样本合并。B48.757292/C48.527301尾部保留，原anchor133.379621/limit140.04860205保持且约束通过。复算先aligned rank-max后median/MAD，正式判定与原API一致。

本次10jobs/20ranks通过，四评价endpoint含40个rank样本、16次全量检查（每次248,879,620元素）、24份各自通信trace；8个首检查为0，8个次检查rank0最大9.5367431640625e-7、rank1为1.1920928955078125e-7，原容差及exact-replica观察通过。质量调用仍是先parent、后setup/capture，不是两个优化稳态检查；无完整原tensor保存或新的source比较。只核8个当前关键文件，不重做历史1947身份审计。

每份trace双方都是8 AllReduce、17Graphlaunch，总497,759,232B不变。B1138kernels/1122Graph/group0；C1138/1130Graph/group1、一小TREE入口+7RING入口，均在Graph外。实际用了新group，不能据kernel入口后缀断言逐次协议。新native `nsys-1e8a160e79` 每rank1128kernels/1120Graph/17个versioned launch/8eagerAR，实际worker channels00/01记录SHM/direct/direct；两个扰动周期NCCL总时长35.787365/189.076747ms，尾部保留，不当wire、关键路径或clean扣减。

独立timeline48.553366/48.447640ms、8AR/67sync/0gradientcallbacks/1bucket；0回调是readiness未知，不是没有梯度。校准三载荷共30合成样本，rank-max中位数0.090959/3.849092/10.499552ms，运行普通collector/default group，不是候选fastgroup或模型重放，也不是传输下界。新tuning两rank确有`NCCL_PROTO ... Simple`配置日志，小载荷TREE/SIMPLE、大片RING/SIMPLE；仍是含setup/warmup的另一周期，未与本次计时逐kernel连接。Torch运行峰值约4.613GB/lifecycle约5.656GB/reserved6.354GB，微小worker差异不代表收益或全部NCCL外部分配。安装耗时6.765395–10.184382ms不是完整setup成本。

真实planner generation7cf8aa3c635c，8输入101,879B，9assistant/14tools/0schema修复；2532展示行核实，observations工具8截断后12续读，8文件均完整暴露。工具10错误根路径保留，11修正读取；计划一次写入无编辑，末工具仅JSON/schema/assertions。未观察到模型/张量/GPU/真实collective/private-config正文或后续答案访问，外部FS非hermetic。provider0.18003949998只是估计。计划自主请求6collector，5个实际执行；`binding_cost`因真实替换关系返回`not_applicable`，无额外ablationGPU。候选诊断每rank6个正常False生成器位置/458returns，无exception，不是observe比较。worker旧deprecation/barrier/AccumulateGrad stream警告保留。

planner“保留父bundle并追加”的说法错误，输入`revisions.json`仅有候选bundles/parentID，未给实际父构成；不是已运行第二个bundle。profileNCCL/clean比例、rank差异的wait-skew归因也不采纳。原计划和实验完全保留。本轮下一步已落实为[通用父子bundle事实输入修正](v6-project-collection-composition-20261031.md)，无新模型链。旧只读运行环境cb34/实验core ed24不变；工作代码的通用数据补充单独记录，不回写已完成实验。

[公开摘要](evidence/v6-ddp-pipeline-next-validation-20261031/public-terminal-audit.json)对应终审 `87bb29646f9fe883dbadb378b1e71ae31470d1e3e17b66f52e1f0c3626a9f7a3`，原数值审计 `3fc88cca827ed437012d9f7658b218478ea9124434a69f6ca8a84c2e1036b9c7`。首次数值审计通过；摘要初稿安装上界不完整，已按原8条setup记录更正且旧稿留服务器，没有重跑workload。原trace/SQLite/全部history/工具输出只留服务器；本地只收取小摘要。实际150GiB盘余9,194,303,488B，GPU空闲；无清理，旧23068已handled且进程退出，无新监听/自动GPU任务。

以下为原声明与启动记录。


[固定B/C](v6-ddp-pipeline-next-fixed-20261031.md)原质量通过，48.194381→47.633574ms仅观察1.16%改善，因此需要原正式噪声判定。本次**仅一次**现有`ProjectSearch.validate_bundle(publish=False)`，fresh AB再BA；保持原工作量、质量、anchor、noise、约束及diagnostic profiles，固定样本不合并，拒绝后不重试或换参数。

实际启动**2026-10-31 00:29:33.892849UTC / 香港08:29:33**，controller **258723**，唯一隐藏监听 **23068**。使用[通用实验编排](../scripts/experiments/v6_validate_fixed_candidate.py)，实际controller SHA **e06f1a99e1759a73398ac6cd898bb35a472dbe41941ceeb6724cc29ebd004844**；复用既有cb34d3ea只读环境，生产core ed24af7…不变。8当前关键身份，没有旧1947文件遍历、模型健康调用、重复环境副本或Git发布门槛。

源`ddp-pipeline-next-fixed-20261031/ddp-fixed` current0bc仍为已接受02b933；C精确staged **805deb71707914fe2eb9ae52259fa66ae536d8470a6d9e7808bb3911f2feada1**，singlefixed-pipeline-next，代码e610e944…28,517B/proposal4a00833b…9,663B。空参数仍调用自身fine8，没有fine9/coarse5或新实现。fork共享原request/prepared7a49/current discovery/只读controls，无新discovery/source-control/reference副本或发布。publishFalse下current不移动与正式是否接受是不同问题。

保留正常resource_feedback profile及真实普通collection-planner，由其自主选择已有collector，控制器不手选。没有新selector、candidate、参数搜索、独立闭合或生产采用阶段。只有真正的新正式接受才能另行决定完整版本独立闭合。

[08:31:26启动记录](evidence/v6-ddp-pipeline-next-validation-20261031/startup-verification.json) SHA **12a2196734d324f332d951f20f2f847dacb42c0aea23476b955da0d46ca7b4ec**：四份AB/BA实际请求及正常profile均验证原request、entry/adapter、bundle、空参数、controls、设备身份；各两rank结果已返回，但这里未作终态质量/噪声接受结论。真实planner generation-7cf8aa3c635c/session ses_e70a1e048ffeLk2OXOm1HxDSXF，8实际输入101,879B与snapshot逐字节相等、core相等，完整工具/消费/collector结果待终态核对。最初目录查找仅限trials未找到snapshot，扩大到本branch实际目录即可；没有重跑角色。

启动150GiB盘余9,316,581,376B，512MiB预测+8GiB告警通过；预测参考前次正式9jobs实际82,251,776B及本次固定约33MB增长，非API/collector cap。08:31余9,245,020,160B约8.61GiB，评价/profile完成后GPU空闲、planner运行，无清理或并行GPU。低磁盘告警不等于完成。[启动](evidence/v6-ddp-pipeline-next-validation-20261031/launch.json)与[唯一监听](evidence/v6-ddp-pipeline-next-validation-20261031/wakeup-setup.json)保留。首次本地launcher文本构造发生PowerShell引号错误及缺文件，均在SSH/远端启动前；改literal here-string后首次真实启动成功，不是实验重试。

原两连续全梯度/更新参数/loss/finite/exact replicas及1e-6/1e-4容差、2warmup/5clean、先rank-max再median/MAD不变。原anchor133.379621→140.04860205不重锚。协议归属和收益因果仍不预断；若正式收益不过噪声，保留拒绝，不追加参数/修复链。原关闭链和全部历史接受/失败保持。
