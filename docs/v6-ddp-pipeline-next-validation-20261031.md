# DDP 新候选原正式配对验证（2026-10-31）

[固定B/C](v6-ddp-pipeline-next-fixed-20261031.md)原质量通过，48.194381→47.633574ms仅观察1.16%改善，因此需要原正式噪声判定。本次**仅一次**现有`ProjectSearch.validate_bundle(publish=False)`，fresh AB再BA；保持原工作量、质量、anchor、noise、约束及diagnostic profiles，固定样本不合并，拒绝后不重试或换参数。

实际启动**2026-10-31 00:29:33.892849UTC / 香港08:29:33**，controller **258723**，唯一隐藏监听 **23068**。使用[通用实验编排](../scripts/experiments/v6_validate_fixed_candidate.py)，实际controller SHA **e06f1a99e1759a73398ac6cd898bb35a472dbe41941ceeb6724cc29ebd004844**；复用既有cb34d3ea只读环境，生产core ed24af7…不变。8当前关键身份，没有旧1947文件遍历、模型健康调用、重复环境副本或Git发布门槛。

源`ddp-pipeline-next-fixed-20261031/ddp-fixed` current0bc仍为已接受02b933；C精确staged **805deb71707914fe2eb9ae52259fa66ae536d8470a6d9e7808bb3911f2feada1**，singlefixed-pipeline-next，代码e610e944…28,517B/proposal4a00833b…9,663B。空参数仍调用自身fine8，没有fine9/coarse5或新实现。fork共享原request/prepared7a49/current discovery/只读controls，无新discovery/source-control/reference副本或发布。publishFalse下current不移动与正式是否接受是不同问题。

保留正常resource_feedback profile及真实普通collection-planner，由其自主选择已有collector，控制器不手选。没有新selector、candidate、参数搜索、独立闭合或生产采用阶段。只有真正的新正式接受才能另行决定完整版本独立闭合。

[08:31:26启动记录](evidence/v6-ddp-pipeline-next-validation-20261031/startup-verification.json) SHA **12a2196734d324f332d951f20f2f847dacb42c0aea23476b955da0d46ca7b4ec**：四份AB/BA实际请求及正常profile均验证原request、entry/adapter、bundle、空参数、controls、设备身份；各两rank结果已返回，但这里未作终态质量/噪声接受结论。真实planner generation-7cf8aa3c635c/session ses_e70a1e048ffeLk2OXOm1HxDSXF，8实际输入101,879B与snapshot逐字节相等、core相等，完整工具/消费/collector结果待终态核对。最初目录查找仅限trials未找到snapshot，扩大到本branch实际目录即可；没有重跑角色。

启动150GiB盘余9,316,581,376B，512MiB预测+8GiB告警通过；预测参考前次正式9jobs实际82,251,776B及本次固定约33MB增长，非API/collector cap。08:31余9,245,020,160B约8.61GiB，评价/profile完成后GPU空闲、planner运行，无清理或并行GPU。低磁盘告警不等于完成。[启动](evidence/v6-ddp-pipeline-next-validation-20261031/launch.json)与[唯一监听](evidence/v6-ddp-pipeline-next-validation-20261031/wakeup-setup.json)保留。首次本地launcher文本构造发生PowerShell引号错误及缺文件，均在SSH/远端启动前；改literal here-string后首次真实启动成功，不是实验重试。

原两连续全梯度/更新参数/loss/finite/exact replicas及1e-6/1e-4容差、2warmup/5clean、先rank-max再median/MAD不变。原anchor133.379621→140.04860205不重锚。协议归属和收益因果仍不预断；若正式收益不过噪声，保留拒绝，不追加参数/修复链。原关闭链和全部历史接受/失败保持。
