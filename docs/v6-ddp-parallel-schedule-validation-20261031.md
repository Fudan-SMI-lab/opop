# 分片排程候选的原正式配对验证

香港2026-10-31 **19:32:38** 实际启动，controller **297985**；唯一隐藏完成/磁盘监听 **7304**。现有通用 `v6_validate_fixed_candidate.py` SHA96586fdc2a611253b32b5a5e63ceb49d63132a63db78bb7cfb3c014cd51229da调用一次 `ProjectSearch.validate_bundle(publish=False)`，fresh AB然后BA。工作runtime/core e893与刚完成fixed一致；没有新框架、prompt、模型、测量或接受改动。

依据[原固定B/C终审](v6-ddp-parallel-schedule-fixed-20261031.md)：原fullchecks/精确副本/约束通过，45.521059→44.334140ms观察2.6074064%，且实际11次分片SUM与更新等待机制执行。这个观察值得一次正式噪声判断；不将模拟预测当收益、不混入fixed样本。

来源 `ddp-parallel-schedule-fixed-20261031/ddp-fixed` CURRENT eb8cd905（9bdd，10calls/9groups）。C精确staged7401459ffc71e929f71ed98e643774d7c23902c3dc2f6abb3ede455990b479ce / single fixed-parallel-schedule，code7e6fb7a723590a93b289f43fcfef84663fc618b5da6d559362144d178827eed3 /35595B，proposal442ac3f39dc9d13f8591f534be804b42b77b625c933a184038de64200a241403 /11159B。params{}调用自身fine8/comm_groups2/tail_splits4；requires[]/replaces fixed-pipeline-parallel。trial-parallel-schedule-validation。不改代码或试其他groups/tail_splits/chunk_plan。

原2rank/FP32/GPT2完整工作负载、2连续full gradients/updated params/loss finite/exact replicas（atol1e-6/rtol1e-4）、2warmup/5clean、先aligned rankmax后median/MAD、原anchor133.379621/limit140.04860205保持。existing fork共享request/prepared EMPTY7a49/discovery/只读controls，不新跑source/discovery/reference副本。正常resource_feedback profile和真实普通collection-planner自主选现有collectors，没有controller选collector。API observe-only无cap，所有失败保留。source/newbranch current仍eb8，无独立阶段或指针发布；仅真正新formal接受后才另行声明必要完整版本独立闭合。

**19:34:17启动实核** [d51fcd81](evidence/v6-ddp-parallel-schedule-validation-20261031/startup-verification.json)：8当前关键身份、四个实际ABBA请求及profile的原request/entry/adapter/controls/UUID、精确9bdd/7e6代码和空参数核对；已有10rank结果文件，只代表文件存在，完整数值/噪声/接受仍待终态核验。此时尚无planner snapshot，不预称其输入/访问。GPU已结束profile为空闲，controller阶段仍运行，df8,986,644,480B。没有historical sweep/modelhealth/新测试。

**19:36:42新增planner实核** [08b1df7d](evidence/v6-ddp-parallel-schedule-validation-20261031/startup-planner-supplement.json)：真实generation9f846b9ebd6d/session ses_e6e42c703ffeqfgZDtZcpNq2FZ，8实际初始输入108,595B逐字节相等，原prompt/model/tools/core保持；revisions正确显示parent fixed-pipeline-parallel被fixed-parallel-schedule替换、无retained bundle。只核新出现的投递，未重复8关键身份；完整实际工具/输出/探测、消费和最终噪声结论待审。GPU空闲处于planner阶段，controller仍运行，df8,983,552,000B。

启动前必要容量处理在GPU全部结束后执行：香港19:32:08–13，仅18份已关闭且正式拒绝trial02ed5436b4的screen/pair-C原metric JSON，从292,225,518B无损冷存到24,872,204B；所有roundtrip hashes验证，1434其他文件metadata保持。所在旧目录虽名ddp-accepted-communication-trial，选中的候选decision实际拒绝；接受baseline/source/materializations/controls/requests/decisions/native/current/model/provider/deps均未改。服务器RESTORE：`/root/autodl-tmp/dev/storage-audit-20261031/ddp-schedule-validation/RESTORE.md`；journal207338be7904eb7869a727c53749f0eaec4682a712b72ded8ac687f8656569c4、planbcd934a9ac327e8b95545064028d634be2bc6647c24d1fb0fb08b003b38e246f。旧journals保持，cold数据未丢失。没有raw本地转移/归档/旧storage controller重跑。

启动150GiB盘余9,058,074,624B约8.44GiB；384MiB增长预算依据近两formal完整root213,245,952/210,182,144B，仍保留8GiB警戒，非API/collector cap。low_disk_space不是完成，不盲杀/删；如有警告先看workers，结束后只按授权保护当前数据做必要无损cold。旧45380/41700/32992/39344/45968均handled、迟到事件去重；fixed短批无监听/无排队事件。

候选原审2f788497、fixed1159b695已完成，不重审旧批次。继承copied inputs/有限guards/shared pool/setup hooks/异常后只恢复grad/PG生命周期/rank非对称等条件性限制保留，不额外故障注入或修复链。原质量首parent次setup并非两个optimized-only检查；先前source比较只hash复用，不新声称通用正确性。NCCL sums/区间重叠不当wire/关键路径或跨capture clean差值。正式结果返回后按真实收益/错误选择下一步；不重试直到通过，不把辅助工作当项目终点。
