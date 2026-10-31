# 分片排程接受版本的正常通信采集

**19:55:28终态已到达**：batch_finished/completed，discovery和preservation结果已写入；唯一监听53652已成功queued完成事件（returncode0）。完整新planner工具/原始trace/native数值审计由该事件续接，当前不预称终态数据或新接受已核验，不重复创建listener。终态观察df8769634304B约8.17GiB，GPU为空。

香港2026-10-31 **19:51:28** 已启动一次原 `ProjectSearch.discover`，controller **304369**，唯一隐藏监听 **53652**。使用[新独立接受版本](v6-ddp-parallel-schedule-independent-20261031.md) **c283f6a55dc1f758fa0fb9575f757b27384b5bb5b381592251c6fa4808c5f6a6**，源 `ddp-parallel-schedule-independent-20261031/ddp-independent` CURRENT仍原始EMPTY7a49，explicit fork accepted exploration。新root `ddp-schedule-communication-evidence-20261031` 仅附加当前registry/plan/discovery，文件/bundle/contract保持。

理由是新接受7e6fb7仍继承旧9bdd的10call/9group registry；正式profile虽也是10call/9group，并没有把新代码registry接上。需要实际新版本调用与通信记录供后续优化使用，不构建分析器、不自动预排candidate。原fixed-parallel-schedule/params{}、自身fine8/2groups/4tail片、code7e6fb7和proposal442ac3f保持，旧物理文件不是多个active bundles。source/formal7401/旧eb8/b769/187c及其他既有指针不变。

现有通用 [v6_discover_accepted_version.py](../scripts/experiments/v6_discover_accepted_version.py) SHA62d4e9bf967d2a90c828304d58f7a164fd5f2b47cf328e95e3d8fa65ea35afa1，直接复用working runtime core e893。六当前关键文件，不复查历史1947hash、不复制runtime、不做model health/额外测试。普通initial collection-planner使用原request/available collectors/new observations/coverage/schema/environment自主选择现有collectors；无开发者审计表、手选collector或算法。API observe-only无cap，普通prompt/model/tools保持，所有错误保留。profile/run-only不是新增全量质量/clean/接受。

**19:53:19实际启动核对** [1ae30303](evidence/v6-ddp-schedule-communication-evidence-20261031/startup-verification.json)：discover5f47565d3a两rank结束，新current **88fa1d104cd1777b723521cfd9bb37540abfcefbb9a5edb3a04ca3cc21f67057** 与原7e6文件/bundle/contract一致，源CURRENT EMPTY7a49、六当前身份保持。真实planner **7e627ef4afe5/session ses_e6e32d78cffekUWW2Ogv5gU7B0**，六实际输入108641B与snapshot逐字节相等，普通初始prompt/model/tools相同、无previous-feedback suffix。fork自动复制原version-df2217162ef2反馈存在，但不在初始六输入内；后续外部访问待终态完整审计，非封闭FS，不能预称无读取。完整工具/输出/探测、新collector语义尚待终态核对。此时GPU空闲、controller planner阶段仍运行，df8793882624B约8.19GiB。

容量所需无损cold仅在GPU结束后 **19:50:54–56** 实施：4份旧正式拒绝02ed C的profile/timeline原trace加12份拒绝e610 C metrics，105566660→8590190B，释放96976470B。roundtrip全部校验，1914其他文件metadata保持；接受/来源/基线/controls/materialization/native/current/model/provider/deps均未修改。服务器RESTORE `/root/autodl-tmp/dev/storage-audit-20261031/ddp-schedule-evidence/RESTORE.md`，journal7a62edaf77df19c3b498f2de608369797a27f0a9fb9164783df6eb870979a9d4、plane06e000b5dbd87a35b60ce27b5e84e2aef8e96e762d72c222853b3981677ea17；旧journal独立保留，cold不是丢失，无raw本地转移。

本批192MiB增长预测依据实际三个完整evidence目录133746688/140967936/139771904B；8GiB警戒不变，非API/collector cap。启动df8810954752B约8.21GiB。low_disk_space不是完成，先看workers；不盲杀/删，必要容量仅按授权在GPU结束后保护当前数据做lossless cold。局部launcher builder的PowerShell/python-c引号SyntaxError发生在脚本创建/SSH/远端启动前，改成文件builder后启动一次，无实验重启。

旧formal7304已处理、迟到事件去重；独立短批未建listener，其他旧watcher不重启。完成后按实际当前registry、普通planner工具访问及真实collector结果选择下一有用优化步骤，不把NCCL sum/重叠或跨capture差值当wire/关键路径，不重复closed角色/参数/修复链。原质量/性能门槛、既有辅助输入与代码条件性限制保持。
