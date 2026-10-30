# 通信研究的有界任务盘点与下一步

2026-10-31用户明确要求继续项目，因此“等待研究范围偏好”不再作为当前阻塞条件。现从已有真实DDP及新增合同证据推进[一次普通项目级选择](v6-ddp-communication-redecision-20261031.md)，由现有selector自主重新判断路线；没有编造新任务或重新开启77ac修复。以下盘点数据及历史状态保持。

2026-10-30 14:54:54UTC完成一次只读盘点，[公开记录](evidence/v6-communication-request-inventory-20261030/inventory.json) SHA `09d773107ed405bd60117cd5e6b8da8c8ad6f05ce5a62b31fc6bf5822b1e5405`，服务器完整清单SHA `9fbcd19b77a6720c88897764b02b88b5bf2dc709514494f8c5730b8dff301103`。这是当前DDP合同修复终态后的明确下一步骤，不是新候选、GPU/collective/model或collector执行。

扫描范围仅 `/root/autodl-tmp/dev/v6-work/*/request.json` 与 `*/*/request.json`，筛选双设备、具有source_root/goal/objective/quality/workload的项目请求。共61份；另19份worker/诊断请求不当作新的真实项目。没有解压冷文件、读取私有provider或复制reference。结论仅适用于这次范围，不声称服务器所有位置或外部仓库都没有其他项目。

| 源入口 | 请求数 | 当前两UUID且prepared | 实际源文件SHA |
|---|---:|---:|---|
| gpt2_ddp_training.py | 48 | 6 | 37d1632436c8f927a52401b6d1fb9e63bd7b8d33d643d0f914916bb89b3364fe |
| train.py（既有FSDP） | 13 | 2 | 6c598dd52cae3decb082f290d36fd0bb157b1ce3ef5966b29be5477d37e1fb1c |

这些是同两类原始项目的分支/目标变体，不能把数十份请求当作数十个独立泛化样本。当前DDP77ac仍有合同反例，不升级到GPU或再修到通过；最新FSDP拒绝不重测，Qwen链不恢复。旧accepted DDP独立36.1578233%仍属计算调度收益，13 AllReduce/497,759,232B不变，不重标为通信接受。

下一步优先**接入一份新的真实分布式任务并完成预检**：固定真实源码/入口、原始工作负载、完整质量/约束、当前设备和空间预算，再由原通用流程理解项目和选择范围。不由开发者指定bucket/算法/实现，不通过修改种子、删工作或换名称制造“新任务”。完整source/controls准备完成前不启动候选或性能接受。若用户选择继续现有DDP，则先单独声明待回答的证据问题；本轮不会自动再生一次相同修复候选或重跑旧测试直到通过。

已通过异步问题询问上述研究范围偏好；当前没有新任务的源码与原始请求，故尚未启动新实验或监听。这是明确的输入待定状态，不是整个研究完成。当前两卡空闲、150GiB配额余8,908,967,936B；本步骤没有清理/生产变更。后续新长任务仍须一次声明、串行GPU、一个隐藏completion/disk watcher，以及完整质量/原约束/独立整步收益和实际通信机制证据。
