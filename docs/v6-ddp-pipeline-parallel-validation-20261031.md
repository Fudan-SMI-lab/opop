# 并发通信候选的正式配对验证

香港2026-10-31 **18:20:43–18:25:23**，controller282491，原 `ProjectSearch.validate_bundle(publish=False)` 接受精确9bdd完整替换版本 `187ccaf61c7737bcb09ffe3b856ff31cc070ef6cbe20bc668bfa158869875986`。源与新branch CURRENT仍为0bc/已接受02b933；accepted exploration与发布指针分开。候选代码、proposal、空参数、自身fine8/两个通信组默认值保持，无参数搜索或手工修改。

[完整新批次审计摘要](evidence/v6-ddp-pipeline-parallel-validation-20261031/public-terminal-audit.json)：服务器audit SHA `6c3a49ad4d1077e708514ff20b9ae797d45038b8edadd3d9d1fd113aec0c067a`，public `d0518cd421a62be559397b58120b2bcf1466cd8e7fc4b5cea645358669588cb0`。9jobs/18ranks、40计时样本、16全量检查、24自身metric trace、2份新native、8个当前关键文件均核对；原请求、adapter、GPU、controls、完整质量、约束及噪声规则保持。

| pair | B中位数ms | C中位数ms | 收益ms | 噪声界限ms |
|---|---:|---:|---:|---:|
| AB | 48.055876 | 45.229003 | 2.826873 | .173532 |
| BA | 48.089818 | 45.453308 | 2.636510 | .342140 |

两对均通过，汇总 **48.072847→45.3411555ms，改善5.6824000875%**，对比对象是已接受02b933。原133.379621→140.04860205约束通过，尾部B48.332631/C45.690359保留，不与固定B/C或独立结果混合。八次首次检查误差0，八次第二检查r0最大9.5367431640625e-7/r1最大1.1920928955078125e-7，原容差与exact replicas通过。质量首次parent、第二次graph setup/reference1，不能称两次优化检查；没有新source比较或全量tensor留存。

每份metric trace B1138kernel/1122Graph/group0，C1138/1130Graph/groups1、2，两臂17Graph launch、8次Graph外AllReduce、相同497759232B。C实际通信重叠17.914697–19.523385ms，B为0（舍入范围内）；这些是同capture描述，不是wire/关键路径/收益因果分解。Torch profile产生10calls/9groups，但没有更新current registry。

真实新Nsys `nsys-0fc3bbc4da` 每rank1128kernel/1120Graph、17个`cudaGraphLaunch_v10000`、8次Graph外通信，streams21:3/25:5；实际日志channels00/01为SHM/direct/direct。SQLite SHA r0 `6a4d118702c6a4500dddc070ed037bb0816b49288326d3c993e8155e14d3cf5c`，r1 `639d22bbf8d9325dd5ce4643b4956fd2600e75752ac7f31556de63a96c574385`，normalized `9055b45d3181b3b94edab3f497fee0e5d46ecfd7714cf098025e96802bfd8d9b`。NCCL sum503.576380/51.862826ms、union259.435006/33.913997ms来自被观察扰动的不同rank周期，长224/233ms尾部保留，不当clean或wire。RING_LL入口不证明逐事件协议；另一次tuning记录包括组预热64B/1MiB和分段、flags/setup，不能按payload跨周期拼接。

Timeline两rank46.638436/46.697853ms、8AR/67sync/0gradient callbacks/1bucket；readiness未知，不是无梯度或完整chunk归属。run-only诊断6个正常False生成器位置共458returns/rank，无异常，不是observe比较。没有本批calibration。Torch内存约4.613GB/runtime、5.656GB/lifecycle、6.354GBreserved，外部NCCL不完全覆盖，无内存收益或新约束。原警告保留，无未捕获traceback；6.86–9.41ms安装不等于完整setup。

普通collection-planner `553f50a8d2d8` / `ses_e6e84b570ffeevkqG2GH0yvFiu`：9assistant/13tools/0schema修复，provider .15197475286仅估计。8实际输入102117B逐字节核对，2540展示行核对；tool8字节截断后tool11读续篇，2008行observations及全部8输入完整暴露。tool7读取真实parent_bundles/added/removed/retained，正确排除非加法替换的binding_cost。四collector由角色选择。tool1/2仅列目录，12一次plan写入无编辑，13仅JSON/schema；完整工具记录未见模型/tensor/CUDA/PG/collective/私人配置或后续答案读取，非封闭FS不作绝对缺失声明。计划把跨周期54ms sum与45ms clean比较推断等待仍不成立，原plan不改。

本批两臂均使用工作runtime **e893f850…**；固定B/C使用只读cb34/core ed24，分别记录，不合并分数。e893只包含此前完成的通用真实父bundle构成输入修正，原prompt/model/采集器/质量/计时/接受规则不变。此前12项测试复用，没有重复测试。通用实验helper记录两个core来源，移除旧core必须相同的版本门槛；没有修改旧冻结runtime或增加case分支。[来源记录](evidence/v6-ddp-pipeline-parallel-validation-20261031/framework-update-reuse.json)。实际当前身份/规划输入启动核验 `ed6f7de9…` 在批次已结束后完成，不当作质量证明。

正式监听45968已handled，排队晚到事件去重。已实际衔接[一次独立完整版本验证](v6-ddp-pipeline-parallel-independent-20261031.md)，不重试已拒绝e610、参数或修复链。
