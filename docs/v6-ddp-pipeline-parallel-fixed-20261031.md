# 并发通信候选的一次原始 B/C 验证

香港2026-10-31 **18:15:48–18:16:31**完成，controller281428。精确 Agent 代码 `9bddb13647eac850b561c1707640e75255f5e9dabf29e1bf241e83a090b5185f`、proposal `94d72e386e9bd229da73f6c46934c352103cf971e80c26b2a588d2eefcfa3cec`，空参数调用代码自己的 fine8/两个通信组默认值，没有人工选分组或改代码。B是已接受02b933，C为单一fixed-pipeline-parallel替换，staged `187ccaf61c7737bcb09ffe3b856ff31cc070ef6cbe20bc668bfa158869875986`。复用只读cb34/core ed24及原controls，9个当前关键文件保持；无新Agent、collector、native、source-control或reference副本。

[终审](evidence/v6-ddp-pipeline-parallel-fixed-20261031/public-terminal-audit.json)对应服务器完整审计 `d078b194908fd9df0ebf1edf6f3fc0b77bf611765debb1a7442f9b439cd69d38`，public SHA `3312811a35c7e5c977b95cea3775714ea314655e5018a9f1ce7d3a79c13d4ed0`。4rank、20计时样本、8次各248879620元素的全量检查、12份自身metric trace均核对。四次首次检查误差0，四次第二检查r0最大9.5367431640625e-7、r1最大1.1920928955078125e-7，原容差和exact replicas通过。实际质量路径仍是首次parent、第二次setup，不称两次优化路径检查；没有保存所有原始tensor或新source比较。

| 项目 | B | C |
|---|---:|---:|
| aligned rankmax样本(ms) | 48.698021 / 48.288781 / 48.291497 / 48.247040 / 48.191589 | 45.584277 / 45.602913 / 45.489269 / 45.310706 / 45.284873 |
| 中位数(ms) | 48.288781 | 45.489269 |
| MAD(ms) | .041741 | .113644 |

观察改善5.7974377113%，原133.379621→140.04860205约束通过。单次B/C不是正式噪声接受，不与后续AB/BA合并。

每份trace均8次AllReduce、17次Graph launch，总payload `6144+6×56702976+157535232=497759232B`不变。B为1138kernel/1122Graph、group0/stream17；C为1138/1130Graph、groups1/2、streams21/25，各3/5次通信，全部通信在Graph外。C同capture NCCL sum-minus-union为18.415434–19.609981ms，确认实际通信区间重叠；它不等于wire时间、关键路径或可回收收益。B重叠为0（浮点舍入范围内）。没有本批Nsys或物理transport证据。C Torch runtime peak约4.614GB、lifecycle约5.657GB、reserved6.354GB，未覆盖所有NCCL外部分配，不宣称内存收益。

原警告和尾部保留，没有未捕获traceback。安装8–9ms不是完整setup/capture成本。首次审计通过，无工作负载重跑。根目录约124MB，384MiB预测+8GiB告警按近期完整固定批次实际大小安排；无清理。批次在监听预检前结束，**没有固定批次listener或排队完成消息**。

决定并已执行：[一次原规则正式AB/BA](v6-ddp-pipeline-parallel-validation-20261031.md)，不调参数或追加修复。
