# 当前设备 Qwen 新接受版本的独立收尾

终态2026-10-29 23:37:02 UTC，冻结1c29b474，控制器69576。四个fresh worker完整20样本/8零误差检查、543冻结文件及请求/source/adapter/bundle/entry/core核验通过。完整审计SHA `d1440c54a8fba6cc14d3cfc5652f9c6301d4e8613c60c5ee2a1d57b37c1f6863`，见[小摘要](evidence/v6-qwen-cache-independent-20261030/terminal-audit.json)。

独立接受 **721.4966495→374.1322145 ms，48.1449824%**。B为746.616983/696.376316，C为374.184622/374.079807；两对收益372.432361/322.296509均超过噪声3.065312/1.212192。C TTFT20.446481/20.379104小于原24.45004275；B首次TTFT24.998813超限保留，不重设锚。探索版本52668e91，publish=False；独立current仍009b，搜索接受current2a42a6ad及旧空parent2a42b9b1不改。搜索48.0946416%与本结果分别列出，不和旧设备47.325%叠加。

只重哈希复用同代码/当前设备原native：32 GraphLaunch、66,056 Graph kernels，未新开profile；不同采样期不作关键路径/host-cost推断。人工源映射/事实辅助及额外owner边界仍保留，无生产采用。Qwen必要收尾完成，后续主线为[当前设备FSDP](v6-fsdp-communication-device-20261030.md)，不自动增加Qwen候选。旧39280监听已处理。

本批是用户转向通信主线前的必要收尾。对[约束反馈搜索](v6-qwen-cache-constraint-feedback-20261030.md)实际接受的 `2a42a6ad`、精确代码 `dbec45af8e995ca52c812806a856b4b85e303e7fc9f808f49086362c45dbdad8`、单 bundle `trial-22524f5ab8-param-a986a194`、空参数，使用现有 `promote_version(publish=False)` 执行一次完整 B/C/C/B。不是新 Agent、搜索、参数选择或生产采用；任何拒绝原样保留，不重测到通过。

新隔离根为 `/root/autodl-tmp/dev/v6-work/qwen-cache-independent-20261030`，基线为当前设备原 prepared `009b4178`，独立分支 current 始终留在该基线；原搜索接受分支和旧空分支 `2a42b9b1` 均不发布或改写。完整源文件、合同、adapter、bundle 顺序、原请求、共享只读 controls、生产 core `ed24af7a` 分别核验。双同步、B1/prompt256/32 greedy、BF16/SDPA/noTF32、完整 logits .02/.02 和精确 tokens、连续两次质量、两次 warmup、每端点五个 clean 样本不变。当前 TTFT anchor23.285755、limit24.45004275ms，不使用旧设备的25.2536046/25.49943795，也不与旧47.325%叠加。

`source-evidence-reuse.json` 只重哈希当前设备 `qwen-device-refresh-fixed-20261030/source-check.json` 及其 source/controls/adapter，复用已审阅的两次完整源比较和 token/logit negatives；不是新 CPU tensor 比较、GPU参考运行或引用复制。外部人工 generated_tokens/step_logits 映射、历史事实辅助和额外 owner 边界继续明示。

新候选在本轮真实 planner 中已有当前设备 Nsight。原 SQLite `c92309b0…` 与 normalized `6f4d6377…` 已逐行重算：66,089 kernels，其中66,056 Graph kernels，32 Graph launch（prefill2134 +31×decode2062），2304 Flash 全在 Graph。独立批仅哈希复用，不再采第二份 C 或 B trace；它是另一个扰动采样期，不能充当 clean score、孤立 host/critical-path 代价或任意并发 owner 证明。

控制器只产生四个小 evaluate 结果，store 内容同盘 hardlink、controls 共享；预留额外32MiB结果预算，要求启动时仍高于8GiB警戒线加该预算。没有本批清理、Nsys、discovery、大参考复制、模型轮询或投递测试。新长工作只允许一个显式隐藏窗口的普通完成/磁盘监听；若在监听预检前已结束，直接审计，不创建队列。原41852已处理，不重启。

收尾后按[通信优先计划](v6-communication-priority-20261030.md)执行当前设备 FSDP 原空基线准备和一轮普通真实通信任务，DDP随后。不得自动延长 Qwen 链、重开提示/解释器/校验者研究或将独立结论冒充无辅助原入口成功。
