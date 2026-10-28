# FSDP 三轮自主搜索与独立完整版本检查

前置[源与生命周期修复审计](v6-fsdp-owner-repair-20261028.md)已通过，修复后的 adapter SHA `8b014f6f…`。该 baseline 经开发者实测反馈辅助修复，后续搜索自主执行；本批不称为新的无辅助原入口 cohort。生产 core `ed24af7a…`、prompt、运行顺序与接受规则不变。

从 `fsdp-owner-repair-20261028/fsdp-repaired` 的空 bundle baseline 创建隔离分支；保留原 source/request/contract/质量/完整工作/显存<=1.05 baseline 约束。共享只读 controls 路径，不重新写多 GB 参考。源质量、修复代码和生命周期在本轮已完成独立核查，不把这些开发者解释作为搜索提示。

固定执行：

1. 框架重新 discovery、Agent 采集规划、目标选范围、候选生成与参数执行、完整质量/约束/配对判据、真实反馈，共三轮。没有人工给组、候选、通信优化答案或旧 bundle。检查实际路径和 fallback，接受后按现有机制重发现/组合，不预设必须生成某种 kernel 或必须获得接受。
2. 若存在 accepted current version，在另一隔离 baseline 分支执行完整版本独立 AB/BA；不改搜索接受结果，不因独立拒绝再抽样到通过。所有质量、覆盖、样本、原始显存约束原样保留。若无版本则明确 not_run。
3. 对存在的 current version 和 baseline 各执行一次独立 nsys 原生路径检查，不把 profiler 耗时纳入 clean 分数；即使独立拒绝，也可用它解释实际执行。通信指标仍通过原来四个显式次级映射分开采集，各自 max rank 后 median。四列不来自同一组 trace，不能交叉相减当物理分解；NCCL sum/union 不等于 wire time/暴露关键路径。

脚本 `scripts/experiments/v6_fsdp_search_20261028.py`，目录 `/root/autodl-tmp/dev/v6-work/fsdp-search-20261028`。启动前约 16.58 GiB，可用空间按实际容器 df 记录；GPU 串行、只运行这一任务，不开额外 Qwen 或大 trace 批次。原始数据留服务器，不本地转存、不清除冻结证据。共享参考避免新增第二套大文件，后续若空间不足需保留现场并基于真实占用处理。

旧 Qwen/ResNet/DDP/TP 的所有接受、拒绝与全部样本均不改变；不采用旧 allocator 合并或 reviewer 建议。真实 API 用量仅统计，不加开发调用配额。每阶段代码、协议与小摘要推送 GitHub v6；一个普通完成监听在终态提交续行，等待不调用模型，无 Goal 或额外投递测试。

启动：冻结 `aa5d497ec9dd54e5abf0daff1bc287bd14499390`，2026-10-28 04:42:52 UTC、controller PID523981；[launch](evidence/v6-fsdp-search-20261028/launch.json)、[protocol](evidence/v6-fsdp-search-20261028/protocol.json)。实际 df 可用 17,794,297,856 bytes（16.57 GiB）。已核实新分支 adapter SHA 不变、controls_path 与修复 baseline 完全相同，未生成另一套参考。唯一普通完成监听 PID33456、waiting，控制器存活，[启动核验](evidence/v6-fsdp-search-20261028/wakeup-setup.json)保存。当前只确认搜索流程启动，尚无本批性能结果。
