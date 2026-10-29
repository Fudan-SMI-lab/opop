# FSDP 原样 Agent 诊断脚本的固定双卡验证

已于 2026-10-29 02:04:05 UTC（香港 10:04）启动，冻结 `0b0e9c44a4a913f4e3acb38df6469ff94b4405d3`，controller PID941199，生产 core `ed24af7a…` 不变。预检 19 个冻结文件/引用哈希通过。02:04:45 核验实际 B request 只有声明中的变化、probe SHA 完全相同，四项原通信诊断保留；双卡 worker PID941299/941300 已运行。此时 C 尚未启动，不预判其结果。见[启动回执](evidence/v6-fsdp-diagnostic-validation-20261029/launch.json)与[实际请求核查](evidence/v6-fsdp-diagnostic-validation-20261029/startup-verification.json)。

唯一普通完成/磁盘监听 PID59148 已核实，旧修复监听标记已处理。监听预检确认本批仍运行，未发送测试消息；低于 8 GiB 时通知。启动实际 df 可用 10,516,926,464 bytes（约 9.79 GiB），无本轮清理或 raw 本地转存。见[监听回执](evidence/v6-fsdp-diagnostic-validation-20261029/wakeup-setup.json)。等待期间不调用模型，完成后核查真实结果。

前置[修复审计](v6-fsdp-diagnostic-repair-20261029.md)通过八项独立 CPU 检查。接下来仅执行生成的 probe `9495eead…` 原始字节，不人工改写；不是新 Agent、候选搜索、性能接受或生产 collector 采用。

顺序固定为派生 baseline B、原失败 candidate C，各一次。两者都复用 `fsdp-search-20261028/fsdp-search/jobs/trial-86e14487d0-param-b3a4c7d3-screen/run/request.json` 的源目录、adapter、controls。B 仅将 `parameters.bundles=[]`，明确是此次派生对照，不是历史已测基线；其物化源目录仍带原 staged revision manifest。C 保留原 bundle 和 `keep_unsharded_blocks=6`。两臂只共同替换 entry/function/output/wait 并增加诊断上下文，独立参考字段为 null。无源文件、控制参考复制或 current 指针发布。

工作量保持 seeded GPT2 Medium 24 层/hidden1024，双 rank B1/S128，FP32/eager/noTF32/dropout0，完整 forward/backward/global-mean gradient/SGD，tied embedding/head 和实际 DTensor shard 映射。不同 shard 不要求彼此相等。保留完整两步 loss/logits/gradients/updated parameters、原 `1e-6/1e-4` 容差、2 次预热、有效时 5 个干净样本，以及原四项独立通信诊断各 3 次 profile/rank。原完成边界、返回 logits、512B 源 scalar 生命周期、峰值约束和 max-rank-then-median 均不改。adapter `8b014f6f…` 的历史源所有权修复属于开发者协助，不称无辅助原入口 cohort。

原 evaluator 每 rank 只调用一次。诊断包装原 compare，先得到原结果再对已存在的观测做完整 CPU 路径统计，不插入新的模型调用或通信。每 rank 写小型 `diagnostic-rank-N.json`，不导出多 GB 状态。审计原 worker 返回/异常、两步观测、样本是否到达、源/入口/参考/参数哈希、hook 恢复和缺失数据。worker 正常完成可以同时 `valid=false`；不能将 controller 完成当作质量通过。

观测的摘要无法独立重建完整张量/digest；`top_mismatches` 不等于仅超容差元素。CPU 遍历、拷贝和引用会改变生命周期/同步及步间隔，rank 间观测工作量可能不同。计数只覆盖属性解析的通信 API，第一段包含 prepare/reset/binding，最后累计可能包含预热/采样/profile；不是物理通信成本。所有观测计时仅作诊断，无 AB/BA 接受、无 Nsight 或新候选收益声明。若新运行意外通过或出现新错误，保留结果，不覆盖旧失败、不重复直到通过。

目录 `/root/autodl-tmp/dev/v6-work/fsdp-diagnostic-validation-20261029`；控制器 `scripts/experiments/v6_fsdp_diagnostic_validation_20261029.py`。启动前核实双卡空闲和实际 150 GiB quota；约 9.8 GiB 可用，参考共享，主要增长预计来自 B 原有四项通信 trace。保留原请求的采集，不为缩小空间悄悄关闭。原始文件留服务器，仅小型回执/摘要进 GitHub v6；无本地大体积转存。单个普通完成/磁盘监听，不启用 Goal 或模型轮询。
