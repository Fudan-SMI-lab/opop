# FSDP 原样 Agent 诊断脚本的固定双卡验证

## 终态审计

本批于 2026-10-29 02:05:43 UTC 完成。四个 rank worker 均正常退出，但 C 两个 rank 的原结果仍为 `valid=false`。B 保留每 rank 5 个 clean 样本、两次全量零误差检查；C 第一次通过、第二次在 logits 拒绝，与原 K6 screen 的完整 checks 相同，未进入 clean 或通信 profile。旧拒绝不变，无新性能接受。

原样 probe `9495eead…` 每 rank 调用原入口一次，写出四份约 0.53/0.68 MB 的严格 JSON；非有限编码计数均为 0，compare 和 13 个通信属性 hook 报告恢复。19 个冻结源文件、参考、入口、probe/plan/current 哈希不变；两臂实际 request 仅有声明变化，worker core 仍 `ed24af7a…`。未复制参考，未发布指针。实际日志仅见已有 barrier、loss_type 和 deprecated all_gather API 警告，无 worker traceback。

每次诊断遍历 587 个路径：loss、logits、292 个命名局部分片梯度、292 个更新参数及 tied_embedding。四个 rank/arm 共八次遍历；读取原 CPU controls 核对路径、shape、dtype、元素数和保存的 top 值中 expected 一侧，重算摘要 fingerprint、计数与 top 值差值。**实际完整张量未保存，不能独立重算所有误差或 actual digest**；这不是一次新的独立源一致性验证。

| 观察 | rank 0 | rank 1 |
|---|---:|---:|
| C 第一步 logits / loss 最大差 | 0 / 0 | 0 / 0 |
| C 第一步梯度非精确路径（均容差内） | 292 / 292 | 291 / 292 |
| C 第一步更新参数非精确路径（均容差内） | 244 / 292 | 243 / 292 |
| C 第一步梯度最大差 | 2.2817403e-7 | 1.8253922e-7 |
| C 第二步 logits 最大差 / 超容差元素 | 4.5299530e-6 / 1898 | 4.5299530e-6 / 1592 |

第二步其他观察路径仍在容差内；rank 1 loss 有 9.5367432e-7 的差异，rank 0 loss 精确。tied_embedding 均为 true，未发现结构变化或非有限差值。新证据表明差异在第一步梯度/更新参数已可见，不能把“第二步第一个比较失败”当成最早数值分歧。没有独立控制隔离唯一 GEMM、通信或其他根因。

脚本的 `top_mismatches` 名称有歧义：本次两 rank 各保存的 8 个最大绝对 logits 差值**全部在各自相对容差内**，1898/1592 个真正失败元素是其他值；不能将 top 列表当作失败坐标。B 相同摘要 fingerprint 也不证明不同 step/rank 的原始张量相同；精确路径未加入原始字节，且不同 rank 是不同 shard。

B 四项通信指标各 3 次独立 profile/rank，共 24 份原始 trace 全部重新解析，数值与结果一致；10 个 clean rank 样本全部保留。max-rank-then-median 的 step 为 182.435090 ms，仅诊断。四项各自聚合为 kernel sum 187.068174、union 179.6178423、overlap 44.7258516、without-compute 142.2106323 ms，来自不同 invocations，不能相减或当作 wire/critical-path 成本。C 因质量失败没有这些计时，不补造性能对照。

通信包装记录 B 每次质量步骤 49 all_gather_single / 25 reduce_scatter_single；C 42 / 24 外加 11 all_gather_into_tensor。这是属性 API 计数，别名和内部调用边界不同，不能直接当作物理 collective 数量或优化归因。B 最终计数还覆盖 2 quality + 2 warmup + 5 clean + 12 profile；第一 delta 含 prepare，all_reduce scalar 属准备阶段。before/after 的 cudnn.allow_tf32 从默认 true 到 prepare 设定 false，不是数学策略违反；原 evaluator 的实际 `math_policy_unchanged` 为 true。

`observer_wall_s` 虽放在 full_traversal 内，代码计时从原 compare **之前**开始；不存在两套可分离的测量时钟。该值约 2.75–5.09 秒，不能扣除为“纯 observer 成本”。CPU 遍历/拷贝/保留对象改变步间隔和生命周期，C 重现原 checks 不证明所有内部时序相同。

审计最初错误地把精简 `original_invocation` checks mirror 当成完整 checks 比较；代码明确省略 atol/rtol。该 CPU 审计断言已保留，改为核对实际字段并另核 `compare_calls.original` 的完整结果，没有重跑 worker 或修改 probe。完整审计 SHA `fc0c65d2d3f1c02e7237e2ddad15ff14e768324ffd3fcb5380f314504a391523`，见[公开摘要与索引](evidence/v6-fsdp-diagnostic-validation-20261029/public-audit.json)。原始 trace/日志/摘要仍在服务器。

结论限定为：第二种任务的 Agent 诊断脚本，在一次真实持久化修复后，能保留原质量拒绝并补出比较中断后的逐路径信息。尚未形成通用自动诊断闭环。下一步按[两任务诊断解读与反馈接口实验](v6-diagnostic-interpretation-20261029.md)检验真实 Agent 能否自行整理可靠的小型反馈，不提供上表或开发者原因、不自动采用生产。实际 df 约 9.55 GiB，GPU 空闲，无本轮清理或本地大体积转存。

已于 2026-10-29 02:04:05 UTC（香港 10:04）启动，冻结 `0b0e9c44a4a913f4e3acb38df6469ff94b4405d3`，controller PID941199，生产 core `ed24af7a…` 不变。预检 19 个冻结文件/引用哈希通过。02:04:45 核验实际 B request 只有声明中的变化、probe SHA 完全相同，四项原通信诊断保留；双卡 worker PID941299/941300 已运行。此时 C 尚未启动，不预判其结果。见[启动回执](evidence/v6-fsdp-diagnostic-validation-20261029/launch.json)与[实际请求核查](evidence/v6-fsdp-diagnostic-validation-20261029/startup-verification.json)。

唯一普通完成/磁盘监听 PID59148 已核实，旧修复监听标记已处理。监听预检确认本批仍运行，未发送测试消息；低于 8 GiB 时通知。启动实际 df 可用 10,516,926,464 bytes（约 9.79 GiB），无本轮清理或 raw 本地转存。见[监听回执](evidence/v6-fsdp-diagnostic-validation-20261029/wakeup-setup.json)。等待期间不调用模型，完成后核查真实结果。

前置[修复审计](v6-fsdp-diagnostic-repair-20261029.md)通过八项独立 CPU 检查。接下来仅执行生成的 probe `9495eead…` 原始字节，不人工改写；不是新 Agent、候选搜索、性能接受或生产 collector 采用。

顺序固定为派生 baseline B、原失败 candidate C，各一次。两者都复用 `fsdp-search-20261028/fsdp-search/jobs/trial-86e14487d0-param-b3a4c7d3-screen/run/request.json` 的源目录、adapter、controls。B 仅将 `parameters.bundles=[]`，明确是此次派生对照，不是历史已测基线；其物化源目录仍带原 staged revision manifest。C 保留原 bundle 和 `keep_unsharded_blocks=6`。两臂只共同替换 entry/function/output/wait 并增加诊断上下文，独立参考字段为 null。无源文件、控制参考复制或 current 指针发布。

工作量保持 seeded GPT2 Medium 24 层/hidden1024，双 rank B1/S128，FP32/eager/noTF32/dropout0，完整 forward/backward/global-mean gradient/SGD，tied embedding/head 和实际 DTensor shard 映射。不同 shard 不要求彼此相等。保留完整两步 loss/logits/gradients/updated parameters、原 `1e-6/1e-4` 容差、2 次预热、有效时 5 个干净样本，以及原四项独立通信诊断各 3 次 profile/rank。原完成边界、返回 logits、512B 源 scalar 生命周期、峰值约束和 max-rank-then-median 均不改。adapter `8b014f6f…` 的历史源所有权修复属于开发者协助，不称无辅助原入口 cohort。

原 evaluator 每 rank 只调用一次。诊断包装原 compare，先得到原结果再对已存在的观测做完整 CPU 路径统计，不插入新的模型调用或通信。每 rank 写小型 `diagnostic-rank-N.json`，不导出多 GB 状态。审计原 worker 返回/异常、两步观测、样本是否到达、源/入口/参考/参数哈希、hook 恢复和缺失数据。worker 正常完成可以同时 `valid=false`；不能将 controller 完成当作质量通过。

观测的摘要无法独立重建完整张量/digest；`top_mismatches` 不等于仅超容差元素。CPU 遍历、拷贝和引用会改变生命周期/同步及步间隔，rank 间观测工作量可能不同。计数只覆盖属性解析的通信 API，第一段包含 prepare/reset/binding，最后累计可能包含预热/采样/profile；不是物理通信成本。所有观测计时仅作诊断，无 AB/BA 接受、无 Nsight 或新候选收益声明。若新运行意外通过或出现新错误，保留结果，不覆盖旧失败、不重复直到通过。

目录 `/root/autodl-tmp/dev/v6-work/fsdp-diagnostic-validation-20261029`；控制器 `scripts/experiments/v6_fsdp_diagnostic_validation_20261029.py`。启动前核实双卡空闲和实际 150 GiB quota；约 9.8 GiB 可用，参考共享，主要增长预计来自 B 原有四项通信 trace。保留原请求的采集，不为缩小空间悄悄关闭。原始文件留服务器，仅小型回执/摘要进 GitHub v6；无本地大体积转存。单个普通完成/磁盘监听，不启用 Goal 或模型轮询。
