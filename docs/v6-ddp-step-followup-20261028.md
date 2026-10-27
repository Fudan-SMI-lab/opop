# DDP冻结路径补核与新原入口整步目标

终态审计：四阶段均正常完成。新原入口任务的最终完整版本独立 AB/BA 通过，`step_ms` 从 118.1469 降至 70.1677 ms（40.61%）。这是新生成 adapter 的新任务，与先前同 adapter 批次的 55.73% 收益分别记录，不能将两次百分比当作配对性能比较。完整审计见 [audit.json](evidence/v6-ddp-step-followup-20261028/audit.json)，原始证据仍在服务器。

## 固定旧候选的路径和通信补核

旧批次候选 `3c0b8d62…` 的真实 discovery：每个 rank 的 GPT2LMHeadModel 从四次 `[1,128]` 调用变为一次 `[4,128]`；call sites 从 3,865 降到 1,102，原生 kernel 从 3,999 降到 1,160，实际执行拼接 kernel。代码保留逐 chunk loss 权重与一次 SGD 更新。两臂每 rank 都是 13 次 NCCL、497,759,232 logical input bytes；没有降低梯度通信量。原生 discovery 有明显扰动，不能拿其中的长 NCCL 时段证明原 clean 测量的等待原因。

单独 B/C/C/B 诊断全部通过质量与覆盖检查，各保留 5 个 clean 样本及 3 个通信 profile 样本；先对齐 rank 取最大，再取中位数：

| 端点 | clean step 中位数 ms | 通信 kernel sum 中位数 ms |
|---|---:|---:|
| B0 | 113.5590 | 49.4692 |
| C1 | 49.9295 | 35.5083 |
| C2 | 49.7551 | 37.0498 |
| B3 | 109.8907 | 38.2410 |

C2 的首个 clean 样本 92.5161 ms 完整保留。通信 baseline 明显漂移；仅描述这组诊断，不作新的通信接受结论。逐 raw trace 重算通信 sum 与记录一致，同臂参数/源文件一致、B/C 原项目文件一致。此次显式替换诊断 request 的 objective.source 才取得通信样本，原 step-goal 正式端点没有该指标，不回填旧端点。NCCL sum 包含等待，不是 wire time 或暴露关键路径。

## 新原入口三轮与独立核查

初始输入只有原项目、命令、目标及通用环境/schema；真实 intake 会话 `ses_e80a6129bffezszCtH2cSpppCe`，快照 `generation-03a5c9d61b0a`。未提供旧 adapter、算子组、bundle 或开发诊断。新 adapter SHA `b276a7ed5524fc0c47d6080a6349a0a65b9e14bda8270cbaf87d6cd81bceb193`，baseline `7a49c637…` 无 bundle。源代码和合同保留 world=2、每 rank 四个微批次、FP32/noTF32、完整梯度/参数/损失、精确 rank 梯度关系、一次 SGD、最终同步及原 baseline 1.05 倍整步约束。代码实际一直使用 eager attention；intake 将 attention 实现选择列为可替换细节，因此不能声称生成合同绝对冻结了该选项。

| 轮次 | 实际执行与决定 |
|---|---|
| `11e4128fb3` | 搜索配对接受，但全部质量/诊断路径是 eager fallback。CUDA capture 遇到非 pinned CPU→CUDA 标量复制错误，Graph 未执行。小幅收益不能归因为 Graph；额外 warmup 的影响仅是解释假设。 |
| `f1097548f9` | 反馈后的 mask 修复消除了第一错误，随后 autograd 的 stale default-stream 引用使 capture 失败；120.3725 ms 超过 114.0241 ms 原始限制，筛选拒绝。 |
| `2401c0bc3a` | Agent 读取反馈并修改 side-stream warmup/capture；Graph build/replay 实际成功，搜索配对接受。 |

第一轮接受后实际重新 discovery（`1a0eb351…`→`d3b4c6ba…`），后续替换完整 invocation bundle；最终 `09b71766…` 只有新 bundle `trial-2401c0bc3a-param-e2183894`。这是同一调用的版本替换，不能称作多个 active bundle 共存。最后一轮之后没有再次 discovery。第二、三轮真实工具记录包含 previous-feedback 读取，第三轮的独立 Nsight 证据每 rank 3,469 个 kernel 中 3,450 个带 Graph 节点标记、13 个 NCCL 均在 Graph 外；不能仅凭 route 标签或空 framework fallback 计数作结论。

最终候选将四个原微批次的 forward/autograd 梯度累积捕获到 Graph，保留 eager bucket allreduce 和 SGD Graph、完整同步。输入张量固定且 reset 原地恢复权重；没有缓存参考答案。框架质量比较保持原容差，连续两步各比较 248,879,620 个元素，独立 AB/BA 候选均为零误差。候选内部 build 检查只是决定是否回退，不替代正式质量判据。该策略依赖当前固定输入/存储；代码中的运行期失败回退仍可能涉及状态恢复，不能从未触发的分支推广为任意动态任务的保证。

源一致性另用原项目串行 global batch 两步轨迹比较新 baseline 的全量 loss/梯度/参数；此次字段/容器直接匹配，无需修复 fixture，全部通过。关系负例在 rank 1 的 `gpt2.transformer.ln_f.bias` 注入实际约 `2.500019e-7`，仍在数值容差内，但精确 rank 关系断言成功拒绝。

独立完整版本 `6d0eebee…` 相对于新 baseline 重装最终 bundle；四端点的全部 5 样本保留，B 中位数 111.8550/124.4388 ms、C 70.0954/70.2400 ms。两对收益都超过噪声界，候选满足 114.0241 ms 约束，最终 40.61% 收益通过；没有外部发布动作。Agent 对通信带宽上限、device occupancy、暴露尾部和可优化边界的部分解释超出了证据，本文不采纳这些推断。

## 下一步与保留项

这批证明原入口能自主生成 adapter、选择整个项目执行计划、根据真实反馈修复候选，并取得独立全版本收益。已确认的框架缺口是：`required_observations` 只检查 discovery 能力；当主目标改为 step_ms 时，不会自动产生通信评测样本。下一步为显式、可选、由采集器注册表解析的次级诊断指标接口；在最小真实测试与真实 intake 对照后再决定生产采用。不通过指标名称硬编码路由、不扩大反馈上下文，不重复 review 直到通过。

终态数据盘约剩 37.66 GiB。保留旧 DDP 通信拒绝、旧 Qwen 拒绝及全部 80 样本、TP 完整 32-token 失败、ResNet 独立拒绝及全部 80 样本，以及另一个 memory adapter 的 36.68% 冻结接受/缺少 images allocation 事实。allocator 合并仍未采用，不进行本地大体积转存。

承接[整步目标55.73%独立收益的审计](v6-ddp-step-goal-20261028.md)。本批保持生产源码、提示、GC及接受规则，补齐已证明存在的信息缺口，再检验完整原入口能力。

固定部分使用`ddp-step-goal-20261028/ddp-step`的baseline与最终候选`3c0b8d62…`，不改其代码、参数或旧结果：

- 单独B/C原生discovery trace，核对实际模型调用形状/次数、当前输入、NCCL执行及原生kernel路径。声明route或空事件均不足以单独证明执行。
- 单独B/C/C/B通信诊断，使用现有evaluator明确指定`torch_communication`的`communication_kernel_sum_ms`；每端点完整质量、原5个clean样本和3个独立profile样本。只为补采缺失的通信维度，不重新接受/发布候选，不向原独立结果补写数据，不把sum解释为wire time或关键路径。所有端点/样本保留。

随后从原项目、原torchrun命令和整步目标开始全新三轮真实Agent流程；不提供adapter、旧候选、分组、上述诊断或审阅回答。保留world_size=2、原微批次/总工作量、FP32/noTF32、完整质量、精确rank关系和1.05倍原baseline整步约束。新adapter和合同由框架自主生成；源审计与关系负例另外执行，若有接受版本，再做完整独立AB/BA。观察字段/容器变动需代码审查映射，不能自动判数值失败或通过。

脚本`v6_ddp_step_followup_20261028.py`；目录`/root/autodl-tmp/dev/v6-work/ddp-step-followup-20261028`。GPU阶段串行，原始数据留服务器，启动前实际容器剩约42.57GiB。没有大型Qwen任务或本地原始数据转存；每阶段留小型索引和摘要到GitHub。只设置一个普通终态监听，不启用Goal或模型轮询。

这不是把固定候选重新测到通过：旧整步独立接受、旧通信独立拒绝都保留；新原入口任务独立记录结果。allocator改动与两次review建议仍不采纳，TP完整质量失败以及历史Qwen/ResNet结果和样本也保持。

启动记录：2026-10-27 21:50:12 UTC冻结`53d2611fa96ec6b27263c8dfeefe8fb08283fe2d`，controller PID372607；核心SHA仍`cb4ea28c…`。[launch](evidence/v6-ddp-step-followup-20261028/launch.json)与[protocol](evidence/v6-ddp-step-followup-20261028/protocol.json)留档。启动实际df可用45,692,874,752 bytes（42.56GiB）。首次检查B原生profile已passed，C的两个GPU worker正在运行，尚未据此宣称新原入口结果。唯一普通监听PID18968、状态waiting、同一controller存活；[设置证据](evidence/v6-ddp-step-followup-20261028/wakeup-setup.json)已归档。没有恢复旧监听或转存控制器。
