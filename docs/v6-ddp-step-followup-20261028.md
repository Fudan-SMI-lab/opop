# DDP冻结路径补核与新原入口整步目标

承接[整步目标55.73%独立收益的审计](v6-ddp-step-goal-20261028.md)。本批保持生产源码、提示、GC及接受规则，补齐已证明存在的信息缺口，再检验完整原入口能力。

固定部分使用`ddp-step-goal-20261028/ddp-step`的baseline与最终候选`3c0b8d62…`，不改其代码、参数或旧结果：

- 单独B/C原生discovery trace，核对实际模型调用形状/次数、当前输入、NCCL执行及原生kernel路径。声明route或空事件均不足以单独证明执行。
- 单独B/C/C/B通信诊断，使用现有evaluator明确指定`torch_communication`的`communication_kernel_sum_ms`；每端点完整质量、原5个clean样本和3个独立profile样本。只为补采缺失的通信维度，不重新接受/发布候选，不向原独立结果补写数据，不把sum解释为wire time或关键路径。所有端点/样本保留。

随后从原项目、原torchrun命令和整步目标开始全新三轮真实Agent流程；不提供adapter、旧候选、分组、上述诊断或审阅回答。保留world_size=2、原微批次/总工作量、FP32/noTF32、完整质量、精确rank关系和1.05倍原baseline整步约束。新adapter和合同由框架自主生成；源审计与关系负例另外执行，若有接受版本，再做完整独立AB/BA。观察字段/容器变动需代码审查映射，不能自动判数值失败或通过。

脚本`v6_ddp_step_followup_20261028.py`；目录`/root/autodl-tmp/dev/v6-work/ddp-step-followup-20261028`。GPU阶段串行，原始数据留服务器，启动前实际容器剩约42.57GiB。没有大型Qwen任务或本地原始数据转存；每阶段留小型索引和摘要到GitHub。只设置一个普通终态监听，不启用Goal或模型轮询。

这不是把固定候选重新测到通过：旧整步独立接受、旧通信独立拒绝都保留；新原入口任务独立记录结果。allocator改动与两次review建议仍不采纳，TP完整质量失败以及历史Qwen/ResNet结果和样本也保持。

启动记录：2026-10-27 21:50:12 UTC冻结`53d2611fa96ec6b27263c8dfeefe8fb08283fe2d`，controller PID372607；核心SHA仍`cb4ea28c…`。[launch](evidence/v6-ddp-step-followup-20261028/launch.json)与[protocol](evidence/v6-ddp-step-followup-20261028/protocol.json)留档。启动实际df可用45,692,874,752 bytes（42.56GiB）。首次检查B原生profile已passed，C的两个GPU worker正在运行，尚未据此宣称新原入口结果。唯一普通监听PID18968、状态waiting、同一controller存活；[设置证据](evidence/v6-ddp-step-followup-20261028/wakeup-setup.json)已归档。没有恢复旧监听或转存控制器。
