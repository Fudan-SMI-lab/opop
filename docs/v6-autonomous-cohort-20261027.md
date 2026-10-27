# 同版本三项目原入口自主验证

本批执行[阶段 D](plan-v6-next-stage-20261027.md)，前提是关系检查提示已通过真实 evaluator A/B、双卡负例和原源码一致性验证。脚本为 `scripts/experiments/v6_autonomous_cohort_20261027.py`，目录 `/root/autodl-tmp/dev/v6-work/autonomous-cohort-20261027`；启动记录提供精确提交和源码摘要。

## 固定任务与顺序

| 任务 | 原项目入口 | 冻结工作量与目标 | 轮数 |
|---|---|---|---:|
| Qwen3-4B | qwen_inference.py 与既有本地权重 | interactive：batch 1、256 输入 token、完整生成 32 token；最小 TTFT | 3 |
| ResNet-18 | resnet_inference.py 与既有真实图片/预训练权重 | offline：batch 32；最小完整推理延迟 | 3 |
| GPT-2 DDP | 原 torchrun 双 rank 命令 | 原累积训练步；最小诊断 NCCL kernel 总时长，保留整步约束 | 3 |

从原项目目录、原启动命令、目标和当前设备开始，全部重新执行 intake、evaluator、发现/分组、选择、生成、回装、评测和反馈。不给现成 adapter、算子组、候选或旧人工诊断。三轮用于覆盖两次后续反馈机会，不是 Agent/API 次数、费用或 token 限制。不强制 Agent 换组或采用某类实现；失败与无可接受候选均是实验结果。

全部任务和后续检查串行使用共享 GPU 租约；每个任务分别记录输入、冻结代码、终态和空间变化。启动前数据盘约 25 GiB 可用；模型资产不复制，独立版本检查复用不可变参考，保留已有冻结实验。只在数据盘新增本批结果。

## 单独的 collector planner 真实对照

自主 cohort 前，重放旧 DDP 第三轮 post-evaluation collection planner 的初始快照 `generation-40fef4adc280.json`，按 control/treatment 各一次真实角色会话。仅将 candidate_diagnostics 的能力描述改为与实现一致：候选文件当前线程的异常与 False 返回；不提供正常调用、其他标量返回或由空事件证明成功分支。输入文件、模型和其他提示不变；保留已有自动契约修复。

旧实际回复曾请求不存在的 `candidate_diagnostics.executed_calls`、`candidate_diagnostics.scalar_returns`，并将其视为路径证明，因此修正有真实问题依据。此对照只检查描述与后续计划，不再执行旧候选采集；它的输出、选组和反馈不进入三个自主任务，也不与上一批 evaluator 对照混为一项实验。

## 独立核验与解释范围

三个自主任务结束后，分别检查新 baseline adapter 的原源码一致性。Qwen/ResNet 使用既有源项目完整参考 logits 和精确 token/class；任何新增观察列出但不因对比子集而被声称已经源验证。输出命名不同导致检查不可用时明确记录并人工核查，不将其等同数值错误。DDP 使用原源码 serial global-batch 两次连续更新和完整梯度副本比较，再执行同一关系负例检验新 evaluator。

若项目当前版本包含接受的优化 bundle，再对整个完整版本运行一次新的标准独立 AB/BA，与该任务原基线比较；不发布这个独立复测，不追加到分数通过为止。无接受候选则记录未执行的原因。

报告区分原入口自主链、角色对照、源一致性与独立性能验证；检查多组共存、接受后重新发现与反馈使用，但不预判这些行为必然出现。Qwen 旧固定候选的 80 次采样仍只属于开发诊断，原标准失败保留。当前仍缺少可信通信优化正例，阶段 E 的目标切换和新通信项目待本批证据继续推进。

## 完成后续行

本批由一个控制器串行管理全部阶段；`launch.json` 与 `followup-launch.json` 是同一进程的启动记录，终态写入 `controller-completed.json` 和 `followup-completed.json`，兼容现有普通守候协议。错误也会写入结果；进程异常退出且无终态时由守候触发核查。

配置独立本机状态 `.scratch/v6-cohort-wakeup.json`，每 30 秒通过普通 SSH 检查终态；结束后向当前会话排入一次核查与继续指令。不启用 Goal，不用模型轮询，不添加投递测试。

## 已启动

冻结提交 **`4cdd076376de`**，于 `2026-10-26T22:02:24.751310Z` 启动控制器，PID `63346`。collector planner control 的真实初始会话已建立并保存生成前快照，三个项目按上述顺序排在角色对照之后。

本机守候 PID `22328` 已启动，真实 SSH 检查确认控制器运行，状态为 `waiting`。模型调用只发生在框架实际工作或批次结束后续行，等待由普通脚本完成。启动时数据盘可用 26,295,762,944 bytes（约 24.49 GiB）。[部署与输入证据](evidence/v6-autonomous-cohort-20261027/README.md)已归档；当前不预判自主优化或独立验证结果。

## 完成后审计

控制器于 `2026-10-27T04:41:53.122740Z` 正常结束，七个阶段均返回 completed；三个项目的 evaluator controls 均通过。这只是执行终态，下面分别列出优化和独立验证结果。期间没有人工插入候选、反馈或修改冻结框架。源码摘要为 `ed1f8a63296fa915e6c18737293e94d9fa3d82fd3918958400e5033754156ebd`。

| 项目 | 按实际生成顺序的三轮接受情况 | 独立完整版本 AB/BA | 源码一致性 |
|---|---|---|---|
| Qwen TTFT | 拒绝、拒绝、接受 | 25.079302 → 21.553870 ms，降低 **14.06%**；两对均超过噪声界 | 两次完整 logits 与生成 token 全量比较通过，零误差 |
| ResNet 延迟 | 接受、接受、接受 | 4.251161 → 3.292693 ms，降低 **22.55%**；吞吐约束通过 | 人工核实字段映射后，完整 logits 和精确类别通过 |
| DDP 通信诊断 | 拒绝、接受、拒绝 | **拒绝**；诊断 NCCL 时间 42.175118 → 36.790167 ms，但候选整步端点 128.534849 ms 超过 121.511032 ms 上限 | 人工核实参数序列映射后，原源码全局 batch、两次连续更新、完整梯度/参数/损失及精确 rank 副本关系通过 |

独立检查使用新 worker、完整最终版本和原 baseline，均未发布独立复测版本，也未追加测量直到通过。Qwen 旧固定 RMSNorm 候选的独立拒绝和全部 80 个诊断样本仍保留；本次接受的是另一个从原入口自主生成的版本。

### 实际执行、反馈与组合

- Qwen 的 `trial-1a25041f29`、`trial-cf72c8a39d` 都实际走 parent-eager。第三轮 `trial-b934b61f27` 的 proposal 明确引用这两个失败及 `_ok=False` 诊断，调整候选自己的安装验证门槛，没有修改冻结质量要求；完整质量和独立测量通过。范围是 prefill/decode 两组 joint，decode 仍透传。原生 sqlite 记录一次 `cudaGraphLaunch`、2,205 个 graph kernel，但 `_k_add_rmsnorm`、`_k_qknorm_rope`、`_k_silu_mul` 均零次。代码在 Triton 不可用/编译失败时可选 Torch composite，因此 `graph-composite` 自报名称不能证明 Triton 融合执行。当前只证实 graph/composite 路径和整项目收益，不能宣称 proposal 所述约 500 个融合 kernel。接受发生在最后一轮，本任务没有接受后的下一轮。
- ResNet 的 `trial-a7f5de7ac7`、`trial-1bfb948628` 为 joint，最后 `trial-abe1619d50` 为完整模型 region。接受后重新发现，再由新候选替换旧 bundle；实际消费前一轮反馈。最终原生 trace 中 46 个 kernel 全部属于 graph，包含 CuDNN 融合计算。候选报告 Triton pool 不可用，不能宣称该优化执行。每个版本只有一个 active bundle，证明多组联合、区域和替换链，不证明多个独立 bundle 同时共存。
- DDP 首轮 block joint 未改善，第二轮 execution_plan 在最后一次 forward 与 backward 之间插入 Gloo barrier，搜索中接受；第三轮基于该版本重新发现并尝试新 NCCL group/hook，违反整步约束且未超过噪声，被拒绝。最终独立测试的是第二轮完整版本。其诊断时间下降包含将等待移到 NCCL 之外的影响，没有减少通信字节或提高链路带宽的证据，尚不是可信的通信效率正例。

生成前快照保存了第二、三轮实际 previous-feedback 文件，proposal 引用了前轮具体测量和失败。Qwen selection 的反馈文件约 60 MB、candidate 约 5.6 MB；这是可读文件大小，不等于全部进入模型上下文。作为后续信息精简的调查线索保留，尚不据此重写反馈机制。

### 原源码检查与负例

冻结检查脚本的 ResNet 初始失败是字段 `top1` 与生成 adapter 的 `predicted_top1` 不同；实际代码返回 logits.argmax(-1)。DDP 初始失败是审计预期 named dict，而 adapter 按 `raw_model.parameters()` 顺序返回 148 个张量的 list，初次比较未检查任何梯度元素。因此这两项不能按原始 false 直接认定数值不一致，也不能忽略。

新增独立审计脚本 `scripts/experiments/v6_cohort_schema_audit_20261027.py` 依据实际源码手工确定映射，另建 `cohort-schema-audit-20261027`，未修改旧结果、adapter、参考或容差。ResNet 两次完整 32×1000 logits 和 32 个类别通过；DDP 两次连续更新、每项 124,439,808 个梯度/参数元素及损失、精确副本关系通过。这是案例审计修正，不是框架硬编码。

新 DDP evaluator 同时通过真实双卡关系负例：rank 1 的一个梯度改变约 `2.500019e-7`，仍在普通容差内，但 observe 的精确副本检查成功抛出异常。负例是在独立 worker 注入，不进入优化性能测量。

### Collector planner 单项对照

实际两臂初始快照的 framework、model、tool settings 和所有输入文件一致，prompt 唯一差异是 candidate_diagnostics 的能力描述。control (`ses_e85c380e6ffevewjE6MVB3ZuzJ`) 仍请求不存在的 executed_calls/scalar_returns；treatment (`ses_e85c16a0bffem0w2gjKEJJK2qu`) 明确将成功路径证明列为 unavailable，不再把空事件当作正常路径证明。两臂均无契约修复。

这支持当前能力说明修正，但不证明 treatment 的所有因果解释都正确：其 calibration 理由仍有把合成参考理解成可分离等待/传输的表述，现有观测不能支持这一推断。此角色对照没有实际执行采集器，也没有给自主 cohort 注入结果。

### 仍需改进与下一步

DDP intake 把 `bucket_cap_mb=25`、`gradient_as_bucket_view`、`no_sync` 等实现选择与数值语义一起写入冻结 request.notes。后续 Agent 据此将这些选择视为不可调整；evaluator 已有的语义区分提示不能撤销上游冻结要求。下一步从本批 DDP intake 的原始快照，真实对照一段通用的“任务要求与源实现选择分开记录”说明，保持所有输入文件不变。先检查结果再决定生产提示是否采用。

阶段 E 先执行相同 ResNet batch-32 的带吞吐约束显存目标，重新从原入口完成三轮，并保留双向指标交叉评价；跨目标分析不修改原合同或将不同 anchor 的 bundle 强行合并。额外通信案例先验证官方 Qwen TP=2 的完整源码基线和通信占比，只有实际符合正确性且有明显通信开销才进入优化；它是新分布式执行案例，不虚称新增模型种类。

数据盘剩余约 11.1 GiB，本批约 14 GiB，主要为 Qwen trace/重复反馈和 DDP 全量参考。系统盘仅约 3.6 GiB，不能作为迁移目标。已验证 XFS reflink 内容独立，但对一份 261 MB 完全相同文件试验后项目配额可用空间未增加，因此未批量操作或删除冻结证据。该次 CoW 替换保持路径、内容 SHA 和元数据不变，详细记录一并保留。下一批先做较小任务，不再同时铺开多份 Qwen 大采集。

三项目共 39 个框架角色会话，provider 报告成本分别 10.6636746687、8.04457475118、9.74973402396（原 provider 单位，非独立账单；不含 planner A/B）。统计不限制后续调用。
