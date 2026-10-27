# 采用 intake 语义说明后的原入口验证

依据[上轮真实审计](v6-semantics-lifetime-20261027.md)，生产仅采纳经过 A/B 的 intake v2 一段通用提示。生命周期提示不变。目录 `/root/autodl-tmp/dev/v6-work/ddp-semantics-20261027`，冻结提交与 SHA 在启动记录保存。

## 固定协议

1. 新 DDP 原入口完整三轮：原项目、原命令、原通信目标及整步约束，不给旧 adapter、分组、候选或人工诊断。真实 Agent 自主 intake、evaluator、发现/采集、选择、生成、回装、评价和反馈；API 用量只统计、不限额。三轮是该批协议，不是会话预算。
2. 源 serial/global-batch 全量梯度/参数、精确 rank 关系负例独立检查；若生成 observation 名称/容器不同，记录需人工审查，不将名称不同自动解释为数值失败。有接受版本则仅一次完整独立 AB/BA；保留搜索接受与独立拒绝的区别。审计说明是否真正允许等价通信策略，且没有省略工作量、质量或聚合语义。
3. TP 诊断串行在其后：原 BF16/SDPA 单次 prefill，单卡与官方双卡 hook 第一 block 内的归一化、Q/K/V、attention 输出投影、MLP 等输入/输出；hook 不改模型返回值。保留实际 shape/DTensor 元信息，区分直接比较与连续 rank 切片假设，不把未匹配形状当错误。
4. 对保存的**相同单卡输入与 checkpoint 权重**，另做两个 row projection 的 full BF16、分半 BF16 后求和、分半 FP32 后求和/转回等局部算术对照。只隔离舍入/分块影响，不声称复现 NCCL 执行，不把 FP32 用到原模型或改变质量门槛。此前 TP 完整32-token logits 失败继续有效；本批不是完整 TP 验收，更不是自主通信优化。

当前 df 可用约10.13 GiB，先前同类 DDP 占约6.2 GiB、独立目录约376 MiB。只开这一个项目，TP 保留第一 block 小型张量、不做大 trace。df 与 statvfs 在此容器不同，脚本分别记录，按 df 判断额度；不删除旧证据。运行代码从 Git archive 单独冻结，使用既有私有配置/依赖 overlay；两个 GPU 作业共享租约串行。

ResNet 同 adapter 显存候选的独立吞吐拒绝、其固定80样本、旧 Qwen 固定候选拒绝及全部80样本、旧 memory adapter 36.68%接受与缺失 images 事实，均保留原结果。本批不继续重测 ResNet 直到通过。

## 验证与自动续行

启动前做脚本编译、实际环境导入、配置与旧 TP 请求结构核查。已有 intake 6项检查通过，采用段落与实际 treatment 逐字一致。启动后记录真实会话/快照和单个普通完成守候。守候仅检查文件/进程，结束或控制器异常退出后 queue 一次；不启用 Goal、额外投递测试或模型轮询。

脚本：`v6_ddp_semantics_20261027.py`、`v6_tp_block_audit_20261027.py`。初始阶段/生成失败也应记录，不把 controller completed 当自主优化成功。终态后审计实际生成代码、inputs、controls、负例、完整版本、TP 输出，再决定下一步最小改动。

## 已启动

2026-10-27 10:52:57 UTC 启动单控制器 PID `240963`，冻结提交 `a44a6f379c93e6384624b5ed6ef3da23b140b47e`，核心 SHA `cb4ea28c3d90ccad5bc723ed5b6dc124488d572fca8a69b3ed3b12ec0e994016`。启动 df 可用10,863,415,296 bytes。脚本编译、实际依赖导入及源/资产/reference 路径检查通过。TP 请求的依赖 overlay 保留，但框架导入路径明确改为本次冻结源码。

真实 intake 会话 `ses_e83020a1dffetsO3CcM8qBPlMI`，生成前快照 `generation-ea83ca2cbb72` 已保存。仅表示实际角色已开始，不预判最终合同与优化结果。本机普通守候 PID `20504`，状态 `.scratch/v6-ddp-semantics-wakeup.json`；真实 probe 确认控制器 running。见[启动证据](evidence/v6-ddp-semantics-20261027/README.md)。

## 终态审计

整批于2026-10-27 14:28:37 UTC结束，三个阶段均执行完成，GPU worker与控制器退出。完成守候成功提交消息 `01a17dc3-9a70-7be1-ba2f-723d5110b8ff`，实际开启本次审计。**DDP最终独立拒绝，TP完整质量仍未就绪；completed 不代表优化成功。**

### Intake、evaluator与源一致性

实际 resolved request 保留 world=2、每rank microbatch=1、序列128、累积4次、全局8样本/1024 tokens、FP32/关闭TF32、SGD lr=.001、完整梯度/参数/精确副本检查。通信目标和step约束都是先对齐sample取rank max，再取sample median；原step上限为120.89597625 ms。reset、检查、单独profile和报告在计时外，完整前后向、NCCL同步、SGD和完成边界在计时内。

新 intake 将 bucket/no_sync/foreach/Gloo 等写为baseline选择，并明确允许语义等价替换；下游 understanding 同时记录baseline假设与必要不变量。三个真实候选随后实际改写通信策略，说明这一自由度传到搜索，不只是角色A/B文本变化。仍有较细的源实现描述，不能推广成所有未来项目都不会过度冻结。

evaluator最初一版 controls 失败：rank0的38,597,376元素张量有容差超限，rank1随后Gloo等待超时。框架把失败反馈给第二次真实evaluator会话；生成代码修正 tied wte/lm_head 的 state_dict 恢复，并在自身连续步骤检查中从上一步分布式参数构建单步串行参考。第二版controls通过，adapter SHA `a4056384d718eebc668c897357ba3468306756aefa07fa77236ffb8feea3ec2d`。这属于原入口工作流内的自主修复，未人工改adapter。

批后通用源码审计抛出 `KeyError: loss`，因新observe输出名为 `accumulated_loss`。代码核对确认其为四个已除4的microbatch loss之和，gradients/parameters使用原named_parameters字典。另建明确映射的独立审计，**仍从原始初始状态做完整两步serial/global-batch轨迹，不沿用adapter的分步anchor，也不修改容差或旧记录**。两rank每步各124,439,808个梯度和参数均通过：梯度最大误差分别7.45e-9、6.80e-8，参数最大误差3.73e-9；平均loss最大误差4.77e-7；每步跨rank梯度/参数精确一致。原报错保留，补充结果见 `manual-audit/source-mapping-audit.json`。

真实容差内梯度负例 `probe_valid=true, detected=true`，精确关系检查有效。质量通过不意味着计时满足目标。

### 三轮实际行为与独立复测

| 轮次 | 实际生成与执行 | 结果 |
|---|---|---|
| 1 `trial-8469963975` | 整步region替换；所有microbatch no_sync，最终pack后一次all-reduce，再unpack与SGD | 通信指标33.121 ms，但step127.067 >120.896，参数筛选拒绝 |
| 2 `trial-7d98173269` | 渐进bucket合并、手动Future、Gloo对齐，尝试恢复计算/通信重叠 | 三组参数均无有效候选，质量检查失败；不计为性能测量成功 |
| 3 `trial-25f69b0d87` | 官方all-reduce future链、逐bucket Gloo barrier，SGD前显式CUDA完成同步 | 搜索完整AB/BA接受，随后独立AB/BA拒绝 |

第二/三轮生成前快照均包含真实先前反馈，提案明确引用前轮耗时/参数错误并修改实现。Agent将第二轮归因于手动Future导致更新读取未完成梯度；第三轮同时改future链和完成同步，质量随后通过，但不能把这当作单变量因果实验。

最终选中 `pre_backward_align=none, per_collective_align=barrier`，bundle `trial-25f69b0d87-param-a8bd4641`。搜索端点通信52.023/53.932→35.210/35.272 ms，step113.960/119.493 ms，满足当次判据。覆盖记录走 `invoke:aligned-official-bucket-allreduce`，无记录的fallback；源码与原生trace交叉确认真实通信执行，每rank13个NCCL `RING_LL` kernel。第一轮原生trace为每rank1个 `RING_LL`，并非Agent预期的SIMPLE路径。

独立完整版本仍质量通过、coverage_errors为空；通信端点47.809/41.236→35.252/35.195 ms，但step116.707/**122.944** ms，后一端超过120.896上限；第一对收益12.557 ms也未超过13.886 ms噪声界。因此同时因 `original_baseline_constraint_failed` 与 `paired_gain_not_above_noise` 拒绝。保留搜索接受与独立拒绝两个事实，不按平均通信下降约20.9%宣称成功。

候选主要将一部分等待移到Gloo/CPU侧。NCCL kernel sum仍不是wire time、链路带宽或暴露关键路径；提案中的“wire cost”等描述未得到这些观测支持。没有据此修改框架或追加长提示。

含intake共13个真实角色会话，provider报告费用10.92258456982，仅统计、不限额。原始参数筛选、完整端点、反馈输入哈希、源码和原生SQLite路径/SHA见证据目录。

### TP第一block诊断

原BF16/SDPA模型的单卡prefill logits零误差。双卡prefill max_abs仍为.4375、首token相同；旧完整32-token max_abs=3.0的拒绝不变。

embedding、第一层input norm输出完全相同。按实际shape的连续rank切片比较，Q投影输出已有差异（rank0 max_abs=.00048828125，rank1=.0009765625），K/V投影为零；这仍明确是分片布局假设下的采样边界比较，不把未匹配布局直接判错权重。o_proj输出max_abs=.015625，第一block末为.0625。差异已在此block的attention输出all-reduce之前出现，不能只归因于NCCL求和。

相同单卡输入/完整checkpoint权重的独立算术对照：full BF16完全复现记录输出；将输入维度分半做BF16 GEMM后相加，o_proj与down_proj分别产生max_abs=.015625/.03125。仅把两个已舍入的BF16部分输出转FP32求和再转回，结果不变；FP32部分GEMM可降低RMS但也不复现原BF16输出。局部算术说明分块/舍入具有影响，不能证明完整TP失败的全部因果，也不是NCCL执行复现或修改后的模型验收。未将FP32诊断作为生产精度修改。

## 后续顺序

优先执行已授权的[全盘清理](v6-storage-cleanup-20261027.md)，不新开大实验。清理后保持当前完整DDP与Qwen/ResNet有效版本可运行。DDP仍缺稳定满足整步约束的通信正例；若继续诊断，围绕同一最终版本的等待转移与端点波动取最小证据，不能反复重测到通过。TP可进一步隔离同输入Q投影的colwise计算差异，但不放宽原容差，不把FP32局部实验或prefill通过冒充完整32-token接受。当前没有新的生产prompt或搜索策略修改。
