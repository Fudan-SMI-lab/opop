# 多 rank 指标聚合顺序：源定义审计与最小修正

## 实际发现

在 2026-10-27 DDP 新原入口实验中，intake 已准确描述原项目的通信指标：对每次 profiler 采样取两个 rank 的最大值，再对三次采样取中位数。原始脚本 `examples/v6_validation/gpt2_ddp_training.py` 对 clean step 也采用逐次最大值后取统计值。

框架此前的 `measurement.metric()` 则对各 rank 已汇总的中位数取最大值。两者一般不等价。本批 prepared 端点的通信读数如下（ms）：

| 采样 | rank 0 | rank 1 | 当次最大值 |
|---|---:|---:|---:|
| 0 | 35.921772 | 42.693940 | 42.693940 |
| 1 | 38.092142 | 34.704439 | 38.092142 |
| 2 | 38.400203 | 35.207955 | 38.400203 |

原项目定义得到 **38.400203 ms**，旧框架得到 **38.092142 ms**。该端点的 clean step 两种计算恰好都得到 116.142063 ms，不能由此推断其他端点也一致。

这是已有样本和源代码支持的评测差异，不能靠给候选 Agent 更多 profiling 文本来修复。正在运行的 `01fcb47` 三轮 DDP 实验继续使用原冻结代码；本修正与其结果分开记录，不向原会话注入诊断。

## 最小实现

在 objective 和各性能 constraint 中增加可选 `aggregation_order`：

- `samples_then_ranks`：保留原行为，也是未指定时的默认值。
- `ranks_then_samples`：对每次对齐的 rank 样本应用已有 `rank_aggregation`，再取中位数。

第二种模式直接使用已有 `metric_samples`，普通指标则使用 clean `samples`；要求每个 rank 的样本数量一致且非空，避免静默截断。只有源项目定义了对齐的分布式调用时，入口 Agent 才应选择该模式。框架不按 DDP、模型名或指标名称自动猜测。

同一选择用于参数筛选分数、最终配对比较、原始约束和给 Agent 的目标参考值。新模式的噪声界也从逐次聚合后的样本估计；旧模式保留原先各 rank MAD 的算法。原始样本不被覆盖。

给 Agent 的改动仅是 intake schema 中这个字段及其说明；未增加源任务特例或失败分类规则。旧请求不要求补字段。

## 验证

受影响的 measurement、diagnostic planning、intake 检查共 **22 项通过**。新增检查覆盖真实 DDP 数组、慢 rank 交替导致的接受结论变化、constraint 独立选择顺序、参数筛选分数与最终端点一致，以及样本缺失时不能静默丢弃。

真实 Agent 对照已完成：从原始 intake 初次生成前的快照分别重放一次 control/treatment，使用同一新增运行代码、模型、prompt 和其他输入，只替换 treatment 的 `inputs/intake.schema.json`。两臂均允许原有自动契约修复，不限制 API 请求或费用。该对照是角色实验，不作为完整自主项目成功证据。

| 对照 | 真实 Agent 结果 | 自动契约修复 |
|---|---|---:|
| control，旧 schema | ready；目标和 step constraint 均未声明聚合顺序 | 0 |
| treatment，新 schema | ready；目标和 step constraint 均自主选择 `ranks_then_samples` | 0 |

使用 `zhipuai/glm-5.3`，两臂的 provider `reported_cost` 分别为 0.16868548、0.10211447，仅作估计统计。这一对结果支持“增加最小能力说明即可让本次 intake 表达正确语义”，不代表所有任务或未来调用的稳定成功率。

目录：`/root/autodl-tmp/dev/v6-work/next-stage-20261027/metric-order-ab`；[对照结果](evidence/v6-next-stage-20261027/metric-order-ab/result.json)。后续项目实验应新建合同，而不是修改本批已冻结的请求。

## 仍需独立核对

新 DDP adapter 已观察完整梯度和更新后参数，但原源码还要求 rank 之间的梯度逐元素完全相同。目前 per-rank 的容差比较并不自动证明这一跨 rank 关系。已排定原源码全局 batch 对照与新 controls 的精确副本检查；它们用于核实当前基线，不证明任意后续候选都遵守该关系。若需要完善 evaluator 对自定义质量条件的执行，应单独保存角色状态并做最小对照，不能把条件仅写在合同文字中就视为已经实现。

后续实测更新：基线原源码全局 batch 对照和精确副本检查均已通过；三轮旧批次按源顺序重算仍被拒绝。另一个独立双卡负例已证明，rank 1 容差内的非零梯度扰动会破坏精确副本关系，但现有全量比较仍通过。已启动仅一段通用关系检查提示的真实 evaluator 对照，见[协议与负例证据](v6-post-batch-audit-20261027.md)。该改进与指标聚合属于两个独立问题。
