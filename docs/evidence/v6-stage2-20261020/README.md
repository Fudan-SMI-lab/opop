# v6：绑定后端、通信反馈与全新输入验证

本轮在 box4-2 双 RTX 4090 上使用实际 GLM-5.3 请求与 GPU 执行。
实现说明见[阶段记录](../../v6-stage2-binding-and-autonomy.md)。

13 次全新启动中，12 次通过生成适配、独立源程序核对并完成两轮搜索；
另一次在准备阶段失败，原记录保留。24 轮中框架接受 9 次、拒绝 15 次。
其中统一批次的一个 Qwen 候选实际全部回退，**不计为有效优化**。
原始接受决策与发布指针没有被人工改写。[汇总审计](execution-audit.json)、
[逐轮结果](outcome-matrix.json)与[原始协议摘要](autonomy-summary.json)分别保存。

## 统一冻结批次

六任务全部使用 `21897c111d648d83a87f87c770b5480a2e8751c9`，
核心源码摘要 `c21f4933a170e24ef433bd4fa79791e84958e5be370d0edd4baec177a03bbc60`。
运行中没有人工修改候选、补充会话信息或调整标准。
每个目标独立生成 evaluator，输入没有历史分组、适配代码或成功候选。

| 任务与目标 | 两轮结果 | 判断 |
|---|---|---|
| ResNet-18 B32，延迟 | 4.436→4.388；4.406→4.370 ms | 两轮质量通过，收益均未超过配对噪声 |
| ResNet-18 B32，工作负载峰值 allocated 显存 | 282,103,808→205,071,872→170,932,224 bytes | 两轮接受；累计约下降 39.4%，原始吞吐约束通过 |
| Qwen3-4B B1，TTFT | 首轮 logits 失败；次轮 24.980→26.986 ms | 次轮质量通过但变慢，拒绝 |
| Qwen3-4B B1，完整生成吞吐 | 首轮 logits 失败；次轮 44.674→46.067 token/s | 分数规则接受，但实际全回退；排除有效收益声明 |
| GPT-2 双 rank DDP，整步时间 | 首轮 Work 返回接口失败；次轮 140.577→234.717 ms | 次轮完整质量精确通过，性能拒绝 |
| GPT-2 双 rank DDP，通信目标 | 两轮完整质量通过，整步分别 253.193/257.871 ms，约束上限 148.377 ms | 两轮拒绝，无通信加速结论 |

Qwen 两目标均为 prompt=256、生成32 token；吞吐包含全部32个生成 token。
ResNet 两目标均为 batch=32、224×224；DDP 两目标均为序列128、累积4、双 rank。
独立生成的 adapter 存在计时边界／胶水代码差异，不能把不同目标间比较当作严格的
同 adapter 因果消融；每条运行内部的父版本／候选配对具有自己的固定协议。

显存任务第二轮保留第一轮 stem bundle，新增覆盖24组的 stage0/stage1 bundle。
两个连续步骤均比较完整32,032个元素，最大绝对误差约 `9.06e-6`；两个 bundle
均有真实执行覆盖，原生 trace 分别记录40与76次关联内核启动。
吞吐为8,329/8,035 image/s，超过原始95%约束6,886.6 image/s。
这验证了同一项目内的多组选择、重分组、候选共存和整版评价。
见[第二轮完整回执](stage2-autonomy-final-21897c1/resnet-memory/trials/trial-e3e197b753/result.json)。

Qwen 全回退案例的每个配对候选端点记录62次替换入口调用、125次父实现调用，
以及64条 `fallback:setup-failure:IndexError` 路径记录。
入口计数不等于优化实现生效；该结果暴露了当前接受规则与有效优化归因之间的缺口。
本轮没有增加新的接受门槛、失败类型路由或信息获取限制。
见[原决策与实际路径](stage2-autonomy-final-21897c1/qwen-throughput/trials/trial-6fbe9313aa/result.json)。

DDP 整步候选的参数写为 `SIMPLE`，但前后原生 trace 均显示 `RING_LL`；
不能声称协议切换已经发生。它的四个配对端点、每 rank 两个连续步骤均比较
248,879,617个元素，误差为零，仍因整步变慢而拒绝。
通信 kernel 累计时长包含等待，不是物理传输时间。

## 开发批次与独立复测

开发主批次冻结在 `0e4c751bf0a6e999f91be4b9f7fe8d271f0739f5`。
首次 ResNet 显存准备失败暴露 `state_protocol` 的类型校验遗漏；
修复后在 `be511646aeee40c0357a4cefdc3ef1540a5c87d1` 重新启动全新运行。
原失败没有被修复后结果覆盖。

| 任务 | 实测结果 | 执行证据与范围 |
|---|---|---|
| ResNet 延迟 | 首轮4.234→3.633 ms；次轮3.630→3.319 ms | 8组后扩展到14组，次轮明确替换旧 bundle；不是两个 bundle 共存 |
| ResNet 显存（修复后的新运行） | 首轮282.104→205.033 MB；次轮无有效参数候选 | 完整质量和吞吐约束通过，保留首轮版本 |
| Qwen TTFT | 首轮质量失败；次轮27.845→23.583 ms | 完整 token/logits 精确一致，2,144次原生 prefill 内核启动 |
| Qwen 完整生成吞吐 | 首轮45.282→43.167 token/s，拒绝；次轮46.249→81.999 token/s，接受 | 完整质量精确一致，62次有效解码调用，66,226次关联内核启动 |
| DDP 整步 | 首轮执行失败；次轮141.694→120.131 ms | 选择26组并改写计算树，双 rank 完整两步质量通过；不是通信加速 |
| DDP 通信 | 两轮均被整步约束拒绝 | 无接受版本，无通信加速结论 |

Qwen 吞吐首轮的两个连续步骤合计仅2次替换执行、60次签名守卫回退。
下一轮 Agent 从完整反馈和旧候选源码定位缓存子类型问题，改为返回准确的
`DynamicCache` 类型，62次调用全部进入 Graph replay 路径。
代码每步复制当前 token、更新实际 KV 状态并运行模型；没有缓存任务输出。
一次父调用用于候选内部正确性核对，不是热路径全部回退。
见[首轮](stage2-autonomy-0e4c751/qwen-throughput/trials/trial-9c4355e642/result.json)、
[次轮](stage2-autonomy-0e4c751/qwen-throughput/trials/trial-bccaddc2f6/result.json)及相邻候选代码。

首次 Qwen TTFT 配对的父端点存在较大波动。所有原自动运行结束后，开发者在
同一冻结源码、同一候选字节上另做 AB/BA：25.087→23.422 ms，约下降6.6%，
两对均超过噪声界限、完整质量通过。方向复现，幅度小于首次约15.3%。
这4个端点是独立复核，不计作无人干预生成或新的发布。
见[复测回执](stage2-qwen-ttft-replication-0e4c751/result.json)。

不同批次独立生成了不同候选，不能把批次间收益差异归因于框架提交的变化。
这些结果覆盖三项目的工作流路径，仍不证明任意项目／目标均能自动获得收益。
模型特定实现来自 Agent 生成的候选，核心框架没有新增按模型名调度的分支。
DDP 使用已声明的随机初始化 GPT-2 与合成 token，只验证参考计算一致性，
不代表训练收敛；本轮也没有验证 NVLink、RDMA 或多机硬件。

## 绑定诊断与反馈证据

`scoped` 保留为可选后端，默认仍为 `legacy`。
初始三项目36个端点、ResNet独立复测24个端点、最终核心版本复测24个端点，
共84个绑定诊断端点通过质量与覆盖。均使用相同候选字节，不发布候选。

最终 ResNet 四对端点的中位数：

| 条件 | legacy | scoped |
|---|---:|---:|
| parent | 1.604 ms | 1.528 ms |
| passthrough | 3.244 ms | 2.939 ms |
| candidate | 3.624 ms | 2.764 ms |

candidate 四对方向均改善，但组合仍慢于 parent。Qwen 的作用域仍覆盖整个
model，未观察到稳定收益；DDP 的无模块站点实际退回 legacy。
首次最终复测因 job basename 重名在端点开始前失败；原错误保留，唯一目录重启
完成验证，后续 driver 使用完整输出路径哈希命名。
[初始汇总](binding-summary.json)、[独立复测](stage2-resnet-binding-recheck.json)、
[最终复测](stage2-resnet-binding-final-21897c1.json)保存原始数值。

完整 Collector 附件带 revision、目标、设备、来源与哈希；已选采集器会在进入
配对评价且质量／覆盖通过的候选上重跑。筛查阶段已淘汰的候选保留原筛查及已有诊断，
不声称它们重跑了全部采集器。DDP 的 Torch 与 execution_timeline 两份观测各有
双 rank 共26个 bucket 所属记录，均能关联到实际组；这不是52个独立物理 bucket，
也不把共享通信时间分摊到各组。

[会话审计](agent-evidence-access.json)记录实际工具输入，包括上一轮反馈、
原生采集附件和 bucket 所属索引的读取。599条路径关联工具记录包含列目录与写文件；
其中150条是显式 read 调用，均不等同于理解正确。未发现历史运行路径的显式引用，
但该观察不构成访问隔离保证。GPU readiness、唯一 consumer 关系、等待与传输分离
仍有 unknown；Agent 的带宽下限、时序因果或 token 分叉解释不自动作为事实。

## 检查、用量与重现

最终核心实现的[相关检查](stage2-autonomy-final-21897c1/tests.json)为33 passed；
历史重叠的20/31/32项及局部检查不累加。
[用量](usage.json)记录77个会话（76个角色及1个子会话）、2,073条 assistant
消息、provider 报告费用估算64.26783885446美元，含失败尝试，不含外层 Codex
或上一阶段评审，不是最终账单。没有 API／token／费用额度。

各批次 `controllers.json`、请求、生成适配、源程序核对、候选源码、参数筛查、
配对检查和采集索引均留档。完整模型、张量与 trace 仍在 box4-2，
[533个原始文件的清单](raw-evidence-manifest.json)记录路径、字节数和 SHA256；
导出的 JSON 是可阅读摘要，不冒充原始文件字节。
`scripts/` 保存导出和只读审计工具；验证入口位于
`examples/v6_validation/run_autonomous.py`、`run_binding_backends.py`。
复跑须使用对应冻结源码与服务器既有配置，不能把不同来源的测量混为同一协议。
