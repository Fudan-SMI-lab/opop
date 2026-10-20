# v6 实施验收与可复现边界

本页对照[实施计划](plan-v6-general-c3-and-communication.md)归集证据，阶段过程见[执行记录](v6-execution.md)，接口与命令见[项目工作流](v6-project-workflow.md)。2026-10-20 已完成 P0–P6 在 box4-2 当前硬件上的主线实现与验收，包括三项目有效多候选、正式对照和自动探索分支的整版推广。这里的完成包含正确拒绝退化组合，不代表所有目标都获得加速；计划中延后的 TP2、NVLink/RDMA 等边界见下文。

## 验收矩阵

| 要求 | 实际证据 | 状态 |
|---|---|---|
| v6 独立开发、旧入口保留 | 独立分支／工作目录，项目命令单独路由；按变更选择旧行为检查，删除依赖缺失历史文件的过时用例 | 通过；未要求完整历史套件 |
| 真实 provider 与 GPU；用量只统计 | 各阶段 session/message、候选与 worker 回执；账户窗口等待后复用原会话；没有开发调用／token／费用上限 | 通过 |
| 设备绑定、跨 rank 作业及回收 | [P0 设备与故障证据](evidence/v6-p0-20261020/README.md)、[P1 多编辑事务](evidence/v6-p1-20261020/README.md) | 通过 |
| 自动生成项目 evaluator | [ResNet](evidence/v6-p2-20261020/README.md)、[Qwen／DDP 独立全量参考](evidence/v6-p5-20261020/README.md) | 三项目通过 |
| 无手工 targets 的发现、分组、选择 | module／functional 真实执行发现；GLM 选择 scope 与组，动态构造目标站点 | 通过 |
| 单组、联合、区域及执行策略 | [自动联合候选](evidence/v6-p3-20261020/README.md)、正式 whole/leaf 对照及实际 Graph／分布式策略 | 已执行；失败不算加速 |
| 旧 TaskSearch/TPE 与真实资源响应 | [参数资源和反馈](evidence/v6-resources-20261020/README.md)、[反馈后的新候选](evidence/v6-feedback-closure-20261020/README.md) | 通过 |
| ResNet 多候选完整回装 | 两个 bundle 同时执行、质量通过但组合退化；整版拒绝保留父版本 | 通过 |
| DDP 多候选跨 rank 共存 | 计算与通信策略在两 rank 共同执行，完整连续质量通过，整步退化拒绝 | 通过 |
| Qwen 两个有效候选共同执行 | batch=8 原 decoder 与新增归一化融合共同执行，完整质量通过；原生 trace 有 4,495 次新增 kernel；360.645→106.296 token/s，整版拒绝并保留父版本 | [有效共存通过](evidence/v6-post-recovery-20261020/p6-qwen-composition-assessment.json)；RoPE 仍回退，未称全部组成功 |
| 可行非贪心分支及自动整版推广 | 四轮完成；从可行非贪心分支生成新候选，经整版 AB/BA 接受并提交，0.731→0.669 ms | 通过；恢复前后源码身份分别记录 |
| 动态重分组与证据失效 | Graph 后内层调用退休；Qwen 新 15,059 组与旧归属算法逐项一致；关联资源带原版本 | 通过 |
| 接受、拒绝、回退与陈旧父版本 | CAS、完整 bundle 集合、覆盖与质量分开；真实外层 Graph 绕过旧候选被拒绝；[整版回传](evidence/v6-p6-branches-20261020/README.md) | 通过 |
| 合法单项与组合交互 | 同父版本实测 A、B、A+B；质量均通过，交互区间跨零；不可拆分候选保持 joint_only | [通过](evidence/v6-post-recovery-20261020/README.md) |
| 通信能力自动启用／缺测降级 | 单卡跳过；双 rank Torch/Nsys/NCCL transport、13/13 bucket 关联、权限不足实测回执 | 通过 |
| PCIe／NVLink 等资源维度 | 实际 NVML PCIe 速率与读取窗口；4090 NVLink 不支持；真实 payload allreduce 校准 | [当前硬件通过](evidence/v6-resources-20261020/README.md) |
| 通信候选质量及整步约束 | 最新策略完整连续质量通过，但约 261.57 ms 超过 161.42 ms 固定整步上限，被拒绝 | 通过拒绝路径；没有可靠通信收益 |
| Agent 实际消费细粒度负反馈 | 失败代码、调用路径、资源与整步退化；19 条历史文件工具记录及后续方案 | [通过](evidence/v6-resumed-20261020/ddp-feedback-consumption.json) |
| 目标切换、保留负载与新入口 | 八目标、跨目标重测；未注册纯函数项目；保留 shape/seed；成功、无收益及守卫回退分开 | 已执行 |
| 正式策略／反馈对照 | 冻结源码的 11 条路径、22 轮，全部失败成本保留 | [完成](evidence/v6-resumed-20261020/README.md)；不证明策略普遍优越 |
| Agent 与 worker 环境一致 | 默认 Python 与 worker 版本对齐；真实角色执行默认／显式解释器查询 | [通过](evidence/v6-post-recovery-20261020/p6-worker-environment-validation-a-result.json) |
| 联合候选的局部回退诊断 | 任一活动站点全部调用父实现即可触发独立追踪；真实完整 Qwen 草稿筛查自动发现 RoPE 编译错误并记录 rank／call | [通过](evidence/v6-post-recovery-20261020/p6-qwen-draft-project-d-result.json)；诊断草稿不冒充正式提交 |

## 如何解释结果

三种模型加一个纯函数入口证明当前工作流可以跨这些结构运行，不能证明任意项目都能自动优化。选组、候选和实际性能来自运行结果，不要求每种目标都获得加速。质量通过、调用覆盖通过、优化路径生效、性能被接受是不同结论。

Qwen TTFT 的独立正例、batch=8 吞吐的新正例、DDP 整步的独立正例、ResNet 的延迟／吞吐／allocator 显存目标以及信号项目反馈后的正例各有独立参考和 workload，不能把各自最佳数字拼成同一系统的累计提升。DDP 新候选在原整步目标通过，但保留形状和通信目标均未获可靠收益；这一差异保留为目标／负载边界。

最终 Qwen 组合是质量成立、两个 bundle 实际贡献输出、性能明显退化的负例。新增融合只覆盖归一化链；RoPE 内部守卫仍回退，其余明确委托的站点也不能算成新优化。两个连续步骤各比较 38,895,872 个元素，最大绝对误差 0.125 在原阈值内，不是逐位一致。此前 batch=1 Graph 的有效共存失败仍保留，不能被 batch=8 的结果追认。

每个性能接受仍要求原始质量与约束、完整候选覆盖及独立 AB/BA。当前采样较小，MAD 判据是实验接受规则，不是跨硬件、跨请求分布的统计保证。额外 profiler、内层发现和干净性能运行分别计量；通信 kernel 时间含等待，PCIe 速率是设备级观测。

## 仍需扩展或另行验证

- 盒子没有 NVLink/RDMA，相关真实硬件采集不能由现有结果推导。
- Qwen TP2 与单卡等价的历史基线问题保留为后续扩展；未放宽原质量门槛。
- 当前实际 worker 以 LocalExecutor 的一 rank 一 GPU 运行；单进程多 GPU、自定义分布式 runtime 和更多 collector 需要对应适配与实测，不能只按机器卡数推断通信已覆盖。
- 未观测的 alias、跨 stream 状态和无法唯一归属的 backward／共享 collective 保持 unknown，不自动拆成逐组因果收益。
- Collector 有配置注册接口；搜索策略和 Agent transport 仍通过各自模块替换，并非所有组件都有动态插件加载。
- 开发过程中补充的跨 run 历史、草稿诊断和静态检查均有回执；正常上一轮反馈、资源、候选代码和运行契约由框架自动传入，但未实现任意历史实验的自动检索。

原始 tensor、完整 trace、模型资产和私有 provider 配置保留在 box4-2。GitHub 保存代码、协议、关键结果与原始文件哈希；按各阶段 manifest 复核完整证据。恢复旧实验应使用该实验记录的冻结源码，不能把最新代码直接混入旧比较。
