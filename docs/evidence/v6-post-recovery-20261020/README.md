# 独立复核、组合交互与真实环境修复

本目录继续记录 provider 恢复后的真实运行。完整结果、生成代码及原始文件哈希在相邻 JSON／Python 文件；失败、无改善和未完成验收保持区分。

## DDP 成功候选的独立复核

固定 `trial-9cf6ebd687-param-cb3e0316` 的代码和参数，使用框架 replay 接口重新绑定相同调用身份，在冻结实现 `a1b4ae1` 下执行：

| 条件 | 父版本→候选 | 判定 |
|---|---|---|
| 原 workload，整步目标 | 146.083→125.422 ms | 两对收益均超过噪声，影子评价通过 |
| 保留 workload，sequence=256、K=2 | 95.412→93.863 ms | 正确性通过；第一对收益未超过噪声，拒绝 |
| 原 workload，通信诊断目标 | 79.184→73.531 ms | 正确性与整步约束通过；通信收益不稳定，拒绝 |

三次均完成两个连续训练步，每步比较约 2.49 亿元素；24 个替换站点实际执行，未观察到 `context.references` 回退调用。保留形状不是全部回退后通过，但仍不能宣称获得可靠性能收益。通信目标的第二对收益为负，两对均未超过各自噪声界限；干净 step 约 121.55／119.79 ms，满足固定原始基线的 161.422 ms 上限。不能将计算优化的整步收益直接改称通信优化收益。

三个 trial 分别为 `replay-93432c81704a`、`replay-f0dc21acf59b`、`replay-0fbd6be241ce`；以各自文件中的完整配对数据和质量记录为准。

## A、B、A+B 的实测交互

`run_interaction.py` 使用同一冻结原始父版本和相同质量／采样协议，分别测量已有计算候选 A、通信策略 B，以及原始两候选组合 A+B。代码和参数保持不变，三个版本全部通过质量和覆盖检查，均未提交。

| 版本 | 本次父版本→候选 step |
|---|---|
| A | 149.738→152.682 ms |
| B | 144.448→247.901 ms |
| A+B | 141.627→228.397 ms |

每行各自独立 AB/BA，父版本身份相同但测量样本不同。基于配对收益计算的交互点值为 **+19.628 ms**；组合各对极值的范围为 **[-2.520, 41.776] ms**，再按端点 MAD 界限扩展为 **[-34.763, 74.019] ms**。范围覆盖零，不能确定交互方向，也不能把点值分配给某一组。这些范围不是统计置信区间。B 与 A+B 的整体退化由各自完整项目测量直接支持。

[交互结果](p6-ddp-interaction-c/interaction.json)、[固定协议](p6-ddp-interaction-c/protocol.json)和三个端点报告保留实际来源。该次 driver 为 `649f16b`，core 为 `8a528212ba65a71f41f7ac938cacdc939c6595fff373d87e43c5c97528eabb9b`；目标单位为 ms。后续 driver 在协议内显式写入 objective、workload 和 request 哈希，不回改该已完成实验。

## Agent 与 worker 的解释器不一致

实际工具记录显示，部分生成角色通过 `/root/miniconda3/bin/python` 检查库源码；实测该环境为 Torch 2.8.0+cu128、Transformers 5.20.1。框架 worker 使用 `/root/autodl-tmp/dev-venv/bin/python`，实际为 Torch 2.13.0+cu129、Transformers 5.16.1。库 API 检查可能因此出错；这不代表已经在正确 worker 上测得的质量／性能结果失效，也不将历史失败全部归因为此问题。

项目配置现在为角色设置 worker Python 优先的 PATH，并提供 `inputs/worker-environment.json`。生成、同会话修复和恢复请求都明确要求使用该解释器。旧单算子配置保持不变；没有修改系统库或私有 provider 配置。

[真实验证请求](p6-worker-environment-validation-a-result.json)使用配置中的 GLM-5.3 实际执行两次 CPU 元数据查询，默认 `python` 与显式路径均得到相同的开发环境版本，4 条 assistant 消息，报告费用约 $0.01596。box4-2 上解释器配置与 provider 恢复的 4 项相关检查通过。正在运行的四个开发会话收到[环境更正](p6-worker-environment-correction.json)，通过服务端 `noReply` 消息补充事实，没有中止或重建原会话；已完成的正式对照未修改。

自动分支的后续推广、Qwen 有效多候选组合和其他正在生成的候选仍需等待真实结果，本目录暂不代表 P6 完成。
