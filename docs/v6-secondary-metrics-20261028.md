# 次级诊断指标接口的隔离验证

## 终态审计与采用范围

2026-10-28 00:30:40 UTC 四阶段完成。五个真实双卡端点（固定 off/on/on/off + intake mapping 集成）质量、两步连续状态、覆盖和 FP32 policy 检查通过；[完整小型审计](evidence/v6-secondary-metrics-20261028/audit.json)保留全部 25 个 aligned clean 样本、18 个 rank-local profile 样本及 raw trace SHA。逐 raw trace 重算通信 sum、先取 rank 最大再取中位数均一致。五个端点除次级映射外参数一致，源文件/adapter/candidate 完全一致；overlay 全部文件与改动清单逐字核验。

| 固定端点 | clean step 中位数 ms | 次级 NCCL sum 中位数 ms |
|---|---:|---:|
| off 0 | 70.0886 | 未采集 |
| on 1 | 70.1034 | 33.4753 |
| on 2 | 69.9316 | 33.3771 |
| off 3 | 70.1820 | 未采集 |
| intake mapping | 69.8401 | 33.0703 |

没有新候选接受，也不据此声称采集无资源成本。原先 Graph 候选 40.61% 独立接受与这些诊断端点分开。

真实 intake 两臂均一次生成成功、零合同修复。control `ses_e8036c9…`，treatment `ses_e802480…`；初始实际快照只改变 schema 文件（8,684→10,621 bytes），prompt/模型/工具/其他输入相同，两臂共同使用同一隔离核心。删除新增 schema 字段/定义后与原 schema 完全相同。两臂保留完整 global batch、FP32/noTF32、全量梯度/参数质量、精确 rank 关系、完成同步、max-rank-then-median 与 1.05 倍原基线限制。treatment 主动选择 `max_rank_nccl_kernel_sum_ms` → `torch_communication.communication_kernel_sum_ms`，映射执行成功。输出文字并不完全等价：treatment 将线程数列入 source-required numerics，control 更明确允许线程设置替换；两者均保留 DDP/no_sync/foreach 的实现自由度。

但本次不能宣称完整请求可直接完成原入口流程：

- treatment 的 `required_observations` 写成 `torch_communication.communication_kernel_sum_ms`，现有 discovery 的真实可用名是 `communication_kernel_sum_ms`。用保留的真实 discovery/capability 字段集合重放现行名称检查，该 required name 不可用；本批仅提取 metric mapping 的集成没有经过这一步，因此没有暴露错误。这是代码/证据重放结论，未新跑原入口 discovery。
- treatment 在 `source.profiling.metric_repeats` 写入 3，但现有采样器读取的是 request.profiling.metric_repeats（默认 3），该嵌套设置被转交 provider 后未使用。实际恰好采到 3 次不能证明该 override 生效。
- CPU fixture 实测默认关闭、双指标输出互不覆盖、旧 objective.source 共存、缺失值/字段/名称冲突拒绝及 inner-loop 聚合；最初 fixture 没有专门注入 NaN/Inf，非有限值分支此前只有代码检查，不能把它当作已完成的动态测试。

据此分开推进：采用已验证的**显式 request 运行时接口**，默认不采集次级指标，空映射保留旧 portable request；不采用自动 intake schema。运行时追加非有限值检查测试并在生产版本做一次真实双卡验证。自动映射只做新的小型 schema 说明对照：解释 discovery 字段名、输出别名和 provider 参数/框架采样配置的区别，不自动修正模型输出，不按错误类型路由，也不以重复直到通过替代分析。并继续既定 Qwen 同 workload decode 目标变体，避免工作停留在 DDP 重复诊断。

当前实际数据盘可用 40,323,956,736 bytes（约37.56 GiB），终态无 GPU worker。旧结果和所有原始样本保持，生产 intake/prompt 与 allocator 均不作自动采用。

起因：[原入口 DDP 整步任务](v6-ddp-step-followup-20261028.md)已获得 40.61% 独立收益，但主目标改为 step_ms 后，`required_observations` 只验证 discovery 能力，不会自动向 evaluator 添加通信样本。不能把后来手动补采的数据回填冻结端点。

本批只在复制的框架源码中验证最小原型，生产代码和提示词不变。`v6_secondary_metrics_prototype_20261028.py` 使用精确锚点生成隔离 overlay，并记录全部改动文件 SHA：新增通用 `diagnostic_metrics` 显式映射，经现有采集器注册表解析 collector/field，不按模型或指标名硬编码。默认空映射；每个已声明指标必须实际采到，否则明确报错。来源未验证或环境不适用时，由 intake 留空，不能编造零值。

采集在全部 clean 测量之后进行，次级样本、采集证据和指标定义单独保存；不改变主目标、clean 样本、质量、约束或接受判据。不同指标使用不同输出目录，避免覆盖旧 objective.source 或其他指标的 raw trace。次级指标带自己的 rank/sample 聚合定义，并传给现有 inner loop 的指标摘要。collector 失败的处置沿用显式采集失败语义，没有新增失败类型路由。

本批顺序固定：

1. CPU 集成检查：实际 evaluator 的 controls、默认关闭、两个次级指标、主目标 collector 与次级 collector 共存、文件互不覆盖、缺失值/字段/名称冲突报错、交替慢 rank 下 inner-loop 聚合。仅测试采集接口，不将 fixture 当作真实通信实验。
2. 固定新 DDP 的已接受 Graph 完整版本 `09b71766…`，按 off/on/on/off 顺序执行；同一原 adapter、候选、参数、完整质量、两步轨迹、5 个 clean 样本。on 额外调用已有通信采集器，另取 3 个 profile 样本。所有样本保留，不进行正式候选接受，不用分数决定是否重跑。
3. 从真实 intake 初始快照 `generation-03a5c9d61b0a` 做一次新会话 control/treatment。两臂同用隔离原型核心，显式记录相对旧快照的共同 core 变化。初始输入只改 `inputs/intake.schema.json`：增加可选字段及其定义/说明，8,684→10,621 bytes；删除这两个新增 schema 节点后与原 schema 完全相等。prompt、原目标、源项目、部署、worker 环境、模型和工具设置不变。检查实际生成请求是否保留完整语义，并由 Agent 自主选到适用次级指标。
4. 若真实 treatment 选择了指标，只提取它的 metric mapping，在既有已核实 adapter/candidate 上执行一次集成检查。没有选择则记录 `not_selected`，不人工补写。这是接口集成，不是新原入口 evaluator 或优化 cohort。

预检：服务器隔离 overlay 的 CPU 集成通过；相关 quality/measurement/intake 测试共 21 项通过。初次 CPU 检查错误地假定样本只有两个 adapter 字段，遗漏现有 `framework.elapsed_ms`，在 `cpu-corrected` 保留修正后的执行；原失败目录保留。首次 pytest 命令使用了不存在的文件名，未运行任何测试；改用实际 `test_v6_intake.py` 后 21 项通过。未修改生产实现来迎合测试。

终态需核查真实快照差异、完整请求/合同语义、所有原始 clean/profile 样本、实际源文件/参数/overlay SHA、rank 聚合和独立 trace 重算。采集发生在 clean 测量后不代表其资源生命周期没有成本；不同端点的数值变化也不能自动归因为此接口。次级数值不等于 wire time 或关键路径。完成审计后再决定生产采用，随后继续目标切换/更强通信案例。

目录 `/root/autodl-tmp/dev/v6-work/secondary-metrics-20261028`。原始数据留服务器，只将代码、索引和小摘要提交 GitHub；无本地大归档。启动前核查实际容器 df 及 GPU 空闲，单控制器、一个普通终态监听器；不启用 Goal、额外送达测试或模型轮询。旧 DDP 两种接受/通信拒绝、Qwen/ResNet 拒绝及全部样本、TP 完整质量失败和 memory adapter 历史事实保持不变。allocator 合并仍未采用。

启动记录：冻结 `f43046fc2df77c0965d33473cf0a479314f34bcb`，2026-10-27 23:52:40 UTC 启动 controller PID412316；[launch](evidence/v6-secondary-metrics-20261028/launch.json)、[protocol](evidence/v6-secondary-metrics-20261028/protocol.json)、[overlay 文件哈希](evidence/v6-secondary-metrics-20261028/overlay-manifest.json)已留档。生产核心仍 `cb4ea28c…`，隔离核心的真实 SHA 由 protocol/worker 逐次记录。启动 df 可用 40,440,446,976 bytes（37.66 GiB），GPU 无计算进程。

启动核查：本批 CPU 检查及固定 off/on/on/off 四个双卡端点全部 valid，全部样本留服务器，尚未把数值变化解释为接口性能影响。真实 control intake 会话 `ses_e8036c9a6ffe7aedh9CJYoD1ck` 已建立，初始快照 `generation-e4376e067093`；等待两臂和条件式集成完成后再审计生成语义与所有输入差异。相关 21 项[预检日志](evidence/v6-secondary-metrics-20261028/precheck-pytest.txt)及[CPU 摘要](evidence/v6-secondary-metrics-20261028/precheck-summary.json)留档。唯一普通监听 PID19852，状态 waiting、controller 存活，[监听设置](evidence/v6-secondary-metrics-20261028/wakeup-setup.json)已保存；旧批次监听已提交一次续行后退出，没有恢复旧归档监听。
