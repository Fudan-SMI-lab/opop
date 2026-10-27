# 次级诊断指标接口的隔离验证

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
