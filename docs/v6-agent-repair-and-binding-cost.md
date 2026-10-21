# Agent 修复与绑定成本采集（2026-10-21）

## 责任边界核验

DDP 的约 100 ms 增量主要来自框架安装的 functional 绑定，不是 adapter Agent 生成的训练代码。
实际链路为：

1. adapter Agent 生成 `prepare/reset/run/observe/metrics/models`，复现原项目。
2. selection Agent 选择 group 与受影响调用闭包。
3. candidate Agent 通过 proposal 与 `build(context)` 声明实际替换站点。
4. `evaluation._worker_entry` 创建 `BindingSession`；后者为 functional 替换启用 `TorchDispatchMode`。
5. evaluator 在完整 `adapter.run()` 周围激活该 mode。

原 DDP 通信候选替换两个 `c10d.allreduce_.default` 站点，每步分别调用 1 次、12 次。
两者的 `site_parent` 都是 `project.invoke|run` 的 ID `5d7d6bb13691b02f533f62b0`。
它不是可解析的 module forward 边界，现有 scoped 后端仍会回退到 legacy。

因此改进责任分为：框架公开实际绑定机制与可测成本；selection/candidate Agent 根据完整项目目标选择
合适边界并生成实现。不能通过让 adapter 优化原基线来消除这项成本，也不能把所有退化归因于候选算法。
本轮先验证现有 module/project 边界能否解决问题，尚未新增绑定后端或改写搜索调度。

## 已实现的通用改动

- 新增可选 `binding_cost` 采集器，接入已有 collector registry 和采集规划 Agent。
  对保留父版本的新增 bundle，执行 parent/passthrough、passthrough/parent 四个独立端点。
  每个端点保留原连续质量检查、预热、重复测量与设备租约。透传省略新增候选的 build 和计算。
- 无候选或替换了父 bundle 时返回 `not_applicable`，不推断不存在的成本；执行或质量失败时保留端点证据，
  不给出耗时对照结论。成本属于完整调用及其组合，不分摊到算子组，不从验收得分中扣除。
- 修正诊断透传：只安装实际声明的 rank-local replacement calls，不把整个 affected closure 都安装成替换。
  这避免 module/project 候选的子算子闭包人为引入全局 functional dispatch。
- binding contract 解释 adapter、选择、候选和执行层的责任，以及 scope 名称与实际绑定站点的区别。
- 可选 `profiling.feedback_agent_plan=true` 复用现有采集规划 Agent，在候选实际结果出来后重新选择
  采集项。参数筛查全部未入选时，向 Agent 提供每个实际参数版本的小型结果摘要，由它明确选择一个
  revision 和采集项；不预设失败分类或下一步优化动作。默认关闭，旧输入保持原采集调用方式。
- 新增 `scripts/v6_diagnostic_feedback_experiment.py`：冻结原失败候选、父版本、常规反馈和任务，
  使用独立真实会话比较单项 `supplemental_evidence` 增补，再交给原参数搜索与配对 GPU 验证。
  结果仅发布到隔离实验分支。新版本按已有机制将完整 collector 记录放到索引附件中。

原项目入口可在部署配置中启用候选后采集规划，采集项仍由 Agent 根据实际端点和现有证据选择：

```yaml
project_defaults:
  profiling:
    feedback_agent_plan: true
```

已有显式 project request 则在自身 `profiling` 中设置同名字段。省略时默认关闭。
`binding_cost` 属于可选诊断，不改变真实候选的绑定策略、质量阈值或配对验收标准。

## 当前验证记录

第一批 collector、snapshot、observation 和 binding 相关测试在 box4-2 通过 14 项。
修正透传站点后，binding_cost、scoped、functional 相关测试通过 9 项。
本机完整 snapshot 测试曾因缺少 optuna 无法通过，已使用实际 worker 环境完成验证。

新采集器真实 DDP 探针的两个 AB/BA 对照，rank 最大完整步中位数分别为：

| 对照 | parent（ms） | binding passthrough（ms） |
|---|---:|---:|
| AB | 143.812453 | 261.473880 |
| BA | 155.667821 | 245.851864 |

四个双 rank 端点完整质量检查通过。它们证明存在显著绑定成本；端点变化包含测量波动，
不把某个差值当作固定开销常数。探针未提交优化版本。

## 真实 Agent 对照

首批实验固定框架 `95280f2`，仅新增独立实验入口，保持原框架执行行为。
每个任务从原失败现场建立 control/treatment 新会话，首次各一份，不声称统计显著性。

| 实验 | 角色 | treatment 的唯一额外事实 |
|---|---|---|
| Qwen TTFT | candidate | capture 改变 mask/GQA 分支的实测因果证据 |
| Qwen throughput | candidate | 有效 KV 窗口数值差异与缓存类型引发回退是独立问题 |
| DDP step | candidate | 实际 boxed ReduceOp ABI 及 P2P 从未执行的证据 |
| DDP communication | selection → candidate | 候选 build/计算完全省略后仍存在的大额绑定成本 |

这些是开发者提供诊断的受控修复实验。新的自主端到端运行需要另外从原项目入口发起，不能混称。
原质量阈值、通信目标和完整步约束保持不变，API 使用仅统计。

运行目录：`/root/autodl-tmp/dev/v6-work/agent-repair-20261021`。
下文分别记录实际决定、执行路径证据、独立复测和未解决问题。

## 中途环境故障与恢复边界

本轮遇到 box4-2 的 `/root/autodl-tmp` 122 GB 数据盘耗尽，四组修复实验和第一批新入口验证中断。
保留中断日志与已经生成的候选，不把缺少完整终态记录的运行当作成功实验。
恢复的 control 使用相同候选字节重新执行完整 GPU 验证；新 treatment 从冻结输入重新发起真实会话。
新入口自主验证另行重跑，明确区别于开发者恢复的修复实验。

空间处理先对已结束历史目录中 819 个内容完全相同的文件逐项校验 SHA-256，以硬链接保留原路径和字节，
共释放约 6.16 GiB（跨系统盘和数据盘）。活动实验、运行数据库和 provider 配置不参与去重。
一个传输 ZIP 在确认本机副本 SHA-256 相同后删除，原诊断文件仍保留。
早期 `stage2-autonomy-0e4c751` 批次的完整冷归档已转存本机，4,445,620,276 字节，
SHA-256 `0f24eae78103c62d2f8e1908d4e769a3f180fbea1f86357c5b0ef2e8fb204d2c`。
本机与服务器归档哈希一致后，移除服务器该批次的展开副本，原路径保留 `ARCHIVED.json` 恢复说明。
数据盘当时恢复至约 14.6 GiB 可用，随后继续修复实验。后续还归档了冷历史目录及大 reference/trace，
最后一个 `stage3-variants.tar.gz` 为 7,692,523,298 字节，SHA-256
`48e23844bb6bce1978feb8e983091146f2a90041e82d78bd09d0a7b407735e09`；本机核验并确认没有
活动进程引用原目录后，服务器仅保留恢复说明，数据盘恢复约 19.27 GiB 可用。
六份私有冷归档均保留在本机 `.scratch/remote-cold-archive`，不会随公开证据上传。

## 采集规划输入的最小修正

真实的候选后采集规划已经自主选择 `binding_cost`，说明新增采集器可以通过现有 Agent 使用。
但首次规划把 intrusive discovery 的通信核时间约 284.7 ms 与干净候选测量直接比较，
而当时输入没有给出原始干净通信基线 94.49 ms。为此仅在 `evaluation.json` 增加
`original_reference`：原始 revision、逐 rank 的干净指标及来源解释；不附加原始大日志。

冻结同一生成前快照、同一 prompt 和模型，仅增加这项输入，在新真实会话中重放：
Agent 改为使用 94.49 → 71.35 ms 的干净测量，保留完整步 227.04 ms 超过 151.86 ms
约束的判断，选择相同候选 revision，仍请求绑定成本诊断；未再用 284.7 ms 宣称优化收益。
这是单次规划对照，证明该信息修正了本次基准混用，不代表优化成功或统计稳定性。

## 已完成的修复对照结果

Qwen TTFT 的两个新会话均恢复了完整质量。各自的 AB/BA 配对结果如下；两组基线不同，
不能把两个候选的绝对耗时差直接解释为诊断的因果收益。

| 分支 | 原路径 TTFT（ms） | 候选 TTFT（ms） | 判定及实际路径 |
|---|---:|---:|---|
| control | 24.2007365 | 23.5309945 | 通过；保持原数值路径的 eager composite |
| control 同代码独立复测 | 24.6940765 | 23.4476975 | 通过；不重新生成或改参数 |
| control held-out：batch=2、prompt=768 | 100.5099285 | 100.653433 | 质量通过，性能拒绝 |
| treatment | 26.261408 | 21.3507085 | 通过；保留 eager mask/GQA 分支后进行 Graph capture/replay |
| treatment 同代码独立复测 | 26.0657465 | 21.485022 | 通过；不重新生成或改参数 |
| treatment held-out：batch=2、prompt=768 | 100.660126 | 100.5209 | 质量通过，性能拒绝；候选签名保护回退原实现 |

treatment 的独立时间线原生记录包含 `cudaGraphLaunch`。原 workload 的收益约 18.7%，
独立复测约 17.6%；held-out 没有有效加速。不能由此声称该固定形状候选已经泛化到新形状，
也不能由单次 control/treatment 生成声称诊断是获得优化的必要条件。

Qwen decode treatment 在两个完整生成上均达到零误差，62 次目标调用全部进入候选，
框架签名 fallback 从原失败案例的 60 次消失。但候选的 Graph 自检失败，实际执行
`eager_window_parent`；配对判定因 `paired_gain_not_above_noise` 和 `objective_not_improved`
拒绝。另一个开发者诊断副本只打印被捕获的原异常，未修改数值计算：
`RuntimeError: captured step diverged from eager parent`。因此问题从原来的数值／cache 接口错误
缩小到 capture/replay 的数值一致性；仍未获得可靠吞吐收益。
复查表明现有 `candidate_diagnostics` 已在参数筛查后自动捕获同一异常；打印副本只是开发者复核，
不应把它描述成框架原本没有的异常采集能力。

DDP step treatment 修复 boxed ReduceOp 判断后，Agent 在真实 CPU Gloo 自检中遇到多次 P2P
匹配超时，最终改用 `reduce_scatter_single` 加 `all_gather_single`。这不是原 P2P 实现成功。
真实双卡连续质量检查零误差，每 rank 的 13 次目标调用均进入候选，没有框架 fallback；
候选自报 `split_allreduce`，与生成代码一致。完整步由 141.1253735 增至 255.8519155 ms，
因此被拒绝。该候选仍安装两个 functional 站点；修正 guard 没有消除绑定层开销。
执行路径被修复与获得性能优化是两个独立结果。
control 也独立生成了 reduce-scatter/all-gather 路径，完整步 146.284576 → 265.7569635 ms，
同样被拒绝。两个分支的原生时间线均在每个 rank 记录 13 个 ReduceScatter 和 13 个 AllGather
kernel，证明新通信实现实际运行；不从 kernel 名称推断实际 NCCL protocol。
因此单次对照不能证明额外 ABI 证据是修复的必要条件，两者都没有解决 functional 绑定的性能障碍。

DDP communication treatment 的 selection Agent 在看到透传成本后自主改选 `model.lm_head`
module 边界，candidate 只安装该 module 替换。三个参数实例均质量通过且满足整步约束：
step_ms 为 144.675915、145.303522、138.327063，原限制为 151.86385935 ms。
正式配对通信目标为 71.2905145 → 63.9656575 ms；两对收益 10.286786、4.362928 ms，
对应噪声界 3.438766、11.268976 ms。第二对没有越过噪声界，框架拒绝该候选。
正式候选整步 140.960518、142.44106 ms，均满足约束。

对 Agent 实际选择的 DDP module 边界另做纯透传探针，四个双 rank 端点质量均通过。
AB 的 parent/passthrough 为 149.690369/136.490750 ms，BA 为 140.377092/143.108565 ms。
没有重现 functional 绑定约 100 ms 的大额增量；两对差值方向不同，不能将其中一个当作固定成本，
更不能据此宣称绑定本身带来加速。

这支持“现有 module 绑定可消除阻断该候选的主要全局 dispatch 开销”，但不支持
“已经获得稳定通信收益”或“PCIe 传输变快”。候选仍调用原 collective，主要改变 host 到达等待。
采集规划 Agent 自主选择的五项后续诊断均已实际完成：binding_cost、execution_timeline、nsys、
nccl_tuning、collective_calibration；其候选选择和结果来源均保留。

## 新入口验证范围

另从原项目、命令和目标启动 Qwen TTFT、ResNet latency、DDP communication 三种新任务，
框架固定为 `88b9651`，启用可选 `profiling.feedback_agent_plan`；未提供 adapter、分组、候选或人工诊断。
Qwen 与 ResNet 的新 adapter 已通过原项目模型／输入构造及完整输出的独立人工核验。
新入口结果与上述诊断辅助修复分开统计。

ResNet 已完成两轮自主优化。第一轮 BN folding 的干净配对延迟为 4.227431 → 4.181138 ms；
第二轮根据实际反馈重新选择 12 个组、28 个 module 站点，以 joint bundle 替换第一轮 bundle，
将 conv+bias+ReLU 以及残差 epilogue 融合，配对延迟为 4.1503545 → 3.455310 ms。
连续完整 logits 与 top-1 检查通过，最大绝对误差 9.06e-6；未改变原质量阈值。
已有自动时间线记录 9 次 `aten::cudnn_convolution_relu` 和 8 次 `_add_bias_relu_kernel`，
为实际原生融合执行提供证据，而不只依赖候选自报路径。

最终完整版本再由开发者启动独立 AB/BA，与其自身原始项目比较：4.225842 → 3.4788785 ms，
延迟降低约 17.7%，质量通过、coverage 无误、两对收益均越过噪声界，验收通过。
独立复测没有重新生成或修改候选，也没有发布到其他任务；这一结果针对当前 batch=32 workload，
没有追加新形状泛化结论。新入口搜索本身没有接收人工诊断。

Qwen 新入口也已完成两轮。第一轮 Graph 自检失败后全部回退原路径，24.8862215 → 24.755876 ms，
配对拒绝。第二轮改为 Triton elementwise fusion，框架配对记录 26.191631 → 24.694290 ms 并通过；
但两个质量调用都自报 `parent_fallback_validation`，执行的是原父路径，不能计为融合优化成功。
完整版本独立复测 23.9387035 → 26.1267055 ms，质量通过但性能被拒绝。

既有 candidate_diagnostics 自动捕获第二轮 `_close()` 返回 False。生成代码将内部容差设为
0.01，严格于冻结的 0.02。为核实差别，开发者诊断副本只在原比较处输出误差统计，保持计算、
返回值、阈值及回退控制不变，也不发布候选：prefill logits 最大绝对误差 0.28125，
超出内部 0.01 的元素 50,297 个，超出正式 0.02 的元素仍有 15,511 个。
因此这次自检确实阻止了不满足正式容差的 prefill 输出，不能靠放宽自检解决。
这个诊断只覆盖早退处实际比较的张量，不冒充完成了后续被跳过的 decode 数值检查。

该新入口仍使用冻结的 `88b9651`，尚不含后文 `6ed684c` 的路径摘要修正。
采集规划的空计划及对第一次“Graph 收益”的误述均保留为真实失败证据；实际第一次 Graph 没有生效。

### 新入口发现并修复的可用性接口问题

第一次新 DDP 入口完成了 intake 和 evaluator 验证，但 profiling 因
`required_observations=['communication_kernel_sum_ms']` 被拒绝。实际字段已经被 normalize 采到，
`available_fields()` 却只报告较粗的 `nccl_kernel_duration` 等 capability 名称。
这不是 adapter 丢失计算、硬件无法采集或 Agent 使用了不存在的 metric。

`e16e80f` 的修正让 `status=observed` 时实际存在的有限数值字段也能满足 required observations，
同时保留原 capability 名称；None、非有限值及 `not_observed` 下的默认零值不冒充测量。
没有增加 DDP／模型名称分支或特定 metric 的别名表。
原请求、原 adapter 不变的双卡 profiling 复测通过，两个 rank 均有 13 个 NCCL kernel；
原始 diagnostic 数值及其 intrusive 属性保留，未作为干净性能分数。相关 16 项测试通过。

该次 DDP adapter 的独立原项目比较通过：原项目串行计算同一 global batch，比较两次连续更新的
完整梯度、参数和 loss，同时检查两 rank 的梯度完全相等。生成 observer 使用 `state_dict()`
包含 tied-weight 别名，人工核验的期望值也采用原模型的完整 `state_dict()`；未改 adapter 输出。
另在 `e16e80f` 从原项目／命令／目标重新启动 DDP，先前失败的新入口运行仍保留。
重跑生成的新 adapter 也通过原项目独立核验：除了完整参数、梯度、loss 和完全相同的 replica
梯度，还逐项检查了它额外输出的每个 rank 的 normalized micro-batch loss；没有删去额外输出。
源码复核也确认 micro-loss 的 detach 累积原项目本就存在，额外 CPU 转换在 `observe` 中；
训练计时主体仍保留相同累积、SGD 更新和 synchronize，不能据此解释约 100 ms 的绑定增量。

该新入口两轮候选均未通过整步约束。第一轮在两个 functional 通信站点创建第二个 NCCL group，
尝试三组 channel/protocol 参数，质量均通过，但整步 253.862262–295.482941 ms，超过
144.80321415 ms 的限制。第二轮继续选择同一组 functional 站点，保留原默认 NCCL group 并
在首 bucket 前加入 CPU 到达对齐；两个 rank 的连续检查零误差，整步仍为 249.802865 ms，
通信目标筛查值为 80.530106 ms，同样没有可行参数，未进入正式 AB/BA 验收。
这不是稳定通信优化成功；也不能将第一轮的全部退化归因于 NCCL 参数本身，两个候选仍承担
全局 functional mode 开销。这里的“全部未入选”指参数筛查，不是 framework fallback：
实际目标调用均进入候选，原 collective 作为候选组合的一部分继续执行。

第二轮后，冻结框架中的采集规划 Agent 自主选择 binding_cost、execution_timeline、nsys、
nccl_tuning、collective_calibration，五项均实际完成。纯绑定对照的 parent/passthrough 为
139.964257/232.282689 ms 和 146.200234/247.006700 ms，再次测得约 92–101 ms 增量。
这次采集没有人工指定采集器或补充诊断，原始两轮流程已正常结束，未产生可接受版本。

随后从这份实际反馈启动一轮隔离续跑，框架冻结为 `ce10994`。开发者只触发续跑，使用的反馈
字节与原 `latest-feedback.json` 完全相同（SHA-256
`5c7eaa68bc4ec3f0975c31c851d444357487f425a4348eaf91c0283a3849c59f`），未指定分组、算法或参数。
它与原始两轮新入口分开记录，不能称为一次从入口开始、无人触发续跑的三轮实验。

这轮续跑已经完成。selection Agent 根据自动获得的 92–101 ms 绑定诊断，选择 project-kind
整体调用组；candidate 只替换整个 invocation，实际 worker 记录为 `effective=direct`、
`reason=no functional replacements`。虽然 selection 的 scope 字段是 `group`，该组覆盖的是
整个项目调用，不能仅根据 scope 名称将其理解为局部算子优化。

Agent 尝试将前三个 no_sync micro-batch 的计算录入 CUDA Graph，保留最后一次 eager DDP
backward、默认 process group、reducer、SGD 和同步。但 capture 失败，候选随后回退父实现。
既有 candidate_diagnostics 在两个 rank 都自动捕获 `_capture_segments` 内 `raw_model(...)`
调用处的 RuntimeError：Graph capture 中的 CPU→CUDA 拷贝要求 CPU tensor 为 pinned memory。
当前诊断没有定位到被调用库内部究竟哪个张量产生这次拷贝，不能将某个 mask/loss 常量认定为根因。

筛查整步为 137.209215 ms，质量通过且满足约束；正式 AB/BA 的通信目标为
66.547669 → 73.144745 ms，两个候选整步为 145.554415、144.248580 ms，其中一组超过
144.80321415 ms 限制。两组通信收益均未越过各自噪声界，最终拒绝，未产生可供独立版本复测的
已接受候选。连续质量检查零误差对应的是内部父路径回退，不能用来证明 Graph 计算正确或优化生效。

本轮后续规划识别了 capture failure 与表面计时的区别，自主请求六项诊断，均已执行：
candidate_diagnostics、execution_timeline、nsys、nccl_tuning、collective_calibration、binding_cost。
项目级纯绑定对照为 parent/passthrough 142.288805/148.414764 ms 和
139.560733/142.112600 ms，没有重现约 100 ms 的 functional 增量，但仍包含若干毫秒的波动／开销，
不能称为零成本。证据支持“自动成本反馈促使 Agent 改选更合适的绑定边界”，不支持
“已经获得稳定通信收益”或“Graph 已成功执行”。

### 根据真实信息缺口保留候选路径记录

新 Qwen 第一轮候选的两个调用均自报 `parent_fallback_setup_failure`，实际 Graph 自检错误
`replay logits differ bitwise from eager parent output` 已被原有 candidate_diagnostics 自动捕获。
首次后续采集规划的 endpoint 摘要只保留调用／reference 总数，丢掉了路径记录；
Agent 选择 binding_cost、execution_timeline、python_runtime，主要讨论约 0.13 ms 的噪声级差异。

冻结该次生成前快照、模型、prompt 和其他输入，仅在 evaluation.json 补回实际
`candidate_declared_routes` 及“自报信息需证据核实”的来源说明，在新真实会话中重放。
Agent 转而指出两个调用都声明 setup fallback，选择 candidate_diagnostics 和 execution_timeline，
将问题收敛到实际 setup 异常和执行路径。代码因此只保留该既有路径字段及来源说明，
不根据标签推断故障类别、改写搜索动作或改变验收。此单次规划对照不证明性能收益。

在第二轮“形式验收通过但全部回退”的真实现场又做一次相同类型的冻结重放：原规划选择空列表；
仅补回实际路径记录后，Agent 明确指出 26.192 → 24.694 ms 的表面收益与两次
`parent_fallback_validation` 的矛盾，改选 candidate_diagnostics、execution_timeline、nsys。
这次仍只验证规划生成变化，未把它算作新优化实验，也未按候选自报标签直接修改验收。

### 原始基线的绑定身份

新 DDP 第一轮三个参数实例的 step_ms 为 253.862262、295.482941、289.416067 ms，
均超过原始 137.907823 ms 的 1.05 倍限制 144.80321415 ms。后续规划错误地认为父子版本
具有相同 legacy 绑定，因而排除 binding_cost；实际上原始 revision 的 bundle 列表为空，
而候选安装了两个 functional 替换。零 framework fallback 也不能排除 dispatch 成本。

冻结该生成前快照，仅在 `original_reference` 加入其真实 revision 的 `bundles` 列表，
新真实会话改为选择 binding_cost、execution_timeline、nsys、nccl_tuning、collective_calibration，
指出整步约增加 116 ms 而通信只增加约 25 ms，不能直接归因于 NCCL 参数算法。
代码据此只保留这份原始 bundle 身份，不填充缺失的 endpoint setup，也不规定必须选哪个采集器。
单次重放证明本次规划选择发生变化，不证明这些额外诊断一定使下一轮优化成功。

一个先行探索重放曾将缺失 setup 归一化为空列表；它的输入、会话和费用均保留，另附来源复核，
不作为代码修改的验证依据。有效对照使用上面的真实 revision bundle 列表。
对应代码 `ce10994` 已在 box4-2 通过 9 项相关测试，并推送到 GitHub v6。

## 结论边界与后续缺口

本轮改进了框架绑定成本的可观测性、诊断采集规划及信息来源，没有实现新的低成本 functional
绑定后端。现有 module/project 边界可以避开部分全局 dispatch 成本；当选定站点仍只能由
functional mode 执行时，这项成本仍在。是否需要新增绑定后端应依据后续真实任务决定。

ResNet 的自主多组联合优化及独立整体收益成立；Qwen 的诊断辅助 Graph 收益成立，但新形状无
有效加速。新的自主 Qwen fusion 未成功，仍需对真实 CUDA 数值路径逐段定位，不能放宽质量要求。
通信实验尚未证明稳定的传输优化；CPU 到达对齐即使降低 NCCL 核时间，也不等于 PCIe 传输变快。

原计时验收可能接受内部全部回退的候选，这次由独立复测和路径证据识别出来。
通用的有效执行路径验收仍是缺口；没有根据候选自报标签添加硬编码黑名单或改写搜索逻辑。
路径摘要与原始 bundle 身份的修正已用冻结输入、真实会话验证规划变化，本轮冻结新入口
不含这些后续提交；不能把分开的证据组合说成最新版已完成一次全新的自主端到端运行。

原始记录、候选、生成前快照和哈希索引见
[本轮证据目录](evidence/v6-agent-repair-20261021/README.md)。

## 失败定位与后续验证（2026-10-21，第二批）

第二批记录位于 `evidence-followup-20261021`。以下首先记录已结束的定位与对照；使用定位信息的
Agent 修复实验，以及冻结 `a6f2b6e` 的全新 Qwen／DDP／ResNet 原入口运行，另行记录终态。
两类运行的输入不同，不能将诊断辅助修复称为无人工信息注入的自主成功。

### Qwen：定位到实际 GPU 数值差异

在原失败融合候选上，仅增加边界读数，未改变计算、内部检查或回退。36 层共 470 个边界记录显示：
第一层 input norm 和 Q/K/V projection 与父实现逐项一致；首个差异出现在融合的 Q/K norm＋rotary，
Q 最大绝对误差 0.0625、134,052 个元素不同，K 为 0.25、33,908 个元素不同。
最终 logits 最大误差 0.28125，15,511 个元素违反正式 `atol=rtol=0.02`，复现此前的诊断。
CPU 拷贝和 hooks 会干扰运行时间，这些读数只用于定位，不作为性能结果。

在同一实际 GPU 张量上进行第一层 shadow 计算：独立 RMSNorm kernel、接原生 BF16 rotary、
融合 kernel 的 norm-only 变体均逐项一致；原融合 kernel 开启 FP fusion 时出现差异，关闭后逐项一致。
这证明第一层差异与该编译选项相关，尚未反汇编确认具体指令，也不证明全模型修复。

进一步只修改两个 norm＋rotary launch 的 `enable_fp_fusion=False`，完整候选仍回退。
边界记录显示前两层一致，第三层 Q rotary 有一个元素相差 0.0078125；最终 logits 最大误差
0.3125，仍有 14,631 个元素违反正式容差。第三层 shadow 计算进一步发现独立 Q RMSNorm
已有同一个单元素差异；K norm 也有一个小差异，但经 rotary 后舍入回一致。
因此还存在归一化计算的数值差异，不能把关闭 FP fusion 或放宽内部检查当作完整修复。
尚未证明具体 reduction／rsqrt 指令是原因。

两个派生诊断的基础 protocol 曾继承“计算未改变”的描述；各自 `intervention-scope.json`
明确更正为修改了上述编译选项。原记录和更正同时保留，避免把干预实验误称为纯读数。

### DDP：捕获到库内部的真实异常来源

在原失败候选的既有异常处理处仅打印完整 traceback，两个 rank 均定位到
`transformers/masking_utils.py:603` 的 `eager_mask`：
`torch.where(mask, torch.tensor(0.0, device=mask.device, dtype=dtype), min_dtype)`。
这里创建标量触发 Graph capture 中未 pinned 的 CPU→CUDA 拷贝；实际训练输入已在 GPU 上。
它是本次第一个已确认的 capture 障碍，不代表消除它后所有后续操作都能捕获。

据此做了一项通用采集修正（`a6f2b6e`）：`CandidateTrace` 在传播到候选的异常上保留 traceback
的文件、函数、行号与源码文本，包括被调用库的位置。仍只 trace 原候选文件，不持有 frame、locals
或 tensor，不改变候选计算和验收。相关 8 项测试通过；使用未修改的原 DDP 候选在双卡复测，
框架自动采集到两个 rank 的同一库内位置，无需开发者插入 print。

### 冻结候选生成的执行证据索引对照

`b6487df` 增加独立实验入口。固定原第二轮候选生成前的快照、模型、prompt、父版本和框架
`88b9651`，control 保留原始输入，treatment 仅在原反馈开头加入已有路由／异常的紧凑索引，
原反馈全文保留。两边分别使用新真实会话，随后执行相同参数筛查、完整质量检查和 AB/BA。

| 分支 | 配对父版本 TTFT（ms） | 配对候选 TTFT（ms） | 结果 |
|---|---|---|---|
| control | 25.672291 / 24.247710 | 25.309170 / 25.057786 | 拒绝；全部 setup fallback |
| treatment | 24.843216 / 31.334290 | 27.482147 / 25.919362 | 拒绝；全部 setup fallback |

两个 Agent 都正确识别了旧候选实际回退，control 也自行找到了原反馈中深层的证据。
两者生成的 Graph 自检仍失败：control 超过正式 0.02 容差，treatment 的最大 logits 差异为
0.3125，超过其 0.015 内部容差。因而这次对照没有证明索引改善修复结果；索引保持为实验工具，
不加入默认候选输入，不据一次比较作因果或统计显著性结论。

第二批原始证据、生成代码、输入快照和哈希索引见
[后续验证证据目录](evidence/v6-evidence-followup-20261021/README.md)。
