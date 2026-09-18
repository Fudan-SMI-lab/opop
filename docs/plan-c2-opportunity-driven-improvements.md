# C2 条件响应驱动的结构机会：完整实现与验证计划

日期：2026-09-18。状态：已获准撰写计划，尚未实现或执行；本文件是本轮唯一规范正文。
执行索引：[短清单](../.omo/plans/v5-c2-opportunity-driven.md)。范围根目录为 `D:/Pyhon_projects/opop/v5`。
本次仅新增两份计划，不提交、不推送、不调用模型、SSH 或 GPU，不激活 boulder，不修改旧计划、源码或结果。

## 1. P0：先改变 C2 如何发现和实现有价值的结构机会

核心问题不是“哪里资源最多”，也不是“怎样做一般性的全任务融合”，而是条件响应揭示了什么值得追求的方向。
固定方法链：**实测条件 objective/parameter/resource response → 固定伙伴下的有价值方向 → 源码限制、成本增长或耦合 → 连贯结构动作 → 预期新有用联合区域 → retune 与合资格原生扩域 → 实现的全任务净收益**。
whole-task 是沿此链核算收益和必要配套的尺度，不替代 C2 机会推理，不把 C2 改成通用 fusion optimizer。
例如局部方向虽有收益，伙伴 tile、流水、布局、物化或同步成本可能抵消它；结构动作须解释怎样改变这些条件。
可能需要更高寄存器或 shared-memory 使用，也可能失去父结构的旧最佳点；资源下降和保留旧点均不是结构接受条件。
不要求存在硬件硬墙：连续成本增长、spill、occupancy 权衡、伙伴参数耦合都可支持有限假设。
端点有限差分只是固定伙伴条件下的响应，不是全局梯度，更不是资源导致延迟变化的因果证明。
aggregate profile 是上下文，不能据其峰值宣布瓶颈；domain 边界不是编译拒绝，未测不能补成零。

### 1.1 提示中实际要完成的推理

1. 指认本次响应所属源码、selected params、目标方向、单位、端点和固定伙伴；保留失败及覆盖缺口。
2. 从有效端点找值得追求的方向或 nominal contrast；说明它只在哪些伙伴条件下成立，不硬造斜率。
3. 对照源码解释限制来自何处：生命周期、重复读取、归约、物化、索引、调度或伙伴耦合；区分事实与推断。
4. 提出一个连贯结构动作及必要配套，解释它为何可能改变该方向的收益或代价，而非列无关优化菜单。
5. 描述目标联合区域及伙伴关系；能命名轴和值时具体写，不能时写数据流目标与未知，禁止伪精确数值。
6. 明确新增成本、正确性风险和可能失效条件；最终由新结构自身调优后的完整任务表现裁定。
7. 结果分析逐环对照“主张、实现、测量”，失败或未进入目标区域也完整保留，不把意图直接当成实现证据。

### 1.2 落到现有接口，而非再建证明系统

`src/kernel_optimizer/agents/method_prompts.py` 增加独立 C2 opportunity guidance，仅在实际供应 conditional responses 时启用。
`whole_task` 的一般任务指令两臂相同；G0 不供 fresh responses，不能仅凭臂名启用 C2 专属机制指令。
空响应、缺测或无有效 contrast 时保持事实边界，允许源码推理并注明响应不足，不捏造机会。
`legacy_local` 保留兼容但本轮不使用；不新增第三种实验臂或另一套一般优化提示。
`agents/modules.py` 中 analyst 的现有 `summary`、`hypotheses.change/expected_effect/risk` 承载上述链条。
rewriter 的现有 `change_summary` 承载证据、价值方向、限制、源码动作、目标区域、伙伴、成本和不确定性。
不新增 mandatory proof schema、额外 LLM deliberation 调用或语言表述验收门槛；不要求量化预期收益。
实际 compiler wall 仍走独立 `wall_text`/`analysis/resource_walls.md`，不与 conditional response、软成本或 analyst 判断混写。
审查现有“profile 已排除假设”“某路径一定最有价值”等强断言，让实测条件证据优先，避免一般提示盖过 C2 主线。

### 1.3 当前 selected 参数必须真实送达

当前 `c2_local_agents.generate_legacy_proposal()` 同时把 `shared.source` 传给 analyst 和 rewriter，未保证其默认值等于 parent params。
`c2_local_inputs.stage_inputs()` 已物化 `common/parent.py`，但上述生成函数没有使用它；文件名 `best.py` 不能证明已在 best params。
采用最小方案：analyst 收原始可调源码及显式 `shared.parent.params`；rewriter 收既有 materialized parent 文件及同一显式 params。
在 `AnalystInputs`/`RewriterInputs` 增加可选 selected-params 上下文，sandbox 写 `tuning/selected_params.json` 并注明源码是否已物化。
优先读取 stage 已生成的 parent，若调用方未 stage 才用现有 `materialize` 一次；不要重复物化或修改原始 `Shared.source`。
修改“best.py already at best-known PARAMS”的无条件断言，使它只在实际物化且核验一致时成立。
测试必须用默认值与 selected 值故意不同的真实输入，检查 agent sandbox 内容、PARAMS 与 endpoint 固定伙伴一致。
原始 tunable source、完整 domain、constraints 与普通 trial 历史仍保留，后续 parameterization 不因上下文物化而缩域。

### 1.4 机会意图贯穿 parameterizer 和 expansion

现有 `rewrite_intent` 传递具体目标区域及伙伴条件，不能只传“优化了代码”这种 diff 摘要；两臂均启用同一传递机制。
生成时保留 `LegacyProposal.change_summary`，初始 parameterization 和 retune 注册均使用它，让 expansion 可继续读取。
当前 `Child` 只保留 path/space/backend；增加可选 `rewrite_intent` 并由 `parameterize_legacy_proposal()` 填入，再送到 `RetuneInputs.rewrite_intent`，不为摘要另调用模型。
parameterizer 允许为参数接线或必要修复改变 computational body，不设 blanket body、dtype 或等价性禁令。
只加小提示：涉及 index、mask、reduction 变化时，在现有参数描述/约束理由中说明与接线或修复的具体关系及风险。
沿用既有 validation 和有界 repair；不建立 AST 等价平台、额外审查模型或强制证明格式。
B0/task21 mask 的固定点正确性伤害已确认，作为针对索引轴的回归；B1/task43 实测近乎中性，不据此禁止非等价源码改写。

## 2. 已核对事实与历史边界

依据：[当前结果报告](result-c2-method-improvements.md)、[15 次固定诊断](../.omo/notepads/c2-parameterizer-impact/measurement.md)、[晋级成本与扩域记录](../.omo/notepads/c2-parameterizer-impact/promotion-cost.md)。
旧完整报告归档 5777ee7 所述核心存在 staging 污染；修复后 A 是混合的探索性结果，不能作为本轮独立 controls。
最新 O/P/D 诊断更新在笔记：task21 O 3/3 pass、P 3/3 fail、仅改 K-mask 的 D 3/3 pass；不重写旧报告历史判断。
task43 O/P 三次重复范围重叠，仅支持该固定点近乎中性，不证明精确等价或任意参数下安全。
旧 core/A 的 `SPACE_EXPANDED=0`，初始 parameterizer 不同 domain 不是原生扩域；不能以 OFF 结果否定扩域效力。
当前 `c2_information.py` 强制 local budget=0，RetuneInputs 也显式传 0，并在 retune 前退出 `with Runtime(...)`。
`c2_retune.RetuneInputs` 默认 expansion=0；`retune(..., runtime=...)` 已要求启用扩域时存在 caller-owned live client。
`Orchestrator._continue_accepted_candidate(..., run_analysis=False)` 已先调优、算 numeric stats，再调用 `_maybe_expand_space`。
因此核心续调能力已存在，但新 C2 指令、selected 上下文和本轮实际调用接线尚未实现，不能写成已完成效果。

## 3. 文件所有权、顺序与最小实现边界

| 优先级/owner | 独占文件或职责 | 交付与边界 |
|---|---|---|
| P0 提示与上下文 owner | `src/kernel_optimizer/agents/method_prompts.py`、`modules.py` | 单一 owner 编辑两文件；C2 链、selected 输入、既有 summary/intent、小型 body 理由提示 |
| P0 生成桥 owner | `scripts/experiments/c2_local_agents.py`，必要时 `c2_local_inputs.py` | 接入实际 parent/params，保留 proposal intent；与提示 owner 先对齐接口后编辑 |
| P0 原生接线 owner | `src/kernel_optimizer/control/orchestrator.py` | 仅在必要时补事件/最小传递；不重做 numeric eligibility 或 TPE，不与提示 owner 共编 |
| P1 runtime/晋级 owner | `scripts/experiments/c2_information.py`、`c2_information_inputs.py`、`c2_retune.py` | live runtime、显式 cap1、多空间计数、full promotion、最终 child3 与 deadline 接线 |
| 程序 owner | 新 `scripts/experiments/c2_opportunity_program.py` | 两波薄入口，调用现有 acquisition、generation、retune、adapter、admission、cost/helpers |
| focused tests owner | 第 10 节四个新测试文件及直接相关旧测试 | 测输入与行为，不匹配整段提示字面；不清理广泛 legacy 文件 |

先锁 P0 测试再写提示/上下文，随后接通 native 与 P1，最后组合薄入口；不让小修复占据方法主体。
快速实现 120 至 180 分钟，其中约 70 至 100 分钟用于 C2 推理、上下文、intent 及其 focused tests，其余 50 至 80 分钟接线和组合验证。
共用辅助模块只做必需的小改动；新入口不是 orchestrator，不复制 evaluator、TPE、witness 或模型调度平台。
复用 `c2_method_admission.py`、`c2_method_protocol.Deadline`、`c2_method_files.py`、`c2_local_adapter.GpuAdapter` 和已有计数/比较辅助函数。
旧 `c2_method_*` 的四臂/分支模型与旧 terminal 的 150 分钟、复用 baseline 协议不直接套入本轮。
新入口本地声明两臂及两波固定规格，heldout 复用测量 helper 而非旧 terminal 主函数；不修改旧计划以迁就新运行。
对外结果只标 G0/C2；内部复用 `InformationRun.group="G0"/"H"` 路由，H 供应完整 responses，两者均 `whole_task` 且传 intent，不复用旧 P/H 共享 rewrite 设计。
新程序绕过旧 `c2_method_native.tune_opportunity()` 的固定 seed0、expansion0 和 quick 晋级路径；不为本轮重写未使用的旧消费者。

## 4. 原生 retune 与扩域：两臂相同的机会实现路径

每机会初始一个 B40，完成后先跑现有 numeric boundary/headroom/min-effect/edge-failure eligibility，再作父子晋级。
即使 child 初始 quick 比父更差，也不能跳过其合资格扩域；扩域上限两臂都是 1，接受后另一个 B40。
不合资格无额外 40；no-op 或 rejection 记录真实状态而非宣称执行成功。本轮不是 S/E factorial。
新程序的实际 `InformationRun`、local config、`RetuneInputs` 消费链均显式传 cap=1，不依赖默认值覆盖。
可保留 `RetuneInputs` 的 legacy 默认 0；必须测试本轮实际消费者全部取 1，禁止只改配置却仍落到默认 0。
扩展 `Runtime` 的 caller-owned 生命周期，包住 generation 及 retune；调用 `retune(spec, local, runtime=runtime)` 后再关闭。
retune 注册时保留 rewrite intent，复用 `_continue_accepted_candidate(..., run_analysis=False)`；不再跑 post-analyst。
只做一次初始 parameterization，原生扩域自身允许既有 parameterizer/repair；禁止再跑第二套完整候选流水线。
保留原始 defaults、witnesses、prior-best anchors、constraints 处理与已修复 staging 的原文件名布局。
只有源匹配时才复用 prior measurement；源变化时配置 anchor 可保留，但旧 measurement 不能冒充新源测量。
跨空间保留原始实测 trial/artifact；扩域失败或更差不能擦掉先前 best，selected source 必须来自对应 measured trial。
扩域改了 body 则单独记录该变化；“域扩展加 body 改写加新预算”的收益不得归成纯 domain expansion。
在扩域前归档 base-best source/config/trial 及分数，不加 GPU finals；before/after 仅作描述性机制证据。
计数遍历所有 `TUNING_DONE`/space 的 asked，而非仅取最后 snapshot；每个已准入 study 完整 B40 才算完成。
归档 eligibility、实际 `SPACE_EXPANDED`、rejection/no-op、各空间 asked；全零扩域写“未覆盖”，不是“扩域无效”。
结果同时保留 selected_trial.space_id 与各 published spaces，不能因最后发布了 expanded space 就把旧空间 best 标成扩域所得。
不增加 mid-B40 expansion、axis locator、S7、scanner；不更改 native TPE 内部配置选择或两臂 eligibility 策略。

## 5. P1：fresh full 晋级与独立 heldout 分开

复用每机会已有 fresh parent3 和全部初始/合资格 expanded tuning 完成后的最终 child3，作为 selection screen。
parent3 可以在生成前取得；两者间隔及 cache/等待作为限制记录。不在每个空间后重复付 child3 finals。
父用原 P1 自身最佳配置，子用其结构跨已运行空间的 native best；禁止要求子在父旧固定配置上赢。
历史 parent quick 只用于描述与 quick/full 冲突识别，不能充当 full 晋级比较分母；native TPE 的内部排序不变。
记每块 latency median 为 x；每臂 screen 集合 S 有 3 块，M(S)=median(S)，R(S)=[min(S),max(S)]。
有效 child 默认在 M(child)<M(parent) 时可晋级，但先执行下列冲突检查；相等保留 parent，无内部 2% 最低改善要求。
若两臂描述性范围重叠或相触，或 quick/full 的严格优劣方向相反，必须做一次 fresh 父子对测，各 1 full job。
原始三块范围只触发确认并作为描述性信息保留报告；累计 min/max 只能向外扩展，初始重叠不可能靠追加块消除，因此不要求累计范围变得分离。
每一对均在父、子各自结构已选中配置上各测 1 full job，契约保持 100 performance samples/5 correctness；第一对 parent→child，第二对 child→parent。
每次用所有 screen+新增 full blocks 计算累计 median，绝不挑最好三块或只用最近一次；最终每臂最多 5 块。
记第 i 对的父/子 block median 为 p_i/c_i，累计父/子 median 为 M_P/M_C；第一对后若 p_1≤c_1 且 M_P≤M_C，立即 retain parent（含平局）。
否则在 deadline 允许时执行第二对，不另加自适应标准；第一对不能直接晋级 child，缺块、失败或截止均不晋级。
两对后仅当 c_1<p_1 且 c_2<p_2 且 M_C<M_P，并且 required correctness 全部有效时 promote child；任何其他情况 retain parent。
历史 quick/full 反向保留标签，只作触发依据，不要求其排序被改写，不补测 quick；两对矛盾即保留父，不做第三对。
每机会最多 2 对、4 jobs，总确认上限仍为 32 full blocks；边际阳性通常需 4 jobs，满足首对负向停止条件只需 2 jobs，不宣称统计显著性。
确认失败、缺块或 deadline 中止不晋级，区分 failed/censored；retain incumbent 不永久删除候选 family，不补抽结构。
screen/confirmation 属于选择数据，不能再称独立 finals；incumbent 固定并记录后才做全新 heldout。
heldout 不再改变 incumbent、续调或回选，旧结果也不得按本协议事后重排；2% 只属于外部论文投入门槛。

## 6. 唯一主实验：两个任务、两次 fresh rep、两个臂

共 `2 tasks × 2 reps × 2 arms = 8` 个固定 candidate opportunities，每机会一个新 rewrite，无 best-of 或补抽。
task43 原始 P1：`cand-b67a1cb4` / `tr-6fb84baa`；task21 原始 P1：`cand-4c96b8c4` / `tr-6afd5e56`。
每次从上述 exact original P1/off-state 及其 source/helpers/reference/params/history 开始，不用旧生成候选或旧成绩替代新测量。
wave2 也是 original P1，不接 wave1 winner；fresh rep 指独立生成调用，不保证 provider 支持或遵守 LLM seed。
G0：whole_task 一般指令、普通历史与资源、同样 intent/小修复/扩域，但无 fresh endpoint 信息。
C2：同样一般条件，加自己当次 fresh responses 与专属机会机制指令；比较的是这两者的 package，不能隔离 guidance 单独作用。
每个 C2 机会最多尝试 12 fresh endpoints，共 4×12=48 attempts；失败消耗预算，不补采到 12 成功。
G0 acquisition=0，空 responses wrapper 仍保留普通 Shared 证据；不复用旧独立核心结果充当对照。
初始 sampler：rep0=0，rep1=2；同 rep 两臂相同；evaluation/validator 均为 0。
expanded study 沿 native `cfg.run.seed`，默认与初始相同，即 0/2；不发明 expansion seed API 或偷偷换 seed。
不同 domain 使用相同 seed 不等于匹配搜索轨迹，报告该限制；不把 rep 解释成模型随机性完全受控。
模型/provider/既有 generation 配置固定，8 条链可并行或流水化；C2 链依赖自己 acquisition，G0 可独立准备。
计划正常路径 8 analyst + 8 rewriter + 8 initial parameterizer；实际 started/finished、格式修复、原生 expansion 调用另计。
8 条链都是预定准入机会，不按预测性能过滤；服务错误与既有有界修复耗尽均保留原机会，不重开一条模型链。
沿用两臂相同有界 format/validation repair；修复不构成新 rewrite draw，不因拒绝增加第九个机会。
拒绝计入 8 机会，可为 asked=0，不替换；质量失败可 retain parent，技术未完成单列 censored。

## 7. 四卡两波、时间准入与同卡比较

| 波/rep | A0 | A1 | B0 | B1 | heldout 比较卡 |
|---|---|---|---|---|---|
| wave1/0 | task43 G0 | task43 C2 | task21 G0 | task21 C2 | task43=A0，task21=B0 |
| wave2/1 | task21 C2 | task21 G0 | task43 C2 | task43 G0 | task21=A0，task43=B0 |

A/B 是两台既有 host，0/1 是其 GPU；跨 rep 同任务换 host 且臂换 GPU，单波任务配对在同 host。
每波两臂完成 selection 后，将 selected source/params/helpers 转移到该 host GPU0，再测共同 fresh parent 和两臂 incumbents。
heldout 三轮固定顺序：parent,G0,C2；C2,G0,parent；G0,C2,parent，各角色恰好三块。
同卡所有 GPU timing 串行；另一 GPU 可准备 CPU/LLM，不得有作业与当前卡计时争用；host 负载记录。
runtime 端口、run/cache/sandbox 目录使用现有隔离配置且每机会唯一，不建新环境或安装依赖。
从第一个 wave1 CLI 启动记录 campaign start 和绝对 deadline=start+4h；两波、heldout、机制检查共用此上限。
跨进程/host 传同一绝对 deadline，换算为各进程剩余时长，禁止把一台机器 monotonic 原点直接传给另一台。
通过既有 admission 在 agent/worker/新 study 准入前检查；已准入作业按既有规则 drain，排程超限后不准入新 job。
记录每阶段 start/end、deadline、未准入项和 drain，wave2 不重置时钟；不把 drain 隐藏在四小时以内。
wave2 无论 wave1 成绩如何都必须执行，除非真实 deadline 或技术 blocker；没有可选 rep、性能早停或择优补测。
第一波通常 45 至 90 分钟，扩域可能更长；首波检查点只更新剩余成本预测，不改 N、B40、seed 或 gate。
若串行两波超 cap，保留 wave2 的 censored 行，不减少 B40、不换候选、不以完成的子集宣称全局阳性。

## 8. 计数、测量契约与有界机制小检查

| 项目 | 固定数/上限 | 计数解释 |
|---|---|---|
| candidate opportunities / new rewrites | 8 / 最多 8 | 拒绝、失败、未准入也占机会；每条链最多一份新结构 |
| C2 fresh endpoints | 最多 48 | 4 机会各 12；G0 无；不是 48 个必然成功点 |
| native studies | 正常 8 至 16 | 8 initial B40 + 最多 8 accepted expanded B40；拒绝/截止可少于 8 |
| asked | 最多 640 | 16×40，含 witness/measurement reuse；不是 640 个全新 full jobs |
| selection screen full blocks | 最多 48 | 8×(parent3+最终 child3)，所有空间结束后仅一组 child3 |
| confirmation full blocks | 最多 32 | 8×2 pairs×2 jobs，停止规则见第 5 节 |
| independent heldout full blocks | 36 | 4 task/rep pairs×(parent3+G0 incumbent3+C2 incumbent3) |
| 科学程序 full blocks 合计 | 最多 116 | 48+32+36；无确认且均有效时 84 |
| optional mechanism QUICK | 最多 8 | 每任务最多 4，不是每 rep 4；不计入 full |

每 full block 保持 100 performance samples、5 correctness，`dual_witness_relaxed` 及原阈值不变，raw FP64 rescue 计数保留。
契约沿用 fp32 reference、pass fraction0.99、cosine0.99985、FP64 multipliers2/3；不把 rescue 通过表述成严格数学等价。
116 full blocks 对应最多 11600 报告性能样本、580 次 correctness；不含 native quick/witness、static checks 或 smoke。
failed child 可没有三块，不能补造有效 latency；若最终 retain parent，heldout 中该臂仍须 fresh 测三块，不复用共同 parent 三块。
明确报告 scheduled/attempted/complete/failed/censored；36 是完整 heldout 协议，deadline 缺失要记缺失，不能写完成。
native asked、validation witnesses、adapter static checks、provider retries 各自列账，不能从 asked 推造总 worker job 数。
机制检查先用已存在 responses、trials、source 和 base-best 档案，不以多做实验代替解释已有证据。
代表案例预先固定为每任务 C2 rep0，不看赢家再选；无效或不可对应写 N/A，不换成 rep1 或 G0。
从调优前 proposal 识别假设轴及伙伴映射；若无法唯一解释或跨结构对应不成立，不强行选点。
有唯一映射时，在额外测量前声明最多四个精确 source/config/partner 点及检验关系，采用两结构各两个对应点。
点只能来自既有合法 source/domain；无法组成该对照则不做该任务小检查，不加新轴、TPE 或 LLM 来寻找好案例。
额外调用固定 QUICK 20 samples/3 correctness，最多每任务 4 次，共 8；失败仍计次数，预计 10 至 20 分钟且在 campaign cap 内。
小检查只检验声称机会是否实现及代价变化，不声称全局因果；不启动独立 ablation campaign。

## 9. 指标、投入门槛与独立交付

主性能只取 heldout 同卡的三块 median-of-medians；每 pair 共用此次 fresh parent 归一化：R_arm=M_arm/M_parent。
改善定义 `100×(1-M_C2/M_G0)`；不跨卡比 raw latency，不用最快块，不用 screen 或 tuning 分数替代 heldout。
完整列四个 pair；invalid 机会的 retained-parent 结果仍计入，失败不删行，censored/missing 不能当作零收益或成功。
外部论文投入 gate：至少 3/4 pair 的 C2 比 G0 median 低至少 2%，且 `max(C2三块)<min(G0三块)`，覆盖两个任务。
任何 missing/censored pair 禁止全局阳性声明；invalid 但有完整 retained-parent heldout 仍留在四 pair 分母中。
三块范围与此 gate 仅描述性投资判断，不是置信区间或显著性；gate 通过也不自动启动长期跟进。
质量、拒绝、rescue、源码/域/body 变化、机会实现比例、扩域资格与实发次数一起报告，负结果不隐藏。
效率报告实际两波共享窗口和 time-to-incumbent，用既有事件边界；model、acquisition、tuning、screen、confirmation 分项。
另列 endpoint acquisition、heldout transfer/wait、额外 mechanism、收集及人工分析成本，避免与包含它们的总窗口重复相加。
资源以寄存器/shared/spill/occupancy 等向量和未知值报告，不合成资源效率单分数，也不把 wall time 当 GPU busy。
共享窗口不是各臂独立部署时长；成本反事实必须标“估计”，不能当实测或据四卡配置宣称 4× 加速。
交付独立结果表：8 机会的状态、seed、domain、asked、expansion、selected、质量和成本，以及 4 heldout 比较及全部块值。
交付两条案例链图/表，明确 measured/inferred 标签、动作和目标区域是否实现；附成本表、失败样例、源码 SHA 与 raw 路径。
结果写新 `docs/result-c2-opportunity-driven-improvements.md`，raw 写新 `results/c2-opportunity-driven/`，不覆盖既有结果。
论文表述限制为 discovery-biased 的已知两个 benchmark 上的 pilot；不宣称通用证明、独立 guidance 因果效应或扩大样本代表性。

## 10. 测试、归档和唯一紧凑核验

未来执行先按先前 GitHub 归档模式单独发布本获批新计划，再开始代码/实验；本次写计划不运行 Git。
focused CPU 测试通过后、任何 GPU smoke 前，发布并部署测试过的精确 source SHA；归档不混入凭据或 raw 大目录。
新增 `tests/test_c2_opportunity_prompts.py`：响应有/无路由、wall 分离、selected 参数真实送达、目标 intent 到 parameterizer。
新增 `tests/test_c2_opportunity_flow.py`：live Runtime、显式 cap1、初始较差仍查扩域、no-op/reject、跨空间 measured artifact 与 staging。
新增 `tests/test_c2_promotion_full.py`：原始范围重叠/相触及 quick/full 冲突触发；初始重叠可在两对一致胜出且累计 median 更低时晋级；两对矛盾 retain；首对明确负向且累计支持父时立即停止；全部 screen/确认测量均贡献累计 median；二对上限、缺块/失败 retain、heldout 不重排。
新增 `tests/test_c2_opportunity_program.py`：8 机会两波原 P1、0/2 seed、共享 deadline、48/640/116 上限、拒绝不补抽、wave2 必须执行。
测试用 stub agents/workers 和真实 sandbox/staging 布局；断言输入、调用顺序、事件与结果，不锁死提示整段文字。
B0 mask 回归检验 [K,N] 尾块中 K-mask 沿行轴，错误列轴案例可检测；保留已完成 O/P/D 证据，不再跑 15-job ablation。
复跑 `test_c2_retune_staging.py`、`test_c2_accepted_continuation.py`、`test_c2_retune.py`、`test_c2_information_execution.py` 相关测试。
CPU 命令：`.venv-v5/Scripts/python.exe -m pytest tests/test_c2_opportunity_prompts.py tests/test_c2_opportunity_flow.py tests/test_c2_promotion_full.py tests/test_c2_opportunity_program.py tests/test_c2_retune_staging.py tests/test_c2_accepted_continuation.py tests/test_c2_retune.py tests/test_c2_information_execution.py`。
对修改 Python 做既有静态检查/LSP，缺工具记录而不安装依赖；测试先有失败证据再最小实现，不作全仓清理。
新薄入口约定 `python -m scripts.experiments.c2_opportunity_program --config CONFIG --inputs INPUTS --output OUTPUT --deadline-unix-s DEADLINE`，所有阶段共用 deadline。
INPUTS 固定本计划两波映射、原 P1 引用和本槽 wave/rep/arm；测试拒绝改 N/B/seed/gate，不接旧 A/B/C phase。
setup 中只做一次已有 staging/mask GPU smoke，最多 3 full jobs、约 10 至 20 分钟；与科学 8 机会分账，无额外 rewrite。
smoke 沿用已知 B0 的 O/P/D 固定输入各一次：O/D 应过、P 应拒绝，检验实际 staged 输入而非重新估计效应；不将 smoke 纳入主性能表。
如果 smoke 不符合已知契约，停止科学准入并修正接线，不能耗尽三次后擅自补跑；异常成本与 blocker 单列。
只设一个未来独立紧凑 review：核对计数、source fidelity、指标及 focused tests；不启动五代理安全/QA 审计。

## 11. 时间账、停止点与明确风险

| 阶段 | 名义墙钟预算 | 包含/不重复计算 |
|---|---|---|
| 实现及 focused CPU | 2 至 3h | P0 占主要精力，复用现有接缝 |
| setup、部署、smoke | 0.25 至 0.5h | 上述 10 至 20 分钟 smoke 已包括在此 |
| 科学实验 | 2 至 3.5h | 四卡两波、8 至 16 studies、heldout、可选机制均在同一 4h cap |
| 分析及一次紧凑核验 | 0.5 至 1h | 表格、案例、计数/源码/指标核验 |
| 名义总计 | 4.75 至 8h，约 5 至 8h | 2+0.25+2+0.5 至 3+0.5+3.5+1 |

若接缝集成破坏，实施 contingency 可到 4 至 5h，总计相应为 6.75 至 10h；不把快速路径或此范围写成保证。
实验最大排程 4h 加单列 drain；expansion、模型排队、确认或 host 争用可突破名义耗时，不延长排程上限来求阳性。
开放风险：selected 上下文原有失配、Runtime 生命周期、跨空间计数和 artifact 对应须由真实输入测试覆盖，计划本身不证明已修复。
开放风险：8 条生成链质量及扩域 eligibility 未知，不能预猜候选表现；零实发扩域只能报未覆盖。
开放风险：小样本噪声、quick/full 反向、soft-cost/伙伴推断和 known-task discovery bias 限制结论强度。
开放风险：native expansion 可能改 body 或被拒绝；4h deadline 可能令 wave2/heldout censored，不能以缩预算补齐表面完整性。
本程序结束即停：无四臂 L/P/H、S/E factorial、A/B/C 分支、第三轮、12h 延长或为论文阳性追加实验。
不建设 safety hardening、freezing、recovery、identity、ledger 平台；不新增依赖、不收窄域、不禁止 dtype，不修改旧计划/boulder。
