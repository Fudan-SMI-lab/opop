# 交接简报:找墙与测斜率的两个设计缺陷(供外部设计者快速进入)

> 写给:接手设计"墙判据 + 斜率估计"改造方案的 agent。
> 你不需要读过本仓库的任何其他文档;本文自足。
> 日期:2026-09-14。所有"实盘"数字来自当时正在 box4 上运行的 step-4 对照实验(约 6.5h 时点)
> 与已跑完的 S7 对照实验,均可从 `events.jsonl` 复验(§8 有复验方法)。

> **⛔ 交付边界(先读这条):你的交付物是设计文档,不是代码改动。**
> - **禁止修改 `src/` 与 `configs/` 下的任何文件**——一对 12h 对照实验正在 box4 上跑,
>   它的 worker/orchestrator 代码就来自这个仓库的检出;判据/机制类改动已被约定推迟到
>   全部实验结束后统一评审("高风险修复只记录,不动运行中的实验")。
> - **可以**:读一切;跑只读探针/重放脚本;跑测试套件
>   (`PYTHONIOENCODING=utf-8 uv run --offline --extra test --quiet python -m pytest -q`,
>   当前基线 1 failed/1188 passed,那 1 个是 §4.7 已记录的 torch-less 既知问题);
>   为验证自己的设计**新增**临时探针/原型脚本(放 `scripts/probes/` 或独立目录,
>   不 import 改动、不打 monkey-patch 进正式模块的运行路径)。
> - 写新文件与提交:设计文档放 `docs/`;不要修改本简报与既有文档正文
>   (发现其中的错误请在你的设计文档里指出,而不是就地改)。

---

## 1. 框架三分钟速览

**这是什么项目**:一个 GPU kernel 优化 agent harness(论文 "Guiding GPU Kernel Structure Search
with Parameter Tuning Feedback")。LLM agent(经 opencode 调 GLM)生成/改写 Triton kernel 候选,
harness 负责一切判定(正确性、计时、预算、收敛)。实验任务固定为 KernelBench
`level3:43 MinGPTCausalAttention`,跑在 RTX 4090(autodl 云盒)。

**候选与参数空间**:一个候选 = 一个 Python 文件,含唯一模块级 `PARAMS = {...}` 字面量;
所有可调 knob(如 `BLOCK_M`、`NUM_WARPS`、`COMPUTE_DTYPE`)经 `PARAMS[...]` 读取。
agent 为候选声明参数空间(每 knob 一个有序 choices 列表,约定便宜→贵)。
每空间约 40–80 个 trial,Optuna **TPE**(multivariate, group)采样;
每 trial = 一次性子进程:编译 → 正确性 → 20 采样计时,产出
`TRIAL_DONE` 事件(含 `params.values`、`latency_ms.{median,mean,samples[20]}`、
`profile.{n_regs,n_spills,shared_bytes,...}`)。
超共享内存上限的配置在编译筛查处被拒,记 `CONFIG_SCREENED_INFEASIBLE` 事件
(键:`candidate_id, kernel, limit, max_shared, params`)——**拒绝是编译器的精确判定,不是估计**。

**四个循环**:A 修复(坏候选喂 repair agent)、B 调参(上述 TPE)、C 结构改写
(analyst 读调参统计出瓶颈假设 → rewriter 产出新候选)、D 新族探索。
所有状态在 append-only `events.jsonl`,报告纯重放再生。

**C2(本简报关心的贡献点)的主张**:调参过程中发现的"资源墙"(某 knob 的取值范围被
硬件资源上限截断)+ 通往墙的延迟斜率,可以引导(i)结构改写(把墙"解开")和
(ii)采样器(往墙下最后一个可行值投点)。机制分三件套,各有开关:

| 机制 | 模块 | 输出 |
|---|---|---|
| 硬墙归因(2e) | `evaluation/wall_attribution.py` | 墙文本简报 `resource_walls.md` → rewriter |
| 软墙(寄存器溢出) | `evaluation/soft_wall.py` | 同上(共用文件名,内容区分) |
| 斜率引导投递(S7) | `tuning/slope_guide.py` | `study.enqueue_trial` 往采样队列插点 |

**至今的两对对照实验结果**(背景,不必细究):
- **S7 对**(唯一变量=投递,已跑完):机制端到端零执行损耗(建议 7 次全被采样器接受),
  但端到端不赚不赔;72 次斜率重算里 63 次(88%)"无墙可用"。
- **step-4 对**(C2 全关/全开,进行中):全开臂 31+ 次重算 **100%**"无墙可用"、**0 投递**;
  8 次归因事件:4 次连墙都没形成,4 次形成了(共 5 道)**全部**被斜率过滤器丢弃。

**⇒ 瓶颈两次独立定位到同一处:墙的判据与斜率的测量,不是机制执行。**
本简报要交接的正是这两处的设计缺陷(用户于 2026-09-14 指出,实盘证实)。

---

## 2. 现有管道的精确描述(改造对象)

### 2.1 数据基础:`latency_by_value`(边际中位表)

`tuning/stats.py:79-84`(`TuningStatsAnalyzer._param_stat`):

```python
grouped: dict[str, list[float]] = defaultdict(list)
for t in complete:                                   # 该空间全部完成 trial
    grouped[repr(t.params.values.get(name))].append(t.latency_ms.robust_ms)
for key, vals in grouped.items():
    lat_by_value[key] = statistics.median(vals)      # 每值一个边际中位
```

**注意**:分桶只按"该 knob 的取值",桶内 trial 的**其余 11 个左右 knob 任意**。
这张表是下游一切斜率/单调性判定的唯一数据源。

### 2.2 硬墙:`find_walls` + `select_for_probing`(`evaluation/wall_attribution.py`)

- `find_walls(stats, refused_params)`(240-288 行):对每个 knob,取被拒配置在该 knob 上的值,
  **判据 = 该值严格在边际已测范围之外**(`v > hi or v < lo`,264-271 行;注释原文:
  "A refused value INSIDE the measured range is not a truncation")。
  成墙后从 `latency_by_value` 取靠墙侧最后 3 个值的中位作 `tail_latencies`,
  算 `monotone`(严格递减)与 `tail_gain_pct`。
- `select_for_probing`(291-308 行):**只保留 `monotone and tail_gain_pct > 0`** 的墙。
- 幸存的墙送**消融探针**:从该空间最优配置 θ\*(以及 top-K 高性能点)出发,
  **固定其余全部 knob、只把这个 knob 改成被拒值**,问编译器是否仍拒
  (这一步是条件化的、正确的;实测从 θ\* 出发 6/6 可归因、从默认配置出发只 1/6)。
- 归因成功的墙渲染成中文简报(`for_prompt`,346-398 行,措辞已条件化:
  "仅在该候选的最优参数点处成立")交付 rewriter。

### 2.3 软墙:`find_soft_walls`(`evaluation/soft_wall.py:230-321`)

适用门:该空间**最优 trial 自己要有 n_spills>0**(249-260 行,这一门是对的,保留)。
然后同样按 knob 分桶(桶内其余 knob 任意),取每值的 spill 中位与延迟中位,
要求 **spill 曲线单调非降** 且 **延迟尾部仍在改善**(292-311 行)——
判据同样跑在边际曲线上。软墙无独立探针(渲染文本已如实声明)。

### 2.4 斜率引导投递:`SlopeGuide`(`tuning/slope_guide.py`)

每 `recompute_every=10` 个完成 trial 重算一次:调 2.2/2.3 的墙查找 →
对幸存墙,**从 θ\* 的搭档出发**、沿该 knob 向墙走到最后一个可行 choice
(`_toward_wall`,319-403 行,条件化的、正确的)→ `study.enqueue_trial` 投递。
计数器齐全(`n_skipped_no_wall` 等);S7 实测投递侧**零损耗**
(n_suggested == 实际投递数,零采样器拒绝)。

### 2.5 一句话概括管道形状

**行动端(消融探针、投递点构造)已条件化在 θ\* 上——是对的;
守门端(墙的范围判据、单调/斜率过滤)跑在未条件化的边际曲线上——是错的。
被污染的边际统计守门,把条件化的正确动作挡在门外。**

---

## 3. 问题一:斜率读的是未条件化边际,测的是采样史不是 knob

### 3.1 机制

桶的成分由 TPE 的采样史决定:TPE 对它看好的值会配好搭档反复采样,
对早期试过就放弃的值只留下坏搭档的残骸。于是 `latency_by_value` 的形状
一半来自 knob 效应、一半来自采样器偏好——一条会被 Simpson 悖论任意扭曲的曲线
(本项目已在寄存器分类上实测过混池反转相关性的先例)。

### 3.2 实盘证据(s4-c2on 臂,可复验)

**例 1**:`cand-94aab0fc` / `G_BLOCK_M`——这是 seq622 被丢弃的墙
(`tail_gain=+72.8%` 但 `monotone=False`):

| G_BLOCK_M | n | 边际中位 (ms) | 该值上最好 (ms) |
|---|---|---|---|
| 32 | 7 | 6.61 | 4.06 |
| 64 | 2 | 19.58 | 11.45 |
| 128 | 17 | **30.67** | **3.28** |
| 256 | 20 | 5.33 | 3.19 |

边际中位 6.6→19.6→30.7→5.3 锯齿(⇒ 判非单调 ⇒ 墙丢弃);每值最好 4.06/11.45/3.28/3.19。
128 桶的中位与它自己的最好差 **9.3 倍**——桶里塞满 `COMPUTE_DTYPE=ieee` 等坏搭档(最差 103ms)。

**例 2**:`cand-4b7a73bf` / `ATT_STAGES`:四个值的 n = 44 / 7 / **1** / **1**。
尾部三值(2,3,4)的"曲线"建立在 n=7/1/1 的桶上,中位 11.89→7.75→30.03——
单调性判定拿单样本桶当曲线点。

**该臂 5/5 道成形的墙全部死于 `monotone=False`**,而上表说明这个"非单调"
主要是桶成分噪声,不是 knob 的真实形状。

(复验口径说明:上表按候选全体完成 trial 分桶,发射端按"空间"分桶且用
`robust_ms`;成分污染的结论不受此差异影响,但逐数字对账时要按空间重算。)

### 3.3 连带后果

- S7 预注册的"斜率越陡的投递点应表现越好"被**反向**否证(浅斜率反而输得少)——
  若 `tail_gain_pct` 本身近于噪声,这不再是谜。
- `select_for_probing` 的丢弃量实测大于保留量(S7 数据上 monotone 单独丢 9 道、保留 8 道)。
- 噪声背景:本任务单 trial 计时 std ≈ 16%,重调同一空间可摆 2.1%,复测缺口 ±2–4%。
  任何斜率判据必须显式对抗这个量级。

---

## 4. 问题二:墙判据被"换搭档跑通"结构性抹掉

### 4.1 机制

判据要求被拒值在**边际**已测范围之外。而共享内存占用 ≈ tile 各维与 stages 的**乘积型**函数:
`BLOCK_K=128` 配小 `BLOCK_M` 装得下、配大 `BLOCK_M` 被拒。所以几乎任何单值都存在可行搭档;
TPE 采样越充分,越容易把被拒值在别的搭档下测通,墙**按定义**消失。
⇒ 这个判据把"墙"定义成了**采样不完整性的残差**:采样器干得越好,墙越接近零,
永远赶不上"每 10 个 trial 重算一次"的消费节奏。

### 4.2 实盘证据

- S7 实测全过程:`NUM_WARPS=16` 在 trial 6 被拒(成墙)、trial 14 换搭档跑通(墙消失);
  墙数随 trial **单调减少**。
- s4-c2on:35 个共享内存拒绝,8 次归因事件里 4 次 `walls_found=0`——
  **每个被拒值都在该 knob 的边际已测范围之内**。
- 拒绝本身全部真实:S7 控制臂 39/39 个拒绝复核为真超限(1.13–2.26 倍),
  编译器消息明确点名 kernel 与字节数。**不缺拒绝,缺的是能从拒绝里存活的"墙"。**

### 4.3 与它相邻、但不同层的两个已知缺口(改造时须一并考虑)

- **§4.5(`docs/next-round-changes.md`)——归因射程**:即使墙成形并过滤存活,
  单 knob 消融也探不到被拒点:被拒配置与 θ\* 中位差 **5–6 个 knob**,
  恰差 1 个 knob 的配对在全部记录里只占 **1%**(19/2326)。
  已量价的修法是"从被拒点向 θ\* 逐维回退求最小可行集合"(见 §4.5,批量粒度是分水岭:
  每层一批 123s / 每墙一批 1.21h,答案相同)。
- **约 2% 的真墙输入丢失**:launch 期 `OutOfResources` 被 `_classify_exception`
  标成 `runtime_error`,进不了 `find_walls` 的输入。

三个缺口的关系:**问题二 = 墙进不来;问题一 = 进来了被假锯齿杀掉;§4.5 = 杀剩的探不到。**
只修任何单独一层,零投递都不会消失(两臂的零投递成因已实测不对称)。

---

## 5. 已经做对、设计必须复用的部分

1. **消融探针**:θ\* 固定其余 knob、动一个,问编译器——条件化,且探针便宜:
   compile-only screen 边际 **7ms/个**,每批固定成本 **~11s**(进程启动),
   按 materialized source 缓存、故意不缓存失败(`correctness.py:170-191, 240-244`)。
   **重要**:编译元数据同时给出 `shared_bytes / n_regs / n_spills`——
   即**整个资源向量都能以 7ms/点条件化测得**,只有延迟必须跑真 trial。
2. **投递路径**:`enqueue_trial` 端到端零损耗已两次实测;剂量开关
   (`recompute_every` / `max_enqueued_per_recompute`)已在。
3. **软墙适用门**("最优 trial 自己要 spill")方向正确,38–43% 适用率是诚实的分母。
4. **条件化措辞**:`for_prompt` 的简报文本已把"仅在最优点处成立"写死,不必改。
5. **计数器纪律**:每种拒绝/跳过原因分开计数(单位不同,不可跨事件求和)。

---

## 5b. 本地代码阅读地图(你就在这个工作目录里,`D:\Pyhon_projects\opop\v3`)

按"要理解什么 → 去读哪里"组织。各模块的 docstring 写得很密,是**设计决策及其实测依据**
的第一手记录(包括踩过的坑),先读 docstring 再读代码体,收益最大。
⚠ 唯一的搜索纪律:**不要在仓库根 `D:\Pyhon_projects\opop` 做无范围的递归搜索**
(那里有一个 14GB 的 .db);把 grep/glob 限定在 `v3/src`、`v3/docs`、`v3/scripts`、`v3/tests`。

**改造对象(问题一、二所在,按管道顺序)**

- `src/kernel_optimizer/tuning/stats.py` — `_param_stat`:`latency_by_value` 边际中位表的
  发射端(问题一的数据源);同文件的 `at_boundary`/`best_trial_value` 锚定逻辑
  docstring 里有一段 1126-knob 的实测,解释了"中位挑值 vs 最快 trial 挑值"为何分开——
  读它能避免把两套挑值逻辑混为一谈。
- `src/kernel_optimizer/evaluation/wall_attribution.py` — 硬墙全链:`find_walls`(范围判据,
  问题二)、`select_for_probing`(单调过滤,问题一的消费端)、消融探针的 origin/top-K 机制、
  `for_prompt`(简报渲染)。模块 docstring 是 C2 硬墙侧的完整设计陈述。
- `src/kernel_optimizer/evaluation/soft_wall.py` — 软墙:适用门、spill 曲线判据
  (同样跑在边际上)、以及"为什么选 n_spills 弃 occupancy"的 2788-trial 实测表。
- `src/kernel_optimizer/tuning/slope_guide.py` — 投递机制:`due`/`suggest`/`_toward_wall`/
  `_incumbent`、全部计数器及其"不是 partition,不可求和"的说明(`snapshot` docstring)。
  docstring 末段"WHAT THE SHIPPING CODE ACTUALLY DID ON THE REAL CORPUS"预言了后来实测到的
  95% 无墙率,值得整段读。
- `src/kernel_optimizer/evaluation/correctness.py` — `compile_screen`/`prescreen_batch`/
  `cached_shared_verdict`:compile-only 探针的调用端、缓存语义(答案缓存/失败不缓存)、
  超时预算。你的方案若用探针,成本与失败语义以这里为准。
- `src/kernel_optimizer/gpu/worker_main.py` — `run_compile_probe`(约 713 行起):
  探针的 worker 侧实现,`warmup=True` 编译不 launch,返回每 kernel 的
  `{shared, n_regs, n_spills, num_warps, num_stages}`——**探针能给全资源向量**这一事实
  的出处;批量语义(`extra_kernel_src_paths`)也在这里。

**接线与消费端(改判据后要跟着改的地方)**

- `src/kernel_optimizer/control/orchestrator.py` — 三件套机制在外环的挂载点:
  搜 `SlopeGuide`、`find_walls`、`find_soft_walls`、`RESOURCE_WALL_ATTRIBUTED` 的发射处,
  可看到判据的输入(refused_sets 从哪来)与输出(事件/简报)如何进出。
- `src/kernel_optimizer/config.py` — 三件套的开关键
  (`v3.wall_attribution.* / v3.soft_wall.* / v3.slope_guide.*`);新开关照这个形状加。
- `src/kernel_optimizer/models/reports.py` — `TuningStats`/`ParamStat` 的字段定义
  (`latency_by_value` 的类型契约在此)。
- `tests/` — 每个上述模块有同名测试;`tests/test_s7_pair_reader.py` 展示了
  "fixture 必须逐字段抄自真实 payload"的纪律,新事件的测试照此写。

**背景文档(按需读)**

- `docs/next-round-changes.md` — 待办缺陷总账:**§4.8 就是本简报两个问题的原始记录**
  (含实盘表格),§4.5 是相邻的"归因射程"缺口(含已量好的多 knob 回退价格表),
  §5 是既有的建议次序。你的方案要与这两节对齐或明确改写它们。
- `docs/design-conditioned-walls-and-slopes.md` — 在你之前已有一版设计稿
  (frontier scan + 实测轴斜率)。**你的任务不是复述它**:独立设计后与它比对,
  指出它的缺陷或给出更优替代都欢迎;若结论趋同,请说明哪些部分是被证据强制的
  (殊途同归本身是信息)。
- `docs/v3-design-resource-ratio-and-conversion-efficiency.md` — C2 资源维度设计的
  上游思路与"预测数字禁止进决策"红线的完整论证。
- 事件读法示例:`scripts/probes/read_slope_guide_counters.py` 与
  `scripts/probes/did_the_enqueued_point_win.py` 的 docstring 记录了两个真实读错数的事故
  (跨事件求和得三角数;中途判胜负被反转),新读数脚本照此防御。

**运行数据在哪**:本机没有 GPU 也没有 run 数据;实盘 events.jsonl 都在 box4
(§8 有路径)。本地能做的是读代码、跑 `uv run --offline --extra test --quiet python -m
pytest -q`(纯 CPU,当前基线:1 failed——§4.7 已记录的 torch-less 既知问题——1188 passed)。

---

## 6. 硬约束与红线(违反任何一条的方案会被否决)

1. **不得缩小/限制搜索空间**:墙只记录、只建议,**永不 enforce**;不得删除取值、
   不得加约束、不得改预算分配。投递只能"加点",trial 预算不变。
2. **禁止针对单 case 的特判/硬编码**;一切修改必须是泛化的整体提升。
3. **进入决策/排序的数字必须是已实现的实测量**:预测数字已被实测否决
   (`predicted_gain_pct`:中位带符号误差 −5.0%、21 次里 8 次符号错)⇒
   **不得用 surrogate/拟合模型的预测斜率做门**。资源映射不可分离(三重否证)⇒
   不得拟合全局资源模型,只能局部实测。
4. **噪声纪律**:效应量未显著超过噪声底(单 trial std 16%、重调摆动 2.1%、复测 ±2–4%)
   时一律报 `unknown`,不报数字。每 trial 的 20 个原始采样都在 `TRIAL_DONE` 里,CI 可算。
5. **正对照义务**:任何判据放宽,必须证明它不会把所有拒绝都变成墙/所有 knob 都变成有斜率
   ——即必须仍存在被判"非墙/无斜率"的输入。常数读数 = 坏探针。
6. **预算现实**:8/8 跑完的 run 都由墙钟结束 ⇒ 一切新增探针按 **run 级墙钟**计价,
   且批量粒度决定成本(见 §4.5 的 36 倍差)。
7. 资源向量不得标量化合成一个数;无跨候选共享;PRUNED 行为不动。
8. 有序域才有斜率;categorical(含 bool)不读斜率(现有 `_as_num` 的排除正确)。

---

## 7. 下游消费者与接口要求(设计的验收面)

| 消费者 | 现在拿到什么 | 改造后必须仍满足 |
|---|---|---|
| rewriter(改写 agent) | `resource_walls.md` 简报:哪个 knob、在哪个点、超限多少字节、延迟趋势 | 措辞条件化;附带的斜率必须是实测且过噪声门;分母(适用性)如实报 |
| 采样器(经投递) | 靠墙侧最后可行值 + θ\* 搭档 | 只加点;剂量有 cap;去重按整配置 |
| 分析/报告 | 事件计数器 + 每墙 payload | 新判据下每种失败原因仍分开计数;判决写 payload 不写事件名 |
| 论文 | P1–P5 类预注册读数 | 新判据的预测须在跑前写进 config 头部 |

---

## 8. 如何自行复验(不动正在跑的实验)

- 实盘数据:box4 `/root/autodl-tmp/opop-workspace/opop-glm/runs-v3/{s4-c2on,s4-c2off}/run-*/events.jsonl`
  (只读);S7 对应 `s7-treatment/s7-control`。**只信盘上 events.jsonl**。
- 读数工具(已修 bug 版):`scripts/probes/read_slope_guide_counters.py`(计数器,防三角数)、
  `scripts/probes/did_the_enqueued_point_win.py`(投递点胜负,只在 RUN_FINISHED 后读)、
  `scripts/probes/rewrite_provenance.py`(改写溯源)。
- 重放工具:`scripts/probes/s7_feasibility_replay.py`(按前缀重放墙查找)、
  `scripts/probes/s7_why_it_declines.py`(逐拒绝原因)。
- 判读纪律三条:投递点胜负只能跑完后读;计数器不可跨事件求和;
  改写文本谈资源 ≠ C2 生效(须查该族是否有已归因墙)。

## 9. 设计任务书

给出一套同时解决问题一、二的设计,须回答:

1. **墙的新判据**是什么?如何不被搭档成功抹掉,又如何保证仍有"非墙"存在(正对照)?
2. **斜率的新测量**是什么?数据从哪来(观测配对/定向采样/探针),噪声门怎么定,
   每点 n 与 CI 如何披露?
3. 与 §4.5 多 knob 回退、2% 误标修复如何**分层组合**?
4. **成本**:每空间/每 run 增加多少墙钟?批量粒度?(参照 7ms+11s/批 与 §4.5 价格表)
5. **兼容**:旧 run 重放不炸;开关粒度足以做单变量对照;事件与计数器方案。
6. **预注册**:新机制上什么读数算成功/失败,跑前写下。
