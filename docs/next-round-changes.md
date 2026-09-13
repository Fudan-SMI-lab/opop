# 下一轮改动清单(已批准 / 待批准 / 明确不动)

**日期** 2026-09-13 · **checkout** `D:\Pyhon_projects\opop\v3` @ `516431e`
**状态**:本文件是**待实施**清单,以下**没有一条已经写进代码**。

**用户已给的边界**(逐条遵守,不再重新征询):
- ✅ **已批准**:放宽墙探针的"仅从最优点出发"约束(§1)。
- 🚫 **明确不动**:PRUNED / FAIL 报法相关的一切改动 —— 用户判断"改动风险较高,建议先不要变动"。
  ⇒ 这同时意味着 **Optuna 内建 `constraints_func` 这条路本轮不走**(它与 PRUNED 报法互斥),
  §4 只记录事实不提改动。

---

## 1. 【已批准】墙探针从多个高性能点出发,而不只从最优点

### 1.1 现状与要改的地方

`orchestrator.py:1494-1507`:

```python
theta_star = self._theta_star(crun)          # 只取 top-1
origins = [("theta_star", theta_star)]
if cfg2e.probe_second_origin:
    default = self._space_default_params(crun)
    if default:
        origins.append(("default", default))  # 第二个原点是"默认配置"
n_probes = self._probe_walls(crun, probe, origins)
```

现在的两个原点是 **top-1** 与 **空间默认配置**。后者的作用是**检验条件性**,不是找更多墙 ——
实测从默认出发只有 **1/6** 个墙能归因(从最优点是 **6/6**),因为默认配置每一维都取最小,
放大一个 knob 仍装得下。

### 1.2 改成什么

把原点从"top-1 + 默认"改为 **"top-K 实测高性能点 + 默认"**,K 可配(建议 3)。

```python
origins = [(f"theta_top{i+1}", p) for i, p in enumerate(self._theta_top_k(crun, cfg2e.probe_top_k))]
if cfg2e.probe_second_origin:
    ...  # 默认配置不变,仍是条件性对照
```

**`_theta_top_k` 的取点规则(必须写死,否则会悄悄变成一个新的启发式):**
按 `robust_ms` 升序取前 K 个**已完成**(`status == "complete"`)的 trial 的参数组 ——
即把现有 `_theta_star`(`orchestrator.py:1522`,它已经正确地用 `robust_ms` 而非 `.mean`/`min`)
的"保留最小值"改成"保留前 K 个"。**不做**任何"预期提升后可能超过 top-1"的外推,见 §1.5。

### 1.2b 真正的落点:`_probe_walls` 的写回分支会互相覆盖(读代码才发现的)

**好消息:批量探针已经就位。** `_probe_walls`(`orchestrator.py:1558`)已经对
**`(wall, origin)` 的全部组合**构造变体、一次 `prescreen_batch` 探完,
docstring 就写着 "Materialize every (wall, origin) ablation, probe them in **ONE batch**"。
⇒ §1.4 担心的 34 倍成本回归**结构上已经不会发生**,加原点只是让 `variants` 变长。

**坏消息,也是这个改动的真正工作量所在**:写回是**二分支**的(`orchestrator.py:1597-1605`):

```python
if oname == "theta_star":
    wall.verdict = verdict; wall.limit = limit; wall.max_shared = max_shared; ...
else:
    wall.second_origin = verdict            # ← 单个标量字段
    wall.second_origin_max_shared = max_shared
```

`else` 分支把**任何**非 `theta_star` 的原点写进**同一个** `second_origin`。
⇒ 直接加第 3、4 个原点会让它们**互相覆盖,只剩循环里最后一个**,
而且不会报错、不会少数据 —— 事件日志看起来完全正常。
**这正是本项目记过的"可信常数"故障形态**(`a-constant-reading-is-a-broken-probe`:
最危险的故障不是 `None`,而是一个看起来可信的值)。

⇒ 所以改动必须是:**先把标量写回改成按原点名索引的字典,再加原点**。顺序反了就会静默出错。

### 1.3 输出怎么变:二值检查 → 稳健性计数

`Wall.second_origin` 现在是单个 `Verdict`。改为**每个原点一条记录**:

```python
# 新增字段。`second_origin` 不删 —— 旧 run 的事件日志要能继续读(replay 兼容)。
origin_verdicts: dict[str, Verdict]        # {"theta_top1": "attributed", "theta_top2": ..., "default": ...}
origin_max_shared: dict[str, int | None]
n_origins_attributed: int                  # 分子
n_origins_probed: int                      # 分母
```

`for_prompt` 的措辞随之改成**带计数**,而不是现在的绝对句:

> `BLOCK_N`:在 **3 个高性能点中的 2 个**上,单独改成 128 会使编译器要求 122880 字节共享内存
> (本卡上限 101376)。**从空间默认配置出发时不触墙** ⇒ 这道墙由该点上其他 knob 的取值共同决定。

**为什么这是纯粹的增益**:它把一个**无法区分**的状态拆开了 —— 现在"这道墙只在唯一那个最优点成立"
和"这道墙在整片高性能区域都成立"在事件日志里**长得一模一样**,而对改写者来说这两件事价值差很多。

### 1.4 代价:可忽略,已实测

一次探针的**边际**成本 **0.49s**(18 个变体在同一个 worker 进程里 8.8s;
单独起进程是 16.7s,几乎全是进程启动 + torch/CUDA/KernelBench import)。
K=3 时每个墙 3 次探针 ≈ 1.5s,每候选 3 个墙 ≈ 4.5s,对 12h 的 run 可忽略。
`probe_total_s` 已经在事件里,可以事后减掉而不用假设。

批量路径**已经存在**(见 §1.2b),所以这里不需要新的成本护栏,只需要一个**回归测试**
锁住"K 个原点仍是一次 `prescreen_batch`"这个既有性质。

### 1.5 明确不做的事(以及为什么)

**不用"斜率预期提升后能超过 top-1"来选点。** 用户提出的这个方向逻辑上成立,但它会构成循环论证:
"预期提升"这个判断**本身就是斜率外推**,而斜率外推的可靠性正是我们还没证实的东西
(`boundary-saturation-does-not-predict-headroom`:文献推荐的最强空位信号,五种读法 ρ 全在
−0.11…+0.24,低于 incumbent 的 0.43/0.52)。用未证实的外推去**选择**探针位置,再用探针结果
去**支撑**那个外推 —— 结论会自证。

**top-K 是它的无循环替代**:用**实测**的高性能点,不用**预测**的。
如果将来 §3 的 P4 把"斜率是可用的分配先验"测实了,再回来加外推选点。

**不因这个改动缩小任何取值域。** 探针只是"多问编译器几次",不删值、不改预算、不改采样。

### 1.6 验收判据(必须先写测试再改)

1. **`test_a_third_origin_does_not_overwrite_the_second`** —— **最重要的一条**,直接针对 §1.2b。
   用 3 个原点跑 `_probe_walls`,断言三个裁决**都能取回**。
   在**当前**代码上这条测试必须**失败**(否则它没有测到那个覆盖 bug),改完才通过。
   这就是 `a-variant-that-changes-no-behaviour-is-not-a-variant` 要求的:变体必须真的改变行为。
2. `test_a_wall_probed_from_three_origins_reports_a_count_not_a_boolean`
   —— 构造一个在 2/3 个原点触墙的 fixture,断言 `n_origins_attributed == 2`,
   且 `for_prompt` 文本里同时出现 "2" 和 "3"。
3. `test_the_origins_are_measured_points_not_extrapolated_ones`
   —— 断言 `_theta_top_k` 返回的每一组参数都能在 `crun.trials` 里找到一个
   `status == "complete"` 的 trial(即**没有**任何被构造出来的点)。
4. `test_k_origins_are_still_probed_in_one_batch`
   —— 回归测试,锁住既有性质:断言 K 个原点仍只触发**一次** `prescreen_batch`。
5. `test_probe_top_k_of_1_is_byte_identical_to_today`
   —— `probe_top_k=1` 时行为与当前完全一致,使这个改动**默认可关**。
6. **一个正对照**:构造一个在 **0/3** 个原点触墙的墙,断言它**不**进 `for_prompt`
   —— 否则"计数"版本可能把 0/3 也渲染出来。
   (`probe-needs-a-positive-control`:阴性结论必须有一个会大声失败的正对照。)
7. `test_theta_top_k_skips_incomplete_and_zero_latency_trials`
   —— 复用 `_theta_star` 已有的过滤(`status != "complete"`、`params is None`、
   `latency_ms is None`、`ms <= 0`),断言这些都不会成为原点。

### 1.7 config

```yaml
v3:
  wall_attribution:
    probe_top_k: 1        # 默认 1 = 当前行为;建议实验用 3
    probe_second_origin: true   # 不变,仍是条件性对照
```

**默认 1**,理由与 `enabled: false` 相同:改动探针数会改变 GPU 时间,
不能让一个 run 悄悄变得与之前的不可比。

---

## 2. 【建议做,未批准】把墙检出扩到 `n_spills` 与 `occupancy`

依据见 `docs/resource-dimensions-and-which-walls-are-detectable.md` 的汇总表。
八维里除已实现的 `shared_bytes` 外,只有这两个可做:

**`n_spills`(优先)**
- 数据**已经在**:`worker_main.py:785` 与共享内存同一次探针返回 `num_spills`;
- 判据是硬边界,不需要标定阈值:某 knob 增大使 spills **由 0 变正**;
- 唯一有稳定资源→性能转化率的维度(`only-n-spills-has-a-stable-conversion-rate`)。

**`occupancy`(其次,且是 `n_regs` 的真正出口)**
- 判据:某 knob 增大使 occupancy 单调下降且触底(<0.30);
- 两个陷阱:极性是**反的**(低才绑定);它是**嵌套字段**
  (`profile.occupancy["occupancy"]`),平读会把"已测"读成"未测"。

**关键区别,决定了这不是"多传一个 limit"**:共享内存是**硬拒绝**(配置跑不起来),
这两个是**软墙**(配置跑得起来但被削)。软墙**不产生任何被拒参数组**,
而 `find_walls` 现在的全部输入就是被拒参数组 ⇒ **需要第二种判据函数**,
不能给现有函数加参数。这是一条独立工作项,不应塞进 2e。

---

## 2b. 【受阻塞,不是被否证】带宽 / 算力作为墙:等一台能开 ncu 的机器

**这一条不是"待批准",是"被测量能力挡住"**,记在这里以免日后误以为这条路被否证了。

**概念上它完全可以是墙**:撞到带宽上限 ⇒ 改算子降低带宽用量 ⇒ 代价是多用别的资源。
这正是 C2 想做的事,而且是算子优化里最主流的考量对象之一。

**挡住它的是"每候选的量测不到"**:
- 现在的 `pct_of_dram_peak` 分子来自 `TaskCost`(*"a property of the task, not of a candidate"*),
  是**任务级常数** ⇒ 一个任务内它就是 `1/延迟` 乘常数,实测 ρ(它, 1/latency) = **+1.000**(4/4 run);
- `FlopCounterMode` 数候选自己的 FLOP:**完全融合的候选报 0** ——
  算术没经过 dispatcher。**候选越好读数越接近 0,信号与目标反向**;
- aten 层字节数:是真每候选测量,但*"blind inside a fused kernel, which is where a good
  candidate works"* ⇒ 同样越好越失真;
- **ncu:`ERR_NVGPUCTRPERM` 在容器里永久封死**,即使 ncu 存在也照样被 gate。

**唯一能解除阻塞的是一台裸机或有 `CAP_SYS_ADMIN` 的机器。** AutoDL 容器给不了。
解析推算(从 tile 几何算字节)**不要做** —— 本项目已两次证明这类推算不可靠:
手写共享内存约束中位只有真值 32%;`BLOCK_M*BLOCK_N*stages` 在 15/15 个候选上
可行集与不可行集重叠。

⇒ **行动项**:如果将来拿到能开硬件计数器的机器,这是第一件该做的事,
因为它一次性把两个最主流的优化对象从"任务级饱和度"升级为"可归因的墙"。
详见 `docs/resource-dimensions-and-which-walls-are-detectable.md` §3。

---

## 3. 【待批准】`categorical_distance_func`,以及 S7

### 3.1 `categorical_distance_func`(低风险)

现在 `tpe.py:64` 对**所有** knob 一律 `suggest_categorical`,于是 TPE 把
`BLOCK_N ∈ {16,32,64,128}` 当成无内部结构的标签:知道 64 好,对 128 的概率**没有任何影响**。
实测后果:有墙 knob 的最高取值被采到的平均归一化位置 **0.52** —— 是**撞**上墙,不是走向墙。

**前提已实测成立**(这是它与其他候选改动的唯一区别):
平均|相邻取值延迟差| / 平均|远距取值延迟差| 的 **p50 = 0.735** < 1 ⇒ 序携带信息。

**"哪些 knob 能给距离"已经用真实语料量化过**(9 份 events.jsonl,999 条 domain 声明):

| | 数量 | 说明 |
|---|---|---|
| `kind="int"` | 858 | 216 个不同 (名字, 取值集);取值个数 3–6 |
| `kind="str"` | 141 | 只有 5 个名字,全是精度/模式开关(`COMPUTE_DTYPE` 等)⇒ **真无序,正该排除** |
| `kind="float"` | 0 | 语料里没出现 |
| **取值全是数字字符串的 `str` knob** | **0** | 我担心的陷阱在真实语料里**没发生** |
| **真是 0/1 开关的 `int` knob** | **0** | "int ⇒ 有序"在这份语料里成立 |
| 取值个数 <3 的 `int` knob | 5 | 距离函数几乎没东西可平滑,应排除 |

**判据必须落在 `choices` 的实际值上,不能只看 `kind` 标签**(标签是 agent 声明的,
虽然 `guard.py:113` 会强制校验类型,但读数不该依赖它):

```python
def ordered_choices(domain) -> list[float] | None:
    if any(isinstance(c, bool) for c in domain.choices):
        return None          # float(True)==1.0 会让开关看起来像有序轴
    nums = [_as_num(c) for c in domain.choices]
    if any(n is None for n in nums):
        return None          # 同时抓住 kind="str" 但取值是 "128" 的情况
    if len(nums) < 3:
        return None          # 2 个点之间的"距离"无法区分平滑与两个孤立标签
    return nums
```

**这套逻辑已在仓库里存在** —— `wall_attribution._as_num`(第 105 行)做的就是这件事,
连 `bool` 排除和理由都写了。⇒ 复用一个已验证的判据,不是新建启发式。
覆盖面 **858/999 = 86%**,其中 211/216 有 ≥3 个取值。

### 3.2 S7:斜率引导采样(高风险,需批准)

**唯一的结构差别**:把 2e 从调参循环**外**搬进**内**,每 10 个 trial 用现成的
`TuningStatsAnalyzer.analyze` + `find_walls`(两者都已存在且是纯函数,不碰 GPU)重算,
再用 `study.enqueue_trial` **插队**"高斜率 knob 取未测过的相邻值,其余取当前最优"。
**只加候选点,不删任何取值**;不改 `trials_per_space`。

**可行性有实测**:回放真实 trial 顺序,半程(40 trial)就有可探针的墙,还剩 40 个 trial 可花。

**三条风险**:
1. **早期排序不稳,符号会翻** —— `BLOCK_N` 在 25% 处 **−89.0%**、50% 处 **+43.7%**
   ⇒ 引导必须建议性、可撤回、每次重算,不能锁定。
2. **墙会出现又消失** —— `cand-948343ba`:25% 有 2 个 → 75% 有 0 个 → 100% 有 1 个(换了 knob)。
   机制上必然:墙的定义是被拒值落在**已测范围之外**,采样扩大后范围会追上被拒值。
3. **最重的一条:"斜率是好的分配先验"这个前提没被证实** —— 见 §1.5 的 ρ ≤0.24。
   而且斜率 vs 覆盖度 ρ=−0.431,覆盖度<1.0 的 24 个参数里 **14 个(58%)有墙**
   ⇒ **朴素的"先测没测过的"已能找到多数墙**。S7 可能把预算花在"斜率大但绝对收益小"的维度上而变慢。
4. **可比性**:改 trial 分配 ⇒ 与之前的 run 不能合并,需要一对 12h 对照臂。

**必须事先声明的可证伪预测**(否则结果无法解读):
- **P1** `RESOURCE_WALL_ATTRIBUTED` 首次出现时间(归一化到调参进度)**提前**;
- **P2** 每候选 probe-worthy 墙数**增加**;
- **P3** 覆盖的族比例从 25% 上升;
- **P4** 若 P1–P3 成立而延迟仍不改善 ⇒ **「更早知道墙」不是瓶颈**,该负结果同样是结论,
  并把 C2 的问题定位到"改写这个动作本身的转化率"。

**做成可控模块:框架里已有同形状先例。** `tuning/deweight.py`(S1b)就是一个可开关的采样偏置:
注入点 `OptunaTPETuner.__init__(deweight_reject=...)`,`None` = 关且**字面等于**旧路径,
施加位置在 `ask()` 循环内,效果经 `snapshot()` 进事件日志**从日志可测而非靠论证**。
S7 照同一套接线,三个开关分开(**不要合成一个布尔**):
`enabled` / `recompute_every`(10)/ `max_enqueued_per_recompute`
—— 这样 P4 那种"机制生效但无收益"能定位到是**时机**还是**剂量**。默认 `enabled: false`。

---

## 4. 【不动】PRUNED 报法:只记录事实,本轮不改

用户判断改动风险较高。**本轮不做任何相关改动**,以下仅为记录,避免日后重复讨论:

**现状**(`tpe.py:138`):硬性不可行(`infeasible_shared_memory` / `guard_rejected` /
`materialize_error`)报 `PRUNED`,其余报 `FAIL`。理由:Optuna 把 pruned 留在 TPE 模型里、
把 failed 丢掉(实测:12 个 FAIL 只剩 1 个可见,12 个 PRUNED 剩 13 个)。

**实测收益是弱的**:每个 pass 前半 vs 后半的不可行比例,**26 个 pass:12 降 / 5 平 / 9 升**
⇒ 这个通道没有明显把采样器推离不可行区。我的判断:PRUNED 的 trial 进了模型但**没有目标值**,
而 TPE 建的是"好组/坏组"密度比 —— 无 value 的 trial 进不了任何一组,只是让 Optuna
"记得访问过"。**"记得访问过"与"知道这里不可行"是两件事。**

**两条已知的债**(不改,但要记住):
- Optuna 内建 `constraints_func` **不对 pruned/failed 调用** ⇒ 与现行报法**互斥**。
  走内建约束就得把这些 trial 改报 `COMPLETE` + 违约量,而它们**没跑过、没有延迟值**可填。
  ⇒ **本轮不走这条路。**
- PRUNED 现在承载**三种**语义(硬不可行 / guard 拒绝 / S1b 降权),事件日志能靠
  `failure_kind` 分开,但在 Optuna 内部**完全无法区分**。将来若引入真正的中途早停,会混在一起。

---

## 5. 建议次序

1. **§1(已批准)** —— top-K 原点。零机时风险,`probe_top_k=1` 时与今天逐字节相同。
2. **§3.1** —— `categorical_distance_func`。唯一"前提已实测成立、不删取值、不动预算、
   不需对照臂"的改动,且它**修的是 S7 也依赖的地基**:若采样器连序都没有,
   S7 插进去的"未测过的相邻值"对模型仍然只是又一个孤立标签。
3. **离线量化(零机时)** —— 族内非最优候选的墙,有多少在最优候选的源码上仍成立。
   这个数决定放宽族覆盖率是扩大样本还是制造噪声。**注意**:我此前说"墙文本只发给被归因的族"
   是不准确的 —— `orchestrator.py:2590` 是**逐族**取自己 `family.best` 的 `wall_text`,
   投递没有偏袒;25% 覆盖是因为**只有 25% 的族其最优候选恰好有一道被归因且斜率为正的墙**。
   ⇒ 提高覆盖率只能放宽四道门之一,那是**改判据**,不是"多发几份",所以我撤回它"零风险"的说法。
4. **§2** —— `n_spills` 墙(需要第二种判据函数)。
5. **§3.2 S7** —— 需批准,一对 12h。
6. **§4** —— 不动。

**无论走哪条,报告都必须保留当前的负结果**:"2e 只进 prompt 时无端到端收益"是已测事实
(同箱同 mode 的干净对比里关掉 2e 的臂快 8.0%,噪声底的 2.0 倍),改进版若成功,它就是对照。
