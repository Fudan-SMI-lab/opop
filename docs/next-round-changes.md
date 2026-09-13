# 下一轮改动清单(已批准 / 待批准 / 明确不动)

**日期** 2026-09-13 · **checkout** `D:\Pyhon_projects\opop\v3` @ `02a40b9`
**状态**:本文件是**待实施**清单,以下**没有一条已经写进代码**。

**用户已给的边界**(逐条遵守,不再重新征询):
- ✅ **已批准**:放宽墙探针的"仅从最优点出发"约束(§1)。
- 🚫 **明确不动**:PRUNED / FAIL 报法相关的一切改动 —— 用户判断"改动风险较高,建议先不要变动"。
  ⇒ 这同时意味着 **Optuna 内建 `constraints_func` 这条路本轮不走**(它与 PRUNED 报法互斥),
  §4 只记录事实不提改动。

---

## 0. 汇总表:所有待实施项一览

| # | 项 | 状态 | 改哪里 | 机时代价 | 需对照臂 | 主要风险 |
|---|---|---|---|---|---|---|
| **1** | 墙探针从 **top-K** 高性能点出发(不只 top-1) | ✅ **已批准** | `orchestrator.py:1494-1507` 取点 + **`:1597-1605` 写回改字典** | ≈4.5s/候选(批量路径已就位) | ❌ 不需要 | 写回分支会**静默覆盖**(§1.2b),顺序必须先改字典再加原点 |
| **2** | `n_spills` 软墙检出 | ✅ **相关性已过关**(ρ 最大 +0.339);待批准实现 | 需**新判据函数**(不是给 `find_walls` 加参数) | **0**(profile 已有该字段,无需探针) | ❌ 不需要 | 无独立探针确认;27/56 峰形;**适用面 43%**(32/56 赢家 spill=0) |
| ~~2'~~ | ~~`occupancy` 软墙~~ | ❌ **已否证(本轮实测)** | — | — | — | 相关性过关但**判据方向与最优点相反**:峰形 37/56、赢家中位位置 0.33、仅 2/56 在顶端。详见 `analysis-soft-walls-spills-and-occupancy.md` §2 |
| **2b** | 带宽/算力作为墙 | 🔒 **受阻塞** | — | — | — | **需裸机或 `CAP_SYS_ADMIN`**;2/2 台 AutoDL 实例已复验被封 |
| ~~2c~~ | ~~`logical_bytes` 维度~~ | ❌ **已撤回(本轮实测否证)** | — | — | — | 对 tile 乘积 **ρ=+0.978**、对 `shared_bytes` **ρ=−0.897**;逐元素算子上恒定 ⇒ 是 knob 的代数改写。详见 §2c |
| **3.1** | `categorical_distance_func`(给采样器"序") | **待批准**,低风险 | `tpe.py:64` | 0 | ❌ 不需要 | 仅作用于数值型 knob;前提已实测(p50 0.735) |
| **3.2** | **S7** 斜率引导采样 | ✅ **已实现**(`tuning/slope_guide.py`,三开关默认全关);**待跑一对 12h** | `tuning/slope_guide.py` + `tpe.py:enqueue/n_told/drawn_keys` + `orchestrator._slope_guide_step`(照 S1b 接线) | **一对 12h** | ✅ **需要** | 破坏 run 间可比性;"斜率是好先验"**未证实**(ρ ≤0.24);**验收实测:仅 10.5% 候选触发,95.1% 的重算"根本没有可行动的墙"** ⇒ S7 **不修**覆盖率而是继承它,见 `analysis-s7-acceptance-firing-rate.md` |
| **4** | PRUNED / FAIL 报法 | 🚫 **不动**(用户判断) | — | — | — | 与内建 `constraints_func` 互斥;实测通道弱(26 pass:12 降/5 平/**9 升**) |
| **5** | 离线量化族覆盖率 | 建议做 | 纯读已有日志 | **0 机时** | ❌ | 决定"放宽族覆盖"是扩样本还是造噪声 |

**建议次序**:1 → 3.1 → 5 → 2 → 3.2。理由见 §5。

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

## 2. 【`n_spills` 相关性已过关;`occupancy` 已否证】软墙检出

**本轮已完成实测** —— 完整数据与流程影响见
[`docs/analysis-soft-walls-spills-and-occupancy.md`](analysis-soft-walls-spills-and-occupancy.md)
(5 个已备份 run、2788 个完成 trial、56 个候选、跨 4090 与 A800)。

### 相关性(§0 定的门槛:|ρ| ≥ 0.9 即不得实现)

| 信号 | vs `n_regs` | vs `shared_bytes` | vs tile 乘积 | vs 延迟 | 结论 |
|---|---|---|---|---|---|
| `n_spills` | +0.339 | +0.257 | **−0.048** | +0.275 | ✅ 过关 |
| `occupancy` | −0.497 | −0.387 | +0.032 | **−0.053** | ✅ 过关 |

两者之间 ρ = **−0.500**(中度相关,不互为改写)。
⇒ **相关性这一关两个都过**,与 `logical_bytes`(+0.978)形成明确对照。

### 但 `occupancy` 过不了"建议方向是否与最优点一致"这一关 ⇒ **不做**

- 候选内部:高 occupancy 的三分之一比低的快 **23.1%**(中位);
- **但** 关系是**峰形**(37/56 个候选中间最快),赢家在自己范围的归一化位置**中位 0.33**,
  在顶端(≥0.8)的只有 **2/56**;
- ⇒ **"提高 occupancy"在实测最优点处是反向建议。** 把它报给改写者会让它追一个测量显示并非最优的方向。
- 顺带得到的有用事实:赢家处的 `occupancy_limiter` 是 **38 个 registers / 18 个 shared_memory**
  —— 这个字段**仍然有用**,但用途是给 `n_spills` 墙提供"为什么"的说明,不是自己当一个墙。

### `n_spills` 通过了同一关 ⇒ **可做**

- 关系形状:27 峰形 / **18 单调符合预期** / 6 反向;
- **赢家在自己 spill 范围的位置:中位 0.00,56/56 全在最低端** ⇒ "减少 spill" 与最优点方向**一致**;
- 判据在真实数据上产出 **49 条结论 / 29 个候选**(454 个 (候选,knob) 对里);
- **适用面 43%**:32/56 个赢家 spill 已为 0,对它们在最优点处无墙可找 ⇒ 报告必须给分母。

**关键区别,决定了这不是"多传一个 limit"**:共享内存是**硬拒绝**(配置跑不起来),
`n_spills` 是**软墙**(配置跑得起来但被削)。软墙**不产生任何被拒参数组**,
而 `find_walls` 现在的全部输入就是被拒参数组 ⇒ **需要第二种判据函数**,不能给现有函数加参数。

**流程影响(四个环节,只改第 ②④ 两处)**:①trial 判定**不变**(软墙不拒绝配置,
不影响成功率/预算/Optuna 状态)⇒ **不破坏 run 间预算对等,这是它与 S7 的根本区别**;
②`find_walls` 输入端需新判据函数;③**无需探针**(spills 已在每个 trial 的 profile 里)⇒ 零 GPU;
④复用现有 `RESOURCE_WALL_ATTRIBUTED` 事件与投递链路。详见该文档 §5。

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
详见 `docs/analysis-can-we-measure-per-candidate-bandwidth.md` §1。

---

## 2c. 【已撤回 —— 本轮实测否证】`logical_bytes` 作为新维度

**我曾建议**:既然计数器不可用,就把 Triton IR 解析扩到 2-D + 循环次数,
算出逐候选逐参数的**逻辑字节数**,做成新维度 `logical_bytes`,与 `n_spills`/`occupancy` 同属软墙。

**实测否证了它**(`scripts/probes/logical_bytes_vs_tile_knobs.py`,box1,
真实 tiled matmul 1024³ fp16,18 个 tile 组合):

| 相关性 | ρ | 读法 |
|---|---|---|
| `per_instance` vs tile 乘积 `BM*BN*BK` | **+0.978** | **几乎就是同一个量** |
| 总量 vs `shared_bytes`(编译器给的) | **−0.897** | 强相关,新信息很少 |
| 总量 vs tile 乘积 | **−0.871** | 同上 |
| 总量 vs 延迟 | −0.307 | 中度 |

**三条独立的否证理由:**

1. **代数上它就是 tile knob。** tiled matmul 的逻辑总量约化为
   `M*N*K*width*(1/BM + 1/BN)` —— 只是 tile knob 与问题形状的函数。
   而 tile knob 采样器本来就在直接调,`shared_bytes` 本来就在跟。
2. **逐元素算子上它是常数。** BLOCK 取 256/512/1024/2048,总量恒为 201326592。
   **没有 knob 能移动它 ⇒ 没有 knob 能被它截断 ⇒ 结构上不可能成为墙。**
3. **L2 盲区被量化**:逻辑总量 / 必需字节 = **12.0x–32.0x**(18 个组合全部),
   而逻辑计数分不出哪些重读命中了 L2 ⇒ 也当不了饱和度的分子。

**这是同一个陷阱第三次出现** —— `pct_of_dram_peak` 对 1/latency ρ=+1.000、
SOL 头寸对 1/latency ρ=+1.000,这次是对 tile 乘积 ρ=+0.978。
**共同形态:一个新"维度"其实是已有量的代数改写。**
⇒ **添加任何新维度前必须先测它与已有量的相关性**;三次都是这一步被跳过。

**保留的部分**:`scripts/probes/ir_derived_bytes.py` 在两个已知真值的 kernel 上
ratio **1.0000**、正对照通过,是一个可靠的**核对工具**
(例如验证某次改写是否真的减少了访存请求),**只是不该升级成维度**。

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

> **实施状态(2026-09-13)**:**已实现**,`tuning/slope_guide.py`,三个开关
> `v3.slope_guide.{enabled, recompute_every, max_enqueued_per_recompute}` 加一个 `use_soft_wall`,
> **默认全关**;46 个测试 + `revert_check_s7.py` **35/35 变体全部被捕获**。
>
> **实现后新增的两项实测,都必须与下面的预测一起读:**
>
> 1. **可行性回放**(`s7_feasibility_replay.py`):墙在半程可得的比例 **25/34 = 74%**、
>    最早前缀中位 **0.38**;但早期 walled knob 到 100% 仍成立的只有 **18/34 = 53%**
>    (12 个消失、4 个换 knob)。⇒ 引导做成**建议性 / 每次重算 / 有剂量上限**,三者都是为这个 53% 设计的。
>    **硬墙版本在本机不可回答**:151 个候选 0 个有共享内存拒绝记录(本地 run 全早于筛查)。
> 2. **验收测量**(`s7_acceptance.py`,把**发货代码**在 19 个 L3 run / 152 候选 / 790 次重算上回放):
>    触发 **16/152 = 10.5%**、共插 48 个点、首次触发中位 **0.25**(够早,P1 前提成立)、
>    被行动 knob 的尾部斜率中位 **+34.8%** —— 但 **751/790 = 95.1% 的重算是因为"根本没有可行动的墙"**。
>    ⇒ **瓶颈是判据而不是接线**;S7 **不修** C2 的覆盖率问题,而是**继承**它。
>    这与 2e 的 25% 族覆盖、item 5 的"提高覆盖只能改判据"是**同一结论的第三个角度**。
>    ⇒ **P2 / P3 是最可能失败的两条**,且失败应归因于**判据适用面**,不是 S7 的时机或剂量。
>    完整解读见 `docs/analysis-s7-acceptance-firing-rate.md`,原始输出见 `docs/s7-acceptance-{soft,hard}.txt`。


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

### 4.1 【本轮不改,已实测定量】共享内存拒绝被误标成 `runtime_error`

**S7 对照跑期间发现**(2026-09-13,box4 s7-treatment `tr-4b6d6cdf`)。

**缺陷**:`worker_main._classify_exception`(`worker_main.py:35-39`)只识别 `out of memory`,
其余一律归为 `runtime_error`。而 Triton 在 launch 期抛的是
`triton.runtime.errors.OutOfResources: out of resource: shared memory, Required: 131072,
Hardware limit: 101376` —— 这是一次**拒绝**(配置的属性、必然复现),不是崩溃。

**为什么平时看不到**:`compile_screen` 在 launch 前就拦住这类配置并写对标签,所以健康的 run 只会看到
十几个 `infeasible_shared_memory` 而没有这种 `runtime_error`。要触发误标,必须**先让 screen 自己失败**
—— 而"screen 失败绝不构成判决"是刻意设计(`correctness.py:299-303`),于是真 trial 照跑并撞上运行时。
本次的触发原因:`ptxas` 在一个 8 MB / 161,770 行的 PTX 上跑爆了 `screen_timeout_s`。

**影响面(三个消费者,方向都是坏的)**:
- `orchestrator.py:1573` —— `find_walls` 的**全部输入**就是 `failure_kind ==
  "infeasible_shared_memory"` 的 trial。误标的墙对 C2 与 S7 **完全不可见**。
- `tpe.py:218` —— 硬性不可行报 `PRUNED`(留在 TPE 模型里),`runtime_error` 报 `FAIL`(被丢掉)。
- `deweight.py:71` —— 把 `runtime_error` 汇总成"整个候选有缺陷"的证据,而单点资源拒绝不是。

**实测频率**(`scripts/probes/mislabelled_walls.py`,box4 全部 7 个 run / 3095 个 trial):
**14 / 725 = 1.9%** 的共享内存拒绝被误标;7 个 run 里 3 个至少出现一次
(arm2 7 个、arm3 6 个、s7-treatment 1 个;两个继承自 box2 的 run 各 0 个)。
被误标的配置全部是大 tile(`BLOCK_M ≥ 128` 且 `BLOCK_N ≥ 64`),墙钟 123–1762 s
—— 也就是说它们同时也是**最贵**的 trial,因为都得等 `ptxas` 跑完才被拒。

**为什么本轮不改(而不是"改了更安全")**:
1. 改标签就等于把这 14 个 trial 从 `FAIL` 翻成 `PRUNED`,直接落在 §4 用户已判定
   "风险较高、本轮不动"的那段逻辑上。
2. worker 侧的改动**下一个 trial 就生效、无需重启**(见
   `opop-v2-worker-vs-driver-fix-propagation`)。对照跑中途改,会在**两臂各自不同的 trial 序号处**
   改变分类语义 —— 那是往一个正在测量 `find_walls` 输入的实验里注入混淆,是制造第二个自变量,
   不是修复。
3. 1.9% 落在本项目已知的噪声底(重测 ±2–4%、每 trial std 16%)以内,不足以左右本对照的结论。

**⚠️ 第 3 条已被削弱(2026-09-14 新证据)—— 按计数看小,按后果看不小。**
第三个现场:S7 控制臂的改写候选 `cand-8071c78e`,同一异常出现在**见证门**上
(`Required: 131072 / Hardware limit: 101376`,记为 `runtime_error`)
⇒ **整个空间被拒**并触发 loop A。而 loop A 的诊断精确到机制:

> `_gemm` 在**操作数 dtype** 上经共享内存暂存,而两个操作数(x 与 `c_attn.weight`)都是 fp32
> —— `COMPUTE_DTYPE='bf16'` 的转换发生在**载入之后的寄存器里**,并不缩小缓冲区。

修法 `GEMM_STAGES` 3→2 使暂存减半到 65536 B,对**每种** `COMPUTE_DTYPE` 都装得下,其余逐字节不变。

⇒ **这是一道真墙,被 loop A 正确解开,而 2e 全程看不到它。**
所以这个缺陷的代价不止"墙数偏低约 2%",还包括:
- 它让**真墙对 C2 隐形**,解墙的功劳记在 loop A 名下 ⇒ 这是 **C2 覆盖率偏低的一个此前未计入的成因**;
- 它把拒绝升级成**整个空间被拒 + 一轮 loop A**(本例花掉一次 repair + 两次 parameterizer 调用),
  而正确标注下这只是一个 `PRUNED` 的点。

**按机时与按后果计价都比按计数计价大。** 结论仍是本轮不改(理由 1 与 2 不变:
worker 侧改动下一个 trial 即生效,中途改会在两臂不同序号处改变分类语义),
但**不要再引用"落在噪声底内所以无关紧要"这句**。

**下一轮怎么改**(方案已定,勿再讨论):在 `_classify_exception` 里按**异常自己的话**判定
—— 匹配 `out of resource: shared memory` / `OutOfResources` 即返回 `infeasible_shared_memory`。
必须连带做的两件事:(a) 从 `Required:` / `Hardware limit:` 抓出数字写进 `failure_detail`,
因为 2e 要区分"超一格"和"超一倍"(`correctness.py:269-282` 已说明这一点);
**范围要说准**:本条只针对 **launch 期** Triton 异常那一种文本(`Required: N, Hardware limit: M`)。
**compile-only screen 的拒绝已经带数字**,格式不同
(`_flash_attn requires 229376 bytes ... per-block opt-in limit of 101376`,**39/39 都有**,
见 §4.5)⇒ 不要写成"这两个数字目前完全缺失"。
真正的共同缺口是它们**只以自由文本存在**,读者必须写正则去挖
(我为此挖错过一次,把 0/39 匹配当成了"字段缺失"的证据)⇒
(a) 的正确形态是**把 required / limit 落成结构化字段**,两种来源统一。
(b) 正对照测试 —— 一个真正的非资源崩溃仍须归为 `runtime_error`,否则"修法"可以退化成
"把所有崩溃都叫墙"。

### 4.1b 【同一缺陷的第二笔代价:机时,不只是标签】

上面说"1.9% 在噪声底以内",那是就**墙的计数**而言。就**机时**而言代价大得多,已在 S7 对照跑现场量出:

处理臂那唯一一个被误标的 trial(`tr-4b6d6cdf`,`BLOCK_M=128, BLOCK_N=128, NUM_WARPS=1,
NUM_STAGES=1, ieee, split3`)墙钟 **1081 s**,占该臂**全部 GPU 时间的 39.9%**(0.753 h 中的 0.30 h)。
它把两臂的 trial 速率从 45.2 : 41.1 拉成 **45.2 : 34.8 trials/h**:

| | 控制臂 | 处理臂 |
|---|---|---|
| 实际速率 | 45.2 trials/h | **34.8** trials/h |
| 若那一个 trial 只花中位数(23 s) | 45.2 | **41.1** |
| 每 trial `job_wall_s` 中位数 | 20.9 s | 23.2 s(**1.11x**,在容差内) |

即**两臂速率差的主体是这一个 trial**,余下部分是 agent 延迟(占墙钟 39% vs 33%),
不是逐 trial 的 GPU 成本差异 —— `check_arm_search_parity.py` 也据中位数把它正确判成
`TAIL (does not compound)`,`PARITY OK`。

**为什么这笔代价是该缺陷造成的**:这个配置在 launch 时才被 Triton 拒(`Required: 131072 /
限 101376`),而它之所以能走到 launch,是因为 `ptxas` 在一个 8 MB / 161,770 行的 PTX 上跑爆了
`screen_timeout_s` —— 于是"screen 失败绝不构成判决"(刻意设计)让真 trial 照跑。
标签修对**不会**省下这 1081 s(那笔钱花在 `ptxas` 上,发生在分类之前),
但它会让这类点进入 `PRUNED`、从而让 TPE 知道这一片区域不可行,**减少后续同类点被重复抽到**。
所以下一轮的收益应按"减少重复撞墙的次数"计价,**不要**记成"每次省 1081 s"。

**已复验的期限(两臂一致,不是猜测)**:`screen_timeout_s` = **120 s**
(`screen_floor_timeout_s` 120 胜过 `prescreen_timeout_s(1)`=33),即 **D9 的修复在本次实验里是生效的**
—— 那个 screen 只花了 120 s,**不是**旧的 1200 s。真 trial 期限
= `build_timeout_s + eval_timeout_s` = 1200 + 600 = **1800 s**,由
`worker_client.py:301` 的 `proc.communicate(timeout=...)` 执行、超时在 310 行 kill,
并且每个 job 独立进程组(288 行),所以一个 job 超时**不会**连带杀掉共享通道里健康的另一个 job。
两臂至今 `job_timed_out` 与 `failure_kind: "timeout"` **均为 0**。

⇒ 修正我先前的一处表述:这 1081 s 里**没有** 1200 s 的 screen 成分,
它几乎全部是真 trial 内部的 `ptxas` 时间。D9 那笔 2.1 h 的账是 box1 旧配置的历史,本次不适用。

参见 [[a-timeout-censors-the-metric-that-would-price-it]]:超时不写 `out.json`,
所以这类最贵的编译在常规统计里**结构性缺席** —— 本次能测到它,只是因为它恰好没超时。

---

## 4.2 【本轮不改,已实测】沙箱救援丢失改写意图 ⇒ 该改写永久不可归因

**现象**(S7 控制臂,`REWRITE_PRODUCED` seq=477):

```
family_id      "fam-5546fb5f"
candidate_id   "cand-8071c78e"
hypothesis_id  ""
change_summary [recovered from sandbox after a transport failure; the agent's own summary never arrived]
```

**救援路径本身是对的**(`modules.py:715`、`base.py:306`):传输失败时把 agent 已写出的候选文件
抢救出来,摘要坦白标注不可用,且仍走 `check_output` ⇒ 半写文件照样被拒。**代码没有缺陷。**

**但产生一个分析层后果,而且是静默的**:该改写的 `hypothesis_id` 为空、意图无记录
⇒ **来源永久不可归因**。受影响的读数:
- **P3(族的墙文本覆盖率)** —— 无法判断这次改写是否收到过墙文本;
- 任何"改写依据什么"的统计(如 `rewrite_provenance.py`)会把它计入"无法归因",
  而"无法归因"与"确认无墙引导"是两件事;
- [[only-nine-percent-of-hypotheses-are-ever-implemented]] 那类实现率统计的分母。

**下一轮的改法(需改 agent 路径,故不中途做)**:救援时**从抢救出的源文件里回填 `hypothesis_id`**。
可行性已实测:处理臂两个正常改写的 `change_summary` 分别以 `H1:` / `H2:` 开头,
即 rewriter 自己会把假设 id 写进产物文本 ⇒ 从源文件首部或注释里正则提取即可。
**必须连带**:(a) 提取失败时**保持为空**而不是猜一个(空是诚实的,猜是伪造归因);
(b) 写正对照 —— 一次真实的救援必须仍然被标为"摘要不可用",
否则修法会退化成"给每个救援编一个 id"。

**代价量化**:本对至今 3 个改写,1 个(33%)意图丢失。样本极小,但方向是 100% 的信息损失
而非部分损失 —— 这个改写在任何来源分析里都只能计入"未知"。

---

## 4.3 【不是代码改动,是实验设计】C2 的新颖性被自己的基线威胁 —— 下一轮必须测三件事之一

**实测(box4,S7 对照跑,`scripts/probes/analyst_does_c2_already.py`)**:

| 臂 | 假设数 | 只提上限 | 只提斜率 | **两者兼具(= C2 形状)** |
|---|---|---|---|---|
| **控制**(`slope_guide: False`,0 道已归因墙) | 24 | 5 | 0 | **4(17%)** |
| 处理 | 18 | 3 | 1 | 1(6%) |

控制臂 analyst 原文:

> `Picks up the boundary trends now blocked: stages 1->4 gave -25% at the winning tile;`
> `BLOCK_N 32->128 was monotone -24%`
> `Unblocks the shared-capped directions: NUM_STAGES 6-8 and/or BLOCK_N=256 at 4 stages`
> `fit under the 101376 B cap (BN=256 bf16 tiles s4 = 98304 B exactly)`

**这是设计使然,不是意外**:analyst 的 prompt(`agents/modules.py:1059-1088`)把
`at_boundary`+方向、effect size、逐值失败率、best 处资源占用、`docs/device.md` 硬上限
全部交给它,并在 **1079 行**明确要求它判断某方向是否 "prevented by a hardware/resource limit"。

⇒ **"2e 让框架能做到以前做不到的事"已被否证。** 这比 BE-CBO 那条更近:那是外部先例,
**这是我们自己的基线**。

**C2 仍可主张,但只能是下列之一,且每条都要单独测量**:

| 主张 | 可测量的形式 | 已知的支持/反对 |
|---|---|---|
| **更早** | 2e 在调参结束即产出;analyst 要等完整统计 ⇒ 比较"首次可得的 trial 序号" | 待测。S7 本就是把它提前到调参**内部**,但 16 次重算 0 投递 |
| **更可靠** | 同一处截断,analyst 命中的比例 vs 2e 命中的比例 | 待测。需先有可归因的墙(本对 0 道) |
| **定量更准** | 墙给出精确 `over_ratio`;agent 是估算 | **有支持**:[[triton-flash-attn-shared-memory-formula]] 实测 agent 手写共享内存约束中位只有真值 **32%** 且从不拒绝 |

**"定量更准"是目前唯一已有证据的一条**,而且它把 C2 从"新能力"改写成"精度替代 agent 估算"——
这是一个**更小但站得住**的主张。下一轮若要保 C2,应优先设计这条的实验。

**不要**据上表声称"控制臂 analyst 比处理臂强":匹配器是粗糙正则,4 vs 1 在 24/18 基数上是噪声。

---

## 4.4 【本轮实测新发现,不是代码改动,是 C1 的一条失败模式】改写声明的资源效应可以是恒等

**现场**:`s7-treatment/run-l3-43-20260913-202332`,`cand-6f5cdc80` → `cand-7d02bbab`(H1),
box4。这是本对第一个**关闭**的改写空间。

改写声明(analyst H1 原文):"Keep Q/K/V/P tiles in bf16 inside `_flash_kernel` …
**frees ~40% of operand registers -> occupancy 17%->~30%** and fewer ALU cast instructions"。

**在父代最优点的同一组 6 个 knob 上复测子代**(`scripts/probes/rewrite_source_vs_knob.py`):

| | parent | child |
|---|---|---|
| latency | 3.3649 ms | 3.3644 ms(**0.015%**,远在 ±2–4% 噪声底之下) |
| `n_regs` | 210 | **210** |
| `n_spills` | 0 | 0 |
| `num_warps` | 2 | 2 |
| `occupancy` | 0.1667 | **0.1667** |
| `shared_bytes` | 18432 | **18432** |
| `occ_limiter` | registers | **registers** |

**六个维度全是恒等。** 而 `rewrite_vs_promise.py` 报的 **+3.06%**(3.3649 → 3.2620 ms)
**全部来自 TPE 在新空间里换到了另一个 knob 点**,不是源码改动的功劳。

### 已离线量化:40 个改写,28 个可判定,**3 个(11%)是空操作**

**零机时**,读 box4 上 6 个 run 的 `events.jsonl`
(`inherited-from-box2` ×2、`arm2`、`arm3`、S7 两臂),
工具 `scripts/probes/rewrite_source_vs_knob.py`(10 项自检,含 identity/moved 与父代时间序两向对照):

| | |
|---|---|
| 改写总数 | 40 |
| **可判定(存在匹配点)** | **28** ← 唯一诚实的分母 |
| 不可判定 | 12(空间仍开 5、取值从未被采样 7、无同族 0、knob 集不交 0) |
| 源码改动**动了** ≥1 个已测维度 | **25 / 28(89%)** |
| 源码改动在全部维度上**恒等** | **3 / 28(11%)** |
| 固定 knob 下源码贡献为负 | 3(−14.34%、−26.07%、−92.89%) |

动的时候动哪些维度:`shared_bytes` 17、`n_regs` 16、`n_spills` 13、`occupancy` 8、
`num_warps` 5、`occ_limiter` 3。

⇒ 按源码自身算,**28 个里 6 个(21%)在匹配点上没有正贡献**(3 空操作 + 3 变差)。

> **⚠ 本节先前的数字(17 可判定 / 4 空操作 24% / 5 负 / 53% 无正贡献)是错的,已作废。**
> 成因:父代取"族内最优的已关闭候选"而未加时间序约束,于是当一个改写被**自己的后代**超过后,
> 它被拿去和后代比,改善被打印成倒退。实测两处:`cand-7d02bbab`(H1)与 `cand-c56d8286`(H2)
> 互为父代,同一对数字打印两遍、符号相反(+10.00% / −11.11%)。
> 修法是按事件 `seq` 约束父代必须**在该改写产生之前**已关闭。修正后可判定数从 17 升到 28
> —— 因为正确父代的最优点更容易在子代空间里被采到 —— 且空操作率从 24% 降到 11%。
> **我先前把 5 个负贡献当作"框架只收集确认"的机制证据,其中 2 个是父代认定错误造成的假负值;
> 方向仍成立(21% 仍不小),但强度被我夸大了。**

**内部一致性检验(不是巧合)**:3 个 identity 案例的固定 knob 延迟差是
**−0.12% / +0.28% / +0.02%,全部落在 ±0.3% 内**;25 个 moved 案例跨
**−92.89% … +29.66%**。
零资源变化 ⇄ 零延迟变化,两个独立读数互相印证 ⇒ 这 3 个确实什么都没做,
不是"匹配逻辑漏了什么"。

**选择偏差已排除(这是我担心的那条替代解释)**:7 个"取值从未被测"若系统性偏向
"footprint 改得最狠"的改写(父代 tile 在子代空间里跑不起来),则空操作率会被**高估**。
实测 `scripts/probes/undecidable_rewrite_bias.py`(3 项自检):
**全部是 TPE 采样巧合,0 个是被拒**(修父代前是 16/16,修后基数变小,结论未变)。
⇒ 该轴上无偏。**注意轴的范围**:排除的是"被拒"这一条路径,不是所有可能的偏差来源。

### 顺带的第二个发现:被接受的改写里有 3 个源码贡献为负

25 个 moved 案例里 **3 个的固定 knob 延迟差是负的**(−14.34%、−26.07%、−92.89%)
—— 也就是说在父代最优点上,新源码**更慢**。这些改写**全部被框架接受**,
因为接受判据是族最优的改善,而族最优由 TPE 在新空间另找的点提供。

⇒ 这给 [[the-framework-only-collects-confirmations]] 一个**机制级**证据,不只是统计观察:
改写"几乎从不变差"是因为**衡量它的量里混着调参**。

### 为什么这比"声明幅度不准"严重一档

已知的声明准确率问题([[declaration-accuracy-is-per-dimension-not-global]]:`shared_bytes` 36%/20%)
是**幅度**问题,符号大体还在。这 3 例连符号都没有 —— 是**空操作**。
而延迟确实改善了,于是框架把它记成**一次成功的改写**,并在族的 `best_history` 上前进一格。

### 读法上的连带教训(我自己差一步就写错)

先按 `rewrite_vs_promise.py` 的"两侧各自最优点"读,得到的是
`n_regs 210→254`、`occupancy 0.1667→0.0833`(减半)、`occ_limiter registers→shared_memory`
—— 看起来是"改写把占用率弄差了",**方向与恒等这一真相相反**。
那三个数全部是**换 tile** 的后果。
⇒ **任何要归因给源码的资源方向,必须在固定 knob 的匹配点上读。**
匹配点不存在时,正确答案是**拒绝出数**,不是退回自有最优点比较。

**第二个连带教训**:父代不在 payload 里,这一个事实制造了**两个**独立缺陷 ——
需要匹配点(否则混入 tile 效应),以及需要时间序(否则拿后代当父代)。
下一轮若在 `REWRITE_PRODUCED` 里加上 `parent_candidate_id`,两个都不必再靠推断。

### 必须承认的口径限制

**"空操作"的准确说法是"在 C1 对齐的那 6 个维度上是空操作"**,不是"什么都没改"。
改指令混合、访存顺序、循环结构而 footprint 不变的编辑,在这里读作 identity 而**并非**普遍意义的无效
—— 但那恰恰是 C1 的问题:**C1 声称对齐的维度看不见这类改动**。
另外匹配点是父代最优点这**一个**点,源码改动可能在别处动了维度而在这里没动;这是固定 knob 的代价。

### 下一轮该做什么

1. **把匹配点复测变成常规产物**,而不是事后探针。
   **代码位置已核实**(`v3` @ `55387c5`):改写走的是 `orchestrator.py:947-965` 的
   `else` 分支,`anchors = tuple(w.params for w in accepted.witnesses)`
   —— 锚点是**两个见证配置**(default 与 minimal),**父代最优点不在其中**。
   K 扩展走的是另一条路(`:2306-2335`),那里**已经**把 `prior_best.params` 前置进 anchors
   并复用其测量、零机时。⇒ 要做的是把同样两行搬到改写分支,
   **不是**"改写已经有了这个机制"。
   **一个必须连带的细节**:`:943-944` 的既有空间分支把锚点截成 `[:2]`。
   若在改写分支加父代最优点而不检查这个上限,会**静默挤掉一个见证**
   —— 见证是空间发布门的证据,丢掉它是在替换一个安全检查,不是加一个锚点。
   今天 28/35 个已关闭改写碰巧有匹配点,靠的是 TPE 偶然采到;
   入队 anchor 让它变成 **35/35**,代价是**一个 trial**,而且该 trial 本身是有用的采样。
2. **在这个匹配点上比对声明**:`REWRITE_PRODUCED` 声明的维度与方向 vs 实测 Δ。
   这给 C1 一个**逐改写**的准确率,而不是今天的逐候选声明准确率。
3. **正对照是必须的**,已内建:必须有改写在匹配点上**确实**移动了被点名的维度(实测 13 个),
   否则"全是空操作"与"匹配逻辑坏了"打印同一个结果 —— 本轮就先坏了一次
   (`median_ms` 拼错 ⇒ 每个 trial 读成 `None` ⇒ 打印"无同族可比",一句关于 run 的假事实)。

**样本量**:28 个可判定改写,来自 6 个 run、两块 4090 与继承自 box2 的记录。
比 n=1 结实,但仍是**单任务(L3:43)**,引用时须说明。
**同时须说明这些数字已因父代认定修正而变过一次**(24%→11%):
在 `REWRITE_PRODUCED` 里加 `parent_candidate_id` 之前,任何逐改写的比例都依赖推断出的父代。

---

## 4.5 【本轮查清的根因,建议列入下一轮前列】归因失败不是标签问题,是**单 knob 消融的射程**问题

**这是 C2 零投递最紧的一条成因,而且可修。**

### 我此前的解读是错的

我一直把 `over_ratio < 1` 读成"θ\* 处没触顶 ⇒ 那次拒绝另有原因(不是共享内存)"。
实测否证。工具 `scripts/probes/refusal_vs_theta_star.py`,读 s7-control 的 39 个被拒配置:

| | |
|---|---|
| 39 个拒绝 | **全部真实**,超限 **1.13x – 2.26x** |
| 消息 | 明确点名 kernel 与字节:`_flash_attn requires 229376 bytes ... limit of 101376` |
| over_ratio 0.485 的 `BLOCK_N` 墙 | θ\* 用 **32768** / 101376(=0.485),被拒点要 **229376(2.26x)** |
| 被拒点与 θ\* 的 knob 距离 | 该点差 **10 个**;另两个差 **8 个**、**3 个** |
| 被墙点名的 knob 确实不同 | **21 / 39**,但同时另有 knob 也不同 |

⇒ **拒绝是真的,标签是对的,墙是真的。** 归因判 `not_attributed` 在它自己的判据下也没错 ——
它做的是**从 θ\* 出发、一次动一个 knob** 的消融,而被拒点在多个 knob 上同时偏离,
每个 knob 单独都装得下。**缺口在方法的射程。**

### 可做的两种改法

1. **扩大起点**:归因起点取更多高性能点,而不只是 θ\*。
   `probe_top_k` 已朝这个方向,但它取"最优的前 K 个",**仍全在 θ\* 一侧**。
2. **多 knob 联合消融**:把被拒点**向 θ\* 逐维回退**,求使其装得下的**最小维度集合**。
   那个集合正是该投给改写的内容,而且比单 knob 更有信息量(它说的是"这几维一起顶住了")。

### 两种改法有明确优先级:**幅度不是障碍,位置才是**(604 个单 knob 配对实测)

工具 `scripts/probes/single_knob_reach.py`(7 项自检),读 5 个 run 里**恰好差一个 knob** 的配置对,
量其 `shared_bytes` 比值 —— 这是**局部实测**,不拟合任何曲面
([[resource-map-is-not-separable]] 已三次否证可分离的全局模型):

| | |
|---|---|
| 单 knob 的 `shared_bytes` 比值 | 中位 **1.25x**、p90 **2.01x**、最大 **4.50x** |
| 幅度最大的 knob | `BLOCK_N` / `BLOCK_M` 各 4.50x、`NUM_STAGES` 4.08x、`COMPUTE_DTYPE` 4.00x |

被拒配置超限 1.13–2.26x,而单 knob 的 p90 已到 2.01x、最大 4.50x
⇒ **需要的倍数在单个 knob 的能力范围内**(中位 1.25x 不够,所以不是"典型情况能做到")。

**⇒ 所以单 knob 归因失败不是因为力气不够,而是因为出发点不对:**
θ\* 是一个**跑通了的**配置,必然在上限**以下**,它附近的已测点也都在以下。
**改法 1(扩大 θ\* 一侧的起点)不管取多少个点,每个都还在限下** ——
只有**从被拒点往回走**(改法 2)才会真正横跨上限。

⇒ **改法 2 优先,改法 1 无法单独解决这个问题。**

**一处我必须收回的读数**:探针最初报"0 / 603 个配对有单 knob 跨过上限(0%)",我据此写下
"幅度够、位置不够"。这个 0 **是被数据布局强制的,不是测量结果**:
649 个观测在限上、2430 在限下,但**没有任何 (group, kernel) 组同时含两侧** ——
因为限下的观测来自 `TRIAL_DONE`(配置跑通了),限上的来自
`CONFIG_SCREENED_INFEASIBLE`(没跑通),而两者描述**不同的 buffer**,分组键本就把它们分开。
⇒ 那是一个**结构上不可能非零**的计数器,与
[[an-unreachable-branch-is-not-a-safeguard]] 同型。探针现在**拒绝打印**该数字并说明为何不可观测。
结论(改法 2 优先)不依赖那个 0,而依赖上表的幅度分布与"θ\* 必在限下"这一事实。

### 代价已量完(零机时,`scripts/probes/price_multiknob_ablation.py`)

读 4 个 run(S7 两臂 + arm2 + arm3)的 **388 个被拒配置**:

| | |
|---|---|
| 被拒点与 θ\* 的 knob 距离 | 最小 1、**中位 5**、最大 15 |
| 贪心最坏情形合计 | **9717** 次评估(实验仍在跑,该数随被拒配置增加而漂移;本表读于 6.0 h) |
| 一级回退在记录里已测过 | **19 / 2326(1%)** ⇒ 几乎全是新点,必须现算 |

**要做的评估不是"编译+计时",而是 compile-only screen** —— 它本来就报
`max_shared` 与 `limit`(`CONFIG_SCREENED_INFEASIBLE` 的键正是
`candidate_id, kernel, limit, max_shared, params`),不需要完整 cubin。
其代价 `correctness.py:170-191` 已实测:**48 个配置一进程共 11.02 s ⇒ 边际 7 ms/个,
但每批约 11 s 的进程启动是固定成本**(单独跑一个是中位 16.7 s,几乎全是 torch/CUDA/KernelBench 导入)。

**⇒ 决定代价的是批数,不是评估次数**(逐维回退每一层是一批,因为下一层的点依赖本层答案):

| 设计 | 评估次数 | 批数 | 代价 |
|---|---|---|---|
| 仅一级回退,一批 | 2350 | 1 | **27 秒** |
| 贪心,**每层一批(所有墙合并)** | 9717 | 5 | **123 秒** |
| 贪心,每个被拒配置一批 | 9717 | 391 | **1.21 小时** |
| 完全不批量 | 9717 | — | 45.1 小时 |

**实现上的分水岭是批的粒度**:每层一批 123 秒,每墙一批 1.21 小时,**差 36 倍,而两者答案相同**
⇒ 功能测试发现不了。**必须有测试钉住"每层一批跨所有墙合并提交"。**

**同一处实测还给了一条上界**:共享内存相关子网格可达 600,000 点 = 7 ms 下 70 分钟
⇒ **穷举筛查不可行**,而逐维回退是**有向**搜索(访问 d、d−1、…,不是网格),这正是它的价值所在。

**两笔尚未计入、提交前必须量的代价**:
(a) screen 按 materialized source 缓存,且**故意不缓存失败**(`correctness.py:240-244`)
⇒ 批次超时的代价是一次重探,不是永久盲区;
(b) **墙钟始终是约束预算**(8/8 跑完的 run 都由它结束)⇒ 即使每道墙一分钟也在和 trial 抢时间,
**须按 run 而不是按墙计价**。

**⚠ 本节的数字我改过两次,两次都是我引错了价**:
先写"`compile_s` 达数十秒 ⇒ 编译密集",实测是中位 **0.42 s**(1821 个 trial,p90 0.45 s、最大 1.53 s)
—— 数十秒和 ptxas 吃 111 GiB 来自某个 agent 自己写的病态 PTX
([[agent-script-can-oom-the-whole-box]]),不是正常编译;
再写"批量 67 秒",那是**只算了 7 ms 边际、漏掉每批 11 s 固定成本**,
且掩盖了真正的设计选择(批的粒度)。**引用本节的数字前请看清是哪一种批法。**

**必须连带的正对照**:任何放宽都要能证明它**不会把所有拒绝都变成"已归因"** ——
即必须存在仍被判 `not_attributed` 的拒绝(例如真的由非资源原因失败的)。
否则"提高覆盖率"退化成"把门拆掉"。

### 与已有条目的关系(两个独立缺口,不要混)

- 本节:**墙进来了,但归因不到**(射程不够)。
- [[a-shared-memory-refusal-can-arrive-labelled-runtime-error]]:**墙根本进不来**
  (`_classify_exception` 把 launch 期 OutOfResources 标成 `runtime_error`,约 2% 的真墙对
  `find_walls` 不可见)。

### ⚠ 两臂的零投递成因**不对称**,修本节不会让处理臂产出墙(2026-09-14,6.4 h 时点)

| | 共享内存拒绝 | `walls_found` | 成因 |
|---|---|---|---|
| **处理臂** | 35 | **0** | **每个被拒值都在该 knob 的已测范围之内** ⇒ 按定义不构成墙 |
| **控制臂** | 68 | 4(2 道过斜率) | 有墙,但**全部卡在归因**(over_ratio 0.485 / 0.970) |

⇒ **处理臂的 0 投递不是 §4.5 造成的** —— 它连一道候选墙都没有,缺口在**更上游**:
墙的判据是"被拒值落在已测范围**外**",而同一取值换搭档跑通后墙按定义消失
([[a-wall-can-be-erased-by-a-later-successful-trial]],实测 `NUM_WARPS=16` 于 trial 6 被拒、
trial 14 跑通 ⇒ 墙数随 trial **单调减少**)。

**这条对下一轮的直接含义**:
1. **只修 §4.5 不够**。即使归因射程放宽,处理臂这类 run 仍会是 0 投递,
   因为它的输入集(`find_walls` 的候选墙)本身是空的。
2. **⇒ 需要一并考虑"墙的判据"**:是否要把"被拒值在已测范围内"也算作墙的一种
   (即"这个取值在**某些搭档下**不可行")。**这是改判据,风险等级与放宽归因相同,
   必须带正对照** —— 否则"墙"会退化为"任何一次拒绝"。
3. **报告 S7 结果时必须逐臂说明成因**,不能合并成"S7 零投递因为归因失败"。

### 一个探针教训,记在这里因为它差点污染本节

我的第一版正则要求 `Required:` 与 `limit:` 同现,而真实消息是
`<kernel> requires N bytes ... limit of M` ⇒ **0/39 匹配**。
探针据此打印"没有一个携带 Required/limit 数字",而我差点把它当作
**§4.1(a)"应把这两个数字写进 `failure_detail`"的证据** —— 字段一直都在,是我的模式写错了。
⇒ **声称某字段缺失之前,先验证解析器。** 修正后 **39/39** 全部解析成功。
这也意味着 §4.1(a) 的措辞需要收窄:要补的不是"把数字写进 detail",
而是**把它们写成结构化字段**(今天要靠正则从自由文本里挖)。

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
4. **§4.5 归因射程(本轮新增,**代价已量完**)** —— **这是 C2 零投递的根因**,
   而且比上面第 3 项更靠前:第 3 项问"覆盖率能不能扩大",§4.5 说明**为什么当前几乎全部归因失败**
   (被拒点与 θ\* 差 3–10 个 knob,单 knob 消融装不下)。
   **代价不再是未知**:391 个被拒配置、贪心最坏 9717 次评估,走 compile-only screen
   **每层一批合并所有墙 = 123 秒**;每墙一批则 1.21 h,完全不批量 45.1 h。
   ⇒ 可做,但**分水岭是批的粒度**(差 36 倍且答案相同 ⇒ 功能测试查不出),必须有测试钉住。
   仍须带"仍有拒绝被判 not_attributed"的正对照,否则放宽退化成拆门。
5. **§2** —— `n_spills` 墙(需要第二种判据函数)。
   **先做 §2c 学到的那一步**:在真实候选上测 `n_spills` 与已有量(`n_regs`、tile 乘积、延迟)的
   相关性。`n_spills` 有先验优势(八维里唯一有稳定转化率),但 §2c 刚证明"先验优势"不能替代实测。
6. **§3.2 S7** —— 需批准,一对 12h。
   **注意本轮结果对它的含义**:S7 的投递路径依赖"有墙被归因",而本对 29 次触发 0 次投递,
   根因是 §4.5 而非 S7 自身 ⇒ **先修 §4.5 再重跑 S7,否则重跑仍会是 0 投递。**
7. **§4.1 / §4.2 / §4.3** —— 不动(§4.1 的 (a) 措辞已按 §4.5 收窄)。

**一条本轮新增的、适用于每一项的门槛:**
**任何"新维度 / 新信号"在实现前,必须先测它与已有量的 Spearman,|ρ| ≥ 0.9 即不得实现。**
这是三次同型事故换来的:`pct_of_dram_peak` 对 1/latency ρ=+1.000、
SOL 头寸对 1/latency ρ=+1.000、`logical_bytes` 对 tile 乘积 ρ=+0.978。
三次都是**先想清了机制、跳过了相关性**,而机制听起来都成立。

**无论走哪条,报告都必须保留当前的负结果**:"2e 只进 prompt 时无端到端收益"是已测事实
(同箱同 mode 的干净对比里关掉 2e 的臂快 8.0%,噪声底的 2.0 倍),改进版若成功,它就是对照。

## §4.6 `scripts/screen_cost.py` keys on three event names that are never emitted

`scripts/check_event_names.py` (2026-09-14) reports three names in `screen_cost.py` that `src/`
never emits, **each the SOLE key for its branch** — so those branches have never run:

| name used | nearest real name |
|---|---|
| `COMPILE_PROBE_FAILED` | `CONVERSION_RATES_FAILED` |
| `SCREEN_FAILED` | `PRESCREEN_FAILED` / `SLOPE_GUIDE_FAILED` |
| `WORKER_TIMEOUT` | (no close match) |

**Certain but not urgent** (a reporting script, not the harness), so recorded rather than fixed
mid-experiment. Fixing it needs the emitting side read first — the nearest-name suggestion is a
string-distance guess, and picking the wrong one would swap a silent zero for a plausible wrong
number, which is worse. Same shape as the reader bugs the checker was written for.
