# 代码修改清单:剩余项的具体情况、修法与风险

**状态更新 2026-09-06(晚)**:P5/P6/P3/P4 已实施并验证;P0 的三个前置缺陷 D1/D2/D3 已修、
预算已抬(`max_families_total` 3→6、`max_families_active` 2→3);**但 D2 的修复引入了 D4
(外环不终止),同日发现并修复** —— 详见 `finding-loop-d-preflight-defects.md` 的 D4 节。
顺带查出一个更普遍的问题:19 个真实 run 里只有 **1 个**是被墙钟结束的,详见
`finding-runs-rarely-reach-their-budget.md`;报告现在会直接写出"run 为什么结束"以及
"改写轮用了几轮 / 共几轮",这个可见性缺失正是 D2 藏了 19 个 run 的原因。

**前置更正**:上一份评估(`assessment-framework-defects-priority.md`)把三个**已经修好**的项当成待修列出了,
这是我核实不足。逐条核实后的真实状态:

| 项 | 我先前说 | 实际 | 证据 |
|---|---|---|---|
| **P0** Loop D | 待修 | **已开启(2026-09-06)** | 抬 `max_families_total` 3→6 后互锁解除;修 D1/D2/D3 + D4 |
| **P1** 见证门精度误拒 | 待做实验 | **已解决** | fp64 门那个 run 剩余 2 次拒绝均为 `runtime_error`(真 bug),精度误判 0 |
| **P2** minimal 见证 fp16 角 | 待修 | **已修** | `validation.py:35 _looks_out_of_range` + `_next_witness` 回退,且比我的提案更精细 |
| quantile 崩溃 | 待修 | **已修** | `worker_main.py:569-570` 已改为采样 + `.sort()`,并包在 try/except 里 |

**你对 P1 的判断是对的**;**P0 的判断需要更正** —— 轮次上限(3→5)提升的是 loop C(在已有族里挖更深),
Loop D 是注入结构上不同的新族,是另一个门(2026-09-06 已打开)。

---

## 需要改代码的,只剩 4 项

按建议顺序:**P5 → P6 → P4 → P3**。P0 是改配置数字(不算代码),但有一个必须先做的代码前置检查。

---

## P5 — 无 `tl.dot` 的 kernel 被误标 `precision: unknown`

### 情况

`orchestrator.py:67 _detect_candidate_precision` 的判定链末端:

```python
    if "tl.dot" in text:
        return "tf32"          # 有点积但没写精度 → Triton 在此代默认走 tf32
    return "unknown"           # ← 没有点积就到这里
```

判定顺序是:先读 PARAMS 里的精度 knob → 再找源码里的 `input_precision` / `float16` / `bfloat16`
→ 最后看有没有 `tl.dot`。**一个完全没有矩阵乘的 kernel 会穿过所有分支落到 `unknown`。**

### 举例

`run-l3-48-20260905-010737` 的获胜 kernel 是 Mamba 顺序扫描,核心是
`state = state * e + bv * xv` 加 `tl.sum` —— 纯逐元素与归约。grep 全部精度信号:

```
tl.dot 0   input_precision 0   float16 0   bfloat16 0   .half( 0   tf32 0
float32 1
```

于是报告写 `precision: unknown`。后果:`_honest_verdict` 拿 `unknown` 去选对比基线,
默认落到 `torch_compile`(fp32)—— **这次恰好是对的,但纯属运气**,代码并不知道原因。
如果哪天默认值改成 tf32 基线,一个 fp32 kernel 就会被拿去和 tf32 基线比,凭空损失约 1.04–1.9× 的加速比。

影响范围:所有 scan / 归约 / 逐点融合 / 转置类 kernel —— 即所有不含矩阵乘的算子。

### 怎么修

在 `tl.dot` 分支之后、`return "unknown"` 之前插一个分支:

```python
    if "tl.dot" in text:
        return "tf32"
    # 没有点积 => 不走张量核 => 算术精度就是存储精度。上面已排除所有低精度构造
    # (float16/bfloat16/.half()/input_precision),所以到这里只能是 fp32。
    # 关系到 _honest_verdict 选哪个基线:一个 fp32 kernel 必须对 fp32 基线,
    # 而 "unknown" 只是碰巧落到同一个默认值。
    if "triton" in text or "tl." in text:
        return "ieee_fp32"
    return "unknown"      # 非 Triton 后端(cuda load_inline)仍然无法判定
```

保留 `unknown` 给 CUDA 后端是必要的 —— 那里没有 `tl.` 记号可读,硬猜会错。

### 风险

**极低。**
- 只影响报告里选哪个对比基线,**不影响任何运行、判决、接受/拒绝**
- **已实测确认结论不变**:`_honest_verdict:117` 的分支是
  `is_tensor_core = precision in ("tf32","fp16","bf16")`,`unknown` 和 `ieee_fp32`
  **都落在非张量核侧**,选同一个基线。实跑验证:

```
unknown    -> compared_against=torch_compile       speedup=9.49
ieee_fp32  -> compared_against=torch_compile       speedup=9.49    ← 完全相同
tf32       -> compared_against=torch_compile_tf32  speedup=9.13
fp16       -> compared_against=torch_compile_tf32  speedup=9.13
```

所以这一项**当前数据的任何数字都不会变**,改的是"为什么正确" ——
把一个碰对的默认值变成一个有依据的判定。价值在于未来:如果哪天默认分支改了,
或者要在论文里声明精度归类方法,`unknown` 是站不住的。

### 怎么验证

- 单测:三个输入(纯扫描 Triton 源 / 含 `tl.dot` 的源 / CUDA 源)分别得到 `ieee_fp32` / `tf32` / `unknown`
- 回放:对 L3:48 那个获胜 kernel 跑一次,确认从 `unknown` 变成 `ieee_fp32`,且 `_honest_verdict`
  选中的基线**仍是** `torch_compile`(证明结论不变)

---

## P6 — 报告把跨精度加速比印在诚实判据之前

### 情况

`reporting/report.py:234` 先写四个原始加速比,`:252` 才写同精度判据。读者先看到的是最大的数字。

三个任务的参考实现**全是纯 fp32**(已读源码确认:无 tf32 标志、无 autocast、无 dtype 参数),
而获胜候选都用更低精度,所以四个原始比里有三个是跨精度的:

```
任务     vs torch_compile   vs c_tf32(诚实)   仅换基线就虚高
L3:21          3.09×              2.27×             1.36×
L3:43          4.23×              2.21×             1.91×
L3:48          9.49×              9.13×             1.04×
```

**L3:43 上光换基线就值 1.91×**,几乎等于全部诚实加速。

### 举例

机制本身是好的,而且在承重 —— 三个历史 run 被它判为 `FAILS`:在自己精度下**比基线更慢**,
却对 fp32 基线显示 1.08–1.86× 的"胜利":

```
run-l3-21-20260904-013056  fp16  vs_compile 1.08×  vs_compile_tf32 0.79×  FAILS
run-l3-43-20260903-145357  tf32  vs_compile 1.72×  vs_compile_tf32 0.87×  FAILS
run-l3-43-20260904-093730  fp16  vs_compile 1.86×  vs_compile_tf32 0.95×  FAILS
```

问题纯粹是呈现顺序:如果有人从报告顶部摘数字写论文,摘到的是 4.23× 而不是 2.21×。

### 怎么修

`reporting/report.py` 里交换两块的顺序,并给原始那组加标注:

```
- **honest same-precision verdict**: ...        ← 移到前面
- speedup vs each baseline (cross-precision, NOT directly comparable):
    - vs `eager`: ...
```

### 风险

**极低。**纯字符串顺序,不动任何计算。唯一注意点:`kernel-opt report --run <id>` 是从
events.jsonl 纯重放再生的,所以旧 run 重新生成报告后顺序也会变 —— 这是想要的效果,不是问题。

---

## P4 — 空间扩展朝已经在失败的边界瞄准

### 情况

`orchestrator.py:293 _requests()` 选扩展方向的过滤链:

```python
{"name": ps.name, "direction": ps.boundary_direction}
for ps in stats.param_stats
if ps.at_boundary and ps.boundary_direction in ("min", "max")
and _is_numeric_knob(ps.name)
and (ps.effect_pct or 0.0) >= min_effect_pct
and not _at_hard_edge(ps.name, ps.boundary_direction)
```

**没有任何一项看失败率**,而 `failure_rate_by_value` 早就在 `TuningStats` 里算好了
(`stats.py:146` 写入,`reports.py:33` 定义),**目前全代码库无人读取它**。

### 举例

`run-l3-43-20260906-091019` 的 24 个被拓宽 knob:

```
10 个(42%)推向了失败率 >=25% 的边界
  新增值在失败边界之后 : 16/37 = 43% 失败
  新增值在健康边界之后 : 13/84 = 15% 失败
```

近 3 倍差距,而失败 trial 不返回延迟数据 —— 纯亏。

最典型的 `cand-60fdcae9`:八个 knob 里七个推向失败率 24–40% 的边界,整次扩展瞄进了父空间
已经证明不友好的区域。

**反例(机制正常工作)**:`cand-9c8d066a` 的 `ATTN_BLOCK_M` 加了 256,采样 2 次得 166.0ms
(差 8.7 倍),TPE 正确留在 128 —— 花 2/40 个 trial 买到一个真实阴性结论。差别在于那个边界是健康的。

全部 67 次扩展的产出分布,说明为什么值得管:

```
完全无变化(|增益|<0.5%)      26  (39%)
边际(0.5–2%)                  19  (28%)
改善且获胜值是新增的           12  (18%)   ← 机制真正生效
改善但只靠重调原有域           10  (15%)
```

**只有 18% 通过它存在的理由获益**,67% 没有有意义变化,每次 40 个 trial ≈ 全项目约 9.5 小时 GPU。

### 怎么修

**我最初想的修法(在过滤链里加一个 veto 条件)是错的 —— 已实测否决。**
`scripts/audit_expansion_failure_veto.py` 在 19 个 run、523 次瞄准上回放了三种设计:

```
A. 硬过滤(我最初的提案)
   压掉全部 155 次失败边界瞄准,但会取消 8 次扩展,其中 3 次历史上是改善的
   —— 而 orchestrator.py:315-331 已经实测过"取消扩展"的代价:扩展给的是
   两样东西(更宽的域 + 一份新的调参预算),返回 [] 把两样都取消了

B. 只 veto winner-anchored 那条臂,让 median 兜底臂救回
   不取消任何扩展,但 median 臂在 8 例中至少 5 例**瞄向完全相同的被 veto knob**
   (NUM_WARPS、BLOCK_D、BLOCK_N/BLOCK_K …)—— veto 恰好在它该起作用的地方被架空

C. 把 veto 当成排序偏好,而不是过滤器   ← 实测最优
   优先选健康边界的 knob;只有当这次扩展里**没有**健康备选时才退回失败边界
     - 避开 131 次失败边界瞄准(它们所在的扩展都还有健康备选,跳过零损失)
     - 8 次扩展仍然瞄向失败边界 = 完全保持今天的行为
   零扩展被取消、零 re-tune 被放弃,拿到 131/155 = 85% 的收益
```

**被 emptied 的 8 例里有两个是当次 run 的冠军候选**(`cand-0d0dcd49`
是 `run-l3-21-20260905-195615` 的最优,`cand-60fdcae9` 是 L3:43 的 8.06ms 冠军)。
方案 A 会取消它们的扩展 —— 这正是你在轮次上限那一题里给的判据
("性能已经很高但还有少量提升空间的族往往产出最强结果")的同型风险,只是发生在扩展层。

所以落地形态是**排序**。具体改动在 `boundary_knobs_to_expand` 内部,
**两条臂共用一个后置步骤**(不动任何现有过滤条件):

```python
def _edge_failure_rate(ps) -> float:
    """要拓宽的那个边界取值的历史失败率;无数据 => 0.0(不干预)。"""
    if space is None or not ps.failure_rate_by_value:
        return 0.0
    try:
        choices = space.domain(ps.name).choices          # 必须从域取,不能从 dict 取
    except KeyError:
        return 0.0
    numeric = [c for c in choices
               if isinstance(c, (int, float)) and not isinstance(c, bool)]
    if not numeric:
        return 0.0
    direction = ps.boundary_direction or _median_direction(ps)
    edge = min(numeric) if direction == "min" else max(numeric)
    return ps.failure_rate_by_value.get(repr(edge), 0.0)  # 键是 repr(choice)

# 在 return 之前,对已经算好的 requests 做一次偏好排序:
def _prefer_healthy(reqs: list[dict]) -> list[dict]:
    by_name = {ps.name: ps for ps in stats.param_stats}
    healthy = [r for r in reqs
               if _edge_failure_rate(by_name[r["name"]]) < max_edge_failure_frac]
    return healthy or reqs        # ← 这一行是全部安全性的来源
```

然后 `requests = _prefer_healthy(_requests(True))`、
`return _prefer_healthy(_requests(False))`。

`healthy or reqs` 是关键:**它保证返回值永远不会因为这个改动从非空变成空**,
所以 `orchestrator.py:926` 的 `if not knobs: return`(会直接取消整次扩展)
永远不会被这个改动触发。

阈值 `max_edge_failure_frac` 加进 `BudgetConfig`,默认 0.30
(实测健康组 15% / 失败组 43%,0.30 落在两者之间)。设为 1.0 即完全关闭。

### 修改后具体会发生什么(实测,非估计)

在 19 个 run / 523 次瞄准上回放:

```
                              今天        改后
每次扩展仍然发生              169+8       169+8      不变(零取消)
瞄向失败边界(>=30%)的次数     155         24        -131
其中"该扩展另有健康备选"       131          0        ← 这 131 次改为先瞄健康 knob
其中"该扩展只有失败边界"        24         24        ← 原样保留,行为不变
```

**具体到一次扩展长什么样** —— `cand-60fdcae9`(L3:43 的 8.06ms 冠军)七个瞄准全部
指向失败率 24–40% 的边界:今天它照原样发出;改后 `healthy` 为空,走 `or reqs`,
**仍然照原样发出**。这个候选完全不受影响。

而 `run-l3-43-20260906-091019` 里另一类扩展(有健康备选的那 131 次)会改变瞄准顺序:
原本推向 43% 失败率的边界,改为先推健康边界(15% 失败率)。按已测的
"失败边界后新增值 43% 失败 vs 健康边界后 15%",这部分预期把扩展带来的废 trial 减少约 2/3。

**代价上限是可以算的**:被改变瞄准的 131 次都属于"另有健康备选"的扩展,
所以损失的最坏情况是"某个失败边界其实是被别的 knob 拖累的假象,而我们先试了另一个" ——
少一次瞄准,不少一次扩展。且因为是排序,`healthy` 用完还会继续用原列表里剩下的。

### 风险

**低。方案 C 的风险显著小于我最初的提案。**

1. **过度拦截 —— 已被方案 C 结构性消除**:排序而非过滤,意味着"拦掉"只发生在
   同一次扩展里存在健康备选时。参数间耦合导致的假高失败率最坏后果是
   "先试了另一个 knob",而不是"这次扩展没了"。
2. **拦成空请求 —— 已被 `healthy or reqs` 结构性消除**,不需要再去确认
   `no_new_choices` 路径是否伤候选(那条路径在 `orchestrator.py:926`
   `if not knobs: return` 确实会直接取消扩展,这也是方案 A 危险的原因)。
3. **剩余的真实风险**:健康/失败的判定依赖单次调参内的失败率,样本可能很薄
   (事件里没有每值 trial 计数,无法在离线回放里核这一项)。缓解:阈值取 0.30
   而非更激进的值;并且因为是排序,判错的代价只是换个瞄准顺序。
4. **一个实现坑**:`failure_rate_by_value` 的键是 `repr(choice)`(`stats.py:124`),
   而**键顺序是 trial 顺序,不是域顺序** —— `_median_direction` 的 docstring 已记下
   同型陷阱:实测某候选 `latency_by_value` 键序是 `['128','64','256','512','1024']`,
   首键不是域最小值。所以边界值必须从 `space.domain(name).choices` 取。
   已有测试 `test_median_fallback_reads_edges_from_the_domain_not_the_latency_dict`
   守着同一条不变量,新代码应加一条同型断言。

**绝对不要做的一件事**:不要把这个信号"升级"成资源模型。我测过 —— 汇总层面
`shared_bytes` 随失败率单调上升,看着像完美预测量:

```
失败率     knob-value 数   regs 中位   shared 中位
0%               974          255       40960
50%+             619          255       86016
```

但 `n_regs` 每桶都是 255(硬上限,零信息),而在**单候选单 knob 内部**(veto 实际需要的粒度)
"shared 最高的取值失败率也更高"只占 **458/935 = 49%,等于抛硬币**。汇总趋势是跨候选聚合的假象。
**只用直接测到的 `failure_rate_by_value`。**

### 怎么验证

- 单测:同一扩展里一个健康边界 + 一个失败边界 → 只返回健康的;**全部失败 → 原样返回全部**
  (这条断言是方案 C 与方案 A 的分界线,必须有)
- **前瞻回放已完成**:`scripts/audit_expansion_failure_veto.py`,19 个 run / 523 次瞄准,
  输出上面那张三方案对比表。改完后重跑,确认 emptied 仍是 8 且都走 `or reqs` 分支

---

## P3 — 报告需要区分"自写核心算子"与"委托给 PyTorch"

### 情况

族抽象允许成员之间结构不一致。`run-l3-43-20260906-091019` 按 `kernel_names` 分类
(H=自写注意力,D=委托 torch SDPA):

```
fam-6eea8eac   best  8.06   5 个候选   D D H D H     ← 血统内部翻转
fam-8fb9b2b8   best  9.43   4 个候选   H H H H
fam-e6706893   best 15.40   3 个候选   D D D
fam-94add40d   best 16.10   5 个候选   H H H H H
```

**最快的族在自己血统内部切换了计算方法**,而获胜的是委托那个。

### 举例

头条数字 8.06ms(复测 8.37ms)来自 `cand-60fdcae9`,`kernel_names` 是
`['_fused_qkv_projection', '_head_layout_projection']` —— **没有注意力 kernel**,
核心是 PyTorch 的 `scaled_dot_product_attention`。我们的贡献是把 `c_attn`+QKV 打包融进
一个 Triton GEMM 并去掉输出转置 —— 真实,但不是注意力 kernel。

纯手写最优是 `cand-9f6af7bd` 的 9.43ms。

**我自己在这上面连错两次**:先说 `fam-e6706893`(16.7ms)是最好的手写结果(错,该族全程委托),
又说 `cand-e29aa508`(16.1ms)是委托型(错,它有 `_flash_causal_attention`)。
**按族归属推断会出错,必须读 `kernel_names`。**

### 是否应该修

**改报告,不要改搜索。**委托 SDPA 对延迟是好棋,搜索找到它是正确行为,不该禁止
(而且"禁止委托"属于你已排除的干预类型)。问题只在报告没区分。

### 怎么修

在 report 里增加两行:按获胜候选的 `kernel_names` 判定,同时给出
**"含委托最优"和"纯自写最优"两个数字**。判定逻辑已在
`scripts/verify_report_against_events.py` 实现并在两个 run 上验证过,搬过去即可。

**关键设计点**:不要硬编码注意力关键词。我脚本里用的
`attention|flash|score|softmax|probab` 是 L3:43 专用的,换任务就失效。
正确的做法是**判断候选启动的 Triton kernel 是否覆盖了参考实现的主要算子** ——
`SEMANTICS_PROBED` 事件已经在探测参考的结构(`training`、`norm_layers`),
可以在那里顺带记录参考的主要算子类型,再和 `kernel_names` 对比。

如果这个通用做法一时做不出来,**退路是只报事实、不下判断**:
在报告里直接列出获胜候选的 `kernel_names`,让读者自己看有没有核心算子。
这个退路零风险且立刻可做。

### 风险

**极低**(只读已有的 profile 元数据,不改判决)。
唯一的风险在"通用判定"那部分:如果关键词或算子对比判错,会给出错误的归属标签 ——
**这比不给标签更糟**。所以建议先落地"列出 `kernel_names`"这个退路,通用判定作为后续。

---

## P0 — 不是代码修改,但有一个必须先做的代码前置检查

> **2026-09-06 更新:前置检查已做,发现 Loop D 路径上还有三个缺陷**,其中一个
> (D2)一旦开启必然触发并提前终止实验。详见
> `docs/finding-loop-d-preflight-defects.md`。摘要:
>
> | # | 问题 | 严重度 | 修法 |
> |---|---|---|---|
> | **D2** | **一次 novelty 失败就冻结所有族并结束整个 run**;而"最后一次 novelty 尝试必然失败"(`accept_novel_seed` 会因 `productive_family_count >= max_families_total` 拒绝),所以**每个开启 D 的 run 都会提前结束** | **高** | 只在确实没有活跃族时才清扫 |
> | **D1** | 外门用 `len(families)`、内门用 `productive_family_count()`,两套规则不一致;improvement E 只改了内门。14 个 run 中 5 个两门判决相反,4 个在 10–21% 预算处结束且有 0–2 个有效族 | 中 | 外门改用同一个函数(1 行) |
> | **D3** | `key = f"novelty:{round_no}"`,而 `round_no` 是局部变量、resume 时归零 → 键碰撞 → 返回 False → 触发 D2 结束 run。`_rewrite_round` 用的是持久化状态,是对的 | 低(仅 resume) | 键改用事件日志派生值 |
>
> **也核实了没问题的**:0.85 相似度门标定良好(108 对刻意不同的种子里只有 2 对超过
> 0.85,即 1.9%),不要放松;novelty 沙箱播种与 prompt 渲染实测正常
> (7 个文件、1350 字符、device.md 正确写入 sm_120);D 成功时可重复触发。

### 情况

`max_seed_candidates: 4` → 4 个族;`max_families_total: 3`;
`_novelty_round` 的门是 `len(families) >= max_families_total` → `4 >= 3` 恒真。
18 个 run 里 `origin:novelty` = 0,`AGENT_CALL_STARTED module=novelty` = 0。

修法是改配置数字(抬 `max_families_total` 到 6,`max_families_total_hard: 6` 已是现成上界)。

### 两个机制问题的确切答案(读代码 + 19 个 run 核实)

**问 1:抬 `max_families_total` 会不会影响初始生成的族数?——不会。**

初始族数由 `max_seed_candidates` 单独决定,`_generate_seeds`(`orchestrator.py:474`)
里两处都只读它:

```python
n_candidates=min(self.cfg.agents.generator.n_candidates,      # :503
                 self.cfg.budgets.max_seed_candidates),
for gen_cand in outcome.output.candidates[: self.cfg.budgets.max_seed_candidates]:   # :508
```

`max_families_total` 在整个 `_generate_seeds` 里不出现,只在 `_novelty_round:1372`
出现一次。19 个 run 全部 `seeds=4, families=4`(种子各自注册一个族),
所以抬 total 到 6 = 初始仍是 4 个族 + 最多再长 2 个新族。

**问 2:是不是要等初始族全部完成调优和改写后才生成新族?——不完全是,比你描述的更早。**

外环是这样的(`orchestrator.py:427-443`):

```python
while True:
    verdict = global_verdict(...)          # 墙钟/全局收敛,freeze 则退出
    if verdict.verdict == "freeze": break
    round_no += 1
    progressed = self._rewrite_round(round_no)     # Loop C
    if not progressed:                             # ← 只有 C 无事可做才进 D
        added = self._novelty_round(round_no)      # Loop D
        if not added:
            冻结所有剩余 active 族
```

分阶段看:
1. `_generate_seeds` → `_pipeline_batch` 把**全部 4 个种子**跑完
   参数化+调优+统计+瓶颈分析(这一段是全部完成的,符合你的描述)
2. 然后进外环。**Loop D 的触发条件不是"所有族改写完",而是
   `_rewrite_round` 返回 `progressed == False`** —— 即这一轮没有任何族真的发起改写。
   而 `_rewrite_round` 只遍历 `active_families()`,后者被 `max_families_active`(=2)截断。

所以准确的表述是:**"当本轮没有任何活跃族能再改写时,才尝试生成新族"**。
一个族在下列任一情况不贡献 `progressed`:已 freeze(收敛或轮次耗尽)、
`best is None`(无正确候选)、或 `report is None`。

这意味着两件对预算判断重要的事:
- 新族**不需要**等到全部 4 个族都用满 5 轮改写。只要某一轮里活跃的那 2 个族都冻结了,
  D 就会被尝试。所以新族有机会在 run 中段出现,不是必须在最后。
- 反过来,`max_families_active=2` 意味着**同时只有 2 个族在改写**,新增的 2 个族会
  排队竞争这 2 个槽位。`active_families()` 的排序规则是
  "未证明的族优先(`rewrite_rounds_used == 0`)→ 按上一轮改善斜率 → 绝对延迟兜底"
  (`families.py:246-251`),所以**新族一旦注册就会因为 `rounds_used == 0` 排到最前**,
  优先拿到改写槽位。这是设计意图(不做早期剪枝),但也是墙钟增加的主要来源:
  新族会挤占老族的改写轮次,而老族此时往往正是延迟最低的那些。

**由此得到一个具体的预算判断**:如果目标是"给新族机会但不牺牲领跑族",
`max_families_active` 可能需要同时从 2 抬到 3;否则新族的优先插队会直接减少
领跑族的改写轮数 —— 这恰好是你在轮次上限那题里想避免的情形。
这是一个需要你决定的取舍,我没有实测数据支持任何一边(Loop D 从未跑过)。

### 但必须先做的代码检查

**Loop D 从未执行过,意味着 `NoveltyGeneratorAgent` 的整条路径从未在任何真实任务上跑过** ——
prompt 渲染、JSON schema 校验、沙箱输入(各族 anchor + summary)、`accept_novel_seed`
的签名去重与 0.85 相似度门,全部未经执行验证。

**所以开启前应先跑 `configs/smoke_l1_novelty.yaml`**(该配置已存在,`rewrite_rounds_per_family: 0`,
显然就是为此准备的),确认这条路径能走通。否则可能在 L3 跑到第 4 小时才发现它抛异常 ——
而按 `AgentModule` 的重试逻辑,novelty 失败会消耗 agent 预算并记 `AGENT_CALL_FAILED`,
不会杀死整个 run,但那一轮的新族机会就白费了。

### 风险

- **配置改动本身低风险**,但会改变搜索投入分配。**注意:我先前说"会明显增加实验时间"是错的,
  已实测更正** —— 改写轮是串行的(`_rewrite_round` 逐族遍历,每次 `_do_rewrite` 阻塞 GPU),
  总轮数上限 = `max_seed_candidates × rewrite_rounds_per_family` = 20,与 active 无关。
  按 38.8 分钟中位,12h 只够约 18 轮,**两种设置下墙钟都是约束**
  (`run-l3-21-20260905-195615` 在 active=2 时就已经跑到 12.82h 超时)。
  active=3 改变的是**分配**:更多族拿到第一轮,每族拿到的轮数更少。
- **未验证代码路径的风险中等** → 已做前置检查,查出 D1/D2/D3(见上表)。
  **D2 必须在开启 `max_families_total` 之前修**,否则每个 run 都会在最后一个新族被接受后
  立刻结束,剩余数小时预算全部浪费 —— 与已记录的
  `docs/finding-run-stops-with-budget-unused.md` 是同一类提前停止。

---

## 建议顺序与理由

```
第一批(改代码,低风险,不动搜索语义)
  P5  precision 分支         ~10 行 + 1 单测   已实测:当前数字零变化
  P6  报告顺序               ~5 行             行号已核实 report.py:234 / :252
  P3  报告列 kernel_names    ~10 行(先走退路,不做通用判定)
第二批(改代码,低风险,动方向选择)
  P4  失败率排序(方案 C)    ~15 行 + 单测     前瞻回放已完成,方案 A/B 已被否决
第三批(改配置 + 前置冒烟)
  P0  L1 novelty 冒烟 → 确认路径可用 → 再抬 max_families_total
```

第一批三项都只碰报告,合计约 25 行,且 P6/P3 直接决定论文数字会不会被误读 —— 应该最先做。
P4 单独一批,因为它是唯一会改变搜索行为的;它的前瞻回放**已经跑完并推翻了我最初的修法**
(见 P4 章),落地形态是排序而非过滤。P0 放最后,因为它需要一次冒烟实验 + 一个预算决策。

**这次核实推翻的两件事**(记录下来以免重复):
1. **P4 不能做成过滤器** —— 会取消 8 次扩展,含两个 run 冠军候选。只能做排序。
2. **"只 veto 一条臂"也不行** —— median 兜底臂瞄向同一批被 veto 的 knob,veto 被架空。

---

## 一个必须声明的覆盖缺口

**GLM 臂零实验数据。**parameterizer / analyst / rewriter / novelty / repair 五个模块
**从未在 glm-5.3 上运行过**,只有 generator 被验证过一次。以上全部分析只在 gpt-5.6-sol 上成立。
