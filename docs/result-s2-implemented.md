# S2 实施完成:按维度独立的多份判决 + 消化层 + G28

**日期** 2026-09-11 · **commit** `867c228` · **本机** 33 passed / 6 skipped(Windows 无 optuna,设计如此)· **反向验证** 14 个错误实现中 12 个在本机被判别,2 个待 A800

---

## 一、修的到底是什么

**不是「标签选错了」,是「必须选一个」这个要求本身。** `classify()` 返回 `kind: str` 一个字符串,9 个 return 点各给一个标签。实测代价:

| run | 同标签比例 |
|---|---|
| run-l3-21-20260908-232211 | **19/20** = `resource_limited` |
| run-l3-48-20260909-115701 | 12/17 = `memory_bound` |

**而分量早就算出来了** —— `evidence` 里已有 occupancy + limiter、n_regs、n_spills、shared 用量、peak_alloc、aten 流量、threads_launched。**向量存在,被 if-else 链塌缩了。**

## 二、三个新增/改动的文件

### `evaluation/dimensions.py`

每维一条 `DimensionRecord{dimension_id, measured, ceiling, ceiling_provenance, verdict, confidence, applicable, not_applicable_reason, higher_is_better, unit}`。

**刻意做成 evidence 的读者,不是第二条测量路径。** 两个理由都来自经验:并行路径会与标签用的那条漂移,让对照 run 无法解读;而这里每个量都已经被验证过在候选间会变(peak 9.6–41.3%、aten 15–82%),重新推导可能静默重新引入任务级常数。

**极性来自 `conversion._DIMENSIONS`,不重述。** occupancy `lower_is_better: False` ⇒ 低才绑定,用**独立的一对阈值**(0.30/0.50),因为「占用率天花板的 90%」读起来意思正相反。

| 维度 | 天花板来源 | 本次实测判决(L3:48 那个候选) |
|---|---|---|
| occupancy | 已是分数,无独立天花板 | **binding**(0.167 < 0.30) |
| n_regs | 本机 device query 的每线程上限 | near-binding(218/255 = 85.5%) |
| n_spills | **天花板恒为 0**,任何 spill 即绑定 | slack(0 个) |
| shared_bytes | 本机 opt-in 每块上限 | slack(0.42) |
| peak_alloc_bytes | 本机 VRAM 总量 | slack |
| candidate_aten_bytes | **无天花板**(是下界) | unknown + confidence 0.5 |
| candidate_aten_ops | 同上 | unknown + confidence 0.5 |
| threads_launched | **无极性** | not-applicable(但 `applicable=True`,见 §四) |

### `evaluation/digest.py`

四元组 `{dimension, severity, root_cause, ranked_recommendation, expected_delta}` + `assert_no_raw_vector()`。

**不是压缩器。** 循环写在 `state.records` 上,所以维度数就是输入数,砍掉一维必须改这个函数的形状 —— **D-9 由结构保证,不靠意图**。方案第一版把它叫「压缩器」,那个读法会在新形状里复刻 v2 自己的缺陷。

`expected_delta` **恒为 `unknown`**。三条实测说明变化率事先不可导出:shared 无闭式(96 点中 0 个准)、代价图不可分离(10 个单步 0 个一致)、**连符号都不可靠**(13 条非单调切片)。给数字就是编造。

`ranked_recommendation` **带上排序依据**:没有「先动哪一维」的规则打得过对照(最好 7.1% vs 21.4%),所以顺序只按 severity + confidence,并且在文本里说明它不是实测优先级。

### `evaluation/dimensions.unreachable_ceilings()` —— J2-4

**只说 evidence 里实测到的三件事:**

1. **后端可达性(G10)** —— Triton 在 fp32 只到屋顶的 84.1%(45.61 vs cuBLAS 54.20,四个精度里差距最大)。「你已到屋顶 84%」和「84% 就是你的后端上限」要的是**相反的动作**,没有这个数就分不开。
2. **DRAM 屋顶不适用(G26)** —— 工作集在 L2 内。
3. **哪个算力屋顶在生效** —— 指令流里没有张量核操作时,百分比是对 fp32 算的,张量核屋顶不是分母。

**刻意不覆盖的一件事**:「这个**任务**根本用不了张量核」。L3:48 上那是**8/8 张量核候选正确性失败**得出的,属于跨候选历史,既不在 evidence 里也不在这个候选自己的测量里。**从单个候选的指令流编出它,与那个虚名屋顶本身是同一类错误** —— 一句没有依据的自信断言。它属于 S3(屋顶出处),不属于这里。

**位置也是判据的一部分**:不可达屋顶**放在 findings 之外的独立字段**,渲染在「阅读顺序」之后。J2-4 的失败条件是「它仍出现在 recommendation 里」,这只有在它结构上位于排序列表之外时才有意义。

### `agents/modules.py` —— G28

`_bottleneck_doc` 新增 `digest_text: str | None`。**vector 模式下它替换掉整个 `## Verdict: **{kind}**` 段与逐键 `evidence` 渲染** —— 那正是 J2-3 禁止的东西。

**参数刻意是字符串,不是 Digest 对象。** 这个函数就是 prompt 边界;若它接对象,它就能伸手取 `.findings[i].confidence` 渲染数字,那道门就变成依赖这个函数的自律而不是依赖它的签名。

**两个模式都保留的部分**:任务代价段、本机天花板段、以及「本分析看不到什么」。前两个描述的是**任务**和**机器**,不是候选的资源向量;第三个若在 vector 模式下丢掉,那是**删掉一条警示而不是改变形式**,对照 run 的两臂就会同时差两件事,J2-5 无法归因。

### `control/orchestrator.py` —— `_dimension_digest`

**两个决定,刻意分开:**

| | 条件 | 理由 |
|---|---|---|
| **记录**(`DIMENSION_STATE` 事件) | **无条件,两个模式都写** | 写它不改变任何 agent 的行为;而且它是让 J2-1 能从**对照臂自己的日志**离线重放的原因。若只在 vector 模式写,对照臂就没有向量可比,J2-1 得多跑一个 run |
| **进 prompt** | `v3.diagnosis.mode == "vector"` | 这才是有外部反例的那一臂(few-shot 范例实测把 KernelBench L1 fast_1 从 10% 压到 6%)⇒ 需要对照 run 而不是论证 |

按 `(candidate_id, structural_signature)` 键入,**绝不按族** —— 按族正是 rewrite 会静默继承父候选向量的路径(J2-7)。

事件里带 `prompt_mode`:后来的读者不应该需要从 config 文件反推这个 run 在哪一臂,config 可能已经改了。

### `evaluation/reading_checks.py` —— P4 缺的那一半

`check_applicability()`,现在检五种不一致。**其中两条是被自己抓出来才写对的**,见下节。

## 三、J2-* 判据落地情况

| 判据 | 状态 | 怎么验的 |
|---|---|---|
| **N1**(必败)`applicable=false` 带 `measured` | ✅ | `test_n1_applicable_false_with_a_measured_value_is_reported`;反向验证:P4 前的 `reading_checks`(grep 确认 0 处 `applicable`)什么都不报 |
| **N2**(必败)原始向量进 prompt | ✅ | 两条:类型(record/state)+ 签名键(evidence dict)。且**行为断言** —— 渲染文本里不得出现 218 / 93.7 / 1048576 / 0.167 这些实际数字(断言键名会在改名时失效) |
| **J2-1** 区分度 | ✅ | 两个候选的**整个判决元组**不同,且墙的位置不同(occupancy vs shared_bytes);**反向对照也写了** —— 同一 evidence 两次必须给同一结果(否则「区分度上升」可能是抖动) |
| **J2-4** 不可达屋顶被标出 | ✅ | 三种情形 + 一条「没测到就不许声称」的对照;并断言它**不在** `ranked_recommendation` 里 |
| **J2-6** 多维同时 binding | ✅ | occupancy 0.167 + spill 48 ⇒ **两条** binding 记录;`ordering_basis` 必须说「同时绑定」而不是悄悄排序 |
| **J2-7** 改写后重测 | ✅ | 父子对,断言键不同**且**记录不同(只断言记录会让「忽略 signature 参数」的实现也通过) |
| **J2-9** 维度不被砍 | ✅ | findings 数 == records 数;且每一维都必须出现在 prompt 文本里(削减同样可能发生在渲染层) |
| **J2-2b(b)** 107.8% 报告而非拒绝 | ✅ 既有 | `bottleneck.py:543` >1.05 报 `unknown` |
| **J2-8** L2 内不报 DRAM 绑定 | ✅ 既有 | `test_g26_l2_working_set_gate.py` |
| **J2-5** 终局不变差 | ⏳ **对照 run** | S2 最大风险点,只能实测 |

## 四、反向验证抓到了什么(比它确认的更多)

`scripts/revert_check_s2.py`:14 个错误实现,每个断言「点名的那些测试必须失败」。**结果:**

### (1) 我自己的三个变体是错的

- 「无极性」变体点了 J2-1 的测试 —— 但 J2-1 那个测试**当时也在动 spill**,而 `n_spills` 是特例、不走 `_band`。于是它在「`_band` 恒返回一个值」的实现上照样通过 ⇒ **它当时是 spill 分支的证据,不是它所命名的分档的证据**。已改成只动被分档的维度、spill 两边都为 0。
- 「G28 未关闭」变体点了「保留 unmeasurable 列表」这个测试 —— 但**旧的标签路径也渲染那个列表**,所以它正确地在该变体下存活。已移出该变体的点名清单(它是下一个变体的证据)。
- 「压缩器」变体点了「有实测值不得渲染成 NOT MEASURED」这个测试 —— 而那个测试当时是 `for line in ...: if 维度名 in line`,**维度被整个丢掉时循环体一次都不执行,空过**。已加 `assert seen == 2`。

### (2) 一个测试点名了错误的规则

「`applicable=False` 无理由」原本只断言 `assert any("no reason" in n ...)`。但**规则 5(`not-applicable` 判决无理由)对同一输入也会开火**,所以规则 2 被删掉时它照样通过。已改成断言规则 2 自己的措辞。两条规则对读者说的是不同的事:规则 5 关于判决,规则 2 关于标志位。

### (3) 一个真 bug

`if verdict is None or not verdict.evidence: return None` **写在 try 之外**。`verdict.evidence` 是对别处构造的对象取属性,它自己就可能失败 —— 而守卫在外面时,那个失败会**逃出下面的 except**,杀掉候选的分析步骤。**那个 except 存在的全部目的就是「诊断层的缺陷不得表现成候选的缺陷」**,而守卫的位置恰好绕开了它。已移入 try。这条是 happy path 永远测不出来的。

### (4) 检查脚本自己读 pytest 状态读错了三次

| 错法 | 后果 |
|---|---|
| 从摘要行里 grep `" passed"` | `-q -q` **完全抑制摘要行** ⇒ 全绿的 suite 报成「基线不绿」 |
| 用子串猜「是否被 skip」 | skip 与「在坏代码上通过」**无法区分** |
| 用 `-rs` 拿 skip 列表 | 它打的是 `file:line`,**不是 node id** ⇒ skip 被当成「在坏实现上通过」 |

**把 skip 报成「passed on the wrong implementation」是对一个测试的诬告,报成 ok 是伪造体检报告。** 最终改用 `-v`,每行同时带 node id 和结局词。

### (5) 我自己的两条记录被自己的检查判为不一致

**这是 P4 检查的第一次实战,它抓的是我。**

| 问题 | 我原来的写法 | 为什么错 | 修法 |
|---|---|---|---|
| aten 两维有实测值但 verdict `unknown` | 规则 3 无条件报「悄悄拒判」 | 它们**没有天花板可分档**(下界,无分母),`unknown` 在那里是诚实的 | 规则 3 加「存在天花板」前提;**并加了反向测试** —— 有天花板却 `unknown` 仍必须被抓,两个方向都覆盖 |
| `threads_launched` 标了 `applicable=False` | 把两个问题混成一个 | 读数**存在且被测到**,不适用的是**分档**(该维无好坏方向)。说机器没有这一维是**假话**,而且把一个真实测量藏在一个意为「不存在」的标志位后面 | `applicable=True` + verdict `not-applicable` + 必带理由;**并加规则 5**:`not-applicable` 判决无理由必须被报,因为它现在有两个成因 |

**这两条同时让 prompt 说了假话** —— 有实测值的维度被渲染成「NOT MEASURED on this candidate」。已修:「go measure it」和「没有东西可比」是不同的动作。

**一条纪律**:修法必须**不弱化检查**。检查存在的全部目的是「`applicable=false` 与 `measured=0` 永远可区分」,所以两处都是**加前提 + 加反向测试**,不是放宽。

## 五、这次没做什么

1. **`classify()` 一行未改。** 标签路径原样保留 —— `v3.diagnosis.mode` 必须可切换才能做对照 run。
2. **`num_warps` 没有记录。** 它既不在 `classify()` 的 evidence 里(grep 0 处)也不在 `_DIMENSIONS` 里,**没有东西可读**。刻意不用默认值造一条:那正是「可信的常数」故障形态,而且会虚增 J2-9 的维度计数却什么都没测。写完后作为死代码删掉了。
3. **四个 SASS 计数器仍未接入**(`shared_load` / `shared_store` / `vec_64` / `barrier`)。它们已在采集、已躺在磁盘上,但接入属于 S2b,而 S2b **因缺正对照被挂起**。
4. **J2-5 未验证。** 只能靠对照 run,这是本阶段唯一无法离线判定的判据。
5. **A800 上 2 个变体未验证**(需要 optuna 的那两个 wiring 测试在本机 skip)。

## 六、待办

- [ ] A800 全套 suite + 在 A800 上重跑 `revert_check_s2.py`,确认那 2 个未验证变体
- [ ] S2d(动作—资源预期账本)
