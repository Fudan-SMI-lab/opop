# S3 实施完成:每维下界 + 天花板出处进字段

**日期** 2026-09-11 · **commit** `dca3e02` + `66d2d7a` · **A800(权威)** 623 passed / 0 failed · **反向验证** 13/13 判别

---

## 一、两个缺口,是两类不同的东西

### (1) 天花板出处

`ceiling_provenance` 在 S2 之前**全仓 0 处**;S2 加的是一个**自由文本字符串** —— 而那正是 S2d 刚在预期那一侧判定为**不可证伪**的形态。

**正对照是一次事故,不是一个假设**:一个 fp16 候选被拿 tf32 天花板当分母,读出 **107.8% of peak**,而它的真实数字约 **54–60%**。而 107.8% 在下游读作「**已到顶,别再优化了**」——**指令被完全反转。**

一句话说「tensor-core ceiling」**无法被问「哪个精度的?」**。所以 `precision` 与 `backend` 是**独立字段**,`precision_mismatch()` 因它们才可能存在。

### (2) 下界

天花板回答「离屋顶多远」,**不回答「还剩多少空间」** —— 因为一个维度的地板通常不是 0。

| 维度 | 地板 | 来源 |
|---|---|---|
| `candidate_aten_bytes` | `compulsory_bytes` | **在参考上实测**(task_cost)。融合能去掉所有中间量,去不掉这些 |
| `candidate_aten_ops` | 1 | 定义式:全融合实现只 dispatch 一个 op |
| `n_spills` | 0 | 定义式 |
| `peak_alloc_bytes` | `compulsory_bytes` | **弱**,confidence 0.5:缓存分配器夹在两者之间 |
| `n_regs` | **UNKNOWN** | 无闭式,**连符号都不可靠**(13 条非单调切片) |
| `shared_bytes` | **UNKNOWN** | 候选闭式公式 96 个点**命中 0 个** |
| `occupancy` | **UNKNOWN** | 它是寄存器/shared/warp 预算的**输出**,不是独立量 |
| `threads_launched` | **UNKNOWN** | **另一个理由**:无极性,「剩余空间」没有方向可言 |

**四个 UNKNOWN 是判据本身,不是欠账。** `LowerBound.source` 是闭合词表,**没有「猜」这个成员**。

## 二、为什么 `unknown` 必须可达

**编造的下界比没有下界更糟。** 一个虚构的地板会让某维看起来快用尽了,把精力从它身上移开;而下游**无法区分**推导来的地板和实测的地板 —— 除非记录说明是哪种。

`test_unknown_is_the_majority_answer_and_that_is_the_point` 把这条写成断言,**并且对着词表检查而不是硬编码列表**:如果将来某次改动让多数维度都报出地板,那正是这条判据要禁止的编造。

## 三、读数低于自己的地板:报告,绝不 clamp

**这不是假设情形。** `candidate_aten_bytes` 本身就是候选真实流量的**下界**(融合过的 kernel 在一次 aten 调用里做完,中间字节从未被 dispatch)⇒ **融合良好的候选会读出低于 `compulsory_bytes`**。

clamp 到 0 会说「**0% 剩余空间,你已在地板上**」—— 对一个**只是无法用这种方式测量**的候选,这是假话。

**与 `check_reading` 的越界规则同一纪律,同一理由:clamp 会保留错误的动作。**

## 四、消费者

`digest` 渲染两者 —— **没有东西渲染的下界等于没实现**,与「`conversion` 零消费者」是同一个论证。

**「剩余空间」自成一节**,因为「离屋顶多远」和「还剩多少空间」是两个不同的问题;把一个 UNKNOWN 混进同时陈述档位的那一行,会让读者把档位读成剩余空间。

## 五、冒烟测试抓到我自己一句自相矛盾的话

`uses_tensor_cores=False` 配一个 tensor-core 标签时,那条 caveat **自相矛盾**:

> the tensor-core roof is **NOT the denominator** ... so the compute percentage above is against the **tensor-core (tf32)** roof

`classify()` 使这个组合不可达(`if peaks and uses_tc:`)。但**一句自相矛盾的话进了 prompt,比它的任何一半都糟**:读者**完全无法据此行动**,也无法判断该怀疑哪一半。现在它被报成一个「分母未知」的不一致。

## 六、反向验证:13 个变体全部判别,其中 4 个是同一个失败族

**刻意向一个方向配重** —— S3 存在的理由就是防止编造地板,而那四个变体每一个都**看起来完全合理**:

| 变体 | 后果 |
|---|---|
| 寄存器地板取 0 | 最诱人的默认值,也是**最强的断言**(说这一维可以被完全消除) |
| task cost 未测时读作 0 | 「你可以消除所有流量」—— 从零数据做出的最强断言 |
| 低于地板的读数 clamp 掉 | 「0% 剩余空间」 |
| 弱分配地板按满 confidence 报 | 邻近量的界被当作本量的界 |

**另外两个是我自己的变体写错了**,由脚本抓出:

1. 一个变体点名了 `test_definitional_floors...` —— 但定义式地板在读 `task_cost` **之前**就返回了,所以它**正确地**在该变体下存活。已移出点名清单。
2. 见 §七。

## 七、A800 抓到一个测试 fixture 的缺陷(第三次同形)

S3 让 `_dimension_digest` 读 `self.task_cost` 与 `self.calibration`,而 `test_s2_wiring.py` 的 fixture 造 Orchestrator 时**没有这两个属性** ⇒ A800 上四个测试报 `AttributeError`。

**这是 G41 那个形态第三次出现:一个不是生产形状的 fixture 什么都不验证** —— 那四个测试当时全都在走 `DIMENSION_STATE_FAILED` 路径,却在断言 `DIMENSION_STATE` 的内容。

**让它可诊断的是 G40**:那个 handler 把 `AttributeError` 变成落盘的错误字符串而不是一个死掉的分析步骤,所以根因**两条命令**就找到了。

三处修改,只有第一处是 bug:

1. fixture 补上两个属性。**设为 None 而不是对象** —— None 是**真实状态**(未标定的机器、task cost 测量失败的 run),下界必须退化成 `source="none"` 而不是抛异常。
2. 新增 `test_the_s3_bound_and_provenance_reach_the_journalled_record`:**S3 落地时缺的就是这个测试**。它驱动真实方法、带真的 TaskCost 与标定,断言地板与出处进了事件日志、断言 `n_regs` **仍然没有地板**、并断言 `DIMENSION_STATE_FAILED` **不在** —— 那正是旧版本看不见的东西。
3. 新增退化路径的测试,让「没有标定/没有 task cost」成为一个**被测状态**而不是一次意外。

**并且把 `test_s2_wiring.py` 纳入 S3 反向验证的范围**,加一个「task cost 变成必需」的变体 ⇒ **这类失败以后由脚本抓,而不是由 A800 抓。**
