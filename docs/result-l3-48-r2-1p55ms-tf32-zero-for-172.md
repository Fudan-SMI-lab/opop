# L3:48 r2 结果:1.5544 ms(fp16),比 r1 差 9.9%;tf32 分支 0-for-172

**状态**:box 1,`run-l3-48-20260909-115701`,2026-09-10 00:15:30 自行 finalize(`RUN_FINISHED` 在盘上)。全部数字来自 events.jsonl 与 report.md,不来自通知内容。

---

## 一、头条数字

| 项 | r2(本次) | r1(`run-l3-48-20260907-202457`) |
|---|---|---|
| 获胜候选 | `cand-efcecff9`(family `fam-e08cafb2`) | `cand-90060cab`(family `fam-110a2dd6`) |
| tuned median | 1.5544 ms | 1.48 ms |
| **final_reeval** | **1.55 ms**(median 1.5616) | **1.41 ms** |
| 精度 | fp16 | fp16 |
| **同精度加速比**(vs `torch_compile_tf32`) | **8.4516x** | **9.2908x** |
| vs eager | 13.2903x | — |
| 结束方式 | wall clock,12.295 h / 12.0 h | wall clock,15.747 h / 12.0 h |
| 改写轮 | 3 across 4 families([1,0,1,1]) | 3 across 4 families([1,1,1,0]) |
| trials | 680(371 complete) | 720 |

**基线在两个 run 里逐项相同**(eager 20.6/20.7、eager_tf32 20.1/20.1、torch_compile 13.7/13.7、torch_compile_tf32 13.1/13.1,同一台 box 同一张 4090),所以这个对比是**同卡同基线**的,不需要折算。

**结论:r2 比 r1 差 9.9%(1.55 vs 1.41 ms)。** 但 r1 的墙钟超支到 15.747 h,而 r2 只用了 12.295 h —— **两者的预算不同,所以这不是一次干净的复现实验**。r1 多出的 3.45 h(28%)是最直接的候选解释,但未验证。

**r2 的获胜者带一个必须说明的警告(report 自己标了)**:`cand-efcecff9` 的正确性**完全依赖 fp64 相对臂** —— 3 of 3 correctness trials 被相对臂救回,主 relaxed 门在这三次上都已失败。report 的措辞是对的:这是合法通过而非漏洞,但**一个速度相近、能直接过主门的 kernel 在数值上是更好的结果**,像对像的比较必须这么说。

**近平局警告(box 侧脚本自算)**:371 个 complete trial 里 **126 个(34%)** 落在获胜者的 1 个合并 SEM 之内,带宽 1.5544–1.6323 ms(5.01%),而合并 SEM 是 0.0857 ms(占 best 的 **5.51%**)。**SEM 比带宽还宽 —— 获胜配置在这个采样量下与它的对手不可区分。** 带内的 `DOT_PRECISION` 分裂是 fp16 111 / ieee 10 / bf16 5。

---

## 二、本 run 最重要的发现:tf32 分支 0-for-172,且 25.9% 的 trial 墙钟花在上面

### 2.1 实测

| 精度 | trials | passed | 占 trial 墙钟 |
|---|---|---|---|
| fp16 | 277 | **257** | 144.0 min(37.8%) |
| **tf32** | **172** | **0** | **98.5 min(25.9%)** |
| bf16 | 127 | 25 | 71.7 min(18.8%) |
| ieee | 104 | 89 | 66.6 min(17.5%) |

**tf32 是 0-for-172,且在全部 10 个候选上都是 0**,包括那些 fp16 拿到 38/38、bf16 10/10、ieee 12/12 的候选:

```
cand-efcecff9      bf16=10/10  fp16=38/38  ieee=12/12  tf32=0/20
cand-6498ec62      bf16=6/6    fp16=15/15  ieee=7/7    tf32=0/12
cand-8a4b7286      bf16=9/11   fp16=33/33  ieee=9/9    tf32=0/27
cand-2c5bf5af      bf16=0/8    fp16=18/18  ieee=7/7    tf32=0/7
(其余 6 个同形)
```

### 2.2 这不是门的问题 —— 跨 run 对照证明了

**关键对照:tf32 在 7 个 L3 run 中有 6 个正常通过。**

| run | 精度 knob | tf32 通过率 |
|---|---|---|
| box1 run-l3-21-20260908-232211 | `COMPUTE_DTYPE` | **146/167** |
| box1 run-l3-43-20260908-121539 | `COMPUTE_DTYPE` | **26/35** |
| box1 run-l3-48-20260907-202457(r1) | `COMPUTE_DTYPE` | 22/72 |
| **box1 run-l3-48-20260909-115701(r2)** | `DOT_PRECISION` | **0/172** |
| box2 run-l3-21-20260909-154359 | `COMPUTE_DTYPE` | **110/155** |
| box2 run-l3-43-20260908-053708 | `COMPUTE_DTYPE` | **97/133** |
| box2 run-l3-43-20260909-015247 | `COMPUTE_DTYPE` | **71/111** |

**所以 tf32 没有被门结构性否决。** 0-for-172 是这一个 run 的性质。

### 2.3 根因:tf32 在所有 10 个候选里都落在未补偿的 `else` 兜底分支

读获胜者 `cand-efcecff9` 的源码(`candidates/cand-efcecff9/source.py:134-170`):

- `DOT_PRECISION == "fp16"` → 有 **`rho` 最大绝对值稳定化**:`rho = tl.maximum(tl.max(tl.max(tl.abs(srun)...)), 1e-30)`,再 `srun / rho`,最后 `yo * (tl.exp(acs) * rho)`;
- `DOT_PRECISION == "bf16"` → 走 **`_dot_bf16_comp` 补偿分裂 dot**(高/低位分裂成两个 bf16);
- `DOT_PRECISION == "ieee"` → 全部 `input_precision="ieee"`,精确;
- **`else:` → 裸 `tl.dot(..., input_precision="tf32")`,零补偿、零稳定化。**

**10/10 个候选都是这个形状**:每一个都给 fp16、bf16、ieee 写了具名分支,**没有一个给 tf32 写具名分支** —— 它总是那个 `else`。脚本对分支体扫补偿标记的结果:

```
cand-efcecff9      fp16=STAB  bf16=STAB  ieee=plain  else/fallback=plain
cand-6498ec62      fp16=STAB  bf16=STAB  ieee=plain  else/fallback=plain
cand-8a4b7286      fp16=STAB  bf16=STAB  ieee=plain  else/fallback=plain
(其余同形;ieee=plain 是对的,它本来就精确)
```

fp64 相对臂的比值证实了这是数值问题而非门问题:tf32 失败的 164 次里 **min 4.609、median 4.982**,阈值是 multiplier 2.0 × 参考自身 rmse。**即使按低精度那档的 3.0 也过不了**(4.982 > 3.0),所以这不是乘数选错。而 `frac_within_tol` 0.97605 与参考自身的 ieee-vs-tf32 spread 0.977768 几乎相同 —— **候选在逐元素指标上跟参考自己一样好,是 RMSE 被少数大偏差拉爆**,正是"未补偿 dot"的指纹。

**这与既有记录完全一致**:`compensated dot passes, plain dot fails` —— 低精度 100% 失败有两个成因,(A) 未补偿的 dot,(B) 精度门控整个算法而非 fp16 的落到未稳定化的 else 分支。**本 run 是 (B) 的最干净实例:同一个 kernel、同一个门,fp16 分支 38/38 过,tf32 兜底分支 0/20。**

### 2.4 为什么浪费不会自行衰减 —— 机制已在代码里复验

`tuning/tpe.py:119-121`:

```python
hard = record.failure_kind in ("infeasible_shared_memory", "guard_rejected",
                               "materialize_error")
self.study.tell(trial, state=TrialState.PRUNED if hard else TrialState.FAIL)
```

`correctness_mismatch` **不在** hard 列表里 → 报 `FAIL` → **Optuna 把这些点整个从 TPE 模型里剔除**。于是 tf32 的采样份额在整个 run 里从不下降:

```
quarter 1 (n=170): tf32 0/44 ok
quarter 2 (n=170): tf32 0/44 ok
quarter 3 (n=170): tf32 0/47 ok
quarter 4 (n=170): tf32 0/37 ok
```

**四个季度、172 个 trial、零通过,采样份额纹丝不动。** 这与 shared-memory 那条 180/1004 的记录是同一个机制的另一个实例。

**注意 `correctness_mismatch` 留在 FAIL 是有意的**,`tpe.py:116-118` 写明了理由:它可能是非确定性的,或是候选自身的缺陷而非该点的性质,教 sampler 避开那片区域等于教它噪声。**这个理由对单个 trial 是对的,对"某个 categorical 取值 0-for-172"是错的** —— 后者不是噪声,是一个已被 172 次实测确立的事实。

### 2.5 见证门为什么没拦住

`tf32` 在**每个** space 里都是 published choice(`['fp16','bf16','tf32','ieee']`),但**从来不是 default**(所有候选的 `PARAMS` 都写 `'DOT_PRECISION': 'fp16'`)。而见证门只判 default 值与 minimal 角。**所以 tf32 从未被见证门测过一次**,它只被 TPE 一个 trial 一个 trial 地试出来 —— 每次 18.6 s,共 98.5 min。

这正是 `opop-v2-witness-default-only-weakness` 那条记录的推广:**见证门测不到的 categorical 取值,其代价由调参预算承担。**

---

## 三、本 run 对"后端切换触发器"的检验:未构成检验

这个 run 原本被安排为 backend-switch 触发条件的首次真实测试(prompt 的既定触发条件是 strict IEEE fp32 dot-bound)。

**实测结果:检验没有发生。**

- 10/10 候选声明 `triton`,`detected_backend` 全为 `None`(该 run 的事件未记录检测字段),**0 处 declared/detected 不一致**;
- origin 分布:seed 4、rewrite 6,**没有任何非 Triton 候选**;
- 而 ieee 在本 run 里是**通过**的(89/104),即任务并未把候选逼进"strict ieee 且被算力顶住"的角。

**所以这条仍然零证据**,与 `opop-backend-choice-has-no-evidence` 的记录一致。要真正检验它,需要一个 ieee 是唯一可行精度且确实 dot-bound 的任务/配置,本 run 不是。

---

## 四、其余事实(供后续分析,未逐条追根因)

- **bf16 25/127**:比 tf32 好但仍差。ratio median 6.483(阈值 3.0),同样是数值问题。**但 bf16 有具名补偿分支**,所以它的失败不能用 §2.3 解释 —— 待查。
- **失败分布**:correctness_mismatch 265、infeasible_shared_memory 22、runtime_error 22。**shared-memory 只有 22 次(3.2%)**,远低于历史的 18% —— F5/P5 的编译期前移在起作用(另有 22 次 `CONFIG_SCREENED_INFEASIBLE` 在 trial 之前就被挡掉)。
- **占用普遍 17–25%,limiter 全是 registers**,与 `opop-triton-caps-regs-instead-of-spilling` 一致。
- **DRAM 93.7–94.4%**:本任务仍在带宽墙上,与 `L3:48 is bandwidth-bound at 90%` 一致(r1 曾测到 94.7%)。
- **参数报告里 4 处 `blocked_by=compile_failure`**(BLOCK_P/BLOCK_J/BLOCK_L 想变小或变大但被编译失败挡住)—— 这是 S1 阶段(把编译期可知的不可行从空间里声明掉)的又一处实证。
- 1 次 `AGENT_CALL_FAILED` + 1 次 `AGENT_ARTIFACT_RESCUE`(救援机制生效)。
- 1 次 `HYPOTHESES_FAILED`。

---

## 五、修复判断

按既定标准(**确定 + 严重 + 泛化**才立即修,高风险/不确定的只记录):

### 立即可修(确定、严重、泛化):无一条是"改门"

**F-A(推荐,泛化,低风险)——「一个 categorical 取值 0-for-N 就把它从域里摘掉」**

不是改门,不是改判据,而是**把 §2.4 的机制缺口补上**:当某个 categorical 参数的某个取值累计 N 次全失败(建议 N = 12,与该取值的最小实测样本量同阶),把它从后续 ask 的域里移除,并在事件里记 `DOMAIN_VALUE_RETIRED`(带取值、样本数、失败种类分布)。

- **为什么泛化**:不针对 tf32、不针对 L3:48、不硬编码任何精度名。任何 categorical 的任何取值都适用。
- **为什么严重**:本 run 25.9% 的 trial 墙钟(98.5 min)零产出;而墙钟是本项目最稀缺的资源(19 个 run 只 1 个跑到预算)。
- **为什么低风险**:只在 **N 次全败**后触发,且 N 可配置;`retired` 只影响后续采样,不改变已有结果,也不影响接受判定。
- **必须带的反向对照(会大声失败的那种)**:构造一个"前 N 次全败、第 N+1 次会成功"的合成场景,断言**关掉该功能时第 N+1 次被采到、开启时不被采到** —— 这样"我们放弃了一个本可成功的取值"这个风险是被度量的、而不是被假设不存在的。
- **与 S1 的关系**:这与 v3 的 S1 是**同一个思想的两半** —— S1 处理"编译期可知的不可行",F-A 处理"实测已确立的不可行"。**F-A 属于 v3,不建议在 v2 上改**(见下)。

### 只记录、不改(高风险或需大改)

- **F-B:让候选给 tf32 写具名补偿分支。** 这是 prompt/契约层的改动,会改变 agent 看到的信息 → 改动前后 run 不可比较,且属于"大量改动"。**记录进 v3 的 S2 消化层设计**:诊断应当能说出"你的 tf32 取值 0-for-172,因为它落在未补偿的兜底分支"——这正是消化层要产出的那种"根因 + 排序建议"。
- **F-C:见证门覆盖 categorical 的非默认取值。** 会显著抬高每个候选的准入成本(每个取值一次见证)。**记录**,与 `opop-v2-witness-default-only-weakness` 归并处理。
- **F-D:bf16 25/127 的根因。** 尚未定位(它有补偿分支,所以不是 §2.3)。**只记录。**

### 明确不做

- **不改 fp64 相对门的乘数。** 实测 ratio median 4.982,按低精度那档 3.0 也过不了 → 乘数不是原因。而 `opop-v2-fp64-relative-gate-works` 记录该门承重且不可欺骗(救回的 ratio 全 < 1.0)。**动它会同时放进真正错误的 kernel。**
- **不改 `correctness_mismatch` → FAIL 的分类。** `tpe.py:116-118` 给的理由对单 trial 是对的。F-A 在**域**层面解决,而不是把噪声教给 sampler。

---

## 六、与 v3 的接口

本 run 给 v3 的实施方案添了三处实证:

1. **S1(编译期不可行声明出空间)**:又见 22 次 `CONFIG_SCREENED_INFEASIBLE` + 22 次 `infeasible_shared_memory` + 4 处 `blocked_by=compile_failure`。
2. **F-A 应当并入 S1**,作为"实测已确立的不可行"这一半 —— S1 的验收判据应当同时覆盖这两半。
3. **S2(消化层)**:本 run 是"消化层该说什么"的教科书例子。原始向量会说"tf32 的 frac_within_tol=0.97605";消化后应当说"**tf32 取值 172 次全败,根因是它落在唯一没有补偿的兜底分支,建议给它写具名补偿分支或从域中移除;预期 Δ:回收 25.9% 的 trial 墙钟**"。

**并且本 run 给近平局纪律加了一个更强的数据点**:合并 SEM 5.51% > 带宽 5.01%,**即 SEM 比整个近平局带还宽**。这比 L3:48 r1 的"16% std vs 9% 跨度"更极端,进一步支持 S4 的"每维报价格区间、区间重叠即拒绝行动"。
