# v2 Harness 改进实施方案

> 日期:2026-09-02(方案)/ 2026-09-03(实施完成)。依据:`framework-diagnosis-and-improvements.md`
> (磁盘核实的诊断)+ `../research/kernelfoundry-findings.md`(KernelFoundry 调研)。
> 本方案给出**每项改进的具体改动点(文件:行)、代码草图、风险、验收**,可直接据此实现。
> **原则**:只借鉴"硬约束/泛化"做法,不写死"性能策略"类提示(见 §0)。
>
> **实施状态(2026-09-03)**:A/B/C/D.1/E/F 全部实施完成,G 已移除。默认行为不变
> (config flag 控制),experiments_l3.yaml 已开启 A(dual_witness_relaxed)+F(repair 3)。
> 全套 **90 passed, 1 skipped**(host 无 torch 的 relaxed_close,WSL 侧已验证)。
> 代码已 push 到 github Fudan-SMI-lab/opop 的 v2 分支留档。实施细节见 `../../HANDOFF.md` §10。
> **下一步**:用改进后代码重跑 level3:43/level3:21 验证通过率与 Loop C trace 的实质改善。

---

## 0. 指导原则(先定边界,避免过度工程)

1. **只加泛化增益,不加 case 补丁**。prompt 只写 Triton 语言/编译器层面的硬约束(违反即编译失败或
   必然数值错),不写"押某种优化方向"的性能策略——后者有 case 依赖、可能反作用。
2. **改判定口径优先于改生成**。提升通过率最根本的一步是让"数值上够好"的候选能过门(双精度见证),
   其次才是减少低级错误(prompt/静态门)。
3. **每项改动可独立开关 + 可回退**,用 config flag 控制,默认行为在验收通过前不变。
4. **不破坏现有 82 测试 + 事件溯源/resume 语义**。新增字段向后兼容(replay 老 run 不报错)。
5. **诚实口径**:任何放宽正确性判定的改动,必须在 report 里注明判定方式与基线精度,
   避免与其他项目产生不可比数字。

---

## 第一梯队 — 直接提通过率(优先做)

### 改进 A:双精度见证正确性门(收益最高)

**动机**(诊断 §2.1):level3:43 四个 seed 因默认 tf32 点积在 attention 长 reduction 上
累积误差 >1e-4 被判 fail。KernelFoundry 的对策:参考同时用 tf32 和 ieee 各算一份,候选匹配**任一**即过
(`kernelfoundry/tasks/kernelbench/task.py:123-145`)。

**关键约束**:KernelBench 的 `eval_kernel_against_ref`(`KernelBench/src/kernelbench/eval.py:394`)
只接受**单个** `precision`,内部自己做 `torch.allclose`(`:804`),**不返回原始张量**,
也没有"匹配任一参考"模式。因此**不能靠改调用参数实现**,必须在 v2 worker 侧新增一个
自建的相对误差正确性 job(复刻 KernelBench 的输入生成 + 参考前向,但比对逻辑换成双精度见证 + 宽松容差)。

**改动点**:

1. `src/kernel_optimizer/gpu/jobs.py`:新增 job 类型 `eval_correctness_relaxed`,
   `make_relaxed_correctness_job(ref_path, kernel_path, backend, num_trials, seed, ...)`。
2. `src/kernel_optimizer/gpu/worker_main.py`:新增 handler `run_relaxed_correctness(job)`。
   复刻 `run_and_check_correctness`(`KernelBench/eval.py:727`)的**输入/种子生成逻辑**
   (必须逐字节一致:`torch.manual_seed(seed)` → 生成 `num_trials` 个 trial_seed → 每 trial
   `set_seed(trial_seed); inputs=get_inputs_fn()`),但比对换成:
   ```python
   # 双精度见证:同一参考跑两遍
   set_fp32_matmul_precision("high")     # tf32
   out_ref_tf32 = ref_model(*inputs)
   set_fp32_matmul_precision("highest")  # ieee
   out_ref_ieee = ref_model(*inputs)
   out_kernel = kernel_model(*inputs)
   ok = _relaxed_close(out_ref_tf32, out_kernel) or _relaxed_close(out_ref_ieee, out_kernel)
   ```
   其中 `_relaxed_close`(复刻 `kernelfoundry/testing.py::all_close_with_slack:33-62`):
   ```python
   def _relaxed_close(ref, got, elem_tol=0.01, frac=0.99, eps=1e-7):
       if ref.shape != got.shape: return False
       rel = (ref - got).abs() / (ref.abs() + eps)
       return (rel < elem_tol).float().mean().item() > frac
   ```
   外加余弦相似度第二判据(可选,≥0.99985)。**shape 不匹配仍硬 fail**(不放宽)。
3. `src/kernel_optimizer/config.py` + `configs/*.yaml`:`evaluation` 段新增
   ```yaml
   correctness_mode: strict        # strict(现状,KernelBench 1e-4) | dual_witness_relaxed
   relaxed_elem_tol: 0.01
   relaxed_pass_frac: 0.99
   cosine_min: 0.99985
   ```
   默认 `strict`,不改现有行为;实验用 config 显式切 `dual_witness_relaxed`。
4. `src/kernel_optimizer/evaluation/correctness.py`:`quick_test`/`full_eval`/`screen` 按
   `correctness_mode` 分派到 relaxed job 或现有严格 job。**注意**:relaxed job 只判正确性,
   计时仍走现有独占计时路径(不要把宽松判定混进计时 job)。

**风险与缓解**:
- **假阳性**(宽容差放过实际错的 kernel):必须配套改进 D(任务敏感性黑名单),
  且保留 `strict` 模式做最终复测的可选二次确认。report 注明用了 relaxed。
- **复刻输入生成不一致**:若 v2 自建 job 的随机输入与 KernelBench 不完全一致,会与 baseline 不可比。
  **验收硬门**:对一个已知正确的 ModelNew,relaxed job 与 KernelBench strict job 在 ieee 下结论一致。
- **基线精度口径**:若采纳,`benchmark.py` 的 baseline 也应记录 tf32 与 ieee 两版(或至少注明用哪版),
  speedup 分母明确。

**验收**:
- 已知正确 kernel:strict 与 relaxed 都 PASS。
- level3:43 的历史 tf32 候选(从旧 run artifacts 取 materialized 源):strict FAIL、relaxed PASS。
- 故意做一个"输出全 0"的错误 kernel:relaxed 仍 FAIL(不能被宽容差放过)。
- 全套 pytest 绿 + 新增 relaxed-correctness 单测。

---

### 改进 B:generator/repair prompt 加 Triton 硬约束(BAD/GOOD 对)

**动机**(诊断 §2.1):generator 反复踩 `tl.next_power_of_2`(设备代码)与 tf32 两坑。

**只加硬约束类**(违反即编译失败或必然数值错),**不加性能策略类**。清单(来自 KernelFoundry
`meta_prompting.py:418-425` + 补充,均为 Triton 语言/编译器事实):

| # | 规则 | 类别 |
|---|---|---|
| 1 | `tl.arange` 越界访问用 `mask=offs < N` | 正确性铁律 |
| 2 | BLOCK 尺寸必须 2 的幂 | 编译硬约束 |
| 3 | 编译期常量必须 `tl.constexpr` | 编译硬约束 |
| 4 | `tl.dot` 两输入维度须被 16 整除 | MMA 硬约束 |
| 5 | `tl.dot` 混精度:转精度后再点积,**累加用 fp32** | 数值通则 |
| 6 | `tl.next_power_of_2` 只能 host 侧调用(`triton.next_power_of_2`),结果作 `tl.constexpr` 传入;`tl.arange(0,X)` 的 X 必须 constexpr 2 的幂 | 编译硬约束(补充) |

**改动点**:
1. 新建 `src/kernel_optimizer/agents/prompts/triton_pitfalls.md`,内容为上表 + 每条一个
   BAD/GOOD 代码对(格式照 `kernelfoundry/optimization_aware_prompts.json` antipatterns.triton)。
2. `src/kernel_optimizer/agents/modules.py`:新增 `_triton_pitfalls_doc()`(照 `_contract_doc():25`),
   在 `GeneratorAgent`/`RepairAgent`/`RewriterAgent`/`NoveltyAgent` 的 `seed_sandbox` 里
   `sb.write_input("docs/triton_pitfalls.md", ...)`,并在各自 `render_prompt` 里加一句
   "backend 为 triton 时先读 `docs/triton_pitfalls.md`,避免其中列出的编译/精度错误"。
   **仅在 backend=triton 时注入**(CUDA 候选不写,避免无关噪声)。

**为什么这是泛化增益、不是 case 补丁**(回应质疑):这 6 条约束的是"合法/正确"边界——
遵守它**不会让任何正确 kernel 变差**(如不需要 mask 的整除尺寸 kernel 加 mask 只是恒真判断)。
反例是性能策略类(如强推 `num_stages=3`、flash-attention 模式),那些有 case 依赖,
**本方案明确不写进静态 prompt**;若要用应走改进 F 的动态注入。

**风险**:prompt 变长增加 token 成本(可接受,每次调用固定几百 token);
过度约束可能让模型保守(低)。**验收**:改后 level3:43 重跑,seed 阶段
`runtime_error`(编译失败)类拒绝显著下降(对比旧 run 的 4/4 编译失败)。

---

### 改进 C:GPU 评估前加零成本静态门

**动机**:`tl.next_power_of_2` 类致命模式,与其等 WSL 编译失败(耗 GPU 配额 + 分钟级),
不如生成后立即 regex 拒绝/反馈。KernelFoundry 提取代码后就跑 `postprocess_code`。

**改动点**:
1. 新建 `src/kernel_optimizer/paramspace/triton_lint.py`:`lint_triton_source(src) -> list[str]`,
   纯 regex/AST 检测(host 侧、零 GPU):
   - `@triton.jit` 函数体内出现 `tl.next_power_of_2(` → 报错。
   - `tl.arange(0, X)` 的 X 不是 `constexpr` 参数名/字面量 2 的幂 → 警告。
   - kernel 签名里用作 arange 上界/循环边界的参数没标 `tl.constexpr` → 警告。
2. 挂载点:`AgentModule.check_output`(`base.py:148`)——generator/rewriter/novelty 输出候选文件后,
   若 backend=triton 且 lint 命中**硬错误**,返回 problem 文本触发**同一次调用内的重试**
   (带具体错误反馈),比丢到 GPU 再失败便宜得多。这复用了现有重试机制,不新增循环。

**风险**:regex 误报(把正确写法当错)。**缓解**:只把"必然编译失败"的模式设为硬错误
(`tl.next_power_of_2` 在 jit 体内是确定失败),其余仅作 warning 附在反馈里不阻断。
**验收**:构造含 `tl.next_power_of_2` 的样例源,lint 命中;正确的 host-constexpr 样例不误报;
单测覆盖。

---

### 改进 D(配套 A):契约文档纠错 + 任务正确性可测性检查

**动机**:两处:
1. **契约文档写错了容差**:`prompts/candidate_contract.md:50` 写 "tolerance atol=rtol=1e-2",
   但 KernelBench fp32 实际是 **1e-4**(`eval.py:95`)。这会误导 agent 以为有 100× 的数值裕度。
   **直接改**:把该行改为准确描述(strict 模式 1e-4;若启用 dual_witness 模式则说明双精度见证语义)。
2. **宽容差前必须做任务敏感性检查**(KernelFoundry `FILTERED_OUT` 19 个低敏感任务):
   放宽容差在"输出方差低/对输入不敏感"的任务上会假阳。新增一次性检查:对参考模型扰动输入
   (相对 1% 噪声)看输出变化幅度,若变化 < 判定阈值则该任务标记"relaxed 不安全",退回 strict。

**改动点**:
1. `candidate_contract.md:44-50` 改写(纯文档,零风险,**可立即做**)。
2. `src/kernel_optimizer/tasks/kernelbench.py` 或 `evaluation/`:新增 `task_sensitivity_probe(task)`,
   在 run 启动(baseline 阶段)跑一次,结果记入 events(`TASK_SENSITIVITY_PROBED`),
   若不安全且 config 为 relaxed 则该任务自动降级 strict 并在 report 注明。

**验收**:probe 对 attention 任务判"安全"(输出对输入敏感);对一个已知低敏感任务判"不安全"。

---

## 第二梯队 — 提搜索有效性

### 改进 E:修 novelty 名额逻辑(dropped 族不占额)

**动机**(诊断 §2.5):`_novelty_round` 首行 `if len(families) >= max_families_total: reject`
(`orchestrator.py:_novelty_round` + `families.py::accept_novel_seed:149`)。seed 全灭时
4 个失败/dropped 族占满 `max_families_total=3` 名额 → novelty 一个都产不出。

**改动点**:`src/kernel_optimizer/control/families.py::accept_novel_seed:149`——
名额计算改为只数**"活的或有 best 的"族**,排除 status 为 dropped/无 best 且已冻结的族。
或更清晰:新增 `active_or_productive_family_count()`,`accept_novel_seed` 用它替代 `len(self.families)`。
需同步 `orchestrator._novelty_round` 里构造 summaries 的遍历(跳过 dropped 族的 anchor 读取)。

**风险**:族总数无上限会爆 → 保留一个更大的硬上限(如 `max_families_total_hard=6`)防失控。
**验收**:构造"3 个 dropped 族 + 预算未耗尽"场景,novelty 能产出第 4 族;单测。

---

### 改进 F:提高 repair 预算 + 失败分级回传

**动机**(诊断 §2.1、§2.5):repair 2 次面对"编译错+精度错"双层问题不够;
且 repair 没拿到结构化的失败类别。KernelFoundry 有梯度化分级(0-5)。

**改动点**:
1. `configs/experiments_l3.yaml`:`budgets.repair_attempts: 2 → 3`(小改,先验证收益)。
2. `src/kernel_optimizer/agents/modules.py::RepairAgent.render_prompt:357`——
   prompt 里显式带上 `failure_kind` 的**分类指引**:数值错→查精度/归约顺序/fp32 累加;
   编译错→查 constexpr/arange 上界/dot 维度。failure_kind 已有(worker 返回),只是没充分用于分流。

**风险**:预算增加线性增 agent 成本。**验收**:level3:43 重跑,repair 后过门率上升。

---

> **改进 G(参考 kernel 库)已从本计划移除**(用户决策,2026-09-02)。理由:该做法依赖模板库
> 质量以及"当前任务是否恰好有对应模板 case",偏离本方案 §0 的泛化增益原则,属于 case 依赖的做法。
> 若将来要做,应作为独立的、可评估收益的实验单独立项,而非纳入本轮泛化改进。

---

## 实施顺序与依赖

```
立即可做(零风险,不依赖 GPU):
  D.1 契约文档纠错 ────────────────────────┐
                                            │
第一梯队(核心,按序):                       │
  B  Triton 硬约束 prompt ──┐               │
  C  静态门 (regex lint) ───┼─→ 一起验证:level3:43 编译失败率下降
  A  双精度见证门 ──────────┘   (A 需要 D.2 敏感性检查配套)
  D.2 任务敏感性检查 ───────────→ A 的安全前提

第二梯队(增量):
  E  novelty 名额修复(独立,小)
  F  repair 预算+分级(独立,小)

每梯队结束跑一次 level3:43 干净实验验证增益,再进下一梯队。
```

**建议的第一个可交付里程碑**:D.1 + B + C + A + D.2 五项做完,跑一次 level3:43 干净实验,
对比旧 run 的"4/4 seed 全灭"看通过率是否实质改善。这是收益最集中、且能直接验证"诊断是否对"的最小闭环。

---

## 每项改动的测试与回归要求

- 所有改动**默认关闭或默认保持现有行为**(config flag),验收通过前不改默认。
- 每项配至少一个单测(lint、relaxed_close、novelty 名额、敏感性 probe)。
- 全套 pytest 必须保持绿(当前 82)。
- 事件溯源:新增事件类型(`TASK_SENSITIVITY_PROBED` 等)必须让 replay 老 run 不报错(向后兼容)。
- 涉及正确性判定口径的改动(A/D),report 必须注明判定方式与基线精度。

---

## 遗留决策(已拍板,2026-09-02)

1. **是否采纳宽松容差**:✅ **采纳建议方案**——启用 tf32+ieee 双精度见证宽松容差,
   但**强制配套任务敏感性检查(D.2)+ report 注明判定方式**,并保留 strict 模式做最终复测二次确认。
2. **基线精度口径**:✅ **采纳建议方案**——baseline 同时记录 ieee 与 tf32 两版参考,speedup 分母明确标注。
3. **改进 G 的模板库**:❌ **移除**——依赖模板库质量与任务 case 匹配,偏离泛化增益原则,不纳入本轮。
