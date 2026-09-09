# 框架现状评估与下一步(基于 L3:21/43/48 + L1 smoke,2026-09-09)

写作时两箱在跑:box 1 L3:48 r2(~270/720 min),box 2 L3:21+ceilings(42 min,generator 首调超时后正在换 session 重试——已知 GLM 首调慢的健全恢复,非卡死)。全部结论按当前 HEAD `14fab15` 复验过代码 file:line,不凭记忆。

---

## 一、框架当前是健康的:核心闭环已被真实证据证实

不是"能跑",是**四循环都在真实任务上产出了可归因的结果**:

- **Loop B(调参)**:TPE 排名可信。L3:21 winner 与场分离 >1 SEM(近平局 2/588),L3:43 per-trial CV 中位 0.3%。median 目标已修(93.2% vs mean 64.8% 排序正确率)。
- **Loop C(改写)**:L3:43 r2 的 winner 是**三代连续改写**(种子 3.591→改写 2.822→改写 2.767),改写改善了全部四个族。这是"结构搜索"这一论文主张的最强证据。
- **Loop D(新族)**:曾在 L1:19 跑通过一次(一个族被接受,`finding-relu-memcpy-is-a-benchmark-property.md`),但**在全部 L3 run 中零执行**——不是坏,是被墙钟挤掉(见下 P-D)。
- **Loop A(修复)**:L3:21 两次 repair 都产出被接受的诊断(cudnn legacy flag、stale BN buffer),artifact rescue 从 transport 失败里救回了写好的改写。

**已确认修复且承重**(memory 里标"已修"的,逐一复验代码在位):
- fp64 相对门(correctness.py:220-222 三处路径接线):L3:21 winner 3/3 靠它过,对抗缺陷以 28-290× 被拒。
- cosine 溢出(worker_main.py:1684-1708,float64+归一化):L3:48 曾丢 2 seed,已修。
- fp16/bf16 天花板(P3)+ 缓存 schema 版本(1f18cd1):否则 fp16 候选被当 >100% 峰值。
- F5 compile screen 已前移进 `guard_ok`(orchestrator.py:1028-1029),P5 已落地——shared-memory 不可行配置在 ask 阶段就被排除,不再每个都浪费 materialize+编译。
- 厂商库许可(F2/P7,candidate_contract.md:201-208)、per-precision trial 表(P6-A/B,report.py:293+923)、rescue 计数进报告(F7)、后端可切换且源码裁决(1ef142d)全部在位。

**下一步不该是"重构框架"**——闭环是好的。

---

## 二、仍然开放的确定性问题(按严重度)

### 严重(影响结果,非仅效率)

**S1. 低精度死值反复被抽,且 Loop A 会被误导去"修"一个非缺陷。**
L3:21 两轮 bf16 都 0-for-100 / 0-for-22(COMPUTE_DTYPE 与 GEMM_PRECISION 两个名字下),占 8.8% trial 预算(D1)。这是效率损失,但真正的严重点是**方向性**:bf16 在 L3:43 却是获胜精度(369✓/30✗)——同门同模型判决相反。这证明门是对的、不该做 dtype 禁令,但也意味着"哪个精度死"完全是任务属性,agent 无法先验知道。**修复(D1):FAIL→PRUNED 升级,按 (candidate,knob,value) 计,失败 N 次零成功后才升级**,绝不跨候选池化。这是唯一"改了搜索行为"的项,必须在两箱都空时统一上,并接受"之后的 run 与之前不可比"。

### 中等(误导性,非失败)

**S2. P6-C 未做:一个候选在 4 个精度上声明支持、实际只有 1 个能跑,报告只测 θ_best。**
final_reeval 仍是单一 θ_best(orchestrator.py:2194),per-precision **最优**未上报。L3:43 首轮 θ_best 有两个自己声明的精度根本跑不起来,这在报告里看不见。P6-A/B 已把 per-precision trial 表做出来了(缓解),但 C(报告每精度最优)未做。成本高、需设计,放后。

**S3. ieee `tl.dot` 无标定 yardstick(P9)。**
strict IEEE fp32 attention 只到 fp32 屋顶 18%,外部 CUDA 55-74%,调参无法弥补(Triton 该 shape 无硬件快路径)。但标定只测 `torch.matmul`(cuBLAS),所以 bottleneck 报告拿 cuBLAS 屋顶比 Triton ieee 候选,报出虚低百分比、诱导 agent 徒劳优化。影响小(这些任务不要求 strict ieee),成本低,防的是一类误导。

### 低(效率)

**S4. F5 prescreen 在多 kernel 候选上净负**(L3:21 −30.5 min,D5)。正确性半边完美(129/129 shared-memory trial 编译期拒绝),只是定价错在"per-kernel 而非 per-variant"。修复面已确认只剩两项:按 kernel 数缩放采样 + 只筛 shared 相关子网格(缓存跨扩展命中已证实,无第三项可捡)。

**P-D. Loop D 在 L3 上零执行,是墙钟调度问题不是 bug。** D 排在 Loop C 之后、且"墙钟到就不启动新 novelty 轮"(orchestrator.py:569-577)。12h 预算 + 10-14 候选的调参几乎必然在 D 被调度前耗尽。若要论文主张的四循环在 L3 上有证据,要么抬预算、要么给 D 保留配额——但这是实验设计决策,非缺陷。

---

## 三、下一步优先级(含 H 卡价值标注)

用户三个候选方向:扩大测试范围 / 支持 CUTLASS-CuTe / 其他改进。我的判断:

### 第一梯队:先做,低风险且解锁后续一切

1. **S1(FAIL→PRUNED 死值升级)** — 唯一确定+严重+泛化的搜索修复。两箱空窗统一上。〔H 卡价值:高。H 卡上精度选择空间更大(fp8 进场),死值浪费会放大。〕

2. **完成当前三任务第二轮**,拿到 ceilings-开/关(D4)、后端切换触发(L3:48 是 strict-fp32 GEMM-heavy,首个真正检验)、bf16-vs-fp16 的干净数据。这是"扩大测试范围"的前置——**在扩大之前先把已有任务的对照做完**,否则新任务只是增加样本不增加洞察。

### 第二梯队:H 卡测试的直接前置(建议现在就动,标进计划)

3. **阈值可移植性硬化** 〔H 卡价值:关键前置〕。`research-bottleneck-classification-and-portable-thresholds.md` 已指出:分类门限历史上是 A100/H100 硬编码常数,我们已改成自校准导出(标定阶段实测 p_c 再留余量)。**但这条链在 4090 之外从未验证过。** 换 H 卡前必须做一次 `doctor` + 标定的干净跑,确认:(a) 校准 schema 覆盖 H 卡新增的量(fp8/TMA/更大 shared);(b) device 块不再谎报(现在 Linux config 存在正是因为 Windows 块静默谎报 sm_120 → agent 被告知错卡)。这是低成本、纯验证,且是 H 卡测试不翻车的保险。

4. **后端识别与 profile 抽取已后端中立**(cuobjdump -res-usage 覆盖 CUDA/CUTLASS/CuTe,commit 6efa850;structural_signature 已含后端前缀 1a2d450)。这意味着**profiling 侧已经为 H 卡上的 CUTLASS 准备好了**——缺的只是生成侧(prompt + pitfalls + 每后端编译错误分类)。

### 第三梯队:CUTLASS/CuTe 生成支持 —— 有价值但应在 H 卡上做,不是现在

**明确结论:CUTLASS/CuTe 生成支持应推迟到 H 卡就绪,不在 4090 上做。** 理由是实测的,不是猜:

- **在 Ada(4090)上 CUTLASS 无优势**:外部团队自己的源码注明"dense TF32 tensor rate = FP32 SIMT rate";我方实测 tf32/fp16/bf16 的 Triton GEMM 已达屋顶 87-93%、与 cuBLAS 差 5% 内。CUTLASS 的价值(`research-kernelpro-borrowable-details.md`:1.23× over Triton)来自 **Hopper 的 CuTe**,不是裸 CUDA C,且那是 H 卡特性。
- **在 4090 上做等于无法验证收益**:写了 CUTLASS 支持,在这张卡上测不出它比 Triton 好,就是白做+无判据。
- **所以正确顺序**:H 卡到位 → 先做第二梯队的可移植性验证 → 再上 CUTLASS 生成(prompt/pitfalls/编译错误分类)→ 在 H 卡上用"同算法双边扫"验证 CUTLASS vs Triton 的真实比值(判据:>1.05× 才值得,`plan-backend-neutral-profiling-and-bottleneck-taxonomy.md:150`)。

**现在能为 H 卡 CUTLASS 预做、且在 4090 上就有意义的**:第二梯队的第 3、4 项(可移植性 + 后端中立 profiling),它们既是当下的正确改进,又是 H 卡的地基。

---

## 四、一句话建议

先上 S1(唯一确定+严重的搜索修复),跑完三任务第二轮拿齐对照数据,同时做阈值可移植性硬化(H 卡前置);**扩大任务范围排在"把已有任务对照做完"之后**;**CUTLASS/CuTe 生成推迟到 H 卡**,但现在把后端中立 profiling(已完成)和可移植标定验证(待做)作为它的地基先立好。不要在 4090 上写测不出收益的 CUTLASS。
