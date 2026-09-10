# 两臂可比性:共享路径逐字段一致,只有候选相关字段不同

**日期** 2026-09-11 · **box1 对照** `run-l3-43-20260911-053020` · **box2 处理** `run-l3-43-20260911-052630` · 均 L3:43,均启动于 `dceda97`

两臂各产出首份 `BOTTLENECK_CLASSIFIED` 后做的检查。**这条是整个对照实验的承重项**:若共享代码路径在两台机器上给出不同数字,最终那个差就不再归因于开关。

## 判据:哪些字段**必须**相同,哪些**应该**不同

`BOTTLENECK_CLASSIFIED` 的 evidence 全部由**同一份代码**算出 —— 两个开关管的是诊断**形态**(label/vector)与账本**渲染**,不是分类器本身。所以阈值和各个 basis 字符串必须完全一致;而 `kind` / `at_limit` / `gpu_ms` 不同是**合法的**,因为两臂调的是不同候选。

| 共享路径字段 | 结果 |
|---|---|
| `thresholds`(4 个标定阈值 + `calibrated`) | **一致** |
| `dram_pressure_basis` | **一致** |
| `compute_pressure_basis` | **一致** |
| `candidate_aten_basis` | **一致** |
| `cpu_over_gpu_denominator` | **一致** |
| `ridge_flop_per_byte` | **一致** |
| `arithmetic_intensity` | **一致** |

7/7 一致。阈值一致是先前把两台机器**共用一份标定**的直接结果(此前 `idle_frac` 差 2.51% vs 门限 2.35%,当时的处理是让分母相同而不是放宽容差)。

| 候选相关字段 | 对照 | 处理 |
|---|---|---|
| `kind` | `resource_limited` | `resource_limited` |
| `candidate_precision` | **bf16** | **bf16** |
| `compute_ceiling_used` | tensor-core (bf16), Triton-measured (above cuBLAS) | 同 |
| `gpu_ms` | 3.1862 | 3.2128 |
| `at_limit` | `spills=6`, `regs=255/255`, occupancy 16.7% (registers) | occupancy 21% (shared_memory) |

## 两个值得记下来的观察

**两臂独立收敛到同一精度**。都是 bf16,都被分类为 `resource_limited`,天花板都取到"Triton 实测高于 cuBLAS"的张量核 bf16 屋顶(G10 的取最大值在两台机器上一致生效)。

**但绑定的维度不同**:对照撞寄存器(255/255,6 次 spill,occupancy 16.7%),处理撞 shared memory(occupancy 21%)。这正是 S2 的每维判决要区分的东西 —— 同一个任务、同一精度、同样"资源受限"的标签下,**实际顶住的维度不同**。单一标签会把这两种情况说成一样。

**当前两臂成绩 3.1862 vs 3.2128 ms,相差 0.83%**,在实测 2.35% 噪声底以内。这不是最终结论(还没到 `final_reeval_ms`,而 `tuned_ms` 系统性乐观 1.5–6.7%),但方向上与 G9 的先验一致:结构化信息的边际收益可能饱和。

## 方法上的一条

比较用的是**解析后的模型字段**而不是文件 diff,和 pre-flight 里两臂配置的比法一致。文件 diff 会把注释和键序算进去,而承重的是解析后进入判决的那些值。
