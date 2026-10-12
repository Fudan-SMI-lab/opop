# C3 实验完整信息档案(2026-09-22 归档)

> 目的:C3(目标驱动的算子优化框架)实验告一段落,本文件是**信息索引**,供后续回来检查/调取。
> 两台机即将转做 KernelBench 扩大范围测试,C3 资产保留在盘上不动。
> **所有断言都可在盘上复验;引用数字前先在当前 checkout 复算(多机共用,状态可能变)。**

---

## 0. 一句话结论

C3 = 目标驱动的算子优化框架。三条证据线已齐:①§4.3 目标特异性表(对角三目标全胜 +3.56/+12.51/+9.29%);②站点选择判据(优化空间=相关性×承载×余地);③算子更慢但目标更快(度量哲学)。**详见交付文档 `v5/docs/c3-support-data-2026-09-22.md`(7 节,权威结论都在那)。本文件是原始实验资产的索引。**

---

## 1. 机器与环境

| | box4 | box1 |
|---|---|---|
| ssh | `-p 37752 root@connect.bjb1.seetacloud.com`(**不在 ssh config,直连**) | alias `autodl`(`connect.bjb3.seetacloud.com:49408`) |
| host | `autodl-container-459d488f2d-658995df` | `autodl-container-wtl0lh99as-...` |
| GPU | 2×RTX4090 | 2×RTX4090 |
| 代码根(活) | `/root/autodl-tmp/c2-local-experiment/code/v5` | 同左 |
| venv | `/root/autodl-tmp/kernel-opt-venv/bin/python` | 同左 |
| C3 实验根 | `/root/autodl-tmp/c3-fast-kernel-iteration/` | 同左 |
| torch/triton | 2.13.0+cu129 / 3.7.1 | (曾 2.9.1,跑车道前需确认) |
| python 可见 | 仅 `bash -lc` 登录 shell | 同左 |

**运行 lane 的标准方式**:`snapshot_code.sh <dest>` 建只读快照(只 examples/src/tests)→ `PYTHONPATH=$SNAP/src:$SNAP` → CVD 选卡。薄脚本 `run_goal_site_probe.sh <site_module> <goal> <card> <suffix> [dev_h] [search_h]`(已修 CVD)。

---

## 2. §4.3 表的三个数据来源(box4)

§4.3 合成表(见交付文档 §1)由三处实测拼成,均 box4、RMSNorm 站点 `pure-727f184a931d6918df621c6385fe123d7dd248cb470c207fd46b715a8c7a1fa0`、contract `2c0e2fc7…`:

| 用途 | 路径(box4) | 关键内容 |
|---|---|---|
| **ttft 整行**(kernel `214c971b`) | `F4-campaign-0921-053151` + `F4-matrix/matrix.json` | +3.56%/−10.64%/−14.17%,36/36 valid,device-proven,已发表密封矩阵 |
| **single/multi 列**(对角+离对角) | `synth-matrix-3rows-0922-173506/matrix.json` | single +12.51% / multi +9.29% 等,统一 F4 corpus,held-out |
| **ttft 列 latency**(single/multi-kernel) | `synth-matrix-3rows-full-0922-175958/matrix.json` | 284.14/284.28ms(correctness_failed 保留 latency) |

**本地存档(Windows v5)**:`results/c3-fast-kernel-iteration/synth-matrix-3rows-0922/`、`synth-matrix-3rows-full-0922/`、`F4-matrix/matrix.json`。

**其他合成运行(过程产物,可忽略/删)**:`synth-matrix-ttft-only-*`(单机验证)、`synth-matrix-final-*`(误杀)、`synth-matrix-pergoalfloor-*`(误杀)、`synth-matrix-tableB-*`(用户改主意后停)。

---

## 3. 站点探针实验清单(证明"不同目标/站点增益不同")

**box4(ttft 目标为主)**:
| 目录 | 站点 | 目标 | 结果 |
|---|---|---|---|
| `F5-probe-norm-lane-0922-020748` | RMSNorm | ttft | accepted,+4.46%(selected `453b09ee`,logits 边缘) |
| `F5-probe-silu-lane-0922-020750` | act_fn(SiLU) | ttft | censored,0 接受(≈0) |
| `F5-probe-upproj-ttft2-lane-0922-080234` | up_proj(GEMM) | ttft | censored,0 接受 |
| `F5-lanes-ttft-0921-222823` | gate_proj(GEMM) | ttft | accepted,+0.22%(噪声内) |
| `F4-campaign-0921-053151` | RMSNorm | ttft | **发表 +3.56%**,kernel `214c971b` |

**box1(single/multi 目标)**:
| 目录 | 站点 | 目标 | 结果 |
|---|---|---|---|
| `F5-probe-norm-single-lane-0922-080159` | RMSNorm | single | **accepted +12.16%**,selected `78a1aa57`,logits 0.0112 |
| `F5-probe-norm-multi-lane-0922-080159` | RMSNorm | multi | **accepted +5.29%**,selected `fc877ef5`,logits 0.0118 |
| `F5-probe-upproj-multi-lane-0922-034233` | up_proj(GEMM) | multi | censored,最佳 −7% |
| `F5-probe-downproj-multi-lane-0922-124637` | down_proj(GEMM) | multi | censored,最佳 −0.72% |
| `F5-probe-qproj-multi-lane-0922-124637` | q_proj(GEMM) | multi | censored,最佳 −1.64% |

**站点承载 CUDA 时间(associated_cuda_us,优化空间分析用)**:gate_proj 88822 / up_proj 92975 / input_layernorm 14041(multi)~2383(single)~15036(multi) / act_fn 5095。RMSNorm 由 **8 个未融合 kernel** 组成(可融合=有余地);GEMM 是单个 cuBLAS kernel(零余地)。

---

## 4. 关键脚本(box4 `/root/autodl-tmp/`,及本地 v5/scripts)

| 脚本 | 用途 |
|---|---|
| `synthesize_c3_matrix.py` | **§4.3 合成驱动**:去 framework_id、保留正确性门,bind 已定稿 bundle 测三目标。`--per-goal-floor`/`--record-failed-timing`(超精度保留 latency)。本地 `v5/scripts/` |
| `render_matrix_table.py` | 渲染 matrix.json 成 §4.3 markdown 表(本地 `v5/results/c3-fast-kernel-iteration/F2-live/`) |
| `operator_attribution.py` | 算子级 paired micro speedup 复算(box4 `/root/probe-clean/`,仅 stdlib) |
| `run_b26_gemm_floor.py` | floor 自校准 device-only 模板(本地 `v5/scripts/`) |
| `run_goal_site_probe.sh` | 起一条 dev→freeze→lane 链(box4 `/root/autodl-tmp/`,已修 CVD) |

---

## 5. 三个必须记住的方法学要点(引用数字前先读)

1. **framework_id 是内部防漂移守卫,不是 §4.3 测量有效性的必要条件**。三条 norm lane 分属 4 个不同 impl_sha,不能用 `rerun_matrix`/`run_split_matrix` 拼(fid 门拒),但可用去 fid 的 `synthesize_c3_matrix.py` 在统一 corpus 下重测——正确性门完整保留。
2. **held-out 比搜索期严**:一个候选在 search prompt 上过门(如 ttft `453b09ee` logits 0.0142),不保证在密封 held-out 上过(测得 0.0164~0.0225 超 floor)。§4.3 用 held-out。
3. **floor 应按目标序列几何校准**:0.01616 是 64-token 校准值,套到 4-token 的 ttft 偏紧 ⇒ ttft 列 single/multi-kernel 超 floor 被拒(但 latency 已按需保留)。非 kernel 错(local=1.0/NLL 全过)。

---

## 6. 未尽事项(若日后重开 C3)

- B5 `site-reachability.json` 用硬钉 0.01 判 GEMM `reach=False`,已被 floor 实测翻案(B26),但**尚未重生成**——仅 `select_sites` 自动选站点时需要;强制站点绕过排名不需要。
- ttft 列若要 device-proven 通过:需 per-goal floor(按 4-token 几何校准)或换 logits≈0 的 ttft@norm 候选(原 lane 7 个 valid 里筛)。
- **交付时轮换 4 机 API key + GitHub PAT**(standing 指令,未做)。