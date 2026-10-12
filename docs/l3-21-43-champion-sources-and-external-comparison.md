# L3:21 / L3:43 冠军源码定位与外部对比(2026-09-23 备忘)

> 2026-09-22 核验完成。回答:两个冠军的源码是否还在、记录的性能是多少、能否比过外部框架最优。

## 1. 源码位置(已逐一核验身份)

| 冠军 | 位置 | 备注 |
|---|---|---|
| **L3:43 `cand-173b12ae`**(2.7628 ms) | 本地 `external_files/box2-runs/runs-l3/run-l3-43-20260909-015247/`(`candidates/cand-173b12ae/source.py` + `report/best_kernel.py` + manifest/events/jobs) | ⚠ box2 已释放,**这是唯一副本** |
| **L3:21 `cand-8a5dcef3`**(3.4545 ms) | box1 `/root/autodl-tmp/opop-workspace/opop-glm/runs-l3/run-l3-21-20260908-232211/`;已于 2026-09-22 镜像到本地 `external_files/box1-runs/runs-l3/run-l3-21-20260908-232211/` | 冠军 candidates 目录 + `report/best_kernel.py` + report.md |

身份核验方式:`best_kernel.py` PARAMS 逐项对上结果文档记录(L3:43:ATTN 128×64/8 warps、GEMM 64×256×32/8 warps、bf16+fp32 acc;L3:21:fp16、BM1=128),且 run 内 `report.md` 的 Best result 段与 git-tracked 文档(`docs/result-l3-21-3p45ms-clock-ended.md`@062f8e8、`docs/result-l3-43-r2-2p76ms-bf16.md`@14fab15)一致。

两个曾经的疑点已排除:
- `results/c2-method-improvements/A/terminal-task43/{G0,H}/selected.py` **不是冠军**(GEMM tile 128×128×32 ≠ 冠军的 64×256×32);其 `"source_fidelity": "failed"` 是 `scripts/experiments/c2_method_a_terminal.py:78` 的 schema 常量(该实验类按设计不保真源码),不是损坏告警。
- 仓库 tracked 的 `results/l3-21-20260905-195615/best_kernel.py`(7.21 ms)是**旧**冠军,已被 3.4545 ms 取代。

## 2. 记录的性能(final re-eval 中位,文档与 run report 两处一致)

| | L3:21 | L3:43 |
|---|---|---|
| 延迟 | **3.4545 ms** | **2.7628 ms** |
| vs eager | 4.46x(15.6 ms) | 7.79x(21.5 ms) |
| vs 同精度基线 torch_compile_tf32 | **3.26x**(11.4 ms) | **3.99x**(11.0 ms) |
| 精度 | fp16 | bf16 + fp32 acc |
| 正确性 | fp64 rescue 3/3(承重) | fp64 rescue 3/3(承重) |

## 3. 与外部框架最优对比(同卡 4090、绝对 latency,规避各家 eager 基线不共享)

| 任务 | 我方 | 外部最优 | 判定 |
|---|---|---|---|
| L3:21 | **3454 μs** | 4410 μs(K-Search,其基线偏低 6.6%)/ 4980 μs(Foundry 稳健值) | ✅ 领先 22–31% |
| L3:43 | **2763 μs** | ~8685 μs(Shinka;K-Search 声称的 21.4x 已排除,理由见下更正) | ✅ 领先 **~3.1 倍** |

> ⚠ **2026-09-23 更正(逐源码核查后)。** 上表 L3:43 一行原写 K-Search 的 21.4x 是
> "多流未同步计时假象" —— **该判定不成立,已撤回**。实际读它的
> `source/candidate.py` 后确认:它的双流同步是**正确的**
> (`current_stream().wait_event(ev1/ev2)` 让默认流等两条支流,CUDA event 计时覆盖全部工作);
> 它也**不是**纯 fp32(用 fp16 hi/lo 喂张量核),并**确实**做了块级 causal 跳过。
>
> 排除它的真实理由是**它自己声明的精度路径与该延迟不自洽**:docstring 与
> `recorded_settings.json` 都声明 c_attn/c_proj 为 "full fp32 cuBLAS (IEEE)"、`precision: fp32`
> 且源码无 `.half()`,而这两个投影合计 309.24 GFLOP —— 在 4090 上即便放宽到 TF32 张量核
> 也需 **1.872 ms**(IEEE fp32 需 3.744 ms),都 > 1.000 ms。
> 完整论证见 `docs/paper-contribution-evidence-map-2026-09-23.md` 的「必须随表携带的口径」第 7 条。
>
> 另一处更正:本文第 2 节的性能数字有效,但 `external_files/box2-runs/.../report/report.md` 里
> 有一句 analyst 文本把任务形状写成 "B=16, T=2048, C=1024, 16 heads" —— **那是笔误**。
> harness 报的 412.317 GFLOP 唯一对应 **B=128 / T=512 / C=768 / nh=8**,与外部包一致,形状可比。

## 4. 注意事项

1. 两个冠军的正确性**完全靠 fp64 相对门通过**(run report 自标)。用户已裁定后续实验沿用我们自己的门为标准;但论文中做逐候选公平对比时须如实标注 rescue 承重。
2. L3:43 冠军 `candidates/.../source.py` 字节 sha(`8ab5ac11…`)≠ 注册 sha(`415416b1…`)是**预期的**——注册身份在 generator sandbox,candidates 下是参数化后版本;交付物以 `report/best_kernel.py`(物化 winner)为准。
3. box1 若释放,L3:21 冠军也只剩本地镜像;两个镜像目录都不要清理。
