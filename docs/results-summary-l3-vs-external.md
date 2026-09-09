# L3 实验结果汇总 — 本框架 vs 外部框架(kernelfoundry MAP-Elites 复现)

数据来源:本框架三个已完成 L3 run 的 `events.jsonl` / `reeval_result.json`(磁盘原始记录,非通知转述);外部数据来自用户提供的外部测试人员表格与 `external_files/l3_best_candidates_21_43_48_20260908/manifest.json`;同卡复测数据来自 `docs/finding-external-cuda-vs-our-triton.md`(box 2,RTX 4090,双方走同一 job builder、同一正确性门、100 次 CUDA-event 采样)。

**跨表比较须知**:两表的 ms 值来自不同机器(均为 RTX 4090,但时钟/驱动/torch 环境不同——同任务 eager 差 10–17%),**绝对 ms 不跨表可比,加速比才是可比货币**;唯一严格可比的是"同卡复测"列(双方 kernel 在我们 box 2 上同门同计时实测)。

## 主表(本框架结果)

| 任务 | eager(ms) | compile 全 FP32(ms / 加速比) | compile 允许 TF32(ms / 加速比) | 本框架最佳(ms / 加速比) | 与外部结果对比 | compile 配置注释 | 必要超参设置 |
|---|---|---|---|---|---|---|---|
| L3:21 | 15.581184 | 14.154176 / 1.101× | 11.386880 / 1.368× | **3.454464 / 4.510×**(vs eager)<br>3.296×(vs compile-TF32,同精度判定基线)<br>Triton,fp16 计算 fp32 累加,5/5 正确性(3/3 经 fp64 相对门,ratio<1 即比参考更准) | 外部最佳 4.785798 / 3.680×(他们的卡)。**同卡复测(box 2)**:外部 CUDA 4.833 ms vs 我方当时最佳 4.414 ms → 我方 1.09×;manifest 宣称 3.67×,同卡实测 3.22×。本表 3.454 为 box 1 最新 run(跨箱同型号卡),较外部同卡值约快 1.40× | 共同:`torch.compile(backend="inductor", mode="default")`,`fullgraph`/`dynamic` 均为默认值(未显式设置),不启用 autocast;计时 = CUDA event,100 采样取**中位数**,不含编译/预热,基线与候选同进程测得。<br>全 FP32:`torch.backends.cuda.matmul.fp32_precision="ieee"` + `torch.backends.cudnn.conv.fp32_precision="ieee"`(torch 2.9 API,同时覆盖 matmul 与 cudnn 卷积;旧 torch 回退 allow_tf32 双旗标)。<br>允许 TF32:同一 API 置 `"tf32"`。 | 输入 = 10×112×224×224;out_channels=192、kernel_size=5、stride=2、expand_ratio=6;**train 模式**(BN 用 batch 统计);KernelBench pin `423217d`;RTX 4090(box 1) |
| L3:43 | 21.451296 | 13.824528 / 1.552× | 10.985984 / 1.953× | **3.012608 / 7.121×**(vs eager)<br>3.647×(vs compile-TF32)<br>Triton,fp16,5/5 正确性 0 救援 | 外部最佳 5.700033 / 3.218×(他们的卡)。**同卡复测(box 2)**:外部 CUDA 8.675 ms(strict fp32)vs 我方 3.0126 ms → 总体我方 2.88×;但**同精度分档**:strict-IEEE 外部胜 1.64×(8.893 vs 14.565),tf32 打平(1.005×),fp16/bf16 外部未实现。manifest 宣称 3.18×,同卡实测 2.47× | 同上 | 输入 = 128×512×768;n_head=8、max_seqlen=1024、attn/resid dropout=0;train 模式;KernelBench pin `423217d`;RTX 4090(box 2) |
| L3:48 | 20.7 | 13.7 / 1.511× | 13.1 / 1.580× | **1.41 / 14.681×**(vs eager)<br>9.291×(vs compile-TF32)<br>Triton,fp16;5/5 正确性**全经 fp64 相对门救援** | **任务参数与外部表不同(batch 2048 vs 16),ms 与加速比均不可直接比,外部规格另起一行见下。**外部 kernel 在**我方规格(batch=2048)同卡实测 1.477 ms**(13.99× vs 我方 eager)vs 我方 1.411 ms → 1.05×,且对方 0 救援直接过主门——**应报平手,不宣称胜**;manifest 的 99.2× 在我方规格同卡只再现 13.99× | 同上;注:此 run 早于 median 修复,基线与 final 为 100 采样**均值**(该箱当时全链均值,基线极紧故加速比不受损,见 memory `opop-selection-chain-is-all-mean`) | 输入 = **2048**×128×8×64;d_state=16、block_len=64;train 模式;KernelBench pin `423217d`(该 pin 的 batch_size=2048);RTX 4090(box 1) |
| L3:48(外部规格) | 0.954896(外部卡) | 0.393304 / 2.428× | 0.399250 / 2.392× | —(本框架无此规格 run) | 外部最佳 0.008918 / 107.076×,**batch=16**——问题规模是我方规格的 1/128;0.0089 ms ≈ 8.9 µs 已低于我方 4090 实测空启动地板 17.4 µs,该数字未经 GPU0 复测(外部 README 自注),不应在任何对比中引用 | 外部自述:`fullgraph=True`、`dynamic=False`、旧接口 `allow_tf32` 双旗标 | 输入 = **16**×128×8×64;d_state=16、block_len=64(与我方 KernelBench pin 的 batch_size=2048 不一致,来源规格差异) |

## 数字口径说明

- **本框架"最佳"一律为 `final_reeval_ms`**(独立新进程 5/5 正确性 + 100 采样复测),绝不引用调参循环内的 `tuned_ms`(系统性偏差 −4.2%~+6.7% 双向皆有实证)。L3:21/L3:43 为中位数,L3:48 首 run 为均值(当时该箱全链均值)。
- L3:43 的 3.0126 来自主动终止后的手工独立复测(`reeval_result.json`,5/5 正确,中位数);该 run 无 RUN_FINISHED。
- 我方另有 eager-TF32 基线(L3:21 13.864 / L3:43 18.241 / L3:48 20.1 ms),外部表无此列,故未列入主表。
- 加速比 = 本表内同 run 同进程基线之比;外部表加速比 = 外部自身基线之比。
- 外部三个 `.cu` 均经 sha256 对 manifest 校验后在 box 2 复测;两侧用同一正确性门(relaxed 双见证 + fp64 相对门)与同一计时路径。

## 进行中(本表不含,完成后更新)

- **L3:43 第二轮**(box 2,首个天花板可见 + 后端可切换版本):~570/720 min,最优 tuned 2.8221 ms 已优于上表 3.0126——**终值须等它自己的 final_reeval**。
- **L3:48 第二轮**(box 1,commit `062f8e8`)2026-09-09 11:57 启动,incumbent 1.41 ms。

## 结论速览

三任务在**同卡、同门、同计时**下,总体延迟我方 2 胜(L3:21 1.09×、L3:43 2.88×)1 平(L3:48 1.05× 但对方正确性余量更大)。我方总体优势主要来自**精度档位的利用**(fp16/tf32 张量核,外部 L3:43 kernel 因错误前提弃用 TF32);外部真正胜出的档位是 strict-IEEE fp32 attention(CUDA 1.64×,Triton `input_precision="ieee"` 无硬件快路径,调不掉)。外部 manifest 的搜索期加速比(3.67×/3.18×/99.2×)在同卡复测下分别为 3.22×/2.47×/13.99×,引用时须注明。
