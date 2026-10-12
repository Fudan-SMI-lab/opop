# 论文数据资产总目录：已有结果、可用主张与候选位置

**已有 paper 可用 C2 正面数据；当前 C3 device 目标未达成不否定旧 C2。** 最优先使用 T4 的 G0/G1/G2 固定状态信息对照：四父状态 × 三组，共 12 次 fresh proposal，G2 对 G0 为 4/4、对 G1 为 3/4 通过预声明观测门槛。这是有限状态的信息增量观察证据，不是总体因果证明。旧 C3 也有有效的模型级部分结果，不能被当前 fast iteration 的失败覆盖。

核对日期：2026-09-20。本文回答“产出了什么、关于什么、论文能怎么用、源码和原始数据在哪”。只整理既有证据，不重测、不运行分析代码、不访问服务器、不修改旧报告或原始记录。所有远端位置均为 **historical receipt，not live verified**，不是当前在线、文件仍在或 GPU 可用的承诺。

## 1. 先选论文材料，再看总量

| 优先级 | 资产 | 最适合支持什么 | 必须同时呈现的限制 |
|---|---|---|---|
| P1 主文 | C2 T4，四父 G0/G1/G2 | 完整 fresh 资源响应在本批固定状态提供正向信息增量信号 | task43/P1 的 G1 明显更好；range 不是 CI；所有 36 child final blocks 用 FP64 rescue |
| P1 主文 | C2 T5，共同 elapsed 比较 | 信息增量不自动转为短闭环净时间收益，2/2 未胜 | G2 相对自身父仍改善；T6 12h 未运行 |
| P1 主文或主消融 | method/A、opportunity、region、helper pilot | 方法迭代中的正负异质性、强基线下的边界 | method/A task21 +41.559%，task43 负；opportunity censored；pilot 仅 1/4 胜 |
| P1 性能表 | 历史 L3:21 同卡 external 对照、L3:43 round-2、A800 E3 | 具体冻结候选在指定 workload/质量策略下的性能 | 混合精度与历史 gate 必须披露；不是等预算搜索优越性 |
| P2 机制 | 固定点、native-retune、unlocked-region、task21 配置诊断 | 资源下降不必然加速，空间可达性、参数和结构交互 | 复用既有结构，不是新生成重复；不能事后改变原 gate |
| P2 C1 观察 | S2 paired、step-3、window-1 corrected closeout | 结构化反馈关联的轨迹、预测与测量身份问题 | 跨 host、双开关或不等信息；不能隔离 ledger 因果效应 |
| P2 模型级部分结果 | 旧 C3 Qwen3-4B 恢复矩阵 | 一份 Python decoder composition 源改写的 single/multi 描述性提升 | 不是新 GPU kernel，不是三目标分别成功；TTFT 差异在波动内 |
| P3 失败与实现边界 | 当前 C3 fast F0/F1、受污染 method/core | 框架准入、污染、计费、缺失结果的诚实报告 | 不得当方法效力负分；control/replay 不等于 agent 成功 |

分类按**具体主张**而非整个目录：`primary-bounded-claim` 为有明确对照的有界主结果，`secondary-mechanism` 为机制/测量证据，`exploratory-inconclusive` 为探索、混杂或缺失，`invalid-exclude` 排除无效数值解释或效力推断。负结果也能是主结果；有效 kernel 的不成立因果解释应排除，但不用删除该 kernel。

### 1.1 不能诚实给出一个“独立实验总数”

历史清单找到 75 个 `docs/result*.md` 文件，其中 12 个 C2、2 个 C3、61 个其他结果文件。**75 是文件覆盖数，不是数据集或独立实验数。** 一次 run 可有进度、终测、纠错、源码审计等多份报告；同候选可以被多次测量；resume 不是 fresh replicate。

| Campaign / 子阶段 | 已产生的单位与数量 | 重用和不可相加项 |
|---|---|---|
| 首轮 C2 单步 | 两固定父 × A/B，4 次 proposal；有效 trial 97、失败 55 | 一次/臂；43/A 静态拒绝仍是机会失败 |
| 首轮固定点复测 | 2 结构 × 2 storage × 3 blocks = 12 full | 复用首轮 task21 两结构，0 新生成 |
| native-retune | 2 结构 × sampler seed 0/1/2，6 B40；240 records；18 final blocks | 12 witness reuse；不是 6 次 C2 生成 |
| T4 信息阶段 | 4 父 × 3 组 = 12 fresh proposals；480 records；72 full blocks | 四父 acquisition 共 48 次，只计一次；24 cached witnesses |
| T5 短闭环 | 4 arms × 2 轮 = 8 opportunities；320 records；48 full blocks | 原 P1 重新生成，不接 T4 winner；16 witness reuse；T6 NOT_RUN |
| unlocked-region | E1/E2 各 B40；3 固定点；7 产物统一卡 21 full | 0 新生成；旧 winner 新测，不能复用旧 latency 充数 |
| method/core | 16 机会，11 完整、1 failed、4 invalid；440 asked；60 full | staging 污染，效力与 intent 消融排除 |
| method/A | 4 两轮轨迹，8 机会，7 完整 B40、1 拒绝；280 asked；45 轨迹 full + 12 terminal full | 同获批程序的修复阶段；不是独立确认研究 |
| opportunity | 8 fresh 机会；13 studies，520 asked；48 screen + 18 heldout full | rep1 两 pair 共 18 规定 blocks 未测；7 child incumbents |
| region follow-up | rep1 新测 18 full + 6 static；helper 集成 4 full + 4 static | 固定旧候选；23-session 自测审计只是衍生分析 |
| helper pilot | 8 fresh 机会；13 studies，520 asked；48 screen + 36 heldout full | 四 pair 全 resolved；usage 的 20 QUICK 不是 20 新候选 |
| task21 配置诊断 | 6 cells × 3 = 18 full，18 static，2 instrumented traces | 0 模型调用、0 调优、0 新生成 |
| 历史主要组 | 早期 H1 四 run；S2 两 run；E1 两 run + E3 一 run；step-3 三 run；window-1 四 run | 只是已定位组的单位，不是完整历史总表；快照、复测和 corpus 重叠 |
| 旧 C3 | 6 fresh 结构机会，5 失败、1 accepted；原矩阵 36 null，恢复矩阵 36 valid | 4 rows × 3 goals × 3 blocks；fallback rows 同源码，不是 4 独立方案 |
| 当前 C3 fast | 2 fresh probes，1 同产物诊断；开发 2 agent / 8 local / 0 model-evaluation charges | setup 另计；formal/heldout 未运行，不是 36 个零分 |

trial、candidate、space、生成机会、correctness input、性能 sample、block、run 是不同单位。一般 full=5 correctness+100 timing samples，3 blocks 是测量重复，不是 3 次独立生成。T4 的 7200 samples 也不是 7200 个优化任务。

## 2. 根目录、服务器与证据层

### 2.1 本地和远端注册表

本地项目根 `L = D:/Pyhon_projects/opop/v5/`；下文 `docs/`、`results/`、`.omo/`、`.deliver/` 均相对 L。外部镜像根 **`E = D:/Pyhon_projects/opop/external_files/`**，是 v5 的兄弟目录，绝不是 `L/external_files/`。

| 身份 | 当时入口 / hostname | GPU、软件与路径 |
|---|---|---|
| 同期 HOST-A / A | `root@connect.bjb1.seetacloud.com:37752`；`autodl-container-459d488f2d-658995df` | 2×RTX4090 24564MiB；Xeon8352V；driver580.105.08；worker `/root/autodl-tmp/kernel-opt-venv/bin/python` |
| 同期 HOST-B / B | 当时 `autodl` 为 `root@connect.bjb3.seetacloud.com:49408`；`autodl-container-wtl0lh99as-5b475d7c` | 2×RTX4090 24564MiB；Xeon8358P；driver580.142；worker `/root/autodl-tmp/orch-venv/bin/python` |
| 历史 box1 | alias `autodl`；不能用今天 alias 反推所有旧 run | 历史 4090；window-1 容器与上述 B 一致，但不声称每个旧批独立验证了该入口 |
| 历史 box2 | alias `autodl2`；`connect.bjb2.seetacloud.com:22010` | RTX4090，Xeon8358P；主要 `runs-l3` 及 external 同卡复测 |
| 历史 box3 | alias `a800`；`connect.nma1.seetacloud.com:46901` | A800 80GB PCIe，sm80，108SM；Xeon6348；Torch2.8.0+cu128/Triton3.4.0 |
| 历史 box4 | window-1 容器 `autodl-container-459d488f2d-658995df`，该收据记录端口37752 | 双4090；只凭 CPU 型号不能唯一证明所有历史身份 |
| 外部 CUDA 原作者机器 | 未给主机/端口 | `/home/kangtian/...` 是外部来源，不得归到 autodl2 |

A GPU0/1 UUID：`GPU-c896ae45-532b-2a1d-147d-ed72e221fa42` / `GPU-682bb59c-17a7-c7bc-2be8-83842d38f2f8`。B GPU0/1：`GPU-9c819228-dcf4-f1da-f21e-6b77b1c58be1` / `GPU-cdff4e6f-7941-db17-7ff1-da75c7d90b24`。同期 C2 worker 为 Torch2.13.0+cu129/Triton3.7.1；C3 另有 Transformers5.16.1。

入口依据 `docs/verification-c2-local-experiment-server-handoff.md:31至75`，以及 `.omo/notepads/v5-c2-method-improvements/server-preflight.md`、`.omo/notepads/v5-c2-opportunity-driven/server-preflight.md`、`.omo/notepads/v5-c2-helper-pilot/server-preflight.md`。region 的补测没有另一次外部 SSH 映射核验；其 localhost 42360/42380 是服务设置。pilot 42510/42511、42520/42521 也不是 SSH 端口。

| 简写 | 远端绝对根，host 必须另带 | 本地对应入口 |
|---|---|---|
| R | `/root/autodl-tmp/c2-local-experiment`，A/B 各自独立 | 首轮、retune、T4、T5、unlocked 分别在不同 `results/c2-*` |
| M | `/root/autodl-tmp/c2-method-improvements` | `results/c2-method-improvements/` |
| O | `/root/autodl-tmp/c2-opportunity-driven` | `results/c2-opportunity-driven/` |
| F | `/root/autodl-tmp/c2-rep1-heldout-followup` | A→`results/c2-rep1-heldout-followup/task21/`；B→`.../task43/` |
| I | `/root/autodl-tmp/c2-formal-selftest-integration`，B 为测量主机 | `results/c2-formal-selftest-integration/`；`host-A/` 仅部署保全 |
| P | `/root/autodl-tmp/c2-helper-pilot/main` | `results/c2-helper-pilot/main/`；按 A/B 导出 manifest 定位，勿盲换前缀 |
| D | `/root/autodl-tmp/c2-task21-config-diagnostic`，B0 | `results/c2-task21-config-diagnostic/` |
| Q | `/root/autodl-tmp/c3-qwen3-4b-operator-goals`，A/B | `results/c3-qwen3-4b-operator-goals/`，导出额外带 host/recovery 层 |
| K | `/root/autodl-tmp/c3-fast-kernel-iteration`，fresh probes 在 A0 | `results/c3-fast-kernel-iteration/`，同样非简单镜像 |

同期代码根为 `R/code/v5`；历史代码根可为 `/root/autodl-tmp/work/opop`。代码 checkout、run root、输入根不是同一目录。相同绝对路径在两台服务器上也不是同一份数据。

### 2.2 找到的是哪一层产物

| 层 | 用途 | 不能混同 |
|---|---|---|
| parameterized source / `source.py` | 可调结构及默认参数 | 默认不等于 best params；历史 source 可能被覆写 |
| selected / materialized trial | 已选配置的执行代码，通常 `selected.py` 或 `trials/<tr>.py` | native winner 不一定晋级为 incumbent |
| `result.json` 的 selected/selected_trial | candidate、space、trial、完整 params 与搜索分数 | 搜索分数不是 fresh final / heldout |
| final/heldout result + event + worker `.out.json` | 正式质量、block medians、raw samples | 聚合不是原始数组；100 samples 不是100重复研究 |
| manifest / export-manifest / setup | 源、reference、配置、SHA、远端到本地映射 | 已有 hash 收据不等于本次重算全量 hash |
| generation events / `sessions.json` | call、session 与候选来源链 | session ID 不等于 run ID，`generation` 也非全局唯一 |

公开层为报告及固定 revision 的代码，URL 可写 `https://github.com/Fudan-SMI-lab/opop/blob/<报告SHA>/docs/<文件名>`。`results/` allowlist、`.omo/`、本地镜像、完整 sessions/数据库/模型均不是已公开数据集。不要为补齐配置而复制私有 provider YAML 或整个 sandbox。

## 3. C2 首选主证据：T4 G0/G1/G2

权威数值：`docs/result-c2-information-stage.md:23至99`；定位与协议：`results/c2-information-groups/REPORT.md`、`docs/c2-information.md`。执行 generation SHA **`519d0374d04cbcca2a9d506194b8b40c72eb48e8`**，acquisition SHA **`dca5c21ba08ffc930a644c92b187b5b78b703d12`**。2026-09-17 04:01:02 至 06:52:36 UTC 的主阶段不含之前 acquisition/准备。

G0 为普通历史信息，包含已有资源信息，但没有 fresh endpoint brief。G1 增加与 G2 **同批 acquisition** 的 endpoint params、validity、J、delta-J/gain，屏蔽 fresh metrics/detail/resource deltas/slopes/unknown-resource 列表及资源敏感原因。G2 收到完整原生 TaskResponse envelope；“完整”不等于所有轴测完。G1 空 metrics 是未提供，不是资源为零。

四父按注册顺序取前两个合格 off 状态，不按新 child 表现挑选：A/task43 P1=`cand-b67a1cb4/tr-6fb84baa` cutoff617，P2=`cand-4e9c333d/tr-19d4ad9c` cutoff345；B/task21 P1=`cand-4c96b8c4/tr-6afd5e56` cutoff960，P2=`cand-03df0141/tr-306cfc41` cutoff1202。

### 3.1 全部 12 行结果与可直接拼接的候选位置

下表 P/C 为各自三个 fresh full block median 的中位数，单位 ms；R=C/P。**不是搜索 best，也不把 300 样本合池。** 每行一次 fresh analyst→rewriter→parameterizer、一个 native B40、sampler seed0。12 行均 selected child，无 fallback。

| Host/task | GPU | 目录键 | Candidate ID | P ms | C ms | R |
|---|---:|---|---|---:|---:|---:|
| A/43 | 0 | P1-G0-519d037 | cand-37e51a4a | 3.492864 | 3.404800 | 0.974787464 |
| A/43 | 1 | P1-G1-519d037 | cand-aad040f7 | 3.431424 | 2.759680 | 0.804237573 |
| A/43 | 0 | P1-G2-519d037 | cand-a40c5d04 | 3.493376 | 3.400704 | 0.973472082 |
| A/43 | 1 | P2-G0-519d037 | cand-a909683f | 3.338240 | 2.712576 | 0.812576683 |
| A/43 | 0 | P2-G1-519d037 | cand-f5a66b0c | 3.406848 | 2.705408 | 0.794108846 |
| A/43 | 1 | P2-G2-519d037 | cand-222393b5 | 3.337712 | 2.602496 | 0.779724578 |
| B/21 | 1 | P1-G0-519d037 | cand-f39709db | 8.466944 | 4.671488 | 0.551732473 |
| B/21 | 0 | P1-G1-519d037 | cand-401f1f58 | 8.365568 | 5.075968 | 0.606769043 |
| B/21 | 1 | P1-G2-519d037 | cand-c2268b83 | 8.465920 | 4.100096 | 0.484306046 |
| B/21 | 0 | P2-G0-519d037 | cand-437adc70 | 9.037824 | 4.211744 | 0.466013056 |
| B/21 | 1 | P2-G1-519d037 | cand-f4be3682 | 9.043968 | 4.236880 | 0.468475754 |
| B/21 | 0 | P2-G2-519d037 | cand-f4b4d40a | 9.037824 | 3.529728 | 0.390550653 |

**每行精确检索公式：** 本地 `L/results/c2-information-groups/HOST-<A或B>/<目录键>/`；远端在该行 host 的 `R/runs/information-groups/<目录键>/`。因此无需另找 candidate 名称即可打开如下文件。

| 需要的材料 | 本地行目录内文件 | 远端原始材料 |
|---|---|---|
| 可调源 / 已选 child / 最终输出 | `child-source.py` / `child-selected.py` / `output-selected.py` | 每组 `retune-manifest.json.inputs.source` 给出生成源绝对位置 |
| 完整 best params / 搜索 J | `retune-result.json` → `selected.candidate_id`、`selected.params.values`、`selected.latency_ms` | 对应 run 的 retune 结果与 `retune/report/selected.py` |
| Fresh P/C、三个块及 samples 关联 | `result.json`、`parent-events.jsonl`、`retune-events.jsonl` | run 下 `parent/`、`retune/` 的 events/jobs；worker 输出含原始结果 |
| 生成、空间、输入交付 | `proposals.json`、`child-space.json`、`generation-events.jsonl`、`actual-delivered-response-views.json`、`input-verification.json` | `generation/` 的 events/jobs/sandboxes；parameterizer ID 以 manifest 为准 |
| 父和 acquisition | host 下 `parents/<P1或P2>/{shared,responses,provenance}.json` | `R/inputs/information-groups/<P>/shared-dca5c21.json`；`R/runs/information-groups/<P>-acquire-dca5c21/responses.json` |
| Runtime 定位，不发布内容 | phase manifests | `R/inputs/information-groups/<P>/group-configs-519d037/<G>.yaml` |

例：A/P1/G2 原始源为 `R/runs/information-groups/P1-G2-519d037/generation/sandboxes/parameterizer-a633ab06/candidate/parameterized.py`。其 bestparams 为 bf16/plain、BM64/BN16/warps4/stages3，搜索值 **3.490816116333008 ms**，fresh C **3.400704 ms**。其他完整参数从逐行 selected 字典取，不能套用此例或 proposal defaults。

### 3.2 门槛、正负例与质量账

range=`[min(C blocks)/max(P blocks), max(C blocks)/min(P blocks)]`；胜出须 R 更低且 G2 上界严格低于对照下界。各对照至少 3/4 且覆盖两任务。这是预声明投资门槛，**不是置信区间、统计显著性或总体因果效应**。

| 固定父 | G2 对 G0 | G2 对 G1 | 对照下界减 G2 上界，G0 / G1 |
|---|---|---|---|
| 43/P1 | 胜 | 负 | +0.000463097 / -0.171422039 |
| 43/P2 | 胜 | 胜 | +0.030933537 / +0.011272383 |
| 21/P1 | 胜 | 胜 | +0.066675977 / +0.121672579 |
| 21/P2 | 胜 | 胜 | +0.074919925 / +0.077305707 |

task43/P1 的 G2 相比 G0 归一化改善仅约 **0.135%**，对 G1 则 R 高约 **21.04%**。不能用总 PASS 遮蔽这行。72 full blocks 均 5/5 correctness、100 samples；**全部 36 child blocks 均 5/5 FP64 rescue**。采用 `dual_witness_relaxed`，不是 strict-allclose 或精度等价。

480 tuner records=393 complete+87 fail，含24 cached witnesses。四父共享 acquisition 共48次、1261.56 worker秒、1失败，不能在三个组重复收费。模型36 module calls/39 attempts；一次 rescued rewriter 的 token/cost 缺失，provider cost 字段0.92049876也未核账，不能当完整货币成本。mask 是交付视图，不是文件系统访问隔离；一份 proposal 自述自测，不能宣称生成期间绝无反馈。

## 4. 早期 C2：固定点、原生重调参、短闭环和补域

### 4.1 首轮与固定点机制证据

`docs/result-c2-local-single-step-experiment.md`，执行 SHA **`21d833419a3ef10e82835e6a6ddf05a13cb86c84`**。43/direct 在 A，21/legacy 在 B；每个 A 普通信息、B 增一次 baked response，各一 proposal。属于探索性单步观察，不是匹配重复研究。

| 任务/臂 | 有效/失败 trial | 搜索 best ms | Fresh parent / child ms | 冻结输出 |
|---|---:|---:|---|---|
| 43/direct/A | 0/32 | 无 | 3.491840 / 无 | parent，静态拒绝不是测得 SDPA 慢 |
| 43/direct/B | 33/7 | 3.000832 | 3.435488 / 2.942976 | child，同臂下降14.336% |
| 21/legacy/A | 27/13 | 9.241600 | 8.374272 / 9.011200 | parent |
| 21/legacy/B | 37/3 | 4.055552 | 8.470528 / 3.941376 | child，同臂下降53.470% |

随后在 B0 复用两份 task21 结构，固定参数做12 full：FP32 storage 下 A/B=6.975568/7.954432 ms，B慢14.033%；FP16 storage 下6.038528/7.214112，B慢19.468%。FP32 无 rescue，FP16 每块5/5 rescue。即使 registers255→168、spills108→0，B仍慢，支持“此固定点资源减压不足以保证加速”，不是纯 streaming-K 因果消融。

本地 `results/c2-local-first-wave/host-A21d/` 的 `direct-{A,B}21d-result.json`、`-events.jsonl` 对应远端 A `R/runs/direct-{A,B}21d/`；B 本地 `host-B21d/legacy-{A,B}21d-result.json` 对应 `R/runs/legacy-{A,B}21d/`。acquisition 文件为 `acquire21d-responses.json`，共享输入 `shared-final21d.json`。
源码：A direct 的 `direct-A21d--task_rewriter-fb2b0181__candidate__child.py`、`direct-B21d--task_rewriter-b8212abf__candidate__child.py`；B legacy 的 `legacy-A21d--parameterizer-7c9f8a0d__candidate__parameterized.py`、`legacy-B21d--parameterizer-87b38eb1__candidate__parameterized.py`。冻结产物同 stem 的 `frozen-child.py`；direct B 同 stem 用 `frozen-child.py` 替换 `child.py`。
legacy 远端源分别为 `R/runs/legacy-A21d/sandboxes/parameterizer-7c9f8a0d/candidate/parameterized.py` 和 B run 的 `parameterizer-87b38eb1/candidate/parameterized.py`。固定点记录为 `results/c2-local-first-wave/followup-fixed21d/{fixed-inputs.json,measurements.jsonl,events.jsonl,summary.json}`，远端 B `R/runs/followup-fixed21d/`。早期 `acquire41f` 零 endpoint 调用是无效仪表尝试，不当作 C2 负结果。

### 4.2 Native-retune：结构 bundle 的三对 seed

`docs/result-c2-native-retune.md`，SHA **`dca5c21ba08ffc930a644c92b187b5b78b703d12`**。不新生成，使用首轮原始 parameterized source，而不是 frozen winner；sampler0/1/2，validator seed0。

| seed / host | A 搜索 / fresh ms | B 搜索 / fresh ms | B fresh 下降 |
|---|---|---|---:|
| 0 / A | 6.107648 / 6.118912 | 4.203520 / 4.085760 | 33.227342886% |
| 1 / B | 6.117376 / 6.142544 | 3.832832 / 3.730432 | 39.268938742% |
| 2 / A | 5.275648 / 5.139968 | 4.055040 / 3.891200 | 24.295246052% |

三对均正，支持“A 初始化遗漏不能解释全部 B bundle 优势”，不证明纯结构因果或全局最优。240记录=198 complete+42 fail，含12 witnesses；18 final blocks 全部5/5 rescue。本地 `results/c2-retune-native/seed<N>-<A或B>/` 有 `source.py`、`selected.py`、`result.json`、`manifest.json`、`study-input.json`、`events.jsonl`、`published-space.json`。
远端按表 host 为 `R/runs/retune-native/seed<N>-<臂>-dca5c21/`，配置 `R/inputs/retune-native/runtime.yaml`，study=`R/inputs/retune-native/studies-dca5c21/seed<N>-<臂>.json`。完整 params 读 `result.selected.params.values`；例如 seed1-B=`cand-573e18cb`。source/reference 的准确输入路径读各 `study-input.json`，不要从一臂猜另一臂。

### 4.3 T5：真正的 common-elapsed 负结果

`docs/result-c2-short-closed-loop.md`，SHA **`c95085b57efaf1f4b60bf6aa88a71b4e9725366b`**。四 arm 两轮，从原 P1 重新出发；A43 G0/G2 在 GPU0/1，B21 G0/G2 在 GPU1/0。共同 H=min(两臂 elapsed)，只使用 H 前真实 ready 的 incumbent。

| task | H 秒 | G0 H时 ms / R | G2 H时 ms / R | G2 晚到终点 ms / R |
|---|---:|---|---|---|
| 43 | 6938.284780 | 2.568192 / 0.735268274 | 2.920528 / 0.851493701 | 2.920448 / 0.851470415 |
| 21 | 6433.833151 | 5.041664 / 0.595272601 | 8.369664 / 1.000000000，原父 | 5.782000 / 0.690828202 |

gate 0/2，两个任务均反向范围不重叠；晚到终点不能回填 H，且仍逊 G0。完整 G2 相对自身父改善14.853%/30.917%，因此不能说“C2 从未改善”。G2额外 elapsed959.257960/1820.106705秒，采集已含在总时间，不再加。320 records=273+47、16 reuse；48 acquisition=46 valid+2 fail；48 full有效。T6条件性12h为 **NOT_RUN**。
本地 arm 根 `results/c2-closed-loop/HOST-<A或B>/<G0或G2>/`，远端对应 host `R/runs/closed-loop/<G>-c95085b/`。arm 有 `result.json/events.jsonl/manifest.json/initial-shared.json`；`round-<1或2>/` 有 `shared.json/responses.json/child-source.py/child-selected.py/ready-artifact.py`，`opportunity/retune/result.json` 给 bestparams，phase `jobs/` 给样本。真正输出看 ready/selection，不把未晋级 child 当 winner。
21/G2/r2=`cand-11efd841`，搜索5.914112091064453不是 fresh5.782000；远端源 `R/runs/closed-loop/G2-c95085b/round-2/opportunity/generation/sandboxes/parameterizer-11fa731a/candidate/parameterized.py`。配置 `R/inputs/closed-loop/configs-c95085b/<G>.yaml`，初始 shared=`R/inputs/closed-loop/shared-original-P1.json`。

### 4.4 Unlocked-region：可达性成立，不追回全部差距

`docs/result-c2-unlocked-region.md`，同 c95085b。B机 E1 只补 BN256，E2再补SM32/64，各B40；0新生成。统一 B0 正/反/正测七产物，21 full，无事后重选。

| 同卡 fresh 产物 | old-G0 | old-G2 | default-N128 | default-N256 | N256-SM64 | E1-selected | E2-selected |
|---|---:|---:|---:|---:|---:|---:|---:|
| ms | 5.019135952 | 5.788671970 | 7.024640083 | 7.786496162 | 5.199359894 | 6.067728043 | 6.217216015 |

只加 N 慢10.8455%；N256/SM64比old-G2快10.1804%，仍比G0慢3.5907%。没有 N128/SM64 配对，新点与old-G2共12参数不同，不能归因纯N/SM。9条N256记录中5成功计时、4 compile screen拒绝；B40没找到已知好点也不证明无更优点。
本地 `results/c2-unlocked-region/inputs/` 有 `source.py`、`reference.py`、`plan.json`、`comparison-plan.json`、`E1-study.json/E2-study.json` 和空间；`frozen-comparison/{old-G0,old-G2,E1-selected,E2-selected}.py` 为冻结源。三个固定点复用 source+plan params，没有证据支持单独 `N256-SM64.py`。
`E1/result.json` 的 winner=`cand-e3bd1a94`，E2=`cand-80870c1f`；比较原始记录为 `comparison-c95085b/measurements.jsonl`，初始探测 `fixed-probes-c95085b/measurements.jsonl`。远端 B `R/runs/unlocked-region/`，E1 output=`E1-c95085b`，源=`R/inputs/unlocked-region/source.py`；其他精确路径从 study/plan 的绝对字段取得。首轮/固定点/补域 runtime YAML 精确文件名未定位，不能补猜。

## 5. 后期 C2：保留全部正、负与 censored cells

以下结果均为各批自己的控制和终测，`gain=100*(1-treatment/G0)`。后期 task21 shape `(10,112,224,224)`，task43 `(128,512,768)`；reference/fp32 I/O 不代表内部严格 FP32。method/A ready 的跨臂自归一化分数与 terminal 同卡分数是两种问题。

| 批次 / task / rep | G0 ms | H/C2 ms | gain | 主张状态 |
|---|---:|---:|---:|---|
| method/core 43 A0 | 3.120127916 | 2.935311913 | +5.923347% | 污染，排除效力 |
| method/core 21 A1 | 5.472256184 | 5.209023952 | +4.810305% | 污染，排除效力 |
| method/core 21 B0 | 8.363519669 | 8.364031792 | N/A | 双方 invalid 后保留父 |
| method/core 43 B1 | 3.391488075 | 2.993151903 | +11.745174% | 污染，排除效力 |
| method/A terminal43 | 2.825183988 | 3.046400070 | -7.830148% | 有效负面终点 |
| method/A terminal21 | 6.177279949 | 3.610064030 | +41.559002% | 有效正面终点，一条两轮轨迹/臂 |
| opportunity 43 rep0 | 3.098608017 | 2.992127895 | +3.436386% | heldout PASS |
| opportunity 21 rep0 | 6.047263861 | 8.360960007 | -38.260215% | FAIL，C2为保留父 |
| opportunity 21 rep1 | 未测 | 未测 | N/A | censored，9块缺失 |
| opportunity 43 rep1 | 未测 | 未测 | N/A | censored，9块缺失 |
| region补测21 rep1 | 6.178864002 | 5.155839920 | +16.556831% | 新批次PASS |
| region补测43 rep1 | 2.951168060 | 2.969599962 | -0.624563% | 新批次FAIL |
| pilot43 rep0 | 2.975200057 | 2.992127895 | -0.568965% | FAIL |
| pilot21 rep0 | 3.675136089 | 5.568000078 | -51.504596% | FAIL，双方胜父不等于C2胜G0 |
| pilot21 rep1 | 4.205567837 | 3.787775993 | +9.934255% | 唯一PASS |
| pilot43 rep1 | 2.944048047 | 3.059711933 | -3.928736% | FAIL |

method/core 的13/15源字节、10/15 body不同，staging污染使 H-G0、P-L、H-P 效力/intent 消融均不可用。B0 key mismatch 是 staging覆盖，不是模型漏 key。method/A 修复后 ready 两任务 +2.218458%/+27.307552%，2/2通过，但不能替代上表 terminal，亦不能隔离 intent/prompt/反馈。
opportunity 全局仍 **INCONCLUSIVE**；region 追加固定产物验证与旧 rep0 合表为描述性 **2/4**，不是回填旧 deadline，也不是新测四 pair。pilot 强 G0 已共享 helper/intent/推荐/扩域/自主工具，仅 C2 增 targeted workflow，**1/4 不支持一般成功**。2%+范围分离均为描述门槛，不是 CI。

### 5.1 Method 与 opportunity 的源码及终测定位

报告分别为 `docs/result-c2-method-improvements.md` 和 `docs/result-c2-opportunity-driven-improvements.md`。本地 M/O 取第2节对应 `results` 根。原始参数以 `retune/result.json` 的 selected/selected_trial 及 `report/selected.py` 为准；执行 trial 模板为 `retune/candidates/<cid>/trials/<tr>.py`。生成 session 位于对应 generation `events.jsonl`，不是完整会话正文。

| Method 数据 | 精确本地模板 | 可直接取用的身份 / 原始测量 |
|---|---|---|
| core，污染但保留 | `M/core/<A0,A1,B0,B1>/<G0,L,P,H>/retune/` | 11个实测child；`M/core/<slot>/finals/jobs/` 为full原件，B0无新winner |
| A task43 G0终点 | `M/A/A0/trajectory/round-2/opportunity/retune/` | `cand-57350b8f/sp-57ebacb2/tr-1ea4cccc` |
| A task21 H终点 | `M/A/A1/trajectory/round-2/opportunity/retune/` | `cand-fe33920c/sp-f4fe5efe/tr-0475bef0` |
| A task21 G0终点 | `M/A/B0/trajectory/round-1/opportunity/retune/` | `cand-242aaad3/sp-acd00341/tr-f854c006`；r2 default拒绝 |
| A task43 H终点 | `M/A/B1/trajectory/round-2/opportunity/retune/` | `cand-fdbb4b2a/sp-5ea65edb/tr-69ccf3ac` |
| A 同卡终测 | `M/A/terminal-task<21或43>/result.json` 与 `jobs/` | task43在A0，task21在B0；非轨迹搜索score |

method/A 原始源从每条 `opportunity/retune/manifest.json` 反查，generation 在相邻 `generation/`；core则读`core/<slot>/<arm>/retune/manifest.json`，生成链在`core/<slot>/generation-<G0或L或P>/events.jsonl`，P/H共享rewrite。task21 H/r2 rewriter=`ses_f4e2d7198ffeJ9pM7BCyiaDEuP`，parameterizer=`ses_f4e23c4dbffeoQgmGt3SQEREcp`。`M/export-manifest-{A,B}.json`、`M/A-export-manifest-{A,B}.json` 给远端绝对映射与既有3253/2347条hash；不能用 slot 的首字母机械猜每次终测host。远端 runtime分别为`/root/autodl-tmp/c2-method-improvements/configs/<slot>.yaml`和`/root/autodl-tmp/c2-method-improvements/A-configs/<slot>.yaml`。

Opportunity 每行本地 `O/wave<w>/<slot>/information/retune/`；**真正 heldout incumbent** 是 `O/wave<w>/<slot>/export/selected.py`。下表均给 native child 身份，1/B1是唯一未晋级者。

| wave/slot；task/arm | candidate / space / trial | initial parameterizer；存在时 expansion |
|---|---|---|
| 1/A0；43/G0 | cand-88a16232 / sp-2c10b493 / tr-dc0f40a3 | 8c83774c；52c36276 no-op |
| 1/A1；43/C2 | cand-74ad95d9 / sp-d1c21d32 / tr-bbec8629 | 5d4443c2；a309c46c |
| 1/B0；21/G0 | cand-6b58050e / sp-e720b5a3 / tr-ac63d235 | 71cb448d；无 |
| 1/B1；21/C2 | cand-48089a01 / sp-cc2e54f3 / tr-5e14a39d | e3b1215c；7b2fc9b6，child未晋级 |
| 2/A0；21/C2 | cand-c2be6230 / sp-6af980fc / tr-0a5332e4 | a54e7ead；9a0b0d7f，base保留 |
| 2/A1；21/G0 | cand-e0b3ce5f / sp-83869f74 / tr-fd5e0eae | 9ed8c02f；c0baf35f，base保留 |
| 2/B0；43/C2 | cand-1c5b1ef0 / sp-2710c4b9 / tr-cf6e3465 | 2d1e1da0；52a74f08 no-op |
| 2/B1；43/G0 | cand-8098f763 / sp-560b108e / tr-cc4ee62c | c66a1d64；a68c8fdd |

initial 源位于每行 `information/generation/sandboxes/parameterizer-<后缀>/candidate/{source,parameterized}.py`；扩域源在 retune 的同类 sandboxes。`selected_trial`、`selected_space` 才决定最后配置，不把 initial manifest 或最后扩域空间当 winner。1/B1实际保留原父 `cand-4c96b8c4/sp-e3acfbaf/tr-6afd5e56`。
heldout=`O/heldout/wave1/{A0,B0}/result.json` 和 `jobs/`；wave2同名 result 明确 censored且三角色数组空，不能声称存在18块jobs。remote roots和hash见 `wave{1,2}-export-manifest-{A,B}.json`、`heldout-wave1-export-manifest-{A,B}.json`、`deadline-closure-export-manifest-{A,B}.json`。源头 session 看 `information/generation/events.jsonl`；例如1/A1 rewriter=`ses_f4b5f6a18ffeF0xwwI5mys0GGC`。

### 5.2 Region：两个 host 的本地 task 层不是远端层

`docs/result-c2-region-followup.md`。本地 `results/c2-rep1-heldout-followup/task21/` 在 A、`task43/` 在 B，**两台远端根都直接是 F，没有额外 task21/task43 前缀层**。每任务 `setup.json` 固定来源绝对路径、SHA256、params、source-result hash；`export-manifest.json` 映射112条既有核验记录。
每任务 `roles/parent/parent.py`、`roles/G0/selected.py`、`roles/C2/selected.py` 可直接取源码；新score=`runs/task<t>/result.json`，该 run 的 manifest/events/worker记录给9 full。21 G0/C2复用 opportunity2/A1、2/A0；43复用2/B1、2/B0，candidate/trial见上表。task43有45rescues，task21零；230.354154秒为新测窗口，不是新搜索。
helper正式集成在 B 的 I：`roles/parent/candidate/parent.py`、`roles/child/candidate/parameterized.py`，params=`plan.json`。`attempts/01-parent,02-child,03-child,04-parent/result.json` 分别8.365056037903、16.514127731323、16.494080543518、8.366080284119 ms；jobs ID依次`df4d23e9/00cec657/ced13899/7f4187b9`，模板`jobs/final-reeval-eval-<id>.out.json`。可证明正式helper集成，不证明新方法或agent自愿采用。
自测审计=`results/c2-agent-self-tests/evidence.json`，23 sessions、137 bash；rewriter8/8、initial6/8、expansion2/7有runtime测试，不是137 benchmarks。W1/B1私有child7.854ms、正式同源同18参数quick16.911423683ms；来源`rewriter-60b0ff95/ses_f4b5f37edffe77GlhCxgFA8nm6`。这排除“仅TPE换参数”解释，但不隔离timer/cache/state原因。衍生JSON的远端副本未定位。

### 5.3 Helper pilot：全部八个候选与四个 pair

`docs/result-c2-helper-pilot.md`。本地每行 `B=results/c2-helper-pilot/main/<host>/wave<w>/<slot>/`，`T=B/information/retune/`。远端应先读 `results/c2-helper-pilot/main/export-manifest-{A,B}.json` 的路径映射，根P本身不保证包含本地host分层。4322条是既有导出核验数，不是本次重算。

| host/wave/slot；task/arm | candidate / space / trial | rewriter / initial parameterizer 后缀 |
|---|---|---|
| A/1/A0；43/G0 | cand-58dccaf8 / sp-c44e8285 / tr-66df2b6e | 53e3248d / 3a2e5db9 |
| A/1/A1；43/C2 | cand-091efa59 / sp-4c958b16 / tr-45f9064b | 2ef9d6e5 / b7d94f66 |
| B/1/B0；21/G0 | cand-24454c71 / sp-a1d16865 / tr-64541a50 | da1b6024 / 325ee0f9 |
| B/1/B1；21/C2 | cand-3a36b550 / sp-f3f4954d / tr-0d087ed9 | b6604a66 / ddc87c6a |
| A/2/A0；21/C2 | cand-9094d915 / sp-19d6f963 / tr-d8144d71 | 54563f47 / bd4dcba4 |
| A/2/A1；21/G0 | cand-ee6a1a23 / sp-19ffd1a8 / tr-d745548b | bd7ba6a2 / 4458c201 |
| B/2/B0；43/C2 | cand-f8fc8aa6 / sp-fb8cb12c / tr-878ac3ea | 2f5a72ec / 84d24931 |
| B/2/B1；43/G0 | cand-bf6b35e1 / sp-85e0d97d / tr-a45002ac | d7c830ab / 3d6e8eeb |

取kernel：`B/export/selected.py`及同目录`imports/`；取native selected：`T/report/selected.py`或`T/candidates/<cid>/trials/<tr>.py`；完整bestparams：`T/result.json:selected.params.values`与`selected_trial.params`。原始可调源在`B/information/generation/sandboxes/parameterizer-<后缀>/candidate/parameterized.py`。
正式分数=`main/<host>/heldout/wave<w>/result.json` 的 `parent_finals/g0_finals/c2_finals`，`jobs/final-reeval-eval-*.json`链接物化源码和`.out.json`样本。session索引=`main/<host>/_agent-evidence/wave<w>-<slot>/sessions.json`；唯一阳性A/2/A0 rewriter=`ses_f482bff51ffek1j4vUFgepBvX7`，initial=`ses_f48210b61ffe87Y4otA3GPu6RA`，expansion=`parameterizer-0900b094/ses_f48061815ffeNq01kgrNx7koaQ`。
usage=`results/c2-helper-pilot/usage-audit.json`：29 named sessions、522工具记录，8/8 rewriter、5/8 initial采用helper。27 CLI尝试中7个worker前错误，20 QUICK为16valid/4invalid；24推荐请求匹配预算内trial，含anchor reuse。只有阳性winner用新增EW1024，但同时改变其他参数且无同域额外B40对照，不能把9.934255%归因扩域。`gate/` 的显式指令readiness为另一个session、2 QUICK，不进自愿采用分母。

### 5.4 Task21 配置诊断：机制正例不是新 C2 winner

`docs/result-c2-task21-config-diagnostic.md`，本地D对应第2节。`sources/G.py`、`sources/C.py`来自pilot B/1/B0和B/1/B1；`cells/<cell>/params.json`、`plan.json`固定配置。六cell均在B0新测，各3 full，90/90 correctness均经FP64 rescue。

| cell | G | C | K32 | K64 | T_SK1 | T_SK4 |
|---|---:|---:|---:|---:|---:|---:|
| median ms | 3.664896011 | 5.561360121 | 5.547008038 | 5.535744190 | 5.096447945 | 6.284255981 |
| 相对C改善 | 对照 | 0% | +0.258068% | +0.460605% | +8.359685% | -12.998544% |

K-only收益<0.5%；T伙伴固定时SK4→SK1下降18.901331%。C原本就是SK1，C→T_SK1不能叫“开启SK1收益”。T_SK1仍比新测G慢39.061188%，仅追回原gap24.514684%；没有postselection heldout，不更新正式winner。
运行顺序=`commands.jsonl`，原始记录=`runs/<编号-cell>/{result.json,raw_worker.json,commands.jsonl,source/}`，例`runs/01-G`链接`jobs/final-reeval-eval-7a3931ee.out.json`。G/C源SHA256分别`fdd95ca58e696ba081a0f742c82fd49a600952ad88e29f3c7e83faeb56ce1f26`、`c4982b91b5d0456c409ccdfda193b1b6931436f07862939702df404c471c3849`。`historical-trial.json`为C旧`tr-5dfdde92`，不是新生成。`profiles/{G,C}/{trace.json,cuda-events.json,key-averages.json,result.json}`各一次instrumented forward，不精确分摊正式协议的latency gap。

### 5.5 后期源码与报告 pins

| 阶段 | 实验/实现 SHA | 报告 SHA；文件位于 docs/ |
|---|---|---|
| method/core | `1090b673d6a7332d20e3f43dc04a9b7b0424dda1`；staging修复`d61b30528b3105aa1012a52c015e857b0dcca267` | `5777ee7223437ba2d4ede51324e4391bc973c68c`；`result-c2-method-improvements.md` |
| method/A | `3ea43cc1cca2454dc26a19c4697dd65c769df7ff` | 同上，不能把core全部追认为修复版 |
| opportunity | `96c8bec3f91071ab668550ff2510f3cd5a545f18` | `f733763a1fafe5c57b548cde80c129e853b27d7f`；`result-c2-opportunity-driven-improvements.md` |
| region | 补测沿用96c8bec；helper实现`8eae0eb4828826ab99c9948ea8160e66e34c3498` | `c7f1d6eeb055aafc2989aaf335146acac5e6ac53`；`result-c2-region-followup.md` |
| pilot | `cda113070c8ba3a91f480f42b68d196187796cda` | `757409bfe38470459bb35aaa4193d22c0bb72580`；`result-c2-helper-pilot.md` |
| diagnostic | evaluator仍cda1130；计划`69d74a307992f5e58c0ff6e74307d141e1c85a12` | `b7d3b20ed4ec5ed0fb55246122a9ae9a012146a6`；`result-c2-task21-config-diagnostic.md` |

region补测早于helper实现，不能归因8eae0eb。后续F1 APPROVE表示交付/解释通过，不把opportunity变成PASS，也不把pilot从1/4改为成功。

## 6. 历史性能、C1、external、A800 与 window-1

### 6.1 主要结果及候选，不按报告标题重复计数

| 资产 / run | Candidate及 final | 论文用途与关键纠正 |
|---|---|---|
| 早期48 `run-l3-48-20260905-010737` | cand-c18203b6，FP32，**mean**1.96ms，6.08h | 机制；std0.829/max10.2ms，推测1.88不是新测量 |
| 早期21 `run-l3-21-20260905-071312` | cand-7dcdbd99，FP16，mean15.80ms；374trials | 2/6预期rewrite执行，80trials死分支；非完整设计 |
| 早期43 `run-l3-43-20260905-091705` | cand-e3a5da01/tr-88446e7a，FP16，final10.2ms | tuned9.73不能替final；row microtile机制 |
| 早期21 `run-l3-21-20260905-195615` | cand-f66890d0，独立100sample11.40ms | 进度非run final；基线双峰，1.323x和1.28x用不同基线 |
| 成熟21 box1 `run-l3-21-20260908-232211` | cand-8a5dcef3，**seed**，final3.4545ms，800trials/12.28h | 同候选box2新测3.619840，external4.8271，报告1.33x；不可混box1时间 |
| 成熟43 box2 `run-l3-43-20260908-053708` | cand-fbd1988e/sp-db9f8108/tr-bc111c61，FP16，3.0126ms | 5/5、零rescue；历史141% compute来自错误TF32 roof，排除该资源解释 |
| 43 round-2 box2 `run-l3-43-20260909-015247` | cand-173b12ae，BF16，2.762752ms；840trials | 报告3.98x对compile-TF32，不是精度等价；新winner无fresh external对照 |
| 成熟48 4090 `run-l3-48-20260907-202457` | FP16，final **mean**1.41ms；external1.477 | 应报告tie非胜；105.3% roof口径不可恢复，排除roof不删除latency |
| 后续48 `run-l3-48-20260909-115701` | final median1.5616ms；cand-efcecff9曾tuned1.5544 | G9/profiling复用不另算search；完整镜像未定位 |
| S2 box1 `run-l3-43-20260911-053020` | label，cand-d02b0742，mean3.11/median3.0935 | 与下行各640trials/16spaces，观察性C1 pair |
| S2 box2 `run-l3-43-20260911-052630` | vector+ledger，cand-2d8eaf9a，mean2.84/median2.8344 | reported8.68%下降；跨box且两个开关，非isolated ledger因果 |
| E1 treatment box2 `run-l3-43-20260911-230736` | cand-eb28dbe5，2.8785ms；12rewrites/12.346h | 有界性能；canonical报告未写winner精度，不猜 |
| E1 control box1 `run-l3-43-20260911-230217` | seed cand-e254236c，4.0852ms；0rewrite/12.506h | 29.5%观测主要比较rewrite与无rewrite，非reserve-round隔离 |
| A800首轮 `run-l3-48-20260911-052647` | cand-56411b63，BF16/split3，0.9728ms，零rescue | resume使实际13.670h，不是12.383h；快照不是额外run |
| A800 E3 `run-l3-48-20260911-231217` | cand-207eabd1，FP16/plain，1.0532ms，13.159h | reported7.654x对compile-TF32；不能用tuned1.0097或写干净12h |

主要出处：`docs/results-summary-l3-vs-external.md`、`result-l3-43-final.md`、`result-l3-43-r2-2p76ms-bf16.md`、`result-paired-arms-l3-43.md`、`results-e1-e3.md`、`result-l3-48-a800-0p97ms.md`。任务48内部 batch2048、外部publication batch16相差128倍，0.008918ms/107.076x不能横比。KernelBench canonical pin=`423217d`；精度写“所报接受策略下最强可用torch baseline”，不要把FP16/BF16与TF32叫同精度。
S2账本纠正后 treatment26hit/9miss/5vacuous，control23/15/1，Fisher单侧p=0.1585；3.7倍noise使用的是correctness fraction2.35%，不是timing噪声。E1声明58/81、round级5/6且p=0.109，不能用依赖声明当独立显著样本。E1 resumed control仍属原run，末阶段结果未完整追踪。

### 6.2 历史候选和 raw 的精确检索

| 数据 / host | 本地与远端根 | 源码、配置、原始数据 |
|---|---|---|
| box2 round-2 | 本地 `E/box2-runs/runs-l3/run-l3-43-20260909-015247/`；远端 `/root/autodl-tmp/opop-workspace/opop-glm/runs-l3/run-l3-43-20260909-015247/` | `candidates/cand-173b12ae/source.py`、`report/best_kernel.py`、`manifest.json`、`events.jsonl`、`jobs/*.out.json` |
| box2成熟43及corpus | 同上父根，另有`run-l3-43-20260908-053708`、`run-l3-21-20260909-154359`、`run-l3-48-20260910-083700` | 四run corpus共2735trials；`E/box2-runs-l3/`旧备份仅metadata/events，不能代替源码镜像 |
| A800 E3/首轮 | 本地 `E/a800-runs-l3/<run>/`；远端 `/root/autodl-tmp/work/opop-glm/runs-l3/<run>/` | E3 `candidates/cand-207eabd1/source.py`、`report/best_kernel.py`；首轮`candidates/cand-56411b63/source.py`和`jobs/final-reeval-eval-5a4c820c.out.json` |
| box1成熟21 | 远端 `/root/autodl-tmp/opop-workspace/opop-glm/runs-l3/run-l3-21-20260908-232211/` | `candidates/cand-8a5dcef3/source.py`；本次未定位完整local mirror |
| E1 control | box1 `/root/autodl-tmp/opop-workspace/opop-glm/runs-v3/run-l3-43-20260911-230217/` | `manifest.json/events.jsonl/report/best_kernel.py`为检索模板，未逐文件验证 |
| S2 pair / E1 treatment /早期run | 第6.1节run/candidate确知 | 完整远端根和本地镜像未建立，不能按日期猜runs-v3/runs-l3 |

box2 round-2 registered source SHA=`415416b161275ed706ee8c3817ac99ca9c2009eab1ae098b56d268d2a8c3f3bc`（events行1197）；E3=`befe5eb71afe5a233ce490eef3a8330aa199786c7922d6714878b74b5393b88a`（行868）。这是注册身份，不声称当前source.py字节hash相同。run内trial执行源优先于可变source；bestparams取manifest/events选中trial及物化winner，不另造`best_params.json`。成熟21代码receipt e2d3d32，43round-2 e840642，S2 dceda97仅为原报告提供的短pin。

External本地包=`E/l3_best_candidates_21_43_48_20260908/`及同名zip，先读`manifest.json`。原服务器未知；原根`/home/kangtian/kernelfoundry-reproduction-main/runs/`，各行run下有`map_elites_events.jsonl`和checkpoint。

| task / 本地包内source | source SHA256 | 原始run / checkpoint |
|---|---|---|
| 21 `L3_21_EfficientNetMBConv/best_kernel.cu` | `393007a36ebb45515008368c5b6e3377c90cbd0ea834e528684ee439dc62f38f` | `mapelites_l3_21_cells27_seed42_20260907T061306Z/checkpoints/trial_0014_branch_006_child.state` |
| 43 `L3_43_MinGPTCausalAttention/best_kernel.cu` | `e9f5821bf3a50a7250dcaef4507a7b72149efc74031b14b54540b0eeff0e2633` | `mapelites_l3_43_cells27_seed42_20260907T061306Z/checkpoints/trial_0014_branch_003_child.state` |
| 48 `L3_48_Mamba2ReturnY/best_kernel.cu` | `b07b30cb8aa9736b84bcfab1c55377d4833c10c7f810adeb8b7939cf6a6d76fb` | `mapelites_l3_48_cells27_seed42_20260907T061306Z/checkpoints/trial_0015_branch_005_child.state` |

内部复测脚本在box2 `/root/autodl-tmp/ext-eval/{ext_eval.py,ext_eval2.py,ours_on_box2.py,verify_claims.py}`，不是已定位raw输出。`docs/finding-external-cuda-vs-our-triton.md`的43严格IEEE CUDA8.893 vs Triton14.565ms，CUDA1.64x；TF32 5.529 vs5.555近似持平，必须保留。撤回的12.339ms IEEE attention及+136%headroom不能用；重测shipped tile6.713/6.723/6.724ms、sweep改善28.6%才是纠正值。

### 6.3 Step-3 与 window-1

| 数据 | arm / host / run | Winner / final | 位置与用途 |
|---|---|---|---|
| step-3 | arm1 box1 `run-l3-43-20260912-214445` | cand-b82b401c，2.83ms；840trials | `E/s3_run_backups/arm1.tar.gz` |
| step-3 | arm2 box4 GPU1 `arm2/run-l3-43-20260912-214326` | cand-c4e6985e，2.49ms；800trials | `E/s3_run_backups/arm2.tar.gz` |
| step-3 | arm3 box4 GPU0 `arm3/run-l3-43-20260912-214326` | cand-c18186fc，2.69ms；760trials | `E/s3_run_backups/arm3.tar.gz`；2e未证明净收益 |
| window-1 | n1-a box4 GPU0 `run-l3-43-20260915-071127` | cand-a14c7861，2.8743679523468018ms | off，720trials；宣称FP16但实际dtype未完整建立 |
| window-1 | n1-b box4 GPU1，同run ID不同arm目录 | cand-e72bb36c，2.5374720096588135ms | off，840trials；宣称BF16 |
| window-1 | pilot-p1 box1 GPU0 `run-l3-43-20260915-070539` | cand-0e23a83e，2.888144016265869ms | active，760trials；宣称BF16 |
| window-1 | pilot-p2 box1 GPU1 `run-l3-21-20260915-070539` | cand-87c56c65，3.60806405544281ms | active，680trials；宣称FP16，与p1不是paired task |

step-3为C1观察性轨迹：arm2半程已到2.488ms，arm1半程3.206ms，但mode/host/CPU/elapsed混杂。文字“80额外trials”不适用于arm1对arm2（表为840/800）；80对应arm3。`E/s3_run_backups/MANIFEST.txt`给5822/5465/5207文件及event hashes，不是样本数；精确远端父根和压缩成员前缀未审，不猜。
window-1远端共同父根=`/root/autodl-tmp/opop-workspace/opop-glm/runs-v4/`，拼`<arm>/<run>/`，host取上表。候选=`candidates/<cid>/source.py`，执行=`candidates/<cid>/trials/<tr>.py`，输出=`report/best_kernel.py`，配置/日志=`manifest.json/events.jsonl`。本地包为`L/.deliver/window1-verification-box{1,4}.tar.gz`和`MANIFEST-box{1,4}.tsv`，压缩成员规范为`runs/<arm>/...`，省掉timestamp run层，**不含jobs或agent sessions**。
box1包SHA=`03f728333b64e3ccc80cd81f687e1ab110e0ccd83609588a32d36763d320f669`；box4=`7b4d7b8f7a55099543d773653dfeb6456bc62f408105676b8662b17b8f920787`。n1启动HEAD=`1fb9c7d4669dd8541b7812bdd3d1a241e9c46f21`，pilot=`f159555892ebc60601364b7636681bf85908f3b1`；dirty patch未知。既有4167/4167文件核验不是本次新验证。
纠正权威为`docs/result-window1-offline-closeout.md`、`docs/closeout-tables/05-p2prime-current.txt`：noleak p1 n=7，marginal/conditional绝对误差中位数0.1390/0.0069，报告20.3x，5项缺分母排除；p2 n=9，0.1766/0.0017，104.8x，2项排除。是付费局部发现对历史prefix，不是等信息比较；旧50.9x/25.8x不可用。
原13对执行源码 **0/13相同**，1.03% median/2.50%p90/5.33%max不是timing noise floor、MDE或tie阈值；33个same-code组都是within-candidate跨space retune。reused组p90由0.55%更正为1.88%，不验证4%阈值。wall_ms/cpu_issue_ms/overhead_gpu_ms始终null，不能补作多目标结果；5/12已查source.py匹配注册SHA，执行trial才是身份依据。

### 6.4 其他已识别历史资产与覆盖边界

G9不是T4的G0/G1/G2：A800 task21 none/verdict/rich各2calls，best7.8679/4.5670/4.7585ms，parent5.4472；quick探索且绕过analyst，rich未胜verdict。`E/a800-runs-l3/g9-20260910-{171936,180344,183406}/`有候选/jobs，远端同A800 runs-l3根；171936为失败尝试，其余精确attempt映射待核。GAP文档中的G1/G2也可能是缺陷编号，不是实验臂。
其他机制资产：A800 `noise_floors_a800.json`、`calibration.json`、`_noise-floor-jobs/`、`_calibrate-jobs/`位于A800镜像及同远端根；所谓fraction是数值一致率而非延迟噪声。162配置/box资源map是硬件可比性证据，不是优化replicates；per-candidate probes两任务×四候选复用历史源。A1从40到51到61 judgments是纠错重分析，资源压力倒数latency不能支持“填满slack”的因果规律。
S7双4090同run ID `run-l3-43-20260913-202332`、不同`s7-control/s7-treatment`，已定位的是3.50h snapshot，无RUN_FINISHED、16次重算/零dispatch，不能当干预后null效果。box2 2735trial corpus不含v3配置，不能当S2/S2d；旧丢失run的38.5% blacklist数字不可复核。537/123 mismatch复核和2790/3396 rescue是check计数，不是新kernel数。
server-handoff仅来源证据：远端父根`/root/autodl-tmp/opop-workspace/opop-glm/runs-v4/`下，A为`m1-a/run-l3-43-20260915-220040`和`m1-b/run-l3-43-20260915-220041`，B为`m2b-{a,b}/run-l3-21-20260915-220113`。各run的`jobs/`下，已纠正m1-b final job=`final-reeval-eval-5c4b9244.out.json`、2.6695680618286133ms；m2b-b=`final-reeval-eval-ef669b67.out.json`、3.9987199306488037ms，均100samples。它们不是本次新测，也不代表window-2完整审计。
`docs/verification-c2-local-experiment-server-handoff.md`还撤回“trial无samples”“agent占elapsed66%至70%”“七父均已证明纯off”；m1-b `cand-b9cf9167/tr-72208736`本来自C4。staged-program只是T4/T5综合，不另算数据。早期dtype、repair、family、ledger、timeout、implementation/revert报告归属上述run或机制组，不把61份历史报告扩成61实验。

## 7. 旧 C3：有效的 36-block 模型级部分结果

`docs/result-c3-qwen3-4b-operator-goals.md`。人工冻结task/evaluator的C3-B，不是C3-A自动任务构建。Qwen3-4B revision=`1cfa9a7208912126459214e8b04321603b3df60c`，BF16/eager/SDPA；58条synthetic prompts=2calibration+14search+42heldout。TTFT为4×4096/1，single为2×256/128，multi为8×512/128的有限同步wave，不是online serving/SLO。
搜索在A0/A1/B0分别TTFT/single/multi；恢复矩阵全部A0串行。六结构机会有5失败、1 accepted；TTFT和single各自baseline fallback。accepted是148行Python decoder composition，替换36层，调用既有F.linear、RMS、SiLU、SDPA，没有新Triton/CUDA设备kernel；“fused_decoder_layer”不等于device fusion。

| 固定row | TTFT ms | Single tokens/s | Multi tokens/s | 相对baseline意义 |
|---|---:|---:|---:|---|
| Baseline | 284.781157 | 25.783499 | 196.880185 | 原模型 |
| TTFT fallback，同baseline字节 | 284.124854 | 26.372324 | 195.102521 | 噪声control，不是优化胜利 |
| Single fallback，同baseline字节 | 284.581298 | 26.451264 | 197.778747 | 噪声control，不是优化胜利 |
| Multi source rewrite | 284.277034 | 29.104201 | 214.785688 | TTFT +0.177021%；single +12.879174%；multi +9.094619% |

每cell三个block的median，完整36分数在原报告:86至103及matrix。TTFT median下降0.504123ms，小于baseline blocks在283.122453至285.098008之间的波动；不能写三目标均优化。single/multi观察范围分离也不是总体统计显著或因果证明。36 quality records通过冻结BF16契约，记录L2 max0/NLL delta0，无全模型FP64 rescue；没有保存完整logit向量，不宣称可独立重算全向量L2。
原heldout harness因128MiB fixture预算在admission前失败，36 null必须保留；修复grouping后新根产出36valid，不能把null写0或追认原矩阵。另有三份真实计算改写被可写parent mirror错误拒绝，不能据此归因LLM不改计算。后续修复`82daf15b460004707e55dbdebb789e2788921527`未用于原search或恢复，不改变旧selection。

### 7.1 旧 C3 候选、源码、矩阵与 raw 精确位置

本地根 `C=results/c3-qwen3-4b-operator-goals/`；远端根Q见第2节。搜索SHA=`af775349a778fa9c62e8fcda6f327578d05ad89d`，恢复SHA=`1870e8e612a1febfec3b77a63b2e091cbc7a6539`，协议SHA=`8cfac7d5a13abafb25f7e658fbf1b6152a0ae0cf`。

| 材料 | 本地 C 下 | Host / 远端 Q 下 |
|---|---|---|
| accepted源与selection | `main-af77534-host-B/main-af77534/multi/selected/operators.py`；该`multi/`目录下`result.json`、`round-2.json` | B：`main-af77534/multi/selected/operators.py`与对应result |
| A侧冻结副本 | `main-af77534-host-A/main-af77534/multi/selected/operators.py` | A：同上相对路径 |
| 恢复执行源 | `recovery-1870e8e/host-A/heldout-recovery-1870e8e/rows/multi/operators.py` | A：`heldout-recovery-1870e8e/rows/multi/operators.py` |
| 原失败矩阵 | `main-af77534-host-A/heldout-af77534/matrix.json` | A：`heldout-af77534/matrix.json` |
| 有效矩阵及逐块记录 | `recovery-1870e8e/host-A/heldout-recovery-1870e8e/matrix.json`，同根`evaluation/raw/` | A：`heldout-recovery-1870e8e/matrix.json`和`evaluation/raw/` |
| 完整数值/来源审计 | `analysis.json`、`source-audit.json` | 本地衍生报告，不猜远端同名副本 |
| 导出索引 | `controller-af77534/export-index-{A,B}.json`、`export-validation.json`；`recovery-1870e8e/export-index.json` | 索引路径以导出收据为准，不直接替换本地host前缀 |

accepted call=`task_rewriter-9efd81e5`，session=`ses_f448e7ab0ffeO2Vxj1cM3USyrA`。agent源由B的`main-af77534/multi-agents/sandboxes/task_rewriter-9efd81e5/candidate/bundle/bundle.json`追踪；输入snapshot是该sandbox的`analysis/task_response.json#/bundle_sources`，可写mirror不是parent权威。其余五call/session及失败返回源hash全在`C/source-audit.json:sessions/guard_defect`，不需要打开完整会话。
source SHA256=`e4753894532cd479749a25774d46844f4760a8274f3956214934d38e04679380`；selection bundle=`8e37a73572d19b4d6379be6fc2af38ef4cb00775142ed0380feb933c97fecbea`；恢复execution bundle=`d80c70baad0962947afa935a01043d163a657140a8491fb00f6b0b4fb8cec1fb`。params为空`{}`，只有一个parameter-free native trial，不是大量TPE优化。
有效matrix SHA=`d14d11fe97b7e754c6b96219b57eead233de240c4e807073156fd5edb89ee4a8`；原失败matrix=`f66d72c69a9b513428fbe68bbe2ff33f88404d4fdf282bd2d761ed014ac3981a`。每个`cells[].evaluation.detail`是JSON字符串，包含**确切**`measurement_raw`与`quality_raw`绝对路径。例baseline/ttft/block0对应A的`Q/heldout-recovery-1870e8e/evaluation/raw/measurement-a2d7288b5c3649a9a269a688313ff476.json`及`quality-540be27fcb3149b89bfdb76289291cf5.json`；本地按上表矩阵根映射即可读token-ready时间戳和逐token losses。
恢复36quality+36measurement、84waves/168requests/15408tokens，10550model forwards；搜索4129forwards不含未知的原失败准备，不能求完整GPU总账。主index219、recovery export111、宽index128是嵌套范围不可相加。旧报告记录F1待审；本文不猜后续状态，也不把review状态当数值有效性的替代。

## 8. 当前 C3 fast：失败来源与 control/diagnostic 分账

`docs/result-c3-fast-kernel-iteration.md`与较新的`docs/handoff-c3-fast-kernel-iteration.md`：**NO_DEV_GO；formal/heldout NOT_RUN；efficacy NOT_ESTABLISHED**。最终review APPROVE仅批准如实交付失败，不是科学成功。F0/F1指开发framework，与最终review F1不是一回事。
两个fresh agent probe各480秒ReadTimeout，均compiled=true、JIT2/launch-count1、IR/局部数值/output-use证据存在，但`cuda_names=[]`，local invalid且latency=null，official model J=null。不能简写“没编译”，也不能用counter替代trace。剩余profiler lifecycle/correlation根因及timeout原因未知，不证明数值kernel有缺陷。

| 阶段 | 结果 / 用途 | 独立记账 |
|---|---|---|
| 初始人工setup | IR归属失败，拒绝保留 | 不是agent candidate |
| 996318d setup recheck | 同control的CUDA/IR/数值/output/suppression通过，wrong control拒绝，3模型检查通过 | setup累计3local/3model/3profiles，另4capture/preparation |
| Fresh F0 | trace空+transport失败，未完成返回 | 1agent / 3local / 0model charges |
| 原F0 owner-RPC replay | 不改source/bundle，local89.088us通过 | 0agent / 2local / 0model；不是整模J或增益，不追认F0 |
| Fresh F1 | owner PID:TID776849:776849仍trace空+timeout | 1agent / 3local / 0model；无repair/redraw |

开发合计2agent/8local/0model-evaluation，8local包含5local operations+3captures，不能再加3。0model charge不等于没跑GPU/模型forward；setup261/275forwards另计。D=2026-09-20T10:30:17.689003Z，连续90min到12:00:17.689003Z，工程准备在D前；无正式S、F*、三目标winner或新36矩阵。新42heldout只有冻结输入，没有分数；开发复用16条，容器58=16+42，不是58条新heldout。

### 8.1 失败源码确切位置

本地 `Klocal=results/c3-fast-kernel-iteration/`；远端K在A。F0执行SHA=`996318df6c7e2e9fcdfac3a8012d4e813b7e732b`，F1=`38e6e10de2e9770692f39ebb8eed86e11eb33df5`，公开报告revision=`8baa00800878ab2326010d89b3deed168e85a91f`。

| 资产 | 本地 Klocal 下 | A远端 K 下 |
|---|---|---|
| F0 helper-admitted源 | `setup-native-recheck-996318d/host-A/development-996318d/F0-evaluation/source-0-0/operators.py` | `development-996318d/F0-evaluation/source-0-0/operators.py` |
| F0失败workspace | 同上development根下`F0-evaluation/C0-workspace/candidate/bundle/operators.py` | 同development相对路径 |
| F0状态、配置、结果 | 同development根`framework-F0.json`、`target.json`、`dev-budget.json`、`dev-state.json`、`F0/result.json` | `development-996318d/`对应文件；远端账本后续含F1 |
| F1导出源 | `T4-native-38e6e10/F1-source/operators.py` | 完整失败workspace为`development-996318d/F1-evaluation/C0-workspace`；导出源以receipt/hash绑定，不猜workspace中最终文件名 |
| F1/helper/终态 | `T4-native-38e6e10/{F1-essential.json,F1-local.json,F1-result.json,terminal.json}` | local原件=`development-996318d/F1-evaluation/raw/fixture-only-0bd149f844c542049c9a7c4d455f9892.json` |
| 原F0诊断 | `T4-native-38e6e10/owner-rpc-diagnostic.json` | 同名T4根；不与fresh F1合并 |
| setup初次材料 | `setup-native-ad0694a/{operators.py,control-result.json,fixture-only-failure.json,profile-summary.json}` | `setup-native-ad0694a/`；bulk profile拷贝未全核验 |
| setup复核 | `setup-native-recheck-996318d/host-A/setup-native-recheck-996318d/{setup-result.json,F0-essential.json,execution.log}` | `setup-native-recheck-996318d/` |

F0 source SHA=`b6b0b447aef27031d0954de3595811e46f405ae81a8e1f8d9885425deefafedf`，bundle=`aba30792ba8a9fb488259ba26cac52a619f10df6fa837b39326738b7fb4c9c71`，params num_warps8。F1 source=`08363d2cb383451d5ffbd98a79410a1c5c5cfa8edfa766b3c25d0d0d19fbbdcb`，bundle=`68a03a7c2df2d6fff4cbd2ea41cde58f78b1aa98336d1621ae5203e5987e6321`，params空。不能把F0诊断成功与不同F1源拼成成功epoch。
F0 framework=`36c96bb19771aa171ac3c0c94849c2634c2383f345ed722011a5c482408b38be`，F1=`d278370b2ff8b9c0ec551dbc4b21c2d873b0f9f5ef755f3fdcfa182dc6c21b67`。完整F1 agents数据库/execution log不在本地T4镜像，完整原件位于远端T4及`development-996318d/F1{,-evaluation,-agents,-baseline}`。来源身份索引先读`F1-essential.json`，不为补session展开私有数据库。
本地F0预算是中途镜像，最终总账读T4 essential/terminal。oracle目录只导出`oracle.json`，不是完整f32 payload；formal连接也未完整就绪。已有模型位于两机`Q/assets/Qwen3-4B/1cfa9a7208912126459214e8b04321603b3df60c/`，不属于公开raw包。旧预算已过期，本目录不授权重跑脚本、resume、F2或新实验。

## 9. 论文图表与可写句

| 建议图表 | 取数与展示 | 可写 / 不可写 |
|---|---|---|
| 图1：T4四父×三组R及range | 第3节12行，全部父与负例；caption写observed range | 可：“本批固定状态完整响应通过对G0/G1的4/4、3/4门槛。”不可：“资源信息因果显著改善总体性能。” |
| 图2：信息收益与净elapsed分开 | T4旁列T5共同H的ready轨迹，晚到终点用不同标记 | 可：“信息方向信号未转为本次两任务净时间胜出。”不可：“C2普遍无效”或“只因probe开销”。 |
| 表1：后期全部pair | 第5节包含method/A正负、opportunity缺失、region单列、pilot四pair | 可：“收益依赖任务与生成轨迹。”不可删除task43或补齐censored制造总体PASS。 |
| 图3：机制链 | 固定点资源/latency、retune三seed、补域七产物、配置六cell分面 | 可：“可达性、调参和资源计数不足以单独保证竞争力。”不可把多参数bundle差异称单轴因果。 |
| 表2：历史性能与精度控制 | 同卡21 external、43 round-2、A800 E3独立行，注明compute/gate | 可：“指定候选的有界性能。”不可pool跨hardware latency或声称等精度/等预算搜索胜出。 |
| 图4：C1轨迹和纠正 | S2/step-3与window-1分开 | 可：“结构化反馈关联更早终点。”不可用伪noise floor、未施加ledger反馈或binding证明全局最优。 |
| 表3：C3两代状态 | 旧4×3矩阵及fallback controls；fast列null/NOT_RUN和失败证据 | 可：“旧源组合在模型级有部分吞吐收益，device目标尚未完成。”不可：“三目标GPU kernels均优化”。 |

质量说明必须随图表携带：C2历史混合精度、relaxed/FP64 rescue与C3 BF16整模质量不是同一政策。多块range、合成prompt和跨host观察均不能改写成CI或总体语义等价。成本图分别标elapsed、worker wall和provider字段，不能把重叠时间加总成GPU busy或完整货币账单。

## 10. 最小只读取数步骤与未覆盖项

1. **T4取一个kernel和正式分数：** 按第3节行键打开`L/results/c2-information-groups/HOST-A/P1-G2-519d037/retune-result.json`，读`selected.candidate_id/params.values`；打开同目录`child-selected.py`和`result.json`。需要原生成源时读`retune-manifest.json.inputs.source`，需要raw samples时沿parent/retune events中的trial/job链接。不要执行candidate。
2. **后期取实际incumbent：** 先读相应`export/selected.py`，不是只拿retune winner；用第5节manifest定位host和remote。region直接读`task21/setup.json`及`export-manifest.json`，pilot读`main/export-manifest-A.json`，不做本地前缀替换猜路径。
3. **旧C3取矩阵原始数据：** 打开第7节有效`matrix.json`，按row/goal/block读取`cells[].evaluation.detail`中的`measurement_raw/quality_raw`，对照`recovery-1870e8e/export-index.json`与本地host-A镜像。选source同时核对selection与execution bundle，不把search-only268.461921tokens/s替代heldout214.785688。
4. **当前C3取失败证据：** 只打开`T4-native-38e6e10/F1-essential.json`、`F1-local.json`、`F1-source/operators.py`和`terminal.json`，F0沿第8节嵌套目录；不要运行`run.sh/run_f1.py`或恢复过期budget。
5. **历史取候选：** 先读本地manifest和表中冻结winner；archive先以已有MANIFEST核对成员路径。包缺jobs就记录缺失，不把报告里的script locator冒充raw，也不为找配置复制完整sandbox。

上述步骤是读现成文件的路线，不是新实验或重算脚本；路径模板中通配符表示由manifest/result选定的已有文件，不保证每个历史run都有同名产物。

明确缺口：最早L1/L2及9月5日前L3、早期四run完整raw、两个box1成熟48镜像、S2完整源码/raw、E1 resume末阶段及后续rerun、S7最终闭合、G9逐attempt映射、external原始timing输出、window-2 M1/M2b完整终局和外围诊断表均未全面审计。外部L2包存在但不在本次优先L3核对范围，不能直接升级论文证据。
配置/身份缺口还包括首轮和unlocked runtime文件名、window-1 dirty启动补丁/实际每trial dtype/worker版本/reuse来源指针、完整provider账单、全部reference字节hash和shape复现链。unknown/null不补零。旧receipt的全量hash验证、本次最小metadata抽查和live检查是三种不同证明；本文只做前两者的引用与核对，没有第三者。

### 可选深层索引，不是读懂本文的前置条件

- `.omo/evidence/paper-data-inventory/core-c2.md`：早期C2逐阶段行号、共享关系和质量账。
- `.omo/evidence/paper-data-inventory/later-c2.md`：method/core全部11 child、A全部7 child、agent/session与worker链。
- `.omo/evidence/paper-data-inventory/historical.md`：历史报告族映射、更多event/archive hashes与纠错出处。
- `docs/handoff-c3-fast-kernel-iteration.md`：当前C3技术边界与旧失败位置；其中未来操作说明不是本次授权。

本次校验范围：完整读取三份输入清单（含historical末尾）、直接核对T4主表12行与门槛、读取两代C3结果和fast交接，并只读补查C3源码审计/目录/矩阵/导出索引。没有重新计算性能、全量hash、统计量或独立实验总数；没有Git、SSH、GPU、测试、依赖安装或源码修改。文档中的既有测试通过、清理、APPROVE和核验计数均属于原收据，不冒充本次执行结果。
