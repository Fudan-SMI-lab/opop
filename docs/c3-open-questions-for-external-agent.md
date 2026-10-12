# C3 待确认问题清单(发给外部 agent)

## 0. 我方目标与范围声明(先读,它决定哪些问题重要)

**唯一核心目标:产出 §4.3 那张真实设备 kernel 的三目标交叉表**,即 `docs/paper-positive-evidence.md` §4.3 当前的 `[PLACEHOLDER]`。表的形状按你们 `c3-goal-workflow-status.md` 第 5 节:4 行(baseline + 三个目标产物)× 3 列(TTFT / single / multi)× 3 blocks = 36 blocks。

**明确缩小范围:C3-A 不做。** 从"模型输入 + 自然语言目标指示"自动构建任务与评测器(目标指标如何检测、优化哪些算子、自定义算子如何接入)这一整步**本轮不实现**,由我们**手工产出**其成果物,直接把框架接到算子优化流程的入口。

因此本轮要验证的科学命题收窄为:**在任务定义明确、评测器正确的前提下,框架能否针对不同的声明目标产生有差别的优化效果。** 这一命题的证据就是 §4.3 那张表,可直接用于 paper。

**我方对已有 §4.1 的判断(供你们校准,不必辩论):** 它是纯 Python 层 decoder composition 改写,不涉及算子;而我们是算子优化框架。我们另外核算了该矩阵 36 个 cell 的 `execution_bundle_sha256`:`baseline` / `ttft` / `single` 三行同为 `9e4ae8972963`(即 ttft 与 single 两行是 baseline 字节的 fallback),只有 `multi` 行是真产物 `d80c70baad09`,而 +12.879% / +9.095% 两个数字都出自这一行。所以它不作为 C3 的主要证据,只作历史边界保留。**如果这个核算有误请指出。**

下列问题按"是否阻塞 §4.3"排序。每题都请给**可核验的答复**(文件路径 + 函数名 + 行号,或明确的 `不存在` / `未实现` / `未知`),而不是能力描述。

---

## 1. 阻塞级:两条代码路径的打通方式(最重要的一组)

我们在当前 worktree(`38e6e10` + 本地未提交改动)读代码得到如下结构结论,**请逐条确认或纠正**:

| | fast 路径 `fast_cli.py` | operator 路径 `operator_cli.py` |
|---|---|---|
| 子命令 | `seal / target / develop / probe / resume` | `readiness / optimize-goal / search / heldout / analyze` |
| agent | `ModelOperatorRewriterAgent`(硬接线) | 可选 `--execution-profile model_operator` |
| 设备 kernel 准入 | **有**:`device_gate.require_device_proof` | **未发现**:全路径不出现 `device_kernels` |
| 产物记录 | `fast_records.Artifact`,含 `kernels: tuple[DeviceKernelDeclaration, ...]` | `search_records.Selection`,**无 kernels 字段** |
| 36 格终测 | **未发现** | `search_matrix.run_matrix`,已产出过 §4.1 |

两条路径在 `examples/c3_qwen3/` 内的唯一共享类型是 `SearchClock`。

**Q1.1** 上表是否准确?特别是:`operator_cli` 的 `search` / `heldout` 链路在任何配置下会调用 `require_device_proof` 吗?我们 grep 到它只有两个调用点(`fast_evaluation.py:149`、`local_eval.py:89`),而 `local_eval.evaluate_local` 只被 `fast_evaluation.py:20` 导入。若确实如此,则 §4.1 那张表在结构上无法证明任何一行含真实设备 kernel —— 这是否与你们的理解一致?

**Q1.2** `fast_development.freeze_framework`(第 82 行)与 `fast_development.formal_goal`(第 102 行)我们在整个仓库(`examples/`、`src/`、`tests/`)grep 到**零调用者**,也没有对应的 CLI 子命令和测试。这是否就是 status 文档所说"最小 driver 须完整接通"的字面所指?**除这两个函数外,formal 链路还缺哪些未接线的部件?** 请给一份缺失件清单。

**Q1.3 (设计裁决,最关键的一题)** 要让 36 格表的每一行都能自证"该行确实含设备 kernel",你们建议哪条路?
  - **(a)** 给 `Selection` 增加设备证据字段,并让 `run_matrix` 强制校验每个非 fallback 行 —— 我方倾向此项,理由是 §4.1 的教训正是"表看起来 4 行、其实 2 行是 baseline 字节",表本身必须能自证;
  - **(b)** 在 fast 路径内新写 36 格终测,完全不复用 `run_matrix`;
  - **(c)** 你们已有的、我们没找到的第三种方案。

若选 (a),`Selection` 是 `FrozenRecord`,新增字段是否会破坏既有 `export_selection` / `load_selection` 的 sha 校验与旧 §4.1 产物的可读性?**有没有会静默失效的兼容点?**

**Q1.4** `run_matrix` 需要 `Readiness`(经 `prepare_readiness`)、`ready.oracles["calibration"]`、`ready.profiles`,并校验 `paths <= traced`(第 139 行:final instrumentation 不得含未 traced 模块)。**fast 路径产出的 target/profile 能满足这套 Readiness 校验吗?** 现有 `setup-native-ad0694a` 的 profile 只覆盖 `goal: "ttft"`(我们读 `profile-summary.json` 确认),而 §4.3 要三个目标各自的行。**三个目标需要三份独立 profile,还是一份 TTFT profile 即可?** 若需三份,这算几次 profile 额度?

---

## 2. 阻塞级:根因与我方已做修复的复核

我们在本地做了三步修复(**尚未提交、尚未同步到任何机器**,box4 仍是 `38e6e10` / dirty=0)。CPU 回归 681 passed(C3 聚焦 56 + runtime 相邻 378 + C2 路径 247)。

**修复内容:**
1. `device_records.py` / `local_device.py`:`KernelEvidence` 增加 `trace_events` / `trace_cuda_events` / `trace_sample` 三个**纯诊断**字段(`require_device_proof` 不读),`prove()` 只读一次 trace 并填充,用于区分"profiler 收到 0 个事件"(收集侧)与"收到 N 个但名字不含声明 kernel"(匹配侧)—— 对应你们诊断顺序第 2 步。
2. `fast_workflow.scoped_config` **删除** `request_timeout_s = 480` 的硬写,改为继承 config 声明值;`fast_generation.py` 的 `prompt_timeout_s` 改为从 `agent.client.timeout_s` 派生。
3. `src/kernel_optimizer/agents/runtime.py`:`OpencodeServer.start()` 的 Popen env 剥离 23 个 CUDA/Triton/profiler-only 变量(`server_env` 仍叠加在后,显式设置照旧生效)。

**Q2.1(根因假设复核)** 我们的判据是 6/6 相关性:4 次**未**起 opencode 子进程的 `prove()` 全部拿到 `cuda_names`(含我们在 box4 上复核的 owner-RPC 诊断 = `["rmsnorm_kernel"]`),2 次起了的全空。机制假设是 **CUPTI activity-session 冲突经继承的 CUDA 环境**。你们在 F0/F1 现场是否有能**支持或否证**这一假设的证据(例如 opencode server 启动与 profiler enter/exit 的时序、server 进程是否出现在 CUDA context 中、`nvidia-smi` 期间快照)?我方前两个假设(`compile_wall`、owner thread)都被自己的数据否证过,这第三个同样只是假设。

**Q2.2(判据陷阱,请特别确认)** `fast_artifacts.identify_framework` 的 hash 文件集**不含** `src/kernel_optimizer/agents/runtime.py`(我们实测:另外 4 个改动文件都在集内)。因此**根因修复本身不改变 framework_id**,F2 的 framework_id 变化只反映诊断字段与 480 的移除。这意味着"修复是否生效"的唯一判据是 receipt 里 `cuda_names` 非空 —— 确认这个理解,并说明这是否算 framework 身份覆盖面的缺口(即 agent runtime 变更不进入 framework 身份)。

**Q2.3(480 的第二处)** `setup/contract.json` 的 `stage_budgets.common.agent_call_hard_cap_s = 480`。我们实测 `FastPrepared.agent_context()` 弹出 `stage_budgets` 后只保留 `development` / `formal` 两键,`common` 整块被丢弃(facet 内搜不到 `480` 也搜不到该键名),且 `examples/` 与 `src/` 中无人读取此键 ⇒ **它既不执行也不传达**。同时该文件被 `CONTRACT_SHA` 钉死,改动会让 `prepare_fast` 抛 `RunnerError`。请确认:(i) 该键确为死值;(ii) 在**不破坏 contract sha** 的前提下,正式实验应如何声明真实的 agent 调用上限。

**Q2.4(实测依据请核对)** 我方删除 480 的依据是:window-2 两个 C2 run 的 rewriter 每次调用墙钟中位数 **519.2s / 736.2s**、最大 **902.8s / 1306.3s**;`config.py` 默认 1500.0 的注释记录"最慢一次成功调用 1167s,留 29% 余量";而 `idle_abort_frac = 0.5` 是**比例**,所以 480 同时把静默容忍砍到 240s。另:F1 总墙钟 481.2s,local gate 在 t=422.2s 返回,只剩 57.8s。**这些数字与你们的记录一致吗?** 你们当初选 480 是否有我们没看到的理由(例如某个外部服务端上限)?

**Q2.5(最小判伪实验)** 我方设想:在已通过的 owner-RPC 诊断中,于**同一个** `prove()` **之前**起一个 opencode server,看 `cuda_names` 是否翻空。成本 1 次 local、**零 agent 调用**。你们认为这个设计能干净地判伪该假设吗?有没有更省或更可靠的对照?**它是否需要单独授权与新输出根**(我们理解旧 ledger 已过期 3h37m 以上、按你们的边界不可复用)?

---

## 3. 阻塞级:DEV_GO 的可达性

`prove_candidate`(`fast_proof.py:28`)要求:agent 作者 + session_id + 有效 model J(score > 0)+ `local_checks_passed == 1` + `logits_relative_l2_max ≤ .01` + `paired_mean_nll_delta ≤ .02` + `completed_requests == goal.requests` + `output_tokens == requests × output_tokens` + 同 stage/bundle/params 的 local receipt + 可核对的 `fast-model` receipt。

**Q3.1** 你们 checklist 里那条"可信实测 local headroom"(收益须超 A/A 噪声)与 `prove_candidate` 不检查 headroom 之间的差别,在**本轮**是否要求补齐?换言之:若 fresh agent 过了 `require_device_proof` 四道门与 `prove_candidate`,但局部收益**不超过** A/A 噪声,我们该判 DEV_GO 还是不判?**这直接决定第一次 develop 的成败口径。**

**Q3.2(承重数字冲突)** handoff §5 记 reference A/A noise = **0.192001462us**,而 `T4-native-38e6e10/F1-result.json` 里是 **1.9359961152**(相差 10 倍)。它是 `local_gain` 的分母,会直接改变"是否超噪声"的判定与 §4.3 将来的增益口径。**哪个是对的?另一个是怎么产生的?**

**Q3.3** 现有 target 是 TTFT、site `pure-727f184a...a1fa0`(73 个兼容 RMSNorm leaves)、4096-token representative 关联 CUDA 14041.134us、exclusive fraction **unknown**。对一个 exclusive fraction 未知的算子,**框架能否判定目标改善的归属**?§4.3 需要三个目标各自的产物 —— **三个目标应共用这一个 RMSNorm site,还是各自重新选 site?** 若各自选,`build_target` 的 `required = 1 if goal == "ttft" else 2`(两个 representative,decode_step ∈ {0,127})意味着 single/multi 需要**含 late-decode 的 profile**,现有 profile 是否具备?

**Q3.4** DEV_GO 只需在**一个**目标上达成,还是三个目标各需一次?`freeze_framework` 只读**单个** `result_path`(一个 DEV_GO)但要求 `targets` 三个齐全 —— 确认:**一次 DEV_GO 即可冻结 F\*,三目标产物在 formal 阶段各自新生成**。

---

## 4. 授权与预算(要执行就必须先明确)

**Q4.1** 完整走到 §4.3 需要几段授权与几个预算根?我方理解是三段:(i) 判伪观测(1 local);(ii) 开发直到 DEV_GO(新 D);(iii) formal campaign(新 S)。**能否合并?**

**Q4.2** formal 预算按 contract 是 agent 6 / local 24 / model 12、每目标墙钟 5400s、cutoff S+9000s、final deadline S+10800s、`independent_final_cell_blocks = 36`。加上 36 格终测本身,**总墙钟估计是多少**?我方粗估 3 小时以上,请给你们的估计。

**Q4.3** 开发阶段需要几轮?F0、F1 各消耗 agent 2 / local 8。若第一次 develop 仍失败,**按什么判据决定"继续修框架"还是"停止并如实记录"**?

**Q4.4** 哪些既有输入获准复用(contract / preflight / development-corpus / target / native profile / calibration oracle),哪些必须新造?我方读到 calibration oracle 的本地副本只有 `oracle.json`、未导出完整 f32 payload ⇒ **本地不足以跑模型,必须用远端那份**。确认。

**Q4.5** 服务器上"已批准的私有 AppConfig"的**真实绝对路径**是什么(你们文档里是 `/ABS/approved-server-private-config.yaml` 占位符)?它是否仍获批准、`request_timeout_s` 当前值是多少?**这项不给,develop 无法启动。**

---

## 5. 环境事实核对(我方实测 vs 你们记录,有冲突)

我方 2026-09-20/21 实测 box4(`connect.bjb1.seetacloud.com:37752`):

| 项 | 你们记录(handoff §4) | 我方实测 |
|---|---|---|
| 模型栈 | Torch 2.13.0+cu129 / Triton 3.7.1 / Transformers 5.16.1 | `orch-venv`:**Torch 2.14.0+cu130 / Triton 3.8.0** |
| A 的 native Python | `/root/autodl-tmp/kernel-opt-venv/bin/python` | 该机上可用的是 `orch-venv`;你们警告"不要混用两套" |
| GPU | A0 = RTX4090 | **两张** RTX4090,均 0 MiB、无 compute apps |
| 代码根 | `/root/autodl-tmp/c2-local-experiment/code/v5` | 存在,`38e6e10` / dirty=0,**但 23M,不含 C3 证据树** |
| C3 运行根 | `/root/autodl-tmp/c3-fast-kernel-iteration/` | 存在且为 **179M**,与代码根是两棵独立的树 |
| contract sha | `2c0e2fc7...996bd8` | 远端逐位一致 ✓ |
| 模型资产 | `.../assets/Qwen3-4B/1cfa9a72.../` | 存在,14G ✓ |

**Q5.1** 我连的这台(`bjb1:37752`)到底是 A 还是 B?你们 handoff 写 A 的入口是 `bjb1:37752` 但 native Python 是 `kernel-opt-venv`;而这台机器上可用的是 `orch-venv`(那被你们标为 B 的解释器)。**跑 §4.3 应当用哪台机、哪个解释器?**

**Q5.2** Torch 2.14+cu130 / Triton 3.8.0 与你们记录的 2.13+cu129 / 3.7.1 不同。**F0/F1 是在哪个栈上跑的?** 换栈会不会改变 profiler/CUPTI 行为(这直接关系 Q2.1 的根因)?**§4.3 是否必须在与 F0/F1 相同的栈上跑**,以免表格与历史证据不可并列?

**Q5.3** 两张 4090 都空闲。§4.3 的 36 格终测要求"同一 A0 物理 GPU 串行"。**能否把 develop/诊断放在卡 1、终测独占卡 0**,还是必须全程单卡串行?(我方已有纪律:同机隔离只靠 `CUDA_VISIBLE_DEVICES`,GPU 锁是 per-run 不是 per-device,两臂两把锁从不竞争 —— 所以"有锁"不构成隔离依据。)

---

## 6. 表格口径(现在定,免得跑完才发现不可用)

**Q6.1** §4.3 每行必须附:bundle/source SHA、选中 target、最终参数、设备证明、模型质量、是否 fallback。**设备证明具体落在哪个文件、以什么字段呈现**,才能让论文读者核验"这一行真的含设备 kernel"?

**Q6.2** 若三个目标最终选出**同一个** winner,或某目标回退到 baseline,**表还能支持"针对不同目标产生有差别优化"这一命题吗?** 若不能,你们建议补充什么证据(例如各目标的搜索轨迹差异、被拒配置)?

**Q6.3** 增益口径(正数为改善)我方按你们 status 第 5 节:TTFT 用 `(base−row)/base`,single/multi 用 `row/base − 1`。**每格 3 个 block 的汇总用 median 还是 mean?** (我方纪律:published `speedups` 是 mean/mean 且被 KernelBench 存成 `.3g`,分辨率地板约 1.3%,不可作差 —— §4.3 请勿沿用该字段。)

**Q6.4** 若某格无效或未跑,你们要求"标无效,不填 0"。**论文表格里该显示什么**(`—`、`invalid`、`NOT_RUN`)?几个无效格以内表仍可发表?

---

## 7. 我方将在得到答复前并行推进的事(告知,非提问)

- **不执行任何 GPU 运行**,不同步代码到任何机器,不碰旧 ledger,不重置 D,不启动 F2。
- 三步修复保持未提交状态,等 Q2.1 / Q2.2 的答复再决定去留。
- 若 Q1.3 得到裁决,我方可在**不烧 GPU、不需新预算**的前提下先写 formal driver(`freeze` / `formal` / `matrix` 三个子命令 + `Artifact`→`Selection` 的设备证据承载),这是当前唯一能并行推进的实质工作。

---

## 8. 最短答复优先级

若时间有限,请**至少**回答这五题,它们各自单独就能阻塞 §4.3:

1. **Q1.3** —— 36 格表如何承载设备证明(设计裁决)
2. **Q1.2** —— formal 链路缺失件清单
3. **Q3.1** —— DEV_GO 是否要求超噪声 headroom
4. **Q3.2** —— A/A 噪声 0.192 还是 1.936
5. **Q4.5** —— 已批准私有 AppConfig 的真实路径
