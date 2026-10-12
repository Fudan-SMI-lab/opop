# C3 外部 agent 问题答复

## 先答五个紧急问题

1. **Q1.3：建议 a'，复用矩阵排序、分组与计量逻辑，增加显式 device 实验 profile、fast preparation/evaluator adapter 和版本化 proof sidecar。** 不另写第二套矩阵引擎，不原地迁移旧 Selection 或 bundle 哈希。非 fallback 行必须闭合设备证据；baseline/fallback 明示例外但仍需身份、质量和新测量。依据 `examples/c3_qwen3/search_matrix.py::run_matrix:74至188`；兼容细节见 Q1.3。
2. **Q1.2：两个 helper 不等于 formal driver。** 最少补齐 headroom 准入、框架源码闭包、持久冻结、三目标 fresh 编排、新 contract/readiness/profile/oracle adapter、可移植 proof 导出、final proof 与三臂终止屏障。`examples/c3_qwen3/fast_development.py::freeze_framework:82至99`、`formal_goal:102至120` 只有定义，证据笔记的仓库检索未发现调用者或专门测试；逐项缺口见 Q1.2。
3. **Q3.1：无可信实测 local headroom，就没有协议合格的 DEV_GO。** 即使当前 `run_probe` 返回该字符串也不够。`examples/c3_qwen3/fast_search.py::local_gain:17至18` 要求每个 fixture 的中位延迟差严格大于对应 A/A 噪声；`C/fast_proof.py::prove_candidate:27至50` 尚未检查它。应在 development/freeze 准入补齐，而非放宽口径。
4. **Q3.2：两个值都对，但不属于同一次 A/A。** F0 为 **0.19200146198272705 us**，F1 为 **1.9359961152076721 us**；路径、哈希、framework 均不同。F1 原始 A/A 数组不在本地，不能独立重算。噪声是差值阈值，不是分母；下文给出完整身份。
5. **Q4.5：历史实际路径是 `/root/autodl-tmp/c3-fast-kernel-iteration/development-996318d/agent-config.json`。** `results/c3-fast-kernel-iteration/T4-native-38e6e10/run_f1.py:34至35,48,52` 与 `D0/run.json:4` 的配置路径记录可核验。仅证实曾使用，当前存在性、批准状态、文件内 timeout 均未知；未读私有配置正文。

## 证据范围与引用约定

核对日期：**2026-09-21，本地只读证据口径**。本答复只新增本文，不修改代码、测试、旧报告、问题清单、封闭状态或预算账本；没有部署、Git、SSH、GPU、provider 调用、新测试或新远端证明。
直接依据 `.omo/evidence/c3-external-questions/architecture.md` 与 `runtime-facts.md`，并逐题读取 `docs/c3-open-questions-for-external-agent.md`。下面引用其已核对的源码位置与记录字段；本次不重新计算历史文件哈希。
问题清单第 19、48至53 行所述 `38e6e10` 已发布基线与本地未发布修改严格分开；本地源码行号可能漂移，不能作为已远端部署的证明。历史报告锚点 `8baa008` 也不代表当前 dirty 改动已发布。
本文是工程建议与事实答复，**不是实施、实验、预算、时钟重置或私有配置复用授权**。C3-A 不做；范围仅为明确任务下的设备 kernel 三目标交叉证据，不建议建设通用测试平台。

所有路径相对 `D:/Pyhon_projects/opop/v5`。为缩短引用，以下缩写可逐字展开，`文件::函数:行号` 或 JSON 字段均为核验入口：

| 缩写 | 完整路径 |
|---|---|
| `C` | `examples/c3_qwen3` |
| `R` | `results/c3-fast-kernel-iteration` |
| `F` | `results/c3-fast-kernel-iteration/T4-native-38e6e10` |
| `D0` | `results/c3-fast-kernel-iteration/setup-native-recheck-996318d/host-A/development-996318d` |
| `U` | `results/c3-fast-kernel-iteration/setup-native-recheck-996318d/host-A/setup-native-recheck-996318d` |
| `O` | `results/c3-qwen3-4b-operator-goals` |
| `M` | `results/c3-qwen3-4b-operator-goals/recovery-1870e8e/host-A/heldout-recovery-1870e8e` |
| `K` | `results/c3-qwen3-4b-operator-goals/recovery-1870e8e/host-A/recovery-1870e8e` |
| `E` | `.omo/evidence/v5-c3-fast-kernel-iteration` |
| `P` | `.omo/plans/v5-c3-fast-kernel-iteration.md` |
| `handoff` | `docs/handoff-c3-fast-kernel-iteration.md` |

## Q0：旧矩阵核算与科学边界

**所述 execution-bundle 分组正确。** `M/matrix.json` 的 `cells[*].{row,goal_id,block,execution_bundle_sha256}` 共 36 条，每行 9 条：
- baseline、ttft、single 共 **27** 条是 `9e4ae8972963bf0acadfc1664e9e091c464ee1510aff336dbd27af179db7d9f8`。
- multi 共 **9** 条是 `d80c70baad0962947afa935a01043d163a657140a8491fb00f6b0b4fb8cec1fb`。
- multi 原 selection bundle 为 `8e37a73572d19b4d6379be6fc2af38ef4cb00775142ed0380feb933c97fecbea`；恢复执行时公共 instrumentation site 重映射产生 `d80...`，不是重新选 winner。见 `O/source-audit.json:40至54` 的 `winner.selection_bundle_sha256`、`winner.recovery_execution_bundle_sha256`。

**+12.879174% single 与 +9.094619% multi 都来自 multi-selected 同一行**，ttft/single 为 baseline fallback。见 `docs/result-c3-qwen3-4b-operator-goals.md:48至61,74至84`。
但“不涉及算子”不准确：它是使用既有 GPU 算子的 agent-authored Python decoder composition/dispatch 改写，包含七次 `F.linear`、eager RMS、SiLU 与配置的 attention；不是 CPU-only，也不是新写 Triton/CUDA kernel。见 `O/source-audit.json:46至48`、`.omo/evidence/v5-c3-qwen3-4b-operator-goals/T9-source-audit.md:79至95`。
旧表保留为历史边界，不能补充为 §4.3 的新设备 kernel 主证据。

## Q1：两条路径与最小 formal 接线

### Q1.1 路由是否准确？

**表格主体正确，“唯一共享类型是 SearchClock”错误。** CLI 模式见 `C/fast_cli.py::Options/main:24至25,47至95`、`C/operator_cli.py::Options/main:39至50,72至159`；fast 的 model_operator 路由见 `C/fast_runtime.py::open_generator:108至119`、`src/kernel_optimizer/wiring.py::build_task_rewriter:187至200`。
operator search 走 `C/operator_search.py::optimize_goal:107至142` → `C/search_session.py::OperatorSession._evaluate:185至188` → `C/manual_task.py::evaluate:25至46` → `C/model_runner.py::ResidentRunner.quality:104至109` → `C/runner_quality.py::evaluate_quality:25至32` → `C/binding_runtime.py::ModelBinding.validate_local:203至216`，是数值 fixture 检查。
heldout 在 `C/search_matrix.py::run_matrix:178至180` 仍走同一 evaluator；**内建配置没有插入设备证明门**。选 device-capable agent 只改变生成，`C/search_helper.py:35至62` 的 session 分支还拒绝 `fixture_only`，不是旁路准入。
fast 的生产调用点是 `C/local_eval.py::evaluate_local:89` 与 `C/fast_evaluation.py::proven_local:141至151`。测试/控制脚本也调用 local evaluator，因此“全仓库只有 fast_evaluation 导入”过强。
共享项还包括 FrozenRecord、GoalId、RunnerError、ResidentRunner、BundleSpec/load_bundle、ParamSet、TaskEvaluation、HelperService，见 `C/fast_evaluation.py:10至22`、`C/fast_artifacts.py:18至20`、`C/search_session.py:14至22`。
`C/fast_records.py::Artifact:44至55` 有 kernels；`C/search_records.py::Selection:70至82` 没有。当前没有 Artifact→Selection proof bridge。旧表不能证明 authored device kernel，不等于旧代码不会调用 CUDA。

### Q1.2 除两个 helper 外缺什么？

**是，这正是“最小 driver 须完整接通”的实质。** `C/fast_cli.py:15,47至62` 未暴露 freeze/formal/matrix；笔记的 `examples/src/tests` 检索未发现两个 helper 的调用者或专门测试。不是说 fast 组件完全没测试。

| 最小缺失件 | 现有接口及应补齐的责任 |
|---|---|
| 准入与持久 F* | `C/fast_development.py::freeze_framework:84至99` 只复核当前候选谓词；补 headroom、目标 key↔goal、原 baseline、实际源码/config 身份闭包及证据依赖封存，不能信任调用方 DEV_GO 字符串。 |
| 三个 fresh formal lane | `C/fast_development.py::formal_goal:102至120` 只是单目标 helper；`C/fast_runtime.py::RunInputs:39` 的 frozen_framework 尚未被消费。接原 baseline、新 session、每目标 ledger、共同 S、accept/confirmation、fallback 与三臂 terminal 屏障。 |
| proof 可移植导出 | `C/fast_artifacts.py::write_artifact/read_artifact:93至110` 只导出 Artifact/bundle sources；`C/fast_records.py:44至65` 的 local/model/confirmation 依赖未形成导出闭包。不能留下临时目录悬空路径。 |
| 新 contract/preparation/evaluator | `C/runner_prepare.py::prepare:46至51` 钉旧 `9d1ef6…`；`C/fast_prepare.py::prepare_fast:16,68至112` 钉新 `2c0e2f…`。`C/manual_data.py::admit_files:58至85` 另钉 interface/corpus/oracle，`C/search_session.py::__init__:63至68` 还依赖旧 proposed-prompts/manual evaluator。 |
| readiness/profile/oracle | `C/search_readiness.py::prepare_readiness:58至85` 要三目标 profile、每目标两次 A/A 与 10 个 oracle 条目。接 device schema、原 calibration、final 每目标/block oracle 与完整 f32 哈希；`C/fast_prepare.py:106至107` 搜索态刻意剔除 final groups。 |
| final proof 与访问门 | `C/fast_budget.py::StageBudget:50至55` 有 final_verification/final lane，但未发现生产 final-verification driver。`C/fast_runtime.py::open_evaluator:63` 不传 formal_complete；`C/fast_prepare.py:72至75` 会拦 final。供应 boolean 不是三臂终止证明。 |
| final 执行与报告 | `C/fast_model.py::evaluate_fast_model:47至53` 不是完整设备门；`C/search_matrix.py:152至157` 重映射 bundle。接目标形状 proof、reference/quality、执行身份映射、36 blocks 与 provenance-aware report；`C/search_analysis.py:35至58` 只有数值汇总，未验设备证明。 |

实现建议仅含这些接口的聚焦验收：旧字节兼容、headroom、freeze mutation、receipt mismatch、fallback、阶段隔离/截止与合成完整 driver；不是新测试平台，也未授权本次实现或测试。现有 final 常量不等于用户批准新额度。

### Q1.3 设计裁决与静默兼容风险

**选 a'：保留现有矩阵语义，新增显式 device profile + fast adapter + 版本化 sidecar。** 复用 `C/search_matrix.py::run_matrix:74至101,128至188` 的排序、分组、导出与计量，不原样调用旧 heldout evaluator，也不 fork 第二引擎。
可新增 `device-experiment-v1.json`，逐行引用冻结的 selection、artifact、target/profile、local/model/confirmation、最终执行证据及文件哈希；这是**建议文件名，当前不存在该完整契约**。设备 profile 对缺失/未知 proof 版本 fail closed。
baseline/显式 baseline fallback 不要求 authored replacement-kernel proof，但仍要身份、质量、新测量，不能写成 device-proven winner。

兼容不能靠“字段 optional”保证：
- `C/runner_records.py::FrozenRecord:22至23` 未设 `extra="forbid"`，旧 reader 按 Pydantic 默认**静默忽略未知字段**，可能把 proof 读掉后导出无证明记录。
- `C/search_bundle.py::export_selection:80至94` 验 bundle/source/effective params，复制源与 bundle，重写含迁移后绝对路径的 selection JSON；`C/search_matrix.py::load_selection:58至65` 也重定位路径，不验设备证据或 selection 文档内嵌哈希。
- 新 optional 字段的 null/default 序列化与路径重定位均可改变字节。`C/manual_data.py::fingerprint:25至26` 哈希原始字节，`run_matrix:116,185至186` 钉 selection 文件指纹，不能重新序列化后声称旧外部 seal 未变。
- `C/model_binding.py::load_bundle:46至68` 对 `BundleDocument.model_dump(mode="json")` 与 source hashes 做 canonical hash；`BundleDocument/Site:16至32` 禁止 extras。给模型加默认字段也可能改变旧输入的 canonical dump/hash，**不要把 proof 塞进旧 bundle 或改旧哈希算法**。
- `C/search_matrix.py:182至184` 的 **selection_sha256 实际赋值为 selected.bundle_sha256**，不是 selection.json 文件哈希。保留旧字段含义，sidecar 用明确的 selection_document_sha256；另保留重映射后的 execution_bundle_sha256。

原 §4.1 文件与哈希必须逐字节保留，新 sidecar 引用旧身份不等于追认旧行设备证明。若以后扩展 Selection 字段，仍须显式版本 dispatch 与双向读写、默认值、未知字段丢失、重定位兼容证据。

### Q1.4 FastPrepared 能直接满足 Readiness 吗？

**不能。需要 adapter；“summary 只有 TTFT”也不成立。** `C/fast_target.py::DeviceProfile/ProfileData:44至54` 是 valid/goal/trace_windows/calls/site_kernels/cuda_intervals；旧 `C/search_session.py::ProfileDocument/ProfileData:25至33` 是 binding/sources/trace/goal，`set_profile:77至82` 还验 goal/contract。
`C/search_matrix.py::run_matrix:108至109,132至139,162至169` 要 prepared 身份、calibration、traced coverage、每目标/block oracle。伪造 `Readiness(valid=True)` 或把 TTFT 文件挂三次都不成立，旧 contract pins 也未解决。
需要**三个目标各自的 profile 观测**，不是必须三个不同算子或三个独立容器。`P:84至86,126,148` 定义三次 setup 窗口；`C/device_profile.py::profile_goal:38至43,82至90` 一次处理一个 goal/window。
`R/setup-native-ad0694a/profile-summary.json:3至8,918至923,1288至1293` 包含 ttft/single/multi；single 的 `lm_head` step 127 在 `970至989` 可见。仅 raw `short-device-profile-17cb1c7131914914bcb7bf55bb0fd2e8.json` 本地可得；single `3c0828…`、multi `d55821…` raw 未找到，完整 late-decode/site/ABI 覆盖未知，不可说从未有过。
`C/fast_target.py::build_target:93至96` 要 TTFT step 0、decode goals 的 {0,127} 两个 representative。从合格旧 profile CPU 构建 target 不消耗新的 profile 调用；**新 profiling 有成本**。`C/fast_budget.py::StageBudget:40至59,100至107` 的历史额度为 setup profile 3，development profile 还计 model，formal-search profile 0；没有免费重开 setup。

## Q2：根因假设与三处本地修改

### Q2.1 CUPTI/environment 假设有证据吗？

**仍未知，6/6 相关性不是根因证明；本地未独立核验那六例完整台账。** 现有证据是：
- `U/setup-result.json` 的 `local[*]` 与 `U/execution.log:5至10`：fresh F0 前正确 control 通过、错误 control 数值失败。
- `R/setup-native-recheck-996318d/F0-essential.json:25至63`、`U/execution.log:15至16`：F0 compiled，但 cuda_names 空，profiler PID:TID 为 `772107:772230`。
- `F/owner-rpc-diagnostic.json:2至26,147至174`：原 F0 source/bundle 在 owner RPC 诊断得到 rmsnorm_kernel；只属诊断，不是候选成功。
- `F/F1-local.json:13至36,92至117`：fresh F1 在 owner `776849:776849` 仍 compiled 且 names 空，说明 owner routing 不是充分修复。

`F/run_f1.py:75至111` 记录 capture→RPC diagnostic→reset/attach→fresh F1 顺序，但没有同步的 server PID、CUDA context 与 profiler 生命周期时间线。F0 log 的打印时间没有显式时区；F1 原 execution.log 未在本地镜像，`F/F1-essential.json:58至70` 是 agent 调用时间，不是 server startup。
仍须区分 Torch/Kineto/CUPTI 初始化与重复 session 状态、线程/flush/event matching、server 或 agent tool-child 交互。环境字符串继承本身不转移父进程的 profiler session；实际子进程 CUDA/CUPTI 行为与具体库版本未知。结束时 0 MiB/无 compute apps 不能排除中途 context；`F/gpu-after.txt:1至3` 的 A0 最后利用率还为 10%。

**三处 patch 的有限评估：**
- trace 字段位于 `C/device_records.py::KernelEvidence:97至106`、`C/local_device.py::TorchLocalDevice.prove:79至87,110至118`，一次读取 events 有诊断价值；设备门不读它们，目前没有修改后的 GPU proof。
- `C/fast_workflow.py::scoped_config:41至44` 继承 timeout、`C/fast_generation.py:160至168` 传实际 client timeout，是行为协议修订，不能当作无害记录修正，见 Q2.3。
- `src/kernel_optimizer/agents/runtime.py::OpencodeServer.start/_start_once:149至160,171至192,223至240` 过滤的是**共享** server 启动环境，server_env 后叠加，已有 server 分支提前返回；可能改变 C2/generic launch。优先 C3 opt-in，或显式共享版本并补兼容证据，不能直接批准为因果修复。
- “CPU 681 passed”是问题清单第 48 行的**外部报告，本文未运行**；不证明 native profiler 恢复，也不证明 C2 行为等价。

### Q2.2 framework_id 与修复判据

**runtime.py 未纳入 source closure 是真实缺口；“永远不改变 ID”和“只看 cuda_names”都不对。** `C/fast_artifacts.py::identify_framework:36至52` 收 fast_*.py、显式 local/device 模块与六个共享文件，不含该 runtime。
固定 revision/config/其余 covered inputs 时，只改它不会改变实现摘要或 framework_id；但 `Framework.revision` 是传入文本，不是自动测量的 checkout/dirty SHA（`C/fast_cli.py:53,70`、`C/fast_records.py:33至42`）。`C/fast_artifacts.py:32至33` 哈希整个 Framework，**改变 revision 文本仍会改变 overall ID**。
建议未来版本纳入实际执行的 runtime/相关配置策略依赖并导出源码哈希清单，不重算 F0/F1 历史 ID。配置摘要含 timeout，但不是全部环境设置；bundle/source/params 指纹也不能代替 framework closure。
`C/device_gate.py::require_device_proof:5至13` 还要编译、launch、声明名字匹配、输出参与、load/store operands 与 skip control 被拒；`C/fast_evaluation.py::proven_local:141至149` 另验质量。非空 names 既不能证明 env patch 的因果作用，也不足以 DEV_GO；还需真实部署身份、fresh lifecycle、完整 receipt、headroom 和 model 证据。

### Q2.3 contract 中未消费的 480 怎么处理？

**执行/传达路径未消费该键，但它仍是冻结的实验规则。** `R/setup/contract.json:248至254` 的 `stage_budgets.common.agent_call_hard_cap_s=480`；`C/runner_records.py::Contract:57至61` 无该预算字段，`C/fast_prepare.py::FastPrepared.agent_context:56至60` 丢 common，`C/fast_generation.py::DraftGenerator.invoke:134至140` 再投影上下文。证据检索未发现应用代码 reader。
`P:138` 明定八分钟，保留 contract SHA 而提高实际 timeout 仍是协议变更。`C/fast_prepare.py:16,32至36,80至83` 钉 contract/facet，不能静默改文件。
未来可选新版本 contract/facet 与 pins，或单独哈希的**前瞻协议 amendment manifest**，引用不变旧 contract、只显式替代对应时限。adapter 必须消费它，F* 封其哈希，并声明 transport timeout、idle fraction/推导时限、retry、阶段 deadline、在途终止规则。不能暗中“config 优先”；本文不批准 1500 或其他新数字。

### Q2.4 480、C2 慢调用与 F1 时间

**480 是快反馈计划的八分钟单调用预算，不是已证实的服务端 cap。** `P:130至140` 配套连续 D=90 分钟、零 transport/generic retry、首反馈软目标 3 至 5 分钟。
`docs/verification-c2-local-experiment-server-handoff.md:120至138` 的 C2 median/max 为 **519.162/902.761s** 与 **736.214/1306.289s**，与你方舍入值一致；每 run 四次 rewrite 的 STARTED/FINISHED 区间可能含重试，不是四次独立 C3 单次成功探针。
`src/kernel_optimizer/config.py:73至98` 确有历史成功最大 1167s、默认 1500s、333s/约 29% 余量注释；它不是私有配置的实值或普遍服务边界。
`src/kernel_optimizer/agents/runtime.py::OpencodeClient.idle_abort_s:479至482` 在 480×0.5 下为 240s；`_watch:628至704` 还要求 sandbox 可读且输出不变，并且 `_session_is_working(...) is False`。busy/unknown 重启观察窗口，不可读不触发该依据，**不是 240s 自动 abort**。历史 F0/F1 记录是 ReadTimeout。
`F/F1-result.json:152至155` 总时长 **481.2264972s**；`F/F1-essential.json:61` agent 开始比 probe 晚约 0.812s。**t=422.2s 返回与剩余 57.8s 未核实**：`F/F1-local.json:133` 只有局部 261.4378ms，无绝对返回时刻，需缺失的原 execution log 与共同 clock origin。延长 timeout 不保证 collection 恢复。

### Q2.5 server-before-prove 最小对照

**有用，但不是完整干净的因果检验，也不能一概算作免费 baseline 加 1 local。** 未来可固定 F0 bundle/params、fixture、GPU、解释器/栈与 owner RPC，先测 server 启动前，再只启动 server、不调用 agent/tool-child 后测；不同时改 timeout/env/stack。身份参考 `F/owner-rpc-diagnostic.json` 的 `report.{bundle_sha256,params_sha256,source_hashes,device_identity}`。
阳性只支持该条件下 server-start 关联干扰；阴性不排除 agent children 或 repeated-session 效应；前后次序仍受进程全局状态混杂。检验 env stripping 还需获准的 inherited/stripped treatment 与匹配 baseline，不能三 patch 一起上就归因。
新增 baseline、capture、before/after profiler 与 treatment 均要据实计量；历史 replay 是 capture+local **两次 local、零 agent**（`E/T4-native-F1.md:19至26`）。只有可用且获准的基线已经存在时，“1 local/0 agent”才可能描述一个新增 treatment。
需要未来明确授权、新输出与独立台账；旧 D 不可复用。可与条件式 development/formal 授权一起批准，不要求为它单独再来一轮许可对话。

## Q3：DEV_GO 与目标证据

### Q3.1 headroom 是必需门吗？

**是。无超匹配 A/A 噪声的可信收益、也无原计划允许的窄范围实测可恢复证据，就不能判协议合格 GO。** `P:87至88,101,119至124`、`docs/c3-goal-workflow-status.md:46,65,98` 保留该要求。
当前 `C/fast_search.py::tune_local:39至41` 仅用 local_gain 决定是否进 TPE，仍可返回无 headroom 的有效 artifact；`C/fast_workflow.py::run_probe:99至106` 和 freeze 重用未验 headroom 的 `C/fast_proof.py::prove_candidate:27至50`，存在准入口径缺口。
应在 development/freeze 验 fixture/device/framework 匹配的实测 headroom，不拿资源计数或猜测可调优代替，不新增任意百分比。局部成功也不保证端到端收益；formal 还需严格优于匹配 baseline 与正向 confirmation（`C/fast_workflow.py:107至113`、`P:124`）。

### Q3.2 两个 A/A 数字及身份

**不是单位换算，也不是同一记录算错。此前 handoff 未加限定的 0.192 是 F0 历史值，不能当成最后 F1 的值；本文纠正该上下文，不改旧 handoff。**

| 项 | F0 | F1 |
|---|---|---|
| noise_us | **0.19200146198272705** | **1.9359961152076721** |
| 原始 A/A 路径 | `D0/F0-evaluation/reference-aa.json` | `/root/autodl-tmp/c3-fast-kernel-iteration/development-996318d/F1-evaluation/reference-aa.json` |
| A/A SHA | `e68034a117085da69574215514ed20d8d23b4e6c556a6fe246f5d36c3310edae` | `68a5587f2a9c241732537f88d6a9358acf3fc045e4e56b665239870119264615` |
| revision | `996318df6c7e2e9fcdfac3a8012d4e813b7e732b` | `38e6e10de2e9770692f39ebb8eed86e11eb33df5` |
| framework_id | `36c96bb19771aa171ac3c0c94849c2634c2383f345ed722011a5c482408b38be` | `d278370b2ff8b9c0ec551dbc4b21c2d873b0f9f5ef755f3fdcfa182dc6c21b67` |
| 链接依据 | `D0/F0/result.json` 的 `target.noise_evidence` | `F/F1-result.json:141至143` 的 noise evidence |

`C/fast_setup.py:45至75` 计算 `max_fixture(abs(median(reference_a_us)-median(reference_b_us)))`，每侧三次 warmup、二十个计时样本，不是标准差。F0 数组在 `D0/F0-evaluation/reference-aa.json:2至68`，两中位数为 **225.11999309062958 / 225.3119945526123 us**。
`D0/dev-budget.events.jsonl:1至3` 的 F0 A/A admission 为 `1789900220.1188936`；F1 起点 `1789904563.7875042`，`E/T4-native-F1.md:32` 记录 fresh capture/A/A。**F1 raw A/A 数组及精确 admission/起止时间未本地镜像，不能重算或重新验证其文件 SHA。**
两次 local 的 fixture identity 相同（`D0/F0-evaluation/reference-aa.json:4至5,56至64`、`F/F1-local.json:28至40`），不意味着不同时期采样延迟相同。`C/fast_search.py::local_gain:17至18` 检查每 fixture 的 `median(reference)-median(candidate) > noise_us`，**noise 不是分母**，更不是最终增益分母。
两个 fresh helper report 均 invalid、latency/J 为 null，无有效 local_gain；RPC diagnostic 的 89.088us 不是 F1 candidate score（`F/owner-rpc-diagnostic.json:2至26`）。不得把 F0 小噪声移植到 F1。

### Q3.3 unknown exclusive fraction 与三个 site

**可测整模型替换效果，不能声称已知独占关键路径占比或 Amdahl 上限。** `D0/target.json:2至4,109至123` 是 TTFT、`pure-727f184a…a1fa0`，associated CUDA **14041.133999994665us**、exclusive fraction null；`C/fast_target.py::build_target:97至109,135至140` 在同名归属不明确时保留 unknown。
同 runner 的受控整模型比较与绑定/proof 身份支持替换效果，不把 inclusive/same-name duration 当独占贡献。各目标必须从各自合格 profile 选 site，可以相同也可以不同，不能只给 TTFT target 改 goal 标签（`P:84至88`）。
`C/fast_target.py::build_target:89至96` 要 decode 的 {0,127}。summary 中 single late 127 可见，但 single/multi raw 不全，**该 RMSNorm 家族对 late decode 的完整 shape/ABI 兼容性仍未知**，见 Q1.4；不是强制找三个不同 operator。

### Q3.4 一个 GO 还是三个？

**一个协议合格 DEV_GO 足够冻结 F*，前提是三个目标 target 预声明且合格；formal 三目标产物全部 fresh。** `C/fast_development.py::freeze_framework:82至99` 读一个结果、要求三目标 key；`formal_goal:109至120` 复验。`C/fast_generation.py::DraftGenerator.invoke:132,144至149` 从原 baseline 出发；`P:101至103,141至146` 禁止把开发 artifact 直接提升为 formal winner。
这不是当前 campaign 已可用的证明，Q1.2 的 headroom/source closure/目标 key 校验/终止屏障仍须接通。未接受候选的 lane 应显式 fallback 或 invalid/censored，不能制造 winner 补齐表。

## Q4：授权、预算与输入

### Q4.1 能否合并授权？

**可以一次授权多个条件式阶段；不要求三次许可对话。** diagnostic/setup、development D、formal S/final 各自保留 ledger、时钟、操作额度、go-gate、停止规则与新输出子根，一个父根即可组织。formal 必须等合格 GO/F* 与 readiness，final 必须等三臂 terminal（`P:130至156`、`docs/handoff-c3-fast-kernel-iteration.md:58至73,237至254`）。
旧 `F/terminal.json` 的 `ledger.spec`：D=`2026-09-20T10:30:17.689003Z`，cutoff=`12:00:17.689003Z`，已关闭/过期。剩余数量不产生 reset/refund/F2 许可。profile/oracle 即使零 provider 也有实际成本，必须列账；本文没有批准新数值。

### Q4.2 总时长与 final reserve

**formal 三小时包络已经包含最后 30 分钟，不是 3h+final。** 历史规则 `P:142至155`：总 agent/local/model=6/24/12，每目标 2/8/4；目标限 5400s 且截于 S+9000，final deadline S+10800，独立终测 36 blocks，不是额外调优。
唯一已测近似参考是旧恢复流程 `K/terminal.json:3至6`：**698.0197567939758s，约 11 分 38 秒，36 blocks**。这是该进程历时，不是新 device final 的预测或 GPU busy time；`K/recovery-provenance.json:10至11` 的 **1505.465713s** 是旧前瞻 planning allowance，不是实际用时。
旧共同 S 至恢复结束为 8029.033s，含 4295s 修复/审阅/部署/等待间隔（`docs/result-c3-qwen3-4b-operator-goals.md:148至163`）。新工程、diagnostic/setup、D 均在 S 外，端到端总时间**未知，不能保证三小时完成**。
新 adapter 的 preparation/oracle/quality/warmup/fixture/device-proof 与矩阵成本尚未完整测定，旧输入不能证明 30 分钟足够；未来若预测不适配，须在候选结果前前瞻修订，不能事后延 S 或缩减科学工作。

### Q4.3 需要几轮，失败怎么办？

**纠正：F0 + 原 F0 replay + F1 合计 2 agent / 8 local / 0 model，不是每次各 2/8。** `R/setup-native-recheck-996318d/F0-essential.json` 的 `budget.used` 为 1/3/0；`F/owner-rpc-diagnostic.json` 累计 1/5/0；`F/F1-essential.json` 累计 2/8/0。
F0、F1 各 capture+A/A+helper=3 local；replay capture+local=2。`F/terminal.json` 中 operations.local=5、capture=3，used.local=8 已包含 capture，不能再加三次；其第 71 行另记 F1 process 两次 model forward。
setup 独立账为 local3/model3/profile3/capture4、recheck forwards=275（`U/terminal.json:379至406`），初始 setup 261 forwards 另记（`E/T3-native-first.md:45至55,72至74`）；oracle 本身 64 forwards/4.665755676s。零 development model evaluations 不等于零 GPU/model 工作。
**新轮数不可预测。** 历史 `P:94至105,132至140` 允许最多三个 framework epochs 共用 D、6/12/4，只在可复现框架缺陷、聚焦回归/C2 证据与剩余获准预算下继续修。普通候选错误/无收益不许可改框架或手工 kernel；无框架缺口、额度耗尽、三 epoch 无 GO 均停报 NO_DEV_GO，拿到 GO 也不刷完余量。旧期限不能重置。

### Q4.4 哪些输入能复用？

**技术可复用不等于本次获准复用。** contract/preflight/interface.md/agent-input-contract.json/development-corpus 有 pinned 身份；16 条 calibration/search 开发记录可作为未来复用候选。target/profile/calibration oracle 须核对 corpus/model/target/stack/payload 与新授权，42 条新 sealed heldout 边界保持不变（`docs/handoff-c3-fast-kernel-iteration.md:104至117,183至187`、`E/T1-inputs.md:11至30`）。
新框架/config/revision 必须有真实新 seal/run 身份；fresh epoch capture/A/A 与 fresh agents 不能拿旧分数替代。旧 ledger/state/results 只作历史，不是可 reset 输入；final oracle/readiness 另行准入与计量。
**该 calibration oracle 的本地副本确实不完整。** `U/raw/oracle-54d86d14db3d4485870fc6969fd80c16/oracle.json:108,248` 指向 `calibration-calibration-00.f32`、`calibration-calibration-01.f32`，本地目录未含 payload。`U/terminal.json:389至392`、`D0/run.json:10` 记录历史远端路径。
需完整远端副本或另一个明确获准且验真的完整副本，不是“当前远端保证可用”。远端存在性/完整性未新核验，且这份 calibration 不是新 final oracle。

### Q4.5 私有 AppConfig 路径与 timeout

历史真实路径：**`/root/autodl-tmp/c3-fast-kernel-iteration/development-996318d/agent-config.json`**。`F/run_f1.py:34至35,48,52,64,105至110` 加载/封存并沿用原 run 输入，`D0/run.json:4` 记录相同绝对路径。
只读运行脚本和输入引用，未打开配置正文。**当前批准、当前存在性、文件内 request_timeout_s：全部未知。** 历史 effective fast timeout=480（`docs/handoff-c3-fast-kernel-iteration.md:25至28`、`E/T4-native-F1.md:32至38`）不等于文件本身值，通用默认 1500 也不能推出私有值。未来使用需在新授权下确认。

## Q5：环境与资源

### Q5.1 bjb1 是哪台，哪个解释器？

**`connect.bjb1.seetacloud.com:37752` 是历史 A；B 是 bjb3:49408。** venv 名不决定主机身份。F0/F1 实际脚本均用 **`/root/autodl-tmp/kernel-opt-venv/bin/python`**（`R/setup-native-recheck-996318d/run.sh:9`、`F/run.sh:9`）。
`E/T1-inputs.md:36至43` 记 A kernel-opt、B orch，并明确 **A 另有 Torch 2.14 orchestrator，但未使用**。发现 A/orch 不证明 A/kernel-opt 不存在；当前两者可用性未知。历史根因对照应提名 A/kernel-opt 并在新授权下验真；新 §4.3 使用预声明 A0 与匹配栈，不能静默替换解释器。
代码根 `/root/autodl-tmp/c2-local-experiment/code/v5` 与证据根 `/root/autodl-tmp/c3-fast-kernel-iteration/` 原本独立（`F/run.sh:3,5,9`、`F/run_f1.py:34至36`）；23M/179M 是你方新观测，本文未复核，不构成历史位置冲突。

### Q5.2 是否必须沿用旧栈？

**F0/F1 实际为 Torch 2.13.0+cu129 / Triton 3.7.1，A0。** 见 `D0/F0-evaluation/reference-aa.json:56至64`、`F/F1-local.json:28至36`、`F/owner-rpc-diagnostic.json:38至46`；Transformers 5.16.1 见 `E/T1-inputs.md:41`，Python 3.12.4 见 `handoff:83至90`。实际 CUPTI 组件版本未知，CUDA build suffix 不能代替库清单。
换 Torch/CUDA/Triton **可能**影响 profiler，但现有记录没有测版本效应。初始根因对照保持旧栈；新 formal 可明确选择新冻结栈，配新 readiness、所需 profile/oracle、匹配 baseline 与身份/授权，此时是新环境 epoch。
final 所有 row-vs-baseline 必须在同一声明栈内匹配；不能把 2.14/orch 与旧 2.13/kernel-opt 病例混池，宣称 env stripping 修复历史因果。

### Q5.3 development 卡 1、final 卡 0 可以吗？

**可作为前瞻资源方案，须明确批准，不是本次许可或 F0/F1 历史改写。** final 36 blocks 必须在同一预声明物理 A0 串行；每阶段记录 UUID、visible/logical 映射、解释器、栈与匹配 A/A/baseline。
历史 `F/gpu-after.txt:2至3`：A0=`GPU-c896ae45-532b-2a1d-147d-ed72e221fa42`，A1=`GPU-682bb59c-17a7-c7bc-2be8-83842d38f2f8`。物理卡 1 可映射成逻辑 cuda:0，逻辑编号不证明物理身份。
GPU1 诊断不能仅与旧 GPU0 相比作为单变量对照。mask/per-run lock 不证明独占；同 GPU 不重叠计时/编译/model，final 前终止本 campaign 搜索/开发，并避免另一 GPU 工作引入同机 CPU/内存/IO 争用（`P:151,162`、`handoff:100,317至319`）。旧空闲快照不是当前预约或许可。

## Q6：表格证据与解释

### Q6.1 proof 文件和字段具体是什么？

**现有每行 matrix 设备证明契约未完成；建议 Q1.3 的版本化 sidecar，不冒充已有输出。** 现有原料是 `C/device_records.py::LocalReport:119至135` 的 stage/bundle/source/params/device/declarations/fixtures/raw，`FixtureResult:109至116` 的 fixture/paired timing/quality/device_evidence，`KernelEvidence:78至106` 的编译、名字、launch、输出、operands、skip-control、质量与诊断。
`C/fast_records.py::CandidateResult:58至65` 链接 locals/model/confirmation；`C/fast_proof.py::ModelReceipt/prove_candidate:15至49` 与 `C/model_runner.py::_raw:47至55` 提供模型绑定身份。Artifact.kernels 是声明，**不是执行收据**。
建议最小 row manifest 字段：schema/profile version、row/goal/fallback reason、F*/protocol SHA、原 baseline、selection_document_sha256、bundle/source/params SHA 与参数值、target/site/group/profile、agent/session/source lineage、哈希化 local/fixture/noise、model quality/goal/confirmation、selected→execution bundle 映射及 final proof/measurement 文件。
**每个非 fallback 行都需覆盖三列的实际 goal/phase/shape**，验证 receipt 与 source/有效参数/F*/device 相配并执行设备门；不能用另行、旧 epoch 或只有 TTFT shape 的证明覆盖全表。`C/search_matrix.py:152至157` 重映射须有可验证映射或匹配新 bundle 的证明。
最终保存目标形状准备/设备证明和实际 final full-model execution/quality/measurement 身份。声明、旧 search proof 或 forward_calls>0 均不能独自证明 final 发射了 authored kernel；诊断 profiling 与 timed blocks 分开计量。baseline/fallback 写 `device_proof: not_applicable_baseline`，不造 positive proof；新增证明调用仍需未来授权。

### Q6.2 相同 winner/fallback 能支持目标差异命题吗？

**相同有效 winner 不支持已实现目标差异化，但仍可产生有效新测量；fallback 也须如实保留。** 有效身份包含 source、params、site mapping，不看文件名。`docs/c3-goal-workflow-status.md:25,90至91`、`P:161至164` 允许重复 winner/fallback，且重复仍要 fresh 测量。
可报告同产物收敛、未证实差异化、混合成功或 fallback；一个 fallback 不自动否定另外两行的真实对比，但该 lane 不是成功。`C/fast_records.py::CandidateResult/ProbeResult:58至84` 的目标条件搜索轨迹、拒绝原因、局部/模型分数与配置选择是辅助机制证据，不能替代失败的正向主命题，也不能 heldout 重选。

### Q6.3 每格汇总与增益公式

**每格取三个有效 block 指标的 median，不用 mean/mean 或舍入后的 published speedups。** `C/runner_measure.py::score_waves:62至95` 先定义每 block：TTFT 为四个独立请求 `(first_ready-accepted)` 毫秒的 median；single 为两个独立请求 `127/(last_ready-first_ready)` 秒制速率的 median；multi 为一次真实 batch=8 的 `1024/shared_wall_s`，含 prefill。
再按 `C/search_analysis.py::analyze_matrix:38至49` 对三个 block 指标取 median；不能混池 token timing 或平均分母。TTFT 改善%= `100*(baseline_ms-row_ms)/baseline_ms`；single/multi 改善%= `100*(row_TPS/baseline_TPS-1)`。保留未舍入 block 值，这只是描述性汇总，不自动代表统计显著。
baseline 与三个目标产物**每行都测 TTFT/single/multi 三列**，每列三 blocks，共 36；不要只填各行自己的优化目标。

### Q6.4 invalid/NOT_RUN/censored 如何显示？

建议分别写 **`NOT_RUN（未运行）`、`invalid（原因）`、`censored（截止/预算，注明未完成 blocks）`**，无数字处不填 0，也不代表 baseline 等价。`C/search_matrix.py::MatrixCell:26至34` 只有 valid/invalid/censored；`expired_matrix:46至55` 填 censored，NOT_RUN 是 `docs/c3-goal-workflow-status.md:73` 的上层缺席状态，不悄改旧模型语义。
`C/search_analysis.py:44至48` 仅在三个 block 全有效时汇总，relative gain 另需 baseline 三个有效值；第 56 行完整性要求 36 个唯一 row/goal/block 有效。selection_unchanged/private-GPU audit 为另列，新 device 表还要 proof/identity/protocol 闭包，数值完整不是全部验收。
**不存在获准“最多 N 个无效格”阈值。** 部分/负向结果可透明发表，但任一缺失或无效 block 都不能声称完成规定 36-block 正向证据 campaign；不能只聚合幸存子集充当完整表。

## 留给用户的前瞻决策清单

- [ ] 是否授权最小 adapter/driver 实现，以及是否一次条件式批准 diagnostic/setup、D、S/final；明确分阶段数量、时钟、输出、go-gates、终止与在途规则。**本文不批准具体新数字，不重启旧 D/S。**
- [ ] timeout 采用新 contract/facet 还是哈希化 amendment；明确生效上限、idle/retry/截止，并封入新身份。共享 env 行为选择 C3 opt-in 或显式版本加兼容证据，因果验证另计成本。
- [ ] 选择历史栈根因对照与未来 formal 冻结栈、物理 GPU/独占安排；逐项批准输入复用，补齐原 profile/oracle payload 与 source/proof closure，不提前开放 sealed final。

覆盖结果：**25 个编号问题 + Q0，合计 26 项，均已答复。** 未知不等于漏答：F1 raw A/A/原日志与返回时刻、六例因果台账、server/CUPTI 时间线与实际库版本、远端当前配置/资源可用性、新 final readiness/成本仍无充分证据。无新远端成功、DEV_GO、F*、formal S 或 §4.3 正向结果可在本文宣布。
