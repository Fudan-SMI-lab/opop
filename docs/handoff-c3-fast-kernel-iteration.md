# C3 fast-kernel iteration 技术交接

## 0. 先读结论与操作边界

**当前是已行政闭合的 `NO_DEV_GO`，不是成功实验。** 两次 fresh agent probe 都在 480 秒 transport 上限结束，均有部分编译、launch counter、数值、IR 和 output participation 证据，但 `cuda_names=[]`，未通过真实 CUDA trace 准入。没有官方完整模型 J、选定 F*、formal S、三目标 winners 或 36 块 heldout 矩阵；formal/heldout 为 `NOT_RUN`，方法效力为 `NOT_ESTABLISHED`。

最终独立 review 的 `APPROVE` 只批准如实交付有界失败。公开报告里“review 尚待交付”是发布时状态，后续 `.omo/evidence/v5-c3-fast-kernel-iteration/F1.md` 已给出最终结论。这里的 review F1 与第二个开发 framework F1 不是同一对象。

**原开发窗口已过期，禁止用旧 ledger 再运行 native 命令、重置 D、退回消耗、改标旧结果或启动 F2。** 本文是文档交接，不是执行授权。未来框架工作和实验需要另行明确授权；新实验还需要独立预算与新输出根，不能把旧窗口换个名字续跑。

用户优先级：尽快获得真实 native 反馈，修框架而非人工修候选 kernel；不扩建测试基础设施、不铺开 broad tests；保留 C2 显式 profiles、默认行为及继承的 dirty worktree。不要清理无关文件，不要自动提交或发布。本文核对时仅本地读取、CPU help/schema 检查和新增此文档，没有 SSH、GPU、provider、下载、安装或 Git 操作。

## 1. 目标与固定输入

目标是 agent 编写实际参与模型输出的设备计算 kernel，依次证明真实编译、CUDA trace-confirmed launch、输出参与、局部数值正确、完整模型质量正确，随后在**同一冻结框架**下分别测量 TTFT、single、multi。Python wrapper/dispatch 改写的收益、声明有 kernel、人工 control 或局部延迟都不能替代这个目标。目前为 C3-B manual-task-first，C3-A 自动任务构建没有执行；现有 fixture 证明能力明确为 Triton、直接 returned-output operands，不能据总目标宣称任意 CUDA backend 已被支持。

| 项目 | 固定值及边界 |
|---|---|
| 最新执行源码 | `38e6e10de2e9770692f39ebb8eed86e11eb33df5`，owner-thread 修复及其既有回归 |
| 公开报告 revision | `8baa00800878ab2326010d89b3deed168e85a91f`，父 revision 为上述执行源码 |
| F0 执行源码 | `996318df6c7e2e9fcdfac3a8012d4e813b7e732b`，IR 归属修复及 setup recheck |
| F0 framework ID | `36c96bb19771aa171ac3c0c94849c2634c2383f345ed722011a5c482408b38be` |
| F1 framework ID | `d278370b2ff8b9c0ec551dbc4b21c2d873b0f9f5ef755f3fdcfa182dc6c21b67` |
| Contract SHA256 | `2c0e2fc70c76d4b780cb1301addcf2d723ec5f19f106af7ac3919fbc86996bd8` |
| 模型 | `Qwen/Qwen3-4B@1cfa9a7208912126459214e8b04321603b3df60c`，BF16/eager/SDPA |
| Agent | 既有 GLM5.3/max 配置；fast scope 固定 timeout 480s，generic/transport retries 均 0 |

上述 revision 是发布/执行 receipts 确认的身份，不是本文对当前 dirty worktree 或远端 HEAD 的新认证。F0/F1 的 schema、prompt、agent-config hashes 相同，F1 改的是实现/revision，不能把不同 framework 的证据合并。

输入分层必须保持：

* `development-corpus.json` 是明确复用的 **16 条**，即 2 calibration + 14 search；开发与 formal search 只接触这一层。
* 新 **42 条 heldout** 在候选前冻结，与旧 58 条及开发输入不交。TTFT 为 12 条 4096-token 输入，single 为 6 条 256-token 输入，multi 为 24 条 512-token 输入。
* `heldout/corpus.json` 是 final runner 使用的 sealed 容器，**16 + 42 = 58 条**，不是“58 条新 heldout”。公开 `corpus.json` 只是 IDs/长度/hashes manifest，不是原始 prompt 数组。本文未读取 sealed 内容。
* TTFT 每请求输出 1 token；single 和 multi 每请求输出 128 tokens，multi 每 wave 为 8 个独立请求。保持原非 thinking、greedy seed 0、EOS/长度规则、fresh KV/state 和 warmup。
* 质量规则不放宽：local `.02/.02`，per-prompt logits relative L2 `.01`，paired mean NLL delta `.02 nat/token`。正式阶段还没运行，不能凭输入已冻结说 final oracle 已就绪。

旧 `.omo/notepads/v5-c3-fast-kernel-iteration/workflow-handoff.md` 早于最新 native probes。它的组合 API 说明可参考，但“runnable F0 / pending”状态及含糊的 “sealed58”措辞由本文和最终 receipts 更正。

## 2. 已知事实、失败分类与未知项

| 证据类别 | 实际观察 | 可以得出的结论 |
|---|---|---|
| 初次人工 setup control | 编译、counter、数值匹配，但 IR 输入/输出归属为空 | 当时框架证明失败，拒绝正确，失败 local charge 保留 |
| 996318d setup recheck | 同一 control 的 IR、CUDA、数值、output-use、skip/suppression 通过；wrong control 被拒；3 次模型 A/A/control 检查通过 | 证明该人工 control 的 harness 路线可用，不是 agent winner |
| Fresh F0 | 一次 primary，480s ReadTimeout；compiled=true、JIT2/launch-count1、正常数值/skip/IR/output 证据存在，CUDA names 空 | local invalid，latency/model J 均 null；`PARENT_TRIAGE / transport_failed` |
| 原 F0 诊断重放 | 不改 F0 source/bundle，经真实 HelperService RPC 和 owner pump 通过；local 89.088us，official_model_score=null | owner-thread 修复在此诊断有效；不能追认 F0 或把 89.088us 当整模收益 |
| Fresh F1 | 新 session/source，owner PID:TID `776849:776849` 上采集，仍 `cuda_names=[]`；再一次 480s ReadTimeout | owner-thread 改动未解决真实 generation 生命周期下的问题；无完成返回 artifact、无 repair/redraw、无 J |

两次 helper 的共同错误是 `declared device kernel was not compiled and CUDA-trace-confirmed launched`。这是组合 gate 文本，**不能简写为“没有编译”**，也不能把 launch counter 当成 CUDA trace PASS。F0/F1 局部编译耗时分别为 156.21/162.7ms，registers 35/33，shared bytes 均 32；这些不是性能收益。

诊断身份：F0 source SHA `b6b0b447aef27031d0954de3595811e46f405ae81a8e1f8d9885425deefafedf`，bundle `aba30792ba8a9fb488259ba26cac52a619f10df6fa837b39326738b7fb4c9c71`，参数 `num_warps=8`。Fresh F1 source SHA `08363d2cb383451d5ffbd98a79410a1c5c5cfa8edfa766b3c25d0d0d19fbbdcb`，bundle `68a03a7c2df2d6fff4cbd2ea41cde58f78b1aa98336d1621ae5203e5987e6321`，参数为空。成功诊断不能与新 source 的失败拼成成功 epoch。

**未知项：**真实 agent 生命周期中 profiler collection/lifecycle/correlation 的残余根因；两个 primary 为什么都未在 480s 内完成；局部失败反馈是否导致 transport timeout。现有证据没有证明数值 kernel 有缺陷，也没有证明加 timeout、再换线程或改算法会修好。优先归类为尚未定位的框架/运行生命周期问题，不人为改 kernel 迎合 gate。

## 3. 预算与不可变历史

原 D 为 **2026-09-20T10:30:17.689003Z**，Unix `1789900217.689003`；cutoff 为 **2026-09-20T12:00:17.689003Z**，Unix `1789905617.689003`。连续 90 分钟包含 reload、框架修复、CPU 回归、诊断、父级/操作等待，没有暂停或延长。工程准备在 D 前就开始了，不能说整个项目只花了 90 分钟。

| 开发组成 | Agent charges | Local charges | Model-evaluation charges |
|---|---:|---:|---:|
| Fresh F0：capture + reference A/A + helper | 1 | 3 | 0 |
| 原 F0 诊断：fresh capture + helper replay | 0 | 2 | 0 |
| Fresh F1：capture + reference A/A + helper | 1 | 3 | 0 |
| 合计 | **2** | **8** | **0** |

`used.local=8` 包含 `operations.local=5` 和 `operations.capture=3`，不是 8 再加 3；profile=0。Model-evaluation=0 不代表没跑 GPU/模型：capture 本身有模型 forward，F1 进程单列 2 次 capture-related forwards。

Setup 单独累计 3 local / 3 model / 3 profiles，另有 4 capture/preparation operations；首次失败未退款。初次 261 与 recheck 275 forwards 是各自 receipt 中的 setup 工作，不是开发 J。

代码的 development 上限为 6 agent / 12 local / 4 model、D+5400，跨 framework 修订共享。capture 同时扣 local，profile 同时扣 model。剩余数量 4/4/4 不构成续跑许可。F1 结束耗时 481.226497s 包含外围工作，transport 上限仍为 480s，drain=0；历史关闭时剩余 572.143s、后续快照 500.868s 是两个采样时刻，**现在均不有效**。

旧 budget/state/result/events/raw 保持不可变。未来新授权必须写清哪些输入可复用、是否允许新 setup/diagnostic、总量与时限、停止规则；使用新根、新 budget/state，不覆盖 `development-996318d`。框架修订不是新实验：在一个尚有效的新窗口内，仍必须沿该窗口原 D、计数及 review/resume 规则继续。

## 4. 环境与路径

### 本地 Windows

工作根 `D:/Pyhon_projects/opop/v5`；既有解释器 `.venv-v5/Scripts/python.exe`。历史 CPU receipts 为 Python 3.12.7、Pydantic 2.13.5、Optuna 4.9.0。PowerShell 的 `PYTHONPATH` 用分号分隔 `src` 和项目根。没有可用 Codegraph index，使用 Read/Grep/目录读取即可，不要为交接新建索引。

### 远端定位，均为历史记录，不是本次连接结果

| 项目 | A | B |
|---|---|---|
| SSH 入口 | `ssh -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=yes -o UpdateHostKeys=no -p 37752 root@connect.bjb1.seetacloud.com` | 同四个选项，目标 `autodl`；既有 alias 指向 `root@connect.bjb3.seetacloud.com:49408` |
| 模型/native Python | `/root/autodl-tmp/kernel-opt-venv/bin/python` | `/root/autodl-tmp/orch-venv/bin/python` |
| 已记录 Python | 3.12.4 | 3.12.4 |
| 代码根 | `/root/autodl-tmp/c2-local-experiment/code/v5` | 同路径，不能据此假定已部署最新源码 |

SSH/解释器来源为既有 `v5-c2-opportunity-driven/server-preflight.md` 与 `v5-c2-method-improvements/server-preflight.md` notepads；C3 的 `T1-inputs.md` 确认两套模型 Python 和包版本，最新 A 路径由 `T4-native-38e6e10/run.sh` 复核。A 实际跑过最新 38e6e10，B 没有对应 fresh probe 证据。不要随手改用 A 的 orch-venv 或 B 的 kernel-opt-venv，它们历史上是另一套 Torch/Triton。

已测模型栈：Torch 2.13.0+cu129、Triton 3.7.1、Transformers 5.16.1。A0 为 RTX4090，GPU UUID `GPU-c896ae45-532b-2a1d-147d-ed72e221fa42`。两机模型资产路径均为：

```text
/root/autodl-tmp/c3-qwen3-4b-operator-goals/assets/Qwen3-4B/1cfa9a7208912126459214e8b04321603b3df60c/
```

远端输入根 `/root/autodl-tmp/c3-fast-kernel-iteration/setup/`；raw 总根 `/root/autodl-tmp/c3-fast-kernel-iteration/`。不用下载或重装。原 provider AppConfig 是**服务器上已批准的私有配置**，本文只写 `/ABS/approved-server-private-config.yaml` 占位符，不提供推测真实路径、不导出配置体或密钥。操作员必须在新授权下确认该配置仍获批准。

Native 进程环境原则：`PYTHONPATH=<代码根>/src:<代码根>`；`PYTHONDONTWRITEBYTECODE=1`、`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`。`CUDA_DEVICE_ORDER=PCI_BUS_ID`，只暴露被分配的物理 GPU；例如 `CUDA_VISIBLE_DEVICES=0` 配合逻辑 `cuda:0`。若物理 mask 为 1，逻辑设备仍为 `cuda:0`。runner 与 `cfg.opencode.server_env` 必须一致。清除该进程的 `http_proxy/https_proxy/all_proxy` 及大写版本，避免旧网络加速 scope 污染 inference/provider/helper；不要改机器全局配置。同一 GPU 上 native timing 必须串行，不并行跑诊断、capture 和模型测量。

## 5. 最小输入与可复用边界

下表远端短路径相对于 `/root/autodl-tmp/c3-fast-kernel-iteration/`，除非另有说明。

| 输入 | 已有内容 | 未来授权运行的要求 |
|---|---|---|
| contract / preflight | `setup/contract.json`、`setup/preflight.json` | 不改科学规则；只有新授权明确允许才复用 |
| contract 同目录依赖 | `interface.md`、`agent-input-contract.json`、`development-corpus.json` | `prepare_fast` 会逐一检查固定 hash；不能只复制 contract |
| target | `development-996318d/target.json` | 复用需确认 profile/target 身份；改 target 必须新声明，不能悄悄换任务 |
| native profile | `setup-native-ad0694a/raw/short-device-profile-17cb1c7131914914bcb7bf55bb0fd2e8.json` | 原完整 profile 在远端；本地 bulk copy 未全部核验 |
| framework | `development-996318d/framework-F0.json`；F1 路径为同目录 `framework-F1.json` | 未来测试源码/config 需新 seal，使用真实 tested revision，禁止填假 `F0` 字符串 |
| calibration oracle | `setup-native-recheck-996318d/raw/oracle-54d86d14db3d4485870fc6969fd80c16/oracle.json` | 必须绑定新 contract 的开发 corpus，且 payload 齐全；不是 final oracle |
| agent_config | 服务器已批准私有 AppConfig | 操作员填写真实绝对路径，不能从本文获取或新造凭据 |
| budget/state/output | `development-996318d/dev-budget.json`、`dev-state.json`、F0/F1 输出 | 旧文件只能做证据；未来独立实验必须新建，不清零旧文件 |

已有 target 是 TTFT，profile 选择 `model.layers.33.post_attention_layernorm`，canonical group 包含 73 个兼容 RMSNorm leaves；site ID 为 `pure-727f184a931d6918df621c6385fe123d7dd248cb470c207fd46b715a8c7a1fa0`。实际代表 fixture 可来自组内 `model.layers.0.input_layernorm`，不能据此误判 target 换了。4096-token representative 的 associated CUDA 时间 14041.134us，exclusive fraction 仍 unknown，reference A/A noise 0.192001462us；禁止把重叠关联时间当独占关键路径或据此许诺收益。

## 6. Runbook

### 6.1 现在可做的 CPU 检查

以下 PowerShell 命令在项目根运行。第一条 help 已在本次文档核对中实际执行，exit 0；其余 help/回归是操作示例，本文没有重跑回归。无需安装任何包，也不需要新测试。

```powershell
$env:PYTHONPATH='D:/Pyhon_projects/opop/v5/src;D:/Pyhon_projects/opop/v5'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
$env:CUDA_VISIBLE_DEVICES=''
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
& ./.venv-v5/Scripts/python.exe -B -m examples.c3_qwen3.fast_cli --help
& ./.venv-v5/Scripts/python.exe -B -m examples.c3_qwen3.fast_cli develop --help
& ./.venv-v5/Scripts/python.exe -B -m pytest -p no:cacheprovider -p tests.c3_search_fakes -q tests/test_c3_fast_helper_threads.py tests/test_c3_framework_iteration.py tests/test_c3_device_kernel_gate.py tests/test_c3_task_routing.py tests/test_c2_agent_compatibility.py
```

最后一条是已有测试的聚焦选择，不是 native 验收。`tests.c3_search_fakes` 提供 fake 边界并抑制 sandbox Git 初始化。只有碰到 IR 归属相关改动时，再单独用既有 `tests/test_c3_device_ir.py`；涉及 immutable-parent 时选择 `tests/test_c3_rewriter_parent_snapshot.py` 中已有相关案例。不推荐全库测试、新测试平台、导出所有旧 sessions 或重新跑 C2 GPU 实验。历史 T4 worker 为 56 passed、5 个既有 warnings，parent 独立检查为 25 tests；这些是既有 receipts，不能相加，也不是上述子集本次结果。

### 6.2 真实 CLI 语法，全部为未执行示例

下面为远端 Bash 语法。`$PY` 取第 4 节批准的解释器；所有 `/ABS/` 路径是占位符，必须替换。**当前不得运行 native develop/probe，亦不得对过期旧状态运行 resume。** seal/target 是 CPU-only，但会创建文件；本次文档任务也没有执行它们。

```bash
"$PY" -B -m examples.c3_qwen3.fast_cli seal \
  --contract /ABS/setup/contract.json --assets /ABS/setup/preflight.json \
  --agent-config /ABS/approved-server-private-config.yaml \
  --revision TESTED_SOURCE_SHA --output /ABS/NEW-RUN/framework-F0.json

"$PY" -B -m examples.c3_qwen3.fast_cli target \
  --profile /ABS/existing-native-profile.json \
  --module model.layers.33.post_attention_layernorm \
  --output /ABS/NEW-RUN/target.json

"$PY" -B -m examples.c3_qwen3.fast_cli develop --inputs /ABS/NEW-RUN/run.json

"$PY" -B -m examples.c3_qwen3.fast_cli resume \
  --state /ABS/NEW-RUN/dev-state.json --review /ABS/NEW-RUN/review.json \
  --framework /ABS/NEW-RUN/framework-F1.json
```

只有 `seal/target/develop/probe/resume` 五个名字，`probe` 是 `develop` 别名。没有 `formal`、`heldout`、`freeze` 子命令，也没有 `develop --device` 或 `--goal`；设备和目标文字放在 JSON，goal ID 来自 target。seal/target 的 output 用独占创建模式，父目录需已存在，不能覆盖旧文件。develop 一次 epoch 后停止，只有 `DEV_GO` 返回 0，其他结果返回 1，不能把 exit 1 自动 retry。

### 6.3 `run.json` 最小可用模板

此模板按当前 `fast_runtime.RunInputs` 字段核对，并用该类型做了 CPU JSON 解析检查。**这只验证 schema，不验证路径存在、资产、授权、hash 或 native 能力。** 模板假设一个未来新授权窗口的 budget 已按批准 D 创建，因此省略两个 clock 字段；不是指向本轮过期 budget。

```json
{
  "contract": "/ABS/setup/contract.json",
  "assets_manifest": "/ABS/setup/preflight.json",
  "agent_config": "/ABS/approved-server-private-config.yaml",
  "framework": "/ABS/NEW-RUN/framework-F0.json",
  "target": "/ABS/NEW-RUN/target.json",
  "oracle_refs": "/ABS/approved-calibration-oracle/oracle.json",
  "budget": "/ABS/NEW-RUN/dev-budget.json",
  "development_state": "/ABS/NEW-RUN/dev-state.json",
  "output": "/ABS/NEW-RUN/F0",
  "goal_text": "在固定质量与输入规则下最小化完整模型 TTFT，提交实际参与输出的设备计算 kernel",
  "device": "cuda:0"
}
```

类型层面必填：contract、assets_manifest、agent_config、framework、target、output、budget、goal_text。`development_state` 与 `oracle_refs` 虽为可空字段，但此 development 入口实际要求它们；缺少 oracle 不能免费生成 reference。其他可选字段为 `started_unix_s`、`deadline_unix_s`、`frozen_framework`、`clock`、`correct_control`、`wrong_control`，不是 formal 已完成的标志，本模板不需要它们。

新 budget 不存在时，必须额外提供批准后记录的数值 `started_unix_s` 和 `deadline_unix_s`，后者严格等于前者 + 5400。不要填原 D，不要为让示例跑起来临时造时钟。预算已存在时省略 clock，或提供完全相同的值；lane/stage/framework/goal 与 clock 必须匹配，不能借加载输入替换旧身份。当前代码在 resident load 前拒绝过期 budget。

输出 `F0` 及同级 `F0-baseline`、`F0-evaluation`、`F0-agents`、`F0-preparation` 均应为该次运行专属新路径。状态由第一次 `develop` 创建；不要手写 ready 状态来跳过限制，不要删除 running/failed 状态重新抽样。每轮最多一份普通 artifact repair，transport failure 不 redraw；framework 最多三个 epoch，是否继续由父级 review 决定。

`resume` 只记录父级诊断和激活新 framework，不执行模型，也不重置 budget。review JSON 类型为 `FrameworkReview`：`result` 必须是最新 result 的绝对路径；`decision` 为 `framework_defect` 或 `ordinary_failure`；`started_unix_s`、`ended_unix_s` 是实际 review 时间；`lines` 为 1 至 10 条原因。framework_defect 还需存在的 `regression_evidence` 与 `c2_compatibility_evidence` 路径，ordinary_failure 转 STOP。类型虽包含 `freeze`，失败 epoch 的 resume 明确拒绝它。review 时间和实际调用时间都不能越过该窗口 deadline。

### 6.4 避免重复模型加载

CLI `execute_development` 经 `open_evaluator` 调用 `load_fast`，每个 fresh 进程都加载模型，finally 关闭 resident；它不会连接以前的驻留模型。为最短 native 反馈，未来获准且已有兼容 resident 时，应复用现有组合 API，而不是并发启动第二个 CLI：

1. `prepare_fast(..., FastAccess(profile="c3_fast_device", phase="development"))`，然后 `attach_fast(resident, prepared)`。attach 恢复 baseline、重置模型状态及 fixtures/sites，不重置 StageBudget。
2. 将该 resident、`TorchLocalDevice("cuda:0")`、同一 StageBudget 放入 `LocalRuntime`；用 `FastEvaluator` 和 `EvaluationFiles` 绑定原 baseline、新输出目录及已批准 calibration oracle。
3. 按 `fast_runtime.open_evaluator` 设置 `engine.check_framework`，以 `identify_framework(frame.revision, scoped_config, prepared)` 校验固定 epoch，不能漏掉这个检查。
4. 使用 `open_generator(inputs, engine)` 的既有 Runtime/factory 路线，再调用 `develop(generator, state_path, output)`。owner 必须是 resident 创建线程；transport worker 跑 agent，owner pump 执行 helper GPU callbacks。
5. 调用方在 finally 关闭 resident，确保 Runtime/helper 退出。只直接调用 evaluator 的重放是诊断，不能替代 fresh agent 生命周期验收。

这些是已有 API 的组合说明，不是一份已经验收的新运行脚本。attach 前还需确认 model path/revision/files 与 resident 相符。不能通过复用 resident 免掉 fresh capture、A/A 或预算 charge。

## 7. 代码职责与兼容边界

除最后一行外，下表路径前缀均为 `examples/c3_qwen3/`。

| 文件 | 接手时关注的职责 |
|---|---|
| `fast_cli.py`、`fast_runtime.py` | CLI 参数、RunInputs、fresh load、oracle/budget/framework admission、Runtime 生命周期 |
| `fast_prepare.py` | 固定输入 hashes、16 条 development 与 final 访问隔离、resident attach |
| `fast_generation.py` | DraftGenerator、真实 helper RPC 接入、ContextVars transport worker、owner callback pump；当前残余生命周期问题的首要接缝 |
| `fast_workflow.py`、`fast_search.py` | 480s/0 retries、capture/A-A、单次 repair、local/tuning/model 顺序及 result；不自动改框架 |
| `fast_budget.py`、`fast_development.py`、`fast_records.py` | 持久 charge、D、review/resume、freeze/formal helpers、证据数据类型 |
| `fast_target.py`、`device_profile.py`、`device_capture.py` | profile 到 canonical leaf group、代表输入、真实设备观察 |
| `device_gate.py`、`local_device.py`、`triton_observer.py`、`device_ir.py` | device proof、局部执行/采集、编译/IR 输入输出归属；不能放宽 gate |
| `fast_evaluation.py`、`fast_proof.py`、`model_runner.py` | local 与 model receipts、完整质量/J、artifact/source/params/stage 对齐 |
| `src/kernel_optimizer/wiring.py` 与 `src/kernel_optimizer/agents/model_operator_rewriter.py` | 显式 execution profile、agent task schema；不是靠 goal 文本隐式选择路由 |

C2 的 default 保持 `existing_generic`；显式 `c2_direct_compat` 对应 `cda113070c8ba3a91f480f42b68d196187796cda` direct anchor，default/legacy 对应 `8526a1390c78346af1a5be112685b7fbe4defa59` 的各自路线。已有有限 CPU fixture 比较了 effective prompt、完整 schema、seeds 和 normalized request，不等于所有历史配置/随机输出/性能等价。保留 immutable-parent 修复 `82daf15b460004707e55dbdebb789e2788921527` 及既有 grouping 修复，不改共享默认值。C3 显式使用 `model_operator` / `model_project_operator`。

## 8. 最短的后续诊断顺序

当前只需阅读已有证据；下面涉及运行的步骤全部等待新授权与预算，不自动执行。

1. **先对齐失败身份和时序。** 从 F1-essential、F1-local、terminal 与已有 execution log 对齐 session/call、bundle/source/params、PID:TID、helper 请求/返回、profiler enter/exit、timeout、close/join。F0 summary 的 locals 为空不能抹掉独立 helper receipt。
2. **区分空采集与名称关联失败。** 先看 profiler 是否产生任何 CUDA activities，再看是否已有 CUDA event 但没匹配声明名称。当前 `cuda_names=[]` 不能直接定性为“名字改一下即可”；检查 collection、flush、同步和生命周期与真实调用顺序的关系，但这些都是待验证假设。
3. **用同一不可变产物做最小对照。** 已有 owner-RPC replay PASS 是对照，不再单独重做一大轮 setup。若批准一次 native 诊断，应优先比较真实 generation/HelperService 生命周期与 diagnostic replay 的差别，而非只测直接 `engine.local`。每次 capture/local 都记账。
4. **单独查 transport。** 确认 helper feedback 是否返回、agent 是否继续、transport worker/owner pump 是否正常退出、join/drain 的顺序。局部 gate 失败与 480s timeout 同时出现不是因果证明；不要先加 timeout 或 retries 掩盖问题。
5. **仅做证据支持的最小框架修复。** 用已有 thread/RPC、workflow/device、C2 聚焦回归检查，不扩建测试平台。修复后既要有真实 lifecycle 证据，也要有 fresh agent 输出的 native gate 证据；仅 replay PASS 不够。若无预算或仍未知，停止并如实记录。

Owner pump 只能路由和关闭 pending/late requests，不能强制抢占正在运行的 GPU 工作。故需保留 deadline、finish、drain 和清理证据，不能只看主调用超时就认定无残留。

## 9. DEV_GO 与 formal 的验收清单

### 下一次获准开发的 DEV_GO

* [ ] 独立授权、新预算根、真实 D/cutoff、tested source/framework/config/contract identities 齐全，保留旧失败且不回填。
* [ ] fresh agent 来源和完成返回 artifact 可追溯，source/bundle/params/session hashes 一致，不以人工 control 或诊断重放替代。
* [ ] 对所声明设备 kernel 有真实 compiled、CUDA trace-confirmed launch、output participation、IR/正常数值与 suppression 证明；launch counter 或声明本身不够。
* [ ] 同一 stage/artifact/params 有有效 local receipt，完整模型 baseline/candidate receipt 与实际 J；quality、请求数/输出长度符合 contract，receipt 中 source 和 forward_calls 可核对。
* [ ] 按批准计划保留可信实测 local headroom：记录有效点的局部收益与对应 A/A 噪声，收益须超出噪声；适用时，慢 default 只能凭实测 tile 利用、compiled 资源或其他诊断支持的具体可恢复配置路径，在剩余额度内评估测前声明的廉价 alternative，再判断 headroom。资源计数本身不是收益证明，不能凭“可能调快”准入。
* [ ] 通过 `prove_candidate` 和 workflow 正常状态迁移，失败/timeout 不追认；**不承诺相对 baseline 的正端到端收益，也不因科学期待要求这种改善；这不豁免上述 local 准入/headroom、真实 device proof 或数值/模型质量条件**。

计划条件与当前实现须分开：计划第 101、119–120 行要求可信实测 headroom，并允许有狭窄实测可恢复配置证据且有余量时进入局部探索；当前 `fast_search.local_gain` 要求每个 fixture 的 reference/candidate median 差超过记录的 A/A noise，`tune_local` 仅在已有通过点满足该条件且参数空间非空时进入 TPE，未实现该狭窄例外。没有超噪声收益时它仍可返回有效 default/alternative，而 `prove_candidate` 不单独检查 local headroom。因此实现返回 DEV_GO 不自动证明计划的 headroom 条件已满足，必须核对相应实测证据，不能降低计划标准。

### Formal 与新 heldout

**现有端到端 formal/new-heldout workflow 尚未完整实现或达到验收就绪。** `freeze_framework` / `formal_goal`、final access 和预算类型已存在，不代表调度、frozen-F* 接入、三目标完成、独立 proof/验证 ledger、36 块矩阵与分析链已经接通。不能把旧 `manual_task` 或 `operator_cli heldout` 静默接上新 contract 来填补空缺。

* [ ] 单独完成/审查缺失的最小 formal 连接，保留 C2/default；不要新建通用安全平台或大型测试设施。
* [ ] 仅在真实 DEV_GO 后冻结 F*、三个预声明 targets 和 compatibility receipt；同一 F* 从原 baseline 分别生成 TTFT/single/multi，不携带开发 winner 充当新答案。
* [ ] 新正式 S 有授权记录：共同 search cutoff 为 S+9000s，final deadline 为 S+10800s，每 goal 90 分钟并裁到共同 cutoff；每 goal 2 agent / 8 local / 4 model，总 6/24/12，不能互相补成额外调用。
* [ ] 完成正式选择后才允许 final runner 接触 sealed 容器，不把 heldout 文本/IDs/reference 泄漏给 agent 或用于调参；新 final references 按授权记账生成，不复用旧 heldout 结果。
* [ ] 原 baseline 与三个 goal-selected rows，4 rows × 3 goals × 3 blocks = 36 块，在规定 A0 上串行独立测量，质量/来源/失败/censoring 明确；无效或未运行不写成零分，也不预设正收益。

## 10. 证据入口与实际文件名

所有本地相对路径以 `D:/Pyhon_projects/opop/v5` 为根。以下按本次实际目录读取列出，不把 receipts 提及的远端文件假装成本地文件。

### 最短阅读路径

1. `docs/result-c3-fast-kernel-iteration.md`：公开失败报告。
2. `.omo/evidence/v5-c3-fast-kernel-iteration/T4-native-F1.md`：原 F0 owner-RPC 对照与 fresh F1 失败。
3. 同目录 `T6.md`：预算、科学边界和闭合；`F1.md`：最终独立 APPROVE 的精确范围。
4. `results/c3-fast-kernel-iteration/T4-native-38e6e10/` 的 `F1-essential.json`、`F1-local.json`、`F1-result.json`、`owner-rpc-diagnostic.json`、`terminal.json`、`gpu-after.txt`。

T4 本地目录还实际有 `F1-source/`、`collect_stop.py`、`deploy.sh`、`run_f1.py`、`run.sh`。这些历史操作脚本**不要重跑**。本地该目录没有 `execution.log` 或完整 F1 agents 数据库；完整原件在远端同名 T4 根及 `/root/autodl-tmp/c3-fast-kernel-iteration/development-996318d/F1{,-evaluation,-agents,-baseline}`。F1 原 local receipt 的精确路径为 `development-996318d/F1-evaluation/raw/fixture-only-0bd149f844c542049c9a7c4d455f9892.json`，失败 workspace 为 `F1-evaluation/C0-workspace`。

### F0 与 setup recheck 的嵌套路径

本地根 `results/c3-fast-kernel-iteration/setup-native-recheck-996318d/host-A/` 下实际是两个子目录，不是把 JSON 直接放在 host-A 根：

```text
development-996318d/
  D-clock.json
  dev-budget.json
  dev-budget.events.jsonl
  dev-state.json
  framework-F0.json
  target.json
  run.json
  operator-terminal.json
  parent-triage.md
  F0/result.json
  F0-agents/
  F0-baseline/
  F0-evaluation/
setup-native-recheck-996318d/
  setup-result.json
  setup-budget.json
  F0-essential.json
  terminal.json
  cleanup.json
  execution.log
  gpu-after.txt
  compute-after.txt
  raw/oracle-54d86d14db3d4485870fc6969fd80c16/oracle.json
```

这是 F0 阶段 essential mirror，**其中 dev-budget/state 不应当作 F1 结束后的最终账本**。最终合计读 T4 的 `F1-essential.json` 和 `terminal.json`；原远端 budget/state 后来继续记录了 F1。本地 oracle 目录实际只有 `oracle.json`，未导出完整 f32 payload，不能宣称本地模型运行所需 oracle 已齐全。Receipts 提及的 `essential-validation.json` 不在上述两个目录的实际列表里，本文不将它作为可直接打开的本地入口。

初次 setup 本地 `results/c3-fast-kernel-iteration/setup-native-ad0694a/` 实际含 `profile-result.json`、`profile-summary.json`、`control-selection.json`、`control-result.json`、`fixture-only-failure.json`、`ir-repro.json`、`failure-evidence.json`、`operators.py`、`control.log`、`raw/` 和 `target-fixture-a5d8382fa6d9497aaafe68e53333fd49.json`。bulk profile 曾超时，raw copy 没全部核验，不作为完整权威 profile dataset。

### 进一步定位

* 输入：`T1-inputs.md`；本地 `results/c3-fast-kernel-iteration/setup/{contract.json,preflight.json,interface.md,agent-input-contract.json,corpus.json,validation-A.json,validation-B.json}`。不打开 sealed prompts。
* C2：`T1-compatibility.md`、`T2-agent.md`；raw `results/c3-fast-kernel-iteration/setup/T2-agent/contract-comparison.json`。
* Device/setup：`T2-device.md`、`T3-native-first.md`、`T3-ir-fix.md`、`T3-native-recheck.md`。
* Workflow/F0：`T2-workflow.md`、`T2-workflow-publish.md`、`T3-F0.md`。
* Owner-thread：`T4-helper-owner-thread.md`，包含既有 CPU 回归和 LSP receipt。
* 发布：`T7.md`；最新解释顺序以最终 `F1.md`、`T6.md`、`T4-native-F1.md` 优先于历史 notepad。`learnings.md` 已追加最新结果。

这些 raw/.omo 不是公开数据集，全文 sessions、数据库、私有 provider 配置和模型未公开。不要为方便交接批量复制它们。

## 11. 清理与交账

历史清理快照：runner 已关闭，PID 776849 不在，无 owned runtime survivors，port 42710 可绑定，A 两卡 0MiB 且无 compute apps。`gpu-after.txt` 中 A0 瞬时利用率为 10%，不能写成所有利用率都为零，也不能据此断言有 owned 残留。**本文没有重新检查远端，这不是当前资源可用或已预留的承诺。**

未来获准运行结束时：先关闭自己拥有的 Runtime/helper/resident，确认线程/子进程退出及 GPU 工作完成；只清理自有资源，不杀他人作业。记录 runner close、owned PIDs、端口、GPU memory/compute-app 快照和时间戳；失败时保留 workspace、immutable snapshots、local raw、transport 错误和 result，不删掉“难看”的记录。

每次交账最少包括：authorization/run ID，源码 revision 和 framework ID，contract/target/oracle identity，source/bundle/params hashes，session/call ID，D/S/cutoffs，start/finish/drain，budget 的 `used` 与 `operations`，capture/profile/模型 forwards，最后状态及失败分类，清理结果。计数按实际 admitted 操作记，失败不退款，profile/capture 的双重 quota charge 不能重复解释成额外测量。没有模型 J 就写 null/NOT_RUN，不用 setup 或诊断数字补齐。

仍待操作员在未来授权中明确的事项只有：新预算范围与 clock、批准的私有配置路径、当时资源/部署状态、输入与 calibration oracle 是否获准复用，以及 formal 缺失链路如何完成。profiler/timeout 根因仍未确定；本交接不承诺修复方式或正性能结果。
