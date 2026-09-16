# C2 局部实验：双 RTX 4090 双主机接入与材料信息（回复）

> **回复对象** `v5/docs/prompt-c2-local-experiment-server-handoff.md`
> **回复方** 内部 session（Claude Code），本地 checkout `D:\Pyhon_projects\opop\v4` @ `db95a8a`
> **采集时间** 2026-09-17 00:09–00:2x CST (UTC+8)；GPU/进程快照均为该时刻只读读数
> **执行边界遵守情况** 只做只读检查：读文件、事件日志、进程元数据、设备状态、`--help`。
> 未启动任何 GPU workload、模型请求、编译、安装、下载、服务重启；未 checkout/reset/pull；
> 未编辑任何配置、代码、测试或旧文档。唯一的写操作是把一个**新增只读探针**
> （`scripts/probes/c2_local_parent_inventory.py`）scp 到两台机器的 `/root/probe-clean/`，
> 用于生成第 5 节的父代索引 —— 未触碰仓库工作树，未改任何既有文件。
>
> **状态标签** `verified_now` = 本次只读核实；`historical` = 仅历史记录；
> `unverified` = 尚待核实；`not_available` = 已确认不可取得或不存在。

---

## 1. 就绪摘要

**可以开始制定局部实验方案。** 两台主机现在**完全空闲**，材料完整度足以固定真实父代并复现决策点。
但有**三个必须先解决的缺口**，其中两个需要开发（待授权），一个需要你们裁决。

| 结论 | 状态 |
|---|---|
| 两台主机可达、四卡全空闲（0 MiB / 0% util） | `verified_now` |
| 生产代码 `src/` + `configs/` 与本地 `db95a8a` **逐字节相同** | `verified_now` |
| **v5 在两台机器上都不存在** | `verified_now` |
| 至少 7 个真实父代材料完整、C2 证据严格早于决策点 | `verified_now` |
| **没有 analyst/rewriter 单步 CLI 入口** —— `agent-smoke` 只支持 generator/parameterizer | `verified_now` |
| **没有 fresh reeval 的 CLI 入口** —— `final_reeval()` 只有库方法 | `verified_now` |
| B=40 子代调参有现成入口（`tune-file`），但需手工提供 space JSON | `verified_now` |

**首要阻塞（按处理顺序）：**

1. **单步入口缺失（需授权开发）。** 你们的第 4 条语义要求"每臂有同等的一次实际结构改写机会"，
   而 `agent-smoke --module` 的 `choices` 只有 `generator` 和 `parameterizer`
   （`src/kernel_optimizer/cli.py:585`）。**rewriter 与 analyst 无单步入口。**
   第 5 条要求"对选出的实际产物做独立最终评价"，而 `final_reeval` 只存在于
   `evaluation/benchmark.py:159`，`cli.py` 里 grep `reeval` **零命中**。
   ⇒ 这两个入口都要新写。属于第 8 节所说的"需要开发，列为待授权，不实施"。

2. **v5 不存在（需授权部署）。** 见 §3.3。因此 "direct 与 legacy 能否分别做" 目前的答案是：
   **legacy(v4) 可做，direct(v5) 不可做** —— 不是缺口未查清，是代码不在机器上。

3. **同机两卡的设备隔离只靠 `CUDA_VISIBLE_DEVICES`，没有设备级锁**（见 §3.6）。
   历史上两臂同机并跑是这么做的，且有 4 个 run 的成功记录；但这是**约定而非机制强制**。

---

## 2. P0 表

| 主机 | 字段 | 值 | 状态 | 核实时间 | 来源 | 缺口 |
|---|---|---|---|---|---|---|
| **HOST-A** | 稳定名称 | `autodl-container-459d488f2d-658995df` | `verified_now` | 2026-09-17 00:09 CST | `hostname` | 容器名可能随重建变化 |
| HOST-A | 历史别名 | **= 历史 box4** | `verified_now` | 同上 | CPU 型号 8352V 与 `v4/docs/experiment-plan-post-v4.1.md`(2026-09-15) 记录一致 | — |
| HOST-A | 接入 | `ssh -o BatchMode=yes -p 37752 -i <IDENTITY_FILE> root@connect.bjb1.seetacloud.com` | `verified_now` | 同上 | 本机 `~/.ssh/` 私钥 | identity 路径按第 2 节要求不在本文给出；密钥内容绝不外传 |
| HOST-A | CPU | Xeon Platinum **8352V** @2.10GHz，2 socket × 32 core × 2 thread = 128 | `verified_now` | 同上 | `lscpu` | 共享宿主，非独占 |
| HOST-A | RAM | 1007 GiB 总 / **942 GiB 可用** | `verified_now` | 同上 | `free -g` | 快照值 |
| HOST-A | GPU0 | RTX 4090，UUID `GPU-c896ae45-532b-2a1d-147d-ed72e221fa42`，PCI `00000000:35:00.0`，24564 MiB，**0 MiB / 0%** | `verified_now` | 同上 | `nvidia-smi` | — |
| HOST-A | GPU1 | RTX 4090，UUID `GPU-682bb59c-17a7-c7bc-2be8-83842d38f2f8`，PCI `00000000:36:00.0`，24564 MiB，**0 MiB / 0%** | `verified_now` | 同上 | 同上 | — |
| HOST-A | 驱动 | **580.105.08** | `verified_now` | 同上 | 同上 | 与 HOST-B 不同，见下 |
| HOST-A | 工作目录 | `/root/autodl-tmp`（194 GiB，**146 GiB 可用**） | `verified_now` | 同上 | `df -h` | 未创建任何测试目录 |
| HOST-A | repo | `/root/autodl-tmp/work/opop`，分支 `v4`，commit `bd65af083fc1a9bef69688e75e1b3c44174a2711`，**dirty=0** | `verified_now` | 同上 | `git -C … rev-parse/status` | 落后本地 9 个 commit，但见 §3.2 |
| HOST-A | orchestrator 解释器 | `/root/autodl-tmp/orch-venv/bin/python`，Python **3.12.4** | `verified_now` | 同上 | `-V` | — |
| HOST-A | **worker 解释器** | `/root/autodl-tmp/kernel-opt-venv/bin/python`，torch **2.13.0+cu129**，triton **3.7.1** | `verified_now` | 同上 | config `wsl.venv` + `importlib.metadata` | **命名与 HOST-B 相反，见 §3.4** |
| HOST-A | opencode | `/usr/local/bin/opencode`，版本 **1.18.29** | `verified_now` | 同上 | `bash -lc "opencode --version"` | 仅 login shell 有 PATH |
| **HOST-B** | 稳定名称 | `autodl-container-wtl0lh99as-5b475d7c` | `verified_now` | 2026-09-17 00:09 CST | `hostname` | 同上 |
| HOST-B | 历史别名 | **= 历史 box1** | `verified_now` | 同上 | CPU 8358P 与同一历史文档一致 | — |
| HOST-B | 接入 | `ssh -o BatchMode=yes autodl`（本地 SSH alias） | `verified_now` | 同上 | 本机 `~/.ssh/config` | alias 依赖本地 config；跨客户端不可移植 |
| HOST-B | CPU | Xeon Platinum **8358P** @2.60GHz，2×32×2 = 128 | `verified_now` | 同上 | `lscpu` | 比 HOST-A 快；历史实测 compile 差约 32% |
| HOST-B | RAM | 755 GiB 总 / **694 GiB 可用** | `verified_now` | 同上 | `free -g` | 比 HOST-A 少 252 GiB |
| HOST-B | GPU0 | RTX 4090，UUID `GPU-9c819228-dcf4-f1da-f21e-6b77b1c58be1`，PCI `00000000:52:00.0`，**0 MiB / 0%** | `verified_now` | 同上 | `nvidia-smi` | — |
| HOST-B | GPU1 | RTX 4090，UUID `GPU-cdff4e6f-7941-db17-7ff1-da75c7d90b24`，PCI `00000000:56:00.0`，**0 MiB / 0%** | `verified_now` | 同上 | 同上 | — |
| HOST-B | 驱动 | **580.142** | `verified_now` | 同上 | 同上 | **与 HOST-A 的 580.105.08 不同** |
| HOST-B | 工作目录 | `/root/autodl-tmp`（250 GiB，**215 GiB 可用**） | `verified_now` | 同上 | `df -h` | — |
| HOST-B | repo | `/root/autodl-tmp/work/opop`，`v4`，**同一 commit** `bd65af08…`，dirty=0 | `verified_now` | 同上 | 同上 | 两机代码完全同版 |
| HOST-B | orchestrator 解释器 | `/root/autodl-tmp/orch-venv/bin/python`，3.12.4 | `verified_now` | 同上 | 同上 | — |
| HOST-B | **worker 解释器** | **`/root/autodl-tmp/orch-venv/bin/python`**，torch 2.13.0+cu129，triton 3.7.1 | `verified_now` | 同上 | config `wsl.venv` | **与 HOST-A 名字相反但工具链相同** |
| HOST-B | opencode | `/root/miniconda3/bin/opencode`，**1.18.29** | `verified_now` | 同上 | 同上 | 路径与 HOST-A 不同 |
| 两机 | provider / model | `zhipuai` / `zhipuai/glm-5.3`，baseURL host `https://open.bigmodel.cn`，SDK `@ai-sdk/openai-compatible` | `verified_now` | 同上 | `.opencode/opencode.jsonc` 结构（值已脱敏） | apiKey 存在，**内容不在本文** |
| 两机 | "max" 的映射 | `glm-5.3.options.reasoningEffort = "max"`（**模型默认**）；`variants.max` 与 `variants.xhigh` 都映射到 `reasoningEffort: "max"` | `verified_now` | 同上 | 同上 | 见 §4.2 的重要区别 |
| 两机 | v5 是否存在 | **不存在** | `verified_now` | 同上 | `find -maxdepth 3 -type d -name v5` 零命中 | 需授权部署 |
| 两机 | 单候选评价入口 | `tune-file`（可 `--trials 1` 近似），无独立 correctness-only 入口 | `verified_now` | 同上 | `cli.py --help` | 见 §4.1 |
| 两机 | B40 子代调参 | **有** —— `tune-file --task … --candidate … --space … --trials 40` | `verified_now` | 同上 | 同上 | 需手工准备 space JSON |
| 两机 | analyst/rewriter 单步 | **不存在** | `verified_now` | 同上 | `cli.py:585` `choices=["generator","parameterizer"]` | **需开发** |
| 两机 | fresh reeval 入口 | **不存在 CLI**；库方法在 `evaluation/benchmark.py:159` | `verified_now` | 同上 | `grep reeval cli.py` 零命中 | **需开发** |
| 两机 | 窗口 1/2 证据 | **9 个 run 全部在盘、全部带 `RUN_FINISHED`** | `verified_now` | 同上 | 见 §5.1 | — |

---

## 3. 环境与接入

### 3.1 主机身份对应关系（依据，不是推测）

历史文档 `v4/docs/experiment-plan-post-v4.1.md`（2026-09-15）记录 box1 = 8358P、box4 = 8352V。
本次实测 CPU 型号：接入点 `connect.bjb1.seetacloud.com:37752` = **8352V** ⇒ **box4**；
SSH alias `autodl` = **8358P** ⇒ **box1**。这是型号级匹配，不是目录名推断。
容器 hostname 本身不含 box 编号，且可能随容器重建变化，所以**不建议**用 hostname 作稳定标识。

⚠ 两台主机**卡型完全相同**（都是 RTX 4090）。差异只在 CPU 与驱动。
按第 10 节的要求：**不可**把跨主机差异归因为路径效果。

### 3.2 代码版本：落后 9 个 commit，但生产代码相同 `verified_now`

两机都在 `bd65af08`，本地在 `db95a8a`，相差 9 个 commit。
**但我核对了这 9 个 commit 的改动范围：**

```
git diff --stat bd65af08..db95a8a -- src/ configs/    → 空输出
git diff --name-only bd65af08..db95a8a | 按顶层目录计数 →
    docs 27 个 / scripts 9 个 / tests 5 个 / .gitignore 1 个
```

⇒ **生产 `src/` 与 `configs/` 逐字节相同**，落后的全是文档、离线探针和测试。
所以"服务器落后 9 个 commit"这个说法会误导 —— **对实验行为无影响**，不需要为了跑实验先 pull。
（本文不建议 pull，那需要另行授权。）

### 3.3 v5：两台机器上都不存在 `verified_now`

```
find /root/autodl-tmp -maxdepth 3 -type d -name "v5"   → 零命中（两机）
```

存在的 repo/工作树只有：
- `/root/autodl-tmp/work/opop` —— 分支 `v4`，dirty=0（**两机唯一的 git 仓库**）
- `/root/autodl-tmp/opop-workspace/opop-glm` —— **无 `.git`**，是 opencode 的 launch_cwd（放
  `.opencode/opencode.jsonc` 与 runs 输出），不是代码副本
- `/root/autodl-tmp/opop-workspace/KernelBench` —— commit `423217d9fda91e0c2d67e4a43bf62f96f6d104f1`，
  **dirty=0**，两机相同

⇒ 回答第 5 节"实际部署版本是否包含 direct 与 legacy"：**只有 legacy(v4)，没有 direct(v5)**。

**若需部署 v5，建议位置**（`unverified`，尚未部署，需另行授权）：
`/root/autodl-tmp/work/opop-v5`（与现有 `work/opop` 平级，不覆盖任何现有工作树；
`/root/autodl-tmp` 两机分别余 146 / 215 GiB）。**我未创建该目录。**

### 3.4 ⚠ 两台机器的 venv 命名相反 —— 这是最容易踩的坑 `verified_now`

| 主机 | orchestrator venv | **worker venv（配置指定）** | worker torch / triton |
|---|---|---|---|
| HOST-A (box4) | `orch-venv` | **`kernel-opt-venv`** | 2.13.0+cu129 / 3.7.1 |
| HOST-B (box1) | `orch-venv` | **`orch-venv`** ← 同一个 | 2.13.0+cu129 / 3.7.1 |

`configs/experiments_v41_m2b_a_box1gpu0.yaml` 的注释自己写明了这件事：

> "BOX 1's WORKER VENV IS `orch-venv`, NOT `kernel-opt-venv` -- the two boxes name them the
> OPPOSITE way round. box1/orch-venv is torch 2.13.0+cu129 / triton 3.7.1, byte-identical
> toolchain to box4/kernel-opt-venv; box1/kernel-opt-venv is an OLDER torch 2.9.1 / triton 3.5.1."

**实测确认**：两个 worker 工具链**确实相同**（都是 torch 2.13.0+cu129 / triton 3.7.1）。
但 HOST-B 上还存在一个 `kernel-opt-venv`，里面是**旧的 torch 2.9.1 / triton 3.5.1**。

⇒ **按名字选 venv 会在 HOST-B 上静默用错工具链。** 必须按 config 的 `wsl.venv` 字段选。
这直接影响你们第 1 条语义（固定评价语义）：换工具链会让 tuned incumbent 不可比。

### 3.5 执行拓扑 `verified_now`

**原生 Linux，无 Docker/WSL/SSH-worker 分层**（尽管配置段名为 `wsl:`，那是 Windows 时代
遗留的段名）。orchestrator 与 worker 都在同一容器内：

```
orchestrator:  <orch-venv>/bin/python -m kernel_optimizer.cli --config <cfg> run --task level3:NN
               cwd = /root/autodl-tmp/work/opop,  PYTHONPATH=src
worker:        一次性子进程，<wsl.venv>/bin/python <repo>/src/kernel_optimizer/gpu/worker_main.py
                 --job <run>/jobs/<id>.json --out <run>/jobs/<id>.out.json
opencode:      opencode serve --hostname 127.0.0.1 --port <4096+>，cwd = launch_cwd
```

**设备映射**：`CUDA_VISIBLE_DEVICES` 在**启动命令里**设置，不落盘到任何事件。
历史命令（见 config 头部注释）：HOST-A 的 m1-a 用 `CUDA_VISIBLE_DEVICES=1`、m1-b 用 `=0`。
⇒ **进程内 cuda index 恒为 0**，物理 index 由环境变量决定。
⚠ 因此**盘上记录无法反查某个 run 用了哪张物理卡** —— 只能在运行期查并存盘。

### 3.6 ⚠ GPU 锁是 per-run，不是 per-device `verified_now`

```
worker_client.py:202   self.lock = GpuRwLock(jobs_dir / "gpu.lock", max_shared=max_shared)
wiring.py:109          jobs_dir = store.run_dir / "jobs"
```

⇒ 锁文件路径是 **`<run_dir>/jobs/gpu.lock`**，粒度是**每个 run 一把**。
两个并跑的 arm 有两个不同的 run_dir ⇒ **它们的锁互不可见，从不竞争**。
锁的语义是"同一个 run 内部：计时独占 / 正确性最多 `max_shared=2` 路并发"。

**后果**：同机两臂的设备分离**完全依赖 `CUDA_VISIBLE_DEVICES` 设对**。
没有任何机制阻止两个 run 同时使用同一张物理卡。
历史上 4 个 run（m1 对 + m2b 对）这样跑成功过，但那是**操作纪律**，不是机制保证。

当前盘上无 `gpu.lock` 残留（`find … -name "gpu.lock"` 零命中）—— 所有 run 已正常释放。

### 3.7 缓存与隔离路径 `verified_now`

每个 run 一套，按 arm 后缀隔离：

| 类型 | 路径模式 | 隔离粒度 |
|---|---|---|
| Triton 缓存 | `/root/autodl-tmp/.triton-cache-v4m1a`（HOST-A 上有 13 个这样的目录） | **每 arm** |
| opencode 状态 | `/root/autodl-tmp/xdg-v4m1a`（`XDG_DATA_HOME`） | **每 arm** |
| run 输出 | `/root/autodl-tmp/opop-workspace/opop-glm/runs-v4/<arm>/<run-id>/` | 每 run |
| job 文件 + 锁 | `<run>/jobs/` | 每 run |
| agent 沙箱 | `<run>/sandboxes/<call_id>/` | 每次调用 |
| 候选物化源码 | `<run>/candidates/<cid>/source.py`、`trials/<trial_id>.py` | 每候选 / 每 trial |

⚠ **共享的是 KernelBench 源与参考实现**（`/root/autodl-tmp/opop-workspace/KernelBench`，只读使用）。
**已知碰撞风险**：若两臂共用同一个 `triton_cache_dir` 或 `XDG_DATA_HOME`，一臂会服务另一臂的
编译结果 / 读到另一臂的 agent 会话状态。现有配置**已经**按 arm 分开，这是必须保持的。

---

## 4. 命令与评价语义

### 4.1 入口清单 `verified_now`（全部来自 `--help`，未执行任何一条）

| 入口 | 状态 | 命令形态 |
|---|---|---|
| **完整 run** | `historical`（成功用过；**本次禁止执行**） | 见下方代码块 |
| **单候选评价** | **部分** —— 无 correctness-only 入口；可用 `tune-file --trials 1` 近似 | `tune-file` |
| **B40 子代调参** | **有** | `tune-file --task … --candidate … --space <json> --trials 40 [--backend triton]` |
| **独立 analyst** | **`not_available`** | `agent-smoke` 的 choices 不含它 |
| **独立 rewriter** | **`not_available`** | 同上 |
| **fresh reeval** | **`not_available`（无 CLI）** | 库方法 `Benchmarker.final_reeval()` |

完整 run 的历史命令形态（来自 config 头部注释，`historical`，**明确禁止本次执行**）：

```
cd /root/autodl-tmp/work/opop
CUDA_VISIBLE_DEVICES=1 XDG_DATA_HOME=/root/autodl-tmp/xdg-v4m1a \
  PYTHONPATH=src /root/autodl-tmp/orch-venv/bin/python -m kernel_optimizer.cli \
  --config configs/experiments_v41_m1_a_box4gpu1.yaml run --task level3:43
```

CLI 全部子命令：`doctor` / `baseline` / `tune-file` / `agent-smoke` / `run` / `resume` /
`calibrate` / `report`。全局 flag：`--config`、`-o/--override`（点号路径，
例：`-o budgets.trials_per_space=8`）。

`tune-file` 完整签名（`cli.py:577-582`）：
```
--task TASK  --candidate FILE  --space JSON  [--trials N (默认 8)]  [--backend BACKEND (默认 triton)]
```
⚠ `--trials` **默认 8，不是 40** ⇒ 要 B40 必须显式写 `--trials 40`。

`agent-smoke` 签名（`cli.py:585`）：
```
--module {generator,parameterizer}  --task TASK  [--candidate CANDIDATE]
```
⇒ **这是第 1 号阻塞的证据**：你们要的"每臂一次实际结构改写"没有现成入口。

### 4.2 ⚠ "GLM5.3max" 的三层区分 `verified_now`

第 6 节要求区分用户称呼 / 已配置值 / 历史实际调用。**三者确实不同：**

| 层 | 值 |
|---|---|
| 用户称呼 | "GLM5.3max" |
| 已配置 | `zhipuai/glm-5.3`，其 `options.reasoningEffort = "max"`（**模型级默认**）；另有 `variants: {low, high, max, xhigh}`，其中 **`max` 与 `xhigh` 都映射到 `reasoningEffort: "max"`** |
| **历史实际调用** | `AGENT_CALL_STARTED.model` = **`zhipuai/glm-5.3`**，**无 `:max` 后缀**（m1-b：analyst 20 次 / parameterizer 21 次 / rewriter 4 次，全部同一字符串） |

⇒ **"max" 是模型默认 options 生效的结果，不是调用点指定的变体。**
harness 把 model 字符串按 `/` 拆成 `{providerID, modelID}`（`agents/runtime.py:535`），
**从不追加变体后缀** —— grep `variant` 在 `runtime.py` 零命中。
若你们需要显式控制 effort，当前路径是改 `.opencode/opencode.jsonc` 的 `options`，
而那**不是** per-call 的，会同时影响两臂。

其他模型服务事实：
- provider SDK：`@ai-sdk/openai-compatible`；baseURL host `https://open.bigmodel.cn`
  （**apiKey 存在但内容不在本文**）
- `opencode.request_timeout_s = 1500.0`（httpx 客户端超时）
- `agents.<module>.timeout_s = 1500.0`，`max_transport_retries = 2`
- `max_retries`：generator/parameterizer/rewriter = 2，**analyst = 1**
- `opencode.agent = "build"`，`permission_mode = "sandbox_config"`
- `port = 4096`，`port_attempts = 3`
- `idle_abort_frac = 0.5`，`memory_abort_frac = 0.92`，`resource_poll_s = 20.0`
- **并发上限：`unverified`** —— 配置里没有会话数上限字段；服务端速率限制未知，
  按第 6 节要求**未发请求试探**

会话隔离：每次 agent 调用建**新** opencode session，`directory` = 该次调用的沙箱
（`<run>/sandboxes/<call_id>/`）。两臂的 `XDG_DATA_HOME` 不同 ⇒ opencode 状态互不可见。
⚠ **但两臂的 runs 目录同在一个文件系统**，沙箱内的 agent 若被给到绝对路径，
理论上可读到另一臂的 run 目录。现有隔离依赖"不把那些路径写进沙箱"，
不是文件系统权限。**这是第 8 节所问"多会话是否可能读到另一臂"的诚实答案：**
`unverified`，机制上未阻断。

analyst/rewriter 是否用 scratch GPU：`unverified`。沙箱权限为 `edit/bash allow, webfetch deny`，
所以 agent **能**跑 bash；是否曾实际触发 GPU 未查（会需要读 45 个沙箱的历史命令）。
⇒ 若要保证不干扰另一臂的正式计时，建议在方案里显式处理，不要假设已隔离。

### 4.3 实际 resolved 评价语义 `verified_now`（读自 `manifest.json`，非默认文件名）

| 字段 | 值 |
|---|---|
| `evaluation.correctness_trials` | 5 |
| `evaluation.perf_trials` | **100** |
| `evaluation.quick_correctness_trials` | 3 |
| `evaluation.quick_perf_trials` | **20** |
| `evaluation.timing_method` | `cuda_event` |
| `evaluation.precision` | `fp32`（参考侧；候选可声明更低精度） |
| `evaluation.correctness_mode` | **`dual_witness_relaxed`** |
| `evaluation.eval_timeout_s` | 600.0 |
| `evaluation.build_timeout_s` | **1200.0** |
| `evaluation.suspicious_speedup` | **`None`** |
| `budgets.trials_per_space` | **40** ← B40 |
| `budgets.rewrite_rounds_per_family` | 5（resolved 值；实验 config 用 3） |
| `budgets.max_families_active` | 3 |
| `budgets.max_seed_candidates` | 4 |
| `budgets.wall_clock_hours` | 12.0 |
| `budgets.repair_attempts` | 3 |
| `budgets.space_expansions_per_candidate` | 1 |
| `gpu.concurrency` | `enabled=true, max_shared_jobs=2, vram_budget_frac=0.45, timing_cooldown_s=2.0` |
| `gpu.compile_screen_enabled` | true |

**B40 的计数单元**（第 7 节明确问的）：`trials_per_space` 是**每个已发布参数空间** 40 个 trial，
**不是每候选**。一个候选若发生 space expansion 会发布第二个空间，于是同一候选可累计 80 个 trial
—— 这**不是超支**。失败的 trial **计入**这 40；标记 `reused_measurement` 的记录也计入
trial 数。**本文只描述，不建议修改预算规则。**

**计时覆盖范围**：`cuda_event` 计时**只覆盖 kernel 执行**，不含编译。
编译耗时单列在 `job_wall_s`（含 lock wait + 编译 + 执行）与 `compile_s`。
⇒ 区分编译与运行计时靠这两个字段，不靠单一数字。

**quicktest / tuning best / 最终评价的区别**：
- quicktest = 3 次正确性 + 20 采样（发布空间前的双见证门）
- tuning best = 每 trial 20 采样的 median，是 TPE 的目标
- 最终评价 = `final_reeval`，**独立新进程 + 100 采样**，结果只写在
  `RUN_FINISHED.summary.best.final_reeval_median_ms`

⚠ **没有独立的 re-eval 事件类型** —— 找 `FINAL_REEVAL_DONE` 会得到 0 个事件。
⚠ `latency_ms` 的键是 `median` / `mean`，**没有 `_ms` 后缀**。写错会让每个 trial 读成 `None`。

**测量复用**：`TRIAL_DONE.payload.reused_measurement` 为 true 表示该记录复用了既有测量。
`tpe.py:191` 有 `enqueue_fresh()`（`skip_if_exists=False`）可强制新测，
但**没有 CLI 暴露它** ⇒ 你们第 5 条要求的"禁用测量复用的独立复测"目前需要开发。

**perf logs / 原始样本**：`BASELINE_DONE.payload.baseline.latency_ms.samples` 有 100 个原始
采样值（已核实 n1-a/n1-b/m1-a 均有）。trial 侧 `latency_ms` 只有 summary 统计量
（median/mean/std/min/max/n_samples），**trial 的原始采样不落盘** ⇒ 按第 7 节要求注明：
**trial 无原始样本**。

**PRUNED 语义**：`enqueue()` 的一个已知返回值是 `already_drawn` —— 重复 enqueue 的点会被
`ask()` 返回后判 PRUNED。**本文只描述，不建议修改**（用户已否决改 PRUNED 行为）。

---

## 5. 父代和历史索引

### 5.1 窗口 1/2 证据完整度 `verified_now`

**9 个 run 全部在盘，全部带 `RUN_FINISHED`：**

| 主机 | arm | run 目录 | 任务 | 大小 |
|---|---|---|---|---|
| HOST-A | `m1-a` | `runs-v4/m1-a/run-l3-43-20260915-220040` | level3:43 | 38 MB |
| HOST-A | `m1-b` | `runs-v4/m1-b/run-l3-43-20260915-220041` | level3:43 | 43 MB |
| HOST-A | `n1-a` | `runs-v4/n1-a/run-l3-43-20260915-071127` | level3:43 | 33 MB |
| HOST-A | `n1-b` | `runs-v4/n1-b/run-l3-43-20260915-071127` | level3:43 | 39 MB |
| HOST-A | `verify` | `runs-v4/verify/run-l1-1-20260915-054554` | level1:1 | 5 MB |
| HOST-B | `m2b-a` | `runs-v4/m2b-a/run-l3-21-20260915-220113` | level3:21 | 50 MB |
| HOST-B | `m2b-b` | `runs-v4/m2b-b/run-l3-21-20260915-220113` | level3:21 | 59 MB |
| HOST-B | `pilot-p1` | `runs-v4/pilot-p1/run-l3-43-20260915-070539` | level3:43 | 41 MB |
| HOST-B | `pilot-p2` | `runs-v4/pilot-p2/run-l3-21-20260915-070539` | level3:21 | 47 MB |

根路径（两机相同）：`/root/autodl-tmp/opop-workspace/opop-glm/runs-v4/`
⚠ **`n1-a` 与 `n1-b` 共用同一个 run_id**，`m2b-a`/`m2b-b` 同样 ⇒ **必须以目录身份区分，不能用
run_id**。这与你们第 4 节的要求一致。

每个 run 目录内容（已核实存在）：`events.jsonl`、`manifest.json`、`state.json`、
`candidates/`、`artifacts/`、`jobs/`、`sandboxes/`、`report/`、`opencode-server.log`。

HOST-B 另有更早的 run 根：`runs-v3`、`runs-l3`、`runs-smoke`、`runs-smoke42`、
`runs-l2-37-orig`（本次未展开）。

### 5.2 可用父代索引 `verified_now`

用新增只读探针 `scripts/probes/c2_local_parent_inventory.py` 生成。
判据按你们第 8 节：物化源码是否在盘、tuned incumbent（**取 MIN 而非 LAST**）、
C2 contrast 是否**严格早于**改写决策、是否已有后代（选择偏差风险）。

**决策时点的近似方式**：取该父代**最早**一个子代的 `REWRITE_PRODUCED.seq`。
⚠ `REWRITE_PRODUCED.payload.candidate_id` 是**子代**，且该事件**不带 `parent_ids`**
（血缘只在 `CANDIDATE_REGISTERED`）。我第一版探针按 child id 比对，得到的是"子代 contrast 全在
自己出生之后"这种**恒真而无信息**的结果，已修正为按 lineage 解析到父代。

#### HOST-A · `m1-b`（active，level3:43）—— 4/4 个种子父代可用

| 父代 | origin | 源码在盘 | tuned min (ms) | contrast | brief | 后代数 | family | C2 时序判决 |
|---|---|---|---|---|---|---|---|---|
| `cand-d4594e97` | seed | 是 | **3.4463** | 2 | 2 | 2 | `fam-78d4d53c` | **可用**：证据早于决策(seq 548) |
| `cand-1653d847` | seed | 是 | **3.6045** | 1 | 1 | 2 | `fam-af049ce1` | **可用**：证据早于决策(seq 844) |
| `cand-d9e9d98c` | seed | 是 | **5.2193** | 2 | 1 | 2 | `fam-86d4a827` | **可用**：证据早于决策(seq 1106) |
| `cand-b9cf9167` | seed | 是 | **7.0942** | 2 | 2 | 2 | `fam-006eae98` | **可用**：证据早于决策(seq 1331) ⚠ LAST≠MIN |

**4/4 都是 `contrast before=N, after=0`** ⇒ C2 证据在决策前已完整可得，无泄漏。

改写子代（**它们的结果我们已知 ⇒ 选择偏差风险，不可作为 A/B 输入**）：
`cand-fd5d4772` 2.7013 / `cand-02fd4efc` 3.2947（父 `d4594e97`）；
`cand-427292c0` 3.2102 / `cand-cc1f81b3` **2.6465**（父 `1653d847`，run 最终赢家）；
`cand-e9d7c244` 3.9629 / `cand-f4742672` 3.1601（父 `d9e9d98c`）；
`cand-6d5655d0` 3.6485 / `cand-72629423` **无 tuned 值**（父 `b9cf9167`）。

⚠ `cand-b9cf9167` 的 **LAST≠MIN** 标记是实测出来的：它的最后一个 `TUNING_DONE` 不是最优的那个。
这正是我们记录过的 reader 脆弱点（本批 8/8 一致，但**这个候选是反例**）⇒ 取 incumbent 必须取 MIN。

#### HOST-A · `m1-a`（off，level3:43）—— 强基线对照，**0 个 contrast**

| 父代 | origin | 源码在盘 | tuned min (ms) | contrast | 后代数 | family |
|---|---|---|---|---|---|---|
| `cand-4e9c333d` | seed | 是 | **3.3603** | **0** | 2 | `fam-f1263c1b` |
| `cand-b67a1cb4` | seed | 是 | **3.5000** | **0** | 2 | `fam-a6e4cb65` |
| `cand-919e19b8` | seed | 是 | **5.6525** | **0** | 2 | `fam-85cf4dc3` |
| `cand-53ded749` | seed | 是 | **7.1956** | **0** | 2 | `fam-43978965` |

4/4 均判 "no contrast at all" ⇒ **off 臂的机制侧结构性为零**，符合设计。
其子代 `cand-682405a1`（2.5155）是该 run 的赢家。`cand-3854ffae` 亦标 LAST≠MIN。

#### HOST-B · `m2b-b`（active，level3:21）—— 3/4 个种子父代可用

| 父代 | origin | 源码在盘 | tuned min (ms) | contrast | brief | 后代数 | family | C2 时序判决 |
|---|---|---|---|---|---|---|---|---|
| `cand-aab9ee80` | seed | 是 | **5.3407** | 1 | 1 | 2 | `fam-4526b58e` | **可用**(seq 479) |
| `cand-6103ab28` | seed | 是 | **5.9141** | 2 | 2 | 2 | `fam-744d8f9d` | **可用**(seq 755) |
| `cand-ca37f117` | seed | 是 | **6.2182** | 2 | 1 | 2 | `fam-7dabd1e3` | **可用**(seq 994) |
| `cand-12a5a601` | seed | 是 | 9.5539 | **0** | 0 | 2 | `fam-cce86a97` | **不可用**：无 contrast |

⇒ **合计 7 个可用父代**（m1-b 4 个 + m2b-b 3 个），横跨两个任务、两台主机。
另有 8 个 off 臂父代可作强基线侧的对照材料。

### 5.3 C2 证据的具体内容 `verified_now`

每个可用父代的 contrast 都能取到：F/N 端点值、固定 partners、参数轴、方向、
discovery/validation 记录。以 m1-b 为例（`noleak` 定义，n=7 可比）：

轴分布：`NUM_STAGES`、`ATT_STAGES`、`GEMM_BLOCK_K`(×4)、`PV_BLOCK_N`、`SCORE_BLOCK_N`、
`BLOCK_N`、`COMPUTE_DTYPE`(×4)。铸币方向：inward 8 / outward 1 / unresolved 4。
简报交付 13 次。m2b-b：`BLOCK_N`、`BLOCK_K`(×5)、`GEMM_STAGES`、`BLOCK_M`(×2)；
outward 4 / unresolved 5；简报交付 9 次。

⚠ 按你们第 8 节的警告：**简报交付数 > 0 不证明 agent 读过或使用了它。**
我们自己的第四档测量（改写文本是否点名该轴）读数为 **m1-b 1/5、m2b-b 1/4**，
窗口 1 两个 pilot 均为 **0** ⇒ 交付与使用之间有实测缺口。

### 5.4 局部单步 replay 的支持情况

| 需要的能力 | direct(v5) | legacy(v4) | 依据 |
|---|---|---|---|
| 固定父代 + 物化源码 | **代码不在机器上** | **支持** | 15/15 候选 `source.py` 在盘 |
| 固定 tuned incumbent | 同上 | **支持** | `TUNING_DONE` 取 MIN |
| 单独跑 rewriter | 同上 | **不支持** | `agent-smoke` choices 无它 |
| 单独跑 analyst | 同上 | **不支持** | 同上 |
| B40 子代调参 | 同上 | **支持** | `tune-file --trials 40` |
| 禁用复用的独立复测 | 同上 | **不支持（无 CLI）** | `grep reeval cli.py` 零命中 |
| 隔离 C2 而不泄漏强基线 | 同上 | **配置层面支持** | `conditional_scan.mode` off/active；实测两对 125 键仅此一项差异 |

**"能否保留强基线同时隔离 C2 而不泄漏"** —— 配置层面已验证可行：
m1 与 m2b 两对的解析后配置各 125 键，唯一值差异是 `v4.conditional_scan.mode`，
其余为必要隔离路径（runs_dir / XDG / triton cache），且 `seed_candidates_dir` 两臂相同。
off 臂五类机制事件全为 0。⇒ **自变量干净**，这套隔离可直接复用。

---

## 6. 耗时与并行判断

### 6.1 实测 wall duration `verified_now`（从 `AGENT_CALL_STARTED`/`FINISHED` 配对算出）

**HOST-A `m1-b`（level3:43，elapsed 13.448h）：**

| 模块 | n | median (s) | mean (s) | max (s) | 合计 (h) |
|---|---|---|---|---|---|
| analyst | 20 | 272.4 | 276.9 | 394.8 | 1.54 |
| parameterizer | 21 | 177.2 | 192.7 | 539.7 | 1.12 |
| **rewriter** | **4** | **519.2** | **585.3** | **902.8** | **0.65** |
| **全部 agent** | 45 | 246.8 | 265.0 | 902.8 | **3.31** |
| trial (`job_wall_s`) | 607 | 19.5 | 30.5 | **1802.4** | **5.14** |
| space prescreen | 20 | 150.3 | — | — | 0.84 |

**HOST-B `m2b-b`（level3:21，elapsed 12.399h）：**

| 模块 | n | median (s) | mean (s) | max (s) | 合计 (h) |
|---|---|---|---|---|---|
| analyst | 21 | 360.0 | 375.0 | 579.0 | 2.19 |
| parameterizer | 23 | 199.3 | 216.4 | 691.9 | 1.38 |
| **rewriter** | **4** | **736.2** | **831.3** | **1306.3** | **0.92** |
| **全部 agent** | 48 | 325.5 | 337.0 | 1306.3 | **4.49** |
| trial (`job_wall_s`) | 743 | 16.9 | — | — | **4.28** |
| space prescreen | 21 | 102.1 | — | — | 0.62 |

**对你们的局部实验最相关的三个数（`verified_now`）：**
- **一次 rewriter 调用：median 519–736 s，max 1306 s**（≈22 分钟）
- **一次 analyst 调用：median 272–360 s**
- **B40 一个空间的 trial 成本：** 按 median trial 16.9–19.5 s × 40 ≈ **11–13 分钟**
  （不含编译尾部；见下方 max）

⚠ **`job_wall_s` 的 max 1802.4 s 是一次病态编译**（`ieee`/`split3`/`NUM_WARPS=1` 生成
16.5 MB / 316167 行 PTX，`ptxas` 烧满 `build_timeout_s + eval_timeout_s = 1800 s` 后被杀，
记为 `failure_kind=timeout`）。**单次事件**，但它占了该 arm 全部 GPU 墙钟的 9.7%
⇒ 局部实验的时间预算必须留这条尾巴。

⚠ 并行/重叠时间**不可直接相加**：agent 调用与 GPU trial 有重叠。
m1-b 的 3.31h agent + 5.14h trial = 8.45h，而 elapsed 是 13.448h ⇒ 有 5h 未被这两项覆盖
（含 prescreen 0.84h、编排开销、以及本次未拆解的等待）。
**旧 12h 整包耗时不能当作本次局部步骤的预算。**

### 6.2 四卡并行可行性（只回答，不试跑）

| 问题 | 答案 | 状态 |
|---|---|---|
| 每台主机能否各承载一个完整 matched A/B 对（每臂一张独占卡）？ | **能** —— 历史上正是这么跑的（m1 对在 HOST-A，m2b 对在 HOST-B，各 2 arm × 1 卡） | `historical` + 当前四卡空闲 `verified_now` |
| 独占由谁确认？ | **没有机制确认。** GPU 锁是 per-run（§3.6），分离靠 `CUDA_VISIBLE_DEVICES` 设对 | `verified_now` |
| 同机两卡共享什么？ | **共享 CPU（128 线程）、RAM、PCIe、磁盘、文件系统**。GPU 显存/SM 不共享（不同物理卡） | `verified_now` |
| 已有争用证据？ | **有间接证据**：agent 调用占 elapsed 的 66–70%，GPU 只占 30–34% ⇒ 关键路径主要在模型服务而非 GPU。两臂同机时 CPU 侧编译会竞争 128 线程 | `verified_now`（比例）/ `unverified`（竞争幅度未单独测） |
| 多个 LLM 会话能否并行？ | **历史上可以**（每对同时跑 2 个 arm，各自 opencode server + 独立 XDG）。**上限未知** | `historical` / 上限 `unverified` |
| scratch GPU / 编译会否干扰另一臂的正式计时？ | **`unverified`。** 沙箱允许 `bash`，未核实 agent 是否曾用 GPU。同机两臂的编译**确实**竞争 CPU | `unverified` |
| 两机能否并行同一路径的完整对？ | **能**（各一对） | `verified_now`（资源）/ 入口缺口见 §1 |
| 两机能否各跑 direct / legacy 的完整对？ | **不能** —— **v5 不在任何一台机器上**（§3.3） | `verified_now` |

⚠ 按你们第 10 节的红线：**不得把 direct 永久固定在一台主机、legacy 固定在另一台，
再把跨机差异归因为路径效果。** 两台主机 CPU 不同（8358P vs 8352V，历史 compile 差约 32%）、
驱动不同（580.142 vs 580.105.08）、RAM 不同（755 vs 1007 GiB）。
若将来要比较 direct 与 legacy，**必须处理主机混杂**（例如两条路径在同一主机上分批跑，
或每条路径在两台主机各跑一次）。

**不保证四卡带来四倍加速。** 明确的串行关键路径与未知项：
- **模型服务吞吐**是主要瓶颈（agent 占 66–70%）；四个并发 arm 会把请求量翻倍，
  服务端速率限制**未知**（按边界未试探）
- 单个 rewriter 调用 median 519–736 s 是**不可并行的串行段**（一臂只有一次改写机会）
- 同机两臂的 `ptxas` 编译竞争 CPU；那条 1802 s 的病态编译期间会占满一个核
- 磁盘：两臂共写同一文件系统（HOST-A 余 146 GiB、HOST-B 余 215 GiB；
  单个 run 目录 33–59 MB，不是约束）

---

## 7. 缺口与授权边界

### 7.1 只需补文件/开发（不需要额外接入信息）

| 缺口 | 需要什么 | 阻塞等级 |
|---|---|---|
| **rewriter 单步入口** | 扩展 `agent-smoke --module` 的 choices，或新增子命令。需读取：父代源码、bottleneck report、可选简报 | **阻塞你们第 4 条语义** |
| **analyst 单步入口** | 同上 | 阻塞（若要复现 analyst→假设 这一环） |
| **fresh reeval 入口** | 暴露 `Benchmarker.final_reeval()` 为 CLI，并接上 `enqueue_fresh` 的禁用复用语义 | **阻塞你们第 5 条语义** |
| **space JSON 提取** | `tune-file --space` 需要一个 JSON；现有 run 里空间定义在 `SPACE_PUBLISHED` 事件内，需一个导出小工具 | 中等（纯读侧） |
| v5 部署 | 建议 `/root/autodl-tmp/work/opop-v5`（未创建） | **阻塞 direct 路径** |

以上全部**列为待授权，本次未实施**。

### 7.2 需要私下补齐的接入信息

| 项 | 说明 |
|---|---|
| SSH identity 文件路径 | 按第 2 节要求不在本文给出。两台机器的认证**当前已在本地可用**，无需新凭据 |
| GLM API key | 存在于两机的 `.opencode/opencode.jsonc`（第 9 行）。**内容不在本文**。若外部执行者需要自己的额度，须另行提供 |

⚠ **一并提醒**：该 API key 目前存在于**四台机器**上（含两台服务器）。
按既有纪律，交付时应轮换。这不影响本次实验。

### 7.3 必须另行授权才能验证的项

| 项 | 为什么本次没查 |
|---|---|
| 模型服务并发上限与速率限制 | 需要发真实请求试探，违反第 2 节边界 |
| agent 沙箱是否曾使用 GPU | 需要读 45–48 个历史沙箱的命令记录；可做但量大，且部分沙箱可能已清理 |
| 同机两臂的 CPU 争用幅度 | 需要负载测试，第 10 节明确禁止 |
| 两臂之间是否真的读不到对方文件 | 需要实际尝试跨臂读取，属于验证性写操作 |
| HOST-B 的 `runs-v3`/`runs-l3` 等更早 run | 本次未展开；若需要可另行索取 |
| 维护窗口 / 共享用户 / 资源预约方式 | AutoDL 容器，`unverified`。**当前空闲不等于未来预约或独占承诺** |

### 7.4 已足够接手的部分

- 两台主机的接入、资源、代码版本、解释器、设备映射、缓存隔离路径 —— **全部 `verified_now`**
- 7 个可用真实父代 + 8 个 off 臂对照父代，材料完整度已逐项核实
- 完整的 resolved 评价语义（含 B40 计数单元、计时覆盖范围、复用判定）
- 实测的 per-module wall duration，可用于估算局部步骤预算
- C2 隔离方案（`conditional_scan.mode`）已在两对实验上验证自变量干净

---

## 附：本次新增的只读探针

`scripts/probes/c2_local_parent_inventory.py`（已 scp 到两机 `/root/probe-clean/`）。
只读 `events.jsonl` 并 stat 候选树，无 GPU、无 torch、无候选执行。
用法：`PYTHONPATH=<repo>/src <orch-venv>/bin/python c2_local_parent_inventory.py <run_dir> …`

它内建三条你们第 8 节点出的判据：
1. tuned incumbent 取 **MIN** 而非 LAST，并在两者不等时打 `LAST!=MIN`（实测已捕到 2 例）；
2. C2 证据的时序按 **lineage 解析到父代**，而非按 `REWRITE_PRODUCED.candidate_id`
   （后者是子代，比对结果恒真无信息 —— 我第一版就是这么错的）；
3. 后代数单列，作为选择偏差风险的显式标记。
