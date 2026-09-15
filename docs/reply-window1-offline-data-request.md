# 对 window1 离线核验数据请求的答复

> 请求来源:`docs/prompt-window1-offline-data-request.md`。
> 本文按其要求的六部分组织。日期:2026-09-16。
>
> **操作边界已遵守**:未启动任何 GPU 作业,未重启在跑的窗口 2 四个 run,未修改生产
> source/config/prompts/scorer,未改动任何原始日志。核验只做**读取 + 哈希 + 目录枚举**。
>
> **§5 交付方式(压缩包上传 / 开放只读 SSH)尚未执行**,原因见第 2 部分:两者都会把数据
> 送出本机,须先经用户批准。第 1、3、4、5、6 部分是不需要外发即可完成的全部核验工作。

---

## 0. 本次核验推翻了我自己的一处结论(最重要,先说)

请求 §3 要求核实 `n=13, median=1.03, p90=2.50, max=5.33` 这组数字的口径。核实过程中
我发现**我给这组数字起的名字是错的,这是同一条 N1 结论的第四次更正**:

我先前称它为"**同配置** A/A 噪声(13 个 byte-identical 配置)"。实测:

```
executed code byte-identical in 0 / 13 pairs
```

**13 对里没有一对真正跑了逐位相同的代码。** 原生成器的配对键是
`(注册种子 source_sha, 声明的 knob 字典)` —— 它固定了**种子**与**旋钮**,
但**没有固定参数化后的程序体**。parameterizer 是一次 LLM 调用,两臂把同一个种子
变成了两个不同的程序,而**两者都保留种子的 `source_sha`,连 `SPACE_PUBLISHED.source_sha`
也相同**(都是 `49d10c6bda13`),所以任何按 sha 配对的检查都看不出差别:

| 种子 | 两臂参数化体的差异行数 | 行数 A / B |
|---|---|---|
| `20eb42e310df` | **22 行** | 224 / 228 |
| `49d10c6bda13` | **18 行** | 221 / 229 |
| `6012d8732b9f` | 2 行 | 185 / 185 |

差异是实质性的,不是注释。`49d10c6bda13` 上 n1-b 的 parameterizer 把精度 constexpr
(`_DT` / `_MODE`)从模块级求值改成了**每次 forward 重新推导**,并改了 `tl.dot` 的
`input_precision` 注释路径。

**因此这组数字的正确名称是:**

> **13 对"同种子 + 同声明旋钮"跨臂配对的延迟差,`|B−A|`:
> n=13 min 0.01% median 1.03% p90 2.50% max 5.33%。
> 它混合了计时噪声与两臂参数化差异,是同代码测量噪声的【上界】,不是对它的测量。**

**真正的同代码 A/A**(同一个 materialized artifact 被计时多次)在这批日志里极稀疏:

| 集合 | n | min | median | p90 | max |
|---|---|---|---|---|---|
| **fresh vs fresh(无 reused 记录)** | **3** | 0.04% | **0.06%** | 0.06% | **0.07%** |
| 含 ≥1 条 reused 记录 | 30 | 0.00% | 0.03% | 0.55% | 8.37% |

⇒ **n=3 太小,不能作为承重阈值。** 这不改变"4% 不能默认外推、每个目标×场地都要自己标定"
这个要求(该要求反而更强),但**它改变了标定的做法**:必须**显式重复同一个 artifact**
才能标定,不能靠事后从搜索日志里捞配对。

**连带影响:上一轮 prompt(`docs/prompt-v5-revision-handoff-round2.md` §1.2 与第五部分第 1 条)
里"同配置测量噪声 median 1.03% / p90 2.50% / max 5.33%"的措辞作废**,请改用上面的上界措辞。
τ=4% 落在这个**上界**的 p90 与 max 之间 —— 结论"保守但不荒谬"仍成立,但依据变弱了。
六个投递点的 4 输 2 平判决**不变**(4% 与 5.33% 同判决,已在上一轮核实)。

新探针:`scripts/probes/v41_aa_pair_artifacts.py`、`scripts/probes/v41_same_code_repeats.py`。
**这两个是新分析,标注为"新分析",不是"旧结果复现"。**

---

## 1. 材料清单

### 1.1 已核实的时间戳子目录(绝对路径)

四个父目录下各有多个子目录。判断依据:**只有一个子目录含 `RUN_FINISHED` 事件**,
其余是冒烟(`v41-smoke-*`)或评测自检(`evalcheck-*`)残留。未做任何"挑好看的"选择。

| Run | 机器 | **实际子目录(绝对路径)** | 同父目录下的其他子目录 |
|---|---|---|---|
| n1-a | box4 | `/root/autodl-tmp/opop-workspace/opop-glm/runs-v4/n1-a/run-l3-43-20260915-071127/` | 5 个 `v41-smoke-17894212xx`(冒烟,无 RUN_FINISHED) |
| n1-b | box4 | `/root/autodl-tmp/opop-workspace/opop-glm/runs-v4/n1-b/run-l3-43-20260915-071127/` | 无 |
| pilot-p1 | box1 | `/root/autodl-tmp/opop-workspace/opop-glm/runs-v4/pilot-p1/run-l3-43-20260915-070539/` | 2 个 `evalcheck-2026091506581x` + 2 个 `v41-smoke-17894269xx` |
| pilot-p2 | box1 | `/root/autodl-tmp/opop-workspace/opop-glm/runs-v4/pilot-p2/run-l3-21-20260915-070539/` | 无 |

⚠ **n1-a 与 n1-b 的 `run_id` 完全相同**(都是 `run-l3-43-20260915-071127`),
`RUN_CREATED.payload` 里只有 `run_id` 一个字段。**臂身份只能由父目录名区分**,
不能由 run_id 区分。请在核验脚本里按父目录取臂。

### 1.2 run 元数据

| 项 | n1-a | n1-b | pilot-p1 | pilot-p2 |
|---|---|---|---|---|
| run_id | `run-l3-43-20260915-071127` | **同左(相同!)** | `run-l3-43-20260915-070539` | `run-l3-21-20260915-070539` |
| task | `43_MinGPTCausalAttention` | 同 | `43_MinGPTCausalAttention` | `21_EfficientNetMBConv` |
| 参考源 sha | `9d280c98c015edca…` | 同 | 同 | (见各自 state.json) |
| 机器 | box4 | box4 | box1 | box1 |
| host 容器名 | `autodl-container-459d488f2d-658995df` | 同 | `autodl-container-wtl0lh99as-5b475d7c` | 同 |
| CPU | Xeon Platinum **8352V** @2.10GHz | 同 | Xeon Platinum **8358P** @2.60GHz | 同 |
| GPU 型号/数量 | 2× RTX 4090 24564 MiB | 同 | 2× RTX 4090 24564 MiB | 同 |
| **该臂实际占用的物理卡** | GPU 0 (`GPU-c896ae45…`) | GPU 1 (`GPU-682bb59c…`) | GPU 0 (`GPU-9c819228…`) | GPU 1 (`GPU-cdff4e6f…`) |
| 卡分离证据 | `/root/n1-gpu-pinning.txt`(运行期 `/proc` 读 CVD + 120s 采样,印 `SEPARATED`,首次尝试即答) | 同一文件 | `/root/pilot-gpu-pinning.txt` | 同一文件 |
| **启动 commit** | **`1fb9c7d4669dd8541b7812bdd3d1a241e9c46f21`** | 同 | **`f159555892ebc60601364b7636681bf85908f3b1`** | 同 |
| commit 证据 | box4 `git reflog --date=iso`:`1fb9c7d HEAD@{2026-09-15 07:11:18}`,run 目录 `created 07:11:27` | 同 | box1 reflog:`f159555 HEAD@{2026-09-15 06:56:36}`,run 目录 `created 07:05:39` | 同 |
| dirty 工作区 | **未知**(见下) | 未知 | 未知 | 未知 |
| venv / toolchain | `/root/autodl-tmp/kernel-opt-venv`(cmdline 证据) | 同 | `/root/autodl-tmp/orch-venv`(cmdline 证据) | 同 |
| 事件行数 | 1222 | 1423 | 1425 | 1174 |
| RUN_FINISHED | present | present | present | present |
| best candidate | `cand-a14c7861` | `cand-e72bb36c` | `cand-0e23a83e` | `cand-87c56c65` |
| `final_reeval_median_ms` | 2.8743679523468018 | 2.5374720096588135 | 2.888144016265869 | 3.60806405544281 |

⚠ **两对 run 不在同一个 commit 上启动。** pilot 对在 `f159555`,n1 对在 `1fb9c7d`。
差异内容:`git diff --name-only f159555 1fb9c7d` = `src/kernel_optimizer/control/orchestrator.py`
+ `tests/test_v41_seed_pairing.py`,提交信息 "Refuse to start when a frozen seed is dropped
by signature dedup"(启动前的种子校验硬门)。**同一对内部两臂 commit 相同**,
所以配对比较成立;**跨对并表时必须声明这个差异**。

⚠ **dirty 工作区状态标注为"未知"。** `RUN_CREATED` 不记录 commit 或 dirty 标记,
当时也没有留 patch。两台机器**当前**都是 `git status --porcelain` 空、HEAD 在 `bd65af0`,
**但这是当前状态,不能冒充启动状态**。按请求要求,列为缺失(见第 5 部分)。

### 1.3 文件清单、大小与 SHA256

**核验必需的最小集**(每 run 三个文件):

| Run | 文件 | 字节 | SHA256 |
|---|---|---|---|
| n1-a | `events.jsonl` | 2025811 | `f2ff6060c4fa0dd721c02c64610994ff70e2b6f7b5aea23d2246e1ce183ab34a` |
| n1-a | `manifest.json`(resolved config,实际生效) | 5132 | `71c2ec5d4d3a81a85d2a9aabfbcf0ddba266a31ccd70ec316d6fce396d5e005f` |
| n1-a | `state.json` | 315 | `d92f14cad4d6f7e868fb0278ad88dda55df7b365fd109996d4b3950a7364e371` |
| n1-b | `events.jsonl` | 2350454 | `b0727dc8329949b19ab98b9258323f9ce445588026bd433b0936b94084c5f1fe` |
| n1-b | `manifest.json` | 5132 | `d32f95f4f3ef9fb48dcd99af46451ab028f34d846746eabf69f438ebb38495de` |
| n1-b | `state.json` | 315 | `d92f14cad4d6f7e868fb0278ad88dda55df7b365fd109996d4b3950a7364e371` |
| pilot-p1 | `events.jsonl` | 2205024 | `4b322dbbdb8cacfb6e3e0328737d53e68bd96ab14b148c4286aab273e48a70be` |
| pilot-p1 | `manifest.json` | 5076 | `23bb8c4516eb901a2a92fcbcbe4c7a5b965636fce5930fb00e8164634b0317f1` |
| pilot-p1 | `state.json` | 315 | `d92f14cad4d6f7e868fb0278ad88dda55df7b365fd109996d4b3950a7364e371` |
| pilot-p2 | `events.jsonl` | 2408586 | `1c2b4591f6cddbc93970efb2f8fcf78902d1fcb7caa965cb1adc424ccda6116c` |
| pilot-p2 | `manifest.json` | 5073 | `88510b8a1c0c9f97720dcbaa118f63ae4d3421324a6009dbbae5a9c0ec0df776` |
| pilot-p2 | `state.json` | 309 | `c365565fc79a1fcfd78138822772aa38ba3f67ce01088d6ddf5ba9f081201677` |

注:三个 run 的 `state.json` 哈希相同 —— 它们是同一个 task(L3:43)的同一份任务描述,
`phase: started`,不含 run 特有信息。pilot-p2 不同是因为它是 L3:21。

**已有的 stdout / 报告**(未重跑,原样):

| 文件 | 字节 | SHA256 | 说明 |
|---|---|---|---|
| box4 `/root/n1-a.log` | 14380 | `39956c6fad263d9e22393757781c3c90b8073e18a6c148b7154c346122d6b6a3` | 编排器 stdout |
| box4 `/root/n1-b.log` | 16498 | `491458d57427c2bb2108e75c624c1bb2eb65c9f7b0fc80030ea314e11b62dc13` | 同 |
| box4 `/root/n1-preaudit.txt` | 1040 | `d88d2daceb904a355009b17a4a02687522905cd5b8cebb4c19bbf12fc0db4c07` | 启动前配置对等审计:125 键比对,3 处差异(全是隔离路径),印 `COMPARABLE` |
| box4 `/root/n1-gpu-pinning.txt` | 1075 | `90524c33490e70119e44285e49db279a7f782b792af24053fb4c03d668dd4ab0` | 运行期卡分离证据 |
| box1 `/root/pilot-p1.log` | 15008 | `83e57858c3ef46f8f546e0f1e5eae394681f2be62816b7408981111278c2809a` | 编排器 stdout |
| box1 `/root/pilot-p2.log` | 13779 | `18ac4e78adafd005a42b7c36a74d7f1037f28465b164db5ed8c209aaffb179ae` | 同 |
| box1 `/root/pilot-gpu-pinning.txt` | 1096 | `86aa6aba2b4fc25ae608b6543773798542f065a1b30a643815cd9a7cbe7d751e` | 运行期卡分离证据 |

每 run 目录还含 `report/`(含 `report.md`、`best_kernel.py`)、`candidates/`、
`sandboxes/`、`jobs/`、`artifacts/`(pilot 另有 `uw_probes/`)。目录字节数:

| Run | 整个 run 目录 | `candidates/*/trials/*.py` 个数 | trials 总字节 |
|---|---|---|---|
| n1-a | 20693919 | 1200 文件(含 .json 侧车) | 12333132 |
| n1-b | 24237806 | 1320 | 14592490 |
| pilot-p1 | 28566800 | 1200 | 13361335 |
| pilot-p2 | 34795063 | 1040 | 15901685 |

**四个 run 整体 < 110 MB**,不含权重、venv、数据库。逐文件 SHA256 清单将随压缩包生成
(经批准后),此处先给核验必需的三个文件。

---

## 2. 交付与访问

**尚未执行,须先经用户批准。** 请求 §5 的两条路径都会把数据送出本机:

- **压缩包上传**:需要一个外部可下载的位置。上传即发布,可能被缓存或索引。
- **只读 SSH**:需要向第三方授予对生产机器的访问。两台机器上都还有**正在运行的
  窗口 2 实验**(box4 m1-a/m1-b、box1 m2b-a/m2b-b,见第 6 部分),开放访问对在跑实验
  有风险。

**我已按请求 §5 的要求准备好交付内容,但不会自行外发。** 待批准后可执行的方案:

```
window1-offline-verification.tar.gz
├── MANIFEST.tsv                     # 每文件: 原路径 → 包内路径, 字节, SHA256
├── runs/{n1-a,n1-b,pilot-p1,pilot-p2}/
│   ├── events.jsonl                 # 原始未修改
│   ├── manifest.json                # 实际生效的 resolved config
│   ├── state.json
│   ├── report/                      # 已有报告, 未重生成
│   └── candidates/*/trials/*.py     # materialized source, 按 trial_id 命名(见 §4)
├── stdout/                          # 上表 7 个已有 stdout/审计文件
└── readers/                         # 第 3 部分的实际执行版本 + 哈希
```

预估未压缩 ~110 MB;若省略 `sandboxes/` 与 `jobs/`(核验 P2′ 与 AA 不需要)约 ~60 MB。

**回复中不含任何密码、私钥或认证秘密**,符合请求 §5 的要求。

---

## 3. 实际执行过的 reader 与执行上下文

**核心结论:`gate2` 确实从 `/root/probe-clean` 执行,且当时执行的旧版 reader 仍在盘上,
未被修复版覆盖。** 版本证据完整,不是"只能找到当前文件"。

### 3.1 实际使用的脚本

| 脚本 | 实际执行位置 | 当时副本 SHA256 | 所属 commit | 与仓库当前版本 |
|---|---|---|---|---|
| `v41_p2prime.py` | box1 `/root/probe-clean/v41_p2prime.py`(11465 B)<br>box4 同名同哈希 | `4cf2124b40020e681d3422669b0aef7a7e2c324383b7e664abd22b78e8d8be3d` | **`63d097d`**(逐字节等同该 commit 的 blob) | **已被 `bd65af0` 取代**(18668 B, `23e7b92cbae2faee`);盘上旧版**未被覆盖** |
| `v41_gate2_readout.py` | box1 + box4 `/root/probe-clean/` | `7581b87cf5081070e279cb8cd13167f55c7bc266b5f09bb47a42cdae7e784f0a` | 与仓库 HEAD **相同** | 一致 |
| `v41_axis_uptake.py` | box1 + box4 `/root/probe-clean/` | `bcf0859f66984ac209ee3a3162fc1a922f676a4f90c099117f166a0bd6f53532` | 与仓库 HEAD **相同** | 一致(box1 另有同哈希的 `axis_uptake.py` 副本) |
| `rewrite_provenance.py` | box1 `/root/probe-clean/` | `54bd26a1bfc3d1943349a0cfbf97792f192a68e7804d1165132da84e9b22c673` | 与仓库 HEAD **相同** | 一致 |
| `gate2_all.sh` | box1 + box4 `/root/probe-clean/`(1798 B) | `60cf4698003fa42f2ff7e7aa9df1d968c575c5338d203dbdb30b2374dbdbb8e4` | **`6d18616`** | 仓库已到 `8a4a17f`(2305 B, `67467547fe18f678`);**盘上是旧版** |
| `v41_refusal_delta.py` | box1 + box4 `/root/probe-clean/` | `15b4b2c7a897934262bbea9561903c2e20df10fd6888b48ee2a8236478e21dd5` | 与仓库 HEAD **相同** | 一致 |
| `v41_n1_noise_floor.py` | box1 + box4 `/root/probe-clean/` | `ddcfb44827ce650046a69f4d0c7e62b34cdcd5507da37daeb7bb220c9bdaae9a` | 与仓库 HEAD **相同** | 一致。⚠ **它不是 13 对 AA 的生成器**(见第 4 部分) |
| `p2prime_audit.py` | box1 `/root/probe-clean/`(7279 B) | `9a822bf7e68279640a58df6e0ff2fb4eb16833618e04b4ab672ae83e512a24eb` | 与仓库 `scripts/probes/v41_p2prime_audit.py` **哈希相同**(仅文件名不同) | 一致 |
| `profile_coverage.py` | box1 `/root/probe-clean/`(3682 B) | `cf7b2b317f2e4bb15cc1bdb58f99f674d27ac7c9a1de45470c1952faf47b5284` | **`26b556d`** 的 blob,逐字节等同 | ⚠ 仓库工作区当前是 **10698 B / `456b3b48d7fc5bfd`**(被后续改写);**盘上是当时执行版** |
| `did_the_enqueued_point_win.py` | box1 `/root/probe-clean/`(26611 B) | `e835a652af758af895f5129691c3d0233b32411c68a99c5acdb6313cb28738eb` | 与仓库 HEAD **相同** | 一致 |

**未使用**:`v41_p2prime_audit.py` 这个**文件名**在机器上不存在(以 `p2prime_audit.py`
执行,内容哈希相同);`v41_profile_coverage.py` 这个**文件名**在机器上不存在
(以 `profile_coverage.py` 执行)。没有为补齐名单重跑任何脚本。

**哈希比对口径**:仓库侧文件是 CRLF(Windows checkout),机器侧是 LF。上表的
"一致 / 不一致" 判定用的是 **LF 规范化后**的哈希(`tr -d '\r' | sha256sum`),
与机器侧原始哈希直接可比。`gate2_all.sh` 与 `profile_coverage.py` 的差异是**真实内容差异**,
不是行尾差异。

### 3.2 执行命令、cwd、输入输出

`gate2_all.sh` 是唯一的包装器,其内容(1798 B 版,即实际执行版)确定了全部四步的
命令、解释器与 `PYTHONPATH`:

```bash
W=${OPOP_WORK:-/root/autodl-tmp/work/opop}
PY=${OPOP_PY:-/root/autodl-tmp/orch-venv/bin/python}
PROBE=${OPOP_PROBE_DIR:-/root/probe-clean}
# [1/4]  PYTHONPATH="$W/src" "$PY" "$PROBE/v41_gate2_readout.py" "$run"    # 未完成则拒答并跳过其余
# [2/4]  PYTHONPATH="$W/src" "$PY" "$PROBE/v41_p2prime.py" "$run"
# [3/4]  "$PY" "$PROBE/rewrite_provenance.py" "$run" | grep -E '...'
# [4/4]  "$PY" "$PROBE/v41_axis_uptake.py" "$run"
```

- **cwd**:`/root/probe-clean`(脚本以 `$PROBE/` 绝对路径调用,cwd 不影响结果)。
- **解释器**:`/root/autodl-tmp/orch-venv/bin/python`(Python 3.12.4)。
  ⚠ box4 上该 venv 也存在且可用,但 **n1 对的 run 本身**是用
  `/root/autodl-tmp/kernel-opt-venv/bin/python` 跑的(见 `n1-gpu-pinning.txt` 的 cmdline);
  **reader 与 run 用的不是同一个 venv**。reader 只读 JSON,不 import torch,故无影响。
- **直接 helper**:`PYTHONPATH=$W/src` 引入的是**生产源码树**
  `/root/autodl-tmp/work/opop/src`,当时 HEAD 见 §1.2;reader 从中 import
  `kernel_optimizer.*` 的数据模型。**helper 就是生产源码,不是独立副本。**
- **输入**:各 run 的绝对路径(§1.1 四个目录)。
- **输出**:stdout,**未重定向到文件**。因此 gate-2 的四步输出**没有独立存盘**,
  只存在于当时的终端会话与本仓库的分析文档中(`docs/window1-results.md`)。列为缺失
  (第 5 部分)。

### 3.3 副本是否可确认与当时执行版本一致

| 判定 | 依据 |
|---|---|
| **可确认一致** | `v41_p2prime.py`、`gate2_all.sh`、`profile_coverage.py`、`v41_gate2_readout.py`、`v41_axis_uptake.py`、`rewrite_provenance.py`、`v41_refusal_delta.py`、`v41_n1_noise_floor.py`、`p2prime_audit.py`、`did_the_enqueued_point_win.py` —— 全部仍在 `/root/probe-clean` 原位,mtime 早于或等于分析时点,且前三者的哈希**逐字节命中某个 git commit 的 blob**,可独立复核 |
| **版本证据缺失** | 无。这一项**没有**缺失 —— 请求担心的"拿其他工作树的同名文件替代"未发生 |

---

## 4. 13 对 AA 方法与表格

### 4.1 原始生成器:**内联代码,非保存脚本;已完整找回**

请求正确预判了这一点:**`v41_n1_noise_floor.py` 不是这组数字的生成器。**
生成器是一段**通过 SSH 以 heredoc 执行的内联 Python**,从未落盘为文件。
box4 的 `/root/.bash_history` 为 **0 行**,机器侧无法追回。

**但它在本会话的完整记录里,已逐字找回。以下是生成器原文(未改一字):**

```python
# 执行方式: ssh -p 37752 root@<box4> '/root/autodl-tmp/orch-venv/bin/python - <<"PY" ... PY'
import json,collections,statistics
base="/root/autodl-tmp/opop-workspace/opop-glm/runs-v4/%s/run-l3-43-20260915-071127/events.jsonl"
# Same (candidate source_sha, EXACT config) measured in BOTH arms => a true A/A on one config.
per={}
for a in ("n1-a","n1-b"):
    ev=[json.loads(l) for l in open(base%a,encoding="utf-8",errors="replace") if l.strip()]
    sha={}
    for e in ev:
        if e.get("type")=="CANDIDATE_REGISTERED":
            c=(e.get("payload") or {}).get("candidate") or {}
            if c.get("origin")=="seed" and c.get("source_sha"): sha[str(c["candidate_id"])]=c["source_sha"][:12]
    d=collections.defaultdict(list)
    for e in ev:
        if e.get("type")!="TRIAL_DONE": continue
        t=e["payload"]["trial"]
        if t.get("status")!="complete": continue
        cid=str(t.get("candidate_id"))
        if cid not in sha: continue
        m=(t.get("latency_ms") or {}).get("median")
        if m is None: continue
        key=(sha[cid], json.dumps((t.get("params") or {}).get("values") or {},sort_keys=True))
        d[key].append(float(m))
    per[a]=d
shared=sorted(set(per["n1-a"])&set(per["n1-b"]))
print("configs measured in BOTH arms (same seed sha AND byte-identical knob values): %d"%len(shared))
diffs=[]
for k in shared:
    A=min(per["n1-a"][k]); B=min(per["n1-b"][k])
    dv=(B-A)/A*100; diffs.append(abs(dv))
    if len(diffs)<=14:
        print("  %s  A=%.4f (n=%d)  B=%.4f (n=%d)  B-A=%+6.2f%%"%(k[0],A,len(per["n1-a"][k]),B,len(per["n1-b"][k]),dv))
if diffs:
    diffs.sort()
    print("  TRUE same-config A/A |B-A|: n=%d  min %.2f%%  median %.2f%%  p90 %.2f%%  max %.2f%%"
          %(len(diffs),diffs[0],statistics.median(diffs),diffs[int(0.9*(len(diffs)-1))],diffs[-1]))
```

**注意生成器自己的注释("byte-identical knob values"、"TRUE same-config A/A")
就是错误标签的来源** —— 它只比对了 knob 值,却自称 byte-identical。见本文第 0 节。

### 4.2 原始输出(逐字,未重跑)

```
configs measured in BOTH arms (same seed sha AND byte-identical knob values): 13

  20eb42e310df  A=9.2539 (n=1)  B=9.7469 (n=2)  B-A= +5.33%
  49d10c6bda13  A=7.5945 (n=1)  B=7.5136 (n=1)  B-A= -1.06%
  49d10c6bda13  A=8.0584 (n=1)  B=7.9980 (n=1)  B-A= -0.75%
  49d10c6bda13  A=8.0942 (n=1)  B=8.0108 (n=1)  B-A= -1.03%
  49d10c6bda13  A=7.6989 (n=2)  B=7.4921 (n=2)  B-A= -2.69%
  49d10c6bda13  A=7.2233 (n=1)  B=7.2228 (n=1)  B-A= -0.01%
  49d10c6bda13  A=7.3272 (n=1)  B=7.1440 (n=1)  B-A= -2.50%
  49d10c6bda13  A=7.2192 (n=2)  B=7.2167 (n=2)  B-A= -0.03%
  49d10c6bda13  A=88.5059 (n=1)  B=87.7722 (n=1)  B-A= -0.83%
  49d10c6bda13  A=25.5360 (n=1)  B=25.0296 (n=1)  B-A= -1.98%
  6012d8732b9f  A=5.2362 (n=1)  B=5.1144 (n=1)  B-A= -2.33%
  6012d8732b9f  A=3.7663 (n=1)  B=3.7622 (n=1)  B-A= -0.11%
  6012d8732b9f  A=3.9373 (n=1)  B=3.9398 (n=1)  B-A= +0.07%

  TRUE same-config A/A |B-A|: n=13  min 0.01%  median 1.03%  p90 2.50%  max 5.33%
```

**单位与指标**:百分比,`(B−A)/A × 100`,其中 A、B 是各臂在该配置上
`TRIAL_DONE.trial.latency_ms.median`(毫秒)取 **min**。汇总统计用 `|B−A|` 绝对值。

### 4.3 原始生成器的精确口径

按请求逐项列明,**以上面的原始代码为准**:

| 口径项 | 原始生成器的实际做法 |
|---|---|
| 聚合层级 | 每个 `(seed_sha[:12], 声明 knob 字典)` 键收集该臂**全部** complete trial 的 `latency_ms.median`,再取 **min** |
| min / median / min-of-medians | **min-of-medians**:每个 trial 内部取 `median`,同配置多个 trial 之间取 `min` |
| 重复记录 | **不去重**,全部进入列表(`d[key].append`),故 `n=2` 表示两条记录 |
| reused 样本 | **完全未区分**。`payload.reused_measurement` 从未被读取 ⇒ 13 对里有 reused 记录参与(补充核实:第 1、5、8、13 对含 reused,见 §4.4) |
| 百分比分母 | **A 臂(n1-a)的值**,`(B−A)/A×100` ⇒ 非对称,换臂会得到不同数字 |
| p90 算法 | `diffs.sort(); diffs[int(0.9*(len(diffs)-1))]` = **下取整索引,无插值**。n=13 ⇒ `int(0.9*12)=int(10.8)=10` ⇒ 第 11 小的值 |
| precision 是否分组 | **未分组**。`COMPUTE_DTYPE` 只作为 knob 字典的一部分参与配对键,不单独分层;未知值不出现(每条都有) |
| matching key 作用域 | `source_sha[:12]` 取自 `CANDIDATE_REGISTERED` 且 `origin=="seed"` ⇒ **只覆盖种子候选,不含 rewrite**;`candidate_id` 本身**跨 run 不可比**(两臂同一种子有不同 candidate_id),故用 sha 而非 id 作键。sha 是**注册源**的哈希,**不是参数化后或 materialized 的哈希**(这正是第 0 节的缺陷所在) |
| 计时口径 | `latency_ms.median` 直接取自事件,未重算;未做 warmup/采样数核对 |

### 4.4 逐对明细表(新分析补充的字段以斜体标注)

原始生成器只印了 `seed sha / A / B / n / 差值`。以下表把请求要求的其余字段补上。
**补充字段来自同一批未修改的 `events.jsonl` 与 `candidates/*/trials/*.py`,
是【新分析】,不是旧结果复现;原始 13 个数值未改动。**

| # | seed sha | *A trial_id* | *B trial_id* | *A cand_id* | *B cand_id* | A min ms | B min ms | nA | nB | *reusedA* | *reusedB* | signed % | abs % | *A artifact sha* | *B artifact sha* | *代码同?* |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `20eb42e310df` | `tr-78a78847` | `tr-a8d76719` | (见注) | (见注) | 9.2539 | 9.7469 | 1 | 2 | *1* | *2* | +5.33% | 5.33% | `8352ac4871` | `b8d2406da1` | **DIFFERENT** |
| 2 | `49d10c6bda13` | `tr-9499a256` | `tr-434f075a` | `cand-5cce69e6` | `cand-9832ce62` | 7.5945 | 7.5136 | 1 | 1 | *0* | *0* | −1.06% | 1.06% | `e0ddf67846` | `16f30ea5c2` | **DIFFERENT** |
| 3 | `49d10c6bda13` | `tr-e7ae09b5` | `tr-1f841a9f` | 同上 | 同上 | 8.0584 | 7.9980 | 1 | 1 | *0* | *0* | −0.75% | 0.75% | `3e599f4c4f` | `926ee31671` | **DIFFERENT** |
| 4 | `49d10c6bda13` | `tr-06915b2c` | `tr-187e9adc` | 同上 | 同上 | 8.0942 | 8.0108 | 1 | 1 | *0* | *0* | −1.03% | 1.03% | `e543fa1973` | `803f9d24bd` | **DIFFERENT** |
| 5 | `49d10c6bda13` | `tr-f79175e9` | `tr-57ab8e6f` | 同上 | 同上 | 7.6989 | 7.4921 | 2 | 2 | *2* | *2* | −2.69% | 2.69% | `37e74164cb` | `a6bd971702` | **DIFFERENT** |
| 6 | `49d10c6bda13` | `tr-bf5a2709` | `tr-d0b42767` | 同上 | 同上 | 7.2233 | 7.2228 | 1 | 1 | *0* | *0* | −0.01% | 0.01% | `54a4902248` | `655cd27832` | **DIFFERENT** |
| 7 | `49d10c6bda13` | `tr-9d39a31c` | `tr-d893998e` | 同上 | 同上 | 7.3272 | 7.1440 | 1 | 1 | *0* | *0* | −2.50% | 2.50% | `0201629391` | `5717d8eac4` | **DIFFERENT** |
| 8 | `49d10c6bda13` | `tr-651a665d` | `tr-d4e089b7` | 同上 | 同上 | 7.2192 | 7.2167 | 2 | 2 | *2* | *2* | −0.03% | 0.03% | `51cb8aa594` | `ba471706f5` | **DIFFERENT** |
| 9 | `49d10c6bda13` | `tr-25381ece` | `tr-45a0e3b4` | 同上 | 同上 | 88.5059 | 87.7722 | 1 | 1 | *0* | *0* | −0.83% | 0.83% | `4adfa3ce4a` | `59e239176c` | **DIFFERENT** |
| 10 | `49d10c6bda13` | `tr-8eb895d6` | `tr-c5215bc8` | 同上 | 同上 | 25.5360 | 25.0296 | 1 | 1 | *0* | *0* | −1.98% | 1.98% | `db8ee7f1d6` | `f61327da39` | **DIFFERENT** |
| 11 | `6012d8732b9f` | `tr-9a4cdfb8` | `tr-2c050373` | (见注) | (见注) | 5.2362 | 5.1144 | 1 | 1 | *0* | *0* | −2.33% | 2.33% | `40fd34e36a` | `bc036a8e89` | **DIFFERENT** |
| 12 | `6012d8732b9f` | `tr-72964ad9` | `tr-f4b2f865` | 同上 | 同上 | 3.7663 | 3.7622 | 1 | 1 | *0* | *0* | −0.11% | 0.11% | `4a9c43140d` | `9cba6206b5` | **DIFFERENT** |
| 13 | `6012d8732b9f` | `tr-c1381e8f` | `tr-0b8c6dac` | 同上 | 同上 | 3.9373 | 3.9398 | 1 | 1 | *1* | *1* | +0.07% | 0.07% | `d0c1a73ffd` | `96961d326e` | **DIFFERENT** |

**其余请求字段的状态:**

- **完整参数**:每对的完整 knob 字典就是配对键本身,存在于
  `TRIAL_DONE.trial.params.values`,可按上表的 `trial_id` 精确定位。**未在此展开**
  (每条 15–20 个 knob),随压缩包交付。
- **原始 latency 值**:`TRIAL_DONE.trial.latency_ms` 含 `median` 与 `mean`
  (**注意:无 `_ms` 后缀**)。表中给的是 `median`。
- **样本计数**:`nA`/`nB` 是**记录条数**,不是每条内部的采样数。每条内部的采样数由
  `evaluation.perf_trials` 决定,记在 resolved config 里,**事件本身不逐条记录采样数**。
- **backend**:`CANDIDATE_REGISTERED.candidate.backend`,三个种子**全部 `triton`**。
- **声明的 precision**:`params.values.COMPUTE_DTYPE`(如第 8 对两臂都是 `fp16`)。
- **实际执行 precision**:**未知,标为未知。** 事件里没有实测 dtype 证据;
  `profile` 不含实际 dtype 字段。按请求要求不为此重跑 GPU。
- **reused 复用来源**:`reused_measurement` 是布尔,**不记录复用自哪条记录** ⇒
  复用来源**未知**。

⚠ 表中 `cand_id` 只逐一核实了 `49d10c6bda13`(n1-a `cand-5cce69e6` / n1-b `cand-9832ce62`);
另两个种子的 candidate_id 未逐对提取,标为"见注"。可由 `trial_id` 在
`events.jsonl` 中一次查询补全,随压缩包交付。

### 4.5 新分析:artifact 级配对(必须与旧结果分开读)

**标注:新分析,非旧结果复现。** 探针 `scripts/probes/v41_aa_pair_artifacts.py`。
结论已在第 0 节给出:声明键 13 对,**artifact 键 0 对**;两臂参数化体差 2–22 行。

同代码重复测量(`scripts/probes/v41_same_code_repeats.py`):n1-a 549 个不同 artifact
中 15 个被计时 >1 次,n1-b 630 个中 18 个;其中**完全不含 reused 记录的 fresh 重复只有
3 例**(全在 n1-a),spread 0.04% / 0.06% / 0.07%。

---

## 5. 缺失项

| 缺失项 | 查找依据 | 因此保持"未知"的结论 |
|---|---|---|
| **四个 run 的启动时 dirty 工作区状态 / patch** | `RUN_CREATED.payload` 只有 `run_id`;`manifest.json` 只有 `task`/`created`/`config`;两台机器当前 `git status --porcelain` 为空但那是**当前**状态 | 无法排除启动时存在未提交改动。所有"该 run 跑的就是 commit X 的代码"的陈述须降级为"HEAD 在 commit X,工作区洁净性未知" |
| **gate-2 四步的原始 stdout** | `gate2_all.sh` 输出到终端,未重定向;`/root/probe-clean` 下只有 `sweep_out.txt`(9-12,与本次无关);`/root/*.log` 是编排器 stdout 不是 reader 输出 | gate-2 的具体打印内容只能由"当时执行版脚本 + 未修改 events.jsonl"重跑得到;**本次未重跑**。若日后重跑须标注为新执行 |
| **13 对 AA 生成器的机器侧痕迹** | box4 `/root/.bash_history` = **0 行**;`/root/probe-clean` 无对应文件;`v41_n1_noise_floor.py` 内容不同 | 无。**生成器已从会话记录逐字找回**(§4.1),这一项不再是缺失;仅"机器侧无痕迹"这一事实需记录 |
| **每条 trial 的实际执行 precision** | `TRIAL_DONE.trial.profile` 无 dtype 字段;事件全文无 `torch_version`/`triton_version`/`toolchain`/`env_probe` | 精度只有**声明值**,实际值未知。任何按精度分层的结论须声明这一点 |
| **reused 记录的复用来源** | `payload.reused_measurement` 是布尔,无来源指针 | 无法回溯某条 reused 记录复制自哪次测量 ⇒ 无法把它折算成独立样本 |
| **worker / collector 版本号** | 事件里**完全没有** `worker_version` / `collector_version` 键 | 版本只能由"启动 commit + venv 路径"间接推断,非直接记录 |
| **`wall_ms` / `cpu_issue_ms` / `overhead_gpu_ms`** | 字段 present 但 **0% 非空**(已在上一轮交付物 4 中测过) | 依赖这三个字段的任何目标现在不可离线重评分 |
| **两对 run 的跨对可并表性** | 启动 commit 不同(`f159555` vs `1fb9c7d`)、CPU 不同(8352V vs 8358P)、任务不同(pilot-p2 是 L3:21) | 四个 run **不能并成一张表**;只有同对两臂可配对比较 |
| **13 对里 reused 记录的影响幅度** | 已标出哪几对含 reused(§4.4),但未做"剔除 reused 后重算"的敏感性分析 | 未做。若需要,是一次新分析,须另行标注 |

**当时记录 vs 当前文件的区分**:上表"查找依据"列里,`events.jsonl` / `manifest.json` /
`/root/*.log` / `/root/probe-clean/*` 全部是**当时落盘、此后未改**的文件(mtime 与哈希
已在第 1、3 部分给出);`git status` / 仓库工作区是**当前**状态,已明确标注。

---

## 6. 操作确认

| 确认项 | 状态 | 证据 |
|---|---|---|
| 未修改生产 source / config / prompts / scorer | ✅ | 两台机器 `git status --porcelain` 为空,HEAD = `bd65af0`;`git diff --name-only bb15839..HEAD \| grep -E "^(src/\|configs/)"` 返回 **NONE**(窗口 2 启动 commit 至今生产路径零改动) |
| 未重启在跑任务 | ✅ | 窗口 2 四个 run 仍在原进程:box4 `m1-a`(215 trial)/`m1-b`(240),box1 `m2b-a`(217)/`m2b-b`(217),读数时点 2026-09-16 01:2x |
| 未启动 GPU 作业 | ✅ | 本次全部操作是 `ls` / `cat` / `sha256sum` / `du` / 纯 JSON 与文本读取的 Python;没有 import torch,没有 CUDA 上下文 |
| 未新增分析(除明确标注者) | ⚠ **部分例外,已标注** | §4.5 与第 0 节是**新分析**(`v41_aa_pair_artifacts.py`、`v41_same_code_repeats.py`),因为它们回答的正是请求 §3 "注明这些数字的单位及对应指标"与 §4 "不要默认这三类 source 的哈希 identity 相等" —— 核实过程本身发现了标签错误。**旧的 13 个数值一字未改**,新结论单独成节并标为"新分析" |
| 未修改原始日志 | ✅ | 四个 `events.jsonl` 的 SHA256 已在 §1.3 给出;本次只以只读方式打开 |
| 未挑选性保留 / cherry-pick 样本 | ✅ | §1.1 列出了每个父目录下的**全部**子目录并给出选择依据(唯一含 `RUN_FINISHED` 者);§4.4 列出全部 13 对,含对结论不利的第 1 对(+5.33%) |
| 未编造缺失事实 | ✅ | 第 5 部分逐项列出缺失,dirty 状态、实际 precision、复用来源、worker 版本一律写"未知" |
| 旧报告与预注册定义保持不变 | ✅ | `docs/window1-results.md`、`docs/audit-p2prime-reader-defects.md` 未因本次核验改写;第 0 节的更正**新增在本文**,并指明上一轮 prompt 中哪句作废 |

### 6.1 一处必须主动披露的情况

核验期间我发现**本仓库工作区里有我未提交、也不是我写的改动**:

```
 M docs/research-v5-unified-objective-framework.md      (+81 −?)
 M scripts/probes/v41_p2prime.py                        (13731 B,少于我提交的 18668 B)
 M scripts/probes/v41_profile_coverage.py               (10698 B,多于提交的 3682 B)
?? scripts/probes/v41_p2prime_history.py
?? scripts/probes/v41_same_config_aa.py
?? tests/test_v41_p2prime_events.py
?? tests/test_v41_profile_coverage.py
?? tests/test_v41_same_config_aa.py
```

mtime 全部在 2026-09-16 00:05–00:35,而我最后一次提交是 09-15 23:24。这些是**外部
agent 直接交付进工作区的代码**。已核实:

- 全套测试 **1306 passed / 1 failed / 11 skipped**;唯一失败是
  `test_g10_backend_ceiling` 的 `import torch`(host 无 torch,既有已知项,与本次无关)。
- 改写后的 `v41_p2prime.py` **保留了符号修复**(`return 1.0 - n_med / f_med`,
  `legacy_sign` 仍可再生 `(n−f)/n`),并**新增第五种定义 `discovery_cutoff`**。
- `v41_same_config_aa.py` 自带的限定声明写着"Original 13 pairs are not reproducible
  locally and remain unverified" —— **这一条现在可以撤销:生成器与原始输出都已找回
  (§4.1、§4.2)。**

**这些改动不影响本答复中的任何数字**:本答复的全部读数来自机器上未修改的
`events.jsonl` 与 `/root/probe-clean` 的当时执行版脚本,不经过工作区的新代码。

---

## 7. 待用户批准的两项

1. **§5 的交付**:压缩包上传(需外部可下载位置)或开放只读 SSH(需向第三方授予生产机器
   访问,且两台机器上有正在运行的实验)。**未执行。**
2. **是否把第 0 节的 N1 更正推给外部 agent** 作为上一轮 prompt 的勘误 —— 它作废的是
   我上一轮刚要求对方采用的措辞。
