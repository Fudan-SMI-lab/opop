# 移交外部 agent 的 prompt:window1 数据交付 + N1 更正(第三轮)

> 用法:把 `=== PROMPT 开始 ===` 与 `=== PROMPT 结束 ===` 之间的全文,连同两个压缩包
> 与附件一起提交。
>
> **压缩包(交付物本体)**
> | 文件 | 字节 | SHA256 |
> |---|---|---|
> | `window1-verification-box4.tar.gz` | 1248201 | `7b4d7b8f7a55099543d773653dfeb6456bc62f408105676b8662b17b8f920787` |
> | `window1-verification-box1.tar.gz` | 1532135 | `03f728333b64e3ccc80cd81f687e1ab110e0ccd83609588a32d36763d320f669` |
>
> **附件(按重要性排序)**
> 1. `docs/reply-window1-offline-data-request.md` —— 六部分答复正文(本 prompt 的证据来源)
> 2. `scripts/probes/v41_aa_pair_artifacts.py` —— 证明 0/13 对同代码的探针(新分析)
> 3. `scripts/probes/v41_same_code_repeats.py` —— 真同代码 A/A 的探针(新分析)
> 4. `docs/prompt-v5-revision-handoff-round2.md` —— 上一轮 prompt(**其中两处被本轮作废**)
> 5. `docs/prompt-window1-offline-data-request.md` —— 你的数据请求(被答复对象)

---

=== PROMPT 开始 ===

你的数据请求(`docs/prompt-window1-offline-data-request.md`)已按要求的六部分答复完毕,
见附件 1。两个压缩包同时交付,离线可核验。

**本 prompt 分五部分:①一条必须先读的更正(它作废我上一轮要求你采用的措辞);
②你担心的两件事都没发生;③交付内容与如何核验;④仍然缺失的项;
⑤两台机器正在跑实验 —— 对你的硬约束。**

**判读纪律(与前两轮相同)**:只信盘上 `events.jsonl`;终值只在 `RUN_FINISHED.summary.best`;
`latency_ms` 的键是 `median`/`mean`(**无 `_ms` 后缀**);不跨机器/跨任务并表;
形如 `*_DECIDED` 的过去分词事件只表示该步跑完,判决在 payload 里。
**未核实的一律标"未核验",不生成看似正确的新数字。**

---

## 第一部分:一条更正 —— **我上一轮要你采用的 N1 措辞是错的**

你请求 §3 要我"注明这组数字的单位及对应指标"。照做的过程中我发现**这组数字的名字错了**。
**这是同一条 N1 结论的第四次更正,错在我这边。**

我上一轮(`prompt-v5-revision-handoff-round2.md` §1.2)让你把 N1 写成
"**同配置**测量噪声 median 1.03% / p90 2.50% / max 5.33%(13 个 byte-identical 配置)"。
实测:

```
executed code byte-identical in 0 / 13 pairs
```

**13 对里没有一对跑了逐位相同的代码。**

### 为什么错

原生成器的配对键是 `(注册种子 source_sha, 声明的 knob 字典)`。它固定了**种子**与**旋钮**,
**没有固定参数化后的程序体** —— parameterizer 是一次 LLM 调用,两臂把同一个种子变成了
两个不同的程序:

| 种子 | 两臂参数化体差异行数 | A / B 行数 |
|---|---|---|
| `20eb42e310df` | **22 行** | 224 / 228 |
| `49d10c6bda13` | **18 行** | 221 / 229 |
| `6012d8732b9f` | 2 行 | 185 / 185 |

差异是实质的,不是注释:`49d10c6bda13` 上 n1-b 的 parameterizer 把精度 constexpr
(`_DT` / `_MODE`)从模块级求值改成**每次 forward 重新推导**,并改了 `tl.dot` 的
`input_precision` 路径。

**而这件事按 sha 查不出来。** 两臂的 `CANDIDATE_REGISTERED.source_sha` **与**
`SPACE_PUBLISHED.source_sha` **都是** `49d10c6bda13`。参数化后的程序体**没有自己的
journal 哈希** ⇒ **任何按 sha 配对的检查在结构上不可能发现这件事**。
唯一能发现的是 `candidates/<cid>/trials/<trial_id>.py` 的字节哈希(该层 100% 覆盖,
已随包交付)。

### 正确的措辞(请按这个改,作废我上一轮的说法)

> **13 对"同种子 + 同声明旋钮"跨臂配对的延迟差 `|B−A|`:
> n=13,min 0.01%,median 1.03%,p90 2.50%(下取整索引、无插值),max 5.33%。
> 单位是百分比,`(B−A)/A×100`,A/B 为各臂该配置下 `latency_ms.median` 的 min(min-of-medians)。
> 它混合了计时噪声与两臂参数化差异,是同代码测量噪声的【上界】,不是对它的测量。**

### 真正的同代码 A/A:几乎不存在

同一个 materialized artifact 被计时多次的情况(`v41_same_code_repeats.py`):

| 集合 | n | min | median | p90 | max |
|---|---|---|---|---|---|
| **fresh vs fresh(无 reused 记录)** | **3** | 0.04% | **0.06%** | 0.06% | **0.07%** |
| 含 ≥1 条 reused 记录 | 30 | 0.00% | 0.03% | 0.55% | 8.37% |

**n=3 太小,不能作为承重阈值。**

### 后果 —— 请逐条落进修订版

1. **"每个目标 × 每个场地都要有自己的 A/A 标定、4% 不能默认外推"这个要求反而更强。**
   但**标定的做法必须改**:标定必须**显式重放同一个 materialized artifact**,
   **不能事后从搜索日志里捞配对** —— 捞出来的配对只固定声明,不固定代码。
   请把这一条写成 v5 的可执行纪律。
2. **τ=4% 落在这个【上界】的 p90 与 max 之间** ⇒ "保守但不荒谬"仍成立,但**依据变弱**。
   不要写成"已由同配置噪声证成"。
3. **六个投递点的 4 输 2 平判决不变**(4% 与 5.33% 同判决,上一轮已核实)。
4. **P3 仍降级,理由仍是"搜索路径分歧是 12h 单 run 的固有方差"**,这一条不变。
5. **新增一条通用纪律,请写进方法学章节**:
   > 报任何"同配置 / 同代码"数字前,必须先说明配对键固定了哪一层 identity。
   > 本项目有三层且互不等价:注册源 sha → 空间 sha(常与前者相同)→
   > **materialized artifact 的字节哈希**(唯一能证明"跑的是同一份代码")。
   > 前两层相同而第三层不同,在本项目里是**常态而非例外**,因为 parameterizer 是 LLM 调用。

### 关于你自己那句限定

`v41_same_config_aa.py` 里你写了 "Original 13 pairs are not reproducible locally and
remain unverified"。**这一条现在可以撤销**:原始生成器(内联 SSH heredoc)与它的完整
输出都已逐字找回,见附件 1 的 §4.1 / §4.2。**但请保留你那个脚本本身的限定语
(match / artifact / precision / context / inference 五条),它们是对的**,
尤其 `artifact` 那条 —— 你写的"registered source_sha 不是 materialized 证明"
正是上面这条缺陷,**你先写对了,我当时没照做。**

---

## 第二部分:你担心的两件事都没发生

| 你的担心(请求 §2 / §3) | 核实结果 |
|---|---|
| gate2 可能不是从 `/root/probe-clean` 执行,拿了别的工作树同名文件替代 | **确实从 `/root/probe-clean` 执行,且当时那版 reader 未被修复版覆盖。** `v41_p2prime.py`(11465 B)哈希 `4cf2124b…` **逐字节等同 commit `63d097d` 的 blob**;`gate2_all.sh` → `6d18616`;`profile_coverage.py` → `26b556d`。**版本证据完整,没有一项是"只能找到当前文件"** |
| 13 对 AA 的原始生成器可能已丢,不能拿现有 N1 reader 冒充 | **你判断正确**:`v41_n1_noise_floor.py` **不是**生成器;生成器是一段 SSH heredoc 内联代码,**从未落盘**,box4 的 `/root/.bash_history` 为 **0 行**。**但它在会话记录里已逐字找回**,连同原始 13 行输出,见附件 1 §4.1/§4.2。**因此这一项不是缺失** |

另外三项你要求核实的,结论如下:

- **时间戳子目录**:四个父目录下的**全部**子目录已列出,选择依据是**唯一含 `RUN_FINISHED`
  的那个**(其余是 `v41-smoke-*` / `evalcheck-*` 残留)。**没有挑好看的。**
- ⚠ **n1-a 与 n1-b 的 `run_id` 完全相同**(都是 `run-l3-43-20260915-071127`),
  `RUN_CREATED.payload` 只有 `run_id` 一个字段 ⇒ **臂身份只能由父目录名区分。**
  你的核验脚本请按父目录取臂,不要按 run_id。
- ⚠ **两对 run 不在同一 commit 启动**:pilot 对 `f159555`,n1 对 `1fb9c7d`
  (差一个"种子被签名去重丢弃时拒绝启动"的硬门)。**同对内两臂 commit 相同**,
  所以配对比较成立;**跨对绝不可并表**。加上 CPU 不同(8352V vs 8358P)与任务不同
  (pilot-p2 是 L3:21),这就是分两个包交付的原因。
- **你请求 §4 "不要默认三类 source 的哈希 identity 相等"** —— 照做并实测:
  `candidate.source_sha == space.source_sha` 在两者都存在时**全部相等**;
  但 `candidates/<cid>/source.py` 与 journal 的 `source_sha` **12 个里只有 5 个匹配**
  —— 注册源实际在 `sandboxes/generator-*/candidates/*.py` 与
  `sandboxes/rewriter-*/rewrites/*.py`,**这两个目录已按此发现刻意加入包内**,
  否则你无法按 sha 定位注册源。

---

## 第三部分:交付内容与如何核验

两个包同构:

```
./MANIFEST.tsv                                       # archive_path \t bytes \t sha256, 逐文件
./runs/<arm>/events.jsonl                            # 原始未修改
./runs/<arm>/manifest.json                           # 实际生效的 resolved config
./runs/<arm>/state.json
./runs/<arm>/report/                                 # 已有报告, 未重生成
./runs/<arm>/candidates/<cid>/source.py              # 参数化后的源(与 journal sha 常不符)
./runs/<arm>/candidates/<cid>/trials/<trial_id>.py   # ★ materialized source, 100% 覆盖
./runs/<arm>/sandboxes/generator-*/candidates/*.py   # ★ 注册种子源的真实位置
./runs/<arm>/sandboxes/rewriter-*/rewrites/*.py      # ★ 注册改写源的真实位置
./runs/<arm>/uw_probes/                              # 仅 pilot 对有
./readers/                                           # 当时实际执行的 reader 原位副本
./stdout/                                            # 已有 stdout + 启动前配置审计 + 卡分离证据
```

**建议的核验顺序**(全部纯 CPU、纯 Python):

1. 先按 `MANIFEST.tsv` 逐文件核 SHA256,确认传输完整。
2. 用 `./readers/v41_p2prime.py`(**当时执行版**,哈希 `4cf2124b…`)对
   `./runs/<arm>/events.jsonl` 重跑,与已发表数字对照。注意:窗口 1 早于 `axis_f_value`,
   所以它走 `UW_PROBE_BATCH.walls` → `SCAN_BLOCK_ADMITTED` 的回退路径;
   **不复刻这条回退会读出 n=0**(我第一版审计就栽在这里)。
3. 核 13 对 AA:用 §4.1 的生成器原文重跑得到 13 对;再用附件 2
   `v41_aa_pair_artifacts.py` 按 `trials/<trial_id>.py` 的字节哈希重配对,应得 **0 对**。
4. `./stdout/*-preaudit.txt` 是启动前的配置对等审计(125 键比对,3 处差异全是隔离路径,
   印 `COMPARABLE`);`*-gpu-pinning.txt` 是**运行期**从 `/proc` 读 CVD + 120s 采样的
   卡分离证据(印 `SEPARATED`)。**卡分离只能在运行期查,盘上不记录设备** ——
   这两个文件是唯一证据,请不要用事后的 `nvidia-smi` 快照替代。

**已排除**(体积主要在这里,核验 P2′ 与 AA 不需要):`sandboxes/` 的 agent 会话记录、
`jobs/` 的 worker 作业 JSON、`artifacts/`(空)。**需要就提清单,补包是只读拷贝。**

---

## 第四部分:仍然缺失的项 —— 请保持"未知",不要补做

| 缺失项 | 依据 | 因此保持"未知"的结论 |
|---|---|---|
| **四个 run 启动时的 dirty 工作区状态 / patch** | `RUN_CREATED.payload` 只有 `run_id`;`manifest.json` 只有 `task`/`created`/`config`;当时没留 patch。两台机器**当前**工作区洁净,但那是当前状态 | 所有"该 run 跑的就是 commit X 的代码"须降级为"**HEAD 在 commit X,工作区洁净性未知**" |
| **gate-2 四步的原始 stdout** | `gate2_all.sh` 输出到终端,未重定向;`/root/*.log` 是编排器 stdout 不是 reader 输出 | gate-2 的打印内容只能由"当时执行版脚本 + 未修改 events.jsonl"重跑得到。**若你重跑,须标注为新执行,不能称"旧结果复现"** |
| **每条 trial 的实际执行 precision** | `profile` 无 dtype 字段;事件全文无 `torch_version` / `triton_version` / `toolchain` / `env_probe` | 精度只有**声明值**。任何按精度分层的结论必须声明这一点。**不要为补这个证据安排 GPU 测量**(见第五部分) |
| **reused 记录的复用来源** | `payload.reused_measurement` 是布尔,无来源指针 | 无法把 reused 记录折算成独立样本 |
| **worker / collector 版本号** | 事件里**完全没有**这两个键 | 只能由"启动 commit + venv 路径"间接推断,非直接记录 |
| **`wall_ms` / `cpu_issue_ms` / `overhead_gpu_ms`** | 字段 present 但 **0% 非空** | 依赖这三个字段的目标**现在不可离线重评分** |
| **13 对里 reused 记录的影响幅度** | 已标出哪几对含 reused(附件 1 §4.4 第 1/5/8/13 对),但未做剔除后的敏感性分析 | **未做。** 若你要做,那是一次新分析,须另行标注 |
| **真同代码 A/A 的可用样本量** | 只有 3 例 fresh 重复 | **n 不足,不能定阈值。** 正确处理是把"显式重放同一 artifact"写成 v5 的标定动作,**不是现在去测** |

---

## 第五部分:两台 Linux 服务器正在跑实验 —— 对你的硬约束

**box4 与 box1 目前都在运行窗口 2 的实验对,预计跑到 2026-09-16 11:30 前后:**

| 机器 | 在跑 run | pid | 卡 |
|---|---|---|---|
| box4 | `m1-a`(off 臂) | 566844 | GPU1 |
| box4 | `m1-b`(active 臂) | 566848 | GPU0 |
| box1 | `m2b-a`(off 臂) | 415002 | GPU0 |
| box1 | `m2b-b`(active 臂) | 415006 | GPU1 |

**实验期间不得进行任何涉及 GPU 的测试或验证。** 请遵守以下四条:

1. **不要提出需要跑 GPU 才能回答的核验项或建议。** 若某个结论缺 GPU 证据
   (实际执行 precision、同代码 A/A 的 n、面板的实际 GPU 成本),
   **正确处理是保持"未知 / n 不足"并列为"待测量"**,不是安排一次测量。
   任何 GPU 测量都要等窗口 2 跑完后另行排期。
2. **不要建议修改生产 `source` / `config` / `prompts` / `scorer`,也不要建议重启。**
   窗口 2 的自变量之一就是共享 prompts,改动会作废在跑的对照。
   现状:两台机器都在 `bd65af0`,
   `git diff --name-only bb15839..HEAD | grep -E "^(src/|configs/)"` 返回 **NONE**。
3. **本次交付的全部材料都是离线可核验的** —— 包内是 `events.jsonl` + resolved config +
   materialized source + 当时执行版 reader,**用 CPU 与纯 Python 即可完成全部核验**,
   不需要 GPU、不需要 torch、不需要访问这两台机器。
4. 需要额外文件就**提清单**,补包是只读拷贝,同样不涉及 GPU。

打包本身已按此约束执行:`nice -n 19 ionice -c3`、对 run 目录只读、未 import torch、
未起 CUDA 上下文。

---

## 交付要求

1. **核验报告**:逐项说明你能确认什么、不能确认什么,不能确认的给出你查了哪些证据。
2. **v5 修订**:按第一部分的五条后果改 N1 与阈值的全部表述;
   **我上一轮关于"同配置测量噪声"的措辞整体作废**,请显式标注这次替换,不要默默改掉。
   其余部分沿用上一轮 prompt 第五部分的六条要求(H2 拆三层、§12.1 处理矩阵、
   离线重评分的资格条件与精度分层、同机配对约束、保留保守措辞与"明确非目标"章节)。
3. **若你认为本 prompt 的某条与证据冲突,指出冲突并给你的依据**,不要沉默照做。
   前两轮你各纠正了我一批结论 —— **这一轮的第一部分正是你 §6 那条指控一路推下去的结果,
   你的 `artifact` 限定语先写对了,我当时没照做。** 继续这样做。
4. **不写"实验显示"而不给数字**;每处依据实测的改动在正文标出数字与来源(哪个 run、n 多少)。

=== PROMPT 结束 ===
