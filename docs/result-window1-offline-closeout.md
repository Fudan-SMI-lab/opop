# window1 离线核验收口简报

> 任务:`docs/prompt-window1-offline-closeout.md`。日期:2026-09-16。
> 输入:`v4/.deliver/` 的两个压缩包 + 两份 MANIFEST。
> 全部核验**纯 CPU、纯 Python**:未碰 GPU、未 import torch、未初始化 CUDA、
> 未访问服务器、未重启实验、未装依赖、未 git commit、未改生产 source/config/prompt/scorer。
> **未修改 `docs/research-v5-unified-objective-framework.md` 正文**;§7 只给替换建议。
>
> 关联表格在 `docs/closeout-tables/`(文件名与本文各节的编号一致)。

---

## 0. 逐项结论表

| # | 核验项 | 结论 | 依据 |
|---|---|---|---|
| 1.1 | 包级哈希 | **已复现** | 两包 SHA256 与 prompt 中预期值逐位相同 |
| 1.2 | 逐文件哈希 | **已复现** | **4167/4167 文件全部匹配**,0 缺失 0 不符 |
| 1.3 | 独立 MANIFEST 与包内副本一致 | **已复现** | 两侧哈希相同 |
| 1.4 | 四个 run 身份映射 | **已复现** | 见 §1;n1a/b 只能以目录身份区分,已确认 |
| 2.1 | 原始 13 pair 复现 | **已复现** | 13 行数值 + 汇总行**逐字符相同**(表 01) |
| 2.2 | materialized 字节相等 = 0/13 | **已复现** | 独立从包内重算(表 02) |
| 2.3 | 逐对原始输入 / 计数 / reused 证据 | **已复现** | 表 07,逐记录展开 |
| 2.4 | 新协议单列不替换原始数字 | **已遵守** | §2.4,原始 13 数一字未改 |
| 3.1 | 同代码重复分组定义 | **已核对** | 33 组全部**跨 space、不跨 candidate**(表 04) |
| 3.2 | trial ID 互异 | **已核对** | 33 组中 **0 组**含重复 trial_id |
| 3.3 | 离散度指标口径 | **已核对** | `(max−min)/min` 无符号,与 13 pair 的 `(B−A)/A` **不是同一统计量** |
| 3.4 | reused 处理 | **已核对** | 见 §3;并发现**我发表的合并行有误**(§3.4) |
| 3.5 | 三组是否三个独立 AA pair | **不成立** | 三组全是**跨 space 重调**,非有意重复测量(§3.2) |
| 4.1 | P2′ legacy reader | **已复现** | 归档执行版逐值复现(表 05 头部与 §4.1) |
| 4.2 | P2′ noleak / discovery_cutoff | **已复现** | 表 05 |
| 4.3 | `UW_PROBE_BATCH.walls` 回退规则保留 | **已保留(位置已迁移)** | §4.3 —— 回退搬到 helper,仍在生效 |
| 4.4 | profile 按 status × reuse | **已复现** | 表 06(当前版)+ 表 10(n1 对) |
| 4.5 | 不得用 p90/max 验证 4% | **已遵守,并给出实测理由** | §5 —— 尾部由 reused 记录承载 |
| 4.6 | 不得称"混合代码与计时差值 = 噪声上界" | **接受,措辞已撤回** | §5.1 |
| — | 缺失项 | **无法核验,保持未知** | §6 |

**"未复现"项:0。** 有两项**"复现但发现新问题"**,均在下文单列:
§3.4(我发表的一个合并数字算错)与 §4.3(回退规则的位置变了,结论未变)。

---

## 1. 输入与身份核对

### 1.1 哈希

```
03f728333b64e3ccc80cd81f687e1ab110e0ccd83609588a32d36763d320f669  window1-verification-box1.tar.gz  1532135 B
7b4d7b8f7a55099543d773653dfeb6456bc62f408105676b8662b17b8f920787  window1-verification-box4.tar.gz  1248201 B
```

两者与 `docs/prompt-window1-data-delivery-and-n1-correction.md` 的预期值**逐位相同**。
解包至独立目录(`%TEMP%/w1co/{box1,box4}`),对原始数据**只读**。
逐文件核 MANIFEST:**box1 2496/2496、box4 1671/1671 全部匹配,0 缺失、0 哈希不符、
0 字节数不符**。独立 `MANIFEST-box*.tsv` 与包内 `MANIFEST.tsv` 哈希相同
(`a2425c92b0bbff7d` / `b50cd3da400be9be`)。

### 1.2 四个 run 的映射

| 包 | arm | runID | task | 目录身份(唯一判别依据) | ref_src_sha | best | `final_reeval_median_ms` |
|---|---|---|---|---|---|---|---|
| box4 | n1-a | `run-l3-43-20260915-071127` | 43_MinGPTCausalAttention | `…/runs-v4/n1-a` | `9d280c98c015edca` | `cand-a14c7861` | 2.8743679523468018 |
| box4 | n1-b | **同上(相同 runID)** | 43_MinGPTCausalAttention | `…/runs-v4/n1-b` | `9d280c98c015edca` | `cand-e72bb36c` | 2.5374720096588135 |
| box1 | pilot-p1 | `run-l3-43-20260915-070539` | 43_MinGPTCausalAttention | `…/runs-v4/pilot-p1` | `9d280c98c015edca` | `cand-0e23a83e` | 2.888144016265869 |
| box1 | pilot-p2 | `run-l3-21-20260915-070539` | **21_EfficientNetMBConv** | `…/runs-v4/pilot-p2` | `24e555726c485612` | `cand-87c56c65` | 3.60806405544281 |

- **n1-a / n1-b 的 runID 完全相同**已核实。可用判别字段是
  `manifest.json → config.run.runs_dir` 的末段(或包内父目录名)。本次全部按目录身份取臂。
- **pilot-p1 / pilot-p2 是不同任务**(L3:43 vs L3:21),`ref_src_sha` 也不同。
  **本简报未把它们当配对 arms,未合并任何效应量**;两者的 P2′ 与 profile 分开报告。
- 臂的处理条件(读自归档 config):

| arm | `v4.conditional_scan.mode` | `v3.diagnosis.mode` |
|---|---|---|
| n1-a / n1-b | **off / off** | vector / vector |
| pilot-p1 / pilot-p2 | active / active | vector / vector |

⇒ **n1 对两臂都是 `off`**,这解释了 §4.2 的 0 个 C4 块(设计使然,不是失败)。

---

## 2. 原始 13 pair 的重现

### 2.1 逐字复现

复用原生成器,**逻辑一字未改**,只把 base 路径指向包内
(`docs/closeout-tables/gen13_original_repointed.py`)。保留:匹配键
`(注册种子 source_sha[:12], 声明 knob 字典 sorted)`、min-of-medians 聚合、
`(B−A)/A` 分母、**floor-index 分位数无插值**、不去重、不区分 reused。

```
configs measured in BOTH arms: 13
  ... 13 行逐字与归档输出相同 ...
  TRUE same-config A/A |B-A|: n=13  min 0.01%  median 1.03%  p90 2.50%  max 5.33%
```

**13 行数值与汇总行全部逐字符相同。已复现。**(表 01)
注:生成器自带的标签 `TRUE same-config A/A` **本身就是错的**,见 §5.1;
本节只报告"复现",不背书该标签。

### 2.2 materialized 字节相等:0/13

独立从包内 `candidates/<cid>/trials/<trial_id>.py` 重算(表 02):

```
executed code byte-identical in 0 / 13 pairs
```

**未预设纠正值** —— 探针输出的是逐对的两个 artifact 哈希,读者可自行比对。
逐种子的两臂参数化体差异:`20eb42e310df` 22 行 / `49d10c6bda13` 18 行 / `6012d8732b9f` 2 行。

### 2.3 逐对明细(表 07 为完整版,含每组**全部**记录)

| # | seed | 声明 dtype | A trial | B trial | A cand | B cand | A ms | B ms | nA(reused) | nB(reused) | signed% | artifact 相等 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `20eb42e310df` | fp16 | `tr-78a78847` | `tr-a8d76719` | `cand-fb77e05b` | `cand-30ec1938` | 9.2539 | 9.7469 | 1(**1**) | 2(**2**) | +5.33% | NO |
| 2 | `49d10c6bda13` | bf16 | `tr-9499a256` | `tr-434f075a` | `cand-5cce69e6` | `cand-9832ce62` | 7.5945 | 7.5136 | 1(0) | 1(0) | −1.06% | NO |
| 3 | `49d10c6bda13` | bf16 | `tr-e7ae09b5` | `tr-1f841a9f` | 同上 | 同上 | 8.0584 | 7.9980 | 1(0) | 1(0) | −0.75% | NO |
| 4 | `49d10c6bda13` | bf16 | `tr-06915b2c` | `tr-187e9adc` | 同上 | 同上 | 8.0942 | 8.0108 | 1(0) | 1(0) | −1.03% | NO |
| 5 | `49d10c6bda13` | fp16 | `tr-f79175e9` | `tr-57ab8e6f` | 同上 | 同上 | 7.6989 | 7.4921 | 2(**2**) | 2(**2**) | −2.69% | NO |
| 6 | `49d10c6bda13` | fp16 | `tr-bf5a2709` | `tr-d0b42767` | 同上 | 同上 | 7.2233 | 7.2228 | 1(0) | 1(0) | −0.01% | NO |
| 7 | `49d10c6bda13` | fp16 | `tr-9d39a31c` | `tr-d893998e` | 同上 | 同上 | 7.3272 | 7.1440 | 1(0) | 1(0) | −2.50% | NO |
| 8 | `49d10c6bda13` | fp16 | `tr-651a665d` | `tr-d4e089b7` | 同上 | 同上 | 7.2192 | 7.2167 | 2(**2**) | 2(**2**) | −0.03% | NO |
| 9 | `49d10c6bda13` | **ieee** | `tr-25381ece` | `tr-45a0e3b4` | 同上 | 同上 | 88.5059 | 87.7722 | 1(0) | 1(0) | −0.83% | NO |
| 10 | `49d10c6bda13` | **tf32** | `tr-8eb895d6` | `tr-c5215bc8` | 同上 | 同上 | 25.5360 | 25.0296 | 1(0) | 1(0) | −1.98% | NO |
| 11 | `6012d8732b9f` | bf16 | `tr-9a4cdfb8` | `tr-2c050373` | `cand-1658a033` | `cand-086a34e8` | 5.2362 | 5.1144 | 1(0) | 1(0) | −2.33% | NO |
| 12 | `6012d8732b9f` | fp16 | `tr-72964ad9` | `tr-f4b2f865` | 同上 | 同上 | 3.7663 | 3.7622 | 1(0) | 1(0) | −0.11% | NO |
| 13 | `6012d8732b9f` | fp16 | `tr-c1381e8f` | `tr-0b8c6dac` | 同上 | 同上 | 3.9373 | 3.9398 | 1(**1**) | 1(**1**) | +0.07% | NO |

**dtype 一列是声明值**(`params.values.COMPUTE_DTYPE`),**不是实测值**;实际执行 dtype 未知。
注意第 9、10 对是 ieee / tf32,与其余 fp16/bf16 不同量级(88ms / 25ms vs 3–9ms),
**该集合本身横跨四种声明精度**。

### 2.4 新协议单列(不替换原始数字)

`scripts/probes/v41_same_config_aa.py`(外部 agent 自写)在同一对 run 上给出:

```
n=9   median_absolute_percent 1.0311   p90 2.5005 (nearest-rank)   observed_max 2.5005
```

**与原始 13 不同,原因由该脚本自报的 protocol 字段完全解释**,不需推测:

| 口径 | 原始生成器 | `v41_same_config_aa.py` |
|---|---|---|
| reused 记录 | **不区分,全部计入** | **"marked reuse excluded"** 前置排除 |
| 每侧聚合 | **min** of medians | **median** of medians |
| 分位数 | floor-index,`int(0.9*(n−1))` | **nearest-rank**,`ceil(0.9*n)` 1-based |
| 权重 | 每记录进列表 | "each matched key once" |
| 分母 | `(B−A)/A` | 同 |

**两组数字都保留,不互相替换。** 原始 13 数(§2.1)是归档结果;n=9 是新协议结果。

---

## 3. 同代码重复的核对

### 3.1 分组定义

分组键 = `sha256(candidates/<cid>/trials/<trial_id>.py)`。核对结果(表 04):

| arm | 重复组数 | 跨 >1 candidate | **跨 >1 space** |
|---|---|---|---|
| n1-a | 15 | **0** | **15** |
| n1-b | 18 | **0** | **18** |

⇒ **33 组全部是同一候选在不同 space 的重调**,没有一组跨候选。

### 3.2 三组 fresh 重复不是三个独立 AA pair

按任务要求逐条核对。三组全文:

| arm | artifact | trial / cand / space | median | spread |
|---|---|---|---|---|
| n1-a | `1e9fb48edf71` | `tr-193ab03c` `cand-a92f8911` `sp-7ae29241` → `tr-1a4826be` 同 cand `sp-61b17e75` | 33.9651 → 33.9860 | 0.0618% |
| n1-a | `71e0f867b7de` | `tr-392048ce` `cand-9f0e6fe8` `sp-f46d81dd` → `tr-798e70c3` 同 cand `sp-66d92ead` | 157.5644 → 157.6258 | 0.0390% |
| n1-a | `cb75883e7748` | `tr-3f535330` `cand-1842bd83` `sp-cd97fb00` → `tr-74ce7747` 同 cand `sp-1d86603e` | 3.8026 → 3.8001 | 0.0674% |

**三条限定全部成立,请照此表述:**

1. **"未标 reused" ≠ 已证明独立。** 三组的两条记录都只是 `reused_measurement` 为 false;
   journal **没有**任何字段证明它们是有意的、独立的重复测量。
2. **materialized 源码相等 ≠ 完整执行上下文相同。** 三组的两次测量落在**不同 space**
   (`sp-7ae29241` vs `sp-61b17e75` 等),即两次调参过程中的不同时点、不同邻居负载;
   包内没有该时点的机器状态证据。
3. **三组不是三个独立 AA pair。** 全部来自 **n1-a 单臂**、全部是**跨 space 重调的副产品**,
   且**三组分属三个不同候选**(3.80ms / 33.97ms / 157.6ms,量级差 40 倍)。
   把它们汇成一个"n=3 的分布"要假定三者同分布,**这个假定没有证据**。
4. **单对最终差值不能称为差值方差。** 每组只有 **2 条**记录 ⇒ 每组给出的是**一个差值**,
   不是方差估计;`min 0.04 / median 0.06 / max 0.07` 是**三个差值的次序统计量**,
   不是任何噪声分布的分位数。

### 3.3 离散度指标不可与 13 pair 混比

| | `v41_same_code_repeats.py` | 原始 13 pair 生成器 |
|---|---|---|
| 公式 | `(max−min)/min` | `(B−A)/A` |
| 符号 | 无符号 | 有符号 |
| 分母 | **较小值** | **A 臂的值** |

⇒ **两者不是同一统计量**,不能并列比较或互相校准。

### 3.4 ⚠ 核对中发现我发表的一个数字算错了

`docs/reply-window1-offline-data-request.md` §0 的"含 ≥1 条 reused 记录"一行发表为
`n=30 min 0.00 median 0.03 **p90 0.55** max 8.37`。

**探针从不打印合并行** —— 它只按臂打印(n1-a n=12、n1-b n=18)。那一行是我手工合并的,
**p90 算错了**。用探针自身的 floor-index 规则重算(表 04 E 节):

```
POOLED 重算:  n=30  min 0.00%  median 0.03%  p90 1.88%  max 8.37%
我发表的:     n=30  min 0.00%  median 0.03%  p90 0.55%  max 8.37%
                                              ^^^^^^^^^ 错
```

**更正:该行的 p90 应为 1.88%,不是 0.55%。** n / min / median / max 不变。
该行本身是"含 reused"的辅助行,**不承载任何结论**,但必须更正。
另注:探针每臂**最多打印 12 行**(n1-a 15 组、n1-b 18 组)⇒ 打印表不是全集,
合并必须从数据重算而非从打印行相加 —— 这正是上面出错的机制。

---

## 4. P2′ 与 profile 的核对

### 4.1 P2′ legacy reader(归档执行版 `4cf2124b`)

用包内 `readers/v41_p2prime.py`(SHA256 `4cf2124b40020e68…`,= commit `63d097d` 的 blob)
重跑,`PYTHONPATH` 指向生产 src(与 `gate2_all.sh` 的 `PYTHONPATH=$W/src` 一致):

| run | n | 边际中位\|误差\| | 条件化中位 | 倍数 | 条件化更近 | 边际符号一致 | 结论 |
|---|---|---|---|---|---|---|---|
| pilot-p1 | 12 | 0.1963 | 0.0039 | 50.9x | 11/12 | 5/12 | **已复现,逐值相同** |
| pilot-p2 | 11 | 0.0802 | 0.0031 | 25.8x | 10/11 | 7/11 | **已复现,逐值相同** |

逐 contrast 的 `g_d / y / |err_c| / g_m / |err_m|` 全部逐值复现(表 05 头部)。
mint 方向分布同样复现(p1 `inward 7 / unresolved 3 / outward 2`;
p2 `unresolved 3 / outward 5 / inward 3`)。

### 4.2 当前 reader:noleak 与 discovery_cutoff

| run | 定义 | n | 边际中位 | 条件化中位 | 倍数 | 条件化更近 | 边际符号 | 排除原因 |
|---|---|---|---|---|---|---|---|---|
| pilot-p1 | old(已撤回) | 12 | 0.1963 | 0.0039 | 50.9x | 11/12 | 5/12 | — |
| | signfix(已撤回) | 12 | 0.1795 | 0.0039 | 46.5x | 12/12 | 7/12 | — |
| | scoped(已撤回) | 12 | 0.2392 | 0.0039 | 62.0x | 10/12 | 6/12 | — |
| | **noleak(选定)** | **7** | 0.1390 | 0.0069 | **20.3x** | 7/7 | 3/7 | `thin_bucket_at_cutoff` **5** |
| | **discovery_cutoff** | **12** | 0.2010 | 0.0039 | **52.1x** | 12/12 | 8/12 | 0 |
| pilot-p2 | old(已撤回) | 11 | 0.0802 | 0.0031 | 25.8x | 10/11 | 7/11 | — |
| | signfix(已撤回) | 11 | 0.0944 | 0.0031 | 30.3x | 11/11 | 4/11 | — |
| | scoped(已撤回) | 11 | 0.2471 | 0.0031 | 79.3x | 11/11 | 6/11 | — |
| | **noleak(选定)** | **9** | 0.1766 | 0.0017 | **104.8x** | 9/9 | 6/9 | `thin_bucket_at_cutoff` **2** |
| | **discovery_cutoff** | **11** | 0.2335 | 0.0031 | **75.0x** | 11/11 | 6/11 | 0 |

- **无运行错误**,两个 run 均正常完成。
- 排除计入 `thin_bucket_at_cutoff`,reader 自己印明**按分母缺失报告,不算条件化胜出**。
- **n1 对:0 个 C4 块** ⇒ reader 印
  `no complete contrast: P2' is unreadable in this run (not a negative result)`(表 09)。
  与 §1.2 的 `conditional_scan.mode=off` 一致 ⇒ **设计使然,不是失败,也不是负结果。**
- **不事后改口径**:五种定义并排输出,`old/signfix/scoped` 保留为已撤回的诊断阶段。
- **新执行不冒充归档 stdout**:本节全部输出是**本次新执行**(表 05 / 09),
  归档包内不含 gate-2 的 stdout(见 §6)。

### 4.3 ⚠ `UW_PROBE_BATCH.walls` 回退:规则保留,但位置已迁移

实测这四个 run 的 `SCAN_BLOCK_DONE` **全部不带 `axis_f_value`**:

| run | `SCAN_BLOCK_DONE` | 其中带 `axis_f_value` | `SCAN_BLOCK_ADMITTED` | `UW_PROBE_BATCH`(含 walls) |
|---|---|---|---|---|
| pilot-p1 | 16 | **0** | 16 | 23 |
| pilot-p2 | 13 | **0** | 13 | 13 |

⇒ **回退路径是这批数字的唯一来源**,必须保留。核对结果:

- 归档执行版 `v41_p2prime.py` 在**自身**第 131–147 行实现回退。
- 当前版 `v41_p2prime.py` **不再含**该代码,乍看像被删;
  实际**搬到了 helper `v41_p2prime_history.py`**(第 184–190 行收集
  `UW_PROBE_BATCH.walls` 与 `SCAN_BLOCK_ADMITTED`,第 212–214 行在
  `axis_f_value` 为空时回填)。当前版仍读出 12 / 11 个 contrast,即**回退在生效**。
- ⚠ 但当前版 `v41_p2prime.py:44-48` 仍留着注释 "Kept for the wall-payload fallback",
  而 `_typed()` 在该文件里**已无调用点**(`grep` 无 `_typed(` 使用)⇒ **死函数 + 误导性注释**。
  不影响任何数字,记为待清理项。

### 4.4 profile 覆盖(status × reuse × 字段 × 有效性)

归档执行版 `profile_coverage.py`(`cf7b2b31`)输出**逐值复现**。当前版
`v41_profile_coverage.py` 增加了 status × provenance × metric × validity 交叉表(表 06):

| 项 | pilot-p1 | pilot-p2 | n1-a | n1-b |
|---|---|---|---|---|
| `TRIAL_DONE` | 760 | 680 | 720 | 840 |
| complete / fail | 612 / 148 | 571 / 109 | 564 / 156 | 648 / 192 |
| profile object / null | 612 / **148** | 571 / **109** | 564 / **156** | 648 / **192** |
| `marked_reused` | 43 | 41 | 41 | 51 |
| **`peak_alloc_bytes` valid_numeric** | 612 | 571 | 564 | 648 |
| **complete_valid_numeric_not_marked_reused** | **569** | **530** | 未跑当前版 | 未跑当前版 |
| `wall_ms` / `cpu_issue_ms` / `overhead_gpu_ms` | present 612 / **nonnull 0** | present 571 / **nonnull 0** | 同 | 同 |
| 声明精度分布 | bf16 437 / fp16 159 / tf32 88 / ieee 76 | fp16 397 / tf32 121 / bf16 84 / ieee 78 | fp16 282 / bf16 265 / ieee 90 / tf32 83 | bf16 372 / fp16 251 / tf32 128 / ieee 89 |
| 按可得性的比例 | 80.5% | 84.0% | 78.3% | 77.1% |

**bad_type / nonfinite / negative 全为 0**(当前版逐字段报告)。
**无 profile 对象的记录 = 全部 fail 记录**,数量精确相等(148=148、109=109、156=156、192=192)。

**三条判读(按任务要求)**:

1. **覆盖状态不自动代表可重评分。** 80.5% / 84.0% 是"字段存在且数值有效"的比例;
   扣掉 `marked_reused` 后可用于**独立**测量的只有 569 / 530。
   任何跨目标冠军必须在该子集内读并报告被排除数。
2. **`wall_ms` / `cpu_issue_ms` / `overhead_gpu_ms` present 但恒 null** ⇒
   依赖它们的目标**现在不可离线重评分**。
3. **dtype 留为未知。** `COMPUTE_DTYPE` / `DOT_PRECISION` 是**声明的 knob 标签**,
   profile 里没有实测 dtype 字段。当前版探针自己印明这一点。
   ⇒ 内存目标的离线重评分必须按**声明**精度分层,并声明"实际 dtype 未验证"。

---

## 5. 关于 4% 阈值与"噪声上界"措辞

### 5.1 我撤回"噪声上界"这个说法

任务明确要求:**不得把混合代码与计时差值称为噪声上界。** 接受,撤回。

我在 `docs/reply-window1-offline-data-request.md` §0 与
`docs/prompt-window1-data-delivery-and-n1-correction.md` 第一部分写的
"**是同代码测量噪声的【上界】**",**措辞不成立**。理由(核对后我同意):
"上界"是一个关于噪声分布的**推断性断言**,而这 13 个差值同时含
(a) 计时噪声、(b) 两臂参数化差异、(c) 声明精度横跨 fp16/bf16/tf32/ieee 四类、
(d) 4 对由 reused 记录构成。**混合量不能宣称界住其中任一分量** ——
它既可能高估也可能低估,方向无法从这批数据判定。

**正确措辞(建议采用)**:

> 13 对"同种子 + 同声明旋钮 + 不同已执行代码"跨臂配对的延迟差 `|B−A|`:
> n=13,min 0.01%,median 1.03%,p90 2.50%(floor-index,无插值),max 5.33%。
> **这是一个描述性混合量**,含计时噪声、两臂参数化差异、四种声明精度与 4 对 reused 记录。
> **它不是噪声上界,不是计时噪声底,不是 tie 阈值,也不是 MDE。**

这与 `v41_same_config_aa.py` 自带的 `inference` 限定语一致
("observed maximum is not a population upper bound, timer noise floor, tie threshold or MDE"),
**该限定语先写对了。**

### 5.2 不用 p90/max 验证 4%,并给出实测理由

任务要求不得从 p90/max 验证 4%。遵守。**而且实测给出了一个更强的理由**(表 08):

```
全 13 对(已发表)                 n=13  min 0.01%  median 1.03%  p90 2.50%  max 5.33%
剔除"两侧最小值均标 reused"的对    n= 9  min 0.01%  median 1.03%  p90 2.33%  max 2.50%
被剔除的 4 对: 0.03 / 0.07 / 2.69 / 5.33
```

⇒ **最大值 5.33% 与第二大 2.69% 都来自两侧最小值均标 `reused` 的对。**
这条分布的**尾部由 journal 未认证为独立测量的记录承载** ⇒
**p90 / max 在此不能用于验证或标定任何阈值**,与所用阈值是 4% 还是别的数无关。

**收敛性证据**:我这条剔除(按"使用的最小值是否 reused",min-of-medians)
与 `v41_same_config_aa.py` 的剔除(按"排除 reused 观测",median-of-medians)
**独立写成,却落在完全相同的 9 个值上**(表 13):

```
mine   : 0.01 0.11 0.75 0.83 1.03 1.06 1.98 2.33 2.50
theirs : 0.01 0.11 0.75 0.83 1.03 1.06 1.98 2.33 2.50
median  1.0300 vs 1.0311    max  2.5000 vs 2.5005
p90     2.33 (floor-index)  vs  2.50 (nearest-rank)   <- 仅分位数规则之差
```

### 5.3 4% 本身:本次不做任何裁决

**本次核验不对 4% 的合法性作任何判定**,既不验证也不否证。可说的只有:
四个 run 里**不存在**可用于标定 tie 带的证据集 —— 混合量不合格(§5.1),
三组 fresh 重复不合格(§3.2)。**要标定必须显式重放同一 materialized artifact**,
那是一次需要 GPU 的新测量,**本次不做、也不建议现在做**(两台机器在跑实验)。

---

## 6. 剩余未知(不补做)

| 项 | 查证依据 | 保持未知的结论 |
|---|---|---|
| 四个 run 启动时的 dirty patch | `RUN_CREATED.payload` 只有 `run_id`;`manifest.json` 只有 `task`/`created`/`config`;包内无 patch | "该 run 跑的就是 commit X"须降级为"HEAD 在 commit X,**工作区洁净性未知**" |
| 每条 trial 的实际执行 dtype | profile 无 dtype 字段;事件全文无 `torch_version`/`triton_version`/`toolchain`/`env_probe` | 精度只有**声明值**;按精度分层的结论必须带此声明 |
| reused 记录的复用来源 | `reused_measurement` 是布尔,无来源指针 | 无法把 reused 折算为独立样本;**§5.2 的尾部因此不可用** |
| gate-2 四步的归档 stdout | 包内 `stdout/` 只有编排器日志与启动前审计、卡分离证据,**无 reader 输出** | 本简报 §4 全部是**新执行**,已标注,不冒充归档 stdout |
| worker / collector 版本 | 事件里无这两个键 | 只能由"启动 commit + venv 路径"间接推断 |
| 计时噪声底 | 无合格证据集(§5.1、§3.2) | **未知。** 需显式重放同一 artifact 才能标定 |
| 三组 fresh 重复是否同分布 | 三个不同候选、量级差 40 倍 | **未知**,不可汇成一个分布 |
| n1 对的 P2′ | `conditional_scan.mode=off` ⇒ 0 个 C4 块 | **不可读**,不是负结果 |

**以上一项都未补做**,未安排任何 GPU 测量,未扩大任务范围。

---

## 7. 对 v5research §§1.1 / 1.2 / 14 的替换建议

**未修改该文件正文**,由上级整合。以下为建议文本。

### §1.1 —— 删除"同配置噪声"的表述

> **替换为**:窗口 1 未提供可用于标定 tie 带的测量噪声估计。可得的 13 对跨臂配对
> (同种子 + 同声明旋钮)**已执行代码 0/13 相同**,且尾部由 reused 记录承载,
> 因此是描述性混合量,**不作为噪声底、上界或阈值依据**。
> 现有 τ 取值在窗口 1 内**既未被验证也未被否证**。

### §1.2 —— 把标定写成可执行动作

> **替换为**:每个目标 × 每个场地必须有自己的 tie 带标定,且标定**必须显式重放同一个
> materialized artifact**(以 `candidates/<cid>/trials/<trial_id>.py` 的字节哈希为身份),
> **不得事后从搜索日志中配对** —— 日志配对只固定声明(种子 sha + 旋钮),
> 不固定已执行代码。标定重复必须是有意安排的、记录为独立测量的重复,
> 而非跨 space 重调的副产品。

### §14(方法学纪律)—— 新增两条

> **(a) identity 三层纪律。** 报任何"同配置 / 同代码"数字前,必须说明配对键固定了哪一层:
> 注册源 sha → 空间 sha → **materialized artifact 字节哈希**。前两层相同而第三层不同,
> 在本项目里是**常态**(parameterizer 是 LLM 调用)。仅前两层相同**不构成**同代码证据。
>
> **(b) reused 与打印上限纪律。** `reused_measurement` 为 false **不等于**已证明独立;
> 为 true 的记录不得进入任何用于标定阈值的分布尾部。
> 探针的打印表可能有行数上限(如每臂 12 行),**合并统计必须从数据重算,不得由打印行相加**
> —— 违反此条已实际产生一个错误的 p90(见本简报 §3.4)。

---

## 8. 精确可运行命令与 helper 版本

**环境**:Windows,`D:/Anaconda/python.exe`(3.12),`PYTHONIOENCODING=utf-8`。
`PYTHONPATH` 指向生产 src 仅为 import `kernel_optimizer.store.read`(与
`gate2_all.sh` 的 `PYTHONPATH=$W/src` 相同做法)。**未安装任何依赖。**

```bash
W=%TEMP%/w1co            # 解包目录
SRC=D:/Pyhon_projects/opop/v4/src
P=D:/Pyhon_projects/opop/v4/scripts/probes

# 1) 逐文件哈希核对(见 §1.1;脚本内联,已附于 closeout-tables)
# 2) 原始 13 pair
python docs/closeout-tables/gen13_original_repointed.py $W/box4          # -> 01
# 3) artifact 重配对
python $P/v41_aa_pair_artifacts.py           <base>                      # -> 02  (base 改为 $W/box4)
# 4) 同代码重复
python $P/v41_same_code_repeats.py           <base>                      # -> 03
# 5) P2' 归档执行版
PYTHONPATH=$SRC python $W/box1/readers/v41_p2prime.py $W/box1/runs/pilot-p1
PYTHONPATH=$SRC python $W/box1/readers/v41_p2prime.py $W/box1/runs/pilot-p2
# 6) P2' 当前版(五定义)
PYTHONPATH=$SRC python $P/v41_p2prime.py $W/box1/runs/pilot-p1           # -> 05
PYTHONPATH=$SRC python $P/v41_p2prime.py $W/box1/runs/pilot-p2           # -> 05
PYTHONPATH=$SRC python $P/v41_p2prime.py $W/box4/runs/n1-a               # -> 09
PYTHONPATH=$SRC python $P/v41_p2prime.py $W/box4/runs/n1-b               # -> 09
# 7) profile 归档执行版 + 当前版
python $W/box1/readers/profile_coverage.py $W/box1/runs/pilot-p1 $W/box1/runs/pilot-p2
PYTHONPATH=$SRC python $P/v41_profile_coverage.py $W/box1/runs/pilot-p1  # -> 06
# 8) 新协议(外部 agent 自写)
PYTHONIOENCODING=utf-8 python $P/v41_same_config_aa.py \
    $W/box4/runs/n1-a $W/box4/runs/n1-b --json                           # -> 11
```

**helper 版本(LF 规范化后 SHA256 前 16 位)**:

| 文件 | 哈希 | 说明 |
|---|---|---|
| `scripts/probes/v41_p2prime.py` | `0ef091a01eaf0374` | 当前版(工作区) |
| `scripts/probes/v41_p2prime_history.py` | `1e04723f610ba4e1` | **承载 walls 回退**(§4.3) |
| `scripts/probes/v41_profile_coverage.py` | `456b3b48d7fc5bfd` | 当前版 |
| `scripts/probes/v41_aa_pair_artifacts.py` | `34c0f4922c249f85` | |
| `scripts/probes/v41_same_code_repeats.py` | `fe54286b9d95b15f` | |
| `scripts/probes/v41_same_config_aa.py` | `f8217c31a658240e` | 外部 agent 自写 |
| `src/kernel_optimizer/store/read.py` | `2882d9e9d27d882a` | 经 PYTHONPATH import |
| 包内 `readers/v41_p2prime.py` | `4cf2124b40020e68`(全) | **归档执行版** = commit `63d097d` blob |
| 包内 `readers/profile_coverage.py` | `cf7b2b317f2e4bb1`(全) | **归档执行版** = commit `26b556d` blob |

仓库 HEAD:`27694e15b70de823fe32f87995b1a056423ccd38`。

### 本次新增的最小 helper(已披露)

按任务"优先复用现有脚本,仅必要时新增最小纯 CPU helper,披露改动":

| 新增 | 用途 | 为何必要 |
|---|---|---|
| `gen13_original_repointed.py` | 重现原始 13 pair | 原生成器是内联代码;**逻辑一字未改,只改 base 路径**(改动即那一行) |
| 内联逐文件哈希核对 | §1.1 | MANIFEST 校验无现成脚本 |
| 内联 `audit3.py`(→ 表 04) | §3.1–3.4 | 探针不打印分组/独立性/合并行,需另核 |
| 内联 `pair_evidence.py`(→ 表 07) | §2.3 | 原生成器只印 min,不印每组全部记录与 reused |
| 内联 `sens.py`(→ 表 08) | §5.2 | reused 敏感性 |
| 内联 `wallsrc.py`(→ §4.3) | §4.3 | 判定 `axis_f_value` 是否存在 |
| 对 `v41_aa_pair_artifacts.py` / `v41_same_code_repeats.py` 的改动 | 指向包内 | **仅 base 路径两行**,逻辑未动 |

**均为纯 CPU、只读、无 GPU / torch / CUDA。**

---

## 9. 边界确认

| 禁止项 | 状态 |
|---|---|
| GPU / torch / CUDA 初始化 | **未做。** 全部脚本只读 JSON 与文本;`test_g10` 之类需 torch 的路径未触及 |
| 服务器访问 | **未做。** 只用本地 `v4/.deliver/` 的两个包 |
| 重启实验 / 新基准实验 | **未做。** 两台机器的窗口 2 实验未受任何影响 |
| 安装依赖 | **未做。** 只用现有 `D:/Anaconda/python.exe` 标准库 |
| `git commit` | **未做。** 本文件与表格已写入工作区,**未提交** |
| 修改生产 source / config / prompt / scorer | **未做。** 只读 `src/kernel_optimizer/store/read.py` |
| 修改 v5research 正文 | **未做。** §7 只给建议文本 |
| v5 实现计划 / 代码 | **未做。** |
| 为消除未知而扩大任务 | **未做。** §6 八项全部保持未知 |
