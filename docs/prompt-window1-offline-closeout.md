# window1 离线核验交接

## 目标与输入
仅用已有数据完成有界收口，不扩展研究、平台或证据台账。
本地输入位于 `v4/.deliver/`：
- `window1-verification-box1.tar.gz`、`window1-verification-box4.tar.gz`
- `MANIFEST-box1.tsv`、`MANIFEST-box4.tsv`
预期哈希及原始 13 pair 生成器见：
- `docs/prompt-window1-data-delivery-and-n1-correction.md`
- `docs/reply-window1-offline-data-request.md`
下文路径均相对 `v4/`。第三轮已纠正 13 pair 身份，
但“混合代码与计时差值是噪声上界”的说法不能沿用。

## 1. 核对输入与身份
校验包级及逐文件哈希；另处解包，只读原始数据。
记录四个 run 的映射：任务、arm、runID、目录身份、源路径。
n1a/b 的 runID 相同，必须以目录身份区分。
pilot p1/p2 属于不同任务，不视为配对 arms，也不合并效应。

## 2. 重现原始 13 pair
复用原生成器，严格保留匹配、聚合及 floor-index 分位数规则。
核验“13 对中 materialized 字节相等为 0/13”，不预设纠正值。
列出精确配对、原始输入、计数及 reused 证据。
相关脚本：`scripts/probes/v41_same_config_aa.py`、
`scripts/probes/v41_aa_pair_artifacts.py`。
新协议输出须单列，不得替换原始数字。

## 3. 核对同代码重复
复用 `scripts/probes/v41_same_code_repeats.py`。
核对分组定义、离散度指标、不同 trial ID 及 reused 处理。
未标 reused 不等于已证明独立；materialized 源码相等
不等于完整执行上下文相同；三组不必然是三个独立 AA pair。
单对最终差值不能称为差值方差。

## 4. 核对 P2prime 与 profile
复用 `scripts/probes/v41_p2prime.py`、
`scripts/probes/v41_profile_coverage.py`。
分别报告 P2prime legacy reader、当前 no-leak、
discovery_cutoff 的结果、n、排除原因及错误。
适用时保留历史 `UW_PROBE_BATCH.walls` 回退规则。
保留原定义，不事后改口径；新执行输出不能冒充归档 stdout。
profile 按 status × reuse 列出存在、非空、数值有效情况。
覆盖状态不自动代表可重评分；缺失 dtype 留为未知。
不得从 p90/max 验证 4% 阈值，也不得把混合代码与计时差值
称为噪声上界。不要求补齐缺失 dirty patch、实际 dtype、
reuse 来源或旧 stdout；缺证据就明确未知。

## 交付与边界
只交一份 `docs/result-window1-offline-closeout.md` 简报，
附少量关联机器可读表格、stdout、精确可运行命令及 helper 版本。
逐项标记“已复现／未复现／无法核验”，列出预期与实际、
差异原因及对应命令；汇总剩余未知，给出 v5research
§§1.1/1.2/14 的简短替换建议。不编造纠正后的数值。
优先复用现有脚本，仅必要时新增最小纯 CPU helper，披露改动。
不要修改 v5research 正文，由上级整合；不做 v5 实现计划或代码。
禁止 GPU、torch/CUDA 初始化、服务器访问、重启实验、
新基准实验、安装依赖、git commit，以及修改生产源码、
配置、prompt 或 scorer。已有数据足够限定核验范围，
不可为消除未知而扩大任务。
