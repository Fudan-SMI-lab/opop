# Window1 离线核验：现有数据与实际执行脚本交付请求

请交付以下已完成运行的现有数据，供 window1 离线核验使用。本次仅收集、核对和打包已有材料，不开展新 GPU 实验，不重写 v5，也不要求新增数据分析。

本地 76 项 CPU 测试已经完成，但这不等于已用实际 window1 数据完成复现。新写的 AA 分析脚本也不等于原始 13 对结果的生成器。请保留旧报告及预注册定义，不覆盖、不改写。

## 1. 四个已完成 run 的原始材料

配置中的父目录基址为 `/root/autodl-tmp/opop-workspace/opop-glm/runs-v4/`，目标如下：

| Run | 机器 | 配置中的父目录 |
| --- | --- | --- |
| n1-a | box4 | `/root/autodl-tmp/opop-workspace/opop-glm/runs-v4/n1-a/` |
| n1-b | box4 | `/root/autodl-tmp/opop-workspace/opop-glm/runs-v4/n1-b/` |
| pilot-p1 | box1 | `/root/autodl-tmp/opop-workspace/opop-glm/runs-v4/pilot-p1/` |
| pilot-p2 | box1 | `/root/autodl-tmp/opop-workspace/opop-glm/runs-v4/pilot-p2/` |

请核实并列出每个 run **实际对应的带时间戳子目录的完整绝对路径**，不能只重复上述父目录。若有多个子目录，请说明哪个对应已完成的目标运行及判断依据，不要自行挑选结果较好的目录。

每个 run 请提供：

- 原始、完整且未经修改的 `events.jsonl`，以及实际生效的 resolved config，而非仅提供默认配置或模板。
- 实际 run ID、task、机器标识、GPU 型号及数量、启动时 commit；若已知存在 dirty 工作区，提供当时的 patch 或已有记录。未知项明确标注，不能用当前状态冒充启动状态。
- 文件清单、每个交付文件的字节大小和 SHA256；已有分析的 stdout 及关联报告、输出文件，若存在则一并交付。

## 2. 实际执行过的 reader 与执行上下文

需要当时**真正执行的脚本**，而不只是仓库当前版本。尤其请核实 gate2 是否从 `/root/probe-clean` 执行，避免拿其他工作树的同名文件替代。

以下文件只交付实际用过的那些；未使用的注明未使用，不要为了补齐名单重新运行：

- `v41_p2prime.py`
- `v41_p2prime_audit.py`
- `v41_profile_coverage.py`
- `v41_n1_noise_floor.py`
- `did_the_enqueued_point_win.py`
- `v41_refusal_delta.py`
- `gate2_all.sh`

对每个实际使用的脚本，请给出当时脚本副本、SHA256、所属 commit（若可确认）、完整执行命令及参数、cwd、实际输入路径和输出路径，以及它直接依赖的 helper 文件副本和哈希。说明脚本副本是否能确认与当时执行版本一致；若只能找到当前文件，请明确标注版本证据缺失。已有 stdout 请与对应命令关联。

## 3. 原始 13 对 AA 结果的生成方法与配对表

此前报告的 AA 结果为 `n=13，median=1.03，p90=2.50，max=5.33`。请交付实际生成这组数字的原始脚本或内联代码、完整命令、已有输出及 13 对配对明细表，并注明这些数字的单位及对应指标。

现有 N1 reader 不能直接视为这组结果的生成器。若原始生成代码或表格已丢失，请列为缺失，本次不要新做分析补充；后续如另行生成，必须标注为“新分析”，不能称为“旧结果复现”。

配对表请尽可能逐对列明两侧数据，缺失字段写明未知，并关联原始证据：

- run ID、candidate ID、trial ID、候选 source identity、完整参数及配对 matching key。
- 原始 latency 值或可精确定位这些值的原始记录、样本计数，以及 reused 标记与复用来源。
- backend、声明的 precision，以及有证据支持时的实际执行 precision；不要把声明值当作实际值。
- 两侧 median、带符号差值、绝对差值及百分比差值，注明方向和单位。

请同时提供当时使用的精确定义：各层级如何聚合，是否取 min、median 或 min-of-medians；重复记录与 reused 样本如何处理；百分比的分母及公式；p90 的分位数算法和插值方法；precision 是否分组及未知值处理；matching key 与各 identity 的作用域，例如仅 run 内唯一还是跨 run 可比。以原始代码或已有记录为准，无法确认的不要推测。

## 4. 候选源码、采集版本与环境证据

- 提供已有 candidate source、parameterized source、materialized source 的 manifest、对应关系、哈希及可访问路径。先交付索引和核验必需的源码即可，不必下载全部候选源码。
- 不要默认这三类 source 的哈希 identity 相等，也不要在没有映射证据时将其合并。请说明各哈希对应的具体对象及作用域。
- 提供已有 worker、collector 的版本信息，以及运行时环境元数据，包括已有的软件版本、backend、dtype/precision 相关记录。当前环境信息若与运行时证据不同，须分别标注。
- 缺少实际 dtype 或 precision 证据时，结论保持“未知”，不要为补证据重跑 GPU。

## 5. 交付方式

**首选：最小必要数据压缩包。** 包内原始文件保持内容不变，附路径映射和文件清单。请返回下载 URL、压缩包字节大小和 SHA256，并保留各原始文件的大小与 SHA256，便于核验传输完整性。

**备选：只读 SSH 下载。** 请提供 SSH alias 或 host、port、用户名、认证方式、四个 run 及脚本材料的精确路径，并明确授权接收方只读访问和下载这些指定材料。回复中不要发送密码、私钥或其他认证秘密，认证材料通过已有安全渠道配置。

不需要模型权重、venv、完整数据库或整个 workspace。仅打包现有核验材料；不要为了整理包而改写原始日志。

## 6. 操作边界与回复格式

保持正在运行的生产任务所用 source、config、prompts、scorer 冻结，不修改、不重启，不启动 GPU 作业。不得修改原始日志、挑选性保留有利数据、为配合旧结论 cherry-pick 样本，或编造缺失事实。保留旧报告和预注册定义，缺失材料直接说明。

请按以下六部分回复：

1. **材料清单**：四个 run 的已核实时间戳子目录、run 元数据、文件列表、大小与 SHA256。
2. **交付与访问**：压缩包下载 URL、大小、SHA256，或只读 SSH 信息、精确路径及读取下载授权。
3. **实际 readers**：实际使用的脚本版本、副本、哈希、命令、cwd、输入输出、直接 helpers 及已有 stdout。
4. **13 对 AA 方法与表格**：原始生成器、命令、精确口径和已有配对表；无法找到的明确说明。
5. **缺失项**：缺什么、已有查找或判断依据、哪些结论因此保持未知，区分当时记录与当前文件。
6. **操作确认**：确认未改生产 source/config/prompts/scorer，未重启、未启动 GPU 作业、未新增分析、未改原始日志或筛选样本，旧报告和预注册定义保持不变。
