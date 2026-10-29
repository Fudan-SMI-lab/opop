# 独立证据核验角色实验

## 终态审计与决定

本批在 **2026-10-29 03:38:01 UTC** 正常结束，两角色都完成生成、逐项覆盖验证和 CPU 传输检查；这不等于事实核验通过。**两份整包报告均不采纳，本轮 verifier 实验关闭，不再堆叠 reviewer 或重复直到通过。** 生产 core、prompt、sampler、allocator、接受规则均未修改，也没有新的 GPU、搜索或性能接受。

审计程序为 `scripts/experiments/v6_audit_diagnostic_verification_20261029.py`。[公开审计回执](evidence/v6-diagnostic-verification-20261029/public-audit.json)对应服务器 `terminal-audit.json` SHA `115be70c156394253dbd73b9df34ef79be8127f6a03a33f96d7b38a05a0bc2ed`；完整工具历史、CPU 检查和反证留服务器 `audit/`。123 个冻结文件保持，两个角色的原四输入、原样 proposed packet、schema 及 worker environment 逐字核验。相同通用 prompt/model/tool 设置，但任务不同，不是单变量 A/B。

| 项目 | Qwen | FSDP |
|---|---:|---:|
| 初始文件／显式 bytes | 7 / 57,776 | 7 / 76,543 |
| schema 修复 | 0 | 0 |
| assistant／工具调用 | 50 / 65 | 65 / 76 |
| 原条目覆盖 | 32：29 supported、3 mixed | 34：32 supported、2 mixed |
| 新 packet bytes | 15,847 | 14,649 |
| provider 费用估计 | 1.5875489554 | 2.1401013024 |

费用只记录 provider 估计，不是账单，也未用于限额。两份 `overall=usable` 都不能作为事实正确的依据。生成文件的精确 SHA、所有引用检查和 snapshot 身份见公开回执；没有手工改写任何生成答案。

Qwen 核验确实用 CPU 读取并比较了四份候选完整输出、controls 和源参考，主要关系有证据支持；但它继续认可“evaluator compare 从未执行”，实际第一 observe 返回后执行过第一次 compare，第二 observe 才阻止第二次 compare。它也未识别原报告将两个 token 负例误写成 token＋logit 负例，以及将 18.75 说成大于 31.125 的错误比较。新 packet 又将 observed baseline 称为 uninstrumented，并将 request 差异缩写为 only entry，遗漏 output 与 diagnostic_context。

另一个新错误是宣称原 CPU 核查脚本找不到：它搜索了错误 sandbox，随后使用过浅的目录深度；原报告给出的 `diagnostic-interpretation-20261029/qwen/sandboxes/diagnostic_interpretation-700d4d3da1f9/checks/verify_artifacts.py` 实际存在，SHA `3452d1a9c5f349bc1487063b9046fc46c8f5d7b05d71f5ed9e5c6ea08e4fd1bd` 不变。引用文件 SHA 匹配并不能证明引用内容支持陈述。

FSDP 核验正确修正了部分相等参数的组成，并实际复算了 163 个文件的 core SHA；但仍认可“差异只能从 backward/reduction 进入”、将 top absolute-error 项当作越容差位置，以及将包含原 compare 的 `observer_wall_s` 当作新增开销。两 rank 的 top8 logits 项全部在相对容差内，真正 1,898 / 1,592 个失败位置没有完整保存，不能从这16项推断失败分布。一个精确 baseline 也不足以证明通用噪声为零或排除环境交互。

FSDP 新 packet 对候选计时/显存“完全不存在”的描述过宽：没有 clean 样本或正式 metrics，但 worker result 中有 setup elapsed 和 allocator 数值；这些属于安装阶段，不能当 warm-model 性能，却不能说没有记录。原报告引用的 mirror.math_policy_unchanged 不存在，实际 worker 字段为 true；核验换了引用但没有指出原 locator 错误。核验输出还有一处少写 `fsdp-` 的不存在引用路径，按原样保留。

完整命令、输出和保留的 CPU 脚本均已检查，未观察到读取后来成功实现、开发者终态答案或另一案例答案；文件系统并非隔离环境，因此不作绝对无访问声明。保留了 Python 类型/索引错误、错误路径、缺失解释器/rg、pipeline exit0 隐藏异常等真实历史。Qwen 曾把 CPU 文件写到 `/tmp/opencode/checks`，之后复制九份文件回 sandbox，不能声称完全遵守仅 sandbox 写入；原研究产物未改。审计自身一次空命令字符串打印 IndexError 也单独留档，未重跑角色或模型。

两份新 packet 在既有 `supplemental_evidence` 中均原样传输、其他路由字段相同；**没有 selector/candidate 接收或使用这两份报告**。下一步按[小型证据索引 candidate 重放](v6-diagnostic-index-candidate-20261029.md)，只给原 candidate 增加已有 Agent 诊断产物的索引，不传这些报告，也不注入人工事实表、原因或修复方案。各任务只做一次真实角色生成，先审计实际读取和代码，再决定是否需要固定执行。

审计后实际150GiB容器剩约 **9.50GiB**，GPU worker 为空；本轮没有清理或 raw 本地转存。旧优化接受、拒绝、全部样本和手工源映射／开发者辅助边界保持。以下为启动时的历史记录。

已于 2026-10-29 03:02:08 UTC（香港11:02）启动，冻结 `9c564293739babd1c40a7232807b23d66370e96e`，controller949392，真实 OpenCode 本地服务健康检查通过。预检 Qwen/FSDP 六显式输入分别 **57,776 / 76,543 bytes**，共 123 个冻结文件；生产 core 不变。见[启动回执](evidence/v6-diagnostic-verification-20261029/launch.json)、[预检](evidence/v6-diagnostic-verification-20261029/preflight.json)。

03:04:36 核验 Qwen `generation-8dcfb0a5babb` / session `ses_e7a6445ebffeREPiWgMj9dZ3cX`：7 个初始文件逐字符合协议，17 条真实 assistant 消息、23 次工具调用；真实 read 已读取 `evidence-index.json` 和完整 `proposed-packet.json`。初始工具访问未见开发者审计/后续答案，完整终态仍需核查；FSDP 后续按序执行。见[初始快照核查](evidence/v6-diagnostic-verification-20261029/startup-verification.json)。

唯一普通完成/磁盘监听 PID50544 已核实实际进程命令，原监听47044标记已处理。当前实际 df 余 10,218,143,744 bytes（约9.52GiB），阈值8GiB；没有 GPU worker、清理或 raw 本地转存。见[监听回执](evidence/v6-diagnostic-verification-20261029/wakeup-setup.json)。完成后按真实结果自动续行，等待期间不调用模型。

当前诊断解读已经能找到关键观测，但两份真实报告同时含有具体事实错误、范围错误和过度因果结论。此实验针对这一已展示的缺口，用相同通用 `diagnostic_verification` 指令，按 Qwen、FSDP 各进行一次新的真实请求。不是重复 interpreter 直到通过，也不直接新增生产审批步骤。

每个角色得到上一阶段原始四个输入的精确字节，以及原样 `proposed-packet.json` 和通用 verification schema；框架另加 worker 环境。不给开发者错误清单、修正事实表、后来成功实现或另一案例答案。允许按索引和原 packet 引用读取已有原始文件、前一角色 CPU 检查及相关源码；实际访问仍须审计，外部文件系统不是隔离环境。

输出 `verification.json`，逐项检查原 summary/facts/limitations/unresolved/hypotheses，区分 supported/mixed/unsupported/contradicted/unverified，附引用和理由。另生成 `supplemental-evidence.json`，只保留有支持且对原失败有用的少量观察、边界和未解决问题；不要求优化实现，hypotheses 可以为空。验证器只检查接口与逐项覆盖，不会用硬编码“正确答案”替代事实审计。

本批无新模型/GPU/collective、诊断 worker、候选搜索、性能接受、reference 复制或 production 改动。允许 CPU 检查已有文件，保留错误和所有工具历史。API 仅统计，不限额。两案例任务与证据不同，不是同快照 A/B，也不能据一次结果宣称稳定泛化。若 verifier 仍产生错误，记录不足，不自动迭代直到通过或继续堆叠角色。

生成后仅在 CPU 经既有 `feedback.route` 核对原样 packet 在 `supplemental_evidence` 中保留、其他路由字段不变。不会自动交给 selector/candidate；后续审计必须分别确认生成完成、事实支持、实际输入投递和下游使用，不能互相替代。

控制器为 `scripts/experiments/v6_diagnostic_verification_20261029.py`，目录 `/root/autodl-tmp/dev/v6-work/diagnostic-verification-20261029`。冻结生产 core `ed24af7a…` 不变；保留上一批 116 个冻结文件，并增加原样 packet/引用文件哈希。启动前检查真实容器 df 和 worker，预计仅小型 role/CPU 产物；raw/private state 留服务器，GitHub 仅代码、哈希和小型摘要。新长工作配置唯一普通完成/磁盘 watcher，不启用 Goal、模型等待轮询或投递测试。
