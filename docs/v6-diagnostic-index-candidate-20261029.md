# 原候选角色按需读取 Agent 诊断产物

已于 **2026-10-29 03:58:31 UTC（香港11:58）** 启动，冻结 `2a76a0af246787d5621f7dc7a0a4052bf452465a`，controller957415。真实 OpenCode 本地服务健康检查通过、120冻结文件核验，core不变。见[启动回执](evidence/v6-diagnostic-index-candidate-20261029/launch.json)与[冻结预检](evidence/v6-diagnostic-index-candidate-20261029/preflight.json)。

04:00:57 实际 Qwen `generation-9e85d07b98af` / session `ses_e7a30a566ffe7OdlVv8eIZKFM7` 核对32初始文件，仅previous-feedback内supplemental_evidence变化，其余输入、prompt、model/tool设置均与原快照相同。已有9条assistant/15次工具；`prt_185cfbf64001opWtVrq2R7Ec6K` 对原反馈grep的实际输出含新增索引（输出SHA `cc07cea59ad023072a128caa685654eb63e3d4f8d44612734931f2c41c71eeb9`）。这证明初始投递及索引文字进入工具输出，**尚不等于已经读取索引文件或正确使用原始观察**。早期完整工具访问未见后来答案，终态仍须审计。FSDP随后按序执行。见[初始输入核查](evidence/v6-diagnostic-index-candidate-20261029/startup-verification.json)。

唯一普通完成/磁盘监听59216已检查实际进程命令，旧50544已处理且退出；见[监听回执](evidence/v6-diagnostic-index-candidate-20261029/wakeup-setup.json)。当前df约9.20GiB，GPU为空；等待不调用模型，终态或低于8GiB时触发后续核查。没有raw本地转存、清理或新的优化接受。

前两案例的诊断脚本已经真实执行并产生有用观察，但解读和独立核验仍留下事实错误。此次直接检验现有 candidate 角色能否按需利用原始诊断证据；不继续增加 reviewer，不向生产添加强制核验层。

控制器 `scripts/experiments/v6_diagnostic_index_candidate_20261029.py` 通过现有 `ProjectAgents.replay_snapshot`，按 Qwen、FSDP 各执行一次新真实 candidate 会话。两者沿用各自历史失败输入的 prompt、模型、工具、选择结果、任务、源码和历史候选。唯一输入变化是 `inputs/previous-feedback.txt` 中既有 `supplemental_evidence: null` 替换为小型索引对象，**各增加592 bytes**。其他 feedback 字节/字段不变，不新增提示段落或手写候选。

| 项目 | Qwen | FSDP |
|---|---|---|
| 原始生成前 snapshot | generation-62e1890de41e | generation-c519752a5f2d |
| 原始输入数 | 32 | 37 |
| previous trial | a254520daa | 86e14487d0 |
| 原 feedback → 新 feedback bytes | 5,200,220 → 5,200,812 | 803,593 → 804,185 |
| 原历史候选 SHA 前缀 | a86694e7 | df3d62ae |
| 按需索引 bytes | 13,127 | 18,476 |

索引直接复用 interpreter **开始之前**准备的 `inputs/evidence-index.json`，只列原任务、原失败、原源码/controls、Agent 生成的 probe/plan 和已完成固定观察的路径及哈希。没有 interpreter/verifier 输出、开发者反证表、后来成功候选或人工数值归因。附带通用范围说明：这些是有扰动的固定观测，不是接受决定或唯一原因；缺失的保存值仍未知。索引位于原 interpreter sandbox 的 inputs 下并不意味着提供其输出，实际外部读取必须检查。

Qwen 从未收到事实的旧失败反馈 candidate 初始状态开始，原错误选择描述也保持；不提供该次后来生成的 ce85 或成功 d5bd。FSDP 从最初搜索第三候选的初始状态开始，只给当时失败的86e实现/反馈，不提供后来 K3 成功答案。FSDP 原反馈包含该试验原有参数记录，新 Agent 观测只对应固定 **K6**，不能推广到其他参数。两角色没有重新做 selector，原选择不变；不是完整自主 cohort，也不是两任务之间的单变量 A/B。

此次控制器仅生成 candidate.py/proposal.json，使用原有 schema 校验与正常修复机制；API 只统计、不限额。不安排 GPU worker、collective、搜索、controls/reference 复制、指针发布或性能接受。原角色工具可能自行进行探针，终态须审计实际命令/返回值及是否超出 CPU 验证范围，不能凭“生成完成”宣称没有 GPU 行为或完整质量通过。

预检已通过：core 为 `ed24af7acc8caa15578e0b859c85ef8a901be51354448ed39094606ed9e10a2f`，两原 snapshot 的模型/角色一致，120 冻结文件通过。实际启动后再核查真实 initial snapshot 和工具读取；输入已投递、工具真正读取、推断正确、代码有效、GPU质量和性能接受分别记录。完整文件可访问，不作隔离或绝对无污染承诺。

输出保持原样。若它没有读索引、错误解释原始数据或生成不可执行方案，保留失败，不重复到通过。若代码和参数适合后续验证，再另行声明固定执行协议；本批不自动采用 production 角色、collector 或接受新版本，也不把人工选定证据入口包装成无人干预端到端。

目录 `/root/autodl-tmp/dev/v6-work/diagnostic-index-candidate-20261029`。启动前核验 GPU 空闲和真实150GiB容器 df，预计只生成小型会话/CPU产物；raw、快照私密配置留服务器，GitHub仅代码、哈希和小型摘要。保留所有历史接受/拒绝及手工源映射、FSDP source-owner 修复边界。新批配置一个普通完成/8GiB磁盘监听，无Goal、模型等待轮询或投递测试。
