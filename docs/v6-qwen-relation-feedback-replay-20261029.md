# Qwen完整输出事实反馈的真实候选角色对照

[固定关系诊断](v6-qwen-relation-diagnostic-20261029.md)证实此前候选推理缺少一项关键观察：greedy已经偏离源，teacher本身与源相同。此实验检验最小反馈补充是否被读取、是否纠正前提并影响生成；不指定底层原因或优化实现。

复用原第三候选初始快照`generation-64bcd2faad41`，并保留其原selection和前一失败候选反馈。Control精确重放原输入；treatment仅在`inputs/previous-feedback.txt`末尾增加一个JSON成员`fixed_full_output_relation_diagnostic`。原文件5,292,135字节，仅增加1,979字节，所有原字段及其值保留；其他32个输入文件、prompt、model/tool配置、生产core完全一致。初始输入在两次真实会话创建时再次核对。外部工具可访问的服务器文件系统不是封闭快照，不能把此实验称为严格隔离的随机试验。

增加内容仅包含已核验的完整输出关系、U/O原错误复现、比较方向/设备限制、observer影响和现有candidate_diagnostics不调用observe/compare的接口范围，以及小型审计路径/哈希。没有手工候选、开发者优化答案、成功代码或新宽泛上下文。尤其不宣称shared pool、alias、cache或算术是唯一原因。这也不是重开已经关闭的accepted-feedback提示词措辞研究，prompt完全不变。

按control、treatment顺序各做一次真实candidate角色，允许现有正常schema repair，保留工具调用、初始快照、输出/错误和费用统计，不设新使用限制。生成后审查实际读取了哪些字段、是否仍误认为teacher-only失败、是否把执行诊断当质量，以及完整工作/质量约束是否保留、代码实际改变了什么。单次生成有随机性，不能把一次差异认定为稳定因果收益。不得反复重跑直到通过。

这一批只进行角色生成和schema验证，不进行新的自主搜索或正式性能接受。Agent可能自主调用工具做探针；这些必须按实际范围记录，不当作独立GPU全质量认证。生成代码需下一阶段审查后才决定是否执行；不自动采纳任何生产修改。

控制器：`scripts/experiments/v6_qwen_relation_feedback_replay_20261029.py`。服务器目录：`/root/autodl-tmp/dev/v6-work/qwen-relation-feedback-replay-20261029`。生产核心仍为`ed24af7acc8caa15578e0b859c85ef8a901be51354448ed39094606ed9e10a2f`。真实启动前完成snapshot哈希、单文件差异、旧JSON字段保留和CLI导入预检，并检查本地OpenCode服务健康。

实际数据盘约13.06GiB可用，当前无需清理。所有raw/私有配置留服务器；GitHub仅代码、小型审计和脱敏元数据。启动后只保留一个普通完成/8GiB磁盘监听，无Goal、模型等待轮询或投递测试。历史接受/拒绝均不改变。


已冻结`826a757e0e50bdf5aba3ff2a1e013cbd5ad033f6`，2026-10-28 20:22:59 UTC启动独立session控制器PID859520，实际OpenCode健康检查通过。见[启动](evidence/v6-qwen-relation-feedback-replay-20261029/launch.json)、[精确输入协议](evidence/v6-qwen-relation-feedback-replay-20261029/protocol.json)、[预检](evidence/v6-qwen-relation-feedback-replay-20261029/preflight.json)。

20:24:32 UTC[实际启动核验](evidence/v6-qwen-relation-feedback-replay-20261029/startup-verification.json)：control真实快照`generation-6673f453e913`全部33输入与原快照相同，prompt/model/tools/core一致；真实provider已有6条assistant记录。当前control进行中，尚未得到两臂最终结果，不提前推断反馈有效或代码正确。GPU暂空闲；实际df剩13,723,664,384字节（12.78GiB）。

唯一普通完成/8GiB监听PID59096已核验waiting，见[监听设置](evidence/v6-qwen-relation-feedback-replay-20261029/wakeup-setup.json)和[实际状态](evidence/v6-qwen-relation-feedback-replay-20261029/wakeup-verification.json)。旧关系诊断监听已标记handled，不重启；等待期间只有普通进程检查，不调用模型轮询。到达终态后继续核查实际输入、工具读取和输出，再决定下一步。


## 终态审计（2026-10-28 21:04:13 UTC）

Control和treatment都完成真实candidate角色，各0次schema repair。重新加载实际初始快照并核对生成后的输入文件：33输入中只有原反馈增加1,979字节，其他32文件、prompt、model/tool配置、core一致，旧JSON字段和值全部保留。模型分别记录费用3.06461253976和3.5705464758（provider估计，非账单）；使用统计没有限额。见[实际输入/工具/输出审计](evidence/v6-qwen-relation-feedback-replay-20261029/terminal-audit.json)。

Treatment通过工具`prt_184415d8d0017jxOmX5QpyntIw`实际打印并读取新增完整事实和限制，最终rationale明确greedy从decode位置1先偏离源、teacher与源相同；没有另行读取所链接audit文件或raw张量。Control读了原失败反馈/候选代码，最终说明greedy与teacher不一致，但没有独立建立哪一侧匹配源。两组都采用原selection已经提出的single StaticCache Graph方向，因此这不是“补充事实才使Agent想到单Graph”的证据，也不能从一次随机生成推断稳定优越性。

两组均保留完整任务/质量/TTFT合同且parameter_space为空；并未执行真实完整GPU质量检查。Control代码SHA`777488e03148618b7a56f2c56302d1e99c5abced778f0f56129040adaf0f4931`，使用显式2D mask、首decode才分配StaticCache，正常步不更新返回DynamicCache的KV，guard fallback时才物化。Treatment代码SHA`313aa7bc7818fa72269b00134d91096102692fb33b56c0c3bb5c96060141d562`，让库的StaticCache路径生成mask，每步把调用方DynamicCache层重绑到真实静态KV的narrow views，使用长度/identity guard。其proposal声称build分配StaticCache，但实际build仅分配static_input，StaticCache在_setup里延迟分配；跨下一次refresh的view稳定性也不能由同一步内声明保证。Control的eliminated_calls为空，treatment声明23,225个；这只是生成的plan声明，不能替代实际路径证据。

Control记录74个工具、treatment89个，工具状态均completed；但输出内分别存在3/2次Traceback。Control早期toy测试token mismatch、shape错位、list传torch.equal失败，treatment出现NameError和计数断言失败；还有control较早mock输出range-overflow为False。后续修改的是Agent自建探针/自身候选，不是开发者运行中修补。全部早期内容和输出已保存在服务器audit session-parts，另保全7份最终临时探针文件。shell管道exit0只可能是tail退出状态，不能把这些失败写成通过。

实际探针为CPU随机小Qwen：control主要FP32、prompt7/new5，treatmentBF16、prompt12/new5；测试runner覆盖了_capture/_replay或_setup/_run_step，绕过真实CUDA Graph。因此零误差或0.0029296875 toy误差均不能证明完整Qwen3-4B BF16/32token GPU质量。双方仍把shared-pool原因和prefill不改保证TTFT说得过满；profile窗口/空隙不能提供新候选的因果成本或原生执行证明。没有据此修改生产prompt/collector或接受规则。

当前GPU空闲，实际数据盘剩13,383,286,784字节（12.46GiB）。下一步按[固定GPU验证协议](v6-qwen-feedback-fixed-validation-20261029.md)，原样执行两份生成代码各一组U/O，不修代码、不重开搜索、不重试到通过。原失败、旧同adapter收益及所有历史决定保持。
