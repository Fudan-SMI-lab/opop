# Qwen完整输出事实反馈的真实候选角色对照

[固定关系诊断](v6-qwen-relation-diagnostic-20261029.md)证实此前候选推理缺少一项关键观察：greedy已经偏离源，teacher本身与源相同。此实验检验最小反馈补充是否被读取、是否纠正前提并影响生成；不指定底层原因或优化实现。

复用原第三候选初始快照`generation-64bcd2faad41`，并保留其原selection和前一失败候选反馈。Control精确重放原输入；treatment仅在`inputs/previous-feedback.txt`末尾增加一个JSON成员`fixed_full_output_relation_diagnostic`。原文件5,292,135字节，仅增加1,979字节，所有原字段及其值保留；其他32个输入文件、prompt、model/tool配置、生产core完全一致。初始输入在两次真实会话创建时再次核对。外部工具可访问的服务器文件系统不是封闭快照，不能把此实验称为严格隔离的随机试验。

增加内容仅包含已核验的完整输出关系、U/O原错误复现、比较方向/设备限制、observer影响和现有candidate_diagnostics不调用observe/compare的接口范围，以及小型审计路径/哈希。没有手工候选、开发者优化答案、成功代码或新宽泛上下文。尤其不宣称shared pool、alias、cache或算术是唯一原因。这也不是重开已经关闭的accepted-feedback提示词措辞研究，prompt完全不变。

按control、treatment顺序各做一次真实candidate角色，允许现有正常schema repair，保留工具调用、初始快照、输出/错误和费用统计，不设新使用限制。生成后审查实际读取了哪些字段、是否仍误认为teacher-only失败、是否把执行诊断当质量，以及完整工作/质量约束是否保留、代码实际改变了什么。单次生成有随机性，不能把一次差异认定为稳定因果收益。不得反复重跑直到通过。

这一批只进行角色生成和schema验证，不进行新的自主搜索或正式性能接受。Agent可能自主调用工具做探针；这些必须按实际范围记录，不当作独立GPU全质量认证。生成代码需下一阶段审查后才决定是否执行；不自动采纳任何生产修改。

控制器：`scripts/experiments/v6_qwen_relation_feedback_replay_20261029.py`。服务器目录：`/root/autodl-tmp/dev/v6-work/qwen-relation-feedback-replay-20261029`。生产核心仍为`ed24af7acc8caa15578e0b859c85ef8a901be51354448ed39094606ed9e10a2f`。真实启动前完成snapshot哈希、单文件差异、旧JSON字段保留和CLI导入预检，并检查本地OpenCode服务健康。

实际数据盘约13.06GiB可用，当前无需清理。所有raw/私有配置留服务器；GitHub仅代码、小型审计和脱敏元数据。启动后只保留一个普通完成/8GiB磁盘监听，无Goal、模型等待轮询或投递测试。历史接受/拒绝均不改变。
