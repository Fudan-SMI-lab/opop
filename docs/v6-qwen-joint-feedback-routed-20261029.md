# 用现有补充证据接口执行一次修正试验

前一批[qwen-joint-feedback](v6-qwen-joint-feedback-20261029.md)确认实验顶层字段被现有route省略，ce85新候选仍质量失败。原失败、完整会话与所有旧结果冻结。本批从同一个原试验前状态另建分支，将**相同**事实放入已经存在的`supplemental_evidence`；不修改生产路由/提示、不添加优化实现或根因解释。

起点仍为`qwen-constraint-continuation-20261029/qwen-continuation`的b61996e1空bundle、a866历史候选以及原反馈SHA`d426e3df`，与前一批一致。原5,211,596字节/所有字段保持，只追加`{supplemental_evidence: {fixed_continuous_full_output_diagnostic: 原事实}}`，新输入5,213,974字节，增加2,378字节（前一批顶层增加2,251）。内层事实与上一批`supplemental-facts.json`严格相等，没有加入ce85代码、ce85失败或后续开发者分析。

运行现有fork_run/CLI resume，一轮普通selector/candidate及默认参数探索。共享原controls/完整discovery/source/adapter/contract，未安装失败候选、不复制参考、不重复发现。保持B1/prompt256/32tokens、BF16/SDPA/noTF32、全量.02/.02/exacttokens、两sync/生命期、原TTFT25.2536046约束。完整质量/约束/噪声/正式搜索接受分别审计；出现新接受后，再单独安排完整独立AB/BA/native；本批没有独立接受阶段。

启动前CPU预检不只检查原始payload：分别经过真实feedback_artifacts和selector/candidate带registry/plan的route，验证只有supplemental_evidence变化且事实保留，原历史代码可读取。实际启动后须再比较初始snapshot，检查事实确实进入角色输入及真实工具读取；因下游selector输出可能变化，不能事先宣称整个candidate角色是单变量A/B，更不能把单次角色差异说成稳定因果优越。即使新候选失败，也保留而不重复直到通过。

控制器沿用修正后的`scripts/experiments/v6_qwen_joint_feedback_20261029.py`，新增`--prior-attempt`用于验证前一终态与事实相等；旧adde9525冻结controller保持。新目录`/root/autodl-tmp/dev/v6-work/qwen-joint-feedback-routed-20261029`。生产core ed24af7a不变；真实API仅统计不限额。实际df约10.32GiB，仅一轮串行，未清理/转存raw。普通完成/8GiB监听只保留一个；旧监听handled，未启用Goal/模型等待轮询/投递测试。全部历史接受/拒绝/样本保留。
