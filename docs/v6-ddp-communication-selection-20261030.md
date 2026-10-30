# DDP 原始拒绝反馈的一次普通选择决策

实际2026-10-30 12:53:29UTC（香港20:53）启动，冻结`07c6a21b01fb75450d7043fa1975bbb1eb99e405`，controller154975；唯一显式隐藏completion/disk监听7584，旧13432已handled，无实验重启。OpenCode本地真实start/health/stop通过，server_url=null，预检没有模型prompt。

12:56:22UTC启动核验SHA `eefdd61b9add9bd80f12666ef6afa8e58b7a830fc49958767f01130c534c0a56`：实际generation-2f2d7ab00b70/session `ses_e73206ce5ffek9yynLuM6ILXrs`，37初始文件27,168,387B与预期、实际snapshot逐字相等；886冻结身份不变。相对原selector10文件变化、5移除、1新增，当前任务/source/registry/common未变，非单变量AB。原feedback14,429,015B/a0234920保持；route10,252,521B SHA `305a2ce51d1f0c5c26938e118c6e0a87d1a4649a266feb3629f5761ecffbb5e2`，supplemental=null，历史失败5e3a不是当前installed ccd90。原prompt/model/tools/core相等。

真实早期10 assistant/16工具，私有history SHA `6c6ac3ead07be1ef560aa12c2e4cd330588297f7efab355055bbd57687eb7490`。Tool14实际读到拒绝原因、paired values、原constraint；tool16只打印历史proposal前6000字符，不称完整阅读。Tool3错误source路径exit2，tool15同类ls错误被后续pipeline exit0掩盖，均保留。早期审查SHA `96551e8e4cb87b920caea28641cec62cbc97855ebdf8f5449d17c05cdffb4a2c`，仅全命令/路径及14/16完整输出，其他为原输入身份和片段/元数据；最终完整访问与推理审计待终态。没有GPU工作或新质量/接受结论。

实际df总161,061,273,600B、余9,095,426,048B，两卡空闲。启动预测384MiB，警戒8GiB；未清理或复制source/reference。小型[启动记录](evidence/v6-ddp-communication-selection-20261030/startup-verification.json)与[早期工具审查](evidence/v6-ddp-communication-selection-20261030/startup-tool-review.json)公开，私有配置/history/snapshot文件留服务器。

以下为启动前协议。


已声明，启动身份另记。用户优先通信，保留通用项目优化目标；前一轮新候选在初始化异常后退回eager、paired拒绝，故不做独立验收或自动完整GPU轮次。当前仅调用现有`ProjectAgents.generate("selection")`一次，让原始失败反馈参与下一步范围/假设决策；不是新reviewer、prompt迭代或重复直到通过。

原source `ddp-accepted-communication-trial-20261030/ddp-search` current1fc24545，仍accepted代码ccd90/单bundle与当前真实discovery。从前一真实selector初始snapshot保留所有任务/source/contract/current observations/search policy，使用现有`feedback_artifacts/candidate_artifacts/route`将previous-feedback及历史候选/previous sidecars更新为新trial-02ed5436b4原始结果、5e3a代码及原始采集输出。没有追加开发者Tensor-index修复、审计因果表、planner错误清单、手选通信组/算法或实现。原始错误proposal/selection仍可经普通反馈出现，不手工纠正。

保持同一普通有feedback的prompt/model/tools/core，自动worker-environment/binding-contract；API observe-only无新caps。新旧多文件自然变化，不是单变量AB或无辅助cohort。原source/metric mapping辅助边界不变。Schema/范围覆盖使用现有check_selection，正常修复和错误保留；schema通过不代表推理正确。

本步仅生成selection.json。没有candidate、参数trial、GPU/model/collective worker、新discovery/controls/source验证/collector、独立验收、指针发布或生产更改。工具仍可访问非hermetic文件系统，终态须审计全部实际访问/输出/CPU或GPU活动，不能只按controller意图判断。未授权手工修复输出或自动执行下一候选；输出后评估假设是否回应原反馈且有通信相关证据，另行声明必要下一步。

启动前两卡空闲、实际150GiB余约8.54GiB；不够完整普通轮次1GiB预测+8GiB警戒。小型selector输入/快照/会话按384MiB增长预测检查，非API用量cap；不清理、不复制references、无并行GPU。冻结旧全部依赖及新增worker/trial/snapshot/审计身份；私有provider/history/snapshot仅服务器。旧13432 handled；新长工作只设一个显式隐藏completion/disk监听，low_disk_space不是completion，不停实验/盲删/重复不变警报。

Qwen链保持关闭，FSDP最新拒绝不重复；旧DDP计算Graph独立36.1578233%不称通信接受。后续仍要求完整质量、原约束、独立整步收益和真实通信机制证据；NCCL sum/不同profile差值不是wire或关键路径。
