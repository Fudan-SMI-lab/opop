# DDP 原始拒绝反馈的一次普通选择决策

后续状态：对应[真实candidate生成](v6-ddp-communication-candidate-20261030.md)已终态审计，代码890659f7/schema修复后替换旧bundle却误认为reference保留Graph；一次CPU取证发现异常返回/重复SUM/错误传播反例。没有GPU或新接受。原selection及其未证实假设保持；下一步仅一次具体接口修复生成，不重新选择组或手改实现。

本批于2026-10-30 13:04:38UTC（香港21:04）以`batch_finished`结束，已完成终态审计：SHA `816bff94f0b660f3a96e388d2a0484c6e428453394e3156fc551054fde280ec2`，小型[公开审计](evidence/v6-ddp-communication-selection-20261030/public-terminal-audit.json) SHA `70f5a107e64d72e7acfd5894a193325743acbc078a9d7d0308354d2c80a8b129`。886冻结文件、37初始文件27,168,387B、实际registry route、source/current/prompt/model/tools/core均核验；控制器已退出，两卡空闲，实际150GiB盘余9,092,583,424B（约8.47GiB）。旧监听7584已handled，不重启；本轮未清理。

真实selector共33 assistant、43工具、0 schema修复，provider估计0.84558406612（非账单）。完整工具输入/输出、内嵌CPU脚本与session parts保存在服务器；1632行read输出逐行核验，其中1条明确截断只验证前缀。未观察到模型/GPU/collective执行，仅JSON/文本处理及已安装PyTorch默认allreduce hook的源码检查；无外部后验答案读取记录，但文件系统并非hermetic。Tool3错误source目录exit2；tool15同类错误被后续pipeline exit0掩盖，tool12 grep无匹配不能按整体exit0算通过，均保留。审计器第一次将文件路径传给要求目录的validator导致NotADirectoryError，保存`terminal-audit-attempt-1.json`后仅修正读取器，没有角色/实验重跑。

`selection.json`仅tool42写入一次，无编辑；从该脚本AST字面量与实际calls重建字节完全相等，SHA `cd3f9b53abd848b891b70743c6195bf3383733b2d867d2345abab945fdd6a7e7`。自主选择execution_plan、项目整步组及两个AllReduce组，1055 affected calls覆盖当前registry。假设为多communicator并发、末桶拆分和按桶提前更新，属于可形成候选的通信调度方向，尚无实现/正确性/收益证据。

不能采纳整份因果说明：它把前轮72→108/111ms变慢当作计算饱和证据，但实际前轮在`params.index(s)`初始化异常后运行原始eager，拟议并发Graph根本未到达。Tool26虽打开原诊断，400字符rank前缀止于异常内容之前；没有读取失败candidate.py或当前已安装优化代码的记录。它也把明确注明“不是lower bound”的合成校准称作硬下界，将独立profile中的NCCL区间与clean整步组合预测58–64ms；这些不能证明可回收关键路径、唯一剩余优化方向或多通信组吞吐。原生RING_LL入口、tuning日志中的SIMPLE/2 channels及synthetic logical rate不是逐step物理transport/带宽因果证明。默认hook除法后SUM的源码不保证新分组/拆块的数值等价；提前更新仍需依赖/生命周期审查，回退中性和小于1ms开销也未验证。原输出保留、不手改、不向后续角色提供此审计纠正表。

下一步明确为[一次既有candidate角色生成](v6-ddp-communication-candidate-20261030.md)：沿用此真实选择、原始失败反馈、当前accepted source/registry，正常API做affected_calls排序/闭包与selected-group路由。只形成可审查代码，控制器没有GPU、参数执行、搜索、collector或验收阶段；代码/参数/真实工具记录审查后再决定是否值得另行固定执行。不是新selector/reviewer/prompt重试，也不是认可预测收益。Qwen关闭、FSDP拒绝及原DDP accepted指针保持。

以下为启动时历史。

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
