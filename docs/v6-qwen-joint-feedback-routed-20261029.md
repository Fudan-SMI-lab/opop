# 用现有补充证据接口执行一次修正试验

2026-10-29 00:02:54 UTC终态：本轮生成新候选并获得**搜索接受**，独立整版验收尚未执行。生产core/prompt/sampler/allocator/接受规则均未改。完整服务器审计SHA `ed65189b46d0c110451cd916233282459a15c31067a25a7b707010bfa3465538`；[公开小证据](evidence/v6-qwen-joint-feedback-routed-20261029/public-audit.json)保留全部搜索样本、质量结果、角色输入索引和原生核查，raw会话/trace仅在服务器。

实际selector与上一批29份初始输入仅previous-feedback变化；prompt/model/tool/core相同。candidate初始输入则另有selection.json和target-resources.json变化，因此整个candidate角色不是同快照单变量A/B。selector及candidate均实际sed读到原事实字段，candidate读工具`prt_184e1b6ab001d3Q4SRl6j55PwF`，并在方案中区分第一次teacher通过、第二次teacher错误。三角色均0schema修复，分别37/37/12工具；provider估算费用合计2.50542406686，仅统计。selector曾出现timeline容器类型错误；candidate两个CPU假cache探针分别因测试cache长度、比较shape出错，第三次修正探针通过。所有原命令/输出保留，不把CPU探针当完整GPU质量。

新trial `9523ac3f5f`，原样代码SHA `d5bd32d47d6b4d8165f617810a4dbeabed5c5c850932e6f74aaafd9adaa4bf2b`，参数空间为空。它为prefill/decode共同生成一个bundle，每次返回重建cache/layer对象，按当前cache长度/值刷新prefix，重置请求状态，分开prefill池并加入连续greedy/teacher自检。这里同时改变多项实现，不能唯一归因到镜像复用或证明事实字段有稳定因果优势。自检直接调用replacement，没有经过framework signature分派；本次质量结论来自真正evaluator通过，不能从自检代码反推。

| 阶段 | B完整生成ms | C完整生成ms | C TTFT ms | 结果 |
|---|---:|---:|---:|---|
| 单参数screen | — | 389.643375 | 21.512766 | 质量/覆盖/原约束通过 |
| AB | 704.489814 | 389.825517 | 21.581847 | 配对增益314.664297，高于噪声3.402914 |
| BA | 687.541333 | 389.857801 | 21.591105 | 配对增益297.683532，高于噪声2.256510 |

全部5端点25样本、10次连续检查zero error，完整32tokens/31decode、BF16/SDPA/noTF32及原始TTFT上限25.2536046保持。中位汇总696.0155735→389.841659 ms，下降43.9895207%。接受版本`a3766b5026d1e6eb137962285bd5300e2e3f579ea09069f5ad18da43cb4771de`，唯一bundle `trial-9523ac3f5f-param-ae0ac4b9`；原始source分支b61996e1保持空bundle。只有一轮，无接受后重发现或多个active bundle共存证明。生命周期peak 8,188,023,808→8,322,147,840B，增加134,124,032B；不是显存优化。

真实planner还自主调用Torch、Nsight、timeline、candidate_diagnostics及binding_cost。13个worker全部执行完成。当前candidate原始SQLite SHA `85ffae22bf3392600dbe5b7c55e99eff49a69f0004fea0160185f15a7962f9f4`逐行重算与normalized相同：68,745 kernels，其中68,712 Graph kernels；32次cudaGraphLaunch，prefill 2,217、31次decode各2,145；2,304 Flash kernels全部在Graph内。这是当前代码的原生证明，不借用旧f12路径。独立profile不作为计分或critical-path归因。

timeline原始trace重算union/gap/clock一致，但不能将其与clean wall相减当host成本。另4个binding端点20样本/8质量检查通过，两个parent/passthrough配对得到负耗时差；这是跨worker波动，不是负的物理绑定开销，也不能归因到具体组。planner把双质量步骤的62次decode描述成单工作单元、以及引用旧Torch覆盖推断新路径的表述均不采纳。run-only candidate_diagnostics并不执行observe/compare。global capture patch、任意并发组合和任意全局owner仍未被一般性证明安全。

审计脚本初次把accepted decision误当仅含accepted/reasons的字典，以及将缺pid的原rank文件传给摘要helper，分别触发审计断言/KeyError；修正只涉及外部审计读取，未重跑模型或改实验结果。重新核查全部materialized source/entry/request/core、43个冻结文件及原source/controls哈希通过。此前手工外部generated_tokens/step_logits映射边界仍存在，本轮是开发事实辅助的续行，不是未经人工介入的原入口成功。

GPU全部退出后，00:12:50–00:13:13 UTC无损压缩原入口旧失败批次的明确worker RUN JSON：11个payload，1,683,716,516→130,199,414B，gzip/解压SHA和126个其他文件逐项验证。实际df净释放1,553,403,904B（1.45GiB），余10,876,112,896B（10.13GiB）。10个当前版本source/bundle及6份完整FSDP参考复核；没有本地raw转存。恢复入口`/root/autodl-tmp/dev/storage-audit-20261029/qwen-routed-independent/run/RESTORE.md`及其journal，见[小清理回执](evidence/v6-qwen-joint-feedback-routed-20261029/storage-postcheck.json)。

下一步按[独立整版验证协议](v6-qwen-routed-independent-20261029.md)固定新版本做一次AB/BA和源检查，复用当前candidate原生trace，仅补原baseline trace。独立结论另存，保留全部旧失败；不修改生产框架、不再生成候选或直到通过才停止。

前一批[qwen-joint-feedback](v6-qwen-joint-feedback-20261029.md)确认实验顶层字段被现有route省略，ce85新候选仍质量失败。原失败、完整会话与所有旧结果冻结。本批从同一个原试验前状态另建分支，将**相同**事实放入已经存在的`supplemental_evidence`；不修改生产路由/提示、不添加优化实现或根因解释。

起点仍为`qwen-constraint-continuation-20261029/qwen-continuation`的b61996e1空bundle、a866历史候选以及原反馈SHA`d426e3df`，与前一批一致。原5,211,596字节/所有字段保持，只追加`{supplemental_evidence: {fixed_continuous_full_output_diagnostic: 原事实}}`，新输入5,213,974字节，增加2,378字节（前一批顶层增加2,251）。内层事实与上一批`supplemental-facts.json`严格相等，没有加入ce85代码、ce85失败或后续开发者分析。

运行现有fork_run/CLI resume，一轮普通selector/candidate及默认参数探索。共享原controls/完整discovery/source/adapter/contract，未安装失败候选、不复制参考、不重复发现。保持B1/prompt256/32tokens、BF16/SDPA/noTF32、全量.02/.02/exacttokens、两sync/生命期、原TTFT25.2536046约束。完整质量/约束/噪声/正式搜索接受分别审计；出现新接受后，再单独安排完整独立AB/BA/native；本批没有独立接受阶段。

启动前CPU预检不只检查原始payload：分别经过真实feedback_artifacts和selector/candidate带registry/plan的route，验证只有supplemental_evidence变化且事实保留，原历史代码可读取。实际启动后须再比较初始snapshot，检查事实确实进入角色输入及真实工具读取；因下游selector输出可能变化，不能事先宣称整个candidate角色是单变量A/B，更不能把单次角色差异说成稳定因果优越。即使新候选失败，也保留而不重复直到通过。

控制器沿用修正后的`scripts/experiments/v6_qwen_joint_feedback_20261029.py`，新增`--prior-attempt`用于验证前一终态与事实相等；旧adde9525冻结controller保持。新目录`/root/autodl-tmp/dev/v6-work/qwen-joint-feedback-routed-20261029`。生产core ed24af7a不变；真实API仅统计不限额。实际df约10.32GiB，仅一轮串行，未清理/转存raw。普通完成/8GiB监听只保留一个；旧监听handled，未启用Goal/模型等待轮询/投递测试。全部历史接受/拒绝/样本保留。

2026-10-28 23:28:52 UTC实际启动，冻结`65402c5afd74ce17b2ce4b45b9cc674a98c644b3`，controller PID916254；OpenCode真实健康预检通过，原controls/discovery哈希共享且无新GPU发现。唯一普通完成/磁盘监听PID49084已运行。首次监听预检的remote-root误指向已结束旧目录，断言阻止启动/投递；修正为新目录后才启动，未重启实验或发送重复通知。旧监听已handled。

23:29:52核查真实初始selection snapshot`generation-ac728bfec604`（session`ses_e7b277aa4ffe3zIbNqeR5HeKdd`）：与旧`generation-dac42ec386f8`的29份输入相比，**只有inputs/previous-feedback.txt不同**；prompt、model/tool settings、core相等。实际路由文件5,735,456→5,737,900字节，内部只有supplemental_evidence从null变为原事实，其余字段严格相同。旧a866候选代码SHA相同，ce85未作为输入加入。真实provider已有8条assistant记录；工具`prt_184d90920001zH8D0YHSMKDj4P`实际读取含新增事实的反馈片段，输出SHA`dbbb0f815bca434747bbc837e9c7164c61a024ba7740afb2e42076e83b2d9337`。这里只证明初始变量控制和实际读取，不预判最终选择/生成/质量/性能或因果优越。

该时刻controller正常、处于Agent阶段，尚无GPU worker；实际df可用10,876,497,920B（约10.13GiB）。启动与初始输入核查见[小证据](evidence/v6-qwen-joint-feedback-routed-20261029/startup-verification.json)。后续完整候选快照/工具记录和所有实际执行结果留待终态；不调用模型轮询等待。
