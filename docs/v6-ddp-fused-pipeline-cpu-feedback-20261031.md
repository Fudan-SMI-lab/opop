# DDP 流水候选的一次CPU事实辅助纠正

已于2026-10-30 21:27:01UTC（香港10月31日05:27:01）实际启动，controller `222432`，冻结 `b0012b65c9afdfed215ddc2ccd55916c49721344`，唯一隐藏plain watcher `31676`。不是仅声明后等待：21:28:58UTC实际启动核验SHA `630dd39011a6ef332322af6e504b494da8f987cda7d1aa91a5007023ce6c0c84`，generation `7e29e7027b24` / session `ses_e714a4bfeffeCNBl9mzP7YT0dY`；37文件8,682,320B逐字一致、1489冻结及原prompt/model/tools/core/current10992保持。真实route18,663B/04595a2c…原字段不改，只有精确CPU事实supplement，不增加修复prompt段落。见[启动记录](evidence/v6-ddp-fused-pipeline-cpu-feedback-20261031/startup-verification.json)和[早期读取核验](evidence/v6-ddp-fused-pipeline-cpu-feedback-20261031/startup-tool-review.json)。

早期7 assistant/15工具已核对2487个编号read行、5个明确截断前缀，review SHA `c7a93523b2574f0bfe12b782b2a353f905c1ec1170d09da494700df0129a0225`。工具5完整688行反馈内的CPU事实、工具14完整501行旧eef828代码实际读取；选择rationale/hypothesis和旧proposal部分长行截断。前15工具尚未单独打开CPU结果/index/fixture或当前collector JSON，但不能据此称CPU事实未读：它们已完整内联在反馈。工具7错误adapter根路径和工具12缺失previous-candidate目录/exit2原样保留；8/12只列目录，opencode.json仅文件名、未读内容。前15无模型/tensor/CUDA/processgroup/collective或私配置内容命令，限早期记录，外部FS非hermetic，最终访问与真实探测仍待完整审核；读取不等于正确理解或质量通过。

启动实核GPU空闲/无jobs、controller存活，150GiB盘余9,888,419,840B约9.21GiB，未清理。真实LocalOpenCode start/health/stop、server_url null通过，无模型预检。旧36164已handled/退出。冻结控制器和launch范围句误写“original33 nonfeedback”，实际是原33总输入中的32份非反馈保持；此处与startup仅澄清数量措辞，不修改冻结控制器、实验输入或原错误。当前仅此一次事实辅助纠正，生成结束须审精确代码与实际工具，再作有界下一决策；不自动GPU或修到通过。

本步骤的依据是前一原样eef828代码上的一次CPU合同取证，不是新的GPU质量拒绝或性能测量。前置完整审计`87cddc2c69cb2287cafd9b566a18badb9eeec0883fc6b01c47c925f108cdc137`已完成；保留其生成输出，暂不GPU。下一有界工作只调用一次既有`ProjectAgents.generate(candidate)`，普通生产candidate prompt、既有feedback suffix、model/tools/validator/正常schema repair、API observe-only/no caps不变，**不增加实验修复prompt段落**。

新root `/root/autodl-tmp/dev/v6-work/ddp-fused-pipeline-cpu-feedback-20261031`。输入基于真实generation076246cfc985的33文件：原32份非反馈逐字不变，原路由字段全保持，只附加supplemental_evidence内精确CPU结果、索引和实验范围；另加前候选/proposal/CPU结果/index4份文件，预期37份8,682,320B，1489冻结。源current10992/508a单bundle及原accepted3cd9/formalbbe28c/1fc/4541/independent7a49/historical09b7全部保持，旧物理ccd90文件不当active bundle。选择原7call/3group/rationale/hypothesis、task/source/adapter/registry/controls不改。

原框架完整版本反馈82,833B/`bbc009e9…`不动，增强容器SHA`28f3809e51caa196934ce9ecab946ea983a04b3c2cfbb8ac785452d940448b83`，actual selected-group route18,663B/`04595a2c17e913963f0bf53fc9f40807322e26c222dc15b0a2af6bad43529742`。原accepted decision/measurement/complete_version不改，不伪造eef828搜索或GPU拒绝。前候选20,997B/eef828与proposal12,510B/0f0a为历史输出，不安装；新CPU result0c40b6ef…与实际fixture42a25312…仅条件依赖记录。index指向精确原执行fixture/result/execution/已安装API源码，不指向整个终审，也不给手写修复。夹具中的惯性Graph/stream/Work和小CPU模型不移植到候选。明确DEVELOPER CPU-FACT assistance，非unaided/单变量AB，原source/metric辅助仍保持。

控制器无GPU/model/collective worker/search/parameter/collector/native/source-control/referencecopy/pointer/acceptance阶段；实际Agent工具仍完整审核，不能从意图推断无探测。由候选自行处理证据，不指定group/bucket/算法/代码；生成后审精确字节、参数、依赖、真实测试与错误再决定后续。仅此一次有界纠正，若仍有具体缺陷保留失败，不自动修到通过；Qwen、最新FSDP拒绝、77ac、observer机制链保持关闭。

原DDP任务、2连续全梯度/updated params/loss/finite/exact replicas(1e-6/1e-4)、2warmup/5clean/先aligned rankmax再median/MAD、原133.379621→140.04860205ms约束不变。旧508a独立53.91%属于计算合批+通信融合联合收益，非隔离通信或volume减少；NCCL sums不是wire/关键路径、不同profile不能减clean。没有新source比较或控制副本。

预检GPU空闲、150GiB盘余9,919,021,056B约9.24GiB，384MiB材料预测+8GiB警戒、不是API cap，无清理。实际启动另记，后续只设一个隐藏plain watcher；旧36164已handled且退出。low_disk_space不是完成，不终止活动实验/盲删；旧cold RESTORE/journals保留。原history/config/snapshots及完整CPU执行记录只在服务器，GitHub仅代码/hash/小摘要。禁止打印secret或metadata.output/display，禁止未经授权轮换配置。完成接回审计和有界决策，不静默结束研究。
