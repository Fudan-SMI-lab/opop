# DDP 原始失败反馈的一轮普通续行

已于2026-10-30 10:23:07 UTC（香港18:23）实际启动，冻结`6c6809926168a000c166e106770a7f05686553d1`，controller125768。唯一隐藏完成/磁盘监听12400于10:24:49启动，旧44088保持handled。实际本地OpenCode start/health/stop通过，没有模型测试prompt。启动可用10,983,202,816B，10:25:34审计时10,932,756,480B约10.18GiB、GPU空闲；尚无GPU job，处于真实selector阶段。

[启动核验](evidence/v6-ddp-communication-feedback-20261030/startup-verification.json) SHA `a8e95bedbdf96c31847f107db2c36824dc1935882adc7b5857e7675a1e7e0b15`：generation46c388e29191/session ses_e73aa1d4effeb9UPgg21bO3JD3，32初始文件20,980,742B逐字相等，312冻结hash与source/current/request/prepared/discovery/core通过。真实feedback518,850B/SHA f7489f9ef80206bc4acc1f8385a6311a64773ee43cf96c579c76038c000c5a7e等于实际registry route，supplemental=null，历史cf5e代码和两个打包after artifact一致。

对比上轮无反馈selector，原25输入不变、新增7个反馈/历史文件；模型和工具设置相同，但**prompt不相同**：现有`ProjectAgents.generate`在有feedback时原本追加“Read inputs/previous-feedback.txt and repair the concrete failure.”，不是本轮修改生产prompt。最初只读审计错误断言prompt相等，保留[startup-audit-attempt-1](evidence/v6-ddp-communication-feedback-20261030/startup-audit-attempt-1.json)，修正reader后核验，无模型/实验重启或输入改动。

早期20 assistant/22工具已留私有history，SHA `5671523fecfbfbd9e4f5993eed8b16912dcdf75995441d50a4296bdebdcf2fa4`。工具10–14真实读取反馈片段/参数搜索/worker结果，17读取旧candidate源码；截断输出不证明所有失败字段或样本均已读、更不证明推理正确。工具3缺失source目录的stderr被抑制后后续命令成功，compound exit0不能逐命令视为通过。[早期读取记录](evidence/v6-ddp-communication-feedback-20261030/startup-tool-review.json)只核对这些输出证据及22工具命令/metadata，全部输出、后续访问、CPU/GPU probes与污染审查留终态完整审计。真实活动/输入交付不等于新候选质量或接受。

用户于2026-10-30香港18:12追问结果分析与自动推进。上轮DDP已完整审计，但主Agent结束后没有启动新controller/监听，服务器确实空闲约7小时；不是实验仍在运行或结果尚未到达。此次继续通信研究，声明一个新的有界步骤，不重启旧轮次、不重复固定旧候选到通过。

新工作根`/root/autodl-tmp/dev/v6-work/ddp-communication-feedback-20261030`。现有CLI `resume --rounds 1`、原scope和默认参数探索/API observe-only；普通selector/candidate依据原始framework failure自行决定范围/实现。源为已审计的`ddp-communication-trial-20261030/ddp-search`，空current `e925da5fa07cb2c7ac12e0c19e1c297b62ee28feaf39fdf4494a00cfe4f31d29`，prepared `7a49c637…`。复用已完成当前设备discovery、共享只读controls/contract，初始不再GPU rediscovery、不重跑源/controls、不复制约4GB参考。源current/历史Graph/current/FSDP拒绝/Qwen关闭保持。

输入为trial `df2eb7acd2`的**原始未修改**latest-feedback，2,855,351B、SHA `3e2ddf153aeb14cb79861d0abf413d75147164e5fc754f039e35cd637c38f5ab`，与原trial result逐字相等；`supplemental_evidence`未添加。历史candidate `cf5e5fa7572e159a2a03828ee83bbbded6f730dd68d5a8488d46327c0af15d05`通过既有historical artifact接口可读，未安装。包含原质量失败、路由、原run-only sidecars和已有timeline；不提供开发者terminal audit、后来的control-loss读取结果、状态重置因果表、手写修复或旧接受实现。原源码和诊断/源映射辅助边界仍在，不称无辅助原入口。

前置CPU核对`feedback_artifacts`及实际registry/plan route、历史代码hash和冻结依赖。此检查只证明接口保留；启动后必须逐字核验实际snapshot、模型/工具/提示/core、真实读取和完整工具输出。新selection会改变candidate输入，不是单变量A/B。外部FS非隔离，审核实际访问路径和内容，不作绝对无额外访问声明。所有角色失败、正常schema repairs、CPU/GPU probes、参数和尾样本保留，pipeline exit0不代表子命令成功。

完整DDP工作保持：GPT2 12层hidden768/12head/vocab50257、两rank B1/S128/accum4、有效8序列1024tokens；FP32/eager/noTF32/dropout0，完整zero_grad/FW/BW/loss÷4/SGD .001，前三microbatch no_sync、最后backward同步。两连续全梯度/参数/loss/finite与exact replica检查、原1e-6/1e-4、2warmup/5clean/两同步/状态owner不变。原当前锚133.379621ms、1.05上限140.04860205ms不重设，逐样本max rank再median/MAD。当前GPU为20e8003c/a24af2e6；没有新显存接受约束或跨设备增量比较。

主收益仍是完整step；现有通信诊断映射独立3次profile/rank在clean之后，planner可自主选既有collector。NCCL sum不等于wire/关键路径，NODE/P2P仅能力，实际transport需worker日志；不能用不同period相减推host成本。只有真正新接受版本，审计完整质量/原约束/配对噪声/实际通信机制后，才另声明独立整版B/C/C/B及必要原生证据；controller完成不等于接受。此controller无独立或专门native阶段，不自动无限追加候选、重开Qwen/FSDP拒绝或改生产prompt/sampler/allocator/acceptance。

当前实查GPU空闲，150GiB盘余10,992,791,552B约10.24GiB，满足8GiB告警加2GiB本轮预算；不清理当前数据、不复制reference。新长批次仅一个显式隐藏完成/磁盘监听；旧44088已handled，任何旧批次/监听不重启。低空间不是完成，不终止实验或盲删；若需维护只按既有授权在GPU退出后处理明确冷文件，保护source/controls/accepted/models/provider/private history。所有SSH/后台/通知子进程CREATE_NO_WINDOW，不关闭VSCode终端。raw/private材料留服务器，GitHub只存代码、hash和小型摘要。

上一轮终态审计`70b28f9d3d09ed8078461a325a40e0ce021a5c00e3b13d12e4bb5fff467746bf`与全部拒绝保持。新轮结果另列；完成通知应接续结果审计和下一证据步骤的判断，不能把“本轮停止”误写为整个通信研究已停止，也不能无证据重测至通过。实际启动及snapshot审计随后追加。
