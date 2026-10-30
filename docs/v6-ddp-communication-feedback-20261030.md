# DDP 原始失败反馈的一轮普通续行

终态已完成并完整审计：2026-10-30 11:16:49 UTC的事件为`batch_finished`，不是磁盘告警。新trial `d06aa22e2d`以空参数接受完整版本`4541830b6c81c1c0c2c2fef6dfb5f2566e312483900238abf1e7ac900a1758e6`，候选SHA `ccd90dc158788a164d6fda242572530e9cdc9ca50563af1a1d9d597f0ac12e0b`，12,536B；proposal SHA `63e8661c273cd117a510d9d5e2006bb156a02ada7c3e5d073dde1b74b1767921`，单bundle `trial-d06aa22e2d-param-a72285d2`。原空source/current、历史接受版本和312冻结文件保持，生产core不变。旧监听12400已退出并标记handled。

完整step搜索结果118.214785→71.725802ms，收益39.3258618%。两组B/C为118.555851/71.824818与117.873719/71.626786，收益46.731033/46.246933ms均超过噪声7.696742/10.436134ms；原锚133.379621、上限140.04860205保持。5个正式端点、两rank共50个clean样本及20次完整零误差检查（各248,879,620元素）全部保留，所有尾样本不剔除。候选每rank首个质量调用走parent，第二个走真实Graph setup/capture，两者通过；不能称两次均为优化路径。全部12 jobs/24 rank正常完成，run-only诊断不替代质量检查。

30条原通信metric trace逐条复算，每rank仍13个AllReduce RING_LL。两份本候选Nsys分别SHA `6b5eecaa078e688d8aec14c127558002d88fcb9abb252576c748f7cfad0672e2`、`146542da81f11ffb5a7394984be8d9d91f14e5038334fa306684d029d4261f14`：各4103 kernels，其中2981在Graph、1122在Graph外，实际2次`cudaGraphLaunch_v10000`，全部13个NCCL均在Graph外。独立Torch profile/timeline各4106 kernels，逻辑AllReduce payload仍497,759,232B；真实channel日志SHM/direct/direct。**这证明计算调度机制发生变化，没有证明通信量减少；不归为通信优化成功。** NCCL sums、校准样本、timeline间隙和设备速率不是wire/关键路径，也不能跨profile相减推clean host成本。

代码Graph A覆盖前三个raw-model microbatch的清零/FW/BW，第4个microbatch保留eager DDP reducer，Graph B执行原SGD更新。setup位于第二次质量调用，不能以约12ms安装时长冒充完整capture成本。候选runtime peak1,498,647,552B、lifecycle peak1,999,242,752B较本轮配对B低，但lifecycle reserved3,193,962,496B高于B2,589,982,720B；仅描述，不新增显存约束。全局capture flag恢复到False、局部对象身份guard、失败后peer一致性和部分更新回退仍有限制，实际本次完整检查不能扩展为任意输入/并发owner或回退中性保证。

三角色全部真实snapshot、126个工具输入/输出记录、截断输出文件、CPU脚本和candidate初稿+4次编辑已审计并逐字重建最终代码。selector/candidate/planner分别39/50/15 assistant、41/66/19工具，0 schema repairs，provider估算合计4.22385320112（非发票）。实际读取原失败反馈/历史candidate、现有profile与安装库，以及官方PyTorch v2.13.0源码；未观察到后来的开发者audit/旧接受答案读取，但外部FS不隔离。候选2rank Gloo CPU玩具验证手工计划、bucket地址变化和连续更新，不是原完整CUDA/Graph或真实candidate框架执行；真实candidate CPU导入仅覆盖不支持设备的回退。缺失目录、KeyError、错误API、grep/sed错误及selector把ns按微秒打印却标ms均保留。工具清理了自己的CPU脚本/cache，原命令源码与完整私有history仍在，不能称全部临时raw保存。

数值审计SHA `b154e700ae58402cef4cfcb9ac5595d07fabf9f50eb663ac638e4fec0e324c9b`；完整终态审计`07fc3b7b3fa2f535204582bc95f7aa027724b4efa9bfdc8d2adfc71d4358d8df`；[公开小摘要](evidence/v6-ddp-communication-feedback-20261030/public-terminal-audit.json) SHA `ba54e7005db588647880c9f7f191c81e787135bab44b1a2d1fe5f7c3c32fbddd`。读取器首次漏掉相对candidate写入路径、首次native查询漏版本后缀/查询不存在DRIVER表，均保留初版/错误后仅修reader，没有模型/GPU/Nsys重跑。完整access review SHA `606ff401ec0f21e29edfa0673098ce390a20bc65630bf95c22d88ba4b11c6b8e`，raw/private留服务器。

因确有新搜索接受，下一步仅[一次独立整版B/C/C/B](v6-ddp-feedback-independent-20261030.md)，原字节、publish=False、原prepared基线与约束，复用精确当前native，不新搜索或重采Nsys。搜索接受与独立结论分列；即使独立通过，也不能将计算Graph收益自动列为通信接受。以下启动描述保留为历史。

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
