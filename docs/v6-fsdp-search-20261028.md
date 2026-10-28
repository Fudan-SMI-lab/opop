# FSDP 三轮自主搜索与独立完整版本检查

前置[源与生命周期修复审计](v6-fsdp-owner-repair-20261028.md)已通过，修复后的 adapter SHA `8b014f6f…`。该 baseline 经开发者实测反馈辅助修复，后续搜索自主执行；本批不称为新的无辅助原入口 cohort。生产 core `ed24af7a…`、prompt、运行顺序与接受规则不变。

从 `fsdp-owner-repair-20261028/fsdp-repaired` 的空 bundle baseline 创建隔离分支；保留原 source/request/contract/质量/完整工作/显存<=1.05 baseline 约束。共享只读 controls 路径，不重新写多 GB 参考。源质量、修复代码和生命周期在本轮已完成独立核查，不把这些开发者解释作为搜索提示。

固定执行：

1. 框架重新 discovery、Agent 采集规划、目标选范围、候选生成与参数执行、完整质量/约束/配对判据、真实反馈，共三轮。没有人工给组、候选、通信优化答案或旧 bundle。检查实际路径和 fallback，接受后按现有机制重发现/组合，不预设必须生成某种 kernel 或必须获得接受。
2. 若存在 accepted current version，在另一隔离 baseline 分支执行完整版本独立 AB/BA；不改搜索接受结果，不因独立拒绝再抽样到通过。所有质量、覆盖、样本、原始显存约束原样保留。若无版本则明确 not_run。
3. 对存在的 current version 和 baseline 各执行一次独立 nsys 原生路径检查，不把 profiler 耗时纳入 clean 分数；即使独立拒绝，也可用它解释实际执行。通信指标仍通过原来四个显式次级映射分开采集，各自 max rank 后 median。四列不来自同一组 trace，不能交叉相减当物理分解；NCCL sum/union 不等于 wire time/暴露关键路径。

脚本 `scripts/experiments/v6_fsdp_search_20261028.py`，目录 `/root/autodl-tmp/dev/v6-work/fsdp-search-20261028`。启动前约 16.58 GiB，可用空间按实际容器 df 记录；GPU 串行、只运行这一任务，不开额外 Qwen 或大 trace 批次。原始数据留服务器，不本地转存、不清除冻结证据。共享参考避免新增第二套大文件，后续若空间不足需保留现场并基于真实占用处理。

旧 Qwen/ResNet/DDP/TP 的所有接受、拒绝与全部样本均不改变；不采用旧 allocator 合并或 reviewer 建议。真实 API 用量仅统计，不加开发调用配额。每阶段代码、协议与小摘要推送 GitHub v6；一个普通完成监听在终态提交续行，等待不调用模型，无 Goal 或额外投递测试。

启动：冻结 `aa5d497ec9dd54e5abf0daff1bc287bd14499390`，2026-10-28 04:42:52 UTC、controller PID523981；[launch](evidence/v6-fsdp-search-20261028/launch.json)、[protocol](evidence/v6-fsdp-search-20261028/protocol.json)。实际 df 可用 17,794,297,856 bytes（16.57 GiB）。已核实新分支 adapter SHA 不变、controls_path 与修复 baseline 完全相同，未生成另一套参考。唯一普通完成监听 PID33456、waiting，控制器存活，[启动核验](evidence/v6-fsdp-search-20261028/wakeup-setup.json)保存。当前只确认搜索流程启动，尚无本批性能结果。

2026-10-28 本机重启后，确认远端 controller 仍为 PID523981、PPID1，实验未重启；本地完成监听恢复为 PID24024。用户随后要求继续监听、完成后推进，并及时处理空间不足。07:00:03 UTC 实际容器 df 可用 17,599,234,048 bytes（约16.39 GiB），仍在 search 阶段，无整批终态。

同一个普通监听增加可选 `--disk-path /root/autodl-tmp --min-free-bytes 8589934592`，每30秒在原 SSH 检查中读取一次 `df -B1`，不使用可能报告宿主机余量的 statvfs。低于8 GiB只产生一次 `low_disk_space` attention 通知并退出，保持 `terminal=false`；这不是实验完成，不停止实验、不限制 Agent/API、不自动删除文件。真正完成事件优先，磁盘查询错误也不隐藏完成结果。已有终态/投递去重机制不变。原监听先正常退出再替换，配置和旧状态见[磁盘提醒部署记录](evidence/v6-fsdp-search-20261028/wakeup-disk-update.json)。

空间通知到达时先查实际依赖；运行期间只处理已核实无用的下载残片、临时垃圾或有保留内容的冗余副本，保护活动/冻结证据、引用、模型和 provider 状态，重型冷数据压缩留待 worker 结束。继续遵守服务器内清理授权，不转存本地、不重启旧归档控制器。处理后若实验未结束，只恢复一个监听，避免对未变化条件反复唤醒。默认未指定阈值的其他监听行为不变。本次只修改实验维护脚本，运行中的冻结框架与候选搜索不变。

验证采用隔离事件 fixture：未启用时不查磁盘、真实 df 输出字段路径、低空间仍非终态、阈值边界、完成优先、df错误不吞掉完成。另通过真实 SSH 检查当前 df/控制器，未投递额外测试消息、未启用 Goal 或模型轮询。

切换完成于07:09:12 UTC，旧监听 PID24024 正常退出，新监听 PID7952；07:10:13 UTC 已实际记录 controller running 和可用17,597,054,976 bytes。5项隔离检查通过。首次本地fixture直接执行Linux的 signal-0 存活探针，在Windows出现中断/非零退出；fixture随后将该远端专用调用替换为模拟存活，再运行退出码0。生产远端存活逻辑未改，本地真实监听采用SSH Linux检查，未发生实验重启。无磁盘警报或额外投递发生。

## 终态审计

本批于2026-10-28 09:59:18 UTC完成，用户重启后于10:05 UTC请求检查，当前真实用户轮次直接读取终态；旧监听最后一次检查为08:35:24 UTC，重启后未运行，因此没有发生本批自动投递。已将旧状态记为 handled_in_user_turn 并写 stop 标记，避免重新监听已处理批次。远端 controller/GPU worker 均已结束，实验未重启。见[恢复记录](evidence/v6-fsdp-search-20261028/reboot-terminal-recovery.json)、[终态](evidence/v6-fsdp-search-20261028/completion.json)和[完整小型审计](evidence/v6-fsdp-search-20261028/terminal-audit.json)。

三阶段执行完成不等于三轮均成功。实际顺序为：

| 自主轮次 | 结果 | 实际代码和检查 |
|---|---|---|
| `89aaa5ab69` | 三个参数均 worker_timeout，无质量结果 | 词表分片 head/embedding 自定义计算；exchange 在双端先 irecv 后 isend，存在相互等待风险。没有通过空异常事件证明具体阻塞位置。 |
| `86e14487d0` | worker 完成，三个参数均质量失败 | 改用 all_gather_into_tensor 交换后可执行；连续第二步完整 logits 最大绝对差4.529953e-6，超过原逐元素容差，拒绝。分片 GEMM/自定义梯度改变数值路径，但尚未独立隔离唯一数值根因。 |
| `4f5fce6e6b` | 搜索及完整版本独立接受 | 放弃自定义 head 算术，只配置 FSDP block 的 reshard-after-forward；参数实测 K=1/3/0，选择3。完整计算通过 parent forward 执行，后向保留参数重用，不改源数学策略。 |

第二、三轮实际生成前快照含前轮反馈，真实工具记录读取拒绝、worker状态、参数观测、质量与候选代码；第三轮正确区分第二轮“执行完成但质量失败”。没有人工插入优化答案。selection/candidate 的“已证实 wire-bound”“减少一次 collective 就等量缩短关键路径”“唯一根因”等说法超出证据，本文不采纳。Agent 自主产出的当前模型调用ID及 block路径属于本任务候选，不进入框架按模型硬编码分流。

完整版本独立AB/BA：B端点中位数176.607457/176.427291 ms，C为170.970796/171.137176 ms；聚合 **176.517374→171.053986 ms，降低3.095099%**。两对收益5.636661/5.290115 ms均超过噪声界1.779594/0.674946 ms。四端点各rank全部5样本保留，其中首个C端点存在232.461366 ms慢样本，没有剔除或扣除暂停。全部两步质量零误差、覆盖通过；每样本 allocated/peak均为1,462,069,760 / 2,239,245,312 bytes，峰值满足原2,351,207,577.6 bytes上限。

已重新核对adapter SHA、request以及此前独立源检查对应的六个完整source/reference文件哈希。复用的基线参考仍与两步独立serial-global source一致；候选全量比较依赖这套已核实参考，没有另造简化质量标准。负例仍沿用前置记录的真实源检查与存储观察量比较，未声称本批再做GPU注入。

独立SQLite原生核查：每rank baseline **49 AllGather + 25 ReduceScatter**，candidate **46 + 25**；总kernel从2115降至2106。这与实际候选保留末尾三个block、避免三次后向re-all-gather一致，形成减少通信操作且整步独立加速的正例，收益幅度仍较小。profile的NCCL sum扰动很明显（baseline rank1为381.15ms，candidate rank1为347.96ms），不拿它解释clean收益或带宽。四项次级指标的每个端点、每rank三份raw trace均独立重算通过；这些字段来自不同profile调用，例如一个candidate union中位数241.48ms，不能与另一组sum170.25ms作同轨迹代数比较，也不能直接等价为暴露关键路径。

本批10个角色会话，最终仅一个active bundle；只有初始baseline discovery，接受发生在最后一轮，没有本批接受后重发现或多bundle共存证据。生产core仍`ed24af7a…`，没有采用新的prompt、allocator或失败路由。审计脚本首次误从metric_samples取step_ms导致KeyError，修正为clean samples与次级metric_samples分别读取后审计通过；仅是审计容器映射错误，冻结实验和所有数值未改。

下一步按[后续三轮协议](v6-fsdp-continuation-20261028.md)从此已接受版本继续，给框架原样前轮反馈，重新发现并自主选择后续范围；不添加开发者优化建议，不重测旧失败到通过。启动前容器约12.17 GiB，先对三个已结束且无活动依赖的旧诊断批次做服务器内无损压缩，保留完整恢复清单，再启动单个GPU批次。
