# Qwen 接受后续行的断电恢复

**终态审计：2026-10-29 17:12:56 UTC 完成但执行出错，没有新接受。** [终态](evidence/v6-qwen-accepted-recovery-20261030/followup-completed.json)、[公开审计](evidence/v6-qwen-accepted-recovery-20261030/public-terminal-audit.json)，服务器完整审计SHA `fe139621b7e07b161705a9302e9eede927fdcdc2b2efe0946837fac043c7bde2`。旧控制器断电记录保持，恢复只生成了原来未完成的第二轮候选；没有第三轮，也没有重复第一轮。

第二轮 `trial-recovered-4f3d7f11aa` 生成52,324B候选 `f5b76ebde100c039d8ad9264e316dbade5acdb74c02f40308e958673e802be7e`，空参数，stage为f211410e。启动前设备解析发现冻结请求中的 `GPU-6c11763a-1e00-9af3-e17d-274f5d7f78ba` 已不在当前主机，因而拒绝启动。该screen没有request/started/receipt，没有质量检查、干净样本或配对决定；独立阶段not_run。生成/分支stage不等于接受，源current009b和续行current eb9保持，仍仅活动f065 bundle。

本次实测GPU0为 `GPU-20e8003c-f531-d50c-084c-a9a97fe94004`、GPU1为 `GPU-a24af2e6-99bb-9729-5d14-50fa10a2b321`，均RTX4090。恢复预检确认可用却漏核对冻结UUID，是此次预检缺口；框架的设备保护正确拦截。不能把这次失败称作数值不忠实，也不能将旧设备TTFT anchor直接用于新设备。当前两卡空闲，150GiB配额余约10.89GiB，无本轮清理。

271旧文件和33原selector输入核验不变；同一持久化会话nonfresh恢复，最终29 assistant/36 tools，原未完成message仍保留。新candidate为36输入、36 assistant/44 tools。六个实际角色全部0 schema repairs；工具数依次为初始planner10、第一selector31/candidate50/反馈planner12、恢复selector36/candidate44。provider估计合计4.59042550536，仅观察记录，非账单。完整message/part、命令/输出、初始写入及编辑字节私存服务器 `terminal-private-history`。实际恢复API把selector的previous-feedback逐字节传给candidate，没有再做普通trial的selected-group route；历史4857代码字节正确。不能假设恢复候选的routing字段等同普通新trial。

原六个GPU作业重新核验：accepted f065的discovery、timeline、Python profile有效；第一候选screen及两项诊断worker均安装失败。discovery原trace核对68,777 kernels、53 call sites/46 groups；timeline从原trace重算设备union/gaps/sync/clock完全一致，Python原pstats已检查。这些是不同受扰诊断调用，不是新完整质量/干净计时，不能相减推出host cost或独占关键路径。

第一候选的candidate-diagnostics实际保留三层传播的同一 `AttributeError: Qwen3Attention has no attribute rotary_emb`：_extract→setup→build。build捕获后返回空表，外层声明位置检查才抛出ValueError。这解释已观测安装链条，不推广为旧数值错误的唯一原因。恢复candidate实际读过该诊断、4857/f065代码和安装库，改为模型级rotary查找，并让setup失败时仍安装两个回退callable。它同时改变了若干数值/helper路径，不只是错误信息修复。

新代码拟用六个Triton kernel、内部实现选择和31 decode Graph；prefill为父调用。bundle明确替换f065，当前完整版本只有新bundle，BindingSession此时捕获原模型forward，因此“继承已接受prefill Graph/TTFT保证”不成立。fallback只处理特定异常，早期context guard仍可能返回空表；自检直接调用替换函数，不等于完整框架质量/别名/并发保证。CPU两层hidden128/BF16/prompt6/3decode探针替换了真实Triton实现：TypeError、KeyError、退出0但NaN/FAILED均保留；最后补齐mock的out.copy_才打印token相等、logit/KV非零误差。它不是原36层/32token CUDA质量证明。工具记录未见后来答案读取，但外部FS非隔离，不能作绝对排除；profile差值、唯一GPU瓶颈和延迟保证不采纳。

下一步为单独声明的[当前设备固定验证](v6-qwen-device-refresh-fixed-20261030.md)：既有prepare_variant仅改devices，重新测原baseline controls及TTFT anchor，复用手工外部映射对新controls作完整源比较，再一次固定B/C执行原样f5b候选。无新Agent/搜索/正式接受；旧两批冻结不重启，旧结果和anchor不改写。原监听23004已处理退出，新长任务只设一个监听。以下保留恢复启动时的历史记录。

**已恢复运行：2026-10-30 00:56香港时间（10月29日16:56 UTC）**，冻结fde0f777，恢复控制器2158。[启动回执](evidence/v6-qwen-accepted-recovery-20261030/launch.json)、[实际协议](evidence/v6-qwen-accepted-recovery-20261030/protocol.json)、[冻结预检](evidence/v6-qwen-accepted-recovery-20261030/frozen-preflight.json)与[原会话健康检查](evidence/v6-qwen-accepted-recovery-20261030/session-health.json)均已保存。没有重新启动旧控制器或重复第一轮。

00:57香港时间的[实际恢复核验](evidence/v6-qwen-accepted-recovery-20261030/startup-verification.json)确认：271既有文件保持；原selector会话产生nonfresh generation-b58ccb84d0a8，历史已记录，33输入全部字节相等、无增删、model/tool设置相同。随后已进入原第二轮的新candidate generation-cb08f1ca8ed2/session ses_e7767536dffec2ZwleHR6E5WFV，36初始文件；第二轮此前没有candidate，这不是第三轮。此时未有新GPU评估/质量/接受结论，既有6个GPU作业与第一轮失败不变，源current009b和分支current eb9保持。

唯一普通监听23004已核实存活，[启动预检](evidence/v6-qwen-accepted-recovery-20261030/wakeup-setup.json)及[等待状态](evidence/v6-qwen-accepted-recovery-20261030/wakeup-verification.json)保存。实际df余11,720,011,776B（约10.92GiB），Agent阶段两卡空闲；本轮未做清理。后续由完成/磁盘事件接续，不轮询模型。以下为恢复前核验和边界。

用户确认服务器欠费关机后重新开机。2026-10-29 16:46 UTC（香港10月30日00:46）检查发现：SSH已恢复，两张RTX4090空闲，150GiB数据盘可用约11.17GiB；容器PID1为本次新启动，原控制器986775不存在。宿主机 `/proc/uptime` 不是容器运行时长，不能据其声称容器没有重启。原批state仍为running/search，但没有completion文件，这是中断残留状态，不是仍在执行或正常完成。

原批为[Qwen接受后两轮续行](v6-qwen-accepted-continuation-20261029.md)，冻结b18cef24，生产核心ed24af7a保持。已有接受版本重新发现完成，当前eb9dfd94仍安装单个 `fixed-index-read-qwen` bundle，源码为f065，未接受新优化。已完成第一轮 `trial-45c44c1619`：空参数、新代码4857a625，worker失败于 `installed candidate sites differ from the declared bundle`，在完整质量/干净计时之前，决策 `no_valid_parameter_candidate`。这不是数值不忠实的证据，不能改写为质量通过或数值拒绝。该轮诊断与所有原始错误保留；本次不重复第一轮。

第二轮原目录 `trial-49db1bb09a` 尚未生成候选。实际selector初始snapshot为generation-8d3438a88a77，原会话 `ses_e78ea1e7dffeMDKhbbIGaT9pxM` 保存27条message、110个part，最后assistant没有completed标记；最后已完成工具为读取selection schema，sandbox尚无selection.json。33个初始文件与快照字节一致。保存完整现有会话历史于服务器private-history，记录哈希，不转存私有内容到GitHub。原155冻结依赖及作业/角色快照/第一轮文件共271文件核验，恢复前记录独立存于 `qwen-accepted-recovery-20261030/recovery-audit.json`，旧批新增中断说明但不覆盖state/logs/失败记录。

恢复仅通过已有 `--resume-generation`，在相同持久化会话和原sandbox续完第二轮，保留其上下文、原始反馈与当前parent。不会启动新的两轮搜索、重新选择参数预算、重新生成controls/reference或重复初始discovery。现有恢复API会为恢复后的trial分配 `trial-recovered-*` 标识，不能当作额外第三轮。既有恢复提示泛称“temporary service error”；本次实际原因是用户确认的欠费关机，外部审计明确记录这项措辞局限，不改生产prompt，也不伪造provider错误。

先运行真实本地OpenCode server健康检查，读取已存会话确认原消息可见；该预检没有模型调用。随后恢复控制器仅执行剩余一轮，API observe-only，不另设调用上限、不提供开发者诊断/手写方案。现有恢复接口会重写worker-environment和binding-contract输入并捕获continuation snapshot；保存恢复前字节哈希，终态报告任何实际变化。恢复会话不是fresh-snapshot A/B，也不是未中断的全自动成功。

只有这剩余一轮真正接受了不同的活动bundle集合，才按照原协议对完整新版本相对原prepared baseline009b做一次publish=False独立AB/BA。无新接受则not_run；不复测旧f065到通过。完整B1/prompt256/32greedy、BF16/SDPA/noTF32、full logits .02/.02/exact tokens、两同步、2continuous/2warmup/5samples以及原TTFT 24.051052×1.05=25.2536046保持。停机前后属于不同测量时期，不把跨期差值视为配对增量，也不换原约束anchor。手工外部源映射和既有诊断辅助边界保持。

本次不清理数据：实测空间高于8GiB警戒线，依赖、模型、controls、冻结和已接受版本保持。旧监听记录已标记由本次用户轮次处理；不得补发旧“controller退出”通知或重新启动旧监听。恢复长任务只设置一个新完成/磁盘监听，不轮询模型或测试投递。发生磁盘注意事件时先检查worker，不能把低空间当完成；授权冷压缩仍只在GPU工作结束后进行。

历史Qwen f065独立47.3251794%保持，未证明其之前有自主搜索接受；新FSDP K2质量/显存通过但独立噪声拒绝保持。旧fact-assisted Qwen独立45.1691695%/搜索43.9895207%、same-adapter fullgen46.9421035%/TTFT14.06%/decode88.8847%与旧固定拒绝all80分开；FSDP K3 3.095099%和同策略续行2.157792%不叠加，DDP40.61%/55.73%与旧communication拒绝、TP full32质量失败、ResNet拒绝all80和单独36.68%显存接受/missing images事实保持。提示/核验/有限阅读研究关闭，allocator coalescing和不支持的唯一原因/关键路径主张不采纳。
