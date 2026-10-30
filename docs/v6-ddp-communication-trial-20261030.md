# 当前设备 DDP 一轮普通通信试验

已于2026-10-30 02:10:29 UTC（香港10:10）实际启动，冻结`65dfbef45891536939f03b9e8c53e1ce663a8f1c`/controller100672。02:12:26启动审计SHA `aa74ad333c8db141755b213d8fb7570f2709eafa1fcee02b48654d3c9cd3abbb`：115冻结文件/request/prepared/source/core/UUID通过，discover-81e52aab57双rank通过，发现附着空revision e925da5f，源7a49不变。真实collection-planner generation-fb697d303320/session ses_e756cd09cffe9R8PAYNHvV31Tc，6初始文件262,446B逐字等于snapshot，11 assistant/13工具，私有历史留服务器；仅启动身份/活动证明，完整后续访问/质量/接受待审计，无旧candidate文件。实际本地OpenCode start/health/stop通过，未发模型测试prompt。唯一隐藏监听44088运行，df11,176,882,176B约10.41GiB，GPU已结束发现、角色阶段空闲。见[启动记录](evidence/v6-ddp-communication-trial-20261030/startup-verification.json)。

空间前置已完成：02:06:48–02:07:39UTC，13个旧派生证据1,453,406,919→92,206,688B，压缩/解压SHA、4,204保护文件/102其余文件/历史与新接受版本保持；净释放1,361,186,816B（1.27GiB），余11,266,043,904B（10.49GiB）。journal SHA `63c7c1e21397ca64c64d7afa218a7c5fbc80d42b7e7b047c2c592e4c5d46bb28`，独立恢复路径见文末。GPU当时空闲，未删数据或改历史决策。

从[当前设备空基线](v6-ddp-communication-device-20261030.md)接续一轮普通现有CLI `resume --rounds 1`，默认scope/参数探索、API只观察。工作根`/root/autodl-tmp/dev/v6-work/ddp-communication-trial-20261030`；实际启动信息后补。FSDP本轮已拒绝、不重测，Qwen链已收尾。本批不指定group、bucket、算法或实现，不提供旧Graph候选/人工因果表/修复答案；普通Agent自主选择通信或计算通信联合范围，不预先假定候选必然减少通信。

父分支`ddp-communication-device-20261030/ddp-current`为空7a49c637，原prepared/current与历史09b7 Graph指针不变。现有fork_run共享新controls/合同/源，不复制约4GB参考、不再source/control执行；重新发现当前空baseline，不装旧discovery。baseline/full-source/negative审计`b6f94dbb…`，源`cbf30d10…`、exact关系负例`3f72d15a…`按精确hash复用；原小非零source误差保持。原adapter b276a7ed字节与core ed24af7a不变。

原完整GPT2 12层hidden768/12head/vocab50257、两rank各B1/S128/累积4、有效8序列1024tokens、FP32/eager/noTF32/dropout0/SGD .001完整step不变。原baseline已前三个microbatch no_sync；完整梯度、更新参数、loss/finite和exact跨rank关系、2连续/2warmup/5clean、两同步/原owner及completion保持。逐样本max rank再median/MAD；原新锚133.379621ms，step≤140.04860205ms，不重设锚、复用旧时间或删长尾。通信诊断原映射每rank3独立profile，不能代替clean score、wire或关键路径。

现有planner可选择native/其他collector，控制器不另加独立或专门Nsys阶段。记录真实输入交付、完整工具/CPU与GPU探测、所有参数、异常/质量/约束/噪声/路径及全部样本。controller完成、worker passed、quality通过、搜索接受与独立接受分别报告。只有真正新接受版本，终态审计后才声明独立整版B/C/C/B和必要原生通信证据；失败保留，不再测到通过。

原始源映射和显式诊断metric映射为已声明辅助，本批不是无辅助原入口或单变量A/B。生产prompt/sampler/acceptance/allocator不变，不加reviewer或重开已关闭提示研究。不同输入副本字节相同不等于Agent全部阅读；外部文件系统非隔离，后续必须审核完整实际工具访问，不作绝对无外部访问声明。

启动要求GPU空闲，150GiB实际quota留8GiB告警加2GiB试验预算。前一准备消耗新controls后余量约9.23GiB不足，故额外清单只针对13个旧派生discovery/native summary JSON和已完成readiness trace，约1.45GB，保护4,204个当前文件；不改旧接受决策、raw native SQLite、源、共享controls、immutable store、accepted实现、模型/provider/private history。计划SHA `5ca80ecdf661f88137d54552f98e172697f267c516efbc29a59aa8edf7412f10`；服务器恢复根`storage-audit-20261030/ddp-communication-trial/run/RESTORE.md`与独立journal。实际校验/净回收后才启动，无raw本地转存。

新长批仅一个显式隐藏完成/磁盘监听，不重启任何已处理批次。低磁盘不视为完成，不中断实验或盲删；raw/private历史留服务器，GitHub只存代码、哈希与小摘要。Qwen既有48.1449824%独立、历史FSDP K3 3.095099%与同有效2.157792%不相加，DDP Graph计算收益不冒充通信接受，所有旧拒绝保持。
