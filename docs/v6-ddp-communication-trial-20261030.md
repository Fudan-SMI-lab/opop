# 当前设备 DDP 一轮普通通信试验

已于2026-10-30 03:11:12 UTC（香港11:11）完成，事件为`batch_finished`。终态 **拒绝**：trial `df2eb7acd2`仅一个空参数候选，`no_valid_parameter_candidate`，无新接受版本、clean样本、配对或独立阶段。终态审计SHA `70b28f9d3d09ed8078461a325a40e0ce021a5c00e3b13d12e4bb5fff467746bf`，小型[公开证据](evidence/v6-ddp-communication-trial-20261030/public-terminal-audit.json) SHA `95a491bf3e0479a15db57eb2942fe93bfd2b5130206980b4d35916bcf1726cc3`；完整工具历史、请求、日志、trace和失败留服务器。本轮已收尾，不重跑候选或启动独立接受；Qwen链与最近FSDP拒绝保持封存。

9个实际GPU job、18个rank worker均`status=passed`，但screen两rank均`valid=false`。第一连续完整检查每rank比较248,879,620元素、误差0；第二次都首先在`output.loss`失败：

| rank | 原第二步期望loss | 实际loss | 绝对误差 |
| --- | ---: | ---: | ---: |
| 0 | 10.896949768066406 | 10.976432800292969 | 0.0794830322265625 |
| 1 | 10.883541107177734 | 10.959263801574707 | 0.07572269439697266 |

原`atol=1e-6/rtol=1e-4`、math flags及先两连续质量后warmup的顺序保持。未进入2warmup/5clean或后续通信metric采样，不能报告step收益、原140.04860205ms约束合格或配对噪声接受。`observe`在compare前已生成全梯度/参数并完成exact跨rank梯度关系及负例；比较在loss处停止，不代表后续观察未执行，也不能宣称它们已经对上第二步control。完整实际张量未持久化。

精确候选8,822B，SHA `cf5e5fa7572e159a2a03828ee83bbbded6f730dd68d5a8488d46327c0af15d05`；proposal SHA `4f840ce064eb31a21b806b32950af8959518c1395f716329807679ec888cf187`。Agent选择execution_plan、一个project组及1526 affected calls；代码独立选择497,759,232B平坦梯度缓冲区、一次SUM AllReduce再除2，并捕获完整SGD step；空parameters、无requires/replaces。selector原设想保留13个bucket并Graph重放，与最终candidate机制不同。没有手选bucket、旧Graph安装或手工代码修复。

候选把首调用称为`warmup_parent`，第二调用才lazy setup，但框架这两次是连续质量更新。screen路由各一次`parent_eager_baseline`和`fallback_setup_failure:RuntimeError`，日志有AccumulateGrad流不一致警告。两份独立run-only sidecar、每rank均记录捕获时“AccumulateGrad保留default stream引用”的一个RuntimeError，沿accumulate_only/full_step/_setup/replacement四帧传播，以及一次普通False peer agreement；不是四个独立错误，也不是sidecar做过observe/compare。screen未保存完整异常文本，不能假定各诊断执行内部状态完全相同。

回退代码先把参数复制为`state.initial`再执行parent。审计仅CPU mmap读取已有controls的loss，没有模型或源重跑：两个第二次实际loss都精确等于各自第一步control。这与回退丢失第一更新的代码顺序一致；不是完整参数状态转移trace或所有捕获问题的唯一原因证明。setup仍执行额外累积/通信预热，“未得到快路径质量”不等于setup内部从未执行。sentinel仅比三份小梯度/loss，容差1e-2/1e-3且参照首调用，不能替代原质量；缺失梯度可跳过、restore异常可吞、后续Graph异常无peer一致处理，无一般状态/owner/并发恢复保证。成功路径代码有三次peer agreement，并非proposal所述两次。

原生证据分别重算：baseline发现每rank3999 kernel、13 AllReduce，逻辑输入497,759,232B（12×28,351,488+157,541,376）。原始发现各rank1523 calls/857 groups，合并后供选择1526 calls/860 eligible groups。baseline两份Nsys SQLite SHA分别`0536ceb9440b74c201002631c75537d6687fc44829c82874ca3974a2fb1c21bd`、`7b191912b2b51468e988b8d80c64012c6c919d750c21e9679691c644cc45bf47`；各3999 kernels/13 AllReduce RING_LL、0 Graph kernel/0 GraphLaunch，NCCL sum分别34.744348/231.368710ms，全部保留。实际发现/Nsys/tuning worker的channel日志为SHM/direct/direct；拓扑能力不能替代它，也不能据此计算物理字节或关键路径。

四条baseline/candidate timeline逐条重算kernel union/gap、通信、同步及clock，均3999 kernels/13 AllReduce RING_LL。候选timeline在reset warmup后记录回退parent，没有candidate Nsys或原生快路径通信优化证据；其NCCL sum rank0/1为44.136769/35.626916ms，不能和其他worker/period相减得收益。tuning日志RING/SIMPLE来自含初始化与预热的另一执行，497MB条目也在baseline日志内，不是candidate成功证据，未与kernel事件逐一关联。

calibration两种payload、每rank各5样本（20个诊断rank样本），逐样本max rank后median/MAD为28,351,488B：1.974057/0.010929ms；157,541,376B：10.553975/0.024177ms。没有测拟议497MB单buffer；逻辑吞吐不是wire上限、原任务调度复现或整步下界。selector/proposal用独立Nsys busy/gap推clean约70ms host成本、宣称接近wire ceiling的推断不采纳。

四个真实角色初始snapshot逐字核对：6/25/28/8文件，262,446/16,940,544/22,758,180/268,263B；13/25/68/11 assistant、15/31/82/17工具，均0框架schema repair。provider估计合计5.26530008804，仅观察非账单。candidate初始write及4次edit精确重构最终字节；完整工具输入输出、截断大输出与内联CPU脚本保留私有。记录访问涉及当前打包输入/源码/诊断、安装的torch/transformers和自写文件；未观察旧优化实现或其他审计/reviewer答案读取，但FS非隔离，不作绝对无外部访问证明。`/tmp/opencode`用于自写内联CPU小模型和eliminated.json，目录共享本身不证明使用其他case代码。

CPU探测不是完整CUDA质量：小GPT2静态mask/梯度view测试；最终body是手写world1 identity-average单步模拟，未导入真实candidate或执行原两rank连续路径。真实candidate build测试仅走CUDA隐藏时unsupported_parent_path。错误保留：错误transformers路径、两次非法条件import、缺失CUDA header；flat toy先7元素装15元素失败，后reference累积错误打印False，再修toy得到True；CPU profiler查设备返回code100而CPU工作继续；stderr/pipeline/compound exit0不当作所有子命令成功。终态reader两次误取早拒绝结果不存在的pairs/diagnostics字段，另有探索reader错误evaluate.py路径，均保留并只修reader。初版审计`fb870617…`在补充tuning/Graph/max-rank前保留为`initial-terminal-audit.json`。

115冻结身份、每worker实际request/materialized manifest/adapter/entry/core/UUID/共享reference路径已复核；source current7a49、branch空e925da5f、历史09b7 Graph与旧拒绝保持。03:27:38UTC实查两4090显存0/利用率0、无GPU worker，150GiB盘余10,992,795,648B约10.24GiB，本轮无清理。隐藏watcher44088已排队后退出并标记handled，未重启/补发；没有后台新实验。后续通信研究仍需守住完整连续状态语义，再判断native机制与独立整步收益；本次不据失败自行追加候选链或生产改动。以下启动描述仅保留历史。

已于2026-10-30 02:10:29 UTC（香港10:10）实际启动，冻结`65dfbef45891536939f03b9e8c53e1ce663a8f1c`/controller100672。02:12:26启动审计SHA `aa74ad333c8db141755b213d8fb7570f2709eafa1fcee02b48654d3c9cd3abbb`：115冻结文件/request/prepared/source/core/UUID通过，discover-81e52aab57双rank通过，发现附着空revision e925da5f，源7a49不变。真实collection-planner generation-fb697d303320/session ses_e756cd09cffe9R8PAYNHvV31Tc，6初始文件262,446B逐字等于snapshot，11 assistant/13工具，私有历史留服务器；仅启动身份/活动证明，完整后续访问/质量/接受待审计，无旧candidate文件。实际本地OpenCode start/health/stop通过，未发模型测试prompt。唯一隐藏监听44088运行，df11,176,882,176B约10.41GiB，GPU已结束发现、角色阶段空闲。见[启动记录](evidence/v6-ddp-communication-trial-20261030/startup-verification.json)。

空间前置已完成：02:06:48–02:07:39UTC，13个旧派生证据1,453,406,919→92,206,688B，压缩/解压SHA、4,204保护文件/102其余文件/历史与新接受版本保持；净释放1,361,186,816B（1.27GiB），余11,266,043,904B（10.49GiB）。journal SHA `63c7c1e21397ca64c64d7afa218a7c5fbc80d42b7e7b047c2c592e4c5d46bb28`，独立恢复路径见文末。GPU当时空闲，未删数据或改历史决策。

从[当前设备空基线](v6-ddp-communication-device-20261030.md)接续一轮普通现有CLI `resume --rounds 1`，默认scope/参数探索、API只观察。工作根`/root/autodl-tmp/dev/v6-work/ddp-communication-trial-20261030`；实际启动信息后补。FSDP本轮已拒绝、不重测，Qwen链已收尾。本批不指定group、bucket、算法或实现，不提供旧Graph候选/人工因果表/修复答案；普通Agent自主选择通信或计算通信联合范围，不预先假定候选必然减少通信。

父分支`ddp-communication-device-20261030/ddp-current`为空7a49c637，原prepared/current与历史09b7 Graph指针不变。现有fork_run共享新controls/合同/源，不复制约4GB参考、不再source/control执行；重新发现当前空baseline，不装旧discovery。baseline/full-source/negative审计`b6f94dbb…`，源`cbf30d10…`、exact关系负例`3f72d15a…`按精确hash复用；原小非零source误差保持。原adapter b276a7ed字节与core ed24af7a不变。

原完整GPT2 12层hidden768/12head/vocab50257、两rank各B1/S128/累积4、有效8序列1024tokens、FP32/eager/noTF32/dropout0/SGD .001完整step不变。原baseline已前三个microbatch no_sync；完整梯度、更新参数、loss/finite和exact跨rank关系、2连续/2warmup/5clean、两同步/原owner及completion保持。逐样本max rank再median/MAD；原新锚133.379621ms，step≤140.04860205ms，不重设锚、复用旧时间或删长尾。通信诊断原映射每rank3独立profile，不能代替clean score、wire或关键路径。

现有planner可选择native/其他collector，控制器不另加独立或专门Nsys阶段。记录真实输入交付、完整工具/CPU与GPU探测、所有参数、异常/质量/约束/噪声/路径及全部样本。controller完成、worker passed、quality通过、搜索接受与独立接受分别报告。只有真正新接受版本，终态审计后才声明独立整版B/C/C/B和必要原生通信证据；失败保留，不再测到通过。

原始源映射和显式诊断metric映射为已声明辅助，本批不是无辅助原入口或单变量A/B。生产prompt/sampler/acceptance/allocator不变，不加reviewer或重开已关闭提示研究。不同输入副本字节相同不等于Agent全部阅读；外部文件系统非隔离，后续必须审核完整实际工具访问，不作绝对无外部访问声明。

启动要求GPU空闲，150GiB实际quota留8GiB告警加2GiB试验预算。前一准备消耗新controls后余量约9.23GiB不足，故额外清单只针对13个旧派生discovery/native summary JSON和已完成readiness trace，约1.45GB，保护4,204个当前文件；不改旧接受决策、raw native SQLite、源、共享controls、immutable store、accepted实现、模型/provider/private history。计划SHA `5ca80ecdf661f88137d54552f98e172697f267c516efbc29a59aa8edf7412f10`；服务器恢复根`storage-audit-20261030/ddp-communication-trial/run/RESTORE.md`与独立journal。实际校验/净回收后才启动，无raw本地转存。

新长批仅一个显式隐藏完成/磁盘监听，不重启任何已处理批次。低磁盘不视为完成，不中断实验或盲删；raw/private历史留服务器，GitHub只存代码、哈希与小摘要。Qwen既有48.1449824%独立、历史FSDP K3 3.095099%与同有效2.157792%不相加，DDP Graph计算收益不冒充通信接受，所有旧拒绝保持。
