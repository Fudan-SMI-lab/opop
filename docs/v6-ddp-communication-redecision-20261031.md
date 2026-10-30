# DDP 通信方向的新一次项目级决策

后续通信观测作者已完成[完整审计](v6-ddp-communication-observation-author-20261031.md)：未生成本selector的计算优化候选。因精确probe的关联/恢复CPU反例，下一步限定[一次观测机制修复](v6-ddp-observation-mechanics-repair-20261031.md)，不重做selector、不恢复77ac、不改生产策略。

本批已于17:35:37UTC（香港01:35）完成，reason=`batch_finished`；controller退出、GPU无worker、余约8.20GiB，旧监听49252不重启。28 assistant/34 tools/0schema修复/provider估计0.91937204936非账单。selection `5878032d`仅whole-step组/1055调用，选择合并前三个microbatch的计算路线、保留13AR。原输出不改，不记新候选/通信质量/接受。

终态审计SHA`94ee7a0e5d43529de7408d00149e17708c84996e5fd1aa3e6361dc63483d02f5`，[公开小型记录](evidence/v6-ddp-communication-redecision-20261031/public-terminal-audit.json)SHA`c61335b2bb38c705ae8c282815a69a99b28b567d367b9af69ef1309196a76208`。41初始文件/27,224,402B、924冻结身份、普通prompt/model/tools/core及current再次核验。2108 read行/1截断前缀核对；全部纯JSON分析输出从原输入独立重建一致，唯一selection写入从literal AST重建精确，没编辑。两次只读审计reader分别误拒绝`with open`与`type(...).__name__`，错误保留，仅修正受限重建reader，没有角色/模型/CPUfixture/GPU重跑。原timeline每rank13AR/497,759,232B与独立校准样本重算，仍不当wire/critical path或新验收。

完整访问审查显示：角色读了current ccd90、原始任务、当前timeline/calibration/tuning及旧失败sidecar；**没有打开新增decision-evidence代码/proposal/CPU结果/索引或fixture**，仅工具9深度截断字段摘要。工具30已看到`params.index(s)`异常，却仍把旧eager fallback误称为三Graph replay+merge并分配66ms成本；工具21明确暴露“非lower bound”限制，却在选择中断言SHM带宽饱和、通信时间/体积硬下界。时序相邻/CPU回调/能力字段不证明完整GPU依赖或wire瓶颈。其计算方案数值、fallback与62–67ms预测均未执行验证。

因此不直接生成该计算候选。明确下一步为[一次当前accepted DDP通信观测作者](v6-ddp-communication-observation-author-20261031.md)：只给原始code/任务/观测与具体证据问题，由作者自主决定可移除的最小观测，澄清同次调用的就绪/排队/完成证据与未知范围。无手选collector/算法、无GPU阶段或生产改变。77ac修复仍关闭，不重复selector直到选通信；通信研究继续推进。下面是本批启动记录。

已于2026-10-30 17:21:18UTC（香港10月31日01:21）实际启动，冻结`5ec3a42d091ddaea56f3a55d3ec1e5de3468831c`，controller **177939**，唯一隐藏completion/disk监听 **49252**。旧28768已handled/退出，没有重启旧批次。监听只检查进程、文件与磁盘，终态接回当前会话审查；磁盘告警不当完成。此时运行的是selector，未调度GPU阶段，不将GPU空闲描述为正在GPU验证。

17:23:54UTC启动核验：[startup-verification](evidence/v6-ddp-communication-redecision-20261031/startup-verification.json)，SHA`2e0d4b4efb68fbdee0f5c01188e6238fa4c28901fb384fcd8e3321f3db67e094`。实际generation`4ac2bc7c0189`/session`ses_e722b4144ffendyK8EedF687dl`，41初始文件27,224,402B逐字一致，原37文件中除feedback外不变；924冻结身份、source/current1fc、生产core、普通prompt/model/tools均核验。实际route反馈10,260,140B/SHA`0a8d6665b150f35dceeabfa484143ccf265e919a228ef08c7c09ea3a37e20b96`，仅supplemental增加声明CPU证据。两卡无worker、实际150GiB配额余8,808,640,512B约8.20GiB；这不是新质量或接受。

早期9 assistant/13 tools的有限审查：[startup-tool-review](evidence/v6-ddp-communication-redecision-20261031/startup-tool-review.json)，SHA`8c7e8fd3301ef8dcfe8fe8698de374fe5e89fc008a2d03496e985f66134121f1`。2060 read行与原文件核对、1显式截断前缀；8/9/10的反馈CPU文本遍历输出独立重建逐字相同，未重跑角色/模型。工具10读原拒绝理由和旧selection主张，13读当前accepted代码；工具9只按深度截断摘要CPU字段，前13工具未打开新增decision-evidence文件。4/7/11目录列表仅元数据/前缀审查。不能把交付/字段暴露当完整消费或正确推理，完整终态访问审计待结果；外部FS非隔离。私有历史SHA`846dcf5de2324e416674aec184c9198b7505efecd1cf07193523cefde814384d`留服务器，未回显provider配置或凭据。

首次启动预检在角色/controller/模型/GPU/监听启动前发现：此前独立CPU fixture执行的是CRLF源文件，Git部署规范为LF，故原SHA检查中止。将本地保留的原执行代码逐字存入服务器旧CPU审计目录（SHA4868f3d3一致），新索引改指向该不可变副本；没有重跑CPU或更改生成候选。失败记录另存launch-preflight-attempt-1，冻结b44f0baa目录保留，不当作已启动实验或重试模型。

用户于香港2026-10-31明确要求继续推进，避免付费服务器无任务空闲。因此不再把“等待新项目偏好”作为当前工作的前提；在已有真实DDP任务和新取得的合同证据上推进一次有界项目级决策。原通信优先/完整质量/原约束/独立整步收益要求不变，Qwen关闭、最新FSDP拒绝保持。

本步只运行现有 `ProjectAgents.generate(selection)` 一次，普通selector提示、模型、工具、可选scope和正常schema修复不变，API observe-only、无新增cap。当前accepted source/current discovery仍为1fc24545/ccd90单bundle；不重复discovery/controls/source比较，不复制reference，不改accepted指针。普通选择范围由角色决定，开发者不指定group、bucket、算法、实现或collector。

上次selector的37个实际初始文件作为基础：除feedback的supplemental字段外其余保持，新增4份精确材料——77ac代码、680b proposal、独立CPU结果a394d889及小型索引。原trial02ed5436b4的framework反馈/配对拒绝/历史5e3a代码保持；新CPU记录不是一个新GPU trial，更不是数值或性能拒绝。普通route只增加显式CPU事实辅助，所有原字段/原history不改。没有整份人工审计、因果答案或手写修复。不是无辅助cohort或单变量AB。

77ac实现修复链仍关闭，本步不执行或继续编辑它。新的问题是：结合精确失败记录与当前accepted项目，是否有证据支持不同的后续优化路线。输出后完整审查真实读取、假设与scope；仅有合理的新路线才另行声明一次candidate生成及后续验证，不无条件接续相同修复。若输出只是重复已失败主张，保留该结果，不通过重复selector/prompt直到得到预期答案。

控制器无candidate/搜索/参数/GPU/model/collective/collector/native/验收阶段；实际角色工具活动仍须审查，意图不代表未发生探针。原DDP两连续完整检查、2warmup/5clean、exact rank replicas、1e-6/1e-4、原step上限140.04860205ms保持。NCCL sum不当wire/critical path，计算Graph收益不记通信接受。

启动预算256MiB基于前小型角色实际约80MiB及本次约27MiB输入，保留8GiB告警线，不是API限额。仅串行，所有后台SSH/监听隐藏，一个plain completion/disk watcher。完成事件接回完整审计并继续下一项有界决策，不将一次完成等同整个研究结束；low_disk_space不是完成。原始材料/私有配置留服务器，不回显凭据，GitHub仅代码/hash/小型元数据。实际启动、输入身份及监听另记。
