# DDP 原始通信观测反馈的一次普通项目决策

实际18:55:18UTC（香港02:55）已启动，冻结`b4e86ec8c106952b8c7a3f2e8d18ca30feb714d5`，controller192276/唯一隐藏监听52336。local OpenCode实际start/health/stop通过，server_url=null，没有model预检请求；旧14060终态已直接审计/handled，排队重复通知去重，不重启。

18:57:04UTC[启动核验](evidence/v6-ddp-observation-feedback-selection-20261031/startup-verification.json)SHA`46bc04d529463a024ce5159f8efae6c3d55b1442e0f7c49c6b3e7cb1a171b4b0`：generation999c08dfa37c/session ses_e71d5303effeplDBk2SALVlQUL，56初始文件27,860,418B与实际snapshot逐字节相同；原41非feedback保持，1005冻结身份及sourcecurrent/prompt/model/tools/core相同。实际route10,261,232B SHA`2d2eeeba11945301a08e797485cfba207e5212f8af52f2f8ab78573bb884ab38`，原字段/CPU事实保留，新原始观测仅supplement索引与15文件。

11 assistant/16工具，[早期审查](evidence/v6-ddp-observation-feedback-selection-20261031/startup-tool-review.json)SHA`3f75c276548fe60897ef943a396940f6dab41b01f3e8f4d8508274bbd8a1594f`核对2023 read行/1截断前缀，head4000与工具10/11/15/16纯JSON输出重建相同。实际读旧request/拒绝/通信资源/5e3a失败代码；tool10展示新增index指针，但前16尚未打开新observation-evidence文件。只说明早期访问，不能推断终态不读或正确消费。无早期模型/GPU/collective probe，无controller jobs；余10,669,785,088B约9.94GiB。完整私有历史留服务器，不打印私有配置。

在新B/C真实观测完成完整终审后，继续一次现有`ProjectAgents.generate(selection)`。这次新输入来自已经实际执行的原始观测，不重复selector直到得到预设方案，也不继续修复observer。原通用项目优化目标和通信优先保持；无手选group/bucket/算法/实现/collector，无生产prompt/model/tools/质量/接受变更。

从前次实际selection generation4ac2bc7c0189的41份输入出发，所有原非feedback文件逐字节保留。原previous-feedback既有字段及CPU事实supplement保留，仅增加`diagnostic_artifacts`索引；另增加15份原始文件：Agent probe/plan、B/C原request和observed request、四worker result和四diagnostic、索引。12条raw metric trace只指向服务器原路径/hash，不复制。不给终态审计表、callback数量解释、GPU时序因果结论或手写修复。之前CPU/source/metric映射辅助边界仍在，不是无辅助入口或单变量AB。

原trial02ed5436b4反馈14,429,015B、SHA`a0234920605970500c870b2ec98fc952b264a79f264bd19d6a53c99b796cf459`不改，原5e3a失败历史与77ac CPU合同材料保留，77ac实现修复链关闭。source仍`ddp-accepted-communication-trial-20261030/ddp-search`的current1fc/精确ccd90/单bundle/emptyparams与实际accepted-code discovery；accepted4541、independent7a49、历史Graph09b7不变，无新discovery/controls/source/reference副本。

普通selection prompt及现有feedback suffix、model/tools/default scopes/正常validator修复不变，API仅记录，不加cap。仅决策生成，没有candidate/search/参数/GPU/collector/native/接受或pointer阶段；实际工具访问/CPU-GPU活动仍需完整审查，交付不等于读取，非hermetic不作绝对无外部读取声明。PRIVATECONFIG/CREDENTIALS不回显。

预估256MiB材料/历史增长，保留8GiB警戒；前一GPU观测后余约10.02GiB，不新增清理。新长任务一个隐藏完成/磁盘监听，旧14060已handled不重启。终态要审核完整快照、实际读取、选择依据与normal repairs；仅有具体新证据支持的新路线才另行声明候选，不因scope名称称通信成功，不自动GPU或无限生成。若仍沿用失证前提，则保留输出、根据实际缺口作有界证据决策。启动与实际输入核验另记。
