# 当前 accepted DDP 的通信观测作者

后续原样92279修复已完整终审并只执行一次[固定B/C观测](v6-ddp-communication-observation-20261031.md)：977身份、8完整零误差检查、12raw trace保持；C Graph路径的Python归属显式unknown，未产生通信接受。旧0244从未GPU执行、原postcheck错误/CPU反例保留。观测/mechanics修复链现收尾，继续[普通项目selector消费原始新证据](v6-ddp-observation-feedback-selection-20261031.md)，不给人工因果答案，不重做本作者。

终态已审计：2026-10-30 18:07:48UTC（香港10月31日02:07），事件为`batch_finished`。真实角色有`PROJECT_AGENT_FINISHED`，71 assistant/87工具/0 schema修复，provider估算2.46235109804（非账单）。控制器`completed_with_errors`仅来自已预告的裸prompt比较，漏算575B正常包装；原控制器与错误保留，未重启角色。审计`658f17abf5d66ffdaa168f993e5d3cd1fff180fde5ece32ba78d04e7ca7ab5e3`，[公开摘要](evidence/v6-ddp-communication-observation-author-20261031/public-terminal-audit.json)`25063561…`。

11初始文件1,404,482B、950冻结身份、原source/current/core/model/tools均核验。完整87工具输入/输出与CPU脚本已审查，3724 read行与实际文件匹配，5个纯JSON打印输出独立重建；探针1次写入+16次编辑、plan1次写入均重建为最终字节。新探针`024441f7d80bddaa51590b3630fbcc3969ac31b05548546ad2089106a10b5e98`/37,167B，plan`862a728f25e8056ed7d61659440f22e323e9e299f57932e35b16f8dbd534bf68`/21,810B。实际读取原输入/原B/C rank0片段、原B metric trace、framework/库源码与自身CPU文件；未观察到私有配置/后续审计/其他候选答案读取，但FS非封闭。

作者实际执行了小型CPU Linear FW/BW/SGD、world_size1 Gloo DDP及broadcast，超出本次明确的“无模型/真实collective”限制，不能称纯mock。CPU profiler出现`gpuGetDeviceCount code100`，未见成功CUDA执行。SyntaxError、缺ijson被复合命令覆盖、两次float/dict错误、Path缺失、param-id误当Parameter、phase/payload断言、UnboundLocalError与cap键错误全部保留；Agent修改代码/fixture后测试通过并重复。其异常检查仅type/message，profiler恢复只查class；所谓ambiguity negative没有覆盖真实歧义。临时输出被覆盖、自己的cache被删除，完整命令及最终fixture保留服务器，不宣称所有早期临时文件仍在。

已另行执行**一次精确原字节CPU机制取证**：[脚本](../scripts/experiments/v6_ddp_observation_cpu_contract_20261031.py) SHA`4349d3cd744984bc47246a13963111215292b35e90f41e5d3d2d800455870b04`，[结果](evidence/v6-ddp-communication-observation-author-20261031/cpu-mechanics-result.json)。无模型、tensor计算、process group、真实collective、profiler执行或CUDA初始化。唯一范围正控通过，但重叠inner ranges仍被标observed；trace索引缺1时文件2被配给ordinal10，unknown trace也被标observed。无Python zero_grad的三段模拟profile事件全部停留ordinal0/quality，说明该ledger不等于真实evaluate调用。4096广播上限后仍追加None，4113条列表没有硬封顶。

原入口普通成功/异常各一次，validfalse与嵌套对象保留、原RuntimeError是同一对象，六个patch均恢复；返回dict本身被浅复制。注入非class profiler API的**条件性部分安装失败**时，原入口调用0次，五个先装patch未恢复；fixture最后自行恢复，不是修复探针。这些是CPU接口/报告反例，不是历史GPU/NCCL失败。探针还在entry返回后调用zeros-like bucket inventory，约497,759,232B额外缓冲；“不排任何kernel”未证实，owner/allocator扰动不可忽略。CPU callback、同trace launch关联、通信union均不能证明GPU就绪、wire time或critical path。

结论：**该0244探针不进入GPU观测。** 原输出/失败/所有旧接受保持。下一有界步骤是[一次观测机制修复生成](v6-ddp-observation-mechanics-repair-20261031.md)，仅原输入加精确前probe/plan、上述CPU结果及fixture索引；显式开发者CPU事实辅助，由原diagnostic_author自主决定最小实现，不手写collector、不优化candidate、不重开77ac。没有自动下一轮修复或GPU执行。终态时两卡无worker、实际150GiB盘余8,784,547,840B；本轮没有清理或reference复制。

实际已于2026-10-30 17:44:50UTC（香港10月31日01:44）启动，冻结`2865d183c4154b612655818f095f93d3a61f1809`/controller **181053**，唯一隐藏完成/磁盘监听 **48992**。原49252终态已处理且退出，不重启。launch时150GiB实际余8,797,040,640B，GPU空闲；本步仅作者生成，不能称GPU正在验证。启动实际快照核验另记。

17:48:37UTC实际核验通过，[startup-verification](evidence/v6-ddp-communication-observation-author-20261031/startup-verification.json)SHA`ab76b9d6dd885fc2b2f99d737a24338c879ec6c61e17505d827c39ad986e5e9d`：generation`bad1395efb6c`/session`ses_e7215a5c7ffe5uCvum0WOqYsk1`，11文件1,404,482B（10显式1,400,912B），950冻结身份/模型工具/core/current1fc保持。真实prompt包含下述正常框架包装，SHA`97ce70b3a5ed066ac77ba8b1f8f1dde5485a56c6df3968236ea54d3ff00fc1d0`。15 assistant/24工具、无jobs/GPUworker、余8,790,978,560B约8.19GiB。

早期仅审查24个工具的命令/路径/状态，输出尚未完整审查；[early review](evidence/v6-ddp-communication-observation-author-20261031/startup-tool-review.json)SHA`cb76d6312da470be8ee8155d5171abaffcafb99d83f449b8fe4d02ad78f31b61`。实际读取index/接口/问题/原任务/source/adapter/accepted代码、索引B请求及rank0、部分Crank0、当前三类观测的有界JSON内容、framework evaluation。工具9错误路径保留，10读正确adapter；23不存在目录被显式ls/head fallback覆盖。记录命令未见模型/GPU/collective调用，仍不能替代终态完整工具/输出/探针审查。私有完整历史SHA`e12902717957656469c62d96b2728857e8fcbdf955f1663f8c96802f11576782`仅服务器保存，无凭据回显。

启动reader首次错误地要求实际完整prompt等于裸实验正文，漏掉原`ProjectAgents.generate`的287B通用前缀和`_complete`的288B环境说明。实际3891B=正文3316B+原框架575B，任务正文/core不变。错误保存在`startup-reader-attempt-1.json`，只读reader按现有源代码重建完整prompt核验；没有角色/模型重启或输入修改。**冻结控制器末尾也有同一错误断言，保持原样，可能在真实角色完成后记为completed_with_errors。** 后续应从真实events/receipt/snapshot/输出分开审计，不能把控制器postcheck错误当生成失败、GPU失败或质量结论。此限制已告知用户。

本步接续已结束的项目级selector审计，保留原始项目优化目标、通信优先及所有质量/接受规则。新selector转向前三个microbatch的计算批处理，并未改变13个AllReduce；它用合成校准/当前trace把通信宣布为不可改善的硬下界，证据不足。因此不直接生成该计算候选，不重复selector以取得预期选择，也不重开77ac修复链。

具体证据问题是：**在当前accepted DDP的同一次原始调用中，哪些通信输入生产/就绪、提交、完成、后续使用时点可以可靠关联，哪些等待或传输部分仍不可辨识？** 由现有`diagnostic_author`自主选择最小有界观测和实现，不人工指定collector、group、bucket、算法或优化代码。这是新的通信证据获取案例，不是reviewer叠加、prompt对照或重复优化直到通过。

输入只有原始任务、adapter、精确accepted `ccd90dc1`代码、原始源码、现有accepted-version timeline/calibration/tuning原产物、路径/SHA/大小索引、通用worker接口及上述问题。索引指向原始B/C完整evaluate请求/结果、只读controls和物化source。不给新selector答案、77ac/5e3a失败实现、CPU合同反例、人工终态审计或因果表。先前source/metric mapping人工辅助边界仍保留。

作者仅生成`diagnostic.py`与`diagnostic-plan.json`；本控制器没有GPU/model/真实collective、候选、参数搜索、collector执行、验收、source/controls重跑、reference复制或指针发布。CPU/API/syntax/mock允许且保留失败，不能当完整CUDA质量；实际工具必须核查。生产core/prompt/model/sampler/quality/acceptance不改；本实验的诊断任务提示单独保存，不加入生产角色。

后续若代码审查允许，才另外声明一次固定原字节观测。生成的入口必须每worker委托原evaluate一次，保留两连续完整检查、2warmup/5clean、exact rank replicas、原1e-6/1e-4、完整FW/BW/SGD、两同步、原返回/内部catch与未捕获异常。不得增加模型/collective执行、替换默认DDP归约语义、改变groups/buckets/backend/精度、强制路径或同步制造测量。任何observer同步、hook、allocation、owner/lifetime和clock/join限制须显式解释，未知仍未知；不得将CPU callback等同GPU ready、Work完成等同wire time，或把不同capture相减成critical path。诊断时间不作clean验收。

目标源current1fc/原accepted4541/独立current7a49/历史09b7不变，13 AR/497,759,232B现有计算Graph收益不改称通信接受。Qwen关闭、最新FSDP拒绝、77ac链关闭。角色生成完成后接回全工具/代码/观察范围审计，再确定下一有界步骤；不是自动GPU、自动采纳或无限生成。

本步预测128MiB：原框架冻结约18.6MiB，显式原始观测约1.4MiB，历史小型diagnostic author材料增长约数十MiB，保留8GiB警戒；这不是API或输出cap。实际preflight检查150GiB配额和GPU空闲。无本轮清理，若后续GPU预算不足，须先在worker结束后做授权范围的服务器无损冷存及恢复校验。全部后台SSH/监听隐藏，一个plain completion/disk watcher；原始历史/配置在服务器，不打印凭据。实际冻结/启动/快照另记。
