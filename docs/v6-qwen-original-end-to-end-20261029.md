# Qwen原入口完整生成目标验证

## 三轮搜索终态（2026-10-28 19:54 UTC）

续行控制器正常结束，但**三轮均拒绝，当前没有active bundle**。因此完整版本独立AB/BA和条件式B/C native阶段均not_run；不能把completed说成自主优化成功。搜索内collector曾独立采集baseline Nsight等诊断，这与未执行的最终整版B/C native不同。[终态与全部审计](evidence/v6-qwen-original-followup-20261029/terminal-audit.json)、[控制器结果](evidence/v6-qwen-original-followup-20261029/followup-completed.json)及[自动投递](evidence/v6-qwen-original-followup-20261029/wakeup-delivery.json)保存。

| 轮次 | 真实结果 | 判定 |
|---|---|---|
| `d5c20fcae5` | 31个decode位置Graph、共享pool；worker在observe的teacher-forced logits关系检查抛AssertionError，1,132,088/4,861,952元素超限（23.3%），max abs1.0 | 参数无效，未产生clean样本或paired结果 |
| `e74f5ba6cf` | 真实读取反馈，增加即时logits clone、cache对象身份及回退处理，仍保留31图/shared pool；相同关系检查错误指纹 | 参数无效，未产生clean样本或paired结果 |
| `b8ecdd4254` | 改为单Graph/StaticCache并增加内部逐步数值门控；门控实际禁用Graph，后续parent路径；全量质量通过 | screen可行，但paired TTFT超原锚且增益未超过噪声，拒绝 |

三轮各实际只执行一个`{}`参数点。第三轮screen end_to_end710.62052ms、TTFT24.581114ms。paired parent为721.015889/758.602095ms，candidate为695.556654/766.685883ms，配对增益25.459235/-8.083788ms，噪声8.35978/71.410706ms；candidate TTFT24.253561/26.186706ms，后一端超过原25.2536046ms上限。质量通过不等于性能接受。五个实际clean端点的25样本与10次全量零误差检查全部保留；前两轮缺少clean样本是质量阶段异常停止，不能补造。

第三轮quality阶段计数62次replacement，明确记录1次graph_disabled_numerics、61次parent_fallback_disabled、65次reference调用。源码的disabled状态在此进程后续测量持续，支持其主要执行parent；空`fallbacks`字典不代表没有回退。没有把该候选当作真实Graph加速。第三轮内部Graph/parent数值门控失败的具体张量差异未在旧结果中保存，不编造唯一原因。

10个真实搜索角色均fresh/0schema修复，原core、六份baseline源文件、request/contract、两个完整参考SHA与原准备指针保持；实际工具读取与反馈输入已索引，未发现对已列旧成功答案路径的引用。只有初始discovery，没有接受后rediscovery、替换或bundle共存。10个会话完成不等于候选质量通过。最新feedback文件约155.5MB，文件大小不能代表模型实际上下文；审计保留逐角色输入/工具输出字节统计，不据此盲目扩充或裁剪上下文。

两个重要证据边界：其一，前两轮异常发生于`observe()`关系检查，早于外层对原参考的compare返回，因此现有错误不能证明“greedy capture已匹配源，而只有replay错”。其二，四次candidate_diagnostics虽执行passed/valid=true，实际均不调用observe/compare，输出已有明确“no quality or performance acceptance”scope。Agent将一次诊断执行成功作为“质量曾通过、因此可能随机clobber”的依据不成立；不能据此认定非确定性、alias或shared-pool唯一因果。亦不把clean wall减另一次profile CUDA总时长认定为精确host开销，或把prefill代码未改当作TTFT必然合格。

首次只读审计将framework_source误读为started记录字段，实际字段在rank结果；修正后完整复核通过，未重跑GPU，[错误说明](evidence/v6-qwen-original-followup-20261029/audit-attempt1.json)保留。原入口准备、外部手工映射边界、旧同adapter46.9421%及全部历史接受/拒绝互不覆盖。

下一步仅补已证实缺失的关系检查输入，按[固定输出诊断协议](v6-qwen-relation-diagnostic-20261029.md)比较greedy/teacher/独立源，先建立数据再决定是否需要改候选信息。生产prompt/sampler/allocator/接受规则不变，不重复自主三轮直到通过。

## 续行已启动（2026-10-28 18:01 UTC／香港10月29日02:01）

冻结`a80d8f8b81adb24ea43960b2876531a82dc12335`，控制器PID757316、PPID1、独立session；生产核心仍ed24af7a，OpenCode自动服务健康检查通过。新目录`qwen-end-to-end-original-followup-20261029`先完成外部映射比较，再通过既有API复用原准备，18:01:53 UTC开始此前未执行的搜索。[启动](evidence/v6-qwen-original-followup-20261029/launch.json)、[协议](evidence/v6-qwen-original-followup-20261029/protocol.json)已保存。原错误批次没有重启。

[独立源比较](evidence/v6-qwen-original-followup-20261029/reviewed-source-check.json)由实际CPU worker执行，复核真实CUDA controls保存的两个完整输出：每个4,861,984元素、logits零误差、exact tokens、finite logits通过；token+1和单个超阈logit均被拒绝。比较使用原0.02/0.02，完整tensor无抽样。它没有重新运行GPU，也不独立认证额外观察字段或所有内存所有权。

[复用身份](evidence/v6-qwen-original-followup-20261029/preparation-reuse.json)确认原baseline009b4178、六个文件、adapter498c90f3、request919166b1、contract52c861f1均不变，空bundle、没有初始feedback，controls路径仍是原批次。原current指针不动，不复制大参考，不给候选提供手工诊断。

18:03:34 UTC的[实际活动](evidence/v6-qwen-original-followup-20261029/startup-verification.json)显示真实collection-planner会话`ses_e7c51ef5dffe4vpnT09RUAmASg`已开始，有5条assistant消息，初始快照`generation-a61334228721`。这只是自主搜索已运行的证据，不宣称新候选或独立收益。

唯一普通完成/8GiB磁盘监听PID55228已核验waiting；状态`.scratch/v6-qwen-original-followup-wakeup.json`、同前缀`-wakeup-launch.json`。见[设置](evidence/v6-qwen-original-followup-20261029/wakeup-setup.json)与[核验](evidence/v6-qwen-original-followup-20261029/wakeup-verification.json)。上一批监听已经完成投递、标记handled并写stop文件，不恢复。当前实际df剩16,858,550,272字节（15.70GiB），无需清理；没有本地raw转存或模型等待轮询。

## 终态与外部审计映射（2026-10-28 17:53 UTC）

原批次以`completed_with_errors`结束：真实intake/evaluator/controls完成，独立源检查返回`observation_schema_requires_manual_audit`，因此search/independent/native均为not_run。不是数值失败，也没有新优化接受。旧监听已实际投递。[终态](evidence/v6-qwen-original-end-to-end-20261029/followup-completed.json)、[原始检查](evidence/v6-qwen-original-end-to-end-20261029/source-check.json)、[完整准备审计](evidence/v6-qwen-original-end-to-end-20261029/preparation-audit.json)保留；下文原运行状态是历史时间点。

新adapter SHA为`498c90f3f6f91b993ac8a586ac3230ffe2e6901752165b26948142d978f7cff7`。实际`observe()`把`run()`返回的tokens称为`generated_tokens`，完整stacked logits称为`step_logits`；外部旧fixture期待`tokens/logits`。源代码人工核对确认只是观察名变化，没有切片、归一化或少做工作。每次tokens形状`[1,32]`、int64；logits形状`[1,32,151936]`、BF16。另有token计数及teacher-forced replay关系检查，需与独立源数值比较分开报告。

两个真实角色初始快照`generation-744292500593`、`generation-1cf1bf54245b`均为fresh，core一致，各0次schema修复。工具输入和初始文件已索引，针对旧优化/测量路径的检查未发现引用；不把此范围检查夸大为整个外部文件系统隔离证明。intake得到正确完整目标与TTFT原baseline比例约束。新completion文字正确描述TTFT和完整生成，不继承旧variant里“ttft_ms objective”的表述。

新controls五个clean样本均保留，完整生成median **690.120524ms**、TTFT **24.051052ms**，原锚上限 **25.2536046ms**。reset两次和identity全量比较零误差，错误输出与缺失覆盖负例通过。这些是基线控制结果，不是优化收益或独立源接受。

代码审查确认完整prefill+31次cached decode、固定32greedy tokens、BF16/SDPA/noTF32、源prompt及两计时结束边界。teacher-forced完整replay在observe中、计时外执行。prepare持有model/tokenizer/inputs和CPU元数据，无adapter新增的CUDA参考logits/cache持有者；每次run的cache/output是局部量，quality结果持有完整logits，clean结果只持有tokens与CPU metrics。框架invocation模式在warmup前释放末次质量输出，并在每次metrics后释放clean结果。源clean代码在peak查询前已丢弃返回tokens，框架保留至metrics；二者计算峰值均包含tokens，但不能宣称framework allocated/lifecycle与源内存统计完全相同，亦不跨adapter推断显存优化。阶段annotation有instrumentation成本，不当作零开销。

下一步在新目录`qwen-end-to-end-original-followup-20261029`进行外部、按已审代码固定的映射比较，然后继续原先未运行的三轮。新控制器`scripts/experiments/v6_qwen_original_followup_20261029.py`只在实验审计里映射`tokens→generated_tokens`、`logits→step_logits`，先核对adapter SHA，再用独立CPU worker完整比较新CUDA controls与原参考，含exact token+1和超阈logit负例。CPU比较不冒称重新执行模型/GPU；不重建参考、不放宽阈值、不删样本。

仅源比较通过才用既有`fork_run`复用已完成的fresh原入口preparation，原adapter/request/contract/controls和空bundle基线不变，共享只读参考；既有CLI resume执行发现及三轮真实搜索，独立/原生路径检查沿用原协议。此处是**外部审计映射经人工核实后的续行**，不是另一个独立evaluator cohort，也不能称为完全无人工间断的整批流程。未给候选Agent提供映射、诊断或旧答案。原批次的失败终态不覆盖。

[CPU接口预检查](evidence/v6-qwen-original-end-to-end-20261029/followup-preflight.json)通过导入/语法/help/执行器schema；真实数值比较由续行的第一阶段执行。生产核心、提示、sampler、allocator和接受规则不变。当前实际df余约16.26GiB，GPU已结束，没有新增清理或本地raw转存需求。

承接[同adapter完整生成目标](v6-qwen-end-to-end-20261028.md)的46.9421%独立收益，下一项问题是：不提交现成adapter/request，框架能否从原项目、启动命令和目标自行得到完整且忠实的评测、分组和优化结果。目录`/root/autodl-tmp/dev/v6-work/qwen-end-to-end-original-20261029`，控制器`scripts/experiments/v6_qwen_original_end_to_end_20261029.py`。

输入只有原`examples/v6_validation`项目、原Qwen推理命令/模型资产路径、目标和设备0。目标仍为batch1/prompt256/full32 greedy token的`end_to_end_ms`最小化，`ttft_ms <= 1.05 × 本任务新原baseline`。BF16/SDPA/noTF32、完整logits0.02/0.02、exact tokens、两同步边界及源生命期要求保持。不给旧adapter、已解析request、手选组、旧候选、优化答案或前一批测量结果。自然语言约束是任务输入，不是向Agent指定Graph或其他实现。

使用既有CLI `--project --command --goal --prepare-only`运行全新真实intake/evaluator/controls。随后外部独立源fixture核对新controls的完整tokens/logits；它使用原始参考而非旧优化版本，结果不进入候选Agent输入。新增observation需要单独审查；命名/容器差异保留为schema/执行问题，并停止依赖搜索，不能自动映射、当数值失败或自动接受。源检查只覆盖可比较值，不替代对返回状态、prepare持有对象、计时与内存边界的代码审查。

准备和源检查通过才使用同request `--resume --rounds 3`。这是新定义的原入口cohort，不是重复旧候选直至通过。保存每次初始snapshot、调用/回复、实际参数与路径、质量、原锚约束和反馈；API只统计不加使用额度。未知错误冻结，既有schema修复由框架正常执行，不加入失败类型硬编码。

有自主接受版本才做新合同内独立完整AB/BA；无版本记not_run。独立完整质量通过后才采集单独baseline/candidate Nsight，核对实际路径，侵入式profile不作为分数。搜索接受、源一致性、独立性能、native证据、控制器完成分别判断。不得宣称多个active bundle，除非实际证据存在。

生产核心继续`ed24af7acc8caa15578e0b859c85ef8a901be51354448ed39094606ed9e10a2f`；不调整prompt、sampler、allocator或接受规则。旧同adapter合同completion的“ttft_ms objective”文字局限保留，本次由新evaluator生成合同，不人为修复旧记录。

启动前真实df余约16.29GiB，上一批GPU与服务器内压缩均结束并核查。单GPU串行，不复制旧controls；新evaluator必要的新参考另行记录。仅一个普通完成/8GiB磁盘监听；低空间不是终态，不停止活跃任务或删除证据。raw留服务器，不恢复旧转存/监听。没有Goal、模型轮询或额外投递测试。历史Qwen、FSDP、DDP、ResNet、TP结果及全部样本保留。

已冻结`16e38722c4482a96da7bcd3400f9d3076f0d5b26`，2026-10-28 17:46:18 UTC（香港10月29日01:46）启动。控制器PID754329、PPID1、独立session；OpenCode server_url=null自动启动服务健康检查通过。CPU导入/语法/CLI原入口参数检查通过，生产163文件core SHA不变。

17:47:39 UTC核验处于prepare，真实`project-intake`会话`ses_e7c613ce3ffe5y0Y4f44bUi4ob`，新快照`generation-744292500593`，已观察到5条assistant消息。GPU空闲符合intake阶段，不是卡住；不提前声明evaluator/source或优化成功。见[启动](evidence/v6-qwen-original-end-to-end-20261029/launch.json)、[协议](evidence/v6-qwen-original-end-to-end-20261029/protocol.json)、[真实活动](evidence/v6-qwen-original-end-to-end-20261029/startup-verification.json)。

唯一普通完成/磁盘监听PID48148核验为waiting，状态`.scratch/v6-qwen-original-end-to-end-wakeup.json`，启动参数同名前缀`-wakeup-launch.json`；每30秒SSH/df检查，终态或8GiB空间告警投递一次，不用模型等待。见[设置](evidence/v6-qwen-original-end-to-end-20261029/wakeup-setup.json)、[waiting核验](evidence/v6-qwen-original-end-to-end-20261029/wakeup-verification.json)。上一批监听已投递且标记handled并写stop文件，不重启。实际df剩17,483,948,032字节（16.28GiB）。本机监听进程需要本机在线，远端任务独立运行。


补充固定诊断已完成，见[关系输出审计](v6-qwen-relation-diagnostic-20261029.md)。C1/C2的greedy在进入observe前已偏离源（max1.0），teacher与源精确相同，greedy在observe前后不变；U/O都保留原失败。旧三轮零接受不变，尚无唯一底层原因。下一项只在原第三候选生成前做真实角色输入对照，不重新开展搜索直到通过。
