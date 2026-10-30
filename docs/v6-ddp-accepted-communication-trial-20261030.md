# DDP 已接受版本重发现后的一轮普通通信续行

终态已审计：2026-10-30 12:40:28UTC（香港20:40）`batch_finished`，不是磁盘警报。controller已退出，两卡空闲；实际150GiB配额约8.54GiB可用。唯一旧监听13432标记handled，不重启，不清理。

原trial `trial-02ed5436b4`、代码 `5e3a9a001ba4c793c602f2b066cb12789550cf09e900cd6518ba821deaefef26`（17979B）、proposal12488B、空参数。仅一个screen，尽管默认参数预算3；execution_plan项目组自主选择，单bundle替换原`trial-d06aa22e2d-param-a72285d2`，不是叠加或共存。新revision01cbc402未发布；current/source evidence仍1fc24545，原accepted4541、independent7a49与历史Graph09b7不变。生产core仍ed24af7a。

| 配对 | 父版本中位数ms | 候选中位数ms | 收益ms | 噪声门槛ms |
|---|---:|---:|---:|---:|
| AB | 71.284842 | 108.537192 | -37.252350 | 4.525422 |
| BA | 72.901868 | 111.151905 | -38.250037 | 0.450708 |

原规则拒绝`paired_gain_not_above_noise`和`objective_not_improved`。描述性聚合72.093355→109.8445485ms，不是收益。原约束140.04860205ms通过，没有重定锚。Screen111.477247ms，124.480528ms尾样本完整保留。没有独立阶段，也不将质量通过改写成性能接受。

9 jobs/18 ranks均status passed、valid true；5 evaluate端点共50个clean rank样本、20次完整零误差检查，每次248,879,620元素，原两次连续更新、2warmup/5clean、完整梯度/参数/loss/finite/exact replicas/math/同步边界保持。每次先对齐rank取最大再median/MAD，两个paired decision均按原代码重新计算。候选质量路线均为第一次`parent_eager_baseline`、第二次`fallback_setup_failure:RuntimeError`，reference两次；framework fallback空字典不能抹掉内部回退。父版本才是第二次`graph_setup_capture`。

两次run-only candidate_diagnostics、各两rank记录同一个具体初始化缺陷：`_setup`行148的`sentinel_idxs = [params.index(s) for s in sentinels]`触发`Boolean value of Tensor with more than one value is ambiguous`，再经replacement行396传播。每worker一异常两frame，不是两次失败；四个独立worker的记录分别保留。该行在私有梯度buffer、Graph捕获、并发/串行基准之前，拟议机制未到达。Run-only不执行observe/compare，不替代上述真实完整质量检查。`context.references`在替换旧bundle后指向原始source，故实际fallback不是proposal所称“已接受顺序Graph/零回退损失”。没有手改代码或额外GPU测试。

全部30条post-clean通信trace逐条按各自窗口重新归一化，每rank13 AllReduce Sum f32 RING_LL，逻辑497,759,232B（12×28,351,488+157,541,376）。两组B/C诊断中位数37.682621/37.751108、36.440123/58.542296ms，各为独立profile时期，不能与clean耗时相减或称wire/critical path。候选post-evaluation Torch profile每rank3999 kernels/1525 calls/858 groups，0 Graph launch；独立timeline工作窗口亦3999 kernels、0 Graph launch，原始全trace另含窗口外kernel。没有新Nsys，旧ccd90 native不作为新候选证明。实际NCCL日志channel SHM/direct/direct；NODE/P2P不是传输证据。

候选runtime峰值1981029376B、lifecycle2479586304B，高于父Graph1498647552/1999242752B；reserved lifecycle候选2589982720B低于父3193962496B，计数分别保留，不追加内存接受规则。安装约10–15ms不代表第二质量调用中的完整setup耗时。

真实selector/candidate/planner初始文件41/44/8，字节50,078,774/54,492,258/270,849；实际route、原历史代码/sidecar、模型工具prompt/core及snapshot字节验证。40/25/14 assistant、44/36/17工具，0 schema repairs；provider估算2.67188117102，仅观察值非账单。完整97工具输入输出、私有sessionparts和脚本保留服务器。候选一次write，无edit；只做库API检查（CUDA不可见）、syntax/AST/schema，没有实际CPU模型/collective或GPU探针，不能把这些检查当运行验证。

Selector实际打开当前timeline/calibration/tuning及旧telemetry，候选也读了部分当前timeline；不再是仅索引曝光。但其“校准hard floor/不可减通信量”、CPU callback按序配kernel作为GPU ready、跨profile拼出clean71.8ms并估算4.7ms开销等推断未采纳。候选私有buffer并发合并是否满足容差、有限sentinel/identity guards、global override恢复False、rank非对称setup和部分step回退仍不构成通用安全/确定性证明；本轮未到达这些路径。

Planner选择candidate_diagnostics、execution_timeline、binding_cost；前两者真实执行，binding_cost返回`not_applicable: passthrough requires additive bundles retaining the complete parent; replacement cost is unknown`，没有binding jobs或成本值。Planner把replacement误称additive、从另一profile的均衡时间排除原因、声称timeline能归因clean回退差值，均不采纳。其读取自身provider配置/git元数据的完整输出仅服务器私有保存，不纳入公开摘要。全记录未观察到外部后续accepted/reviewer/审计答案读取；非hermetic文件系统不能证明绝对无外部访问。

所有read输出逐行比对对应文件：selector1962/candidate5909/planner2166行，3行原工具明确截到2000字符。审计初次错误地要求截断行全文相等，原错误保留，reader改为验证已显示前缀；没有Agent/模型/CPU探针重跑。初版审计adcfdcc4也保留。Launcher旧脚本名预检失败在实验启动前，原记录保留。原样本/错误/代码均不修正。

最终[公开小摘要](evidence/v6-ddp-accepted-communication-trial-20261030/terminal-audit.json)，完整服务器审计SHA `0ce0048441d64d0dd4fd20dcf52463b83561c24a91f4cefa07d0518a7f542131`，公开SHA `2a621fc7805a7385d23148d1d8a552c7f2b5d6c0b6e9158fa5304b0245978b03`；numerical SHA `5d1133b887c73208c82384f1b0737efe7e0bc5bacbf3e7716961e6f732f07954`。647冻结文件、实际materialization/source/adapter/entry/core/UUID、每worker请求与旧指针均验证；无新source比较，既往人工source/metric mapping边界保留。

下一步：[一次普通selector决策](v6-ddp-communication-selection-20261030.md)，仅更新原始失败反馈和历史候选/原始诊断，当前accepted source/registry不变。先看其是否提出有证据依据的后续假设；不自动生成修复候选、执行GPU、重复验收或增加reviewer。Qwen关闭、FSDP拒绝保持。

以下为启动协议及历史记录。


实际启动：2026-10-30 12:03:42 UTC（香港20:03），冻结 `2a0bb13a7ab8f46b27c58d77f3c38f81521f5a1a`，controller145514，唯一隐藏监听13432；旧50252已退出并handled。真实local OpenCode start/health/stop通过，server_url=null，无模型测试prompt。首次部署launcher错误地校验旧脚本名，已在创建新root/controller/session/model/watcher前中止；`launch-preflight-attempt-1.json`及原脚本保留，随后只修正部署路径，精确冻结代码/生产core未变，没有实验重启。

12:05:30 UTC启动审计SHA `96f1586200fe20d45d99f5e6e68b504a8af906fa8a7fd15ddbf6b4686538f5bd`：647冻结文件、source/branch current1fc、原request/prepared/discovery/controls/core一致。真实selector generation-deeff48fed0b/session ses_e734df353ffew5IaE4RafG3kbI，41初始文件50,078,774B逐字等于snapshot，原route反馈15,504,256B/SHA84e77ac656c0a7b818be825c89653ca31c9807f91b1bf60d55e72ee7bb312b31，supplemental=null；历史ccd90与实际installed_bundles/current source已交付。与此前普通feedback selector的prompt/model/tools相同，14输入相同、17变化、10新增、1删除，不是整个角色单变量A/B。

早期12 assistant/16 tools，私有history SHA a457fc9df023df57ce90462944a5a02f3398d0dc7455e54ceb2d3f0819d50d4b。tool11实际打印旧accepted质量/配对噪声/约束及parent+Graph setup路线；tool12打印installed bundle身份但affected_calls截断；tool13打印当前三collector索引，tool15读取候选代码。最初16工具尚未打开当前collector JSON正文，不能将交付/索引当作消费，也不能据早期状态预判最终不读。早期只完整复核11/12/13输出，其余参数/状态及部分输出；终态必须完整审计。详见[启动记录](evidence/v6-ddp-accepted-communication-trial-20261030/startup-verification.json)及[早期读取范围](evidence/v6-ddp-accepted-communication-trial-20261030/startup-tool-review.json)。当时无GPU jobs、两卡空闲，150GiB额度实际余9,812,717,568B约9.14GiB；角色仍运行，没有新候选质量或接受结论。

声明一轮有界试验，根目录 `/root/autodl-tmp/dev/v6-work/ddp-accepted-communication-trial-20261030`。上一[证据批次](v6-ddp-accepted-communication-evidence-20261030.md)已经刷新已接受执行的registry，并取得实际bucket、tuning、提交时序和校准材料；通信仍13 AllReduce、497,759,232B逻辑输入，计算调度接受不等于通信成功。下一步由现有普通Agent自行判断当前证据支持的范围和候选，不重复旧失败、不承诺一定选到通信组或得到收益，不开启无限候选链。

源为 `ddp-accepted-communication-evidence-20261030/ddp-evidence` 的1fc24545217087adefa1248704619fcdc882a19057336b231d8e02918bf4d0d8；原代码ccd90dc158788a164d6fda242572530e9cdc9ca50563af1a1d9d597f0ac12e0b、单bundle trial-d06aa22e2d-param-a72285d2、空参数、contract保持。原accepted source4541、证据source1fc、独立current7a49和历史Graph09b7保持；隔离新分支可按原接受规则演进。采用existing fork_run和CLI resume rounds1，默认scopes/parameter exploration，API observe-only/no caps。不重复初始GPU discovery/source/controls，不复制大参考。

原框架accepted trial-d06aa22e2d feedback逐字40,982,219B、SHA35bcf4de383882fee05ba7fe66e8afa6dbdd78a54999587872f3c4dbd0808494，零新增bytes、无supplemental。现有route针对新registry执行，历史candidate/原sidecars按普通接口打包；当前discovery的collector材料另按原common inputs交付。没有开发者planner错误表、独立终态表、因果答案或手写实现。现有selector输入本来包含search.installed_bundles/source；必须核验实际snapshot、读取、恢复/替换关系和工具行为，不能以交付代替消费证明。整组输入变化，不是单变量A/B或无辅助原入口。

完整原DDP任务、2连续质量/2warmup/5clean/rank、全梯度/参数/loss/finite/exact replica、1e-6/1e-4、原133.379621→140.04860205ms约束、rankmax先于median/MAD/noise保持；新设备20e8003c/a24af2e6、adapter b276a7ed、原controls只读，生产core ed24af7acc8caa15578e0b859c85ef8a901be51354448ed39094606ed9e10a2f不变。原始source/metric映射辅助边界保持，不新增FSDP式显存约束或跨设备anchor。

不手选组/bucket/算法/collector，不修改生产prompt/sampler/质量/接受，不并行GPU，不自动安排independent/native；真实planner可选已有collector。完成后审计完整工具输入输出、初始代码及编辑、所有参数/worker/raw trace/尾样本。仅真实新接受才声明独立整版B/C/C/B；通信结论另需实际通信机制，不能以Graph或较低NCCL sum替代。拒绝/异常原样保留，不重复到通过。Qwen关闭、FSDP最后拒绝保持。

空间依据：旧普通trial262,836,224B，feedback640,086,016B，本次刚完成证据125,440,000B，均为实际allocated du；现余约9.28GiB。预算1GiB增量、8GiB告警线，属于估算而非API/collector cap；本轮不清理/复制参考。低磁盘不等于完成，不能终止活跃实验或盲删。新批次仅一个隐藏普通completion/disk watcher，旧50252处理后不重启；所有后台/SSH/queue显式CREATE_NO_WINDOW。raw/private configs/history留服务器，代码/hash/小型metadata进GitHub。


审计输出处理补记：历史planner曾读取自身provider配置。审计者查看完整工具记录时，未先过滤这一输出，导致其中私有配置意外进入工具回显。未写入公开JSON、代码或GitHub；后续回显已屏蔽配置内容，建议更换相关API凭证。该补记不修改冻结原记录、实验输入、数值审计或当前运行配置；没有擅自轮换凭证。此处不重现任何凭证。
