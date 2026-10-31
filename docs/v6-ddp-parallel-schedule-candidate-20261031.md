# 已接受双通信组版本的后续排程候选

香港2026-10-31 **19:20:51** 正常完成，reason=`batch_finished`。本次聚焦终审 [2f788497](evidence/v6-ddp-parallel-schedule-candidate-20261031/public-terminal-audit.json) 完成：39输入8,988,022B、原35非反馈文件、12当前关键身份、普通prompt/model/tools/core e893保持；28assistant/39tools/0 schema修复，provider估计1.14335997286（非账单）。code一次写入无修改 `7e6fb7a723590a93b289f43fcfef84663fc618b5da6d559362144d178827eed3` /35595B，proposal一次写入无修改 `442ac3f39dc9d13f8591f534be804b42b77b625c933a184038de64200a241403` /11159B。全部命令、输出和新代码依赖已审；5058显示行、4截断前缀和8个JSON/源码输出或失败stdout核对，完整历史仅服务器。

实际读取当前9bdd、原训练/adapter/binding/selection/feedback，timeline/calibration有打开、tuning前缀/resources摘要；本次未打开nsys和device_telemetry JSON。工具19在输出两rank通信后因rank1的devices['0']报KeyError；28假定队列模拟超时，29误将每片算成整个尾部，30只修正模拟尺寸。这些是CPU算术假设，不能证明收益；31库API、33整数排程覆盖、36不支持CPU路径回退、38schema、39语法均不构成原GPU质量。没有记录到私有配置/后来答案、模型训练、数值tensor、CUDA工作负载、PG或真实collective命令；cuda.is_available查询不是GPU正确性，外部文件系统不封闭。

新增helper按实际尺寸/顺序/通信槽生成分片和排程，没有新增实验路径/hash/模型名/答案分支；仍继承原12block任务程序专用结构，不是任意模型实现。默认fine8/2组/4尾片，尾部四片各39,383,808B；预计8→11次SUM，总497,759,232B不变。每片Work.wait在对应通信stream完成事件前；整chunk更新等待全部片完成和原参数消费者事件。小chunk通信及更新后移。共享pool/改变update顺序、copied inputs、有限guards、group初始化早于peer一致性、异常后只恢复grads不恢复weights等条件性限制保留，不追加故障注入或修复链。原full evaluator是质量依据；本地gate和CPU模拟不是分片通信证明，预测2–4ms/3–8ms、wire/等待归因不采纳。

已明确选择并于香港 **19:27:50** 实际启动[一次原B/C验证](v6-ddp-parallel-schedule-fixed-20261031.md)，controller296856。空参数使用候选自己的默认值，未人工修改算法、分片或参数；原质量/性能规则保持。旧45380完成事件已处理，不重启旧批次。以下启动记录为历史。

香港2026-10-31 **18:55:34**实际启动一次普通候选，controller **293560**，唯一隐藏完成/磁盘监听 **45380**。源是最新独立接受9bdd及实际新registry，依据刚完成的原项目selection提出的排程假说继续优化；不采用其预估收益/等待因果作为事实，不增加人工分片、排序或算法实现。

[实际启动](evidence/v6-ddp-parallel-schedule-candidate-20261031/launch.json)。使用现有 `ProjectAgents.generate(candidate)`，普通生产prompt与既有feedback suffix、model/tools、API observe-only无cap、正常validator repairs。controller仅生成及schema，不含GPU/model/collective/search/parameter/collector/native/source-control/reference副本/指针发布/接受。实际Agent工具必须终态检查，意图不是无探测证明。返回后审精确code/params/dependencies、数值和泛化范围，再决定必要的原GPU验证；没有自动修复或fine9/coarse5/组数1/3/4循环。

源 `/root/autodl-tmp/dev/v6-work/ddp-parallel-communication-evidence-20261031/ddp-evidence` CURRENT `eb8cd90563b6c4dea210b819d5fff65a82cd80818bbde076d300c71c0b68a214`，新10calls/9groups，单一fixed-pipeline-parallel/空参数。现有代码 `9bddb13647eac850b561c1707640e75255f5e9dabf29e1bf241e83a090b5185f` /30949B、proposal `94d72e386e9bd229da73f6c46934c352103cf971e80c26b2a588d2eefcfa3cec` /8069B，自身fine8/comm_groups2。旧物理文件不是多active bundle；current/独立accepted b769/formal187c/旧0bc73ac928db等保持，没有新discovery/source-control或reference副本。

原selector generation77f50f443211 / session ses_e6e715c7affeyVGC1ihzC9L5v1 的36输入为基底；原35非反馈文件逐字节相同，普通添加selection/targets/target-resources，**39实际文件8,988,022B**，12当前关键文件。原selection `12033944e081ce35bc89aa864c24e3bdc99ba9a26ab059bb89609b14c4fccd01` 只做现有containment/sort canonicalization，canonical `b4d993db17a24b280d6c52db79f7a6d23f42468b9446d13caae1e5d911844a51`，4groups/10calls/rationale/hypothesis/scope保持，并非原JSON逐字节相同。原反馈 `f32b54b717993be4c22e310ba9d32c9afbd8a97ec099079b2181947b9b015245` 不变，按selected-group普通route `7977e38951c684c9017f3b505645ec27b14b88660cd79403e101bcf4189cfe52` /4813B。旧接受decision属于9bdd，不是新candidate质量。已有用户要求supplement保持，没有新analysis、用户要求、终审表、协议答案或手写实现。既有source/metric/CPU/user辅助仍在，不称unaided或同snapshot A/B。

复用现有工作runtime `/root/autodl-tmp/dev/opop-v6`，core `e893f850dff70908e6100b7f6cc0773fbeac03e201759d688eda4d3682b3d0d8` 与selector相同。[通用实验编排](../scripts/experiments/v6_candidate_from_selection.py) SHA `a53979eb3703fa34ca9d38ba5cda0051ba9da0487f602c1ebbf3fe5458ef5c3b` 仅新增可选 `--forecast-bytes`：默认384MiB保持、要求正值、8GiB警戒保持，protocol记录本次预测。它没有case分支，不改生产core/prompt/model/collector/quality/acceptance；语法和实际preflight通过，没有新增模型健康或冗余测试/环境副本/版本门槛。

启动150GiB盘余8,847,880,192B约8.24GiB。近期两次完整候选root实际73,924,608B与20,754,432B，本次[显式192MiB增长预测](evidence/v6-ddp-parallel-schedule-candidate-20261031/disk-budget.json)+8GiB警戒通过；这是预算，不是API cap或运行量限制。初步考虑的冷存没有实施：当前剩余plain旧FSDP metrics位于source/baseline批次，保留不动。没有清理或杀worker；以后GPU需要重新按实际空间预算，不能直接沿用本次余量。已授权无损冷存只可在GPU结束后保护active/source/controls/accepted/models/provider/deps，原RESTORE journals保持。

香港 **18:57:06** [启动实核](evidence/v6-ddp-parallel-schedule-candidate-20261031/startup-verification.json) SHA `c623f0c42727d5f59f182567f2f57aa0e4127dd67396090d7d298aec6cac9e0f`：generation **c13a0147a674** / session **ses_e6e662803ffeY7VxdH96qMHEDd**，39实际输入逐字节一致、原35非反馈相同、12当前关键文件/sourcecurrent/core保持、普通prompt/model/tools相同。9assistant/16tools只做metadata/read path记录，完整输出/探测/code审查待终态，不能预称无外部访问或已正确实现。private early history `040ade1e4b693455377fb0762fdf9c596b76f71cd832c6ed628ccd2b7fd55179` 仅server，未输出metadata.output/display/preview。GPU空闲，df8,826,101,760B约8.22GiB；controller运行中、无startupreader错误/重启。

原数值/全step要求保持：2连续full gradients/updated params/loss finite/exact replicas（atol1e-6/rtol1e-4）、2warmup/5clean/alignedmaxrank-before-median-MAD、原anchor133.379621/limit140.04860205，不重新设anchor或内存约束。先前质量首步parent、第二步setup并非两个optimized checks的范围限制保持；source证据只复用先前实际原设备比较/replica-negative记录，不新声称source等价。若实现引入复制tokens、缓存/PGlifetime、ready/update依赖、fallback或泛化错误，按实际代码处理，不以schema或CPU toy当全GPU证明，也不自动追加修复链。

原正式9bdd相对02b9335.6824%、完整版本独立对EMPTY58.4625%保持，不相加；本次尚无新代码质量/收益。新原selection已完整终审f0488777，旧evidence e92a/independent6448/formal6c3a无需重做。旧41700、32992、39344、45968及更早事件handled/迟到去重；只一个隐藏新listener，SSH/queue CREATE_NO_WINDOW，无model polling/delivery tests。Qwen/FSDP/77ac/observer/eef828关闭链和所有拒绝保留。
