# 分片排程接受版本的普通优化选择

香港2026-10-31 **20:14:15**正常完成，reason=`batch_finished`。聚焦终审 [2d94b241](evidence/v6-ddp-schedule-communication-selection-20261031/public-terminal-audit.json)完成：37实际输入7,558,327B、六当前关键身份、原prompt/model/tools/core e893及source CURRENT88fa保持。17assistant/28tools/0 schema修复，provider估计0.7565482316，非账单。原selection `4a363faac6ed5d19766ee04f8b6e830e3d96dabf639b7798dfed016eb4db0bd1` /4631B，一次写入无编辑，4groups/10calls/execution_plan。全部命令、实际输出和selection已审，无新模型质量/性能接受。

7641展示行、3明确截断前缀、工具23/24/25/28纯JSON/源码输出逐项重建。20/22完整保存stdout分别1,574,450B/1,573,581B与snapshot原heads/targets精确重建，但角色仅看到返回尾部，没有读取保存全文。实际完整读了当前7e6、原训练、binding、原独立反馈与用户要求、calibration924行和tuning342行；resources仅1995/2853行、timeline仅1705/18296行。search虽只有一物理行，实际前缀截断，不是全文暴露；审计表的all_lines_exposed只计行号覆盖。Nsys JSON打开解析后只输出顶层metadata，未展示kernel数组；ownership顶层摘要、adapter仅13个签名行。旧物理候选源码heads/targets实际打开，不能据此称多active bundle或完整语义消费。

工具7的错误相对路径/缺失previous-candidate目录、stderr隐藏和最终exit2保留。其他目录中opencode.json仅名称。所有28工具未记录模型/tensor/CUDA/PG/真实collective、私有配置或后来审计/评审答案内容命令；外部FS并不封闭，不作绝对缺席结论。私有完整历史/命令留服务器。审计reader初次误要求ls目录名带斜杠inputs/，仅改成实际inputs输出断言，失败reader保留，未重跑角色/探测/GPU。

原selection提出更早的producer边界、首批通信以及按片完成事件启动更新，并考虑搬移1/world缩放。当前代码确实先等完整两层producer段、末尾更新等四片全部完成，这是支持一次实现的具体依据。合成速率/SHM并不证明32ms传输下界或41.5ms结构极限；通信未覆盖计算的区间不等于可回收关键路径，非对称首kernel不证明rank等待。预先折半与SUM后缩放不保证所有FP32范围逐位等价；44.255/44.594是独立pair中位数，不是下一次ABBA噪声门槛；original adapter fallback也不是自动保留优化parent或回滚已更新weights。原预测1.5–3ms保留为假说，不采纳为收益。

已经于香港 **20:20:58**实际启动[一次普通候选生成](v6-ddp-schedule-overlap-candidate-20261031.md)，controller309340。只使用原selection/current与普通原证据，未给上述审计纠正/手写算法；返回后审代码再决定原GPU验证，不追加修复/参数直到通过。旧46500已处理，不重启。以下为本批历史启动记录。

香港2026-10-31 **20:02:40** 已实际启动一次普通 `ProjectAgents.generate(selection)`，controller **307469**，唯一隐藏完成/磁盘监听 **46500**。generation33724c362c75/session ses_e6e28c725ffeuId4jt1xGE5WHB。来源是[新7e6fb7当前版本discovery终审](v6-ddp-schedule-communication-evidence-20261031.md)，不是把辅助分析当优化终点，也不手工指定算法。

source `ddp-schedule-communication-evidence-20261031/ddp-evidence` CURRENT **88fa1d104cd1777b723521cfd9bb37540abfcefbb9a5edb3a04ca3cc21f67057**，新实际registry10call/9group、已独立接受c283f6的精确code7e6fb7a723590a93b289f43fcfef84663fc618b5da6d559362144d178827eed3/35595B和proposal442ac3f39dc9d13f8591f534be804b42b77b625c933a184038de64200a241403/11159B。singlefixed-parallel-schedule、params{}是自身fine8/comm_groups2/tail_splits4；旧物理文件不是多个active bundle。原accepted/formal/来源CURRENT均保持。

现有通用[编排](../scripts/experiments/v6_selection_from_feedback.py) SHAaed0367b02a348347046127a5c55510788ba3847e2191001eda521f7576abe52，直接复用working runtime/core e893。原生产prompt+既有feedback suffix/model/tools、group/joint/region/execution_plan默认scope、leafFalse/agent/full、API observe-only无cap/普通validator repairs保持。只复用ProjectSearch._trial的只读common assembly前段SHA89e9338af6c3d3fffaa63ea88ae16485e2611b9cea668c1864b5c53a31c67f2c，不创建trial/runtime优化工作。controller不生成候选、不跑GPU/模型/collective/search/参数/collector/native/source-control/reference副本或发布指针；Agent实际工具仍需终态完整审阅。

原version-df2217162ef2完整版本反馈82990B SHA0952e57700aa8d256bae7750bd7f9ae2faa6e819365dc1ab1a6db0a497421c17保留，augmented中的全部原字段逐项相同；仅既有显式用户约束/“为什么没有继续推进”上下文补充。正常route4781B SHA481468bc7dd73e3ae8df98529853550205765dcb570ad40af288771c1ec34609，原accepted decision/完整版本信息保留，并不包含所有pair/rank原始质量记录。user-constraints SHA34ae723134b9114d19d36e5e3cf16ad876a2287ffaabbd53327ebf7119240c8e，沿用前一accepted/fresh-discovery上下文。普通current四collector JSON/source/task/ownership/bundle输入；无开发者终审表、corrected planner事实、因果答案、手选group或代码。历史source/metric/CPU/user辅助边界保持，非unaided或single-variable AB。

**20:03:23实际启动核对** [35b3178e](evidence/v6-ddp-schedule-communication-selection-20261031/startup-verification.json)：37实际输入7558327B与snapshot逐字节相等，六当前关键身份/source CURRENT88fa/core/普通prompt-model-tools保持；原feedback与route/hash/accepted decision核对。4assistant/8tools仅startup metadata和read paths，完整实际command/output/probe以及selection语义待终态审计，不预称消费/正确性/访问缺席。FS非封闭，私有配置/完整历史留服务器，严禁输出metadata.output/display/preview。此时GPU为空，controller活跃，df8752332800B约8.15GiB。没有startup读取器错误、model-health、历史sweep或新测试。

启动150GiB quota余8770781184B约8.17GiB；**128MiB预测+原8GiB警戒**依据实际最近完整selector目录69906432/27320320B，使用既有通用forecast参数，非API cap，无生产代码改动/本批清理。后续GPU须重新看实际空间预算。此前16关闭拒绝C旧trace的lossless cold/RESTORE journal7a62edaf保持，冷数据未丢失；low_disk_space不是完成，不盲杀/删，只在GPU结束后按既有授权保护active/source/sharedcontrols/accepted/models/provider/deps做必要cold。

旧53652 evidence已完整终审cd52/review3ffa并handled；7304及更旧批次已处理，短独立无listener，迟到事件去重。返回后先完整审真实selection：只有有依据的新路线才执行一次普通候选生成，再审代码后决定原GPU验证；若重复已拒绝/已关闭前提则保留结果并选择有用具体问题，不重试selector直到得到期望算法。原质量/性能接受规则、闭合链和泛化要求不放宽。
