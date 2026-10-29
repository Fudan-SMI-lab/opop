# 原生成诊断脚本的固定GPU验证

已执行并审计完成。冻结`ad3c962c54912c9ff7ff898b85ebd7f0b4374597`、controller932124于01:00:31 UTC启动，01:01:09 UTC结束，两个worker都已回收。监听启动预检发现批次已终态，因而在创建后台监听/排队消息之前退出；本轮直接审计，没有重复通知或重跑。见[启动记录](evidence/v6-diagnostic-author-validation-20261029/launch.json)、[完整小型审计](evidence/v6-diagnostic-author-validation-20261029/terminal-audit.json)，服务器审计SHA `aefb611a7a19da32c1896ae1acb9c12431f219719f2d8e06008d2aa746c87f37`。

原样生成代码真实可执行：baseline通过两次完整质量及5个clean样本；候选仍抛原teacher token关系异常，无clean样本，未变为valid。两臂各保存4份完整greedy/teacher输出文件，共8份，每份tokens为1×32、logits为1×32×151936 BF16。独立CPU逐份重算：baseline全部、候选两次greedy和第一次teacher均与源精确相同；仅候选第二次teacher有12个token差异，位置1/3/6/10/14/19/22/23/24/27/29/31，logits max31.125，prefill位置0仍精确，从decode位置1开始偏离。全部finite，CPU float32相对source容差外4,596,832元素；不冒充原GPU BF16断言计数。

生成索引自身的两对关系、差异位置、逐位置max和前三个差异位置的top2数值均重算一致。第一步两边相同、第二步teacher错误的核心证据由Agent自行设计的探针获得，不是开发者替它补hook。源全量逐份核查仍是外部审核步骤，不能把该人工核查写成生产框架自动完成。索引中独立参考只比首greedy、没有after-observe快照的限制仍存在。

两臂事件各记录原evaluator入口一次；baseline为9次run（2quality+2warmup+5clean）、2observe正常返回，候选为2run、第一次observe返回及第二次observe异常。8个hook均报告恢复。候选记录tracked prefill2/decode62、untracked prefill2，符合原代码的阶段和绑定筛选限制，但只是callback证据，不是native证明或“teacher全部经过候选”。原异常类型/消息在持久化后保留，失败端点没有外层checks返回，不能补成完整外层检查数组。额外CPU持有/同步可能扰动生命周期，本次无性能验收、无唯一原因结论。

25个原文件、probe/plan、source/controls及两个current指针哈希不变；actual entry/provenance与两份ExecutionRequest差异符合声明，生产core ed24af7a保持。所有原始文件留服务器；df余10,553,479,168B（约9.83GiB），没有本地raw转存或清理。该结果支持**单案例诊断作者可行性**，尚未证明通用闭环；下一步按[FSDP不同任务诊断作者协议](v6-fsdp-diagnostic-author-20261029.md)检验分片训练与“正常返回但质量失败”，先只生成/审查，不自动GPU执行或生产采用。以下是冻结执行前协议。

本阶段承接[真实诊断作者审计](v6-diagnostic-author-20261029.md)，检验Agent原样输出能否在真实框架完整评测中保存所缺证据。不是候选优化、搜索、正式接受或新生产collector。原作者终态与CPU错误保留；脚本SHA `16af16d674dc05872e5ff835a5b165c5473819ed9b4e743be12bb2ef64e74110`，不进行手工修补。

固定顺序为baseline-observed、candidate-observed各一次。沿用作者输入索引中的两个原ExecutionRequest及a866失败候选，参数为空。只替换entry/function/output/wait，并添加通用diagnostic_context（原入口/函数、独立参考路径、描述性arm）；请求中的原项目、adapter498c90f3、精度、完整工作、两次连续质量、两次warmup和五次clean重复不改。worker输出目录由executor创建，既有源/controls参考共享只读，不复制、不发布pointer。原未观察失败保持冻结；新观察即使意外通过也不回写旧失败或重复至通过。

完整B1/prompt256/32新token、BF16/SDPA/noTF32、全logits .02/.02、exact tokens、首/末同步边界保持。TTFT原anchor24.051052ms、上限25.2536046ms仍不变，但本次所有耗时仅诊断，不进行性能接受。既有人工外部源字段映射和开发事实辅助历史边界不因本次作者试验消失。

完成后逐项核查：原evaluator是否只调用一次；完整run/observe连续顺序、失败位置与缺失后续数据；原异常是否保存后传播；实际entry/core/源/候选/合同/请求哈希；所有到达的质量与clean样本；完整greedy/teacher原始张量对源及彼此的token、finite、逐位置logits关系；hook恢复、事件顺序、binding标签与实际筛选边界。缺失after-observe快照必须明确，不反推不存在变更。脚本CPU摘要不能替代完整原质量语义；独立摘要只覆盖第一次greedy，需据原始张量另核。

已知事件kind覆盖、CPU副本与持有对象/异常帧、额外allocator查询及模块残留均作为观察限制。负例实际改token，不冒充logits负例。基线通过只能证明该基线条件可运行；不能证明候选无观察效应、唯一根因或所有生命周期保持。执行/诊断持久化、原质量失败、源一致性及性能接受必须分开。若有接口缺陷，先冻结实际错误，再决定明确的最小修正/真实请求；本批不隐式修复、重启或重试。

控制器`scripts/experiments/v6_diagnostic_author_validation_20261029.py`；服务器目录`/root/autodl-tmp/dev/v6-work/diagnostic-author-validation-20261029`。CPU请求/23文件哈希预检通过，真实BindingSession接口匹配；尚不代表GPU有效。生产core ed24af7a、prompt/sampler/allocator/接受规则不变。当前约9.9GiB可用，预计小型索引与约80MB输出张量，无大型trace、参考副本或本地raw下载。本轮无需清理；真实df监控维持8GiB提醒，GPU结束前不冷压缩。新长任务配唯一普通完成/磁盘监听，无Goal、模型等待轮询或投递测试。
