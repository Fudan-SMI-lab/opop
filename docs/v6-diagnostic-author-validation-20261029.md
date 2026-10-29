# 原生成诊断脚本的固定GPU验证

本阶段承接[真实诊断作者审计](v6-diagnostic-author-20261029.md)，检验Agent原样输出能否在真实框架完整评测中保存所缺证据。不是候选优化、搜索、正式接受或新生产collector。原作者终态与CPU错误保留；脚本SHA `16af16d674dc05872e5ff835a5b165c5473819ed9b4e743be12bb2ef64e74110`，不进行手工修补。

固定顺序为baseline-observed、candidate-observed各一次。沿用作者输入索引中的两个原ExecutionRequest及a866失败候选，参数为空。只替换entry/function/output/wait，并添加通用diagnostic_context（原入口/函数、独立参考路径、描述性arm）；请求中的原项目、adapter498c90f3、精度、完整工作、两次连续质量、两次warmup和五次clean重复不改。worker输出目录由executor创建，既有源/controls参考共享只读，不复制、不发布pointer。原未观察失败保持冻结；新观察即使意外通过也不回写旧失败或重复至通过。

完整B1/prompt256/32新token、BF16/SDPA/noTF32、全logits .02/.02、exact tokens、首/末同步边界保持。TTFT原anchor24.051052ms、上限25.2536046ms仍不变，但本次所有耗时仅诊断，不进行性能接受。既有人工外部源字段映射和开发事实辅助历史边界不因本次作者试验消失。

完成后逐项核查：原evaluator是否只调用一次；完整run/observe连续顺序、失败位置与缺失后续数据；原异常是否保存后传播；实际entry/core/源/候选/合同/请求哈希；所有到达的质量与clean样本；完整greedy/teacher原始张量对源及彼此的token、finite、逐位置logits关系；hook恢复、事件顺序、binding标签与实际筛选边界。缺失after-observe快照必须明确，不反推不存在变更。脚本CPU摘要不能替代完整原质量语义；独立摘要只覆盖第一次greedy，需据原始张量另核。

已知事件kind覆盖、CPU副本与持有对象/异常帧、额外allocator查询及模块残留均作为观察限制。负例实际改token，不冒充logits负例。基线通过只能证明该基线条件可运行；不能证明候选无观察效应、唯一根因或所有生命周期保持。执行/诊断持久化、原质量失败、源一致性及性能接受必须分开。若有接口缺陷，先冻结实际错误，再决定明确的最小修正/真实请求；本批不隐式修复、重启或重试。

控制器`scripts/experiments/v6_diagnostic_author_validation_20261029.py`；服务器目录`/root/autodl-tmp/dev/v6-work/diagnostic-author-validation-20261029`。CPU请求/23文件哈希预检通过，真实BindingSession接口匹配；尚不代表GPU有效。生产core ed24af7a、prompt/sampler/allocator/接受规则不变。当前约9.9GiB可用，预计小型索引与约80MB输出张量，无大型trace、参考副本或本地raw下载。本轮无需清理；真实df监控维持8GiB提醒，GPU结束前不冷压缩。新长任务配唯一普通完成/磁盘监听，无Goal、模型等待轮询或投递测试。
