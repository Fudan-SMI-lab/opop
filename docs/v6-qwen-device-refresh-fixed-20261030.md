# Qwen 重启后设备迁移与原样候选固定验证

这是[断电恢复审计](v6-qwen-accepted-recovery-20261030.md)发现设备UUID变化后的独立实验，不重启旧两批。旧第二轮候选尚未启动GPU，没有数值/性能结论。生产core仍ed24af7a，prompt/sampler/allocator/接受规则不变；不新增Agent、参数搜索、正式性能接受或Nsight。

现有prepare_variant仅将旧请求devices改为实际GPU0 UUID `GPU-20e8003c-f531-d50c-084c-a9a97fe94004`。源、目标、合同、adapter498c90f3、质量、工作量、资产和采样不改。变更设备后生成新的原始baseline controls与参考，保留2continuous/2warmup/5clean；TTFT上限仍为该新原baseline的1.05倍，不复用旧25.2536046或其他目标anchor。此前prepared009b及接受版本、参考和记录只读保留。

新controls通过后，沿用已审查的外部generated_tokens/step_logits映射，CPU比较新真实CUDA controls的两次完整32 tokens及1×32×151936 BF16 logits与独立源参考af929f42，检查finite/exact token及token+1/logit超差负例。该fixture仍是手工映射边界，不认证任意额外owner；不重新生成独立源，不将结果提供Agent。controls或源检查失败即停止依赖阶段，保留错误。

通过后使用既有stage_version导入第二轮完整stage f211410e，验证合同/原source anchor相同及文件顺序/哈希相同。唯一活动bundle为trial-recovered-4f3d7f11aa-param-2eb959a5，候选f5b76ebd、空参数；原样保留其代码、proposal及实际自检/回退行为，不修代码，不导入旧f065活动bundle。然后固定新B、新C各一次，B质量/覆盖失败则不运行C。每端点保留原全部工作、2continuous/2warmup/5clean（若到达），检查新原anchor约束。时序指标仅诊断，无ABBA、配对噪声或正式接受；不能视为旧47.325%的增量或恢复搜索接受。

代码有真实风险待执行观察：模型级rotary修复、六种Triton kernel、内部实现选择/31decode Graph及fresh mirror；setup异常回退不保证任意数值/owner安全，直接自检非实际框架dispatch。prefill父调用在单新bundle下指向原模型forward，不能宣称仍用已退役f065 Graph。CPU微型探针失败/修正保留，不能用最终CPU打印代替原模型质量。新运行无额外observer，原错误和样本不覆盖，不重试到通过。

预检实际两张4090空闲，150GiB磁盘余约10.89GiB，核验286冻结身份及旧UUID缺失/新UUID存在。估计仅新controls及普通B/C输出，无大profile或模型复制，不需清理。原始数据和私有设置留服务器，GitHub仅代码/哈希/小摘要；长任务只设一个完成/8GiB监听。低空间不是完成，不停活跃实验或盲删，授权冷压缩只在GPU结束后进行。
