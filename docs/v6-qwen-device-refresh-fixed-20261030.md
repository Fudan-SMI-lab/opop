# Qwen 重启后设备迁移与原样候选固定验证

已于2026-10-29 17:32:47–17:34:22 UTC执行完毕，冻结b8712f1b/controller8350。实际GPU UUID、请求、worker entry/core及286冻结身份已核验；[公开审计](evidence/v6-qwen-device-refresh-fixed-20261030/public-terminal-audit.json)对应完整审计SHA `3b0a9bcbf5aec79c59976fff51518a60e60e6e937bd482eccc5b00178ad6593a`。[启动核验](evidence/v6-qwen-device-refresh-fixed-20261030/startup-verification.json)执行时已终态；[监听预检](evidence/v6-qwen-device-refresh-fixed-20261030/wakeup-setup.json)发现完成，未创建listener或排队通知。旧监听23004保持已处理。

新controls reset两次/identity一次通过，负例和missing coverage拒绝；独立源CPU两次完整tokens/logits零误差、finite/exact及两负例通过。新生成reference SHA恰与旧94887086相同，但其真实GPU controls本次执行且文件位于新目录，并非拷贝旧参考；prepared revision仍009b是内容身份，不意味着复用旧计时。新原baseline TTFT **23.285755ms**，上限 **24.45004275ms**。

固定B/C各两次完整零误差检查、各5样本，加controls共15干净样本全部保留。B full701.762096ms、TTFT24.5339，后者超过新上限；C full674.079858ms、TTFT23.550697合格。B门控按协议只看质量/覆盖，所以仍执行C；B的765.755993ms长样本不删除、不改anchor。没有配对噪声或正式接受。C质量阶段实际记录62次 `parent_fallback_setup_failure`、2次 `prefill_parent_passthrough`，对应父调用62/2；framework fallbacks为空不能抹去候选内部回退。**这是父路径质量通过，未证明新快路径成功。** setup3899.385769ms在warm score之外；C生命周期峰值增加119,403,008B、runtime峰值增加99,271,168B，非显存优化。

为定位被捕获的setup异常，仅补一次既有run-only candidate_diagnostics，17:38:24 UTC结束，worker passed/valid但没有observe/compare/干净计时；[审计](evidence/v6-qwen-device-setup-diagnostic-20261030/terminal-audit.json)和289冻结身份保留。实际在 `_check_logits`→setup→build三层传播同一 `decode logits mismatch (eager_vs_parent_step_1)`，不是三次模型失败；252个normal False返回也不是错误。候选内部检查为0.008/0.008，比原任务0.02/0.02严格；没有保存差值或完整输出，不能推出外层质量超差、唯一kernel/算术原因。token检查位于该断言之前；单独诊断受instrumentation扰动，不证明此前固定C所有内部状态相同。没有改候选、放松质量或重测到通过。

审计辅助程序最初把evaluate的math_policy_unchanged字段用于controls，发生KeyError；保留audit-attempt-1.log后按实际controls schema修正，只重算审计，未重跑GPU/fixture或修改候选。真实evaluate字段均为true。磁盘仍约10.81GiB，无清理。下一步[新设备一次普通反馈试验](v6-qwen-device-feedback-20261030.md)，以这份原样候选及实际setup证据为历史输入，新原baseline fresh discovery，原质量/新anchor保持；不把内部严格自检失败伪造成外部质量失败，也不提供手工修复实现。以下保留启动前协议。

这是[断电恢复审计](v6-qwen-accepted-recovery-20261030.md)发现设备UUID变化后的独立实验，不重启旧两批。旧第二轮候选尚未启动GPU，没有数值/性能结论。生产core仍ed24af7a，prompt/sampler/allocator/接受规则不变；不新增Agent、参数搜索、正式性能接受或Nsight。

现有prepare_variant仅将旧请求devices改为实际GPU0 UUID `GPU-20e8003c-f531-d50c-084c-a9a97fe94004`。源、目标、合同、adapter498c90f3、质量、工作量、资产和采样不改。变更设备后生成新的原始baseline controls与参考，保留2continuous/2warmup/5clean；TTFT上限仍为该新原baseline的1.05倍，不复用旧25.2536046或其他目标anchor。此前prepared009b及接受版本、参考和记录只读保留。

新controls通过后，沿用已审查的外部generated_tokens/step_logits映射，CPU比较新真实CUDA controls的两次完整32 tokens及1×32×151936 BF16 logits与独立源参考af929f42，检查finite/exact token及token+1/logit超差负例。该fixture仍是手工映射边界，不认证任意额外owner；不重新生成独立源，不将结果提供Agent。controls或源检查失败即停止依赖阶段，保留错误。

通过后使用既有stage_version导入第二轮完整stage f211410e，验证合同/原source anchor相同及文件顺序/哈希相同。唯一活动bundle为trial-recovered-4f3d7f11aa-param-2eb959a5，候选f5b76ebd、空参数；原样保留其代码、proposal及实际自检/回退行为，不修代码，不导入旧f065活动bundle。然后固定新B、新C各一次，B质量/覆盖失败则不运行C。每端点保留原全部工作、2continuous/2warmup/5clean（若到达），检查新原anchor约束。时序指标仅诊断，无ABBA、配对噪声或正式接受；不能视为旧47.325%的增量或恢复搜索接受。

代码有真实风险待执行观察：模型级rotary修复、六种Triton kernel、内部实现选择/31decode Graph及fresh mirror；setup异常回退不保证任意数值/owner安全，直接自检非实际框架dispatch。prefill父调用在单新bundle下指向原模型forward，不能宣称仍用已退役f065 Graph。CPU微型探针失败/修正保留，不能用最终CPU打印代替原模型质量。新运行无额外observer，原错误和样本不覆盖，不重试到通过。

预检实际两张4090空闲，150GiB磁盘余约10.89GiB，核验286冻结身份及旧UUID缺失/新UUID存在。估计仅新controls及普通B/C输出，无大profile或模型复制，不需清理。原始数据和私有设置留服务器，GitHub仅代码/哈希/小摘要；长任务只设一个完成/8GiB监听。低空间不是完成，不停活跃实验或盲删，授权冷压缩只在GPU结束后进行。
