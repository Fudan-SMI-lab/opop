# Qwen完整输出关系检查的固定诊断

问题来自[原入口搜索三轮失败](v6-qwen-original-end-to-end-20261029.md)：前两轮在teacher-forced replay关系检查处抛错，未保存greedy与teacher各自对独立源的差异。现有数据不能判断哪一侧先偏离，也不支持唯一alias/shared-pool原因；candidate_diagnostics执行成功没有进行质量认证。

新目录`/root/autodl-tmp/dev/v6-work/qwen-relation-diagnostic-20261029`，控制器`scripts/experiments/v6_qwen_relation_diagnostic_20261029.py`。固定顺序为 **B-observed、C1-unobserved、C1-observed、C2-unobserved、C2-observed**，每项仅一次。C1/C2分别是原`d5c20fcae5`和`e74f5ba6cf`的确切代码/参数；B为原基线。U使用原执行entry与evaluate；O使用外部诊断entry包装生成adapter的run/_infer/observe，观察点位于完整greedy或teacher输出之后，并在observe退出后再次检查greedy输出。只记录该具体失败所缺的完整输出，不加大trace、before_run观察器、GC控制或CPU计数读取。

所有嵌套request、候选、adapter、工作量、数学政策、controls、质量阈值、warmup/repeats/连续步骤不改。观测层保存完整CPU tokens/logits、哈希和小摘要，比较各输出与独立源、teacher与greedy、greedy在observe前后是否改变；额外按run/observe阶段计数已有candidate route回调，不改变原tracking状态。不用单个route标签证明实际原生kernel，也不从没有标签推断没有回退。

CPU复制会增加host工作、同步及CPU对象生命周期，可能改变复现条件；这就是保留U对照的原因。O的内部计时或framework.elapsed不用于性能接受，不减去观察开销。若某次重放通过，也保留原失败并明确观察/重放差异，不重试到通过。全部原始值留服务器，无本地raw下载。一次失败可能使该端点没有clean样本，必须如实记录。

B-observed为实际CUDA观测器控制；随后C1/C2两种路径保留原异常、traceback、所有结果。observed finally写旁文件，controller“completed”只表示诊断尝试结束，worker失败及完整质量另报。该批没有Agent生成、没有自主搜索、没有formal acceptance，也不自动采纳任何生产改动。下一项真实Agent实验必须等此次证据明确所缺信息，再做最小调整，不能把尚未隔离的原因写进提示。

源码与参数身份在执行前后核对。先前全量源参考和manual映射保持，当前只有约13.17GiB可用；预计新增完整CPU输出百余MB、不产生大型profile。GPU工作串行，单个普通完成/8GiB监听；无Goal、模型等待轮询、本地大转存或旧controller重启。若低空间先查df/活跃worker，只有GPU结束后才做授权无损冷压缩，保存全部原始内容。

[接口预检查](evidence/v6-qwen-original-followup-20261029/relation-preflight.json)通过导入/语法/CLI和三个冻结screen执行请求schema。实际CUDA观测行为由首个B端点验证，不以CPU接口检查冒充实验结果。
