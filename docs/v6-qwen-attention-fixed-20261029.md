# Attention事实候选的原样固定执行验证

终态审计（2026-10-28 22:07:45 UTC）：三端点均执行通过，15个原始clean样本、6次完整连续质量检查全部保留，每次4,861,988比较元素且max error 0。B与C-observed共12份完整greedy/teacher/observe后张量逐元素重读，均与独立source完全相同、tokens精确、有限；所有request、adapter、contract、bundle和source/controls哈希、两处current指针已复核未变。此前四种失败实现和原搜索失败均不覆盖。

| 固定端点 | 全程ms（5样本中位数） | TTFT ms | 原25.2536046上限 | 质量 |
|---|---:|---:|---|---|
| B observed | 689.836474 | 24.530360 | 通过 | 通过 |
| C unobserved | 395.008819 | 26.510787 | 失败 | 通过 |
| C observed | 394.668520 | 25.917823 | 失败 | 通过 |

这些是固定诊断，B含观察器、没有配对clean baseline，不能计算正式收益或接受。两个候选均违反原约束，不对同一候选重复计时直到通过。U/O质量coverage各62条graph_replay，O跨2quality+2warmup+5clean共279条run阶段callback，未见fallback标签。结合原代码在标签前调用graph.replay和不同setup失败标签，支持本次走过setup/replay分支；仍需独立native trace，不能单凭标签声称31Graph/具体kernel。Teacher stride不匹配框架signature，缺少teacher callback也不是完整native证明。

U安装耗时3759.306374ms，O3819.432944ms，均在声明的warm-model计时外；两者生命周期peak 8,296,180,736B，相对B 8,188,023,808B增加108,156,928B。运行peak增加60,299,776B，retained master/logits/Graph pool由实际计数体现；不能等同物理节省，也不能据两次相同输出证明任意并发/全局所有权稳定。临时全局mask patch恢复由代码finally保证，但本批未记录独立恢复事件。

完整[审计](evidence/v6-qwen-attention-fixed-20261029/terminal-audit.json)SHA`7acebb8f8c4231735ea3b18b012a2237e768d19ef00de4d990d89e9db79c940c`。接下来做一次单独native路径核查，并按[约束反馈续行](v6-qwen-constraint-continuation-20261029.md)将真实质量通过/TTFT失败及候选历史代码交回普通selection/candidate流程；不手工选择prefill/group、不提供实现答案，不把本次角色诊断说成原始入口全程无人辅助。生产core/prompt/规则不改，GPU已退出，实际df余约11.50GiB。

已启动（2026-10-28 22:06:37 UTC，香港10月29日06:06）：冻结`4d7dd205`，生产core不变；控制器PID898327，隔离候选stage完成。22:06:59启动核查时B-observed的真实GPU worker PID898399运行，尚无完成端点，不能报告质量结论。唯一普通完成/8GiB监听PID48488已核验waiting，旧attention-evidence监听标记已处理；没有Goal或模型等待轮询。实际df余12,442,140,672字节（11.59GiB），没有清理或raw本地转存。证据：[启动核查](evidence/v6-qwen-attention-fixed-20261029/startup-verification.json)、[原样版本](evidence/v6-qwen-attention-fixed-20261029/staged-identities.json)、[监听回执](evidence/v6-qwen-attention-fixed-20261029/wakeup-setup.json)。

前置[真实角色审计](v6-qwen-attention-evidence-20261029.md)完成；候选`f12e2f50…`只经过CPU探针和生成校验，完整CUDA结果未知。下一批不再调用Agent，不修改生成代码，不开展参数搜索或正式接受；使用既有bundle_from_outputs/stage接口，从原发现版本b61996e1的空bundles建立隔离分支，保留原proposal声明和空参数空间。

固定顺序：B-observed、candidate-unobserved、candidate-observed，各一次。U使用原evaluate；O使用已核验的外部观察器`a06b6c0e…`，保存greedy/teacher/observe后的完整CPU张量和phase routes。O改变CPU工作、同步和生命周期，计时只能诊断，不能扣除观察成本或当接受；缺少route标签不证明无native/fallback。每个端点保留原warmup、2个连续质量步骤及若到达则全部5个clean样本。失败可能早于clean，不能把控制器完成当质量通过。

完整B1/prompt256/32tokens、BF16/SDPA/noTF32、logits .02/.02、exact tokens、两个同步边界、原adapter498c90f3和source/controls均保持。原TTFT固定上限仍25.2536046ms；质量、覆盖、约束分别报告，timing不构成正式接受。共享只读参考，不重新生成controls/大参考副本，不发布source或隔离分支current指针。候选的临时masking全局patch是生成实现的一部分，仅在独立worker执行，原样验证其状态刷新、完整质量、Graph/fallback路径及生命周期；不将候选断言当事实。

本批没有Nsight或正式AB/BA。只有原样完整质量通过并确认实际候选路径，才有依据安排下一步独立性能/完整native核查；不能从setup内部自检、CPU模拟或观察标签直接宣称优化成功。保留旧四候选完整失败、所有历史接受/拒绝和样本，不重复直到通过，不降低阈值。

控制器`scripts/experiments/v6_qwen_attention_fixed_20261029.py`；服务器目录`/root/autodl-tmp/dev/v6-work/qwen-attention-fixed-20261029`。生产core仍ed24af7a，预计小输出/完整观察张量，无大trace。启动前核查真实df和GPU空闲，约11.72GiB无需清理；原始数据留服务器，GitHub只代码/哈希/小摘要，无本地大转存。启动后只设一个普通完成/8GiB监听，不启用Goal或模型等待轮询。
