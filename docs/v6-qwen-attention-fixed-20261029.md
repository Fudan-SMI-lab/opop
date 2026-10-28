# Attention事实候选的原样固定执行验证

前置[真实角色审计](v6-qwen-attention-evidence-20261029.md)完成；候选`f12e2f50…`只经过CPU探针和生成校验，完整CUDA结果未知。下一批不再调用Agent，不修改生成代码，不开展参数搜索或正式接受；使用既有bundle_from_outputs/stage接口，从原发现版本b61996e1的空bundles建立隔离分支，保留原proposal声明和空参数空间。

固定顺序：B-observed、candidate-unobserved、candidate-observed，各一次。U使用原evaluate；O使用已核验的外部观察器`a06b6c0e…`，保存greedy/teacher/observe后的完整CPU张量和phase routes。O改变CPU工作、同步和生命周期，计时只能诊断，不能扣除观察成本或当接受；缺少route标签不证明无native/fallback。每个端点保留原warmup、2个连续质量步骤及若到达则全部5个clean样本。失败可能早于clean，不能把控制器完成当质量通过。

完整B1/prompt256/32tokens、BF16/SDPA/noTF32、logits .02/.02、exact tokens、两个同步边界、原adapter498c90f3和source/controls均保持。原TTFT固定上限仍25.2536046ms；质量、覆盖、约束分别报告，timing不构成正式接受。共享只读参考，不重新生成controls/大参考副本，不发布source或隔离分支current指针。候选的临时masking全局patch是生成实现的一部分，仅在独立worker执行，原样验证其状态刷新、完整质量、Graph/fallback路径及生命周期；不将候选断言当事实。

本批没有Nsight或正式AB/BA。只有原样完整质量通过并确认实际候选路径，才有依据安排下一步独立性能/完整native核查；不能从setup内部自检、CPU模拟或观察标签直接宣称优化成功。保留旧四候选完整失败、所有历史接受/拒绝和样本，不重复直到通过，不降低阈值。

控制器`scripts/experiments/v6_qwen_attention_fixed_20261029.py`；服务器目录`/root/autodl-tmp/dev/v6-work/qwen-attention-fixed-20261029`。生产core仍ed24af7a，预计小输出/完整观察张量，无大trace。启动前核查真实df和GPU空闲，约11.72GiB无需清理；原始数据留服务器，GitHub只代码/哈希/小摘要，无本地大转存。启动后只设一个普通完成/8GiB监听，不启用Goal或模型等待轮询。
