# Qwen完整生成目标与TTFT约束的自主搜索

承接方案E及[同核心目标对照](v6-qwen-goal-contrast-20261028.md)。新任务检验框架能否围绕整个生成流程优化，并同时遵守另一阶段的性能约束。不是重测旧候选直到通过，不要求Agent必须选择两个组或使用特定组合。

目录`/root/autodl-tmp/dev/v6-work/qwen-end-to-end-20261028`，控制器`scripts/experiments/v6_qwen_end_to_end_20261028.py`。主目标为完整32-token的`end_to_end_ms`最小化，新增任务约束`ttft_ms <= 1.05 × 本任务新controls的原baseline TTFT`。这是新实验目标/约束，不追溯添加到旧结果，比例不是框架硬编码规则。原batch1/prompt256/full32tokens、BF16/SDPA/noTF32、full logits 0.02/0.02、exact tokens、两同步边界、状态持有和完整工作保持。

通过既有`--reuse-evaluator`只复用原baseline五个源文件及adapter24ebefd7；不复用优化bundle、旧分数锚点或人工诊断。真实variant_contract只适配invariants，框架更新goal/objective，其他字段继承。继承completion文字仍把首token边界称为“ttft_ms objective”，不得声称已自动改写；须结合新goal/objective/invariants核对TTFT实际作为约束、完整生成为主目标。保存初始快照、执行新controls，再用已有独立源参考对新controls做两次全量logits/exact tokens比较；核查结果不传给搜索Agent。

准备和源检查通过后，CLI `--resume --rounds 3`进行发现、采集计划、目标选择、候选生成/回装、参数筛选和反馈迭代。不给手选组、旧T/D答案或交叉测量结论。每个真实会话/API只统计，不设置新费用或请求限制；三轮不是API额度。阶段失败保留并停止依赖阶段，不循环到通过。

存在搜索接受版本时，在新合同内从新baseline做完整独立AB/BA，TTFT约束始终锚定本任务原controls。仅在独立完整质量通过且存在新bundle时采集单独B/C Nsight；侵入式profile不进入clean分数。搜索接受、源一致性、独立质量/性能、实际路径和执行完成分别报告。

这是同adapter目标变体，不是原入口重新生成evaluator的自主cohort。不承诺多个active bundle，只有实际重发现/替换/共存记录能证明。生产core、prompt、sampler、allocator和接受规则保持，成功反馈提示研究已结束。

[预检查](evidence/v6-qwen-goal-contrast-20261028/end-to-end-preflight.json)通过schema、语法与原baseline约束接口；正式任务重新测量anchor，不使用该CPU接口检查里的旧值。服务器内无损整理后实际余16.40GiB，raw留服务器，单GPU串行；一个普通完成/8GiB告警监听，无Goal、模型轮询、本地大转存或额外投递测试。所有历史接受/拒绝及样本保持。
