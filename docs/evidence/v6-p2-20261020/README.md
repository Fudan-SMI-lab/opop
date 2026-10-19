# P2：真实项目理解与评测生成

新增 `optimize-project --request ... --output ... --prepare-only`，从项目源码、固定 workload、目标、质量与资产定位生成 adapter 和项目理解记录。请求不包含人工算子组或目标组选项。生成使用真实配置的 GLM-5.3，框架独立运行评测控制，候选不得改动被冻结的 evaluator 文件。

box4-2 的 ResNet-18 单图请求通过，见 [prepared.json](prepared.json) 和 [生成的 adapter](generated_adapter.py)。实际检查覆盖连续两次调用、reset 重放、identity replacement、错误输出负控制、未执行替换的覆盖负控制。全部 1,000 个 logits 和 top-1 整数均检查；整数不使用浮点相对容差。

另以先前独立运行原项目保存的 `online-reference.pt` 复核，全部 logits 一致，最大绝对误差为 0，见[独立对照](independent-reference-check.json)。原始目录为 `/root/autodl-tmp/dev/v6-work/p2-resnet-a`。此处约 1.63 ms 的短测只证明评测可运行，不作为优化收益或与不同日期运行的性能对照。

相关检查 [24 passed](tests.log)。之后为冻结 evaluator 编辑增加的一项断言随事务文件的 4 项检查再次通过；未重跑完整旧套件。

当前完成的是 ResNet 的自动理解和评测路径。项目仍需通过 P3/P4 的发现、选择、优化、回装及反馈，以及 Qwen 和 DDP 的泛化与状态检查。`--prepare-only` 明确不宣称已完成优化。
