# Qwen原入口完整生成目标验证

承接[同adapter完整生成目标](v6-qwen-end-to-end-20261028.md)的46.9421%独立收益，下一项问题是：不提交现成adapter/request，框架能否从原项目、启动命令和目标自行得到完整且忠实的评测、分组和优化结果。目录`/root/autodl-tmp/dev/v6-work/qwen-end-to-end-original-20261029`，控制器`scripts/experiments/v6_qwen_original_end_to_end_20261029.py`。

输入只有原`examples/v6_validation`项目、原Qwen推理命令/模型资产路径、目标和设备0。目标仍为batch1/prompt256/full32 greedy token的`end_to_end_ms`最小化，`ttft_ms <= 1.05 × 本任务新原baseline`。BF16/SDPA/noTF32、完整logits0.02/0.02、exact tokens、两同步边界及源生命期要求保持。不给旧adapter、已解析request、手选组、旧候选、优化答案或前一批测量结果。自然语言约束是任务输入，不是向Agent指定Graph或其他实现。

使用既有CLI `--project --command --goal --prepare-only`运行全新真实intake/evaluator/controls。随后外部独立源fixture核对新controls的完整tokens/logits；它使用原始参考而非旧优化版本，结果不进入候选Agent输入。新增observation需要单独审查；命名/容器差异保留为schema/执行问题，并停止依赖搜索，不能自动映射、当数值失败或自动接受。源检查只覆盖可比较值，不替代对返回状态、prepare持有对象、计时与内存边界的代码审查。

准备和源检查通过才使用同request `--resume --rounds 3`。这是新定义的原入口cohort，不是重复旧候选直至通过。保存每次初始snapshot、调用/回复、实际参数与路径、质量、原锚约束和反馈；API只统计不加使用额度。未知错误冻结，既有schema修复由框架正常执行，不加入失败类型硬编码。

有自主接受版本才做新合同内独立完整AB/BA；无版本记not_run。独立完整质量通过后才采集单独baseline/candidate Nsight，核对实际路径，侵入式profile不作为分数。搜索接受、源一致性、独立性能、native证据、控制器完成分别判断。不得宣称多个active bundle，除非实际证据存在。

生产核心继续`ed24af7acc8caa15578e0b859c85ef8a901be51354448ed39094606ed9e10a2f`；不调整prompt、sampler、allocator或接受规则。旧同adapter合同completion的“ttft_ms objective”文字局限保留，本次由新evaluator生成合同，不人为修复旧记录。

启动前真实df余约16.29GiB，上一批GPU与服务器内压缩均结束并核查。单GPU串行，不复制旧controls；新evaluator必要的新参考另行记录。仅一个普通完成/8GiB磁盘监听；低空间不是终态，不停止活跃任务或删除证据。raw留服务器，不恢复旧转存/监听。没有Goal、模型轮询或额外投递测试。历史Qwen、FSDP、DDP、ResNet、TP结果及全部样本保留。
