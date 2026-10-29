# 两份真实生成候选的固定执行验证

本批于 **04:45:02 UTC（香港12:45）** 启动，冻结 `3c287e9ca67d283f61d305d98147a470eff7822c`，controller965586，04:47:19四端点已完成。见[启动](evidence/v6-diagnostic-index-fixed-20261029/launch.json)、[冻结预检](evidence/v6-diagnostic-index-fixed-20261029/preflight.json)与[执行身份／初步终态](evidence/v6-diagnostic-index-fixed-20261029/startup-verification.json)。实际六rank启动记录、四request的materialized代码/adapter/entry/bundle参数、131冻结文件和两source/分支指针核验通过；production core不变。

初步回执四端点均worker passed/valid_run=true；Qwen B/C TTFT分别24.458036/21.579238ms，均低于原25.2536046；FSDP B/C peak均2,239,245,312B，低于原2,351,207,577.6B。**这只是worker/质量覆盖/约束回执，尚待全样本、数值检查、实际路径和48份预期通信trace的完整终态审计，不作性能接受。** 没有新Agent或正式AB/BA。

普通监听14176于04:46:30启动，04:47:31成功排队一次完成通知（[回执](evidence/v6-diagnostic-index-fixed-20261029/wakeup-delivery.json)）并退出；旧59216已处理，没有重复监听或额外投递测试。实际GPU已空闲，df约8.48GiB，未清理或转存raw。新回合将完整审计本批后继续。

初次启动审计工具误从rank-started读取不存在的framework_source而KeyError；真实started只含revision/binding/request引用，core在完成的rank结果中。已单独保留[审计工具错误](evidence/v6-diagnostic-index-fixed-20261029/startup-audit-fixture-error.json)，修正读取位置后核验，没有改动/重启任何实验或候选。

本批验证[索引重放](v6-diagnostic-index-candidate-20261029.md)生成的精确原代码：Qwen `2c08b3b1137d754917dce96c358cc20061aad16b362491327cd2f3b83790cafd`，FSDP `206a02b88dcfff432a8a7f43cbb0adb766a7f04842d88a8e914c9c52c8f4ab8b`。不手工修代码，不新增Agent、不再生成候选，不将未读取索引的结果称为诊断反馈闭环。

04:43:22 UTC预检通过：131冻结文件、两任务原calls/groups、portable request、adapter和共享controls身份一致；FSDP source current仍为历史接受3007940f，但新分支明确从空bundle fa5f081d开始，不导入K3优化bundle。两个原request都保留2continuous/2warmup/5clean，Qwen额外diagnostic_metrics为空，FSDP为原四项。实际df 9,790,251,008 bytes，core不变。

按 **Qwen B→C、FSDP B→C(K2)** 串行各执行一次。Qwen参数空间为空；FSDP固定生成代码DEFAULT_KEEP=2且proposal声明primary的K2，不运行其余K1/K3、不做最优参数挑选。使用既有fork_run/bundle_from_outputs/stage/execute，在隔离分支安装；原始空bundle发现revision分别为b61996e1与fa5f081d，原source当前指针即使后来接受也保持不变。原discovery逐字对照生成前calls/groups，不重新GPU发现；controls/reference只读共享，不复制多GB张量。

保持各自原request、contract、adapter、evaluator、精度/工作量/完成边界、连续两步quality、2次warmup和全部5次clean样本。Qwen完整B1/prompt256/32token，BF16/SDPA/noTF32、完整logits .02/.02和exacttokens，原TTFT约束上限25.2536046；不沿用另一adapter目标变体的锚。FSDP seeded GPT2Medium24layer/hidden1024、双rank B1/S128、FP32/eager/noTF32/dropout0，全FW/BW/global-mean梯度/SGD/tied/shard映射及完整两步数值1e-6/1e-4保持，原peak<=1.05baseline/maxrank后median保持。不同shard不是副本；developer-assisted512B source-owner repair边界不变。

新baseline失败则不运行该任务候选，保留实际错误；新候选失败不重试到通过。两任务独立，第一任务失败不会无故取消第二任务。evaluator原返回valid=false、worker异常、quality/coverage、constraint各自记录；controller completed不等于优化成功。没有新observer、Nsight、接受决策、AB/BA、指针发布或正式性能比较。所有固定计时只是诊断，即使两个候选通过质量和约束也需后续声明正式完整版本验证。

FSDP保留原四个diagnostic_metrics，各自3次独立profile/rank；如质量失败，这些阶段可能不执行。逐项核查真实trace及每rank样本，不能跨metric减中位数或把NCCL sum当wire/关键路径。Qwen本批不新增native profile；route labels、自检和代码包含graph.replay都不能替代真实原生执行证据。重点检查setup失败退回parent是否被误当优化路径，及新代码的连续调用是否真正保持完整输出。

控制器 `scripts/experiments/v6_diagnostic_index_fixed_20261029.py`，目录 `/root/autodl-tmp/dev/v6-work/diagnostic-index-fixed-20261029`。阶段前/后冻结源文件、source指针、原controls/诊断raw及新生成代码；production core `ed24af7a…`不变。新分支current始终是其原baseline，不能把stage等同publish。source-fidelity仍依赖已审查的原adapter/参考，不假称本批重复了独立源全量审计。

启动前检查真实150GiB容器df和GPU worker；当前约9.12GiB，预计仅常规worker结果及FSDP原诊断trace，无大型Nsight/新reference。raw/private配置留服务器，不作本地大体积转存；若需要清理，按既有授权在GPU结束后无损压缩冷数据。配置唯一普通完成/8GiB监听，无Goal、模型等待轮询或投递测试。所有历史接受/拒绝/全部样本保持，API使用规则不变。
