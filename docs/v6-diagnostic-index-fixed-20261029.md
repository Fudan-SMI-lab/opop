# 两份真实生成候选的固定执行验证

终态审计完成：四端点/六rank均正常退出，原连续两步质量检查全部零误差；30个clean rank样本、12次完整检查、48份独立通信trace全部保留并复核。[小型审计回执](evidence/v6-diagnostic-index-fixed-20261029/public-audit.json)对应服务器完整审计SHA `2ff41895f1c816ba578cee46fafcdd9a5bdc99162dfa5fa91993ed0b759cb646`。131冻结文件、源码/adapter/entry/core、原request/controls/contract和source/分支current保持。以下计时是固定诊断，**没有正式性能接受、独立AB/BA或新源全量复测**。

| 固定端点 | 主指标中位数（ms） | MAD（ms） | 原约束 |
|---|---:|---:|---|
| Qwen B | 706.337804 | 7.782570 | TTFT 24.458036 ≤ 25.2536046 |
| Qwen C | 389.965840 | 0.133004 | TTFT 21.579238 ≤ 25.2536046 |
| FSDP B | 178.612929 | 2.651011 | peak 2,239,245,312 ≤ 2,351,207,577.6 B |
| FSDP C K2 | 172.504023 | 0.028346 | 同一peak约束通过 |

Qwen全部样本保留32生成token、31decode token和完整生成/首token边界；两次quality各比较4,861,988个元素。C遥测为2次prefill_graph_replay、62次graph_replay，无记录fallback，setup约7,850.74ms在warm-model计时之外。实际代码调用Graph replay且完整quality通过，但本批没有该精确代码的native trace，不能移用旧d5bd的32次Graph证据。C生命周期peak从8,188,023,808增至8,400,915,456B（+212,891,648B），runtime peak增加40,781,824B，不是显存收益；全局patch并发和任意owner组合仍未证明。

FSDP各rank每次完整检查分别361,257,090/361,255,042元素，均零误差。C两次quality确实调用parent forward，两个reference调用属于组合机制，不是自动认定fallback。48份raw trace逐项截取各自workload窗口并重算：每rank每次B为49 AllGather+25 ReduceScatter，C为47+25，均实际RING_LL kernel。这支持K2少两次AG，不支持整个24-callsite消除、固定每collective收益、wire bandwidth或暴露关键路径。

四指标各有自己的3次profile/rank，按每次先max rank后median：B/C kernel_sum为172.724565/168.156001ms，union为175.688406/233.630156ms，compute_overlap为45.415140/42.426688ms，without_compute为124.863435/124.393839ms。union原样本含287.770269与277.287201ms等长值，全部保留；这些是不同profile周期，不能相减归因，也不能用rank各自median替代对齐聚合。FSDP生命周期peak两臂均2,264,977,408B。

前序Agent未打开新增索引的结论不变，固定GPU通过不追认诊断消费或稳定因果优势。两份代码留作固定验证产物，不发布current或覆盖历史接受。下一步按[有限证据阅读提示实验](v6-diagnostic-index-read-20261029.md)只处理这个实际输入消费缺口；不增加reviewer、不自动采用生产提示。早期只读盘点曾误以为worker目录含result.json，实际聚合回执位于根目录；修正读取位置后完成审计，没有重跑模型或改结果。

另用真实prepared evaluation/原objective重新调用约束逻辑，逐项与controller原结果一致，见[原anchor复算](evidence/v6-diagnostic-index-fixed-20261029/constraint-recomputation.json)，不是只把已知阈值写入断言。

04:57:27–04:58:27 UTC，确认GPU退出后，仅压缩明确完成的旧cohort DDP/ResNet及两个DDP step批次worker RUN JSON。229份原payload 3,089,776,337→261,729,157B，压缩/解压SHA逐项核验，1,601其他文件不变；10当前版本源码/bundle和6份FSDP完整源参考后验核验。实际df净释放2,825,908,224B（2.63GiB），剩11,934,175,232B（11.11GiB）。无本地raw转存，当前索引/诊断/源码/controls/模型/private状态不在压缩范围。恢复入口 `storage-audit-20261029/diagnostic-index-read/run/RESTORE.md` 与独立journal，见[维护回执](evidence/v6-diagnostic-index-fixed-20261029/storage-postcheck.json)。规划时误写两个独立分支目录名，断言在压缩控制器启动前停止；枚举真实目录修正清单后执行，原planning错误留服务器。

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
