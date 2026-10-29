# 第二种任务的诊断作者：FSDP完整质量拒绝

Qwen原样Agent探针已在真实GPU保存缺失连续输出并保留原异常，但它专门围绕推理adapter的_infer/observe设计，尚无跨任务证明。此处用已有FSDP训练失败现场，检验同一作者接口能否根据不同源码选择必要观测点。不是重新优化失败候选，也不自动采纳Qwen诊断代码。

固定输入来自原`fsdp-search-20261028`第二轮第一个参数端点`trial-86e14487d0-param-b3a4c7d3-screen`，参数keep_unsharded_blocks=6，不从三参数中按新结果挑选。两个worker当时正常完成，但result.valid=false：连续第一步全量比较通过，第二步在output.logits首先拒绝，未进入clean计时。这个断言只指出比较遍历首先失败的位置，不直接证明执行中差异最先发生在logits或第二步；原文件/数值/其余参数结果保留。

六份显式输入仍为task、原worker记录、adapter、候选、证据索引及run(context)接口；框架另加worker环境。使用相同通用作者说明，追加一段通用要求：保留失败返回值、多rank/真实分片语义、避免额外collective和rank不对称操作、按信息缺口选择小型证据而非复制整套状态。本实验改变了任务与这段运行语义提示，不是单变量A/B或稳定泛化证明。没有提供Qwen探针/结果、开发者FSDP诊断、后续reshard接受版本或唯一原因答案。外部文件系统仍可访问，必须在终态审查实际工具路径和输出。

原任务为seeded GPT2Medium24层/hidden1024、两rank B1/S128、FP32/eager/noTF32/dropout0，完整forward/backward/global-mean-gradient/SGD与两次连续更新，tied embedding/head、真实named DTensor布局、完整logits/gradients/更新后参数，容差1e-6/1e-4保持。不同rank分片不是副本相等。沿用经过开发者生命周期反馈修复的adapter8b014f6f及原controls共享引用，既有辅助边界不抹除；不给Agent后来的数值归因和优化答案。

当前只执行一次真实diagnostic_author生成与语法/schema检查，无GPU观察、搜索、正式接受或新原入口cohort。已有controls可能很大，只索引/读取，不复制。没有为此次诊断预跑新的baseline；将来匹配baseline可由同一原请求显式清空bundles派生，必须记录为新派生控制，不能宣称已有该控制的历史测量。可选independent_reference为空，完整独立源一致性不是本轮作者输入自动提供的答案。

完成后审核生成代码是否适配原返回/异常、两rank完整执行顺序、真实质量与测量生命周期；额外引用、CPU拷贝、统计、通信/同步及原始数据体积是否可行。模型或参考大文件不得新增复制；输出选择与大小须实际审查后再安排独立固定GPU观察。CPU mock不证明CUDA/通信可行；正常worker与索引不等于质量通过。必要修正应保留原失败、另做明确真实请求，不手工隐藏修复或反复直到通过。若不适合执行，先记录具体缺陷。

控制器`scripts/experiments/v6_fsdp_diagnostic_author_20261029.py`，目录`/root/autodl-tmp/dev/v6-work/fsdp-diagnostic-author-20261029`；生产core ed24af7a/prompt/sampler/allocator/验收不变。真实API仅统计不加配额，唯一普通完成/磁盘监听；不使用Goal或模型轮询。当前df约9.83GiB，无GPU、大型trace/参考副本或raw本地下载。本轮无需清理，继续保护共享references、模型、接受版本和全部历史失败/样本。
