# 事实辅助Qwen候选的独立整版验证

2026-10-29 00:21:30 UTC终态已审计：**独立整版质量、覆盖、原TTFT约束及配对噪声判据全部通过**。独立完整生成710.8134195→389.7449015ms，下降45.1691695%；旧搜索43.9895207%是另一测量阶段，不能相加或覆盖。审计SHA`384c2938633a9cd76be84419b9d332c183737296fbeb83c13b25382909c640f1`，见[全部小样本/身份/原生摘要](evidence/v6-qwen-routed-independent-20261029/terminal-audit.json)和[原终态](evidence/v6-qwen-routed-independent-20261029/followup-completed.json)。

| 独立配对 | B完整生成ms | C完整生成ms | C TTFT ms | 增益 / 噪声界ms |
|---|---:|---:|---:|---:|
| AB | 686.777612 | 389.603713 | 21.591340 | 297.173899 / 2.534682 |
| BA | 734.849227 | 389.886090 | 21.595316 | 344.963137 / 6.982088 |

四个fresh worker、20个原始clean样本、8次连续完整质量检查zero error；没有删样本或补测。逐样本32生成tokens/31decode、计时分解及吞吐公式、逐端点median、配对MAD和原24.051052×1.05=25.2536046约束均重算一致。CPU全量源检查两次zero/exact/finite，整数+1与超容差logit两个负例都被拒绝。5个GPU worker（四评测加一个baseline profile）全部原请求/source/adapter/entry/core身份核验通过。

独立临时整版revision b3d1a4fc，publication=exploration（publish=False）；原接受source仍a3766b50、独立current仍009b4178、原pre-trial分支仍b61996e1空bundle。候选代码d5bd32d4、一个joint bundle/两个组、完整原baseline/controls/reference未变。本批没有新角色生成或搜索；额外角色观察/任意global owner没有由tokens/logits源检查自动认证。

新baseline原生SQLite SHA`c5f28a28b7ab775586b7c453a1ab3a70afceec5863736611b0a59f120b9a36f7`逐行与normalized一致：66,441 kernels、0 Graph、2,304 Flash。复用候选原始SQLite SHA85ffae22同样逐行重算：68,745 kernels、68,712 Graph kernels、32 GraphLaunch、2,304 Flash均在Graph内。候选kernel总数并未减少，实际变化是Graph执行路径；不能据此唯一归因CPU调度或将不同profile计时相减当关键路径开销。两个trace来自不同capture，不冒充同窗口原生配对。

此次结果证明这份**开发者事实辅助生成的候选**有完整、可复测的收益；不能追认原先无人干预搜索成功，也不能证明单字段反馈稳定优于原输入或唯一定位旧失败根因。手工源字段映射、人工设计关系观测仍是泛化工作流缺口。下一步按[诊断采集脚本生成实验](v6-diagnostic-author-20261029.md)，用原始失败现场而非已知答案检验真实Agent能否承担该步骤；先生成审阅，再决定实际worker验证，不自动加入生产或新增硬编码失败路由。

本轮无新清理，实际150GiB quota余10,653,446,144B（9.92GiB）、GPU worker已退出；原冷压缩恢复索引保留，无raw本地传输。历史同adapter46.94%、TTFT/decode接受以及所有原失败/样本保持各自结论。以下为启动时协议及历史状态。

2026-10-29 00:19:21 UTC已启动，冻结`8e90cf1a7952e02d8bda43b3dd7b27ff57be5367`，远端controller PID925432。实际CPU源检查两次完整结果均零误差/exact/finite，两个负例被拒绝；已进入独立AB/BA，启动核查时已有两个真实worker记录。新分支共享原prepared/request/controls，原接受pointer及独立baseline pointer未变。实际df余10,836,918,272B（约10.09GiB）。唯一普通完成/8GiB监听PID52480已运行，旧监听已标记handled；等待不调用模型。见[启动核查](evidence/v6-qwen-routed-independent-20261029/startup-verification.json)、[监听回执](evidence/v6-qwen-routed-independent-20261029/wakeup-setup.json)。此时未完成独立验收，不预判后续结果。

上一批[正确接口反馈续行](v6-qwen-joint-feedback-routed-20261029.md)新代码d5bd32d4通过搜索质量、原始TTFT约束和配对噪声判据，完整生成下降43.99%。本批固定该接受版本a3766b50，使用既有fork_run/promote_version API从原prepared baseline009b4178创建独立分支，`publish=False`。无新Agent、搜索、手工改代码/参数或重试至通过；旧搜索接受与本次独立结论分开保留。

顺序为：原adapter498c90f3对应的已审核外部字段映射，CPU重新核对既有真实CUDA controls与独立source的全部tokens/logits及负例；然后四个fresh worker按B/C/C/B执行，保持每端点2次连续质量、2warmup、5原始clean样本和全部生命周期；最后在质量通过时单独采集原baseline Nsight。当前候选的原生trace已经由真实planner采集并逐行审计，直接复用SHA85ffae22的SQLite和SHA65b267c8的normalized，不重复候选profile。原生计时不计入接受分数，不跨trace相减作host/critical-path成本。

保持B1/prompt256/32greedy tokens、BF16/SDPA/noTF32、完整logits0.02/0.02/exacttokens、两sync、原state所有者。目标end_to_end_ms，约束仍来自本fresh task原controls：24.051052×1.05=25.2536046ms。不使用历史同adapter任务的25.49943795上限。不复制controls/source reference；完整源检查只证明既定tokens/logits映射，不泛化为所有额外owner验证。原source、接受分支current和历史原始结果不变，独立分支也不publish。

本轮有前序手工外部字段映射和开发者实测事实辅助，不能称为无人工介入原入口cohort。单bundle覆盖两个组，不等于多个active bundle共存；旧失败数值、同adapter46.94%及旧TTFT/decode结果不覆盖。若独立质量/约束/噪声拒绝，保存并报告，不自动重测。

控制器`scripts/experiments/v6_qwen_routed_independent_20261029.py`；新目录`/root/autodl-tmp/dev/v6-work/qwen-routed-independent-20261029`。生产core ed24af7a保持，API统计模式保持但本批不生成新角色。前置冷压缩已释放1.45GiB，约10.13GiB可用；仅四个clean worker及一个baseline trace串行，预计远小于新搜索批次。raw/private文件仍在服务器；GitHub只保存代码、哈希和小回执。一次普通完成/磁盘监听，8GiB告警不等于终态；不启用Goal/模型轮询/投递测试。
