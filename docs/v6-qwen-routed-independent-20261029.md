# 事实辅助Qwen候选的独立整版验证

2026-10-29 00:19:21 UTC已启动，冻结`8e90cf1a7952e02d8bda43b3dd7b27ff57be5367`，远端controller PID925432。实际CPU源检查两次完整结果均零误差/exact/finite，两个负例被拒绝；已进入独立AB/BA，启动核查时已有两个真实worker记录。新分支共享原prepared/request/controls，原接受pointer及独立baseline pointer未变。实际df余10,836,918,272B（约10.09GiB）。唯一普通完成/8GiB监听PID52480已运行，旧监听已标记handled；等待不调用模型。见[启动核查](evidence/v6-qwen-routed-independent-20261029/startup-verification.json)、[监听回执](evidence/v6-qwen-routed-independent-20261029/wakeup-setup.json)。此时未完成独立验收，不预判后续结果。

上一批[正确接口反馈续行](v6-qwen-joint-feedback-routed-20261029.md)新代码d5bd32d4通过搜索质量、原始TTFT约束和配对噪声判据，完整生成下降43.99%。本批固定该接受版本a3766b50，使用既有fork_run/promote_version API从原prepared baseline009b4178创建独立分支，`publish=False`。无新Agent、搜索、手工改代码/参数或重试至通过；旧搜索接受与本次独立结论分开保留。

顺序为：原adapter498c90f3对应的已审核外部字段映射，CPU重新核对既有真实CUDA controls与独立source的全部tokens/logits及负例；然后四个fresh worker按B/C/C/B执行，保持每端点2次连续质量、2warmup、5原始clean样本和全部生命周期；最后在质量通过时单独采集原baseline Nsight。当前候选的原生trace已经由真实planner采集并逐行审计，直接复用SHA85ffae22的SQLite和SHA65b267c8的normalized，不重复候选profile。原生计时不计入接受分数，不跨trace相减作host/critical-path成本。

保持B1/prompt256/32greedy tokens、BF16/SDPA/noTF32、完整logits0.02/0.02/exacttokens、两sync、原state所有者。目标end_to_end_ms，约束仍来自本fresh task原controls：24.051052×1.05=25.2536046ms。不使用历史同adapter任务的25.49943795上限。不复制controls/source reference；完整源检查只证明既定tokens/logits映射，不泛化为所有额外owner验证。原source、接受分支current和历史原始结果不变，独立分支也不publish。

本轮有前序手工外部字段映射和开发者实测事实辅助，不能称为无人工介入原入口cohort。单bundle覆盖两个组，不等于多个active bundle共存；旧失败数值、同adapter46.94%及旧TTFT/decode结果不覆盖。若独立质量/约束/噪声拒绝，保存并报告，不自动重测。

控制器`scripts/experiments/v6_qwen_routed_independent_20261029.py`；新目录`/root/autodl-tmp/dev/v6-work/qwen-routed-independent-20261029`。生产core ed24af7a保持，API统计模式保持但本批不生成新角色。前置冷压缩已释放1.45GiB，约10.13GiB可用；仅四个clean worker及一个baseline trace串行，预计远小于新搜索批次。raw/private文件仍在服务器；GitHub只保存代码、哈希和小回执。一次普通完成/磁盘监听，8GiB告警不等于终态；不启用Goal/模型轮询/投递测试。
