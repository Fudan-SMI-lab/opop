# 完整连续输出事实交回普通搜索

[固定关系诊断](v6-qwen-joint-relation-20261029.md)确认a86694e7候选的第二次teacher输出失败，其他保存输出精确匹配源。框架现有candidate_diagnostics不执行observe/compare，不能提供这项定位信息。本批只向已有失败反馈追加这项实际缺口，检验Agent能否据此推进，不引入自动硬编码错误路由或手工候选。

从`qwen-constraint-continuation-20261029/qwen-continuation`当前b61996e1空bundle分支通过现有fork_run复用原controls、request、adapter498c90f3、contract和完整discovery。原`latest-feedback.json`为5,211,596字节，SHA`d426e3dfe24be2cd47107ddaed95c1d656fd4066c4328680fbed5e87101f246b`；保持所有字段和值以及最终右花括号前的全部字节，仅追加`fixed_continuous_full_output_diagnostic`2,251字节，形成5,213,847字节输入。新字段为两步greedy/teacher/source关系、实际回调、限制和小证据路径/SHA，不包含根因答案、手工修复、旧成功bundle或性能建议。

既有`candidate_artifacts`确认历史代码精确为a86694e7、所需产物可用；该失败候选没有安装。普通CLI `--resume --rounds 1 --feedback`运行新真实selector/candidate，沿用所有允许scope、默认参数探索和正常schema修复。没有新discovery、reference复制或本批独立AB/BA。必须分别报告执行/质量/约束/噪声/自主接受；如果出现新接受版本，下一阶段先审核再做独立完整版本与native核验，失败则保留，不重复直到通过。

任务仍完整B1/prompt256/32tokens、BF16/SDPA/noTF32、logits.02/.02及exacttokens、两同步边界及原生命期，TTFT约束25.2536046来自原24.051052基线。生产core仍ed24af7a，prompt/sampler/allocator/接受规则不改。该批是开发者事实辅助的普通反馈续行，不是新无辅助入口cohort，也不是同快照单变量角色A/B。原始手工外部source字段映射边界继续标明；全量输出比较不代表所有额外owner语义均已证明。

控制器`scripts/experiments/v6_qwen_joint_feedback_20261029.py`，输出`/root/autodl-tmp/dev/v6-work/qwen-joint-feedback-20261029`。真实预检已验证core、空bundle、追加字节、原反馈语义/字节、历史candidate及共享controls；API费用只观察、不限额。df约10.75GiB，仅一轮，不并行大型任务；保留唯一完成/8GiB普通监听。raw/private配置留服务器，GitHub仅代码/小摘要/哈希。所有历史接受/拒绝/样本保持，未重启旧实验或启用Goal/模型轮询/投递测试。
