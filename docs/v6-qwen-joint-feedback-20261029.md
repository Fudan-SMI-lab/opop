# 完整连续输出事实交回普通搜索

**启动后核查更正：本批新增事实未进入实际selector输入。** 冻结controller把字段加在顶层，而现有`feedback.route`只转发约定字段，其中已经有`supplemental_evidence`。当前实验保留运行，实际应归类为“普通原失败反馈续行，实验性新增事实传递失败”，不能把结果归因为Agent使用了新增事实。以下协议保留原始意图；没有修改运行中的输入、重启或同时启动修正版。

[固定关系诊断](v6-qwen-joint-relation-20261029.md)确认a86694e7候选的第二次teacher输出失败，其他保存输出精确匹配源。框架现有candidate_diagnostics不执行observe/compare，不能提供这项定位信息。本批只向已有失败反馈追加这项实际缺口，检验Agent能否据此推进，不引入自动硬编码错误路由或手工候选。

从`qwen-constraint-continuation-20261029/qwen-continuation`当前b61996e1空bundle分支通过现有fork_run复用原controls、request、adapter498c90f3、contract和完整discovery。原`latest-feedback.json`为5,211,596字节，SHA`d426e3dfe24be2cd47107ddaed95c1d656fd4066c4328680fbed5e87101f246b`；保持所有字段和值以及最终右花括号前的全部字节，仅追加`fixed_continuous_full_output_diagnostic`2,251字节，形成5,213,847字节输入。新字段为两步greedy/teacher/source关系、实际回调、限制和小证据路径/SHA，不包含根因答案、手工修复、旧成功bundle或性能建议。

既有`candidate_artifacts`确认历史代码精确为a86694e7、所需产物可用；该失败候选没有安装。普通CLI `--resume --rounds 1 --feedback`运行新真实selector/candidate，沿用所有允许scope、默认参数探索和正常schema修复。没有新discovery、reference复制或本批独立AB/BA。必须分别报告执行/质量/约束/噪声/自主接受；如果出现新接受版本，下一阶段先审核再做独立完整版本与native核验，失败则保留，不重复直到通过。

任务仍完整B1/prompt256/32tokens、BF16/SDPA/noTF32、logits.02/.02及exacttokens、两同步边界及原生命期，TTFT约束25.2536046来自原24.051052基线。生产core仍ed24af7a，prompt/sampler/allocator/接受规则不改。该批是开发者事实辅助的普通反馈续行，不是新无辅助入口cohort，也不是同快照单变量角色A/B。原始手工外部source字段映射边界继续标明；全量输出比较不代表所有额外owner语义均已证明。

控制器`scripts/experiments/v6_qwen_joint_feedback_20261029.py`，输出`/root/autodl-tmp/dev/v6-work/qwen-joint-feedback-20261029`。真实预检已验证core、空bundle、追加字节、原反馈语义/字节、历史candidate及共享controls；API费用只观察、不限额。df约10.75GiB，仅一轮，不并行大型任务；保留唯一完成/8GiB普通监听。raw/private配置留服务器，GitHub仅代码/小摘要/哈希。所有历史接受/拒绝/样本保持，未重启旧实验或启用Goal/模型轮询/投递测试。

2026-10-28 22:58:46 UTC实际启动，冻结`adde952565b4bd5655b0ff1b15436272e5c23393`，控制器PID909953，core仍ed24af7a。真实OpenCode健康预检通过。唯一普通监听PID56328于22:58:49启动、30秒文件检查/8GiB提醒，旧joint-relation监听已handled。23:00:42真实selector会话`ses_e7b4309cdffeYSILjYx1S2qXu4`已有20条assistant记录；快照`generation-dac42ec386f8`29份初始输入，历史a866代码SHA正确，目标保持end_to_end。该时刻尚无GPU worker，控制器正常运行，df余11,330,641,920B。

检查初始输入时发现`previous-feedback.txt`经过普通路由后为5,735,456字节，使用`previous_trial`等已有字段，`supplemental_evidence=null`，不含新增字段/事实。启动核查脚本最初错误假设文件未经路由并读取`trial`而KeyError；此处没有生产执行错误。随后`v6_check_qwen_feedback_routing_20261029.py`在服务器真实CPU复现完整`feedback_artifacts→route(registry)`，得到与实际snapshot逐字相同的输出：原feedback和顶层加字段feedback输出相同。将相同事实放入现有`supplemental_evidence`可保留内容且其他路由字段相等；**这个修正payload只做CPU序列化检查，未提交角色/GPU**，也未改生产路由。

本批保持冻结等待其原始结果；后续是否再通过正确现有接口续行，取决于实际结果，不自动重复直到通过。须检查candidate初始快照与实际工具读取；外部文件可访问，初始输入没有事实不代表后续绝无可能从其他路径发现事实。不能将潜在成功/失败解释为新增诊断效果。单监听的未来通知已追加此更正，无新通知/重启。详见[启动证据](evidence/v6-qwen-joint-feedback-20261029/startup-verification.json)、[路由实证](evidence/v6-qwen-joint-feedback-20261029/routing-interface-audit.json)及[原问题记录](evidence/v6-qwen-joint-feedback-20261029/startup-discrepancy.json)。
