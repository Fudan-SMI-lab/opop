# 新联合候选的完整输出关系诊断

[一轮约束反馈搜索](v6-qwen-constraint-continuation-20261029.md)的联合prefill/decode候选a86694e7在teacher/greedy token关系上失败。当前记录没有完整greedy/teacher值，不能判定哪一侧先错；candidate_diagnostics只记录正常False，不能补上该信息。下一批仅为这个明确缺口保存原始输出，不提供手工实现或新因果答案。

固定B-observed、C-unobserved、C-observed，各一次。B复用上批基线fixture，C使用原失败screen请求与同一materialized source/candidate、空参数。U只改变output目录，O再换为既有完整观察器a06b6c0e并加入独立reference路径；保存完整greedy/teacher/observe后tokens/logits、有限性、逐位置误差及phase route callbacks。观察器的CPU复制改变同步/host工作/生命期，结果不能用作接受或扣除观察成本；任何意外通过/不同错误都保留，旧失败不改，不重复直到通过。

保持完整B1/prompt256/32tokens、BF16/SDPA/noTF32、原.02/.02和exact token检查、原warmup/2连续质量步骤/5clean样本（若到达）、两同步边界、adapter498c90f3、原TTFT25.2536046。用实际数据核对greedy/teacher各对source、before/after；不把token关系错误自动解释成teacher单独错误、pool别名或cache镜像问题。缺少callback不是无fallback或native路径证明。没有新角色、参数搜索、正式接受、Nsight、discovery或reference复制，不改变source/current/已接受版本。

控制器`scripts/experiments/v6_qwen_joint_relation_20261029.py`，目录`/root/autodl-tmp/dev/v6-work/qwen-joint-relation-20261029`。43个冻结路径哈希及原请求预检通过。原始证据留服务器，GitHub只小摘要/元数据/代码；约10.86GiB可用，预计仅小输出和完整观察张量，无需清理。单普通完成/8GiB监听，无Goal、模型进度轮询或投递测试；既有全部历史接受/拒绝/样本保持。
