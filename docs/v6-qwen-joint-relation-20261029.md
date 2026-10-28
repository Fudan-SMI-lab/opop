# 新联合候选的完整输出关系诊断

[一轮约束反馈搜索](v6-qwen-constraint-continuation-20261029.md)的联合prefill/decode候选a86694e7在teacher/greedy token关系上失败。当前记录没有完整greedy/teacher值，不能判定哪一侧先错；candidate_diagnostics只记录正常False，不能补上该信息。下一批仅为这个明确缺口保存原始输出，不提供手工实现或新因果答案。

固定B-observed、C-unobserved、C-observed，各一次。B复用上批基线fixture，C使用原失败screen请求与同一materialized source/candidate、空参数。U只改变output目录，O再换为既有完整观察器a06b6c0e并加入独立reference路径；保存完整greedy/teacher/observe后tokens/logits、有限性、逐位置误差及phase route callbacks。观察器的CPU复制改变同步/host工作/生命期，结果不能用作接受或扣除观察成本；任何意外通过/不同错误都保留，旧失败不改，不重复直到通过。

保持完整B1/prompt256/32tokens、BF16/SDPA/noTF32、原.02/.02和exact token检查、原warmup/2连续质量步骤/5clean样本（若到达）、两同步边界、adapter498c90f3、原TTFT25.2536046。用实际数据核对greedy/teacher各对source、before/after；不把token关系错误自动解释成teacher单独错误、pool别名或cache镜像问题。缺少callback不是无fallback或native路径证明。没有新角色、参数搜索、正式接受、Nsight、discovery或reference复制，不改变source/current/已接受版本。

控制器`scripts/experiments/v6_qwen_joint_relation_20261029.py`，目录`/root/autodl-tmp/dev/v6-work/qwen-joint-relation-20261029`。43个冻结路径哈希及原请求预检通过。原始证据留服务器，GitHub只小摘要/元数据/代码；约10.86GiB可用，预计仅小输出和完整观察张量，无需清理。单普通完成/8GiB监听，无Goal、模型进度轮询或投递测试；既有全部历史接受/拒绝/样本保持。

实际于2026-10-28 22:50:38 UTC启动，冻结`1dc4e139526e47510bc7fea786533a8e8112e3e9`，core仍为`ed24af7a`。控制器PID908055已脱离SSH会话；22:50:59核实基线观察worker908125正在GPU执行，revision仍为b61996e1、空bundle，实际df可用11,599,859,712B。该核查仅证明启动正常，不是完成/质量/接受结论。

唯一普通完成/磁盘监听PID5932于22:50:41启动，state=waiting，30秒文件/SSH检查，低于8GiB提醒，不调用模型等待。上一批constraint监听已标为handled并写stop，不再启动。启动、协议、43哈希预检、worker状态及监听回执见[小证据目录](evidence/v6-qwen-joint-relation-20261029/)。

2026-10-28 22:51:37 UTC终态为completed，即三个尝试完成。B通过完整质量、五clean样本及两次连续检查；C-U/C-O均在原teacher token断言失败，零clean样本，无正式接受。43个冻结文件、原请求差异、adapter/core/entry、原始参考、source/current均复核，空bundle当前b61996e1未变。

独立重读12份完整CPU输出，每份32tokens和1×32×151936 logits，重算SHA、全值、有限性、逐位置误差及关系。B的六份输出与源逐元素一致。C第一次greedy/teacher/after、第二次greedy/after均与源逐元素一致；只有**第二次teacher**发生偏离：prefill位置0仍完全一致，从decode位置1开始logits不同，最大绝对误差31.125，CPU float32容差公式下4,596,832个logit元素超限。12个token不同，位置为1/3/6/10/14/19/22/23/24/27/29/31。两个greedy结果在observe前后均保持不变。因此不能沿用早先四份候选“greedy先错、teacher精确”的解释；这是另一实现、另一实际失败。

O的实际callback为run阶段2次prefill_graph_replay和62次decode graph_replay、observe阶段2次prefill_graph_replay；不能据此声称teacher没有parent路径或新候选已有native证据。代码中teacher输入stride与framework decode signature不同；prefill返回复用mirror、decode可经parent更新cache，但没有此批逐调用cache/layout记录，尚不将具体mirror/pool/生命周期假设写成已证明根因。观察器CPU复制改变同步和生命期；U/O均出现同类断言不能证明所有内部路径相同。失败worker未返回外层完整checks数组，不能据此补写U的成功步数。

审计脚本`v6_audit_qwen_joint_relation_20261029.py`真实重算通过，服务器terminal-audit SHA为`5c1bdffaafe9a628643828f60067fc641799a3a26adf03b75b58a0e38df6b26c`，小摘要见[terminal-audit](evidence/v6-qwen-joint-relation-20261029/terminal-audit.json)。追加current核查时曾把字符串传给RunStore.open导致审计脚本TypeError；改为Path后重算通过，没有重跑模型、改实验或生产代码。旧失败、5个基线样本及12份raw张量全部保留。

下一步按[完整连续关系事实反馈续行](v6-qwen-joint-feedback-20261029.md)，在上一轮框架原反馈5,211,596字节末尾仅追加2,251字节事实/索引，保留原字段及字节。通过已有resume/feedback运行一轮普通selector/candidate，不手工指定组、代码、唯一原因或改生产提示。属于开发者事实辅助续行，不是无辅助新cohort或角色单变量A/B；任何接受仍需后续独立完整版本核验。实际df余11,541,364,736B（约10.75GiB），GPU已空闲；本轮不清理、不下载raw。
