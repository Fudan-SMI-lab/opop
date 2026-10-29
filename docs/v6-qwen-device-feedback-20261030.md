# 当前设备原基线上的一次普通反馈试验

这是[新设备固定B/C及setup诊断](v6-qwen-device-refresh-fixed-20261030.md)之后单独声明的一次普通试验，不是旧两轮恢复的第三轮或重新启动旧批。使用现有fork_run/CLI resume，从当前设备新prepared原baseline009b、空活动bundle开始，执行一次fresh discovery和一轮真实selector/candidate/default参数探索。无新intake/evaluator/源模型运行、controls重测或参考复制；已完成的新controls与独立源检查只读复用。生产core/prompt/sampler/allocator/接受规则不变，API observe-only，无新增上限。

历史实现仅为确切f5b76ebd/原proposal/selection及固定B/C原结果，未安装该候选或旧f065。开发者创建的diagnostic-feedback.accepted=false明确是“固定诊断且快路径关闭”标记，不是假造配对拒绝。完整实际candidate_diagnostics通过既有resources.extra_collectors/feedback_artifacts提供，少量范围说明经existing supplemental_evidence传递。CPU预检确认接口保留事实和历史代码；实际角色是否读取、如何解释，仍要查初始快照和工具，不能从transport通过推断消费。

反馈范围是可核对的当前事实：固定C完整质量通过但质量阶段62次内部setup回退，单独run-only诊断首decode内部logit自检失败；内部0.008/0.008与原0.02/0.02不同，未保存数值/完整张量，不能判定原质量违反或唯一原因。不提供修复代码、分组、算术归因或旧成功方案。普通selector自行根据新原baseline的实际discovery选择范围。此前父路径质量/固定时序不是新快路径优化接受；B TTFT超限和全部长样本保留。

请求保持完整B1/prompt256/32greedy BF16/SDPA/noTF32、完整logits.02/.02/exact tokens、两同步/owner、2continuous/2warmup/5clean。当前设备原TTFT23.285755×1.05=24.45004275，不使用旧25.2536046。新搜索仍需原screen/配对噪声/约束规则；只运行这一新trial，保留生成/worker/质量失败，不重试到通过。没有在本controller安排独立整版验证或固定native；真实planner可按现有机制采集。若有新接受，须后续审计再决定独立验证，不预设优化成功或多bundle共存。

这是设备迁移与开发者事实辅助的同adapter续行，不是未辅助原入口或单变量A/B。源手工映射和以前owner辅助边界不消失；旧接受47.325%及其他历史决定不改写，不作跨设备增量。空间约10.81GiB，一次串行试验，无并行大Qwen/参考副本；299冻结身份预检。启动时验证实际UUID和真实本地OpenCode健康，长任务只设一个完成/磁盘监听，无模型轮询/投递测试。raw/private配置留服务器，GitHub仅小摘要/代码/哈希。
