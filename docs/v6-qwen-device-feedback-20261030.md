# 当前设备原基线上的一次普通反馈试验

**已于2026-10-29 17:49:17 UTC（香港10月30日01:49）启动**，冻结21aafdb7/controller10522。[启动回执](evidence/v6-qwen-device-feedback-20261030/launch.json)、[真实本地服务健康](evidence/v6-qwen-device-feedback-20261030/session-health.json)、[299身份/实际UUID/CPU反馈路由预检](evidence/v6-qwen-device-feedback-20261030/frozen-preflight.json)已保存。生产核心仍ed24af7a，无本轮清理。

17:53的[实际启动核验](evidence/v6-qwen-device-feedback-20261030/startup-verification.json)确认新设备原baseline fresh discovery worker10662及Python runtime worker11472均通过，均实际绑定GPU-20e8003c、entry aed604ee/core相同；discovery后current2a42b9b1仍空bundle。共享的是本次新controls路径，299冻结文件保持。初始真实collection-planner generation-bce225ade347/session ses_e7736e9faffezOSCwKwO7Cvjyv，6输入/当时11工具；该角色没有previous-feedback输入，**此时尚不能声称后续selector/candidate收到或读过事实**。discovery的60次进程占用快照有1次不可用，不能据无外部client记录排除全部干扰；诊断不是干净时序。

唯一普通完成/8GiB监听 **44008** 已实际运行，[监听启动](evidence/v6-qwen-device-feedback-20261030/wakeup-setup.json)和[等待状态](evidence/v6-qwen-device-feedback-20261030/wakeup-verification.json)留档；旧23004已处理，两项短批未创建监听。核验时controller运行、Agent/collector阶段GPU暂空，磁盘约10.2GiB。后续由完成/磁盘事件接续，无模型轮询或额外投递测试。以下为本批预先声明范围。

这是[新设备固定B/C及setup诊断](v6-qwen-device-refresh-fixed-20261030.md)之后单独声明的一次普通试验，不是旧两轮恢复的第三轮或重新启动旧批。使用现有fork_run/CLI resume，从当前设备新prepared原baseline009b、空活动bundle开始，执行一次fresh discovery和一轮真实selector/candidate/default参数探索。无新intake/evaluator/源模型运行、controls重测或参考复制；已完成的新controls与独立源检查只读复用。生产core/prompt/sampler/allocator/接受规则不变，API observe-only，无新增上限。

历史实现仅为确切f5b76ebd/原proposal/selection及固定B/C原结果，未安装该候选或旧f065。开发者创建的diagnostic-feedback.accepted=false明确是“固定诊断且快路径关闭”标记，不是假造配对拒绝。完整实际candidate_diagnostics通过既有resources.extra_collectors/feedback_artifacts提供，少量范围说明经existing supplemental_evidence传递。CPU预检确认接口保留事实和历史代码；实际角色是否读取、如何解释，仍要查初始快照和工具，不能从transport通过推断消费。

反馈范围是可核对的当前事实：固定C完整质量通过但质量阶段62次内部setup回退，单独run-only诊断首decode内部logit自检失败；内部0.008/0.008与原0.02/0.02不同，未保存数值/完整张量，不能判定原质量违反或唯一原因。不提供修复代码、分组、算术归因或旧成功方案。普通selector自行根据新原baseline的实际discovery选择范围。此前父路径质量/固定时序不是新快路径优化接受；B TTFT超限和全部长样本保留。

请求保持完整B1/prompt256/32greedy BF16/SDPA/noTF32、完整logits.02/.02/exact tokens、两同步/owner、2continuous/2warmup/5clean。当前设备原TTFT23.285755×1.05=24.45004275，不使用旧25.2536046。新搜索仍需原screen/配对噪声/约束规则；只运行这一新trial，保留生成/worker/质量失败，不重试到通过。没有在本controller安排独立整版验证或固定native；真实planner可按现有机制采集。若有新接受，须后续审计再决定独立验证，不预设优化成功或多bundle共存。

这是设备迁移与开发者事实辅助的同adapter续行，不是未辅助原入口或单变量A/B。源手工映射和以前owner辅助边界不消失；旧接受47.325%及其他历史决定不改写，不作跨设备增量。空间约10.81GiB，一次串行试验，无并行大Qwen/参考副本；299冻结身份预检。启动时验证实际UUID和真实本地OpenCode健康，长任务只设一个完成/磁盘监听，无模型轮询/投递测试。raw/private配置留服务器，GitHub仅小摘要/代码/哈希。
