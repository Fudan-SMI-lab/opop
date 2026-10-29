# Qwen 接受后续行的断电恢复

用户确认服务器欠费关机后重新开机。2026-10-29 16:46 UTC（香港10月30日00:46）检查发现：SSH已恢复，两张RTX4090空闲，150GiB数据盘可用约11.17GiB；容器PID1为本次新启动，原控制器986775不存在。宿主机 `/proc/uptime` 不是容器运行时长，不能据其声称容器没有重启。原批state仍为running/search，但没有completion文件，这是中断残留状态，不是仍在执行或正常完成。

原批为[Qwen接受后两轮续行](v6-qwen-accepted-continuation-20261029.md)，冻结b18cef24，生产核心ed24af7a保持。已有接受版本重新发现完成，当前eb9dfd94仍安装单个 `fixed-index-read-qwen` bundle，源码为f065，未接受新优化。已完成第一轮 `trial-45c44c1619`：空参数、新代码4857a625，worker失败于 `installed candidate sites differ from the declared bundle`，在完整质量/干净计时之前，决策 `no_valid_parameter_candidate`。这不是数值不忠实的证据，不能改写为质量通过或数值拒绝。该轮诊断与所有原始错误保留；本次不重复第一轮。

第二轮原目录 `trial-49db1bb09a` 尚未生成候选。实际selector初始snapshot为generation-8d3438a88a77，原会话 `ses_e78ea1e7dffeMDKhbbIGaT9pxM` 保存27条message、110个part，最后assistant没有completed标记；最后已完成工具为读取selection schema，sandbox尚无selection.json。33个初始文件与快照字节一致。保存完整现有会话历史于服务器private-history，记录哈希，不转存私有内容到GitHub。原155冻结依赖及作业/角色快照/第一轮文件共271文件核验，恢复前记录独立存于 `qwen-accepted-recovery-20261030/recovery-audit.json`，旧批新增中断说明但不覆盖state/logs/失败记录。

恢复仅通过已有 `--resume-generation`，在相同持久化会话和原sandbox续完第二轮，保留其上下文、原始反馈与当前parent。不会启动新的两轮搜索、重新选择参数预算、重新生成controls/reference或重复初始discovery。现有恢复API会为恢复后的trial分配 `trial-recovered-*` 标识，不能当作额外第三轮。既有恢复提示泛称“temporary service error”；本次实际原因是用户确认的欠费关机，外部审计明确记录这项措辞局限，不改生产prompt，也不伪造provider错误。

先运行真实本地OpenCode server健康检查，读取已存会话确认原消息可见；该预检没有模型调用。随后恢复控制器仅执行剩余一轮，API observe-only，不另设调用上限、不提供开发者诊断/手写方案。现有恢复接口会重写worker-environment和binding-contract输入并捕获continuation snapshot；保存恢复前字节哈希，终态报告任何实际变化。恢复会话不是fresh-snapshot A/B，也不是未中断的全自动成功。

只有这剩余一轮真正接受了不同的活动bundle集合，才按照原协议对完整新版本相对原prepared baseline009b做一次publish=False独立AB/BA。无新接受则not_run；不复测旧f065到通过。完整B1/prompt256/32greedy、BF16/SDPA/noTF32、full logits .02/.02/exact tokens、两同步、2continuous/2warmup/5samples以及原TTFT 24.051052×1.05=25.2536046保持。停机前后属于不同测量时期，不把跨期差值视为配对增量，也不换原约束anchor。手工外部源映射和既有诊断辅助边界保持。

本次不清理数据：实测空间高于8GiB警戒线，依赖、模型、controls、冻结和已接受版本保持。旧监听记录已标记由本次用户轮次处理；不得补发旧“controller退出”通知或重新启动旧监听。恢复长任务只设置一个新完成/磁盘监听，不轮询模型或测试投递。发生磁盘注意事件时先检查worker，不能把低空间当完成；授权冷压缩仍只在GPU工作结束后进行。

历史Qwen f065独立47.3251794%保持，未证明其之前有自主搜索接受；新FSDP K2质量/显存通过但独立噪声拒绝保持。旧fact-assisted Qwen独立45.1691695%/搜索43.9895207%、same-adapter fullgen46.9421035%/TTFT14.06%/decode88.8847%与旧固定拒绝all80分开；FSDP K3 3.095099%和同策略续行2.157792%不叠加，DDP40.61%/55.73%与旧communication拒绝、TP full32质量失败、ResNet拒绝all80和单独36.68%显存接受/missing images事实保持。提示/核验/有限阅读研究关闭，allocator coalescing和不支持的唯一原因/关键路径主张不采纳。
