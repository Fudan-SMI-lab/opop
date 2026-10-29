# Qwen执行接口失败后的单次候选生成

**已启动：2026-10-29 21:59:05 UTC（香港10月30日05:59），冻结61a968f34b45bc00f7efe53f39728c1812b6d901，controller52752。** [启动](evidence/v6-qwen-cache-interface-candidate-20261030/launch.json)、[492身份预检](evidence/v6-qwen-cache-interface-candidate-20261030/frozen-preflight.json)、[本地服务实际健康检查](evidence/v6-qwen-cache-interface-candidate-20261030/session-health.json)通过。

22:01:21 UTC[实际启动审计](evidence/v6-qwen-cache-interface-candidate-20261030/startup-verification.json)确认generation `a8d232f6cb9b` / session `ses_e76533c6cffeeLrihDWiZF4r03`，33初始文件/130,630,326B逐字匹配，prompt/model/tools/core相同，registry+plan路由一致、supplemental_evidence=null，历史6ec4代码及492冻结身份保持。11条assistant消息/18工具；实际读取新失败sidecar和完整原始traceback（工具12）、原target签名（工具13/15）。这只证明早期失败信息读取，不是正确修复或质量证明。工具14错误dev-v6-work路径被stderr抑制且管道exit0保留；早期无GPU worker。

数据盘余8,707,149,824B（约8.11GiB），无清理。用户随后报告本地PowerShell弹窗：检查工具确实使用PowerShell，且隐藏Python监听的SSH子进程未显式隐藏，是短暂控制台弹窗的可能来源，无法追溯所有已消失窗口。现已给本地SSH/通知子进程设置CREATE_NO_WINDOW；活动监听652正常停止、未投递事件后替换为17992，[记录](evidence/v6-qwen-cache-interface-candidate-20261030/wakeup-window-fix.json)保留。仍只有一个普通完成/8GiB监听，实验没有重启，未动VS Code终端。只改变本地编排，不改冻结服务器框架或模型请求。

在[CPU事实反馈普通试验](v6-qwen-setup-cpu-feedback-20261030.md)的四个worker均因 `_CacheHandle.get_seq_length` 缺失而失败后，声明一次现有candidate角色生成，根目录 `qwen-cache-interface-candidate-20261030`。这次没有新selector、完整搜索、参数执行、GPU worker、接受或指针发布。它不是重启旧trial，也不是重复评分直到通过。

输入以实际generation `d9a0f3917237` 的原任务、已选decode scope、source/adapter/contract/targets/current discovery为基础；更新为本轮framework原始latest-feedback、精确失败代码 `6ec4e97a…` 的历史产物和原始失败collector包。通过既有feedback_artifacts/candidate_artifacts/registry+plan route及ProjectAgents.generate交付，使用原candidate提示/model/tools，正常schema修复保留。旧selection中有关opaque handle/stride/TTFT的错误推断不手工改写，实际失败反馈可供角色核对。不是整角色单变量A/B。

CPU预检通过：492冻结文件，33预期输入共130,630,326B；更新6文件、移除旧binding_cost包、增加失败Nsys包。routed反馈5,305,639B/SHA `c68771a7585cd4418a1da229f977e504b37ffc8dfab534d7b393fc2edc3f04a6`，supplemental_evidence为null。没有提供开发者CPU binding控制、终态事实表、因果答案或修复实现；CPU控制仅作外部审计。此前开发者CPU事实与人工源映射辅助边界仍存在，不能称为无辅助原入口结果。外部文件系统可访问，后续必须检查完整工具路径、输出及CPU探针，不以初始文件名推断没有污染。API仅观察费用，不设新上限；不改生产提示、采样、allocator或接受规则。

源current保持空bundle `2a42b9b1…`，prepared009b及当前20e8 GPU原controls/discovery只读共享；不复制reference或运行新发现。完整B1/prompt256/32 greedy BF16/SDPA/noTF32/full logits .02/.02/exacttokens、两同步和状态生命期不改。当前原TTFT锚23.285755、上限24.45004275保持，旧设备指标不合并。

该控制器只生成candidate.py/proposal.json，语法/schema完成不是执行有效性。终态需要审查实际返回缓存、框架签名派发、参数、fallback、CPU模拟范围及所有错误；适合时另声明精确字节固定执行，不能隐藏修复或自动GPU验证。没有新collector或强制review层，关闭研究保持关闭。

启动前核对实际150GiB df与GPU，约8.37GiB可用，两卡空闲；本批仅角色输入/会话/小代码，没有大型profile或GPU源副本。raw/private配置与会话留服务器，GitHub仅代码/哈希/小摘要。新长工作只设一个普通完成/8GiB监听，low_disk_space不是完成；旧40560事件已处理，不重启旧批次。
