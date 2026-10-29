# Qwen原样setup观察产物的一轮普通反馈续行

在[固定观察](v6-qwen-setup-observation-20261030.md)完成并审核后，声明一轮新的普通selector/candidate试验。原来的`batch_finished`通知已在上一轮处理；本次先核对相同终态与audit SHA，再决定新请求，不重启观察、旧试验或旧监听。服务器两卡空闲，150GiB实际余约9.55GiB，无本轮清理。

起点是`qwen-device-feedback-20261030/qwen-continuation`当前空bundle版本`2a42b9b182b77d54a4028e779d8ba8f8bee5dabec4a83917015ed52e486e0dd1`，共享当前设备原controls和已完成baseline discovery。原prepared baseline仍`009b4178…`，没有再次controls、源比较、初始GPU发现或参考复制；既有fork_run复制discovery元数据并共享不可变artifacts/reference。原source/current及历史接受分支不动，新隔离分支只按现有搜索规则形成自己的结果。

原framework latest-feedback为13,935,818B、SHA`2cf4eb2a2476f398bcbe0cd045b0790374d51852e31bea86d64416480e87b25c`，来自trial0db61e9938的`no_valid_parameter_candidate`。保留原全部字段和值、末尾右括号之前的字节，仅新增现有` supplemental_evidence `容器下的diagnostic_artifacts索引指针。索引只列原样probe/plan、既有B/C原请求、观察请求、worker结果和观察JSON的路径/SHA/大小，以及通用观察范围限制；不加入开发者审计事实表、cache长度结论、实现修复、其他已接受候选或解释/核验角色答案。原04df代码仍通过历史candidate_artifacts提供，不安装到起点。CPU预检实际feedback_artifacts与registry/plan route，要求其他路由字段和原打包产物相同；真正新角色的初始快照和实际读取仍需单独核验。

这是对新增数值与输入来源证据的一次普通使用，不是同快照单变量A/B、原入口无辅助cohort或稳定提示因果证明。生产selector/candidate提示保持，不重新打开已关闭的bounded-reading/解释/核验研究，也不强制选组或手写候选。既有CLI `--resume --rounds 1 --feedback`使用默认作用域/参数探索；API仅观察用量，不设新上限。保留正常schema修复、工具错误、CPU/GPU探测和所有失败，不重试至通过。原角色工具可能自行探测，不能只凭controller配置宣称没有GPU活动；外部FS非隔离，终态审查完整实际访问。

完整B1/prompt256/32 greedy、BF16/SDPA/noTF32、full logits .02/.02/exact tokens、两同步边界/状态生命周期、两次连续质量/两次warmup/每端点五样本保持。当前GPU0是GPU-20e8003c-f531-d50c-084c-a9a97fe94004；当前原TTFT anchor23.285755、limit24.45004275，不能用旧25.2536046/25.49943795或跨设备合并anchor。adapter498c90f3及manual generated_tokens/step_logits源映射边界保留，来源参考/controls只读。本批没有独立完整版本或专门native阶段；真实planner可请求既有collector。若新版本被搜索接受，终态审计后才安排另一次独立完整版本验证；controller完成或run-only诊断valid都不等于质量/性能接受。

固定观察既有结论不改：370身份、10样本/4完整零误差检查，候选62父decode回退+2父prefill，观察TTFT24.589549超限，无快路径接受。新probe保存198比较中的2失败；256个保存超差值算术已核验，完整操作数未保存，聚合数不能从raw重算。首setup存在超原容差的保存项；E最后两parent输入长度不同是代码/顺序推断，不是所有差异唯一原因。这些开发者结论只在本计划说明，不作为新角色事实表或修复答案输入。原probe、原失败、原timing全部冻结。

新任务根`/root/autodl-tmp/dev/v6-work/qwen-setup-feedback-20261030`。预检检查actual UUID、df、冻结文件、原始反馈和历史代码；本地OpenCode实际start/health/stop、不发送模型提示，私有设置只留服务器。仅代码、SHA、脱敏小摘要发布GitHub v6，raw和完整会话不转存。启动后只设一个普通完成/8GiB监听；低磁盘不是完成，不终止实验或盲删。既有cold JSON恢复journal保持。生产core `ed24af7acc8caa15578e0b859c85ef8a901be51354448ed39094606ed9e10a2f`、历史接受拒绝及所有关闭研究不变。
