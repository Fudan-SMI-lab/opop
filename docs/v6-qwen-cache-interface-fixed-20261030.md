# Qwen缓存接口候选原字节固定执行

声明一次B后C固定诊断，使用[真实角色审计](v6-qwen-cache-interface-candidate-20261030.md)的精确代码 `7e78b82dd21391e3db3d6292e3798feff1908f53259aa732a3c011660c28fb92` / proposal `048d30b638809aad76ef99e757be41ea20345d1def2f72da5c332cc56aaf94b6`，空参数。无新Agent、搜索、手改、调参、观察器、Nsys或正式性能接受。生成自检的flag/所有权风险保留，由原外层评价器检验真实安装、派发、完整质量与约束。

根目录 `qwen-cache-interface-fixed-20261030`；既有fork_run/bundle_from_outputs/stage/execute从当前设备空parent `2a42b9b182b77d54a4028e779d8ba8f8bee5dabec4a83917015ed52e486e0dd1` 建隔离分支。原request/discovery/adapter/contract与controls只读复用，不重新发现或复制参考张量。源和隔离current均保持parent，stage不是publish。B仅以完整质量/coverage为C前置门，B的TTFT不重新定义约束；B无效则C不运行。

原始B1/prompt256/32 greedy、BF16/SDPA/noTF32、全logits .02/.02/exacttokens、两同步/状态生命期、2连续质量/2warmup/5clean（若到达）保持。当前20e8 GPU原TTFT锚23.285755，上限24.45004275；不采用旧设备锚。保留所有尾样本、内部fallback、安装错误/未到达阶段，不重试到通过。固定计时不等于配对或独立接受，不借旧f065/d5bd trace。

真实DynamicCache支持父方法接口，但31个cache复用且ownership主要查第一层keys指针；teacher框架stride不匹配仍可能走父路径。直接自检未覆盖真实框架两次run/observe，verify异常未清fast_enabled，eager回退后的标签可能仍写Graph。原字节执行检验这些风险，不据此宣称唯一原因或任意并发生命期安全。

启动前df约8.11GiB、GPU空闲；分支discovery约68.9MB、原固定C约20.2MB，预计仅小规模结果/元数据增长，无大型profile或参考复制。本轮不清理、raw/private会话不出服务器。接近8GiB监听线需按实际df判断，告警不是完成，不终止实验或盲删。新长工作使用一个隐藏普通完成/磁盘监听，旧生成监听已处理。生产core `ed24af7acc8caa15578e0b859c85ef8a901be51354448ed39094606ed9e10a2f`、prompt/sampler/allocator/接受规则不变；人工源映射/此前CPU辅助边界及历史接受拒绝保持。
