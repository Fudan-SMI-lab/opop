# DDP 长尾与审阅输入呈现的最小对照

承接[固定DDP/TP批次审计](v6-ddp-tp-diagnostics-20261028.md)。上一批完整执行，未做新的正式接受；C/N共7个约445–519 ms长尾集中在第8/13次附近，关闭Gloo barrier没有消除它们。GC与逐次CPU限流尚无直接证据，因此先做本批诊断，不修改生产执行逻辑。

## 固定 DDP GC 对照

所有端点使用同一个C版本 `75e28d3d2ccd5fab06175ef1b827a325ed2159d011758c7b2deccda6173b9e2d`、相同adapter、参考、工作量、参数、质量与整步约束。顺序 **U/O/D/D/O/U**，每端点20样本、两次warmup。U调用原evaluator；O增加GC回调及计时外CPU计数读取；D与O只差在measure阶段关闭循环GC，离开该阶段恢复。这里measure包括warmup、reset和测量样本，D会改变对象生命期/资源使用，明确是开发干预，不能作为候选收益或从旧分数扣除。

GC回调记录start/stop、generation、线程、单调时钟；generation2另保存少量触发栈位置，不操作张量、不加CUDA同步或强制GC。计时外记录进程CPU、缺页、切换和容器CPU限流计数。事件包围范围比真正计时略宽，容器计数也并非某个worker独占；分别保留这些局限。观察器本身会产生Python对象与计数读取开销，U用于检查其影响。原始事件与所有样本完整保留；不按结果追加试次，不调整原约束，不把无GC的测量解释成正式优化。

GPU作业串行执行，直接复用原参考，不复制多GB控制文件。错误/质量失败完整写入receipt。样本级GC重叠与限流变化只是定位证据，需要结合干预、触发位置及观察器影响判断原因。

## 真实审阅角色单文件对照

从真实初始快照 `generation-d880018dc713` 分别创建control和treatment新会话。保持原prompt、模型、工具配置、生产核心和其它9个输入文件完全一致；只将 `inputs/tp-summary.json` 从94,970 bytes改为30,049 bytes，比较对象作为显式键、数值作为表格行。

全部6组projection、全部变体对三个reference的status/exact/elements/max/RMS逐项验证不变；保留权重/输入映射和原始观测/输出路径，移除重复output metadata等展开信息，可回查原始索引。**不加入人工纠正答案、GC新结果或新的因果提示。** 两臂实际生成前快照再次核对唯一文件差异。外部文件系统/provider状态并未完整冻结，一次对照不代表稳定普遍收益。无论角色回答如何，本批不自动修改生产prompt或采集器。

## 准备与执行

脚本 `v6_ddp_gc_probe_20261028.py`、`v6_gc_review_batch_20261028.py`。CPU隔离检查已验证回调捕获generation2、返回值不变、GC启用状态/loader/回调列表恢复；TP呈现转换逐项数值相等。该检查不代表GPU质量或长尾因果成立。

工作目录 `/root/autodl-tmp/dev/v6-work/gc-review-diagnostics-20261028`。源代码Git冻结，使用现有真实provider配置，API/费用只统计、不限额。启动前数据盘约48.91 GiB可用。本批不启动大型TP重测，不大体积转存本地；原始trace/张量/样本留服务器，GitHub仅存代码、协议和小型证据。

旧DDP独立拒绝、TP完整32-token失败、旧Qwen拒绝及80样本、ResNet独立拒绝及80样本、旧不同adapter的36.68%接受与missing images事实均保持。长批次挂唯一普通完成监听；不启用Goal、不增加投递测试或模型进度轮询。

## 已启动

2026-10-27 18:31:35 UTC（香港时间10月28日02:31）启动，冻结提交 `00a8d89e12b3a77ecc3984abba32824632b35e88`，控制器PID `309314`。生产核心SHA仍为 `cb4ea28c3d90ccad5bc723ed5b6dc124488d572fca8a69b3ed3b12ec0e994016`。冻结环境已核对真实重放的model/agent/tool settings与原快照一致。

启动核查时首个U端点 `valid=true`，后续双卡worker已运行，数据盘剩52,405,944,320 bytes（约48.81 GiB）。普通完成监听PID `52568`，状态 `.scratch/v6-gc-review-wakeup.json` 为waiting，已真实探查控制器存活。这里只有启动与首端点确认，不预判GC因果或角色对照结果；详见[启动证据](evidence/v6-gc-review-diagnostics-20261028/README.md)。
