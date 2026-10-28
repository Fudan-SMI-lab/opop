# FSDP2 分片训练源就绪性与新原入口检查

承接 [Qwen decode 与指标接口审计](v6-metric-mapping-decode-20261028.md)。目标切换已有新的独立 decode 正例；通信方面仍需区分 DDP 的 host dispatch 改善与通信本身的收益。旧 TP BF16 完整 32-token 质量失败保持不变，本批不继续扩大其诊断上下文或放宽数值标准。

新任务位于 `examples/v6_fsdp_training`：GPT-2 Medium（24 层、hidden 1024、约 355M 参数），FP32、eager attention、noTF32、dropout=0，双 rank 各一条长 128 的确定性 token 输入，完整 forward/loss/backward/global mean gradient/SGD update。权重和 token 为固定 seed 生成，用于执行语义与通信验证，不是预训练语言质量实验。模型形状和任务约束属于独立项目输入，框架没有按模型名分流。

采用 [PyTorch 官方 FSDP2](https://docs.pytorch.org/docs/main/distributed.fsdp.fully_shard.html) 的 bottom-up `fully_shard` 默认策略，每 block 后 root；不插入每层屏障、强制串行、小 bucket 或禁用默认预取。它提供区别于既有 DDP all-reduce 的 all-gather/reduce-scatter 工作流，但是否真正成为通信敏感的可优化案例，要看实测。分片策略和实现可替换，完整工作与质量必须保留。任务主目标为完整 `step_ms`，并保持 source 定义的 `peak_allocated_bytes <= 1.05 baseline`，以保留分片任务的显存预算；适用通信采集为独立诊断，不伪称 wire bandwidth 或暴露关键路径。

固定阶段：

1. 独立源就绪性：对每 rank 的完整 logits、本地实际 DTensor shard 全量梯度与更新后参数，比较未分片 serial global-batch 的两步连续轨迹。reference 顺序执行两个 rank 输入，每项 loss 权重 1/world，再做一次 SGD；不调用 sharded `step()`。容差固定 atol=1e-6、rtol=1e-4，+1 logits 负例必须拒绝，embedding/head 必须保持共享。分片不是副本，不能要求各 rank 不同 shard 数值相等；布局需要按实际 placement 映射。保留全量 source observation 在服务器，供后续独立 adapter 映射检查。
2. 源性能与通信：reset 在计时外，完整 step 前后 CUDA 同步；2 warmup、5 clean，各 rank 对齐后先 max 再 median。完整 logits 持有到内存指标读取之后。另做 3 次通信 profile，记录 all-gather/reduce-scatter 原生 kernel、NCCL sum/union 与完整 profiler 窗口。嵌套 CPU 范围不累加，profiler 与 clean 时间不混用。
3. 只有完整源质量通过，才用新生产核心从 project/command/goal 原入口执行真实 intake → evaluator → controls → discovery。不给现成 adapter、分组、候选、开发者诊断或 readiness 数值。新可选 diagnostic_metrics 由 Agent 自主选择，未选或不支持均照实保留；不补答案。此批止于 `--discover-only`，下一轮先审计实际源与 adapter 的完整语义/所有权/布局，再决定搜索。controls 通过不等于独立 source fidelity 通过。

预检使用单独的 2 层 hidden128 API fixture，两 rank 两步各 441,857 个元素零误差，负例拒绝；只证明本机 API/布局操作可运行，不代替完整 source 质量。目录 `fsdp-readiness-precheck-20261028` 保留，不能混入完整工作样本。生产 intake schema 已核对与前批 treatment schema 完全一致，核心 SHA `ed24af7acc8caa15578e0b859c85ef8a901be51354448ed39094606ed9e10a2f`；27 项相关检查通过，prompt/allocator 未改。

数据盘启动前可用约 33.40 GiB。预计 source 全量两步梯度/参数分片证据约 5.6 GiB，新的 evaluator 参考可能再占相近空间，trace 另计；不启动并行 Qwen 或多个大 trace。保留已有模型/接受版本/冻结结果，只保存新必要工作集；raw 不下载本地，GitHub 仅代码、协议、索引、小摘要。旧 Qwen 拒绝/all80 与 TTFT14.06%/decode88.88%分别记录；DDP Graph40.61%/合批55.73%/旧通信拒绝，TP失败，ResNet拒绝/all80，另 memory adapter36.68%接受/缺 images 事实不变。

执行脚本 `scripts/experiments/v6_fsdp_readiness_20261028.py`，输出 `/root/autodl-tmp/dev/v6-work/fsdp-readiness-20261028`。新批冻结后只启用一个普通完成监听器，终态后自动审计；无 Goal、模型轮询、额外投递测试、旧归档控制器或本地大转存。本批没有候选搜索，也不会产生新的正式性能接受。

## 完整源执行与后处理修正

首次冻结 `21809f6c`，03:56:22 UTC 启动、03:56:57 终态。完整 GPU 源执行成功，两 rank 两步均零误差（每步分别比较 361,257,089 / 361,255,041 个元素，因 vocab 分片不等长），完整五样本和六个 profile 都已写出；source step 中位数 176.6510 ms、allocated peak 2,239,245,312 bytes。这只是源基线，不是优化接受。

控制器后处理断言失败：同一个 `record_function` 在 profiler 中有 `user_annotation` 和 `gpu_user_annotation` 两种同名区间，原 parser 错误要求只出现一次。原结果、日志、冻结代码及终态 `completed_with_errors` 全保留，后续真实 intake 尚未执行。启动监听前检查已经发现终态，所以未启动第一个监听、也没有提交额外续行。

最小修正只在实验 parser 明确选择 CPU `user_annotation` 完整窗口；不改源代码、production、GPU执行或样本。新目录 `fsdp-readiness-followup-20261028` 从原目录只读解析原六个 trace、核对全部源文件 SHA 后继续真实原入口阶段，不重新跑源性能。复用的 source tensors 仍原地保存，未复制第二套。当前实际 df 可用约 27.96 GiB，后续参考体积继续核查。

[完整源审计](evidence/v6-fsdp-readiness-20261028/source-audit.json)包含所有五样本、两步质量、source observation SHA 与六个 profile 索引。每 rank 每次有 74 个 NCCL kernel，实际名称为 AllGather_RING_LL / ReduceScatter_Sum_f32_RING_LL；sum 约 167.82–178.01 ms，独立 profiler 完整窗口约 184.62–193.93 ms。kernel 等待/同步可能包含在这些区间中，不能据此宣称等量的可消除通信成本。原始[后处理失败](evidence/v6-fsdp-readiness-20261028/initial-completion.json)保留。

后续启动：冻结 `fe3b5cebd77aa3ccba080d2a79d0eb2b312315d3`，04:00:15 UTC、controller PID514609；[launch](evidence/v6-fsdp-readiness-20261028/followup-launch.json)、[protocol](evidence/v6-fsdp-readiness-20261028/followup-protocol.json)。旧源证据重新解析通过，真实 intake 会话 `ses_e7f55659fffek6273dGICeudTy`、初始快照 `generation-6351535b2857` 已建立；此时 evaluator/discovery 尚待完成，不预判质量或泛化成功。唯一普通监听 PID26684，[启动核验](evidence/v6-fsdp-readiness-20261028/wakeup-setup.json)为 waiting、controller 存活。实际 df 可用 30,006,849,536 bytes（约 27.95 GiB），未转存任何 raw 到本地。

## 原入口终态与独立审计

04:19:01 UTC，followup 两阶段完成，普通监听投递本轮。原始 source GPU 测量没有重跑。新真实 intake/evaluator/discovery 执行完成，**本批没有候选搜索或正式优化接受**。[终态审计](evidence/v6-fsdp-readiness-20261028/terminal-audit.json)保存实际生成输入索引、完整 request/contract、controls、全部五个 clean 样本与每个次级指标三份 profile 的 SHA/重算值；私有模型配置只记录哈希，原始配置留服务器。

Intake 自主选择四个次级字段：communication_kernel_sum_ms、communication_union_ms、compute_overlap_ms、communication_without_compute_ms，全部使用已安装 torch_communication 接口、max rank 后 median。主目标、原始 1.05 显存约束、FP32/noTF32/eager、完整两 rank 工作、共享 embedding/head、两步质量与完成边界均传到 request 和 contract。四个指标分别来自四组独立 profile 调用，每组每 rank 3 次，**不是同一组 trace 的四列**，不能将跨字段端点中位数相减当作同时发生的物理分解。

Evaluator 会话 `ses_e7f517c42ffe479VAZZwcOBCNV`，初始快照 `generation-0f8efd0b671d`，零 contract repair；adapter SHA `788366bf0c3961e1ce15408873bf271c9867ec9eb7092ea9903dc40891012ecd`。实际代码保留每 block/root fully_shard、完整 loss/backward/SGD、reset、两次同步、返回 logits 至 metrics 读取、tied_embedding 精确布尔观察。双 rank reset/identity/negative/coverage controls 通过。

独立检查直接比较此前 source 写出的四个全量 observation 与新 controls 的两 rank 两步 reference。经代码审阅，loss/logits/gradients/parameters 四个名称和嵌套容器完全一致，新增 tied_embedding=True 单独检查；两步各 rank 分别比较 361,257,089 / 361,255,041 元素，全部零误差。对存储观察量添加完整 logits +1 和改 tied_embedding=False，实际 comparator 均拒绝；这是 CPU 观察量负例，不冒充实时 GPU 模型注入。源码本身的真实 GPU +1 负例此前已经通过。

Discovery 每 rank 2,248 个 call site、809 个 group，实际 49 次 all-gather、25 次 reduce-scatter；74 个 collective 全匹配，payload coverage 74/74，logical input 2,733,563,904 bytes/rank。peer access 为 0，实际 NCCL 日志为 SHM/direct/direct；这不是 wire 流量计。自动 planner 选择 nsys、nccl_tuning、execution_timeline、device_telemetry，均 observed；collective_calibration 为 unavailable（当前实现针对 all-reduce，不能作为 FSDP 两类 collective 的校准结果）。planner 对拓扑/字节计/跨 rank 时钟的限制大多说明清楚，但 rationale 把扰动 discovery 中的 NCCL 非重叠区间称为 exposed 时间并推断主要杠杆，证据不足，不能直接采纳该因果结论。尤其 rank0 discovery NCCL sum 564.99 ms、rank1 174.15 ms，与 clean step 177.60 ms 不在同一测量条件。

### 实测的 512-byte 源生命周期差异

完整数值通过，但资源读数并非严格源一致：source allocated/peak 为 1,462,069,760 / 2,239,245,312 bytes，adapter 为 1,462,069,248 / 2,239,244,800 bytes，均少 512。代码审阅定位到 source `main()` 的 GPU `passed` 标量在 quality 后一直被局部变量持有，adapter 的 prepare/state 未保留它。

独立双卡 [A/H/H/A 探针](evidence/v6-fsdp-readiness-20261028/owner-probe.json)已实证：A 使用未改 adapter，H 只在计时外增加并持有源脚本同款 CUDA scalar/all_reduce；每臂三次完整 step、两 rank 的 allocated/peak 差值都严格为 512，H 与源读数相等。每臂末次完整 logits/梯度/参数比较仍零误差。没有改旧分数、减去开销或进行接受。探针首次配置遗漏 `function='worker_entry'`、在 adapter 执行前失败，原[错误记录](evidence/v6-fsdp-readiness-20261028/owner-probe-initial-error.json)保留；只修入口函数后在新目录执行。

这是一处有实证的 evaluator 生命周期遗漏，大小约为源峰值的 0.000023%，不影响数值通过结论，但本任务存在显存约束，继续搜索前应由负责 evaluator 的 Agent 修复。使用既有 `--feedback` 接口，提供当前 adapter、这一处 source owner 及隔离实测证据；不增加生产 prompt、硬编码路由或全局审核门槛。原错误 adapter/参考及全部结果保持冻结，新输出目录执行真实修复、controls、完整 source 比对和原生内存读数检查，再人工审阅实际 ownership 后决定搜索。该后续属于开发者证据辅助的 evaluator 修复，不再称为完全无辅助的新原入口 cohort。

审计时实际 df 可用 23,854,616,576 bytes（约 22.22 GiB），GPU 无进程。下一批预计新增约 5.4 GiB controls 参考，无其他大型并行实验；raw 留服务器。详见[修复批次](v6-fsdp-owner-repair-20261028.md)。
