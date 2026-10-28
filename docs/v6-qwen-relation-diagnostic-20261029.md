# Qwen完整输出关系检查的固定诊断

问题来自[原入口搜索三轮失败](v6-qwen-original-end-to-end-20261029.md)：前两轮在teacher-forced replay关系检查处抛错，未保存greedy与teacher各自对独立源的差异。现有数据不能判断哪一侧先偏离，也不支持唯一alias/shared-pool原因；candidate_diagnostics执行成功没有进行质量认证。

新目录`/root/autodl-tmp/dev/v6-work/qwen-relation-diagnostic-20261029`，控制器`scripts/experiments/v6_qwen_relation_diagnostic_20261029.py`。固定顺序为 **B-observed、C1-unobserved、C1-observed、C2-unobserved、C2-observed**，每项仅一次。C1/C2分别是原`d5c20fcae5`和`e74f5ba6cf`的确切代码/参数；B为原基线。U使用原执行entry与evaluate；O使用外部诊断entry包装生成adapter的run/_infer/observe，观察点位于完整greedy或teacher输出之后，并在observe退出后再次检查greedy输出。只记录该具体失败所缺的完整输出，不加大trace、before_run观察器、GC控制或CPU计数读取。

所有嵌套request、候选、adapter、工作量、数学政策、controls、质量阈值、warmup/repeats/连续步骤不改。观测层保存完整CPU tokens/logits、哈希和小摘要，比较各输出与独立源、teacher与greedy、greedy在observe前后是否改变；额外按run/observe阶段计数已有candidate route回调，不改变原tracking状态。不用单个route标签证明实际原生kernel，也不从没有标签推断没有回退。

CPU复制会增加host工作、同步及CPU对象生命周期，可能改变复现条件；这就是保留U对照的原因。O的内部计时或framework.elapsed不用于性能接受，不减去观察开销。若某次重放通过，也保留原失败并明确观察/重放差异，不重试到通过。全部原始值留服务器，无本地raw下载。一次失败可能使该端点没有clean样本，必须如实记录。

B-observed为实际CUDA观测器控制；随后C1/C2两种路径保留原异常、traceback、所有结果。observed finally写旁文件，controller“completed”只表示诊断尝试结束，worker失败及完整质量另报。该批没有Agent生成、没有自主搜索、没有formal acceptance，也不自动采纳任何生产改动。下一项真实Agent实验必须等此次证据明确所缺信息，再做最小调整，不能把尚未隔离的原因写进提示。

源码与参数身份在执行前后核对。先前全量源参考和manual映射保持，当前只有约13.17GiB可用；预计新增完整CPU输出百余MB、不产生大型profile。GPU工作串行，单个普通完成/8GiB监听；无Goal、模型等待轮询、本地大转存或旧controller重启。若低空间先查df/活跃worker，只有GPU结束后才做授权无损冷压缩，保存全部原始内容。

[接口预检查](evidence/v6-qwen-original-followup-20261029/relation-preflight.json)通过导入/语法/CLI和三个冻结screen执行请求schema。实际CUDA观测行为由首个B端点验证，不以CPU接口检查冒充实验结果。

已冻结`ba50554ffe266771e4ba9d13360224f507bc09cd`，2026-10-28 20:09:18 UTC启动，控制器PID857099、PPID1、独立session，生产core仍ed24af7a。见[启动](evidence/v6-qwen-relation-diagnostic-20261029/launch.json)、[冻结协议](evidence/v6-qwen-relation-diagnostic-20261029/protocol.json)。

20:10:00 UTC核验：B-observed通过两个连续全量质量检查；其6份完整观察均匹配独立源，teacher/greedy与greedy前后比较零误差。C1-unobserved已按固定协议执行并失败，C1-observed的真实GPU worker正在运行，后续C2尚未完成。这里只确认观测器baseline控制和实际进度，不提前认定候选的原因或任何优化收益。见[真实运行记录](evidence/v6-qwen-relation-diagnostic-20261029/startup-verification.json)。

唯一普通完成/8GiB监听PID1180已核验waiting，状态`.scratch/v6-qwen-relation-wakeup.json`、同前缀`-wakeup-launch.json`；旧搜索监听已投递并标记handled/stop，不重启。见[设置](evidence/v6-qwen-relation-diagnostic-20261029/wakeup-setup.json)与[核验](evidence/v6-qwen-relation-diagnostic-20261029/wakeup-verification.json)。实际df剩14,081,044,480字节（13.11GiB）。本机在线时终态自动唤醒，本批无模型等待轮询或新Agent生成。


## 终态核查：2026-10-28 20:10 UTC（香港10月29日）

五次固定尝试均结束，但只有观察baseline通过，四次候选worker均保留原AssertionError；controller的completed表示尝试完成。没有新Agent生成、搜索或正式接受。[完整小型审计](evidence/v6-qwen-relation-diagnostic-20261029/terminal-audit.json)逐项核对请求、源码、entry、controls/reference SHA及12份完整CPU张量，原始张量仍留服务器。B保留5个clean样本、2次全量检查及6份零误差观察；四个失败候选均在quality阶段停止，没有clean样本。

| 固定端点 | 实际worker | greedy对独立源 | teacher对独立源 | observe前后greedy |
| --- | --- | --- | --- | --- |
| B-observed | 通过 | 两步零误差 | 两步零误差 | 逐元素相同 |
| C1-unobserved | 原关系断言失败 | 未额外保存 | 未额外保存 | 未额外观察 |
| C1-observed | 原关系断言失败 | max abs 1.0 | 零误差 | 逐元素相同 |
| C2-unobserved | 原关系断言失败 | 未额外保存 | 未额外保存 | 未额外观察 |
| C2-observed | 原关系断言失败 | max abs 1.0 | 零误差 | 逐元素相同 |

C1/C2的完整greedy tokens/logits也逐元素相同；两者32 tokens全部正确且logits有限。prefill位置0零误差，首个decode位置1已出现0.1875最大误差，后续位置都有误差。故“greedy capture正确、只是后续teacher/replay出错”的前提不成立。输出在observe前后不变，仅排除该记录区间内的值变化，不能排除生成期间的问题；立即clone的C2与C1相同，也不构成排除全部alias机制的证明。未唯一定位shared pool、cache、alias或算术原因。

原GPU断言为teacher(actual)对greedy(expected)：1,132,088/4,861,952不匹配，max abs1.0；保存后CPU同方向比较为1,130,457。额外只把这份保存张量送到CUDA做原assert_close（没有重新运行模型），精确复现1,132,088及原最大误差位置。CPU greedy对source则为1,131,606，预期值方向不同。设备上的BF16比较与相对阈值方向必须分开，不能把计数差解释成观察器改变输出。所有比较仍失败，阈值保持0.02/0.02。

U请求保留原evaluate entry；O仅更换外部observer entry/function、添加诊断源路径和输出目录；嵌套request、参数、adapter、原质量/工作量及controls完全一致。observer SHA为`a06b6c0e14495bffb68aa498c845ce66378223e9c95579b0b66a7bd463cc566c`，原entry为`aed604ee3c454ba7ef3bec09d74c48e1f92ab0ec873c3f74d2c2e2f76c6bd0af`。两候选观察run各记录31个graph_capture回调；这不替代原生kernel证据，teacher缺少该回调不证明没有原生执行或回退。O的CPU复制改变同步、host工作和生命期；不使用它的计时作接受，不扣除观察开销。

服务器GPU已空闲，实际150GiB配额剩约13.06GiB。没有本地raw下载或清理；旧源命名映射的人工边界、原入口三轮零接受、旧同adapter46.9421%及其余历史结果全部保持。

下一步是[已有反馈单文件事实补充对照](v6-qwen-relation-feedback-replay-20261029.md)：从原第三候选初始状态各启动一次真实角色，验证是否读取和正确使用这项缺失观察，不提供优化实现答案，不自动改生产prompt。
