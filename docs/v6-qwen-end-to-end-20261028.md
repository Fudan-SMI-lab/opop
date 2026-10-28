# Qwen完整生成目标与TTFT约束的自主搜索

## 终态审计（2026-10-28 17:25 UTC，香港时间10月29日）

批次已正常结束，prepare/source/search/independent/native均执行完成。完整版本独立AB/BA通过：**705.5819415→374.366936 ms，降低46.9421%**。这是一项同adapter目标变体的结果，不是重新从原入口生成evaluator的结果。审计代码为`scripts/experiments/v6_audit_qwen_end_to_end_20261029.py`；[逐端点与角色证据](evidence/v6-qwen-end-to-end-20261028/terminal-audit.json)、[终态](evidence/v6-qwen-end-to-end-20261028/followup-completed.json)、[自动投递回执](evidence/v6-qwen-end-to-end-20261028/wakeup-delivery.json)已保存。下文启动状态为历史记录。

原baseline五个文件与adapter24ebefd7保持相同；新controls通过reset、identity、错误输出和缺失覆盖负例。独立源检查两次均比较4,861,984个元素，full logits与exact tokens零误差。额外adapter观察值单独报告，未冒充独立源验证。审计包含16个clean端点、80个样本、30次evaluate全量检查，以及controls自身检查；没有删样本。实际job物化源码、adapter/bundle、request、core及两份Nsight SQLite身份均核验。

新controls首token基准24.285179ms，因此固定上限为**25.49943795ms**，不是跟随优化版本变化的锚点。真实variant_contract把完整生成设为主目标并保留TTFT约束，source workload、状态所有权、两同步边界和数学政策不变。继承completion文字仍称首token边界为“ttft_ms objective”；这段文字未自动重写，但新invariants、选择理由和实际计分使用正确目标。未据此推断执行语义错误，也未修改生产API。

| 轮次 | 实际实现与结果 | 接受判定 |
|---|---|---|
| `7cbb441641` | 单decode组，静态KV与逐位置Graph；完整质量通过，paired全生成701.623035→378.897129ms | TTFT一端25.856583ms超固定上限，另一端24.950436ms；拒绝 |
| `f71b5c1677` | 读取上一轮超限反馈，一个joint bundle覆盖prefill和decode两组，共享静态KV/Graph；paired708.006893→374.194465ms | 两端TTFT21.357271/21.411066ms，全量质量通过；接受 |
| `7c85e77d6f` | 尝试融合norm/RoPE/SiLU/GQA；筛选实际走parent prefill和普通算子decode Graph，融合未生效；完整质量零误差 | 筛选TTFT27.404051ms超限，无valid参数，不进入paired；保留第二轮 |

每轮只有一个实际参数点`{}`，不能把默认parameter_trials=3写成已执行三个参数。第三轮`fallbacks`空但有两次parent reference，不能以空fallback字典推断没有回退。内部prefill setup/融合禁用的唯一原因未被记录，不能把此轮说成数值失败或融合成功。

接受代码经过人工源审查：完整36层prefill与逐token decode仍调用原权重与全量算子，按实际输入复制/执行，输出clone；静态cache和DynamicCache视图衔接，signature/cache不符时有parent路径。未见以参考答案替代计算。当前质量阶段路由为2次prefill graph和62次decode graph，无parent/fallback。12个真实角色均有新会话/初始快照、core一致；记录了实际工具读取与反馈消费。针对旧对照/旧decode/旧TTFT trial路径的工具输入检查未发现引用，检查范围不等于对整个文件系统的隔离证明。无schema修复。第二轮接受后真实rediscovery发生，第三轮提出替换；最终一个joint bundle，**尚不证明多个active bundle共存**。

独立AB/BA两个baseline端点731.601546/679.562337ms，candidate端点374.347311/374.386561ms；保留baseline漂移。配对收益分别357.254235/305.175776ms，超过各自8.797894/2.446156ms噪声判据。TTFT21.324534/21.355557ms均满足原上限，完整质量通过。独立候选评测不发布另一current指针，保留自主接受版本。

单独Nsight实测baseline66,441 kernels、0 Graph launch；candidate66,089 kernels，其中66,056 graph nodes，**32次cudaGraphLaunch**。这支持完整prefill与31次decode的实际Graph路径，不仅是标签自报；profile为另一次受扰执行，不进入clean得分。候选对host overhead/bandwidth的解释仍是机制假设，不是已分解的因果成本。

第一次只读审计错误假设controls也有evaluate的`math_policy_unchanged`字段，已保留[审计错误记录](evidence/v6-qwen-end-to-end-20261028/audit-attempt1.json)。修正controls schema读取后只读复核通过，没有重跑GPU或改变结果。

## 服务器空间与下一步

GPU结束后，于17:36:26–17:38:08 UTC执行授权的服务器内无损压缩，仅选择三批已完成Qwen worker的`run`目录。55份7,854,774,040字节JSON转为545,946,786字节gzip，逐个验证压缩/解压SHA，625个其他文件SHA不变；实际df净释放**7,308,292,096字节（6.81GiB）**，余17,496,178,688字节（16.29GiB）。9个当前版本的源制品/bundle与六份完整FSDP参考后验核查通过。没有本地raw转存，controls/reference/source/store/model/private配置均不在压缩范围。

见[范围清单](evidence/v6-qwen-end-to-end-20261028/storage-plan.json)、[校验](evidence/v6-qwen-end-to-end-20261028/storage-postcheck.json)、[恢复说明](evidence/v6-qwen-end-to-end-20261028/storage-RESTORE.md)。原run JSON缺失时应查`/root/autodl-tmp/dev/storage-audit-20261029/qwen-original-entry/run/cold-json-journal.jsonl`，不是丢失证据。其他早期压缩journal分别保持。

下一步按[原入口完整生成协议](v6-qwen-original-end-to-end-20261029.md)验证入口/evaluator泛化，不复用adapter或优化bundle。生产core/prompt/sampler/allocator/接受规则不改，成功反馈提示研究保持关闭。历史TTFT14.06%、decode88.8847%、旧固定Qwen拒绝/80样本及其他任务接受/拒绝分别保留，不累计收益或覆盖。

承接方案E及[同核心目标对照](v6-qwen-goal-contrast-20261028.md)。新任务检验框架能否围绕整个生成流程优化，并同时遵守另一阶段的性能约束。不是重测旧候选直到通过，不要求Agent必须选择两个组或使用特定组合。

目录`/root/autodl-tmp/dev/v6-work/qwen-end-to-end-20261028`，控制器`scripts/experiments/v6_qwen_end_to_end_20261028.py`。主目标为完整32-token的`end_to_end_ms`最小化，新增任务约束`ttft_ms <= 1.05 × 本任务新controls的原baseline TTFT`。这是新实验目标/约束，不追溯添加到旧结果，比例不是框架硬编码规则。原batch1/prompt256/full32tokens、BF16/SDPA/noTF32、full logits 0.02/0.02、exact tokens、两同步边界、状态持有和完整工作保持。

通过既有`--reuse-evaluator`只复用原baseline五个源文件及adapter24ebefd7；不复用优化bundle、旧分数锚点或人工诊断。真实variant_contract只适配invariants，框架更新goal/objective，其他字段继承。继承completion文字仍把首token边界称为“ttft_ms objective”，不得声称已自动改写；须结合新goal/objective/invariants核对TTFT实际作为约束、完整生成为主目标。保存初始快照、执行新controls，再用已有独立源参考对新controls做两次全量logits/exact tokens比较；核查结果不传给搜索Agent。

准备和源检查通过后，CLI `--resume --rounds 3`进行发现、采集计划、目标选择、候选生成/回装、参数筛选和反馈迭代。不给手选组、旧T/D答案或交叉测量结论。每个真实会话/API只统计，不设置新费用或请求限制；三轮不是API额度。阶段失败保留并停止依赖阶段，不循环到通过。

存在搜索接受版本时，在新合同内从新baseline做完整独立AB/BA，TTFT约束始终锚定本任务原controls。仅在独立完整质量通过且存在新bundle时采集单独B/C Nsight；侵入式profile不进入clean分数。搜索接受、源一致性、独立质量/性能、实际路径和执行完成分别报告。

这是同adapter目标变体，不是原入口重新生成evaluator的自主cohort。不承诺多个active bundle，只有实际重发现/替换/共存记录能证明。生产core、prompt、sampler、allocator和接受规则保持，成功反馈提示研究已结束。

[预检查](evidence/v6-qwen-goal-contrast-20261028/end-to-end-preflight.json)通过schema、语法与原baseline约束接口；正式任务重新测量anchor，不使用该CPU接口检查里的旧值。服务器内无损整理后实际余16.40GiB，raw留服务器，单GPU串行；一个普通完成/8GiB告警监听，无Goal、模型轮询、本地大转存或额外投递测试。所有历史接受/拒绝及样本保持。

已冻结`387ae6bc2b110676e4dfb1bcfda4d5068e9a79f8`，15:02:55 UTC启动远端controller PID707981、PPID1、独立session；生产core仍ed24af7a。OpenCode自动启动服务健康检查通过，server_url=null。15:03:56实际处于prepare的variant_contract真实会话`ses_e7cf6d14fffeojk3mtdVSt1xz8`，初始快照generation-bf6faced27a2，已观察到6条assistant消息；GPU空闲符合尚在合同生成的阶段，不表示卡住。尚未宣称controls、源检查或优化通过。见[启动](evidence/v6-qwen-end-to-end-20261028/launch.json)、[协议](evidence/v6-qwen-end-to-end-20261028/protocol.json)、[真实活动](evidence/v6-qwen-end-to-end-20261028/startup-verification.json)。

唯一普通完成/8GiB磁盘监听PID54204已核验waiting，状态`.scratch/v6-qwen-end-to-end-wakeup.json`，启动参数`.scratch/v6-qwen-end-to-end-wakeup-launch.json`；每30秒只用SSH/df查询，终态或低空间时才投递一次。[设置回执](evidence/v6-qwen-end-to-end-20261028/wakeup-setup.json)保存。上一批Qwen对照监听已完成投递，标记handled_in_user_turn并写stop文件，不恢复。启动后实际余17,600,995,328 bytes（16.39GiB）；本机重启不影响远端任务，但需要恢复此批监听。
