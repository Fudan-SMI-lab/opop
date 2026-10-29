# Qwen新CPU接口证据的一轮普通续行

后续固定执行已完成：[7e78原字节B/C](v6-qwen-cache-interface-fixed-20261030.md)完整质量通过但C TTFT26.076953超原24.45004275，保留10样本/4检查、无正式接受/指针发布。原6ec4的四个接口异常结果不覆盖，不能把新候选通过追认为旧实现有效，也不能据单次序列声称CPU事实段落稳定因果优越。

后续2026-10-29 22:19 UTC：[单次原失败反馈候选生成](v6-qwen-cache-interface-candidate-20261030.md)已完成审计，492身份/33输入保持，47assistant/57工具/0 schema修复。角色实际读到原始AttributeError及target签名，产出7e78真实DynamicCache实现；只做CPU模拟，仍有verify异常flag、第一层ownership及标签限制。原6ec4四worker失败与本页拒绝保持，不追认为GPU修复成功。下一步[原字节固定B/C](v6-qwen-cache-interface-fixed-20261030.md)，无手改/搜索/正式接受/生产采用。

## 终态审计：执行失败，未产生质量或性能结论

2026-10-29 21:42:39 UTC（香港10月30日05:42）确为 `batch_finished`，不是磁盘告警。控制器完成一轮；trial `2a7d20b4d9` 的唯一空参数候选 `6ec4e97a…` 被记录为 `no_valid_parameter_candidate`。四个worker——screen、candidate_diagnostics、Nsight、timeline——全部在父模型调用 `past_key_values.get_seq_length()` 时抛出 `AttributeError: '_CacheHandle' object has no attribute 'get_seq_length'`。没有返回完整质量检查、clean样本、约束或配对噪声判断；不能把此错误称作数值质量拒绝。独立验证未运行，current仍是空bundle的 `2a42b9b1…`。

[公开审计](evidence/v6-qwen-setup-cpu-feedback-20261030/terminal-audit.json)对应服务器完整审计SHA `04ee9d9d9d2c3d464891978de5da2b5db3336f52283a43816e814c9a687d2621`。422冻结文件重新校验；四份实际请求的materialized source、adapter、bundle、entry、core及GPU UUID分别核验。原始request/discovery/prepared、controls、旧拒绝和接受版本未变。生产core仍为 `ed24af7a…`，当前设备TTFT限制仍为 `24.45004275ms`，没有重锚定。

### 实际反馈消费和生成记录

三个真实角色均0 schema repairs。selector为30初始文件/94,796,452B、35条assistant消息/43工具；candidate为33文件/137,079,958B、49条/61工具；planner为8文件/16,272B、14条/21工具。provider估算合计4.20129993434，仅观察，不是账单。初始文件逐字匹配快照，selector/candidate反馈重放实际registry/plan路由一致，历史4693代码及完整打包产物保持。文件大小不表示全部进入模型上下文。

selector工具18只打印新CPU事实的嵌套字段名、范围和索引，深度限制使关键嵌套数值未展开；candidate工具6完整打印CPU结果，工具30再次明确读取singleton stride和两次CPU cache增长。两者均未另行打开新index/fixture/result文件，候选通过内联反馈读到了事实，不能再沿用上一轮“未读取新事实”的结论。仍不构成稳定因果优势或无辅助闭环；这是明确的开发者CPU事实辅助。完整命令、输出、初始生成和六次编辑均保存服务器，重建后与最终31,327B候选SHA `6ec4e97a4bd39a7d16c82109b49b841f0258c039d56de15eeb6f80696e4906fa` 完全一致。完整记录未见后续成功实现/解释核验答案读取；外部文件系统并非隔离快照。

候选工具错误保留：sidecar层级KeyError、错误installed路径且管道exit0、CPU cache被CUDA guard拒绝、4×1误作1×1、尝试赋值只读Tensor.device。最后CPU测试手动重写fast-tail，而非经过真实replacement/框架dispatch；`assert … or True` 恒真，不能证明engagement gate。语法/import/schema通过不补足这些缺口。selector的原始质量/关键路径/TTFT保证和planner的采集预期也不作为结论。

### 返回对象与实际派发边界

新代码改为opaque `_CacheHandle(runner, step)`，没有父模型Cache接口；只在首decode吸收真实prefill DynamicCache，后续希望以handle推进。它移除了候选内部stride限制及旧 `not_attempted` 成功分支问题，但框架签名仍要求 `DynamicCache` 类型和 `[1,1]` stride。直接调用replacement的greedy/teacher自检绕过了这层签名匹配，无法证明真实两次run/observe可执行。另有代码层面的潜在问题：verify在 `fast_enabled=True` 后抛异常时catch没有清除此标志；未将其归因为本次已观察错误。

独立[一次CPU接口控制](evidence/v6-qwen-setup-cpu-feedback-20261030/cpu-binding-interface.json)使用原签名仅将device改为cpu、精确生成的handle类、真实BindingSession的diagnostic_passthrough和只调用get_seq_length的合成父方法。DynamicCache匹配；handle不匹配并在父方法复现同一异常；DynamicCache配teacher `[32,1]` stride也不匹配。1次匹配/2次fallback、恢复通过、CUDA未初始化。没有运行候选build/replacement、模型或Graph；它支持接口/派发解释，不是历史GPU逐调用cache轨迹或数值证明。没有给候选手工修复代码。

candidate_diagnostics只保存一次普通generator False聚合252次，未记录candidate-frame异常；真正失败位于父模型/框架，故空异常列表不是成功。Nsight和timeline均在采集前的invoke失败，前者日志明确“No reports were generated”，没有SQLite/raw timeline，不能沿用旧代码native计数。候选setup内部实际执行与完整32-token快路径质量仍分开；不从标签推断native。

审计读取曾猜错planner文件名（实际collection.json），首审计又假设failed binding含occupancy而KeyError；错误记录保留，修正只读审计后通过，没有模型/CPU夹具重跑或候选改动。CPU接口控制只运行一次。

### 后续边界

两卡空闲，150GiB实际df余约8.37GiB；本轮没有清理或raw本地搬运。完成监听40560标记已处理，不重启。下一步仅[一次候选角色生成](v6-qwen-cache-interface-candidate-20261030.md)：沿用此次真实选择，交付此次原始framework失败反馈、精确历史6ec4代码和原始失败collector产物；不把本次开发者CPU接口结果/因果表注入，不改prompt，不启动GPU或搜索。输出先审查再决定是否执行，避免靠重复GPU试验修复已明确的接口错误。所有历史拒绝、辅助边界和关闭研究保持。

以下为原启动记录。

**已启动：2026-10-29 21:04:45 UTC（香港10月30日05:04），冻结426f21982299c6a8130b0aee4a8f1139ecee66bb，controller43572。** [启动记录](evidence/v6-qwen-setup-cpu-feedback-20261030/launch.json)、[422身份预检](evidence/v6-qwen-setup-cpu-feedback-20261030/frozen-preflight.json)、[真实本地服务健康](evidence/v6-qwen-setup-cpu-feedback-20261030/session-health.json)通过。首次部署快进因已上传audit文件CRLF与提交LF字节不同而中止；核验归一化内容相同、部署精确git blob后才启动，[错误记录](evidence/v6-qwen-setup-cpu-feedback-20261030/launch-preflight-attempt-1.json)保留。错误发生在controller/session/model/GPU/监听创建之前，不是重启实验或改生产。

21:05:39 UTC[实际启动快照](evidence/v6-qwen-setup-cpu-feedback-20261030/startup-verification.json)：selector generation52eb49771097/session ses_e7684d8d0ffeOcUPWcQBSlHLes，30初始文件/94,796,452B逐字匹配；反馈5,894,790B/SHA30459adc210d4848abd1b0c7754ab9ea3bf70b19c2b37266d0f31433641f8462与真实registry route相等，新CPU事实送达、历史4693和原打包产物保持。与上轮selector提示/模型/工具相同，8输入改变、1旧Python观察输入移除，没有新增输入文件，非整个角色单变量A/B。422冻结身份及共享discovery/prepared/request核验；当时7 assistant/14工具，尚无新GPU jobs。早期读取原request/source/adapter/contract/search等，不能由实际送达推断事实已被阅读或正确使用；后续完整审查仍必要。

唯一[完成/8GiB监听](evidence/v6-qwen-setup-cpu-feedback-20261030/wakeup-setup.json)40560及[存活核验](evidence/v6-qwen-setup-cpu-feedback-20261030/wakeup-verification.json)运行，旧7324已处理。启动核验两卡空闲，实际df余9,260,822,528B（约8.63GiB）。无新质量/接受结论，无本轮清理。

在[上一轮终态审计](v6-qwen-setup-feedback-20261030.md)后声明一次新普通trial，根为`qwen-setup-cpu-feedback-20261030`。它不是重启旧试验、同快照A/B或无辅助原入口证明。依据是对4693原码新增的一次CPU接口控制，而非重复旧优化直到通过。没有手写候选/实现修复、人工选组、新提示或接受规则变更；关闭的提示/解释/核验研究保持关闭。

起点复用上一轮空parent2a42b9b1、prepared009b、当前设备已完成discovery/controls，既有fork_run及CLI resume rounds1使用默认作用域/参数探索。原latest-feedback、旧4693候选和两个原run-only sidecar经现有历史产物机制进入；只在现有supplemental_evidence加入紧凑CPU结果、范围界限和原fixture/结果/sidecar索引。保留所有原反馈字段与字节前缀，先核对真实registry/plan路由，再核对实际新快照。开发者辅助事实边界明确，不声称这次端到端自动发现了CPU问题。

新CPU结果仅说明：singleton切片contiguous保持stride[32,1]，原replacement guard送父路径，CPU同cache示例258→259→260；另一个明确stub setup/verify成功的控制最终仍disabled/not_attempted。前者不是历史GPU逐调用cache轨迹，后者是潜在成功分支缺陷而非本次先发生的E断言原因。fixture无CUDA初始化/模型/Graph执行，既不是完整数值质量，也不给修复代码。原4693 E失败、父回退质量通过但TTFT26.232951超限的拒绝及所有样本保持。

工作量、原始完整质量、两同步/状态生命周期、BF16/SDPA/noTF32、B1/prompt256/full32 greedy、两连续检查/两warmup/五clean不改。当前20e8 GPU原TTFT23.285755、上限24.45004275，禁止旧设备anchor或重新锚定。源映射/此前辅助边界保留，原current及旧接受版本冻结。没有本批独立完整版本/专门native阶段；若新接受，先审核再声明后续独立验证。生成、run-only valid、父回退质量、搜索接受和独立接受分别报告。

API仅观察费用，不加上限。角色实际工具可探测，终态需审核全部命令/输出和初始生成/编辑；外部FS非隔离。没有大raw本地转存或本轮清理；启动时两卡空闲、df约8.83GiB。新长工作只设一个完成/8GiB磁盘监听，低磁盘不是完成，不停止实验或盲删。生产core仍ed24af7a…；raw/private配置/完整会话仅服务器，GitHub仅代码/哈希/小摘要。
