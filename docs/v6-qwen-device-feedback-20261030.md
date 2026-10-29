# 当前设备原基线上的一次普通反馈试验

后续证据状态：独立[setup诊断作者](v6-qwen-setup-author-20261030.md)已完成，原样生成脚本的128记录上限会丢失后续失败，CPU接口负例已确认；因此未执行GPU观察，仅安排[一次窄范围真实修复](v6-qwen-setup-diagnostic-repair-20261030.md)。这是观察器证据保留缺陷，不改变本页04df父路径质量通过/当前TTFT拒绝，也不提供模型数值原因或手工候选修复。

**终态已审计：2026-10-29 18:31:36 UTC（香港10月30日02:31）完成，未接受新版本。** [终态](evidence/v6-qwen-device-feedback-20261030/followup-completed.json)、[审计小摘要](evidence/v6-qwen-device-feedback-20261030/public-terminal-audit.json)、[原样候选](evidence/v6-qwen-device-feedback-20261030/candidate.py)与[proposal](evidence/v6-qwen-device-feedback-20261030/proposal.json)保留；服务器terminal-audit SHA为`bf1d6eb32eb5161cb43211418fbc99c768b62add39230186e0ffbc26b84b1e1c`。299冻结文件、11 worker的源文件/adapter/entry/core/设备及当前空bundle均核验。事件是真正完成，不是空间警告；两卡空闲，df余10,307,497,984B（约9.60GiB），本轮无清理。旧监听44008已投递退出并标记处理，不重新启动。

## 执行、质量与接受边界

一次trial `0db61e9938`生成35,199B候选`04dfc6c2…`，empty parameter_space只实际筛选一个`{}`（默认budget3不代表三次执行），revision `f297bf39…`。联合选择prefill/decode两组、两个call，requires/replaces/eliminated_calls均空；新原baseline没有旧f065 bundle。screen worker正常返回valid=true，两次全量检查各4,861,988元素零误差，但质量阶段实际记录62次`parent_fallback_setup_failure`、2次prefill parent passthrough及对应reference调用。框架fallbacks为空不能抹掉内部回退，**通过的是父路径质量，不是新快路径质量**。

原样五个end_to_end样本为736.774652、729.372678、736.591518、752.020505、770.869476ms；TTFT为24.431067、25.307677、24.679412、24.638489、36.634282ms，长尾完整保留。中位full736.774652ms、TTFT24.679412ms；后者超过当前原anchor23.285755×1.05＝24.45004275，参数结果无效，最终`no_valid_parameter_candidate`。未进入配对噪声接受，未产生活动bundle、接受后rediscovery或独立整版验证；current仍`2a42b9b1`空bundle。不能用旧25.2536046上限改变决定，也不能从本期父路径时序推断旧设备增量。

setup8535.081059ms在warm-model分数外。生命周期peak8,419,632,128B、runtime peak8,271,469,056B；相对新原baseline分别增加231,608,320B、103,039,488B，不是显存收益。外部源比较没有重跑，沿用已核验的新设备controls和显式人工字段映射；它不认证任意新增owner或并发组合。

## 实际反馈、代码与探针

四个新真实角色依次初始planner/selector/candidate/后置planner：6/28/31/8份初始文件、9/31/49/9 assistant消息、11/37/61/12工具，框架schema修复均0，provider估算总4.08980644584（非账单）。初始快照、当前registry/plan参与的完整CPU route复放匹配；selector与candidate都收到原样f5b历史代码及supplemental/full candidate_diagnostics。selector实际tool14读原始sidecar，candidate在错误dev-v6-work路径后tool16读打包诊断、tool20确认installed_bundles=[]。初始planner没有previous-feedback输入；不追认启动时尚未发生的读取。完整记录的命令/输出/CPU脚本中未观察到后续成功实现或reviewer答案读取，外部FS仍非隔离环境。

新实现保留parent linear/SDPA，fused模式有3个Triton核，exact模式手工重组原模块运算；31个长度相关decode Graph、master KV、返回新cache/layer镜像，prefill始终parent。不等于kernel-for-kernel或一般bitwise保证。代码`fast_enabled`初始化false，直到setup最后才置true，因此A/B/C/D等经replacement调用的自检实际先走parent；“自检覆盖快路径”声明不成立。Chain E先调用parent decode，再把同一个cache传给下一次调用；尚未记录两次调用的实际cache状态，不能把其失败唯一归为算术、别名或Graph pool。

两个独立candidate_diagnostics均记录同样两次内部断言：`fused_eager_vs_parent_step_1`和`exact_chain_E_absorb`，每次在check/setup/build三个frame传播，不能算六次独立失败。504个普通False返回不是错误。内部FP32阈值.012/.012不同于原任务.02/.02；sidecar没有原始比较logits、最大差或cache状态，因此不能据此证明外层质量超差。collector只run，不observe/compare；有效返回不是额外质量认证。

Agent自测错误全部保留：selector本地schema校验先漏affected_calls后自行补齐，未触发框架repair。candidate四层CPU cache mock出现缺方法、层长度不一致、torch.Size错误比较、切片预期错误、混合storage检查、zip解包和mock parent不接受位置参数等失败；最后完整guard probe仍exit1，仅前半absorb/guard检查输出成功，最终通过的是syntax/import/schema smoke。其后还有代码编辑，不能称最终全套CPU测试或真实Triton/CUDA数值测试通过。原写入、历次edit、全部工具与private session留服务器；仅清理过自身__pycache__。未采纳“唯一手写attention原因”“同op必bitwise”“prefill不变保证TTFT”等主张。

## 发现和诊断证据

当前原baseline fresh discovery实际26,910 calls、17,852 groups、66,441 CUDA kernels，raw trace SHA/数量重新核对；不是旧f065 discovery复用。11个GPU jobs全部正常结束：discovery、两次Python profile、screen、两次candidate_diagnostics、timeline、四个binding-cost端点。screen加binding端点共25clean样本/10完整零误差检查；后四端点是诊断，不能当候选配对接受。

当前候选timeline raw重算66,441 kernels、active union386.450276ms、kernel span1356.825792ms、gap970.375517ms、4个CUDA同步API，与保存报告完全相符；profiled wall1357.522011ms。两份Python原pstats的calls/self/cumulative逐项复核。它们是独立受扰采集，不能用clean wall减profile CUDA作为可回收host成本或关键路径。无当前快路径Nsight证据，旧f065/d5bd Graph数不移植。binding两组parent/passthrough差分别+53.335404和−40.588801ms，符号翻转不代表负物理开销或独立组归因。

审计自身第一次CPU路由复放漏传registry/plan，仅routing字段断言失败；保留audit-attempt-1，按实际调用补齐后两角色完整反馈一致。随后只增加discovery raw校验，原c4a06762审计另存，不重跑Agent/GPU或改生产。

下一步是[内部setup失败的单次诊断作者](v6-qwen-setup-author-20261030.md)：原请求/代码/原始sidecar，不给上述代码推断、修复答案或其他候选；只生成有界观察器，先审查原样输出，再决定单独执行。原优化失败保持，不手工修候选、放宽阈值或重复性能到通过。生产prompt/质量/接受/allocator不变，已关闭研究不重开。以下保留历史启动协议。

**已于2026-10-29 17:49:17 UTC（香港10月30日01:49）启动**，冻结21aafdb7/controller10522。[启动回执](evidence/v6-qwen-device-feedback-20261030/launch.json)、[真实本地服务健康](evidence/v6-qwen-device-feedback-20261030/session-health.json)、[299身份/实际UUID/CPU反馈路由预检](evidence/v6-qwen-device-feedback-20261030/frozen-preflight.json)已保存。生产核心仍ed24af7a，无本轮清理。

17:53的[实际启动核验](evidence/v6-qwen-device-feedback-20261030/startup-verification.json)确认新设备原baseline fresh discovery worker10662及Python runtime worker11472均通过，均实际绑定GPU-20e8003c、entry aed604ee/core相同；discovery后current2a42b9b1仍空bundle。共享的是本次新controls路径，299冻结文件保持。初始真实collection-planner generation-bce225ade347/session ses_e7736e9faffezOSCwKwO7Cvjyv，6输入/当时11工具；该角色没有previous-feedback输入，**此时尚不能声称后续selector/candidate收到或读过事实**。discovery的60次进程占用快照有1次不可用，不能据无外部client记录排除全部干扰；诊断不是干净时序。

唯一普通完成/8GiB监听 **44008** 已实际运行，[监听启动](evidence/v6-qwen-device-feedback-20261030/wakeup-setup.json)和[等待状态](evidence/v6-qwen-device-feedback-20261030/wakeup-verification.json)留档；旧23004已处理，两项短批未创建监听。核验时controller运行、Agent/collector阶段GPU暂空，磁盘约10.2GiB。后续由完成/磁盘事件接续，无模型轮询或额外投递测试。以下为本批预先声明范围。

这是[新设备固定B/C及setup诊断](v6-qwen-device-refresh-fixed-20261030.md)之后单独声明的一次普通试验，不是旧两轮恢复的第三轮或重新启动旧批。使用现有fork_run/CLI resume，从当前设备新prepared原baseline009b、空活动bundle开始，执行一次fresh discovery和一轮真实selector/candidate/default参数探索。无新intake/evaluator/源模型运行、controls重测或参考复制；已完成的新controls与独立源检查只读复用。生产core/prompt/sampler/allocator/接受规则不变，API observe-only，无新增上限。

历史实现仅为确切f5b76ebd/原proposal/selection及固定B/C原结果，未安装该候选或旧f065。开发者创建的diagnostic-feedback.accepted=false明确是“固定诊断且快路径关闭”标记，不是假造配对拒绝。完整实际candidate_diagnostics通过既有resources.extra_collectors/feedback_artifacts提供，少量范围说明经existing supplemental_evidence传递。CPU预检确认接口保留事实和历史代码；实际角色是否读取、如何解释，仍要查初始快照和工具，不能从transport通过推断消费。

反馈范围是可核对的当前事实：固定C完整质量通过但质量阶段62次内部setup回退，单独run-only诊断首decode内部logit自检失败；内部0.008/0.008与原0.02/0.02不同，未保存数值/完整张量，不能判定原质量违反或唯一原因。不提供修复代码、分组、算术归因或旧成功方案。普通selector自行根据新原baseline的实际discovery选择范围。此前父路径质量/固定时序不是新快路径优化接受；B TTFT超限和全部长样本保留。

请求保持完整B1/prompt256/32greedy BF16/SDPA/noTF32、完整logits.02/.02/exact tokens、两同步/owner、2continuous/2warmup/5clean。当前设备原TTFT23.285755×1.05=24.45004275，不使用旧25.2536046。新搜索仍需原screen/配对噪声/约束规则；只运行这一新trial，保留生成/worker/质量失败，不重试到通过。没有在本controller安排独立整版验证或固定native；真实planner可按现有机制采集。若有新接受，须后续审计再决定独立验证，不预设优化成功或多bundle共存。

这是设备迁移与开发者事实辅助的同adapter续行，不是未辅助原入口或单变量A/B。源手工映射和以前owner辅助边界不消失；旧接受47.325%及其他历史决定不改写，不作跨设备增量。空间约10.81GiB，一次串行试验，无并行大Qwen/参考副本；299冻结身份预检。启动时验证实际UUID和真实本地OpenCode健康，长任务只设一个完成/磁盘监听，无模型轮询/投递测试。raw/private配置留服务器，GitHub仅小摘要/代码/哈希。
