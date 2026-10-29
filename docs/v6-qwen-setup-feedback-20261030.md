# Qwen原样setup观察产物的一轮普通反馈续行

后续已完成[CPU事实反馈试验](v6-qwen-setup-cpu-feedback-20261030.md)：新6ec4代码的四个worker全部因opaque `_CacheHandle` 与父Cache接口不兼容而执行失败，未返回质量检查或clean样本。候选实际读到了内联CPU事实，不能把这次失败归为未送达；也不能宣称事实辅助提高了质量或性能。旧4693的父回退质量/TTFT拒绝、旧CPU证据与所有原始样本保持。下一步仅一次原始失败反馈的候选生成，不自动运行GPU或改变生产规则。

本批于 **2026-10-29 20:48:04 UTC** 完成，唯一trial `bf84d454d6` 被拒绝：`no_valid_parameter_candidate`。重启中断的是后续审计，恢复时远端已终态，没有重启批次。两卡空闲、UUID仍20e8/a24a，150GiB实际余约8.83GiB，无清理；旧监听7324已处理。

[终态小摘要](evidence/v6-qwen-setup-feedback-20261030/public-terminal-audit.json)对应服务器audit SHA `34724d510e99df73d6cbfcc82ebbc83b639ec55c0aec2abea3bc1600b944273d`。382冻结身份、8个worker的source/adapter/entry/core/UUID、原discovery/prepared/request及实际selector/candidate registry/plan路由重新核验。源和分支current仍2a42b9b1、空bundles，无配对接受、重发现、共存或独立阶段。原controls/source映射没有重跑。

三个角色初始31/34/8文件、98,228,556/140,516,871/18,213B；assistant36/51/9、工具40/63/12，均0框架schema修复，provider估算共4.19619301566，仅观察用量。完整命令/输出/CPU脚本/初始代码/三次编辑保留服务器，重建最终代码逐字相等。**selector和candidate最终均未打开新增index/probe/plan/观察JSON**；selector工具10只打印指针，主要消费原打包sidecar、旧04df和库源码，故不能把结果归于新观察消费。外部FS非隔离，完整记录未见后续接受实现/审阅答案读取。错误source目录、evaluation为null的AttributeError、错误transformers导入、CPU DynamicCache的layers参数TypeError保留；后续CPU cache/inference操作、语法/schema通过，非完整模型/GPU质量。Agent删除自己的__pycache__，会话及截断search输出保留；文件大小不是上下文使用量。

新代码32,394B/SHA `4693d1472b012cfa786209e8ce2ede85a84ae443d7cc1c2212d2680ff42dcd38`，revision `e666c201…`，空参数仅一实例。单decode组，删除Triton/prefill binding，手工展开模块算子、GQA无mask SDPA、31长度Graph、masterKV和预建mirror。mirror并非每次返回新建；直接setup链不等于框架完整run/observe次序。内部转FP32的.02/.02不等于外部BF16关系算术；代码存在bitwise失败后第二次Graph replay逻辑，不能声称首次全部通过。

| 端点 | end_to_end_ms中位数 | TTFT中位数 |
|---|---:|---:|
| 唯一screen | 788.271980 | 26.232951 |
| binding pair0 parent / passthrough | 729.987131 / 720.858054 | 25.715212 / 25.133755 |
| binding pair1 passthrough / parent | 716.999294 / 731.796445 | 24.561506 / 25.397064 |

25样本/10完整零误差检查全保留，正式screen仅5样本/2检查，其余20/8属于绑定诊断。screen质量路径为62次父setup回退，TTFT超过原23.285755×1.05=24.45004275；32.126278ms TTFT、810.598709ms整段尾样本未移除。setup7047.981304ms在warm计时外；runtime/lifecycle peak增加105,030,656/165,080,064B，不是显存收益。framework fallbacks空不抵消内部回退。

两个run-only sidecar均记录63次input_ids guard异常，以及一次E断言经三帧传播；252个generator False和1个master-view False不是失败。E新消息仅报告舍入max22.0312，无完整新操作数/缓存元数据，不能复制旧观察计数当新事实。run-only valid不调用observe/compare，也不建立快路径完整质量。

[独立CPU接口控制](evidence/v6-qwen-setup-feedback-20261030/cpu-control-flow.json)一次运行原样新代码，未初始化CUDA/执行模型：shape[1,1] teacher切片已contiguous，`.contiguous()`仍返回同对象/stride[32,1]；真实replacement guard送父路径，CPU DynamicCache例子两次增长258→259→260。结合B/R各31次和E一次，支持对63次guard和E位置错配的代码/顺序解释，非逐调用GPU缓存轨迹或唯一数值根因。另一个分离控制只stub setup/verify成功，原build仍因`failure="not_attempted"`未清除而关闭快路径并清空Graph；这是潜在成功分支缺陷，不能解释本次先发生的E断言。无手改候选。

timeline原trace/clock/union/gap/sync已重算：66,441 kernels、active union385.976993ms、span1481.939818ms、gap1095.962825ms、4次sync API、profiled wall1482.621509ms。不能从clean wall相减获得host成本。绑定passthrough较parent少9.129077/14.797151ms不是负物理成本或候选收益。无新Nsys/加速Graph原生证明。prefill不变保证TTFT、旧fused仅因严容差、唯一host/带宽原因以及planner“无Graph路径问题”等主张未采用。

下一步只把新CPU控制的实际结果/界限作为明确开发者辅助反馈，经既有supplemental_evidence声明一次普通续行，不提供实现修复、不改提示、不重开已关闭索引阅读或审阅研究。旧失败保留，新的生成/质量/接受须分别审计。以下为原启动记录。

**已启动：2026-10-29 20:16:48 UTC（香港10月30日04:16），冻结`6cf7bfe305d5724768cd022cd8277802cc25c1ca`，controller33739。** [启动记录](evidence/v6-qwen-setup-feedback-20261030/launch.json)、[本地服务健康](evidence/v6-qwen-setup-feedback-20261030/session-health.json)和[冻结预检](evidence/v6-qwen-setup-feedback-20261030/frozen-preflight.json)通过。实际OpenCode start/health/stop，无模型提示预检。原反馈新增503B变为13,936,321B/SHA`76a2adc5fe5ecaf73a7b68fa8737e6e121d0696ada287120bfa727e728a903c8`；索引3,511B/SHA`9d7a6653adbb0f8447f4f333c8f09a0697cbe97e26604fe9424f1354f6113784`。[现有接口预检](evidence/v6-qwen-setup-feedback-20261030/transport-preflight.json)保留所有原字段/打包产物，只有supplemental不同。

20:19:37 UTC[实际启动核验](evidence/v6-qwen-setup-feedback-20261030/startup-verification.json)：selector generation-d3a31bd6978a/session ses_e76b0be25ffe704tjialDbRUBy，31初始文件/98,228,556B与快照逐字相等，实际feedback5,816,140B/SHA`bb318e94f19e22af2f462addca31b3ab16dd0ba8716834c4a9b90aa17212a744`匹配真实registry route。原04df历史代码、全部原打包产物、382冻结文件核验；discovery/prepared/request和原source/current保持，没有新GPU jobs。与上一普通selector相比prompt/model/tools相同，但6输入改变、3额外历史诊断输入新增，故不是整个角色的单变量A/B；总文件大小不代表全部进入模型上下文。

早期17 assistant/19工具已保留服务器private history，SHA`1c7cca6bdb4548d4104626350978ce363bf74db82e53afc2101412f34184815f`。[早期工具检查](evidence/v6-qwen-setup-feedback-20261030/startup-tool-review.json)显示工具10 `prt_1894fa97b001zfvk9CM224yMVn`实际打印索引指针及SHA，输出SHA`04e1c92922496a8c2a19f3581169a976f2a9dfcf75b194737cb1690540673951`；这19工具尚未打开新index/probe/plan/观察JSON。读取的是原反馈/代码及既有sidecar/binding/timeline，不能把送达或指针暴露称为新证据消费。工具3尝试错误source/目录exit2保留；输入树实际为inputs/source。此为早期记录，后续完整访问/代码/质量仍待终态审计。

唯一[普通完成/8GiB监听](evidence/v6-qwen-setup-feedback-20261030/wakeup-setup.json)PID7324及[存活核验](evidence/v6-qwen-setup-feedback-20261030/wakeup-verification.json)通过，旧31864已处理退出。20:19核验controller运行、GPU空闲，真实df余10,034,954,240B（约9.35GiB）；尚无新候选质量或接受结论。没有清理/raw转存或生产改动。

在[固定观察](v6-qwen-setup-observation-20261030.md)完成并审核后，声明一轮新的普通selector/candidate试验。原来的`batch_finished`通知已在上一轮处理；本次先核对相同终态与audit SHA，再决定新请求，不重启观察、旧试验或旧监听。服务器两卡空闲，150GiB实际余约9.55GiB，无本轮清理。

起点是`qwen-device-feedback-20261030/qwen-continuation`当前空bundle版本`2a42b9b182b77d54a4028e779d8ba8f8bee5dabec4a83917015ed52e486e0dd1`，共享当前设备原controls和已完成baseline discovery。原prepared baseline仍`009b4178…`，没有再次controls、源比较、初始GPU发现或参考复制；既有fork_run复制discovery元数据并共享不可变artifacts/reference。原source/current及历史接受分支不动，新隔离分支只按现有搜索规则形成自己的结果。

原framework latest-feedback为13,935,818B、SHA`2cf4eb2a2476f398bcbe0cd045b0790374d51852e31bea86d64416480e87b25c`，来自trial0db61e9938的`no_valid_parameter_candidate`。保留原全部字段和值、末尾右括号之前的字节，仅新增现有` supplemental_evidence `容器下的diagnostic_artifacts索引指针。索引只列原样probe/plan、既有B/C原请求、观察请求、worker结果和观察JSON的路径/SHA/大小，以及通用观察范围限制；不加入开发者审计事实表、cache长度结论、实现修复、其他已接受候选或解释/核验角色答案。原04df代码仍通过历史candidate_artifacts提供，不安装到起点。CPU预检实际feedback_artifacts与registry/plan route，要求其他路由字段和原打包产物相同；真正新角色的初始快照和实际读取仍需单独核验。

这是对新增数值与输入来源证据的一次普通使用，不是同快照单变量A/B、原入口无辅助cohort或稳定提示因果证明。生产selector/candidate提示保持，不重新打开已关闭的bounded-reading/解释/核验研究，也不强制选组或手写候选。既有CLI `--resume --rounds 1 --feedback`使用默认作用域/参数探索；API仅观察用量，不设新上限。保留正常schema修复、工具错误、CPU/GPU探测和所有失败，不重试至通过。原角色工具可能自行探测，不能只凭controller配置宣称没有GPU活动；外部FS非隔离，终态审查完整实际访问。

完整B1/prompt256/32 greedy、BF16/SDPA/noTF32、full logits .02/.02/exact tokens、两同步边界/状态生命周期、两次连续质量/两次warmup/每端点五样本保持。当前GPU0是GPU-20e8003c-f531-d50c-084c-a9a97fe94004；当前原TTFT anchor23.285755、limit24.45004275，不能用旧25.2536046/25.49943795或跨设备合并anchor。adapter498c90f3及manual generated_tokens/step_logits源映射边界保留，来源参考/controls只读。本批没有独立完整版本或专门native阶段；真实planner可请求既有collector。若新版本被搜索接受，终态审计后才安排另一次独立完整版本验证；controller完成或run-only诊断valid都不等于质量/性能接受。

固定观察既有结论不改：370身份、10样本/4完整零误差检查，候选62父decode回退+2父prefill，观察TTFT24.589549超限，无快路径接受。新probe保存198比较中的2失败；256个保存超差值算术已核验，完整操作数未保存，聚合数不能从raw重算。首setup存在超原容差的保存项；E最后两parent输入长度不同是代码/顺序推断，不是所有差异唯一原因。这些开发者结论只在本计划说明，不作为新角色事实表或修复答案输入。原probe、原失败、原timing全部冻结。

新任务根`/root/autodl-tmp/dev/v6-work/qwen-setup-feedback-20261030`。预检检查actual UUID、df、冻结文件、原始反馈和历史代码；本地OpenCode实际start/health/stop、不发送模型提示，私有设置只留服务器。仅代码、SHA、脱敏小摘要发布GitHub v6，raw和完整会话不转存。启动后只设一个普通完成/8GiB监听；低磁盘不是完成，不终止实验或盲删。既有cold JSON恢复journal保持。生产core `ed24af7acc8caa15578e0b859c85ef8a901be51354448ed39094606ed9e10a2f`、历史接受拒绝及所有关闭研究不变。
