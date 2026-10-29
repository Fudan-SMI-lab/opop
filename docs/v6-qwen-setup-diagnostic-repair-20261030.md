# Qwen setup观察器后续失败证据的有限修复

本次修复和条件固定观察均已完成，原错误保持。后续[普通反馈续行](v6-qwen-setup-feedback-20261030.md)只引用原样观察产物，不再生成/修补本probe，不重复观察到通过，不自动安装生产collector。

后续[原样固定B/C观察](v6-qwen-setup-observation-20261030.md)已在19:56:24UTC完成：370身份、10样本/4完整零误差检查保持，198次安装比较中的首发和晚发seq198失败均保存；256个保存超差项独立核算。候选最终仍父回退、TTFT超当前限制，无新接受。该执行使用本页新bc4751 probe，旧8ab9 probe与CPU截断失败保持；以下为19:41修复终态和原协议。

**终态已审计：2026-10-29 19:41:01 UTC完成，独立CPU边界通过，可安排一次原样固定B/C观察。** [审计小摘要](evidence/v6-qwen-setup-diagnostic-repair-20261030/public-terminal-audit.json)、[原样probe](evidence/v6-qwen-setup-diagnostic-repair-20261030/diagnostic.py)、[原样plan](evidence/v6-qwen-setup-diagnostic-repair-20261030/diagnostic-plan.json)保留；服务器terminal-audit SHA `8d2363dd0f11f91c558b71bf5f8d83ec423f0f50e9eb66de541c45ca91dd8432`。这是完成事件；真实150GiB余约9.56GiB、两GPU空闲，无清理。此前生成脚本没有执行GPU，下一步为[独立声明的固定观察](v6-qwen-setup-observation-20261030.md)，不是优化接受。

10初始文件/九显式131,084B逐字保持，原六75,682B不变；365冻结文件及core/tool/model settings核验。真实25 assistant/39工具（15 bash、14 read、3 write、7 edit）、0框架schema repairs，provider估算0.73219624902，非账单。probe32,998B SHA `bc4751ca9b77f9edaf5269c10dd053298e287d50452f51d113f1555a33265ff5`，plan40,464B SHA `33fdb9239bc1420293fd89fe74a139a081276e2c955aca411db4b9c1bdef63a6`；两者均一次写入后未编辑，首次生成等于最终。完整private history、工具命令/输出、测试初稿/edit及最终fixture留服务器。

实际修复仅改变Recorder及常量/说明：每次比较先记录轻量来源，原comparator执行一次，返回或异常之后才决定保存详情；最多128条详细记录，其中失败最多32条。后续失败挤出最早成功记录并留下stub，超过失败上限则明确计数。最多64个未保留site键加other，共可能65键；原reference前2048条上限未变。没有按候选错误文本路由。run、安装包装及数值统计helper的AST保持，但CPU拷贝从原比较前移到后，省略的成功项也不再复制；“只改变保留、不改变同步时机”说法不成立。

独立复用原CPU序列一次：因旧fixture固定旧SHA/缺陷断言，显式保留原文件并仅改新SHA和预期结果，执行顺序不变。128成功+第129次失败现保存127成功和1失败，原失败传播、seq129/max1/2超差项及seq1淘汰stub正确。NaN/Inf真实失败与strict JSON标记保持；valid=false同返回对象、未捕获RuntimeError同异常对象、entry各一次/hooks恢复，无CUDA初始化。新增混合边界168调用=132成功+36失败，128条详细记录/32失败、32成功淘汰、4成功省略+4失败省略均对账。实际BindingSession安装CPU小builder的内部晚发异常仍由builder捕获，原成功返回保持、失败详情记录；KeyboardInterrupt保持同一对象并记unfinished，不伪造完成记录。

新增安装fixture首次在模块级提前import BindingSession，早于probe.run安装类hook，导致空window和审计IndexError。该夹具/输出/log保留，另建entry将import放进函数补测，匹配原真实evaluate的导入位置。未重复原始序列/混合边界、未改生成probe或模型；这是审计夹具错误，不是被测Qwen异常。

完整实际访问覆盖原输入、命名CPU fixture/result、原baseline entry及当前framework；未观察到后续接受代码、开发者terminal/public audit、其他任务答案读取，FS非隔离。作者实际读candidate440–700行，自行计算约197次exact比较，不是新输入提供的开发者计数。作者测试先有错误line0预期、两处缺rtol的ValueError、bound-method对象身份错误，修正的均为自有CPU测试；后续仍有line0错误，最终10项通过并重复运行一次。其BindingSession测试采用diagnostic_passthrough，未实际加载候选；after-restore测试调用的是恢复后的原assert，不验证存续reference闭包。所有错误和最终文件保持，覆盖的早期临时输出不声称仍存在。

剩余限制不采纳为保证：只存摘要/至多64有限超差项，没有完整操作数，不能独立重算全部统计；setup CPU FP32容差比较不等于原全输出或源质量。非有限计数与有限超差计数分开，后者零不代表通过。每对操作数在describe/statistics共四次CPU复制，非“两次”；临时内存、延迟和1e-5开销未测。闭包、异常帧、同步和CPU工作影响生命周期；不证明零CUDA allocator变化、任意并发/owner安全。first-layer/object-id/length仅部分来源，非全层alias或唯一原因。setup确实执行内部实现，旧outer父回退质量不认证快路径；相同路线/结果不证明全部内部状态相同。文件哈希实际entry前读取，非写证据时。计划提出的额外profile/library实验未采纳；cache推进/舍入原因仍是假设。

原旧probe截断失败、04df原TTFT拒绝及全部历史结果保持，生产core不变，不新增审查层或重新打开已关闭研究。以下启动和原协议按历史保留。

**??2026-10-29 19:30:11 UTC???10?30?03:30???**???`b2151e27735f71268240831834c90281a38f705e`?controller28429???core???[????](evidence/v6-qwen-setup-diagnostic-repair-20261030/launch.json)?[??????](evidence/v6-qwen-setup-diagnostic-repair-20261030/session-health.json)?[????](evidence/v6-qwen-setup-diagnostic-repair-20261030/frozen-preflight.json)?????launcher?????FSDP????argparse?????????controller/session??????[???](evidence/v6-qwen-setup-diagnostic-repair-20261030/launch-preflight-attempt-1.json)????????????????????

19:32:33?[??????](evidence/v6-qwen-setup-diagnostic-repair-20261030/startup-verification.json)???generation-4328159a6ab9/session ses_e76db93dcffeXZLTnW6am6tdss?10????????131,084B?????365???????10??assistant/18??????probe/plan????????CPU fixture/result???installed???rg???pipeline exit0???????????????????????????????FS??????????SHA19139306b602665960f7bf3593052efc47bb62e37f145d3035884497fd49dac7?????

??[????/8GiB??](evidence/v6-qwen-setup-diagnostic-repair-20261030/wakeup-setup.json)PID43664?[????](evidence/v6-qwen-setup-diagnostic-repair-20261030/wakeup-verification.json)?????13092??????????PID?????conhost??????????GPU???df?10,263,711,744B??9.56GiB????????raw??????

准备一次真实diagnostic_author修复。上一份[生成观察器](v6-qwen-setup-author-20261030.md)保留原样：probe `8ab9078e…`、plan `c0dc33c9…`。独立CPU接口测试发现128次成功比较耗尽上限后，第129次失败仍原样传播，但操作数和失败结果均未保存。尚未执行该probe的GPU观察，不将这个持久化缺陷解释为模型数值原因。

使用既有ProjectAgents/Runtime/snapshot接口生成一次修复，保持通用作者指令，只追加有界后续失败保留要求。原六输入75,682B逐字不变，另加原probe25,567B、原plan28,511B和1,324B执行失败记录，总九显式输入131,084B，另有自动worker-environment。失败输入只含真实CPU截断结果、fixture源码与原结果路径/SHA；不提供开发者terminal audit、候选chain E计数分析、手写修复、其他任务答案或后来接受版本。命名CPU fixture可以读取，外部FS非隔离，终态审核完整实际访问。

修复应保持有界资源、真实总计/省略范围/比较来源，保留原一次entry、return/validity、内部catch/fallback、未捕获异常和hook恢复。不得删除所有上限、按固定错误文本路由、改候选/质量阈值、强制快路径或重跑模型。不提供实现答案；原输出与失败证据冻结。controller仅生成与syntax/schema校验，无GPU/model/collective执行、搜索、性能接受、生产collector或提示采用。

终态先审核实际diff、完整工具/CPU失败、保留上限语义与资源占用，再在新目录对原CPU边界做独立检查。若适合才另行声明原样固定B/C观察；否则保留具体缺陷，不隐藏手修或反复执行到通过。365旧文件预检；共享controls/source/接受版本和生产core `ed24af7a…`不变。

当前设备原TTFT anchor23.285755、limit24.45004275及B1/prompt256/32tokens、BF16/SDPA/noTF32、full logits .02/.02/exact tokens保持。前序04df只在父回退路径质量通过且TTFT拒绝，仍无新接受。人工源映射、跨设备测量时期、观察器CPU拷贝/同步/持有影响均保留。已关闭提示/解释/核验研究不重开，不增加reviewer。

启动前检查真实150GiB df与GPU，当前约9.57GiB空闲、无GPU进程；本轮不清理、不复制参考、不下载raw/private历史。仅代码/哈希/小摘要进GitHub v6。新长任务只设一个普通完成/8GiB监听，旧13092完成事件已处理，不重启。
