# Qwen内部setup失败的诊断作者

**终态已审计：2026-10-29 19:12:25 UTC完成；发现具体CPU证据保留缺陷，未启动GPU观察。** [审计小摘要](evidence/v6-qwen-setup-author-20261030/public-terminal-audit.json)、[原样probe](evidence/v6-qwen-setup-author-20261030/diagnostic.py)、[原样plan](evidence/v6-qwen-setup-author-20261030/diagnostic-plan.json)保留。服务器terminal-audit SHA `6e55536a0dd76551ad729f8760fac96f9133b3b79271a1d08dfa2f17437ca6d1`。事件为batch_finished，GPU空闲，实际150GiB余约9.57GiB；无清理或raw转存。

7初始文件、六显式输入75,682B逐字核验，357冻结身份/core保持。真实generation-ffd3bf5531c4共66 assistant、72工具（45 bash/15 read/4 write/8 edit）、0框架schema repairs；provider估算2.32093677894，非账单。最终probe25,567B SHA `8ab9078e71390b240d8a471cc765084a58b9681d3170f6a8a4050a7221fb2ad4`，plan28,511B SHA `c0dc33c969b45437517d2f2282138a6504f62f2014c4b06b0dc992decf097c8`（完整准确值见审计摘要及文件SHA）。完整会话、工具原始输入/输出、大输出未截断副本、初次生成和edit序列、最终CPU fixture均留服务器。

实际代码包装BindingSession.install、reference和torch.testing.assert_close，在原比较前记录CPU摘要，原entry调用一次；安装后reference闭包仍保留，记录窗口关闭时不再采样，但有额外Python调用/owner。没有重新执行模型或强制快路径。摘要含非有限标记、全量统计和至多64个有限超差项，没有保存完整操作数，不能从保存数据独立重算所有统计。setup操作数的CPU FP32 .02/.02比较不等于原全输出质量；非有限元素排除在有限超差计数之外，计数零不代表通过。

**关键缺陷：`_MAX_ASSERT_SITES=128`把成功比较也计入限额，此后所有记录都丢弃。** 独立fixture只运行CPU real torch comparator：128次成功后第129次失败，总计129、保存128次成功、overflow1，原AssertionError仍传播，但失败操作数/结果完全未存。原04df代码若到达已记录的末尾E断言，前序包括1次fused、31次exact、72次KV及A/B/D各31次成功检查，E是第198次；这是条件代码计数，不是新的GPU轨迹。因此原probe不能保证覆盖本案所需后续失败，不能直接启动GPU。

同一新目录还检查NaN/Inf strict JSON保存，以及真实probe.run委托CPU stub：valid=false返回保持同一对象；未捕获RuntimeError保持同一异常对象；两者各调用一次、hooks恢复、CUDA未初始化。非有限测试的有限超差计数为零但原比较失败，未把NaN/Inf解释为实际Qwen结果。原probe和CPU失败均冻结，下一步仅[一次窄范围真实修复](v6-qwen-setup-diagnostic-repair-20261030.md)，不给手写实现/模型原因或其他后续答案。

完整实际访问审核覆盖供给输入、索引中的原请求/results/source、框架及installed transformers、自产CPU测试；未观察到后来developer audit/接受代码/其他诊断答案读取。共享/tmp/opencode/diagtest含旧文件，但本次记录使用自己重写的命名fixture，实际import/traceback指向框架，未观察到旧fixture导入；FS仍非隔离，不能作绝对无访问证明。

作者错误完整保留：错误installed路径、rg缺失但pipeline exit0、DynamicCacheLayer不存在、layers参数错误、None cache、空git历史、初始持久化语法错误、mock shape/引用ID错误、Path.sep错误。Agent自行修正后tiny CPU BindingSession/cache与stub测试通过；随机mock max1实际1.0000001192造成断言，改的是CPU测试精度至1e-5，未改模型质量。dormant测试只比较两次report，未调用dormant wrapper；边缘恢复断言含`or True`是无效检查，另一个安装fixture实际检查身份。重复测试覆盖早期临时结果，最终fixture/完整工具历史保留，不声称所有临时raw存在。

未采纳plan的全覆盖/零扰动/性能保证：每个操作数先describe复制CPU、比较又复制，两份操作数合计四次复制；具体peak未测。拷贝/同步/metadata/异常帧/闭包改变生命周期；约1e-5开销无证据，文件哈希实际在entry前做。模型内部实现会在setup执行，不能说从未执行；单次baseline匹配不量化观察扰动。同量元素但不同shape会flatten，非一般结构质量；first-layer metadata不是所有层KV/alias证据。cache推进/舍入等仍是假设，没有唯一因果或新优化接受。

审计自身两次重建错误也保留：首次假设edit.oldString严格匹配，但tool45使用模糊匹配；第二次统一用display diff，而tool61展示缩进被归一。最终按精确输入替换及tool45实际diff重建并核对最终字节，没有重跑作者/模型/CPU fixture或改生成代码。以下启动记录按历史保留。

**已于2026-10-29 18:53:20 UTC（香港10月30日02:53）启动**，冻结`c8965d34`、controller21633。[启动回执](evidence/v6-qwen-setup-author-20261030/launch.json)、[真实本地服务健康](evidence/v6-qwen-setup-author-20261030/session-health.json)、[冻结预检](evidence/v6-qwen-setup-author-20261030/frozen-preflight.json)已保存。生产core仍ed24af7a，未清理磁盘。

18:56:05的[实际启动核验](evidence/v6-qwen-setup-author-20261030/startup-verification.json)确认fresh generation-ffd3bf5531c4/session ses_e76fd52cdffeUfX5DBS0vwSyQX，7份初始文件、六显式输入75,682B逐字相同，357冻结文件保持。16个真实assistant消息/19工具，已读取原输入、索引、原B/C请求/结果/source/framework；一次错误installed路径查询保留。此时尚未生成有效性结论，完整访问/CPU测试/最终代码仍须终态审计。GPU空闲、df余10,289,156,096B（约9.58GiB）。唯一普通完成/8GiB监听PID13092已核验等待，[启动](evidence/v6-qwen-setup-author-20261030/wakeup-setup.json)与[实际状态](evidence/v6-qwen-setup-author-20261030/wakeup-verification.json)留档；旧44008已处理退出，无重复通知测试。

这是[当前设备普通试验](v6-qwen-device-feedback-20261030.md)之后的一次独立真实diagnostic_author生成。外层评估在父路径质量通过，但新候选两种模式都内部自检失败；现有run-only sidecar只有异常位置/传播，没有完整比较值或输入状态，无法判定原.02/.02是否超差或比较是否等价。以这个具体缺口检验作者能否编写有界观察器，不是新增候选搜索、优化接受或重新开启已关闭的prompt/verifier研究。

复用既有ProjectAgents/Runtime/快照接口；沿用通用作者instructions/interface，增加一段通用的“内部捕获异常/回退/有界比较证据”说明。六份显式输入是原task、原始candidate_diagnostics sidecar、原adapter、确切04df候选、原请求/worker/source/只读controls索引和接口，另有自动worker-environment。没有开发者terminal/public audit、fast_enabled/chain E推断、因果答案、手写修复、其他作者输出或旧成功实现。实际文件大小/快照/完整工具读取须终态审计，外部FS可访问，不承诺隔离。

此controller只生成diagnostic.py与diagnostic-plan.json并验证syntax/schema，禁止作者会话GPU/model/collective执行。CPU/library/mock允许但必须保留失败，成功mock不等于原完整GPU质量。原entry应每worker只调用一次，保持完整连续质量、warmup/repeats、bindings/math/同步/返回值及内部catch/fallback；不得启用被禁快路径、放宽自检/外层容差、重放模型或人工构造替代结果。选择小规模比较值/必要状态元数据，不dump全部KV/model；缺失与非有限值须如实可序列化，原return/exception保持。观察带来的CPU拷贝、同步、owner和生命周期扰动必须明确。

生成后先审核原样代码、真实访问/CPU测试/体积，合适才另行声明固定观察执行；不自动安装生产collector或新增审批层。不把作者schema完成当执行正确、质量或性能接受，不反复生成到通过。源/当前/旧接受版本、全部旧失败冻结；共享controls不复制，未重新做源比较，既有人工源映射边界保持。

当前设备TTFT原anchor23.285755/limit24.45004275及完整B1/prompt256/32tokens、BF16/SDPA/noTF32/.02/.02/exact tokens不变。检查实际150GiB df与live GPU，约9.60GiB余量；仅小文件会话，未安排大trace/参考副本或清理。raw/private历史留服务器，GitHub只代码/哈希/小摘要。启动后设唯一普通完成/8GiB监听，旧44008已处理；没有模型等待轮询、Goal或投递测试。
