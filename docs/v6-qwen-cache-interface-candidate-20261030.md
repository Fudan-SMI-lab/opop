# Qwen执行接口失败后的单次候选生成

后续[原字节固定B/C](v6-qwen-cache-interface-fixed-20261030.md)已于22:31:35 UTC完成并审计：500身份保持、10样本/4完整零误差检查；新C全生成380.435753ms，但TTFT26.076953>24.45004275，不接受。实际62个Graph标签不是native证据，setup gate/所有权限制保持；不重复同版本到通过、不采用prompt。另对工具6完整2,766,678B输出按已记录纯JSON打印算法重建，与服务器原始截断输出文件逐字相同，补充证明输出来自原初始反馈，未重跑Agent/模型。

**终态：2026-10-29 22:19:01 UTC，batch_finished。** 审计SHA `eba45afbb2a83f8d267affc822a9a55ee7463f55c801525d664407b447728267`；[公开小摘要](evidence/v6-qwen-cache-interface-candidate-20261030/public-terminal-audit.json)。独立重放实际registry/plan路由，33初始文件/130,630,326B、492冻结身份、prompt/model/tools/core、源空current2a42全部保持。47条assistant/57工具（34 bash、10 read、4 write、9 edit）、0 schema修复；provider估计2.39222807294，非账单。没有GPU模型、搜索或质量结果。

最终代码33,276B/SHA `7e78b82dd21391e3db3d6292e3798feff1908f53259aa732a3c011660c28fb92`，proposal9,799B/SHA `048d30b638809aad76ef99e757be41ea20345d1def2f72da5c332cc56aaf94b6`，parameter_space={}、eliminated_calls=[]、requires/replaces=[]。初始代码及4次修改重建精确匹配；工具34是fuzzy edit，实际返回diff中的旧拼写与请求oldString不同，按真实diff保留。完整会话、大型截断输出原文件、CPU脚本及修改历史留服务器，没有raw转存。

角色实际读了父方法AttributeError（工具12）、target签名（13/15）及安装的Cache/DynamicLayer/Qwen/SDPA源码（21–27），改为真正DynamicCache/DynamicLayer。完整命令/输出/脚本审查未见后续开发者绑定审计、接受或核验答案读取；文件系统非密闭，不能作绝对无访问证明。新反馈supplemental=null；此前CPU事实与人工源映射辅助边界仍存在。

代码预建31个可复用cache，内部K/V是共享master视图，并非每次返回新对象。活跃对象及第一层keys指针用于识别父调用后的修改，不能证明所有层/values/混合别名/并发所有权。teacher stride[32,1]仍不符合框架[1,1]。A/O/P自检直接调用replacement，未覆盖原框架完整两次run/observe；最终KV比较仅初始oracle链。明确代码风险：verify前置fast_enabled=True，但verify异常catch未清掉它，disable_reason非空时仍可能安装；Graph异常后走eager interior仍会发fast_graph_replay标签。以上是代码审查，不是已发生当前GPU错误或native证据。

CPU错误保留：30不支持layers构造参数、40缺step_caches、42错误要求未写尾部也全1、45期待CPU输入走CUDA快路径、48错误路由标签，均exit1。14错误路径抑制stderr、50拼错dev-venchain后继续测试、55管道tail不能单看exit0。最终3层BF16 CPU测试手工模拟copy/replay/clone/cache尾部，真实replacement调用受CPU设备guard限制；其stale-lineage/_absorb标签不证明进入后续分支。工具31更新layer1却检查layer0，32才更新layer0。通过后55再次运行；不是完整GPU质量。Agent删除自身__pycache__，早期覆盖的临时输出非全部存在，完整历史与最终夹具保留。

不采用“内部已GPU证明”、自检保证无状态泄漏、prefill不绑定保证TTFT、profile gap可回收为关键路径或显存必降等推断。下一步为[一次原字节固定B/C](v6-qwen-cache-interface-fixed-20261030.md)，保留风险，通过原外层完整质量/coverage检查判断实际行为；无手改、观察器、搜索、调参、Nsys或正式接受。当前TTFT上限24.45004275，原失败不覆盖。审计时GPU空闲、150GiB盘余约8.11GiB，无清理；旧17992完成通知处理，不重启。

**已启动：2026-10-29 21:59:05 UTC（香港10月30日05:59），冻结61a968f34b45bc00f7efe53f39728c1812b6d901，controller52752。** [启动](evidence/v6-qwen-cache-interface-candidate-20261030/launch.json)、[492身份预检](evidence/v6-qwen-cache-interface-candidate-20261030/frozen-preflight.json)、[本地服务实际健康检查](evidence/v6-qwen-cache-interface-candidate-20261030/session-health.json)通过。

22:01:21 UTC[实际启动审计](evidence/v6-qwen-cache-interface-candidate-20261030/startup-verification.json)确认generation `a8d232f6cb9b` / session `ses_e76533c6cffeeLrihDWiZF4r03`，33初始文件/130,630,326B逐字匹配，prompt/model/tools/core相同，registry+plan路由一致、supplemental_evidence=null，历史6ec4代码及492冻结身份保持。11条assistant消息/18工具；实际读取新失败sidecar和完整原始traceback（工具12）、原target签名（工具13/15）。这只证明早期失败信息读取，不是正确修复或质量证明。工具14错误dev-v6-work路径被stderr抑制且管道exit0保留；早期无GPU worker。

数据盘余8,707,149,824B（约8.11GiB），无清理。用户随后报告本地PowerShell弹窗：检查工具确实使用PowerShell，且隐藏Python监听的SSH子进程未显式隐藏，是短暂控制台弹窗的可能来源，无法追溯所有已消失窗口。现已给本地SSH/通知子进程设置CREATE_NO_WINDOW；活动监听652正常停止、未投递事件后替换为17992，[记录](evidence/v6-qwen-cache-interface-candidate-20261030/wakeup-window-fix.json)保留。仍只有一个普通完成/8GiB监听，实验没有重启，未动VS Code终端。只改变本地编排，不改冻结服务器框架或模型请求。

在[CPU事实反馈普通试验](v6-qwen-setup-cpu-feedback-20261030.md)的四个worker均因 `_CacheHandle.get_seq_length` 缺失而失败后，声明一次现有candidate角色生成，根目录 `qwen-cache-interface-candidate-20261030`。这次没有新selector、完整搜索、参数执行、GPU worker、接受或指针发布。它不是重启旧trial，也不是重复评分直到通过。

输入以实际generation `d9a0f3917237` 的原任务、已选decode scope、source/adapter/contract/targets/current discovery为基础；更新为本轮framework原始latest-feedback、精确失败代码 `6ec4e97a…` 的历史产物和原始失败collector包。通过既有feedback_artifacts/candidate_artifacts/registry+plan route及ProjectAgents.generate交付，使用原candidate提示/model/tools，正常schema修复保留。旧selection中有关opaque handle/stride/TTFT的错误推断不手工改写，实际失败反馈可供角色核对。不是整角色单变量A/B。

CPU预检通过：492冻结文件，33预期输入共130,630,326B；更新6文件、移除旧binding_cost包、增加失败Nsys包。routed反馈5,305,639B/SHA `c68771a7585cd4418a1da229f977e504b37ffc8dfab534d7b393fc2edc3f04a6`，supplemental_evidence为null。没有提供开发者CPU binding控制、终态事实表、因果答案或修复实现；CPU控制仅作外部审计。此前开发者CPU事实与人工源映射辅助边界仍存在，不能称为无辅助原入口结果。外部文件系统可访问，后续必须检查完整工具路径、输出及CPU探针，不以初始文件名推断没有污染。API仅观察费用，不设新上限；不改生产提示、采样、allocator或接受规则。

源current保持空bundle `2a42b9b1…`，prepared009b及当前20e8 GPU原controls/discovery只读共享；不复制reference或运行新发现。完整B1/prompt256/32 greedy BF16/SDPA/noTF32/full logits .02/.02/exacttokens、两同步和状态生命期不改。当前原TTFT锚23.285755、上限24.45004275保持，旧设备指标不合并。

该控制器只生成candidate.py/proposal.json，语法/schema完成不是执行有效性。终态需要审查实际返回缓存、框架签名派发、参数、fallback、CPU模拟范围及所有错误；适合时另声明精确字节固定执行，不能隐藏修复或自动GPU验证。没有新collector或强制review层，关闭研究保持关闭。

启动前核对实际150GiB df与GPU，约8.37GiB可用，两卡空闲；本批仅角色输入/会话/小代码，没有大型profile或GPU源副本。raw/private配置与会话留服务器，GitHub仅代码/哈希/小摘要。新长工作只设一个普通完成/8GiB监听，low_disk_space不是完成；旧40560事件已处理，不重启旧批次。
