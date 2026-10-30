# 已接受联合版本上的一次普通项目选择

后续candidate已完整终审87cddc2c…：eef828确实实现分段反向/3chunk异步规约与更新，但一次精确CPU合同取证暴露消费依赖和参数生命周期问题，暂不GPU；选择预测不当成功证明。继续[一次明确CPU事实辅助纠正](v6-ddp-fused-pipeline-cpu-feedback-20261031.md)，原选择未改、未重做selector、不手写实现或改变生产prompt。详见[候选终审](v6-ddp-fused-pipeline-candidate-20261031.md)。

下一项已经实际运行：[普通pipeline candidate](v6-ddp-fused-pipeline-candidate-20261031.md)，香港04:49:08 controller217716/冻结0c0ae017/唯一隐藏监听36164。04:51:10真实33输入/1469身份及原prompt/model/tools/current508a通过，原选择仅正常canonicalization，无审计纠正输入。尚非新代码/质量或接受；本selector不重跑。

本批2026-10-30 20:40:18.787766UTC终态为`batch_finished`，已完整终审；[公开摘要](evidence/v6-ddp-fused-communication-selection-20261031/public-terminal-audit.json) SHA`e5ce211d7a55318aec7d37573ad885c37ef514fc6c9b0454b852777b4ce5aca5`，完整审计`a922faa5157b6db9b26a2591c4440aa6aae8ce6c766fd2ce5760abed5351e083`。1454冻结身份、30初始文件7,194,917B、原始反馈/实际registry路由、source current10992、单508a bundle、core/prompt/model/tools均保持。没有candidate、GPU或新接受。实际150GiB磁盘余约9.27GiB、GPU空闲；旧50516完成已handled且进程退出，无清理或重启。

完整24 assistant/31工具/0 schema修复，provider估算0.46256337238不是账单。2415读取行和2处明确截断前缀核实，13条JSON/算术/管道输出逐字重建；selection仅一次写入、无编辑，SHA`b43cc9fe941d950240cc2255af55a42db8e18c8f40bd8d319da165a5d41a3437`。它实际读取当前508a、原source/adapter/contract、完整小型独立反馈，以及全部四份current collector JSON；校准rank记录完整打印，timeline/ownership/plan只打印部分，Nsys完整数组参与算术不代表逐条全文阅读。31次实际访问未见模型/tensor/CUDA/processgroup/collective或私有配置、外部审计/答案内容读取；opencode.json仅目录名曝光，非封闭文件系统/绝对不存在证明。私有完整history留服务器，未输出秘密。

选择为execution_plan、3组/7调用，自主提出分段合批反向、分块SUM/divide和SGD调度，区别于关闭的77ac多通信组/Future修复。保留原选择，不手改方案。当前代码确为反向→整块AR→SGD的串行结构，同次native也未见计算/通信重叠，这足以提出实现假设，不足以证明收益。32.51ms校准不是固定wire成本、不可突破下界或唯一剩余方向；497,759,232B是当前载荷，不是普遍不可减少下界。29–37ms预测忽略总传输、依赖、启动与资源争用；梯度就绪不证明参数可安全提前更新，尤其共享embedding；分块相同不自动保证浮点顺序、完整连续状态、exact replicas或失败回退。

[同次原SQLite算术复核](evidence/v6-ddp-fused-communication-selection-20261031/native-arithmetic-check.json)确认工具28/29将ns除1000却标ms，实际输出单位是µs。最终文字52.2/17.87/31.53/2.80ms数值与正确ns/1e6一致；错误标签仍保留。rank0通信前955 kernels共17.867725ms、通信31.532702ms、通信后149 kernels共2.799854ms，不能把全部20.7ms非NCCL当通信前可隐藏工作；rank1另一受扰周期321.602219ms也不用于clean/wire/关键路径归因。没有重采集或重跑模型。

下一有界决策为[一次普通pipeline candidate生成](v6-ddp-fused-pipeline-candidate-20261031.md)：精确原选择、当前source/registry、原完整版本反馈和正常collector材料，由普通角色自行实现；不交付上述终审纠正或手写实现，不自动GPU。预检1469冻结/33输入8,618,008B、路由2980B/supplemental=null，原字段保持。审查精确生成字节、依赖、参数与错误后再决定后续，Qwen/FSDP拒绝/77ac/observer关闭状态保持。以下为本批历史启动记录。

实际已于2026-10-30 20:34:53UTC（香港10月31日04:34:53）启动controller215396，冻结`d7f2f98d55be812cbe5dcbdc4fb0e76d22f7e048`。唯一隐藏plain监听50516已armed，旧51188已handled且进程不存在。local OpenCode真实start/health/stop通过、server_url=null、无模型prompt预检。启动磁盘9,970,700,288B，无GPU作业或清理。

20:36:52[实际启动核验](evidence/v6-ddp-fused-communication-selection-20261031/startup-verification.json) SHA`c0273f4318e1462396f2f5772b8e95b6bead533bd4ba74cf648bb2f6b184cf0d`：generation8e004497fd3f/session ses_e717a062cffeDJvMrMHNl3Am74，30文件7,194,917B与snapshot/普通组装完全相同，1454冻结/实际source current/core/route/prompt/model/tools核对通过。相对前selector999c：15输入相同、12改变、3新增、29移除，非单变量AB；变化源于新接受版本/新discovery与原完整版本反馈。此时19assistant/26tools、controller存活、GPU空闲，df9,953,460,224B约9.27GiB。

早期工具审阅`83f557f33c1097051ec8c372672f69640e1f27c69784646dcbf1e06892dcd707`留服务器：2415read行/2显式截断前缀，8个只读JSON输出重建完全一致。工具7读完整路由判定/measurement，15读当前508a；19实际打开全部四个current collector JSON但仅打印嵌套短摘要，20打印Nsys前6000字符，22打印完整小型校准rank记录，24打印部分timeline和crossrank。不是整个Nsys/timeline已消费，也不是正确推理或最终访问证明。5/10目录输出及23仅命令/状态核对，后续完整工具仍待终审；前26无模型/GPU/collective/私有配置命令，不能推广到后续。私有history SHA`a32c46664cf42713e72eaae1b4154cc03966320c6aaf7f6934062f7e975518b3`留服务器。

早期reader首次把read工具明确截断的project长行当完整行，AssertionError保存在startup-reader-attempt-1；仅修reader为显式前缀核对，未改或重启角色/模型/实验。以下是启动前声明，已由上述实际启动记录补全。

前置发现已终审7cd9fa02…，原样508a版本的新注册表为9calls/8groups，原完整载荷校准、新timeline/Nsys及tuning均可经现有普通输入路径提供。声明一次`ProjectAgents.generate(selection)`，决定新的有界方向；不预先指定group、bucket、算法、实现或collector，不自动生成候选/GPU执行，不重启已关闭链。具体启动与实际输入校验另记，声明不等于已启动。

来源`ddp-fused-communication-evidence-20261031/ddp-evidence` current`10992e0d329f45c941eb2edcd31f60fc1e7f17c7a6bba8241bad93edde56a979`；精确508ad382…/14898B、单fixed-observation-route/emptyparams。新决策不修改来源current、accepted3cd9/formalbbe28c/source1fc/旧accepted4541/independent7a49。原controls/source/adapter/request/prepared只读，不新发现、不复制references，不改变生产core。

输入组装直接执行现有`ProjectSearch._trial`中创建trial之前的只读common前缀，默认group/joint/region/execution_plan、非leaf限定、agent/full反馈；前缀SHA`89e9338af6c3d3fffaa63ea88ae16485e2611b9cea668c1864b5c53a31c67f2c`。使用原selection_prompt及既有反馈后缀、原model/tools，API observe-only无新cap。预检1454冻结身份，30预期文件7,194,917B。源中保留的旧ccd90文件是历史文件，实际installed_bundles只有fixed-observation-route，不把两个文件当共存两个active bundles。

latest-feedback是既有fork_run复制的原始`version-aefa021e172f/result.json`，82,833B/SHA`bbc009e9cdef66582f9360791f47271ea1e0e91d0df943edd71aafee0c117937`，零添加。普通route生成2964B/SHA`735271eb027472417490bfce171392ee48b9258e6248a9cef0bb0ecda6cb96df`，保留原decision/measurement/complete_version，supplemental=null。现有完整版本route不会把pairs全部rank原始检查嵌入这2964B；不能称所有raw质量均进入角色上下文。普通source材料含实际508a、原task/adapter及current四collector原始JSON，未加入人工终审/修正表、旧CPU事实补充或新观察器。历史人工source/metricmapping/CPU事实对当前版本的援助边界仍保留，非unaided或单变量AB。

前置planner确实外读历史计划和控制脚本，其原始collection计划仍作为正常resources的一部分提供，不将有争议推理修成手工答案。后续应核对实际snapshot、完整工具访问与输出、消费与推理，尤其区分读取指针/文件大小与实际证据使用。私有配置/credential不回显；完整history留服务器。此阶段无controller GPU/模型/collective/collector/search/parameters/acceptance，实际工具仍需终审，不能用意图证明无探针。

原GPT2两rank/B1/S128/accum4/effective8seq、FP32/noTF32、full FW/BW/globalmeanSGD、2连续全量质量/精确replicas/2warmup5clean、133.379621→140.04860205约束不变。上次独立53.91019995%是计算合批+通信合并联合案例，与formal24.07796199%及旧ccd9036.1578233%不相加；13→1AR但497,759,232B不变。校准不是下界，NCCLsum不当wire/criticalpath，不跨capture扣clean时间。

空间预检约9.30GiB、GPU空闲，384MiB增长预算加8GiB警戒而非APIcap，无清理/并行GPU。使用一个隐藏plain监听，旧51188已handled，不重启。结果返回后完整审计，仅有证据支持的新方向才能另行声明一次candidate；若没有支持，保留输出并选择具体证据问题，不重试selector直到想要的算法，也不静默结束项目。
