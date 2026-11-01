# v7 提示词版本与一次有界验证

2026-11-01用户授权提示词改进，并要求独立branch/worktree、保留旧文件用于对比和回退。v7从v6提交`42d6d093`建立；本地`D:/Pyhon_projects/opop/v7`，服务器使用独立`/root/autodl-tmp/dev/opop-v7`。v6工作目录和历史提示词不修改。

现有四个角色的`*_prompt.md`保留为v6，默认配置仍使用这些文件。新candidate全文保存在`src/kernel_optimizer/project/prompts/candidate/v7.md`，只增加一段通用实际路径核对要求：依据真实模块属性、参数形状、状态转换判断守卫；CPU检查使用提交文件；区分复制helper、直接调用、真实绑定与GPU验证。没有任务路径、模型名或预期答案，没有新schema、模型设置或接受规则。

通过现有配置覆盖选择，无需换文件：

```sh
kernel-opt --config private.yaml --override project_defaults.prompt_versions.candidate=v7 optimize-project ...
kernel-opt --config private.yaml --override project_defaults.prompt_versions.candidate=v6 optimize-project ...
```

也可在YAML中设置`project_defaults.prompt_versions: {candidate: v7}`。未设置的角色保持v6；不存在的版本报错，不静默降级。普通生成及恢复selection后的新candidate均使用所选版本；已存在的会话保留原会话历史。现有generation snapshot保存实际完整提示词和输入，继续作为对比依据。新增版本使用新文件，已用于实验的版本不得覆盖更新。

主线仍是C1资源证据→C2响应引导改写→C3原任务完整验收。本次只执行一次候选生成对照：重用已保存的候选初始快照、原反馈和选择，换用candidate v7全文，其他角色/模型/工具/输入保持。旧结果作为历史对照，不重启已关闭的两轮任务，不注入审计答案、不手改优化器。一次生成不能隔离随机性或证明跨项目收益；是否执行声明改写及原完整质量/配对性能须据实际结果判断。若有可执行候选，再单独声明一次原评测；不修到通过、不扩成第三轮搜索。

清理已释放约21.81GiB；原始配置、快照及角色历史仍只在服务器。分支、提示词版本管理服务于此次验证，不成为独立的辅助工程终点。

## 实际执行

实现提交`62d19bae89217976896260c967891f778732999c`已推送独立v7分支。服务器14项相关测试通过，涵盖原文保持、显式版本/错误版本、实际普通生成快照及selection恢复后的candidate选择；本地11项通过、3项因缺少optuna未运行，未安装或改动依赖。Windows新工作树的自动CRLF转换已在v7内恢复原LF，四份旧提示词与v6字节一致。服务器GitHub fetch超时未返回后仅终止自身fetch，改传364,726B代码提交bundle建立v7工作树；没有更新旧v6工作树。

香港2026-11-01 **08:53:56**实际启动一次候选生成，controller`425680`，根目录`/root/autodl-tmp/dev/v7-work/candidate-path-selfcheck-20261101`。v7核心`383699d78527de3cdf7034b91d3e13ceb90d66b7d3ed9877abba4ca6c9fe0f5e`；该变化是版本加载与已声明新提示词，不是旧核心字节不变。复用原候选初始快照`259b793b561a`，无新selector、人工算法、审计答案或GPU验证。

08:54:28实际新快照`5ba2d16262ff`、session`ses_e6b66a8c4ffeEyrswmL7waVoK0`：31份输入129,065,077B与原快照逐字节相同，模型及工具设置相同；完整提示词恰为声明的v6→v7文本替换，SHA`52acbfa5f6de597f90de58b5b4022d9479456f5f0a47eef8a9e2bebf2055403b`。源快照/request/current三项未变。这里只是启动身份核对，实际工具访问、输出代码和执行路径尚待返回后检查，不称优化成功。

启动可用23,263,260,672B，生成阶段增长预测512MiB，非API/工具上限；当次无清理/引用复制/新模型运行。实际waiting预检后只启动一个隐藏完成监听`39708`，旧批事件去重不重启。角色返回后据真实构建/分派路径决定一次原质量及配对性能验证，保持失败、无手修或重复生成直到通过。

## 候选完成与原任务验证

09:10:49生成正常结束：63条assistant消息、76次工具调用、0次合同修复，provider估计费用3.26168（非账单）。最终代码`e72f1c7a666f8d941418da6536899a96837beb72298f8739885f03de6d4caa87`，14,795B，一次写入零编辑；proposal `766c8176979532a34b631c83967ce25c660ecc50e46a6ffb6f9ef8409152e12f`，97,644B，一次写入零编辑，空参数/依赖/替换列表。原输入与模型工具的启动检查复用；三个源关键文件终态未变，不重复历史全量哈希。

候选实际读取模型配置，按真实权重形状和可选bias构建，并返回真实DynamicCache。CPU小模型探测导入提交文件并调用build；调度探测是绑定守卫的模拟，不是生产BindingSession或原GPU验证。两次fixture错误（teacher索引和phase未复位）保留，角色只修改fixture；shell管道exit0不掩盖Python错误。所谓空cache“adoption”实际回退，不能作为快路径证据。owned cache回退后再进入快路径的状态一致性仍有条件风险，未追加故障注入或手修。预测收益、跨profile归因及普遍bitwise等说法均未采信。

终态聚焦审查记录`5bbf4a68393d2cf79dacba051785f7653c5c83e9c5caaef42229677b2e0706e0`保存在服务器，复用前71工具/代码检查，仅补末尾5工具与最终proposal重建。所有命令已审，早期输出审查有明确的相关范围，不宣称每行源码都完成全语义分析或文件系统隔离。原始历史仍只在服务器。

**09:18:31已启动一次原任务正式ABBA**，controller `430467`，根目录`/root/autodl-tmp/dev/v7-work/candidate-path-validation-20261101`。通用实验脚本`v7_validate_generated_candidate.py`调用一次现有`validate_bundle(publish=False)`，以原发现版本`4f0bbecd…`作为基线；旧已接受但全回退的`7456056b…`原指针保留。空参数，复用原request/controls，无新selector、candidate、参数搜索、预备fixed gate、源模型重跑或参考复制；正常反馈planner自主选择现有collector。

两臂均用v7核心`383699d7…`、原B8/256→32完整生成、精确tokens/完整logits、原质量及配对噪声规则。启动可用22,996,221,952B，增长预测4GiB依据此前单capture及归一化约501MB并预留正常多采集/角色波动；非API/collector上限，8GiB仍是告警而非启动硬门槛。当次未清理。通过与否均保存，不自动重试或修到通过。

按用户最新节省token要求，等待期间由一个普通完成监听器处理；不再反复模型轮询、读取未完成草稿或重复中间审计。只在实际完成/新异常/用户指令时接续。当前Goal保持active，这一步不是整个主线完成。
