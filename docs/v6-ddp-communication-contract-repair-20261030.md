# DDP 通信候选的一次接口修复生成

本批于2026-10-30 14:37:18UTC（香港22:37）以 `batch_finished` 完成，**不是磁盘警告**。终态结论：原三个 Future 反例已修复，但新边界反例仍成立，**77ac 候选不进入 GPU，本次有界修复链关闭**；不手改、不第二次修复到通过，不记为 GPU 数值拒绝。原 accepted/current、Qwen 关闭和 FSDP 拒绝均保持。

[公开终态审计](evidence/v6-ddp-communication-contract-repair-20261030/terminal-audit.json) 对应服务器完整审计 SHA `52a6abc9ed434f1ca1e1f690ef2595de96271774339fe1b23eb042b61c4621e2`，公开文件 SHA `fb023bbe9e3f0da5bbe4d28fdbb08afd56acc80f3c0366d22f5fc272d63996ed`。44初始文件32,032,275B、原40文件逐字保持、914冻结身份、原feedback route/model/tools/生产核心与source/current全部复核。实验增加860B段落和4份材料，仍明确为开发者CPU事实辅助，不是无辅助普通cohort或同prompt A/B。

代码26,479B，SHA `77ac04e8eb86f5b7662d1a96e1152a7c59ea7c6120de5db7a8a2a9ee085b672e`；proposal14,520B，SHA `680b3cd8b661145b047d5542eda6eb0f873f4906ec59c97051555a878876adba`。各写一次、无编辑；空参数/requires空/replaces原单bundle。59 assistant/67工具/0框架schema修复，provider估算3.33825099514，非账单。完整会话、初始写入、CPU测试写入及两次测试编辑留服务器；4,842行read输出和3条截断前缀核对，大输出6/8/9按原反馈纯JSON打印逻辑逐字重建，包含8的完整outputPath。

新代码确实吸收了当前accepted计算Graph：前三microbatch GraphA、第四DDP eager backward、GraphB SGD；7个同名函数AST相同，但整体setup/hook/控制流不同，不能说整份实现逐字等同。`context.references`仍是原始eager adapter，旧bundle被替换；本轮不再依赖“旧Graph仍藏在reference中”的假设。两新NCCL组、奇偶bucket路由及大bucket分半是真实生成方案，尚无该版本原生执行或完整质量证据。

一次独立[CPU合同复核](evidence/v6-ddp-communication-contract-repair-20261030/cpu-contract-result.json) SHA `a394d889157e3ce0e7e17422f89e9ceb72f427fb1dcf19639a8b9c9264e39134` 使用精确代码/原20,000,000元素阈值、真实CPU tensor/Future/嵌套hook；collective为明确的等peer SUM替身，无CUDA初始化/模型/真实collective。旧四个hook序列仅执行一次，预期值改为新结果，增加的检查由实际分支差异支持：

| 检查 | 实际结果 | 判断 |
|---|---|---|
| 正常分半 | 完整20,000,000元素、值4 | CPU正控制 |
| 第二半enqueue抛错 | 完整桶、值4、sequential | 原半桶返回修复 |
| 双半enqueue后get_future抛错 | 两次SUM、值4 | 原整桶重复SUM修复 |
| 异步子Future失败 | 传播异常并降级 | 原吞错修复 |
| 小桶enqueue后get_future抛错 | 再次SUM，4变8，成功返回 | 仍重复计算 |
| 第一半get_future及wait均失败 | 吞掉错误、补半桶后成功返回 | 仍隐藏完成错误 |
| 真实哨兵False且fired=13 | 记anomaly/清哨兵，仍graph_hook/concurrent | 未实际降级 |

最后一项仅将捕获的Graph执行闭包替为计数/返回替身，真实state guard与sentinel执行并记录False；不是历史GPU漂移。Agent原测试fired=0，短路未到sentinel比较，名称不能证明该分支。独立真实BindingSession另验证unsupported CPU路径原reference/same return/once/恢复，不验证CUDA安装或完整训练。以上是条件性接口反例，不是历史NCCL故障或两rank完整质量结果。

完整工具审查保留错误：tool54假Graph碰到未mock的CUDA synchronize；55隔离栈实际尝试CUDA lazy init后报NoCUDA GPUs available，不能说“没有CUDA尝试”。56只修测试同步替身；57管道exit0隐藏返回identity断言失败，58改测试期望，59通过；`assert ... or True`无效检查和sentinel短路范围保留。未观察到真实模型训练/进程组/collective/GPU worker。25错误目录exit2、40隐藏grep无输出、63误把bucket.index当Tensor列表index、66空git log保留。67只删自身cache，最终测试/命令留存，不声称所有覆盖前临时输出仍在。没有观察到后续审计/reviewer答案读取，文件系统非密闭。tool3私有配置仅在服务器作元数据/脱敏审查，本轮未回显凭据。

其他未证明项：scratch6×13顺序+6×14并发+4warmup=166次AllReduce，再1次三int共识，proposal约85次/两int不正确；scratch214,244,352B不包含communicator/allocator，总peak保证不成立。setup先collective后共识、rank局部fallback、Graph部分更新后再eager、全局override恢复False和有限指针guard不是通用异常/并发/owner保证。相邻调用小片段sentinel不是完整参考；没有early SGD。校准硬下界、8–14ms预测和跨profile gap归因未证实；NCCL sum不是wire/critical path。

审计reader首次误把tool8截断显示当完整输出，失败保留；只修reader读取已保存完整outputPath，角色/CPU fixture/模型/GPU均未重跑，见[reader记录](evidence/v6-ddp-communication-contract-repair-20261030/terminal-audit-reader-note.json)。控制器退出、两卡空闲、df约8.30GiB，没有清理。旧28768通知handled且进程退出，没有无任务监听。

下一有界步骤已执行为[真实通信请求只读盘点](v6-communication-request-inventory-20261030.md)，不启动Agent/GPU/collector，不继续修复链。建议下一步为新真实分布式请求的接入预检，已询问新任务或重新判断现有DDP证据缺口的偏好。**当前没有后台实验**；缺少新任务源码/原始目标时不编造负载或重命名重跑旧失败，整体研究也未宣称完成。

以下为已结束的启动与预声明历史。

实际2026-10-30 14:14:31UTC（香港22:14）启动，冻结`f3d245c26fddcf96c497f8819ef62943e2ea9daf`，controller168132，唯一隐藏completion/disk监听28768。旧40444已handled且进程退出，没有重启旧批次。OpenCode本地start/health/stop通过，server_url=null，无模型prompt预检；无清理。实际launch余8,984,240,128B。

14:17:31UTC[启动核验](evidence/v6-ddp-communication-contract-repair-20261030/startup-verification.json) SHA `09d57fc21172fb6fad6b4ef3ebedd4e334ea6d59c129b79ebba80298e4ca32f7`：generation9e73cbd5bad8/session `ses_e72d636caffeTRMjABjohzBoIo`，44初始文件32,032,275B逐字相等，原40文件不变，914冻结身份保持。实际feedback route仍82abd303/supplemental=null，model/tools/core/生产prompt文件不变，实验仅增加860B窄修复段落。source/current不变、无GPU jobs/两卡空闲、盘余8,916,619,264B（约8.30GiB）。

早期12 assistant/16工具，[工具核验](evidence/v6-ddp-communication-contract-repair-20261030/startup-tool-review.json) SHA `135f77688d4ae8cfd178dab6fd288be28a4194931fed16c3d98734cf09c4bc57`。3,312行read输出核验，3条明确截断；tool10实际读取CPU结果、11索引、12前代码890659、13前proposal、16命名CPU fixture。16命令/路径已审查，但bash6/8/9完整输出终态再审；不能由输入交付/读取推出正确修复、最终访问边界或质量。当前16命令无模型/GPU/真实collective执行。私有history SHA `ac1a2746137b3e9a68037f2c198640d2543c7a6d6a3d99bbb5776ad7417fc13b`仅服务器。

另记[输出处理失误](evidence/v6-ddp-communication-contract-repair-20261030/audit-output-handling-note.json)：tool3读取自身私有provider配置，审计者查看时未先过滤并回显凭据；已告知用户并建议轮换。凭据未写入公开JSON/代码/GitHub，未擅自修改配置/密钥。后续仅脱敏/元数据输出；原私有记录留存。本事件独立于实验身份/数值，不隐藏或混作候选错误。

以下为启动前声明。

本步只针对890659f7候选在原字节CPU取证中已证明的接口和异常分支缺陷，使用现有`ProjectAgents.generate(candidate)`一次。不是固定GPU重测、参数调优直到通过、追加reviewer或prompt效果研究。生产core、原task/quality/constraints、source/current、通信优先方向不变，Qwen关闭、FSDP拒绝保持。

原generation413f4674224c的40输入全部逐字保留，包括原selection/原framework反馈route82abd303、已安装ccd90/current accepted discovery和历史失败5e3a。只增加`repair/previous-candidate.py`、`repair/previous-proposal.json`、精确`cpu-observations.json`和小型fixture/result/execution索引。CPU记录来自一次执行，候选代码未改；fixture使用真实BindingSession/原闭包hook/CPU tensor和torch Future，受控collective替身，不是GPU/NCCL/model验证。没有整份终态审计、开发者算法或修复代码。原有source/metric mapping和新CPU事实辅助显式声明，不能称无辅助cohort或单变量AB。

实验prompt增加一段窄修复请求，要求核查实际记录和边界，自主处理接口缺陷、保持原完整任务，不执行GPU/模型/真实collective探针；原生产prompt文件不编辑。模型/工具设置不变，API observe-only，无新caps。正常schema修复、工具失败、初始输出和修改历史都保留。新候选仍先审查，不自动执行、不重复直到通过。

控制器无新selector/搜索/参数执行/collector/native/源或controls重跑/reference复制/指针发布/验收阶段。原accepted4541/current1fc、独立current7a49及历史Graph09b7保持。若以后执行，仍须原2连续完整loss/grads/params/finite/exact replicas、2warmup/5clean、原140.04860205ms约束，通信机制及独立整步收益；本步没有这些结果。

磁盘预估256MiB，依据前批32MB输入和约86MB实际总体增长、此次新增约35KB，无worker或大trace；预测不是API上限。启动前实际df需保留8GiB警戒，GPU空闲；无清理/大文件传输。私有历史/配置/完整输入只在服务器，GitHub仅代码/hash/小型元数据。后台SSH/监听显式隐藏、仅一个新completion/disk watcher；low_disk_space不是completion，不重启旧批次或监听。实际启动和输入核验另记。
