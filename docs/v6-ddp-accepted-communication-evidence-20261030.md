# 已接受 DDP 版本的正式发现与通信诊断规划

????2026-10-30 11:43:03 UTC???19:43????`21bcce8615d27c89afee7466dce33be2856a0cfb`?controller142528???????50252??53536???/handled?????OpenCode start/health/stop????????prompt?11:44:10????SHA `ba0fd682282213b90807f0472f93a7051a703119a7e6fea64f93c29d876e1217`?548???????discover-27942843ca?rank?????branch?registry??`1fc24545217087adefa1248704619fcdc882a19057336b231d8e02918bf4d0d8`?????/bundle/contract?source current4541???

??collection-planner generation-fa06929739bd/session ses_e7360b60effeKmiOf83y9r5NRJ????6????262,792B????snapshot?9 assistant/11?????history????????????planner generation-fb697d303320??prompt/model/tools????????request/actual observations/coverage/available collectors/schema/worker environment????????????????/?????????????collector???GPU?????????????????/??/??collector?????worker????????planner???????9,988,993,024B?9.30GiB?

[独立整版验证](v6-ddp-feedback-independent-20261030.md)已接受36.1578233%完整step收益，原搜索39.3258618%另列。但实际通信仍13 eager AllReduce/497,759,232B逻辑负载，当前证据支持计算Graph调度，通信优化成功缺口未闭合。实际检查发现source `ddp-communication-feedback-20261030/ddp-search/discovery.json`仍引用旧空baseline e925，而current为新接受4541；此前post-evaluation Torch profile没有更新其registry。不能据旧可优化范围继续候选链。

因此声明一个有界证据步骤，根`/root/autodl-tmp/dev/v6-work/ddp-accepted-communication-evidence-20261030`：从精确接受版本4541830b分叉，仅调用一次现有`ProjectSearch.discover()`，真实profile后由既有collection-planner按原输入/模型/工具/提示选择既有collectors。此API正常将新plan/registry附在隔离分支；源码、bundle、contract逐字不变，source accepted current4541和独立current7a49保持。**不运行selector、candidate、trial、参数探索或新独立验证**，不手选组/bucket/算法/collector，不改生产prompt/sampler/acceptance。诊断结果只决定下一证据步骤，不自动追加candidate或无限链。

精确代码`ccd90dc158788a164d6fda242572530e9cdc9ca50563af1a1d9d597f0ac12e0b`、单bundle `trial-d06aa22e2d-param-a72285d2`、空参数；不导入旧历史Graph。原current-device任务/request/adapter/controls和完整工作/精度/owner保持，原anchor133.379621/limit140.04860205ms不重设。共享只读约4GB controls，没有源/model/control重跑或参考复制。既有manual源/metric映射辅助与当前代码对象/回退/全局flag限制保持。

collection-planner只收到框架原request、available collectors、此次真实communication observations、coverage、schema及自动worker environment；没有开发者终态事实表、独立接受结论、手写通信方案或reviewer答案注入。必须审计实际快照/读取/全部工具/正常schema repairs，外部FS不是隔离；禁止凭输入交付宣称正确消费。planner本来只规划不自行跑工具实验，其实际行为仍须核查。它选择的collector可以启动真实worker，不能称整个步骤无GPU活动。没有人为强制Nsys或“必须减少AllReduce”的实现答案。

profile/collector完成不是新的完整GPU质量或性能接受；run-only没有observe/compare。新registry lineage只说明调用身份/已装bundle包含关系，不证明多个active bundles、独立组归因或唯一机制。通信trace逐条自己的时间窗口复算，真实kernel/API/channel和逻辑payload与物理wire/关键路径分开。不能把不同profile减去clean wall，也不能从NODE/P2P能力推实际transport。当前完整质量由已审计版本独立验证支持，不能推广为任意owner或并发安全。

控制器[scripts/experiments/v6_ddp_accepted_communication_evidence_20261030.py](../scripts/experiments/v6_ddp_accepted_communication_evidence_20261030.py)只编排既有API，core保持`ed24af7acc8caa15578e0b859c85ef8a901be51354448ed39094606ed9e10a2f`。预检548冻结文件/两卡空闲/10,058,711,040B余量通过。实际先前同代码profile占45,445,120B、六诊断合计39,698,432B；给新discovery/planner/collectors预留1GiB并保留8GiB告警线。没有清理或并行GPU，不根据完成标签预设collector成功。

实际launch/startup另记，长步骤仅一个隐藏完成/磁盘监听。旧独立53536已handled，旧批次不重启；所有后台/SSH/queue显式CREATE_NO_WINDOW，不关闭VSCode终端。raw/private configs/snapshot/history/trace留服务器，GitHub仅代码/hash/小摘要。若低空间先查实际阶段/GPU，不终止实验或盲删；按既有授权只在GPU退出后处理明确冷范围并保存恢复journal。

通信优先、通用优化目标不变。Qwen链关闭，FSDP最后拒绝不重测，既有DDP失败保留。此步骤使研究继续获取当前已接受执行的有效证据，既不把一个批次收尾当作全项目停止，也不以重复搜索到通过替代证据。
