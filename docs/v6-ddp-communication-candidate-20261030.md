# DDP 通信选择后的一次普通候选生成

实际2026-10-30 13:16:17UTC（香港21:16）启动，冻结`5e04391abdbcb2c36e4330bde35564ffdcb810bd`，controller158256，唯一显式隐藏completion/disk监听40444；旧7584已handled，无重启或清理。OpenCode本地start/health/stop通过、server_url=null，无预检模型prompt。实际launch盘余9,083,072,512B。

13:18:55UTC实际快照核验SHA `2444e4c4e451ffe537878ce5c3195e84269c42e781f139efcd4f0682e79100e3`：generation413f4674224c/session `ses_e730b8f16ffeQoPR0S3pmDkxVN`，40文件32,000,403B逐字相等，899冻结身份保持。原selection cd3f9b53经普通排序为f23d5bef，语义不变；真实candidate route6,265,380B SHA `82abd303d6352d8c9632531a6b5f6571298cac4974bf7c5a79307af66febc92f`，supplemental=null，原prompt/model/tools/core一致。相对前普通candidate12输入变化、1新增、5移除，不是AB。当前source/current/prepared/discovery保持，GPU空闲、无jobs、df余9,003,180,032B（约8.39GiB）。

早期11 assistant/18工具，私有history SHA `0f8e51e97d536fdc964c55f4e7e60b2b097fa2e07bc6097bf00e02ef76d7abb1`；[早期工具审查](evidence/v6-ddp-communication-candidate-20261030/startup-tool-review.json) SHA `3b0758f27402f8463b595a9e1e6fd4c2a855c55bd2e4d5724fc8324969e70dfa`。18条命令/路径已看，4247行read输出核验（4条明确截断），bash12/17/18完整输出与源JSON重建一致：12读完整选择，14读历史失败candidate.py，17读拒绝/配对/原约束，18实际读两rank完整质量/reference路由中的`fallback_setup_failure:RuntimeError`。当前collector JSON在这18条中尚未打开，未见CPU模型/GPU/collective执行；这仅为早期记录，其他bash仅命令/源身份审查，不能替代最终全部访问、推理、代码和探针审计，也没有质量/接受预言。

小型[启动核验](evidence/v6-ddp-communication-candidate-20261030/startup-verification.json)和[预检](evidence/v6-ddp-communication-candidate-20261030/frozen-preflight.json)公开，配置/快照/原始history仅服务器。以下为启动前协议。

已声明的下一步仅为一次现有`ProjectAgents.generate("candidate")`。上一步真实selector自主选了整步和两个AllReduce组，其多通信组并发假设尚无实现；原始失败材料仍包含可自行核查的初始化异常。保留未编辑selection和原始framework反馈，让普通candidate角色完成实现，不向角色添加审计纠正、手写代码、分桶/算法指定或后验事实表。审计发现的错误与预测收益不被采纳为事实。该步骤结束后先审查实际代码、参数、工具与失败，再决定是否进行另行声明的执行，不自动接续无界候选链。

来源为`ddp-communication-selection-20261030`的实际generation2f2d7ab00b70/selection SHA `cd3f9b53abd848b891b70743c6195bf3383733b2d867d2345abab945fdd6a7e7`。当前source仍`ddp-accepted-communication-trial-20261030/ddp-search` current1fc24545，已安装ccd90、单bundle、空参数及真实accepted-code discovery不变；原accepted4541、独立current7a49、历史Graph09b7不变。

输入复用实际selector快照的全部common材料。按现有ProjectSearch普通路径做containment closure、affected_calls排序、targets/target-resources映射以及selected-group反馈路由；不改选择的范围、组、rationale或hypothesis。target-resources对象只使用稳定键顺序，映射与普通API相同。原selection字节与规范化后输入分别记录SHA，不把JSON序列化差异说成原字节相同。历史5e3a为previous-candidate，不是installed代码；原trial02ed5436b4 feedback14,429,015B/a0234920逐字保留，supplemental=null，没有新增事实。普通candidate prompt/model/tools/core不变；与前轮多输入自然变化，不是单变量AB或无辅助cohort。

控制器无GPU/model/collective worker、参数trial、搜索、collector、native、controls/source验证或references复制，无指针发布/验收/生产变更。Agent工具实际是否执行CPU/GPU仍须完整审计，不能只凭控制器意图断言。正常schema修复、工具错误及全部session parts保留，API observe-only无新caps；语法/schema通过不是执行或质量证明。生成结果不手工修复、不重复直到通过。

原完整DDP任务、连续两次full loss/grads/params/finite/exact replicas、1e-6/1e-4、2warmup/5clean、两次同步、state owners、原step约束140.04860205ms均不变；本步骤不会执行这些检查。若以后执行仍要全质量、原约束、真实通信机制和独立整步收益，不能仅凭NCCL sum或计算Graph收益认定通信接受。先前source/metric mapping人工辅助边界仍在。

启动前要求实际GPU空闲，150GiB配额中保留8GiB警戒和384MiB角色材料增长预测；预测不是API cap。本轮无清理，不复制大型reference。冻结上一步886依赖及新selection/snapshot/events/审计身份；私有配置、历史和原始输入仅服务器。只设置一个显式隐藏completion/disk监听，旧7584已handled；low_disk_space不是完成，不停实验或盲删。Qwen链关闭、FSDP拒绝不重跑。实际启动及快照交付另记。
