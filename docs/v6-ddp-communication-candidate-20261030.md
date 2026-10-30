# DDP 通信选择后的一次普通候选生成

已声明的下一步仅为一次现有`ProjectAgents.generate("candidate")`。上一步真实selector自主选了整步和两个AllReduce组，其多通信组并发假设尚无实现；原始失败材料仍包含可自行核查的初始化异常。保留未编辑selection和原始framework反馈，让普通candidate角色完成实现，不向角色添加审计纠正、手写代码、分桶/算法指定或后验事实表。审计发现的错误与预测收益不被采纳为事实。该步骤结束后先审查实际代码、参数、工具与失败，再决定是否进行另行声明的执行，不自动接续无界候选链。

来源为`ddp-communication-selection-20261030`的实际generation2f2d7ab00b70/selection SHA `cd3f9b53abd848b891b70743c6195bf3383733b2d867d2345abab945fdd6a7e7`。当前source仍`ddp-accepted-communication-trial-20261030/ddp-search` current1fc24545，已安装ccd90、单bundle、空参数及真实accepted-code discovery不变；原accepted4541、独立current7a49、历史Graph09b7不变。

输入复用实际selector快照的全部common材料。按现有ProjectSearch普通路径做containment closure、affected_calls排序、targets/target-resources映射以及selected-group反馈路由；不改选择的范围、组、rationale或hypothesis。target-resources对象只使用稳定键顺序，映射与普通API相同。原selection字节与规范化后输入分别记录SHA，不把JSON序列化差异说成原字节相同。历史5e3a为previous-candidate，不是installed代码；原trial02ed5436b4 feedback14,429,015B/a0234920逐字保留，supplemental=null，没有新增事实。普通candidate prompt/model/tools/core不变；与前轮多输入自然变化，不是单变量AB或无辅助cohort。

控制器无GPU/model/collective worker、参数trial、搜索、collector、native、controls/source验证或references复制，无指针发布/验收/生产变更。Agent工具实际是否执行CPU/GPU仍须完整审计，不能只凭控制器意图断言。正常schema修复、工具错误及全部session parts保留，API observe-only无新caps；语法/schema通过不是执行或质量证明。生成结果不手工修复、不重复直到通过。

原完整DDP任务、连续两次full loss/grads/params/finite/exact replicas、1e-6/1e-4、2warmup/5clean、两次同步、state owners、原step约束140.04860205ms均不变；本步骤不会执行这些检查。若以后执行仍要全质量、原约束、真实通信机制和独立整步收益，不能仅凭NCCL sum或计算Graph收益认定通信接受。先前source/metric mapping人工辅助边界仍在。

启动前要求实际GPU空闲，150GiB配额中保留8GiB警戒和384MiB角色材料增长预测；预测不是API cap。本轮无清理，不复制大型reference。冻结上一步886依赖及新selection/snapshot/events/审计身份；私有配置、历史和原始输入仅服务器。只设置一个显式隐藏completion/disk监听，旧7584已handled；low_disk_space不是完成，不停实验或盲删。Qwen链关闭、FSDP拒绝不重跑。实际启动及快照交付另记。
