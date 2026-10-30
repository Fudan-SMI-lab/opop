# DDP 通信候选的一次接口修复生成

本步只针对890659f7候选在原字节CPU取证中已证明的接口和异常分支缺陷，使用现有`ProjectAgents.generate(candidate)`一次。不是固定GPU重测、参数调优直到通过、追加reviewer或prompt效果研究。生产core、原task/quality/constraints、source/current、通信优先方向不变，Qwen关闭、FSDP拒绝保持。

原generation413f4674224c的40输入全部逐字保留，包括原selection/原framework反馈route82abd303、已安装ccd90/current accepted discovery和历史失败5e3a。只增加`repair/previous-candidate.py`、`repair/previous-proposal.json`、精确`cpu-observations.json`和小型fixture/result/execution索引。CPU记录来自一次执行，候选代码未改；fixture使用真实BindingSession/原闭包hook/CPU tensor和torch Future，受控collective替身，不是GPU/NCCL/model验证。没有整份终态审计、开发者算法或修复代码。原有source/metric mapping和新CPU事实辅助显式声明，不能称无辅助cohort或单变量AB。

实验prompt增加一段窄修复请求，要求核查实际记录和边界，自主处理接口缺陷、保持原完整任务，不执行GPU/模型/真实collective探针；原生产prompt文件不编辑。模型/工具设置不变，API observe-only，无新caps。正常schema修复、工具失败、初始输出和修改历史都保留。新候选仍先审查，不自动执行、不重复直到通过。

控制器无新selector/搜索/参数执行/collector/native/源或controls重跑/reference复制/指针发布/验收阶段。原accepted4541/current1fc、独立current7a49及历史Graph09b7保持。若以后执行，仍须原2连续完整loss/grads/params/finite/exact replicas、2warmup/5clean、原140.04860205ms约束，通信机制及独立整步收益；本步没有这些结果。

磁盘预估256MiB，依据前批32MB输入和约86MB实际总体增长、此次新增约35KB，无worker或大trace；预测不是API上限。启动前实际df需保留8GiB警戒，GPU空闲；无清理/大文件传输。私有历史/配置/完整输入只在服务器，GitHub仅代码/hash/小型元数据。后台SSH/监听显式隐藏、仅一个新completion/disk watcher；low_disk_space不是completion，不重启旧批次或监听。实际启动和输入核验另记。
