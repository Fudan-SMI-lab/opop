# 成功反馈的最小真实角色对照

依据[FSDP continuation审计](v6-fsdp-continuation-20261028.md)，第一轮新bundle仍选旧K=3执行策略；新的K=4/5未被该轮3次参数筛选访问。selection提出预取而candidate选择扩大旧策略参数域，后两轮真正的新路径均被噪声判据拒绝。原结果不改变，不再次复测旧候选直到通过。

真实生成前快照为`fsdp-continuation-followup-20261028/fsdp-search/generation-snapshots/generation-914fba16905e.json`。当前提示即便反馈accepted仍命令repair the concrete failure。只改变这一句话：要求先从证据判明前次结果，从当前安装版本继续目标，说明本次具体变化与反馈/选择的关系，避免预设已有失败。原始输入文件、selection、反馈、模型、工具配置、source/core全部不变；不追加FSDP优化建议、审计结论、手写候选或新的diagnostics。

控制臂与处理臂各一个新真实candidate会话，保留现有schema修复流程及所有错误。不设置API/费用上限、不重复直到成功。两臂初次快照逐字检查，仅prompt中声明的单句不同。只验证生成/合同及实际代码与参数空间；本批不执行GPU搜索、质量或性能验收，也不自动采用生产提示。

独立CPU预检查已用原记录分数重放TaskSearch，精确复现seed0、3次筛选K=1/2/3及选择3。它解释采样结果而不产生新性能数据。后续应区分：角色是否理解成功反馈、是否提出具体新实施、参数空间是否能在实际筛选次数内覆盖新值、代码是否仍执行旧路径。当前实验只检查一条提示，不同时修改调参或去重逻辑；不把一个对照结果视为普遍有效。

代码`scripts/experiments/v6_accepted_feedback_replay_20261028.py`，新目录`/root/autodl-tmp/dev/v6-work/accepted-feedback-replay-20261028`。启动前已完成服务器内空间整理，raw仍留服务器；本批不生成大型profile或另一套reference。使用一个普通完成/8GiB空间监听，终态自动审计，无Goal、模型轮询或额外投递测试。

已冻结`d5b6ce6ed191ab3f41496ea38102b68c0888254f`，12:54:35 UTC启动远端PID676688，PPID1、独立session；生产core仍`ed24af7a…`。真实OpenCode服务健康预检通过，server_url保持null由框架启动，CPU sampler重放通过。控制臂真实会话`ses_e7d6c535effeamprJU7o5DiCDo`及初次快照已生成，source_change=false；当前只确认运行，不预判输出。见[启动](evidence/v6-accepted-feedback-replay-20261028/launch.json)、[协议](evidence/v6-accepted-feedback-replay-20261028/protocol.json)、[真实会话](evidence/v6-accepted-feedback-replay-20261028/startup-verification.json)。

唯一新监听PID13320，状态`.scratch/v6-accepted-feedback-replay-wakeup.json`，启动参数`.scratch/v6-accepted-feedback-replay-wakeup-launch.json`；12:56:36 UTC实际记录running、磁盘约12.11GiB。旧FSDP followup监听已标记handled_in_user_turn并写stop文件，不能恢复。[监听设置](evidence/v6-accepted-feedback-replay-20261028/wakeup-setup.json)保存。仍每30秒普通SSH检查，结束或低空间时才投递一次；本机重启后需按该新批实际状态恢复，不能重复启动服务器实验。

## 终态审计

本批13:20:38 UTC完成，普通监听13:21:03提交终态消息，实际启动本次核查；不是磁盘告警。controller及GPU进程均已退出，实际df余12,905,521,152 bytes（12.02GiB），无需立即清理。[完成文件](evidence/v6-accepted-feedback-replay-20261028/completion.json)、[实际投递](evidence/v6-accepted-feedback-replay-20261028/wakeup-delivery.json)、[逐项审计](evidence/v6-accepted-feedback-replay-20261028/terminal-audit.json)保留。

两臂首个快照与原快照的43份输入逐字相同，模型、工具配置及core完全一致；控制臂prompt原样，处理臂仅替换已声明的一句话。两臂均无schema repair。控制会话`ses_e7d6c535effeamprJU7o5DiCDo`，处理会话`ses_e7d63c934ffe5AfmxpOgaLqqR6`，初始快照分别为generation-22e84c81423f、generation-1c4ed3d2013f；私有工具配置仅发布SHA，不导出凭据。

**这次没有证明处理臂更优。** 两臂均实际读取了选择、旧反馈和旧候选，都明确承认前次accepted；原措辞的控制臂也直接写明“there is no failure to repair”。两者均显式替换原K=3 bundle、重新建立keep-tail政策并实际调用公开set_modules_to_forward_prefetch API，实现本次selection提出的预取；不是只换名称而继续原K=3代码。第一轮历史重复策略结果不能据此归因于单句提示，生成随机性尚未隔离。生产提示仍未采用处理文本。

参数域：control为keep0..4与depth0..2，共14个合法组合；treatment为depth0..2与keep0..3，共12个合法组合。未执行的CPU startup ask预览分别为`(K1,D2)/(K0,D0)/(K4,D0)`及`(K2,D1)/(K3,D1)/(K3,D0)`；没有tell分数，也没有据此宣布候选可行或获益。原3次筛选不足以覆盖全域，但不能简单归因于信息缺失：当前request没有显式parameter_trials，旧反馈却含整数3，处理臂在两次将其误当列表的检查出错后实际读出了该数值。因此本轮不添加重复上下文、不修改TPE或既定参数试验数，更不加入按模型/失败类型硬编码的路由。

控制臂执行了CPU对象替身检查。处理臂使用单rank/Gloo、小型CPU FSDP模型探测API，最初的placeholder parent函数及绕开module hooks分别产生AttributeError、Tensor/DTensor混用错误；带tail的管道exit0没有让这些检查变成成功。Agent随后修正toy probe，并在若干参数点检查loss/梯度及两种回滚路径。初始写入版本、修改与工具输出均保留。这不等同于双4090、24层GPT2Medium、完整两步loss/logits/gradients/updated-parameters源一致性，也不足以证明预测的10–18ms收益或严格内存上界；本次不采纳这些推论。

下一阶段见[真实执行及失败反馈回归协议](v6-feedback-execution-validation-20261028.md)：固定两臂共同合法的K=3、D=0/1，对源已核实的同一父版本做真实双卡完整质量和实际路径核验；独立另做一次历史质量失败输入的中性提示真实角色回归。不会重放成功两臂直到通过，也不将开发固定参数验证包装为新的自主优化或正式性能验收。

13:58 UTC后续执行已通过完整双卡质量/原peak约束，50样本和120份通信trace核验；独立失败反馈会话识别历史质量失败，但仍有未证实的因果解释。两臂有效而无提示优越性证据，生产提示维持，结束该提示对照分支。详见[终态](v6-feedback-execution-validation-20261028.md)，转回同核心目标交叉测量。
