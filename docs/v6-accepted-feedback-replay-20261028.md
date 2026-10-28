# 成功反馈的最小真实角色对照

依据[FSDP continuation审计](v6-fsdp-continuation-20261028.md)，第一轮新bundle仍选旧K=3执行策略；新的K=4/5未被该轮3次参数筛选访问。selection提出预取而candidate选择扩大旧策略参数域，后两轮真正的新路径均被噪声判据拒绝。原结果不改变，不再次复测旧候选直到通过。

真实生成前快照为`fsdp-continuation-followup-20261028/fsdp-search/generation-snapshots/generation-914fba16905e.json`。当前提示即便反馈accepted仍命令repair the concrete failure。只改变这一句话：要求先从证据判明前次结果，从当前安装版本继续目标，说明本次具体变化与反馈/选择的关系，避免预设已有失败。原始输入文件、selection、反馈、模型、工具配置、source/core全部不变；不追加FSDP优化建议、审计结论、手写候选或新的diagnostics。

控制臂与处理臂各一个新真实candidate会话，保留现有schema修复流程及所有错误。不设置API/费用上限、不重复直到成功。两臂初次快照逐字检查，仅prompt中声明的单句不同。只验证生成/合同及实际代码与参数空间；本批不执行GPU搜索、质量或性能验收，也不自动采用生产提示。

独立CPU预检查已用原记录分数重放TaskSearch，精确复现seed0、3次筛选K=1/2/3及选择3。它解释采样结果而不产生新性能数据。后续应区分：角色是否理解成功反馈、是否提出具体新实施、参数空间是否能在实际筛选次数内覆盖新值、代码是否仍执行旧路径。当前实验只检查一条提示，不同时修改调参或去重逻辑；不把一个对照结果视为普遍有效。

代码`scripts/experiments/v6_accepted_feedback_replay_20261028.py`，新目录`/root/autodl-tmp/dev/v6-work/accepted-feedback-replay-20261028`。启动前已完成服务器内空间整理，raw仍留服务器；本批不生成大型profile或另一套reference。使用一个普通完成/8GiB空间监听，终态自动审计，无Goal、模型轮询或额外投递测试。
