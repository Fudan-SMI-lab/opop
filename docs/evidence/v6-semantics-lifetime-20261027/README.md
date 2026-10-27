# 语义、生命周期及冻结 evaluator 目标变体证据

协议见[执行文档](../../v6-semantics-lifetime-20261027.md)，远端 `/root/autodl-tmp/dev/v6-work/semantics-lifetime-20261027`，冻结提交 `6cd5c80f637b`。

- 两个 launch 文件均指向同一控制器，记录提交、核心 SHA、argv 和磁盘余量。
- protocol 固定两项独立单段角色对照、TP prefill/权重诊断、三轮 goal variant 及其独立核查顺序。
- intake/evaluator 对照的实际生成前快照用于核实唯一提示差异；初始快照出现不代表结果通过。
- manifest 记录逐字远端导出的路径/长度/SHA；wakeup-setup 是本机普通守候及真实 SSH probe 记录。

不归档私有 provider 配置、权重或完整大张量。所有前批失败、接受及审计结论保持在原目录，结果终态后逐项审计再更新。
