# 新 DDP 原入口验证与 TP 第一 block 诊断的启动证据

协议见[执行文档](../../v6-ddp-semantics-20261027.md)。远端 `/root/autodl-tmp/dev/v6-work/ddp-semantics-20261027`，冻结提交 `a44a6f379c93e6384624b5ed6ef3da23b140b47e`。

- launch/followup-launch 指向同一控制器，记录源码 SHA、命令和真实 df 额度。
- protocol 与 ddp-communication-protocol 固定原入口任务及后续独立检查，未提供旧优化产物。
- 实际 fresh intake 快照记录新提示及相应模型/输入，不代表角色已通过或优化已成功。
- wakeup-setup 为普通守候部署和真实远端 probe；终态将单次唤起当前会话核查。

不归档私有 provider 配置、密钥、权重和大参考张量。此前所有失败、接受和追加诊断均保留原记录，未用此批覆盖。
