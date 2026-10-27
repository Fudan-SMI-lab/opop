# 新 DDP 原入口验证与 TP 第一 block 诊断的启动证据

协议见[执行文档](../../v6-ddp-semantics-20261027.md)。远端 `/root/autodl-tmp/dev/v6-work/ddp-semantics-20261027`，冻结提交 `a44a6f379c93e6384624b5ed6ef3da23b140b47e`。

- launch/followup-launch 指向同一控制器，记录源码 SHA、命令和真实 df 额度。
- protocol 与 ddp-communication-protocol 固定原入口任务及后续独立检查，未提供旧优化产物。
- 实际 fresh intake 快照记录新提示及相应模型/输入，不代表角色已通过或优化已成功。
- wakeup-setup 为普通守候部署和真实远端 probe；终态将单次唤起当前会话核查。

不归档私有 provider 配置、密钥、权重和大参考张量。此前所有失败、接受和追加诊断均保留原记录，未用此批覆盖。

终态已补充：actual intake/request、两版evaluator及understanding、所有fresh角色快照、三轮候选/选择/采集方案、完整控制器和TP结果。`terminal-export-manifest.json`记录远端导出字节SHA；`manual-audit/trial-audit.json`保留端点、反馈输入哈希与原trial路径/SHA；`parameter-native-audit.json`记录参数筛选和原生kernel；`source-mapping-audit.json`为代码审查后的loss名称映射和真实GPU独立两步源码数值审计。原始 `KeyError: loss` 和独立性能拒绝没有覆盖。

`wakeup-delivery.json`确认本次为实际队列消息开启的新轮次。清理另有授权与归档清单；本批完整运行状态作为保留集合，未因磁盘清理改变结果。
