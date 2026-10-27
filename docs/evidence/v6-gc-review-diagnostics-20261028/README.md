# GC / 输入呈现对照证据

主文：[固定协议](../../v6-gc-review-diagnostics-20261028.md)。服务器工作目录 `/root/autodl-tmp/dev/v6-work/gc-review-diagnostics-20261028`。

`preflight.json` 是真实服务器CPU隔离检查与冻结TP数据转换检查，不是性能或质量验收。后续记录启动/普通完成监听；原始GPU receipts、GC事件、所有样本和agent输入快照留服务器，只保存小型公开记录。

`launch.json`记录冻结代码和真实启动；`startup.json`记录首个U质量通过与双卡worker；`wakeup-setup.json`记录唯一新普通完成监听。两次review在GPU阶段之后，本次启动记录不代表已收到其回答。
