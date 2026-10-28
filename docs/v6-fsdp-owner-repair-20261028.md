# FSDP evaluator 源生命周期修复

前置证据见 [FSDP 终态审计](v6-fsdp-readiness-20261028.md)。原入口生成的 adapter 全量数值与独立 serial-global source 一致，但遗漏源 main() 一直持有的 GPU 质量标量；双卡 A/H/H/A 实证两种 allocated 指标均差 512 bytes。

通过已有 `--request ... --feedback ... --prepare-only` 执行一次新的修复 workflow。反馈仅包含原 adapter 路径/SHA、遗漏 owner、原始数值和隔离探针索引，要求保留完整计算/质量/观察结构/同步及原生显存查询，以实际对象所有权恢复源生命周期，不得给 metrics 加常量。生成者可以使用框架原有 controls 反馈修复循环；不因性能分数重复请求，不限制 API 用量。

生产 core 仍为 `ed24af7a…`；intake/prompt、框架执行顺序、接受规则不变。这不是新的无辅助 original-entry cohort，也没有人工编辑 adapter。旧原入口 request/source/adapter/controls/discovery 保持冻结，新目录 `fsdp-owner-repair-20261028/fsdp-repaired` 建立自己的 controls 和参考。

控制器随后以已审阅的原 observation 四字段映射比对两 rank 两步完整源张量，并比较全部五个样本的 allocated/peak 原始值。若新生成结构不同，保留检查错误供审阅，不将 schema error 当数值失败，也不自动发明映射。即使这些数值相等，也需后续审阅实际代码以排除常量修正或其他源实现漂移；本批止于准备与独立检查，不搜索候选、不接受性能版本。

执行脚本 `scripts/experiments/v6_fsdp_owner_repair_20261028.py`。预计新 controls 参考约 5.4 GiB，启动前实际容器剩约 22.22 GiB，只执行这一批，无 Qwen 大任务和本地 raw 下载。旧成功/失败/原始样本均保留，已拒绝的候选不因本批而重新接受。代码/索引/小摘要提交 GitHub v6，私有 provider 配置保持服务器本地。单个普通终态监听完成后唤醒审阅；无 Goal、模型轮询或额外投递测试。
