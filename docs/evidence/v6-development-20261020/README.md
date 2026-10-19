# 开发约定调整与真实请求验证

日期：2026-10-20。环境：box4-2。准备存档：`104e44a`。本次修改允许显式关闭 OpenCode 请求截止时间，删除过时的历史文件依赖测试与执行顺序断言，并调整实施计划。

- [test-receipt.json](test-receipt.json) 与 [pytest.log](pytest.log)：203 passed，28 条 Optuna experimental warnings；显式使用 v6 源码，未运行完整历史套件。
- [initial-test-receipt.json](initial-test-receipt.json) 与 [initial-pytest.log](initial-pytest.log)：初次选择 205 passed、29 errors。29 errors 都发生在旧 manual-task fixture 读取缺失的历史 `contract.json` 时；本次仅要求修改过的入口用例，其余不作为开发门槛。
- [provider-receipt.json](provider-receipt.json)：真实 `zhipuai/glm-5.3` 请求；`request_timeout_s=null`，`idle_abort_frac=0`；51.01 秒，4 条 assistant 消息，2 次 read、1 次 bash、1 次 write，全部完成。
- 会话 API 返回的 assistant 消息按 ID 去重后，报告成本合计 $0.0269474；最终响应单条为 $0.0044788。这是 provider 报告值，不是账单验证。探针没有按费用、token、调用次数或总时长终止的逻辑。
- [assessment.json](assessment.json) 是 Agent 原始分析。探针只给了 `common.py` 和 `resnet_inference.py`，所以其中“缺失 workload/assets”的判断描述的是该分析目录，不能外推为完整验证任务包缺文件。

审阅发现，回复中的增加 batch、改变精度及启用其他后端均为未验证假设。其中增加 batch 不符合固定 workload 比较条件；改变精度必须先满足原数值契约。未接受或执行任何优化候选。由此明确 P2 的请求应携带完整冻结 contract、workload 和可用资产说明，不能只给源码后把生成建议直接当作合法编辑。此要求已进入执行记录；当前探针不代表 Project/Eval Agent、自动分组或项目优化闭环已经实现。

可重放驱动：[probe_no_deadline.py](probe_no_deadline.py)。私有原始目录：`/root/autodl-tmp/dev/v6-work/development-checks-20261020/real-request`。原始消息、sandbox 和 provider 配置不在公开回执中。运行时须指定一个新的输出目录。

旧默认请求截止时间仍为 3600 秒。box4-2 的 v6 部署配置显式设为 null；旧配置备份位于服务器 `configs/box42-v6.before-no-deadline.yaml`。通用框架的完整会话用量账本与受控 GPU helper 仍是 P0 待办。
