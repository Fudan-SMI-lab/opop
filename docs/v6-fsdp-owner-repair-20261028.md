# FSDP evaluator 源生命周期修复

前置证据见 [FSDP 终态审计](v6-fsdp-readiness-20261028.md)。原入口生成的 adapter 全量数值与独立 serial-global source 一致，但遗漏源 main() 一直持有的 GPU 质量标量；双卡 A/H/H/A 实证两种 allocated 指标均差 512 bytes。

通过已有 `--request ... --feedback ... --prepare-only` 执行一次新的修复 workflow。反馈仅包含原 adapter 路径/SHA、遗漏 owner、原始数值和隔离探针索引，要求保留完整计算/质量/观察结构/同步及原生显存查询，以实际对象所有权恢复源生命周期，不得给 metrics 加常量。生成者可以使用框架原有 controls 反馈修复循环；不因性能分数重复请求，不限制 API 用量。

生产 core 仍为 `ed24af7a…`；intake/prompt、框架执行顺序、接受规则不变。这不是新的无辅助 original-entry cohort，也没有人工编辑 adapter。旧原入口 request/source/adapter/controls/discovery 保持冻结，新目录 `fsdp-owner-repair-20261028/fsdp-repaired` 建立自己的 controls 和参考。

控制器随后以已审阅的原 observation 四字段映射比对两 rank 两步完整源张量，并比较全部五个样本的 allocated/peak 原始值。若新生成结构不同，保留检查错误供审阅，不将 schema error 当数值失败，也不自动发明映射。即使这些数值相等，也需后续审阅实际代码以排除常量修正或其他源实现漂移；本批止于准备与独立检查，不搜索候选、不接受性能版本。

执行脚本 `scripts/experiments/v6_fsdp_owner_repair_20261028.py`。预计新 controls 参考约 5.4 GiB，启动前实际容器剩约 22.22 GiB，只执行这一批，无 Qwen 大任务和本地 raw 下载。旧成功/失败/原始样本均保留，已拒绝的候选不因本批而重新接受。代码/索引/小摘要提交 GitHub v6，私有 provider 配置保持服务器本地。单个普通终态监听完成后唤醒审阅；无 Goal、模型轮询或额外投递测试。

## 修复终态与采用范围

冻结 `b15397c6`，04:30:04 UTC 启动、04:36:02 完成。设置监听前已经完成，所以没有启动此批监听或额外投递；当前轮直接核查结果。真实 evaluator 会话 `ses_e7f3a37a7ffeijHzQPeJhYnlxC`，初始快照 `generation-610b14ecec55`，零 contract repair。相较原生成初始状态，实际增加 2,022-byte `inputs/previous-feedback.txt`，prompt 增加现有工作流的“读取反馈并修复”句子；其余输入、core、model/tool 配置一致。不是只改一文件的 A/B，也不据此宣称成功率提升。

[实际代码与输入审计](evidence/v6-fsdp-owner-repair-20261028/repair-audit.json)：新 adapter SHA `8b014f6ff6ca1305ee0399eeaa6136653fe3f347f1accbefd83c37bdff67fba6`。代码仅增加 `quality_passed=torch.tensor(1, device=device)`、计时外 MIN all_reduce，以及 `state['quality_passed']` 的存活 owner；将原 before_run 的同步加 CUDA 条件，本机实际 CUDA 路径保持相同。run/reset/observe/原生 metrics 未改，没有给内存值加常量，source/request 未变。标量保留 source 成功质量检查后的状态，框架自身完整质量判断仍独立执行，没有用这个值替代质量验证。

新双 rank controls 全通过；[完整独立源检查](evidence/v6-fsdp-owner-repair-20261028/source-check.json)两步分别全量比较 361,257,089 / 361,255,041 个元素，全部零误差，负例拒绝。每 rank 全部五个 clean 样本的 allocated 和 peak 都与 source 原生值一致：1,462,069,760 / 2,239,245,312 bytes。新代码所有权已人工核查，可以作为后续搜索 baseline。此处采用的是 Agent 生成的本任务 evaluator 产物，生产框架和 prompt 没有修改。

实际 df 余 17,800,138,752 bytes（约 16.58 GiB）。下一批从此 baseline 新建隔离分支，复用只读 controls 引用，避免再生成/复制约 5.4 GiB 参考。执行[三轮自主 FSDP 搜索](v6-fsdp-search-20261028.md)，保留修复辅助与自主搜索的证据边界。
