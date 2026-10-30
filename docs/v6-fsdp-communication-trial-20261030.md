# 当前设备 FSDP 一轮普通通信优化试验

空间门槛已满足（2026-10-30 00:08:21 UTC）：19个旧文件2,072,388,577→152,991,041 B，压缩与解压SHA、616保护文件/67其他文件/10历史current及新Qwen接受版本全部通过；净释放1,919,410,176 B（1.79GiB），余11,513,827,328 B（10.72GiB）。journal SHA `523a6c2d72bf2e2e35393db38307d98bfd656692cc9ddf70165cc8df93b87e55`。13个旧拒绝trial result与6个完成RUN保持全部原字段/样本/错误，恢复路径见文末；无GPU运行期间清理，无新参考复制。

用户要求保持通用项目优化目标，将后续研究优先转向通信。本批在[新设备完整基线](v6-fsdp-communication-device-20261030.md)审计通过后，声明一轮现有CLI普通真实搜索，DDP随后；不再延长Qwen候选链。工作根为 `/root/autodl-tmp/dev/v6-work/fsdp-communication-trial-20261030`。

从当前空prepared7231e858通过现有fork_run共享新controls/合同/源，重新执行该baseline发现。原历史K3 current3007940f不安装、不改变；没有手选group、K、实现、历史优化bundle或开发者优化答案输入。使用原完整step目标与四项通信诊断、默认scope/参数探索、原生产prompt/model/tools/core `ed24af7a…`；API只观察费用，不增设上限。Agent自行决定通信或计算通信联合范围，不能预先宣称会选择减少通信。

原任务seed20261028/GPT2Medium24层hidden1024，两rank各B1/S128，FP32/eager/noTF32/dropout0，完整FW/BW/global-mean-gradient/SGD、tied embedding、named DTensor实际分片；连续两次完整loss/logits/grads/updated parameters比较，atol1e-6/rtol1e-4，2 warmup/每rank5 clean样本，两同步与返回logits/512B标量owner不变。不同rank分片不是相等副本。adapter8b014f6f曾有开发者源owner修复，本批不是无辅助original-entry。

新设备UUID为20e8003c和a24af2e6两张4090，旧两UUID已失效。新基线实际aligned-rank-max step中位数175.753368 ms，peak2239245312 B，原1.05约束上限2351207577.6 B。新controls已真实执行，两rank各2 reset/1 identity与负例通过；原只读独立源四次完整CPU比较零误差，logits+1/tied=False负例被拒绝，source-check SHA `30db47da3d1cc770408b9a7ba45eedd4df8fdb62aeb54c3fd173348a27c2a07b`。本试验复用该证据与只读controls，不声称再次源比较，不复制5.78GB参考，也不复用旧设备分数。

新发现后由现有collector planner、selector、candidate按普通流程运行一轮。所有实际参数、屏幕检查、配对质量/约束/噪声、回退、失败和慢样本均保留。controller完成/worker passed/quality通过/搜索接受分别报告。若新版本被接受，先审计实际代码与原生通信机制，再另行声明独立整版B/C/C/B；本控制器未安排独立或专门Nsys阶段，真实planner可按既有能力申请collector。没有有效候选则保留失败，不重跑到通过。

四个原通信指标仍分别执行每rank三次独立profile；各自复算、逐样本max rank后median/MAD，不跨metric时期相减。原生kernel/API计数、逻辑负载与实际路径分别记录；NODE/P2P-CNS只说明能力，RING_LL kernel名不独自证明SHM/NET物理transport，NCCL sum不是wire/关键路径。完整质量、原约束、独立整步收益和通信机制证据齐备才记当前期通信相关成功。旧K3 3.095099%与同有效2.157792%不相加，K2噪声拒绝和DDP通信拒绝保持。

启动要求GPU无活跃worker、150GiB额度至少留8GiB告警余量加2GiB试验预算。前一轮新controls之后空间不足以直接启动，故只在worker退出后按清单无损冷存19个旧拒绝trial result/完成RUN JSON，保护616当前依赖与67其他文件，旧接受版本/源/controls/model/provider不动。完整样本和决策按解压SHA保留，恢复根 `storage-audit-20261030/communication-trial/run/RESTORE.md` 与journal；旧journal独立。完成与实际净空间另行记录后才启动。

仅一个显式隐藏窗口的完成/磁盘监听，无并行GPU批次、Goal、模型轮询、投递测试或旧控制器重启。低磁盘不是完成，不停止实验或盲删。raw/private配置、会话和完整产物留服务器，GitHub只发布代码/哈希/小摘要。关闭的提示/解释/验证研究保持关闭，无生产采用或硬编码失败路由。
