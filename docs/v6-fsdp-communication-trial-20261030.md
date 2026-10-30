# 当前设备 FSDP 一轮普通通信优化试验

## 终态：通信次数下降，整步配对拒绝

2026-10-30 01:11:01 UTC完成，事件为`batch_finished`。trial-4c587f8ea9仅一个空参数候选，完整质量与原显存约束通过，但`paired_gain_not_above_noise`、`objective_not_improved`拒绝。没有新接受bundle、独立重测、参数追加、发布或生产采用；DDP作为下一任务，Qwen链保持关闭。

完整审计 SHA `e4b798cab65826efff5e3a57b2235f93b34a11e9ed56a5193d43b270d6f828c6`，数值审计`63c9da8eb35e5806a0bbfb9bf394fb7e7a221a20f97ab81751c2c73a5ed67464`；[公开小摘要](evidence/v6-fsdp-communication-trial-20261030/public-terminal-audit.json) SHA `245f079b052183e49076bc214c6c38d08346016ca581c4f987f8df8c1f232d16`。18 jobs/36 ranks全部worker passed，9个评估及binding端点共90个clean rank样本、36次完整零误差检查保留；其中正式screen和配对50样本/20检查，binding40/16。全部请求、materialized源/adapter/entry/core/UUID、84冻结身份、source/current和历史K3指针核验。

| 配对 | B中位数 ms | C中位数 ms | B−C ms | 噪声门槛 ms |
| --- | ---: | ---: | ---: | ---: |
| 0 | 176.015316 | 136.227115 | 39.788201 | 15.807892 |
| 1 | 176.066436 | 223.383338 | -47.316902 | 190.558990 |

每样本先取两rank最大再median/MAD。描述性整体176.040876→179.8052265ms不是接受收益。候选screen中428.455136ms、配对中429.403298/441.058846/413.561318/427.127466ms全部保留，没有扣暂停或筛样本。C runtime allocated peak2,134,215,680B，低于新baseline2,239,245,312B和原1.05上限；reserved反而约4.17–4.28GB，高于B约3.68GB，不写成普遍显存改善。setup约33–40ms位于warm score外，仍非零分配/开销证明。

## 实际通信与候选机制

Agent自主选择execution_plan，原selection16组/43直接call ID由既有框架展开至2259依赖call；一个`project.invoke`替换、空参数、无requires/replaces。代码 SHA `02fb050e80f08f74cc5f550274806d2d42a24cf98cd920230af57932d2ea91f1`，12,243B，初始write与8次edit精确重建。候选让全部24 block在backward前保持unsharded，forward后释放本地shard storage，reshard前从all-gather缓冲重建本rank片段。保留原zero_grad/完整forward/backward/SGD，单次策略在finally还原；永久hooks、方法wrapper和cell仍存在于worker生命周期。部分free/copy/record_stream异常可被吞掉，不能据当前通过推断通用回滚、并发或别名owner安全。

216条通信metric原始trace逐一复算；原四metric各3次profile/rank/端点保持，不跨时期相减。Torch profile与4份实际Nsys SQLite均支持每rank AllGather **49→25**、ReduceScatter **25不变**，kernel算法/协议名RING_LL。逻辑collective输入2,733,563,904→2,128,945,152B/rank不是物理wire bytes。Nsys B/C kernel数2115/2043，真实worker NCCL日志出现`SHM/direct/direct`，只限定这些捕获中的channel，不能从NODE拓扑泛化。

Nsys SQLite B rank0/1 SHA `97e915312174a8b1839f739f23a1c16143681e88144c3a6cfa868c3f6482cb90` / `dc3b0a044129780cd74e53bce1812429d9ff53a32d3b96cc0bfab5de8aaee603`；C `8070c481f49c18dcd5f9dcbc5f2c2ad3cbc9499bc328f0479f989ec70989a32a` / `5c38489c60f50ecc906aa3b412d68239d1c3638fb5c17bb97961b86a63c8930d`。不同profile窗口不是clean时间、wire time或暴露关键路径。较少通信已获机制证据，但本轮没有独立整步成功。

两rank timeline raw/clock复算，gap不能从clean wall相减当host成本；400条同步/等待API记录不等于400次显式全卡同步。binding差值−0.058937/+2.273778ms不代表负物理开销或组因果收益。collective_calibration前后均unavailable：只支持full-world SUM allreduce，原任务是AG/RS；没有校准GPUworker。run-only candidate_diagnostics两rank空events，没有observe/compare，不是普遍无fallback证明。遥测为device-wide速率，不作带宽瓶颈判断。

## 真实角色、错误与边界

四角色实际初始snapshot/inputs逐字核验；assistant数量10/33/66/12，工具14/38/90/16，均0框架schema repair；provider估算总4.80784602466，仅观察非账单。完整工具输入输出、长输出索引、CPU脚本、初始代码/edits留服务器。记录访问为本任务packaged输入、安装库与自身临时测试；未观察到旧接受实现、其他case/reviewer/auditor答案读取，外部文件系统非隔离，不能绝对排除未记录访问。

候选实际执行双进程Gloo小GPT2 CPU测试（2层/hidden32/vocab64/seq8/SGD .01），非原双卡完整CUDA质量。初版测试误在连续更新间reset导致loss失败；移除reset后继续。候选缺失私有属性使free guard静默失败，Agent改为getattr。模拟中途异常后立即恢复断言失败，Agent把断言移至两次stock step之后；最终通过说明该toy的后续恢复，不说明立即还原。部分pipeline退出0仍有traceback，全部保留。最终两次连续候选更新及后续stock检查通过，post_backward hook字段仍None。实际完整质量由正式worker建立。

Selector漏affected_calls的自建schema检查与candidate错误路径/import/shape错误均在角色内保留，非框架repair。审计reader首次遗漏binding_ablation端点，失败日志留存后只修读取分类并重新CPU复算raw；此前Nsys list/dict探索reader错误亦保留，未重跑GPU/model/源fixture。候选零setup/零分配、96%通信活跃/74%暴露、12.5GB/s wire-bound及唯一优化方向主张均不采纳。

原adapter的开发者512B源owner修复、当前controls此前完整独立源验证边界保持；本搜索不称新源比较或无辅助原入口。分支current仍空ad259e1f，源7231、历史K3 current3007940f不变。历史K3 3.095099%和同有效2.157792%不相加，后续噪声拒绝全部保持。

以下保留启动协议与历史状态。

启动审计00:16:46 UTC通过，SHA `348a1f2e30c24a3bc9cf1238f4489ffa7962509762af88e624468c988d9c041c`，见[实际启动快照](evidence/v6-fsdp-communication-trial-20261030/startup-verification.json)。84冻结文件、request/prepared/共享controls与新设备原源一致，discover-b47113febf双rank通过；发现附着新空revision ad259e1f，源current7231不变。真实collection-planner generation-d20b253a56cf/session ses_e75d717dbffeVCkcBlecaHG1DM已有8 assistant/12工具，6初始文件267,681 B逐字匹配快照，无历史candidate文件；完整早期私有history服务器留存。此为启动交付/活动核验，未据此声称完整工具访问审计、候选质量或接受。当时GPU已结束发现、df11,436,781,568B约10.65GiB，controller继续。

已于2026-10-30 00:14:12 UTC（香港08:14）实际启动，冻结 `f200b79f0a78b6538ec8cfddad45b1d592f84ab1`，controller75690。冻结预检84个身份、新devices/源比较/原显存上限/空bundle通过，实际OpenCode start/health/stop通过且未发模型测试prompt。00:14:36普通CLI进入新发现，`discover-b47113febf`两rank worker75863/75864在两张当前4090运行；没有旧K3安装。唯一隐藏监听18872于00:14:26建立，旧45956基线通知已处理，不重启。启动df11,504,537,600B约10.71GiB；这是启动状态，不是候选质量或优化接受结论。

空间门槛已满足（2026-10-30 00:08:21 UTC）：19个旧文件2,072,388,577→152,991,041 B，压缩与解压SHA、616保护文件/67其他文件/10历史current及新Qwen接受版本全部通过；净释放1,919,410,176 B（1.79GiB），余11,513,827,328 B（10.72GiB）。journal SHA `523a6c2d72bf2e2e35393db38307d98bfd656692cc9ddf70165cc8df93b87e55`。13个旧拒绝trial result与6个完成RUN保持全部原字段/样本/错误，恢复路径见文末；无GPU运行期间清理，无新参考复制。

用户要求保持通用项目优化目标，将后续研究优先转向通信。本批在[新设备完整基线](v6-fsdp-communication-device-20261030.md)审计通过后，声明一轮现有CLI普通真实搜索，DDP随后；不再延长Qwen候选链。工作根为 `/root/autodl-tmp/dev/v6-work/fsdp-communication-trial-20261030`。

从当前空prepared7231e858通过现有fork_run共享新controls/合同/源，重新执行该baseline发现。原历史K3 current3007940f不安装、不改变；没有手选group、K、实现、历史优化bundle或开发者优化答案输入。使用原完整step目标与四项通信诊断、默认scope/参数探索、原生产prompt/model/tools/core `ed24af7a…`；API只观察费用，不增设上限。Agent自行决定通信或计算通信联合范围，不能预先宣称会选择减少通信。

原任务seed20261028/GPT2Medium24层hidden1024，两rank各B1/S128，FP32/eager/noTF32/dropout0，完整FW/BW/global-mean-gradient/SGD、tied embedding、named DTensor实际分片；连续两次完整loss/logits/grads/updated parameters比较，atol1e-6/rtol1e-4，2 warmup/每rank5 clean样本，两同步与返回logits/512B标量owner不变。不同rank分片不是相等副本。adapter8b014f6f曾有开发者源owner修复，本批不是无辅助original-entry。

新设备UUID为20e8003c和a24af2e6两张4090，旧两UUID已失效。新基线实际aligned-rank-max step中位数175.753368 ms，peak2239245312 B，原1.05约束上限2351207577.6 B。新controls已真实执行，两rank各2 reset/1 identity与负例通过；原只读独立源四次完整CPU比较零误差，logits+1/tied=False负例被拒绝，source-check SHA `30db47da3d1cc770408b9a7ba45eedd4df8fdb62aeb54c3fd173348a27c2a07b`。本试验复用该证据与只读controls，不声称再次源比较，不复制5.78GB参考，也不复用旧设备分数。

新发现后由现有collector planner、selector、candidate按普通流程运行一轮。所有实际参数、屏幕检查、配对质量/约束/噪声、回退、失败和慢样本均保留。controller完成/worker passed/quality通过/搜索接受分别报告。若新版本被接受，先审计实际代码与原生通信机制，再另行声明独立整版B/C/C/B；本控制器未安排独立或专门Nsys阶段，真实planner可按既有能力申请collector。没有有效候选则保留失败，不重跑到通过。

四个原通信指标仍分别执行每rank三次独立profile；各自复算、逐样本max rank后median/MAD，不跨metric时期相减。原生kernel/API计数、逻辑负载与实际路径分别记录；NODE/P2P-CNS只说明能力，RING_LL kernel名不独自证明SHM/NET物理transport，NCCL sum不是wire/关键路径。完整质量、原约束、独立整步收益和通信机制证据齐备才记当前期通信相关成功。旧K3 3.095099%与同有效2.157792%不相加，K2噪声拒绝和DDP通信拒绝保持。

启动要求GPU无活跃worker、150GiB额度至少留8GiB告警余量加2GiB试验预算。前一轮新controls之后空间不足以直接启动，故只在worker退出后按清单无损冷存19个旧拒绝trial result/完成RUN JSON，保护616当前依赖与67其他文件，旧接受版本/源/controls/model/provider不动。完整样本和决策按解压SHA保留，恢复根 `storage-audit-20261030/communication-trial/run/RESTORE.md` 与journal；旧journal独立。完成与实际净空间另行记录后才启动。

仅一个显式隐藏窗口的完成/磁盘监听，无并行GPU批次、Goal、模型轮询、投递测试或旧控制器重启。低磁盘不是完成，不停止实验或盲删。raw/private配置、会话和完整产物留服务器，GitHub只发布代码/哈希/小摘要。关闭的提示/解释/验证研究保持关闭，无生产采用或硬编码失败路由。
