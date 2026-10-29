# 索引阅读后两份候选的完整版本独立验证

后续已于2026-10-29 08:51 UTC从本批独立接受的Qwen e5ad版本启动[两轮普通续行](v6-qwen-accepted-continuation-20261029.md)，原样framework反馈、共享原controls，源current仍009b；仅新分支执行接受后重新发现。本批Qwen接受和FSDP K2噪声拒绝均冻结，不因续行覆盖。启动前仅对明确旧FSDP worker RUN JSON作无损冷压缩，净释放3.62GiB，恢复journal独立保存；本批149依赖与保留版本核验通过。

终态08:03:25 UTC（香港16:03），[完整审计摘要](evidence/v6-diagnostic-index-independent-20261029/public-audit.json)SHA `16c0defb5354e819a32def84d56b34cb343b3641f8bcb4f7d36586bc2268febc`。60 clean rank样本、24完整连续质量检查均保留并零误差，96份FSDP通信trace逐份重算，149冻结文件、8个evaluate请求及1个native请求、实际source/entry/adapter/bundle/core/controls和所有current核验；14份worker日志无traceback。无新Agent、参数调优、源数值重跑、生产采用或指针发布。

| 独立结果 | Qwen f065，空参数 | FSDP beb5，固定K2 |
|---|---:|---:|
| B配对端点中位数ms | 780.318781 / 700.028521 | 185.370158 / 179.720920 |
| C配对端点中位数ms | 389.895496 / 389.874789 | 177.609311 / 180.985418 |
| 汇总B→Cms | 740.173651→389.8851425 | 182.545539→179.2973645 |
| 配对收益ms | 390.423285 / 310.153732 | 7.760847 / -1.264498 |
| 原MAD噪声界ms | 12.997442 / 4.782218 | 9.000480 / 7.672918 |
| 原约束 | TTFT21.640562/21.611258<25.2536046 | peak2,239,245,312<2,351,207,577.6 |
| 独立decision | **接受，47.3251794%降低** | **拒绝：paired_gain_not_above_noise** |

FSDP汇总描述性1.7794%不能写成接受收益，两pair均未通过噪声界，第二pair反向。保留Qwen B首样本829.202506ms、FSDP B209.047534ms/C201.960307ms及全部通信长样本，不重复直到通过。FSDP每份trace仍B49AG+25RS、C47+25 RING_LL；四独立metric窗口不能相减、NCCL不是wire/关键路径。每rank runtime/lifecycle peak仍相同，质量通过不覆盖性能拒绝。

Qwen新精确代码的SQLite SHA `a4826ec6f781bbc5a0f2bce9553db9f6a92b8d067ccbbd31afa488623505d047` 与normalized全部rows逐行相等：68,745 kernels、68,712 Graph kernels、32 cudaGraphLaunch，prefill2,217+31×2,145decode，2,304 Flash全部在Graph中。不是沿用d5bd/其他candidate旧trace；本批无新baseline native，不能据此隔离host成本或完整加速原因。生命周期peak增加123,020,288B，runtime peak增加40,781,824B，不是显存收益。setup在warm-model评分之外，临时全局patch/预建decode mirror/任意owner并发组合仍无通用保证。

Qwen完整版本revision e5ad9a5b…，decision accepted但publication=exploration，independent current仍009b4178；FSDP revision65e1d1f… rejected，current仍7231e858。原fixed currents b61996e1/fa5f081d、更早K3 current3007940f保持。它们来自隔离candidate角色的固定输出，没有本批前置自主搜索接受，不能追认为无人干预原入口成功、稳定prompt因果优势或多bundle共存。

源证据仍只为已有独立检查的哈希复用，Qwen手工外部映射与FSDP512Bowner辅助修复保持。初次只读检查误输出native JSON内嵌kernel明细到工具日志，已中止该SSH/helper；后续全部服务器内重算并仅取小摘要，未启动raw归档下载/旧存档控制器，记录在服务器audit-helper-notes.json。最初本地审计脚本路径猜测不存在、改为实际文件名，不影响实验。

下一步：FSDP本候选拒绝封存，不重测K2直到通过；只对Qwen独立接受的精确版本开展[接受后两轮续行](v6-qwen-accepted-continuation-20261029.md)。原样完整版本latest-feedback交给既有CLI，fresh discovery，不提供人工诊断/组/实现方案；验证接受后反馈与替换/组合是否真实发生，不预设更好候选或多bundle。仅新bundle集合被接受才做新的整版独立AB/BA。当前GPU已退出，df约8.88GiB；先按已授权server-only范围压缩旧FSDP终态worker RUN JSON，保护所有149冻结依赖/当前接受版本/参考/模型，再启动，不转存本地。

已于 **2026-10-29 07:58:05 UTC（香港15:58）** 启动，冻结 `9b2487a5057fc499c9eaf172956aca65ac6e36f6`，controller981096，生产core保持ed24af7a。见[启动](evidence/v6-diagnostic-index-independent-20261029/launch.json)、[预检](evidence/v6-diagnostic-index-independent-20261029/preflight.json)、[源证据复用](evidence/v6-diagnostic-index-independent-20261029/source-evidence-reuse.json)和[准备复用](evidence/v6-diagnostic-index-independent-20261029/preparation-reuse.json)。149冻结哈希预检通过，无新参考复制或source模型运行。

07:58:39[实际启动审计](evidence/v6-diagnostic-index-independent-20261029/startup-verification.json)确认首个Qwen baseline worker981170真实执行；request/source/adapter/entry/controls、原2/2/5配置、source/fixed/independent current和149文件保持。此时仅初始端点，不能提前宣告独立结果。唯一普通完成/8GiB监听42372已运行，[正确root预检回执](evidence/v6-diagnostic-index-independent-20261029/wakeup-setup.json)与CIM确认无重复；上一监听11872已处理/退出。df10,833,186,816B（约10.09GiB），不作本轮清理/raw本地转存。后续由完成/磁盘事件接续，不调用模型等待或额外投递测试。

接续[已审计固定验证](v6-diagnostic-index-read-fixed-20261029.md)，使用既有fork_run与promote_version，各任务仅一次B/C/C/B完整版本验证，publish=False。无新Agent、参数搜索、代码修改、正式搜索接受或production采用；完整版本独立decision单独记录，失败/质量失败/噪声拒绝均不重试。运行完成不是接受，当前版本尚无独立性能结论。

Qwen source是新固定分支，candidate224d820b/codef0656da9、一个joint bundle、空参数；原prepared baseline009b4178。FSDP candidate4fa94d6d/codebeb542b6、一个reshard bundle、实际默认K2；原prepared baseline7231e858。两个source fixed current仍为b61996e1/fa5f081d；更早FSDP已接受K3 current3007940f不导入、不覆盖。两新独立分支current保持各自prepared baseline，无论decision接受与否均不发布。

保持原request/adapter/contract、两连续质量/两warmup/每rank全部5clean样本，原约束/聚合/噪声规则不变。Qwen完整B1/prompt256/32greedy BF16/SDPA/noTF32，完整logits.02/.02/exacttokens/两同步和生命周期，TTFT固定原24.051052×1.05=25.2536046，不重新锚定。FSDP24层1024hidden、2rankB1/S128 FP32/eager/noTF32/dropout0、完整FW/BW/global mean SGD/tied/named shard loss/logits/grads/updatedparams、1e-6/1e-4、512Bowner和返回logits，原peak<=1.05baseline/max-rank后median。不同DTensor shard不是相等replica。

源证据只做身份复核后复用：Qwen旧source-check的两步完整tokens/logits零误差、finite/exact和两负例，原手工generated_tokens/step_logits映射；FSDP原source-check及六个完整source/controls参考、数值/内存owner辅助修复边界。没有本批CPU张量重比较或GPU源重跑；哈希相同证明复用对象不变，不扩展为任意额外owner/观察认证。原controls只读共享，不复制大参考。预检149冻结文件，包括原133、两个source current/prepared/request、源报告和六参考。

顺序：Qwen完整AB/BA，再FSDP完整AB/BA，最后仅在新Qwen独立quality/coverage通过时采一次精确候选Nsys。独立性能即使被拒绝也保持其真实路径采样边界，不用profile评分；没有第二次候选trace或baseline Nsys。旧其他Qwen代码的32Graph trace不能代替当前f065；单次新trace只证明实际路径，不隔离完整加速原因/host成本。FSDP原四diagnostic_metrics每个端点各3独立profile/rank保持，预计96raw trace；无额外FSDP Nsight，不相减跨metric中位数或将NCCL时间称wire/关键路径。

两case相互独立，一个执行失败不冒充另一失败。失败endpoint依既有promote_version停止后续配对，无重试或调参。所有样本/日志/返回检查/条目身份保留，终态需逐项重算质量、max-rank聚合、中位数/MAD/噪声/约束以及actual native rows；验证source/fixed/independent pointers全部保持。

此步骤补齐已生成实现的独立证据，不构成新原入口cohort、诊断闭环通用证明或提示词因果优越性。两candidate均读取索引/部分摘要但未读producer诊断代码，且包含因果越界，bounded prompt study关闭。此前interpreters/verifiers全packet和allocator coalescing仍不采用；历史Qwen/FSDP/DDP接受、TP/ResNet/旧Qwen拒绝与所有样本不改写。

控制器 `scripts/experiments/v6_diagnostic_index_independent_20261029.py`，目录 `dev/v6-work/diagnostic-index-independent-20261029`。实际150GiB盘约10.10GiB余量；依刚完成固定批大小估计本批约1–2GiB级常规结果/通信trace与一份native，串行且无大参考拷贝。先检查df/GPU；无需本轮清理，低于8GiB由唯一普通监听告警，不因告警终止实验或盲删。只在GPU完成后做已授权服务器内无损冷压缩，禁止raw本地下载/旧存档控制器。只推代码、哈希与小型脱敏回执到GitHub v6；无Goal/模型轮询/额外投递测试。
