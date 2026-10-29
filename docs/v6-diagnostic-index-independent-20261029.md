# 索引阅读后两份候选的完整版本独立验证

接续[已审计固定验证](v6-diagnostic-index-read-fixed-20261029.md)，使用既有fork_run与promote_version，各任务仅一次B/C/C/B完整版本验证，publish=False。无新Agent、参数搜索、代码修改、正式搜索接受或production采用；完整版本独立decision单独记录，失败/质量失败/噪声拒绝均不重试。运行完成不是接受，当前版本尚无独立性能结论。

Qwen source是新固定分支，candidate224d820b/codef0656da9、一个joint bundle、空参数；原prepared baseline009b4178。FSDP candidate4fa94d6d/codebeb542b6、一个reshard bundle、实际默认K2；原prepared baseline7231e858。两个source fixed current仍为b61996e1/fa5f081d；更早FSDP已接受K3 current3007940f不导入、不覆盖。两新独立分支current保持各自prepared baseline，无论decision接受与否均不发布。

保持原request/adapter/contract、两连续质量/两warmup/每rank全部5clean样本，原约束/聚合/噪声规则不变。Qwen完整B1/prompt256/32greedy BF16/SDPA/noTF32，完整logits.02/.02/exacttokens/两同步和生命周期，TTFT固定原24.051052×1.05=25.2536046，不重新锚定。FSDP24层1024hidden、2rankB1/S128 FP32/eager/noTF32/dropout0、完整FW/BW/global mean SGD/tied/named shard loss/logits/grads/updatedparams、1e-6/1e-4、512Bowner和返回logits，原peak<=1.05baseline/max-rank后median。不同DTensor shard不是相等replica。

源证据只做身份复核后复用：Qwen旧source-check的两步完整tokens/logits零误差、finite/exact和两负例，原手工generated_tokens/step_logits映射；FSDP原source-check及六个完整source/controls参考、数值/内存owner辅助修复边界。没有本批CPU张量重比较或GPU源重跑；哈希相同证明复用对象不变，不扩展为任意额外owner/观察认证。原controls只读共享，不复制大参考。预检149冻结文件，包括原133、两个source current/prepared/request、源报告和六参考。

顺序：Qwen完整AB/BA，再FSDP完整AB/BA，最后仅在新Qwen独立quality/coverage通过时采一次精确候选Nsys。独立性能即使被拒绝也保持其真实路径采样边界，不用profile评分；没有第二次候选trace或baseline Nsys。旧其他Qwen代码的32Graph trace不能代替当前f065；单次新trace只证明实际路径，不隔离完整加速原因/host成本。FSDP原四diagnostic_metrics每个端点各3独立profile/rank保持，预计96raw trace；无额外FSDP Nsight，不相减跨metric中位数或将NCCL时间称wire/关键路径。

两case相互独立，一个执行失败不冒充另一失败。失败endpoint依既有promote_version停止后续配对，无重试或调参。所有样本/日志/返回检查/条目身份保留，终态需逐项重算质量、max-rank聚合、中位数/MAD/噪声/约束以及actual native rows；验证source/fixed/independent pointers全部保持。

此步骤补齐已生成实现的独立证据，不构成新原入口cohort、诊断闭环通用证明或提示词因果优越性。两candidate均读取索引/部分摘要但未读producer诊断代码，且包含因果越界，bounded prompt study关闭。此前interpreters/verifiers全packet和allocator coalescing仍不采用；历史Qwen/FSDP/DDP接受、TP/ResNet/旧Qwen拒绝与所有样本不改写。

控制器 `scripts/experiments/v6_diagnostic_index_independent_20261029.py`，目录 `dev/v6-work/diagnostic-index-independent-20261029`。实际150GiB盘约10.10GiB余量；依刚完成固定批大小估计本批约1–2GiB级常规结果/通信trace与一份native，串行且无大参考拷贝。先检查df/GPU；无需本轮清理，低于8GiB由唯一普通监听告警，不因告警终止实验或盲删。只在GPU完成后做已授权服务器内无损冷压缩，禁止raw本地下载/旧存档控制器。只推代码、哈希与小型脱敏回执到GitHub v6；无Goal/模型轮询/额外投递测试。
