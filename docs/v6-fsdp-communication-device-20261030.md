# 通信优先：当前设备 FSDP 基线准备

?????2026-10-29 23:49:26UTC???[???](evidence/v6-fsdp-communication-device-20261030/space-completed.json)?589?????7,589,275,690B?563,758,300B???SHA/???snapshot payload?????72manifest?3270?????547?????10??current??Qwen??/???????FSDP???????????7,024,779,264B?6.54GiB???15,646,564,352B?14.57GiB??????5.78GB?controls+1GiB???????8GiB????journal SHA `6c6fdbd1d4103c1233c7e860d1221a0a7447c9be21c7291eebb7115e34850f02`????????????????/??/?????????sandbox????????????snapshot loader?????????

??????????15????source-check SHA6f83d6ea????UUID?????devices?????baseline????????????GPU controls??????????????????


[用户通信优先要求](v6-communication-priority-20261030.md)不改变通用项目优化目标。Qwen 必要独立收尾已完成，随后优先原双卡 FSDP 全训练任务；DDP作为下一个通信场景。历史K3确有3.095099%独立整步收益和每rank49→46 AllGather，后续同有效K3的2.157792%不叠加；新K2噪声拒绝与旧DDP通信拒绝保留。不能把DDP计算Graph收益或NCCL kernel sum当通信接受。

本批根 `/root/autodl-tmp/dev/v6-work/fsdp-communication-device-20261030` 只准备新设备基线，不生成候选或搜索。以 `fsdp-search-20261028/fsdp-search` 的原prepared `7231e858`空bundle为源，使用现有 `prepare_variant`，**请求仅修改两张GPU UUID**；历史current `3007940f`的K3不会安装。原生产core `ed24af7a`、source/train.py、adapter `8b014f6f…`、合同/质量、目标和四个通信诊断保持。不存在手选group/K/实现或生产提示修改。

原任务是seed20261028的GPT2Medium24层/hidden1024，两rank各B1/S128，FP32/eager/noTF32/dropout0，完整forward/backward/global-mean-gradient/SGD、tied embedding、实际named DTensor分片；连续两次更新比较完整loss/logits/gradients/updated parameters，atol1e-6/rtol1e-4，保留返回logits、源512B标量owner和两同步边界。不同rank的分片不是相等副本。已审阅adapter的开发者source-owner辅助边界不变，不称无辅助原入口。

当前设备为 `GPU-20e8003c-f531-d50c-084c-a9a97fe94004` 和 `GPU-a24af2e6-99bb-9729-5d14-50fa10a2b321`，旧两UUID已消失。原请求的完整step_ms目标、max-rank-before-median/MAD规则、peak_allocated_bytes≤1.05新baseline不变。必须真实执行新controls及五样本锚点，不能复用旧分数。四个通信诊断各三次独立profile/rank，共24条原始trace；各自重算，不跨采样期相减或当wire/critical-path时间。NODE/P2P-CNS仅硬件能力观察，不是本次NCCL transport证明。

新controls后复用四份只读独立源step/rank张量作CPU完整比较，验证tied及logits+1负例。源/旧controls六文件重哈希，不能复制它们冒充新controls。新controls两文件合计预计5,780,839,642B，此外预留1GiB profile/metadata及8GiB磁盘告警余量；达到预算才允许GPU启动。本批不运行旧源模型、不安装旧K3、不接受优化，源比较失败即保留失败并停止依赖阶段，不修到通过。

源、旧prepared/current和已接受版本保持。基线完成后先审计两rank控制/全部样本/24条trace/独立源关系，再另行声明**一轮普通真实通信任务**：现有selector/candidate自行选择通信或计算通信联合范围、默认参数探索，完整step收益为主，通信原生证据为机制核验。无预设成功；新接受才另做独立整版和必要native证据。之后DDP，不自动回到Qwen链。

空间准备只在全部GPU退出后进行。当前新冷计划包含589个已结束旧会话的重复输入JSON/文本JSON（7,589,275,690B），与72份不可变generation snapshot所指原始store内容逐字节核验；原snapshot/store payload保持在线，3270其他文件、547当前冻结依赖、历史/新接受版本及六份FSDP参考保护。仅冷压缩冗余物化路径，记录压缩/解压SHA与恢复位置；不删内容、不动模型/provider/source/controls。恢复日志位于 `/root/autodl-tmp/dev/storage-audit-20261030/communication-priority/run/RESTORE.md` 和 `cold-json-journal.jsonl`。旧物化路径缺失表示已冷存，可从日志恢复；原snapshot loader仍可直接获得原字节。

所有raw/private配置与历史保留服务器端，仅小审计摘要和代码发布v6。无Goal、模型轮询、投递测试、并行GPU批次或旧控制器/监听重启。新长工作仅一个显式隐藏窗口的完成/磁盘监听；告警不是完成，不终止实验或盲删。
