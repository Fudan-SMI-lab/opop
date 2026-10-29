# 证据阅读重放后候选的原样固定验证

2026-10-29 07:19:57 UTC（香港15:19）启动，冻结 `9b333cd609e7579143321d642ef6ae1293225e2f`，controller978153，生产core不变。见[启动](evidence/v6-diagnostic-index-read-fixed-20261029/launch.json)、[预检](evidence/v6-diagnostic-index-read-fixed-20261029/preflight.json)。07:21:15的[启动身份检查](evidence/v6-diagnostic-index-read-fixed-20261029/startup-verification.json)核对已产生的Qwen B/C及FSDP B请求、133冻结文件、source/branch指针、原adapter/entry/bundle/controls及已完成worker core；当时FSDP B两rank运行，未预设最终结果。

本地唯一普通监听11872已恢复，[正确root预检](evidence/v6-diagnostic-index-read-fixed-20261029/wakeup-setup.json)确认运行后才启动。新批于 **07:22:28 UTC（香港15:22）** 完成，监听07:22:29真实排队完成通知并退出，见[投递回执](evidence/v6-diagnostic-index-read-fixed-20261029/wakeup-result.json)；没有投递测试、重复监听或重启旧批。该通知承接下一轮完整结果审计，当前不存在仍需等待的本批GPU实验。

[终态回执](evidence/v6-diagnostic-index-read-fixed-20261029/terminal-receipt.json)报告四端点worker passed、valid_run=true、原约束通过、冻结文件保持：Qwen B/C TTFT25.229420/21.598306ms均低于25.2536046；FSDP B/C peak均2,239,245,312B，低于2,351,207,577.6。这里只记录控制器报告，尚未在本轮重算全部样本/两步完整质量/48通信trace及最终路径，不能据此宣布正式性能接受或提示因果优势。下一轮按通知审计所有原始结果后决定后续。实际df余10,849,050,624B（约10.10GiB），本轮未清理或转存raw数据。

对[已完成阅读提示实验](v6-diagnostic-index-read-20261029.md)的两份精确生成代码做Qwen B→C、FSDP B→C(K2)各一次。Qwen `f0656da9…` 空参数；FSDP `beb542b6…` 固定实际DEFAULT_KEEP_UNSHARDED_BLOCKS=2及proposal声明默认，choices0/1/2/3/4其余不执行。无手动改代码、参数搜索、新Agent、observer、Nsight、正式性能接受或指针发布。新段落未进生产，本批不评价稳定提示因果优势。

使用已有fork_run/bundle_from_outputs/stage/execute。原空bundle父版本Qwen b61996e1、FSDP fa5f081d；FSDP source历史K3 current3007940f保持，不导入。各source与新分支current始终保持各自原值。原calls/groups/selection/request/contract/adapter/controls逐项核验，共享controls/reference只读，无重复发现或多GB参考复制。相应B不合法则不运行C；其他任务独立，失败保留，不重试到通过。

Qwen adapter498c90f3及最初人工外部generated_tokens/step_logits映射边界保持；完整B1/prompt256/32greedy BF16/SDPA/noTF32、logits .02/.02/exacttokens、两同步和状态生命周期不变。原TTFT anchor24.051052→上限25.2536046。新代码prefill fresh对象但decode预建mirror复用，首层指针与完整shape guard/shared master/capture全局patch/setup直接调用检查均有局限；必须看实际setup成功/回退、完整连续quality和路由，不能移用其他候选旧Graph trace或自检承诺。

FSDP adapter8b014f6f保留实际512B source scalar，仍为developer-assisted owner修复。seeded GPT2Medium24层/hidden1024，2ranks B1/S128、FP32/eager/noTF32/dropout0、全FW/BW/global mean梯度/SGD/tied/named DTensor shard映射；完整两步loss/logits/gradients/updated params，原1e-6/1e-4不放宽。不同shard不是副本。原peak<=1.05baseline、maxrank后median、returned logits/完成边界保持。每rank 2continuous/2warmup/全部5clean样本；原四diagnostic_metrics各3独立profile/rank按request执行，质量失败则可能不运行。每项必须用自己的raw重算，不跨metric减median或把NCCL sum称wire/关键路径。

原evaluator入口不换observer；worker passed可能valid=false，controller结束仅表示尝试完成。所有计时仅固定诊断，不作AB/BA正式接受、独立源复测或旧K3增量提升。若代码通过，也不能认可proposal的唯一cause、无条件显存/生命周期保证。本批只是新精确生成字节的执行检验，未形成新的自主项目cohort或诊断闭环通用证明。

控制器 `scripts/experiments/v6_diagnostic_index_read_fixed_20261029.py`，目录 `dev/v6-work/diagnostic-index-read-fixed-20261029`。启动前真实df约10.74GiB、GPU空闲，预计常规四端点/原FSDP traces，无新大profile或reference。无需本轮清理；raw/private状态留服务器，GitHub仅代码/哈希/小型摘要。新批使用唯一普通完成/8GiB监听，本机重启前的旧监听已直接处理不补发。无Goal、模型轮询或投递测试。
