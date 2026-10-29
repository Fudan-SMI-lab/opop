# 证据阅读重放后候选的原样固定验证

本批已完成原始结果审计，[公开摘要](evidence/v6-diagnostic-index-read-fixed-20261029/public-audit.json)对应服务器完整审计SHA `1eef939109d9751bbf8affe7861a3a6190c98c2526a87047b1328a0c27d67758`。四端点/六rank、30 clean样本、12连续完整质量检查均保留并零误差；48通信raw trace各自重算，133冻结文件、materialized source/adapter/entry/bundle/core、原request、controls、source/branch current核验保持。FSDP六个完整源/controls参考及旧独立源报告另行复核哈希，没有新源模型或CPU张量数值重跑。Qwen原手工外部映射、FSDP512B源owner辅助修复边界保持。

| 固定诊断 | B | C | 限制 |
|---|---:|---:|---|
| Qwen完整生成中位数ms | 738.254567 | 389.882535 | 各5样本，非正式AB/BA |
| Qwen TTFT中位数ms | 25.229420 | 21.598306 | 原固定上限25.2536046，均通过 |
| Qwen生命周期peak B | 8,188,023,808 | 8,311,044,096 | 增加123,020,288B，不是显存收益 |
| FSDP max-rank后median step ms | 178.084178 | 173.265275 | 每rank5样本，非正式接受 |
| FSDP runtime peak B | 2,239,245,312 | 2,239,245,312 | 原上限2,351,207,577.6，通过 |
| FSDP生命周期peak B | 2,264,977,408 | 2,264,977,408 | 无下降 |

Qwen全部5样本仍含B首个762.111310ms和TTFT29.379337ms，未去除尾部；合同约束使用中位数。每次32tokens/31decode、TTFT+decode=full及tokens/s计算复核。C质量覆盖为2prefill+62decode，声明路线分别prefill_graph_replay/graph_replay，无记录fallback；setup7737.692354ms在warm-model评分之外，runtime peak增加40,781,824B。这些是实际质量和遥测，不是该代码的native Graph证明；未借用其他候选trace，也不把无fallback列表当无parent别名调用。代码只在prefill fresh cache，decode预建mirror复用等已审阅局限仍在。

FSDP每份trace的实际NCCL RING_LL：B49 AllGather+25 ReduceScatter，C47+25，两rank一致；C每步委托parent forward，质量阶段每rank2次reference。4指标来自4套各3invocation/rank，合计48trace；max-rank后median B/C分别：sum191.266425/201.720637ms、union193.416933/184.948420ms、overlap36.211170/37.335179ms、without_compute164.879739/148.504799ms。C sum包含689.455370ms、without包含286.926313ms原样本；不以删尾/重试获得通信正例，不相减不同采样窗口，不称wire/暴露关键路径。仅验证原样K2减少2个AG操作和本次固定整步计时，不能叠加历史K3提升。

六个worker日志完整读取：Qwen仅加载进度；FSDP barrier设备提示、loss_type默认提示和profiler开始/结束均非执行失败，没有异常堆栈。辅助只读源报告检查曾把FSDP独立映射摘要当作worker rank结构取status，KeyError保留于服务器audit-helper-notes；纠正读取schema，没有重跑模型/改结果。首次审计898baa58保留，之后只增加六参考身份检查，非重试实验。

决定：固定质量/约束通过，尚无本版本正式性能结论。下一步按[一次整版独立验证](v6-diagnostic-index-independent-20261029.md)分别对精确f065/beb5版本使用原prepared baseline执行B/C/C/B，publish=False；Qwen仅补一次候选native，FSDP沿用原请求中每个端点的四类独立通信采样。不产生新Agent/调参/搜索，不采纳已关闭的阅读提示，不因GPU通过认可其唯一原因等过度推断。df约10.10GiB，GPU空闲，本轮无清理/raw转存；所有历史拒绝/接受保持。

2026-10-29 07:19:57 UTC（香港15:19）启动，冻结 `9b333cd609e7579143321d642ef6ae1293225e2f`，controller978153，生产core不变。见[启动](evidence/v6-diagnostic-index-read-fixed-20261029/launch.json)、[预检](evidence/v6-diagnostic-index-read-fixed-20261029/preflight.json)。07:21:15的[启动身份检查](evidence/v6-diagnostic-index-read-fixed-20261029/startup-verification.json)核对已产生的Qwen B/C及FSDP B请求、133冻结文件、source/branch指针、原adapter/entry/bundle/controls及已完成worker core；当时FSDP B两rank运行，未预设最终结果。

本地唯一普通监听11872已恢复，[正确root预检](evidence/v6-diagnostic-index-read-fixed-20261029/wakeup-setup.json)确认运行后才启动。新批于 **07:22:28 UTC（香港15:22）** 完成，监听07:22:29真实排队完成通知并退出，见[投递回执](evidence/v6-diagnostic-index-read-fixed-20261029/wakeup-result.json)；没有投递测试、重复监听或重启旧批。该通知承接下一轮完整结果审计，当前不存在仍需等待的本批GPU实验。

[终态回执](evidence/v6-diagnostic-index-read-fixed-20261029/terminal-receipt.json)报告四端点worker passed、valid_run=true、原约束通过、冻结文件保持：Qwen B/C TTFT25.229420/21.598306ms均低于25.2536046；FSDP B/C peak均2,239,245,312B，低于2,351,207,577.6。这里只记录控制器报告，尚未在本轮重算全部样本/两步完整质量/48通信trace及最终路径，不能据此宣布正式性能接受或提示因果优势。下一轮按通知审计所有原始结果后决定后续。实际df余10,849,050,624B（约10.10GiB），本轮未清理或转存raw数据。

对[已完成阅读提示实验](v6-diagnostic-index-read-20261029.md)的两份精确生成代码做Qwen B→C、FSDP B→C(K2)各一次。Qwen `f0656da9…` 空参数；FSDP `beb542b6…` 固定实际DEFAULT_KEEP_UNSHARDED_BLOCKS=2及proposal声明默认，choices0/1/2/3/4其余不执行。无手动改代码、参数搜索、新Agent、observer、Nsight、正式性能接受或指针发布。新段落未进生产，本批不评价稳定提示因果优势。

使用已有fork_run/bundle_from_outputs/stage/execute。原空bundle父版本Qwen b61996e1、FSDP fa5f081d；FSDP source历史K3 current3007940f保持，不导入。各source与新分支current始终保持各自原值。原calls/groups/selection/request/contract/adapter/controls逐项核验，共享controls/reference只读，无重复发现或多GB参考复制。相应B不合法则不运行C；其他任务独立，失败保留，不重试到通过。

Qwen adapter498c90f3及最初人工外部generated_tokens/step_logits映射边界保持；完整B1/prompt256/32greedy BF16/SDPA/noTF32、logits .02/.02/exacttokens、两同步和状态生命周期不变。原TTFT anchor24.051052→上限25.2536046。新代码prefill fresh对象但decode预建mirror复用，首层指针与完整shape guard/shared master/capture全局patch/setup直接调用检查均有局限；必须看实际setup成功/回退、完整连续quality和路由，不能移用其他候选旧Graph trace或自检承诺。

FSDP adapter8b014f6f保留实际512B source scalar，仍为developer-assisted owner修复。seeded GPT2Medium24层/hidden1024，2ranks B1/S128、FP32/eager/noTF32/dropout0、全FW/BW/global mean梯度/SGD/tied/named DTensor shard映射；完整两步loss/logits/gradients/updated params，原1e-6/1e-4不放宽。不同shard不是副本。原peak<=1.05baseline、maxrank后median、returned logits/完成边界保持。每rank 2continuous/2warmup/全部5clean样本；原四diagnostic_metrics各3独立profile/rank按request执行，质量失败则可能不运行。每项必须用自己的raw重算，不跨metric减median或把NCCL sum称wire/关键路径。

原evaluator入口不换observer；worker passed可能valid=false，controller结束仅表示尝试完成。所有计时仅固定诊断，不作AB/BA正式接受、独立源复测或旧K3增量提升。若代码通过，也不能认可proposal的唯一cause、无条件显存/生命周期保证。本批只是新精确生成字节的执行检验，未形成新的自主项目cohort或诊断闭环通用证明。

控制器 `scripts/experiments/v6_diagnostic_index_read_fixed_20261029.py`，目录 `dev/v6-work/diagnostic-index-read-fixed-20261029`。启动前真实df约10.74GiB、GPU空闲，预计常规四端点/原FSDP traces，无新大profile或reference。无需本轮清理；raw/private状态留服务器，GitHub仅代码/哈希/小型摘要。新批使用唯一普通完成/8GiB监听，本机重启前的旧监听已直接处理不补发。无Goal、模型轮询或投递测试。
