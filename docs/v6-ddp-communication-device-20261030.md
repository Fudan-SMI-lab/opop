# 当前设备 DDP 通信任务准备

本批接续[本轮FSDP拒绝审计](v6-fsdp-communication-trial-20261030.md)，按[通信优先计划](v6-communication-priority-20261030.md)转入DDP。仅准备原空baseline的当前设备controls、完整独立源验证与源要求的exact跨rank负例；没有Agent、候选、搜索、参数选择、优化接受或旧Graph安装。根 `/root/autodl-tmp/dev/v6-work/ddp-communication-device-20261030`；启动状态后补，不能把本协议当已运行结果。

源为`ddp-step-followup-20261028/ddp-step-original`，原prepared空版本`7a49c6370b52d881a52550766b279df22684adc16cf3c91be56275f8fe2e1df9`，历史current `09b717664af4cfdce3978a3001b1e69d4f0d7a9fda43cba20a99e6c08653b772`保持。adapter `b276a7ed5524fc0c47d6080a6349a0a65b9e14bda8270cbaf87d6cd81bceb193`原字节，不安装其中历史优化bundle。不是故意低效baseline：原实现已对前三个microbatch使用no_sync，最后一次backward完成累积梯度同步。

请求差异只有两项：旧device UUID迁移至当前`GPU-20e8003c-f531-d50c-084c-a9a97fe94004`、`GPU-a24af2e6-99bb-9729-5d14-50fa10a2b321`；通过既有`diagnostic_metrics`显式将原notes已要求的通信诊断映射到`torch_communication.communication_kernel_sum_ms`，每rank三个独立profile，逐样本取rank最大后median。旧request只有required_observations，后者不产生评估metric样本；该显式映射属于开发者辅助，不称无辅助原入口。目标、工作、原容差、精度、adapter、同步、完成边界及生产core `ed24af7acc8caa15578e0b859c85ef8a901be51354448ed39094606ed9e10a2f`不变，不触发合同Agent改写。

原模型GPT2 12层/hidden768/12head/vocab50257，FP32/eager/noTF32/dropout0/use_cache=False，seed20261020，输入seed31000+rank；两rank各microbatch1/sequence128/累积4，共8序列1024tokens。一次计时包括zero_grad、四次FW/BW（loss/4）、SGD(.001,momentum0,weight_decay0,foreachFalse)、原CUDA同步，完整返回loss/梯度/更新参数。2连续质量、2warmup、每rank5clean，梯度/参数atol1e-6/rtol1e-4及exact跨rank梯度相等不变。DDP为副本关系，不能套用FSDP分片不等关系。

原主指标为aligned max-rank step_ms，再取median/MAD；原step≤1.05×baseline约束以新设备真实controls为锚，不复用旧分数或重设锚。通信sum另行profile，含等待，不是wire time、暴露关键路径或整步成本；NODE/P2P能力不是NCCL transport证明。

顺序固定：现有prepare_variant重新执行双rank完整controls，成功后使用既有`ddp_reference`在一张当前GPU按原source串行处理相同全局8序列，连续两次SGD，逐rank比较完整梯度/更新参数及平均loss，并CPU检查两个rank梯度bitwise相同。旧DDP独立源没有保存可复用的完整tensor，所以此次明确运行新的原source验证，不称旧hash复用。随后仅一次既有exact-rank负例worker：完整DDP step后只改变rank1最后梯度一个元素atol/4，证明标量容差内但违反exact关系时observe能拒绝。它在独立worker、非计时/候选中执行，不改变原controls或质量。任何失败保留并停止后续搜索，不手改或重试直到通过。

准备审计需核对所有新请求/rank结果、2连续检查/identity/reset/负例、全部10clean样本、六条各自通信trace、完整源/关系负例、adapter/entry/core/UUID/原指针及冻结文件。源映射是已审阅的实验辅助，不能推广为任意额外owner证明。通过后才声明一轮普通真实通信优先DDP试验，由Agent自主选范围/参数/实现；不手选group、bucket或算法、不续Qwen/FSDP失败链。新优化接受后才安排独立整版B/C/C/B和必要原生通信证据。

启动预算为新controls预计3,982,429,030B，加1GiB profile/metadata及8GiB告警底线。当前空间不足，先在GPU退出后执行明确新清单的server-only无损冷存，逐文件验证压缩/解压SHA与恢复journal；原始immutable snapshot payload仍在线，当前FSDP证据、DDP源/controls、accepted实现、模型/provider/private history均保护。第一份4.22GB清单仅规划、没有压缩；补充旧已完成trace/已关闭输入副本后重新评估，实际结果另记。无raw下载或旧维护controller重启。

新长批次仅一个隐藏完成/磁盘监听，低磁盘不视为完成，不终止实验或盲删；所有本地helper/SSH/queue显式隐藏，不关闭VSCode终端。只发布代码/哈希/小摘要，raw、私有配置与会话留服务器。
