# 当前设备 DDP 通信任务准备

已实际完成并审计：2026-10-30 01:56:59–01:58:19 UTC，冻结`db863f634e8c79043d9a605d6cd4487a8cc9d55d`/controller98876。两rank新controls、单GPU原source串行全局batch、双rank exact关系负例依次运行；36冻结身份、源码/adapter/entry/core/UUID/请求/旧指针核验通过，当前空baseline7a49、旧Graph current09b7均保持。短批在监听预检前已终态，未建立listener或排队消息。

审计 SHA `b6f94dbb2651438ccd66253559fa516ac6a79ad38ccc1ca1e92c098d0f8b0553`，[小摘要](evidence/v6-ddp-communication-device-20261030/public-terminal-audit.json) SHA `82ba578709fa9f12d881828eb34b6331ffece30ad616ba14a68b461d85816eff`。10个clean rank样本、4 reset完整零误差检查（每次248,879,620元素）、2 identity及覆盖/负例通过。aligned-rank-max样本128.751741/140.158400/131.167867/137.112747/133.379621ms，中位数133.379621/MAD3.733126ms，原step约束上限140.04860205ms；单个140.158400ms样本保留，按原中位数判定。

六条独立通信trace全部复算，均为每rank13个`ncclDevKernel_AllReduce_Sum_f32_RING_LL`；一次最终microbatch的同步阶段包含多个bucket collective，不把no_sync语义说成原生只有一个kernel。通信sum aligned-max三样本51.264631/49.577684/41.154589ms，中位数49.577684ms，不是wire time或整步关键路径。runtime allocated peak1,977,712,640B；本任务原约束是step，不新增FSDP式显存接受条件。

新完整controls共3,982,429,030B实际生成，rank0 SHA `1b1c2ddc768e6d1651dd2575f4506616fd1126a838bab680566c88e9b67ba49f`、rank1 `b1a93181d73466f7c0b75765578ad201e11e1f90cb985ed5e85731d95fdbe80a`。源检查`cbf30d10062ab7a4bce98e8003d0829ed91268e2f4ea0c9685e385bad268611c`：两步/两rank各124,439,808个梯度与更新参数比较全部通过；梯度最大差分别7.45058e-9、6.79865e-8，参数3.72529e-9，平均loss第一步4.76837e-7、第二步0，不能称完全零误差。两次全梯度跨rank检查则exact。源张量未另存，终态读者核对真实执行/代码/结果，不冒充重新从源raw复算全部数值。

关系负例`3f72d15a1a2bf83733ae9a5b39c0e3596b227aae7e0257213a60645be85023d8`：rank1 `transformer.ln_f.bias[0]`实际改变2.500019036e-7，原标量容差内但exact不同，原observe正确Assertion拒绝；rank0不变、比较通过。两worker均正常结束，检测结果与worker status分别审计。一次只读审计脚本缺少空格触发SyntaxError，发生在比较前，留存attempt1后只修reader；无GPU/source/negative重跑。

前置冷存01:45:10–01:47:45UTC：389个旧JSON/trace/冗余输入5,682,348,535→475,816,817B，净释放5,206,204,416B（4.85GiB）；3,283保护文件、724其余文件、14快照manifest/对应immutable payload、10历史current/新Qwen接受及exploration/6完整FSDP参考校验。恢复`storage-audit-20261030/ddp-communication-expanded/run/RESTORE.md`及journal SHA `8fee373067e7ebb087687ad8e04e02df2a686cd35606f6faad75b0501aba2f59`。最初4.22GB清单仅规划未执行，保留于`ddp-communication/plan.json`。旧断电Qwen的关闭证据取自独立recovery终态，原stale state不改写。

准备后GPU空闲、df约9.23GiB。新controls不压缩、不复制；继续[一轮普通DDP通信试验](v6-ddp-communication-trial-20261030.md)前按原授权补足8GiB底线加2GiB预算。以下为冻结准备协议，非新增重复执行。

本批接续[本轮FSDP拒绝审计](v6-fsdp-communication-trial-20261030.md)，按[通信优先计划](v6-communication-priority-20261030.md)转入DDP。仅准备原空baseline的当前设备controls、完整独立源验证与源要求的exact跨rank负例；没有Agent、候选、搜索、参数选择、优化接受或旧Graph安装。根 `/root/autodl-tmp/dev/v6-work/ddp-communication-device-20261030`；启动状态后补，不能把本协议当已运行结果。

源为`ddp-step-followup-20261028/ddp-step-original`，原prepared空版本`7a49c6370b52d881a52550766b279df22684adc16cf3c91be56275f8fe2e1df9`，历史current `09b717664af4cfdce3978a3001b1e69d4f0d7a9fda43cba20a99e6c08653b772`保持。adapter `b276a7ed5524fc0c47d6080a6349a0a65b9e14bda8270cbaf87d6cd81bceb193`原字节，不安装其中历史优化bundle。不是故意低效baseline：原实现已对前三个microbatch使用no_sync，最后一次backward完成累积梯度同步。

请求差异只有两项：旧device UUID迁移至当前`GPU-20e8003c-f531-d50c-084c-a9a97fe94004`、`GPU-a24af2e6-99bb-9729-5d14-50fa10a2b321`；通过既有`diagnostic_metrics`显式将原notes已要求的通信诊断映射到`torch_communication.communication_kernel_sum_ms`，每rank三个独立profile，逐样本取rank最大后median。旧request只有required_observations，后者不产生评估metric样本；该显式映射属于开发者辅助，不称无辅助原入口。目标、工作、原容差、精度、adapter、同步、完成边界及生产core `ed24af7acc8caa15578e0b859c85ef8a901be51354448ed39094606ed9e10a2f`不变，不触发合同Agent改写。

原模型GPT2 12层/hidden768/12head/vocab50257，FP32/eager/noTF32/dropout0/use_cache=False，seed20261020，输入seed31000+rank；两rank各microbatch1/sequence128/累积4，共8序列1024tokens。一次计时包括zero_grad、四次FW/BW（loss/4）、SGD(.001,momentum0,weight_decay0,foreachFalse)、原CUDA同步，完整返回loss/梯度/更新参数。2连续质量、2warmup、每rank5clean，梯度/参数atol1e-6/rtol1e-4及exact跨rank梯度相等不变。DDP为副本关系，不能套用FSDP分片不等关系。

原主指标为aligned max-rank step_ms，再取median/MAD；原step≤1.05×baseline约束以新设备真实controls为锚，不复用旧分数或重设锚。通信sum另行profile，含等待，不是wire time、暴露关键路径或整步成本；NODE/P2P能力不是NCCL transport证明。

顺序固定：现有prepare_variant重新执行双rank完整controls，成功后使用既有`ddp_reference`在一张当前GPU按原source串行处理相同全局8序列，连续两次SGD，逐rank比较完整梯度/更新参数及平均loss，并CPU检查两个rank梯度bitwise相同。旧DDP独立源没有保存可复用的完整tensor，所以此次明确运行新的原source验证，不称旧hash复用。随后仅一次既有exact-rank负例worker：完整DDP step后只改变rank1最后梯度一个元素atol/4，证明标量容差内但违反exact关系时observe能拒绝。它在独立worker、非计时/候选中执行，不改变原controls或质量。任何失败保留并停止后续搜索，不手改或重试直到通过。

准备审计需核对所有新请求/rank结果、2连续检查/identity/reset/负例、全部10clean样本、六条各自通信trace、完整源/关系负例、adapter/entry/core/UUID/原指针及冻结文件。源映射是已审阅的实验辅助，不能推广为任意额外owner证明。通过后才声明一轮普通真实通信优先DDP试验，由Agent自主选范围/参数/实现；不手选group、bucket或算法、不续Qwen/FSDP失败链。新优化接受后才安排独立整版B/C/C/B和必要原生通信证据。

启动预算为新controls预计3,982,429,030B，加1GiB profile/metadata及8GiB告警底线。当前空间不足，先在GPU退出后执行明确新清单的server-only无损冷存，逐文件验证压缩/解压SHA与恢复journal；原始immutable snapshot payload仍在线，当前FSDP证据、DDP源/controls、accepted实现、模型/provider/private history均保护。第一份4.22GB清单仅规划、没有压缩；补充旧已完成trace/已关闭输入副本后重新评估，实际结果另记。无raw下载或旧维护controller重启。

新长批次仅一个隐藏完成/磁盘监听，低磁盘不视为完成，不终止实验或盲删；所有本地helper/SSH/queue显式隐藏，不关闭VSCode终端。只发布代码/哈希/小摘要，raw、私有配置与会话留服务器。
