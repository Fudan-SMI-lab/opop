# 新入口七案例的原项目一致性核验

七个原始完整流程均冻结于 `187617c`，使用原项目、自然语言目标和启动命令。
本目录是开发者独立核验，不是框架自动生成的语义一致性证明，也没有改变原流程输入或候选。

`final-source-audit/result.json` 中七个案例均为 worker 成功且 `result.valid=true`：

- ResNet 两个目标：用原 `infer()` 与原权重、真实图片构造，核验每个 logit 和 top-1。
  两个新 adapter 均保留完整 images fixture；统计查询在 metrics 中。
- Qwen 两个目标：用原 `infer()`、原模型及 prompt 构造，核验完整 greedy tokens 和逐步 logits。
  TTFT 保留 first-token-ready 同步边界。throughput 仍使用外层 elapsed_ms，
  而原脚本使用内部生成计时，因此绝对吞吐的测量边界仍有区别。
- DDP 两个目标：原源码对完全相同的全局 batch 做串行累积，核验连续两次更新的完整梯度、
  参数和平均 loss；另检查两次更新的跨 rank 梯度严格相等。两个新 adapter 均在 run 内清梯度。
- GraphSAGE：原模型构造和原 `infer()` 核验完整输出 tensor。
  随机权重/图的数值一致性不是下游任务准确率。

每个案例的 source 文件和 adapter 哈希见 `source-hashes.json`；所有保存的源文件与 intake
冻结源字节一致。显存绝对值的分配历史差异仍不能仅凭数值输出一致排除。

## 实际发现的未见输入失败

图网络第一次 adapter 路径错误由框架自行反馈并重新生成。重新生成的 adapter 把节点数和
边数写为常量，标准输入通过，但请求 7,168 节点 / 143,360 边时仍输出 `[4096, 64]`，
原脚本输出 `[7168, 64]`。`graph-frozen-heldout-probe/result.json` 的进程状态虽然是
`passed`，内部数值检查为 `valid=false`，不能按进程退出成功当作验证通过。

从该次真实生成前的快照启动独立对照，仅追加一条通用接口说明：在 prepare 中从 request
读取工作负载相关的尺寸和工作量；单次运行内这些值仍冻结。
对照和处理各一个新会话；两者都生成了读取 request 的版本，且标准/未见规模四个 GPU
检查全部通过。生成随机性明显，不能声称这条文字已证明消除该问题。

接口说明已加入后续版本；原始 `187617c` 流程及其失败保留。新版本的完整图网络流程另行
启动、记录，不替换原批次。没有模型名匹配、自动源码修补或失败类型路由。
