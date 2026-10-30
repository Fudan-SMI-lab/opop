# DDP 新接受版本的独立整版验证

独立接受结论保持。后续[已接受版本正式发现与通信诊断](v6-ddp-accepted-communication-evidence-20261030.md)已终态审计（51dfcb11），更新隔离registry而未变代码/单bundle/contract；仍13 AllReduce/相同payload，不能把36.1578233%计算调度收益改称通信接受。已声明[一轮基于新registry的普通续行](v6-ddp-accepted-communication-trial-20261030.md)，无独立重跑或自动无界链。以下为独立批次原记录。

迟到通知去重（香港19:46）：独立批次11:34:33UTC的完成事件在其完整审计及后续证据批次启动后才送达。重新核对terminal hash、审计9d85463c及source/independent指针一致，按已处理事件去重；没有重跑模型、样本或旧监听。当前controller142528/唯一隐藏监听50252仍运行，GPU暂闲、实际盘余9,988,984,832B。本轮同时修复上述新增中文段落经PowerShell管道编码变成问号的问题；原实验/审计JSON/冻结输入不变，错误文档版本保留在git历史。

终态11:34:33 UTC（香港19:34）完成且已完整复核，独立接受探索版本`fd66b05491e6356e7d08cb402f1ddd4a1566314c44b894d4f7cb3ef928df28f0`，publish=False。整步112.6878425→71.9423715ms（36.1578233%）；两组B112.694293/112.681392、C71.120617/72.764126，收益41.573676/39.917266超过噪声3.941248/6.569288，原140.04860205约束通过。所有40个rank样本、16次零误差完整检查、24条各自period通信trace与483冻结身份通过；123.521373/76.954514ms等尾样本保留。原source接受4541、独立current7a49不变，搜索39.3258618%另列。

候选每rank首个质量调用parent、第二个graph setup/capture，原两连续完整检查保持；不是两次均优化路径。各trace仍13 AllReduce RING_LL；两个C通信profile rank-max中位35.150752/40.757182ms，B38.301596/46.109457ms，分属不同采样窗口，不能相减解释clean/wire成本。精确旧native两SQLite重核实际4103/2981 Graph kernels、2版本化GraphLaunch和13个非Graph NCCL，没有新native运行。显存与搜索配对一致，lifecycle reserved较B高，不新增显存接受结论。原源比较/负例仅hash复用，不是新源运行。

完整审计SHA `9d85463cf952cf3dce386d9b39e604be293fb141a861cc0a2f94cd51d6958c38`，[公开小摘要](evidence/v6-ddp-feedback-independent-20261030/public-terminal-audit.json) SHA `f053f3090b317e7796d7c31b71dcd593347f451368cb31ff70db5992c5a6fc97`。一次只读审计通过，worker日志仅原DDP/loss/profiler提示。仍归为计算调度收益，不能冒充通信量减少或无辅助原入口；生产不采用。独立监听53536已完成/handled，不重启。

下一有界步骤为[已接受版本正式发现与通信诊断规划](v6-ddp-accepted-communication-evidence-20261030.md)：实际旧discovery仍e925空baseline，而当前4541已改变执行；只刷新隔离分支registry/plan并交既有collection-planner决定诊断，暂不生成候选或搜索。必要收尾完成不等于通信研究结束，也不自动接无限候选链。以下启动记录为历史。

实际启动2026-10-30 11:32:42 UTC（香港19:32），冻结`49aa288fccec091e2ecfd25090a5eeb8408034ca`，controller140320，唯一隐藏监听53536；旧12400已退出/handled。11:33:54启动核验SHA `883e55712c1864761c945c70e2280c38bddf43672823c4ccd3694e141481d84a`：483冻结文件、source/current/原request/prepared、三个已生成B/C/C请求的entry/adapter/UUID/controls一致，实际两rank GPU worker141122/141123运行，磁盘余10,206,941,184B。当时未作独立接受预言；终态另审全部四端点/样本/质量/24通信trace。没有新Agent或初始discovery。

因[原失败反馈普通试验](v6-ddp-communication-feedback-20261030.md)真实接受新版本，声明一次现有`promote_version(publish=False)`的B/C/C/B，根`/root/autodl-tmp/dev/v6-work/ddp-feedback-independent-20261030`。不是新Agent、搜索、参数探索、固定旧失败重试或生产采用；独立拒绝/错误照实保存，不重跑至通过。

精确来源版本`4541830b6c81c1c0c2c2fef6dfb5f2566e312483900238abf1e7ac900a1758e6`，代码`ccd90dc158788a164d6fda242572530e9cdc9ca50563af1a1d9d597f0ac12e0b`、单bundle/空参数；对原当前设备prepared空baseline `7a49c6370b52d881a52550766b279df22684adc16cf3c91be56275f8fe2e1df9`。新branch current始终保留baseline，源accepted current保留4541，原source空e925及历史Graph不变，不发布指针。

原GPT2 12层hidden768、两rank B1/S128/accum4有效8序列、FP32/eager/noTF32/dropout0、完整FW/BW/global-mean SGD保持；两连续全梯度/参数/loss/finite和exact replica检查、1e-6/1e-4、2warmup/5clean、两同步/owner保持。逐样本先max rank再median/MAD/原配对噪声；step原锚133.379621、上限140.04860205ms不重设，无新显存约束。每端点clean后原3次独立通信profile/rank，若全完成共24 trace，全部样本和尾值保留。

源证据只按精确hash复用此前真实当前设备controls/完整源比较/replica负例：source-check `cbf30d10062ab7a4bce98e8003d0829ed91268e2f4ea0c9685e385bad268611c`，不是新CPU全tensor比较或源/model/control运行。先前梯度/参数/loss有微小容差内差异，不写零误差；手工源/通信metric映射辅助边界、未保存完整源raw的限制保留。共享只读约4GB controls，不复制reference。

本精确代码/当前设备的两份Nsys已完成实际SQLite/Graph/API/NCCL审计，hash复用，不再采baseline或candidate native。每rank4103 kernels/2981 Graph/2 Graph launches、13 eager AllReduce和497,759,232B逻辑payload说明计算调度改变，通信数量/负载未减少。NCCL sum不是wire/关键路径；通过完整step独立验证也不能自动称通信优化成功。搜索39.3258618%与独立结果分列，不跨期相加。

控制器为[scripts/experiments/v6_ddp_feedback_independent_20261030.py](../scripts/experiments/v6_ddp_feedback_independent_20261030.py)，生产core仍`ed24af7acc8caa15578e0b859c85ef8a901be51354448ed39094606ed9e10a2f`。预检483冻结文件、实际两卡UUID20e8003c/a24af2e6、GPU空闲及10,368,204,800B余量通过；8GiB告警上方预留1GiB输出预算，不清理、不并行GPU。实际启动及请求身份另记；新长工作仅一个显式隐藏完成/磁盘监听，旧12400已handled，不重启旧批次或投递测试。

通信优先级保持，Qwen关闭、FSDP拒绝不重测。必要独立收尾之后，依据本轮真实缺口决定下一有界通信步骤，不能将整个研究默认为停止，也不自动无限追加candidate链。raw/private记录留服务器，GitHub仅代码/hash/小摘要；所有后台子进程CREATE_NO_WINDOW。
