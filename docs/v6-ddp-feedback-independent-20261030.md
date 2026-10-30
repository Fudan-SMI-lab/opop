# DDP 新接受版本的独立整版验证

因[原失败反馈普通试验](v6-ddp-communication-feedback-20261030.md)真实接受新版本，声明一次现有`promote_version(publish=False)`的B/C/C/B，根`/root/autodl-tmp/dev/v6-work/ddp-feedback-independent-20261030`。不是新Agent、搜索、参数探索、固定旧失败重试或生产采用；独立拒绝/错误照实保存，不重跑至通过。

精确来源版本`4541830b6c81c1c0c2c2fef6dfb5f2566e312483900238abf1e7ac900a1758e6`，代码`ccd90dc158788a164d6fda242572530e9cdc9ca50563af1a1d9d597f0ac12e0b`、单bundle/空参数；对原当前设备prepared空baseline `7a49c6370b52d881a52550766b279df22684adc16cf3c91be56275f8fe2e1df9`。新branch current始终保留baseline，源accepted current保留4541，原source空e925及历史Graph不变，不发布指针。

原GPT2 12层hidden768、两rank B1/S128/accum4有效8序列、FP32/eager/noTF32/dropout0、完整FW/BW/global-mean SGD保持；两连续全梯度/参数/loss/finite和exact replica检查、1e-6/1e-4、2warmup/5clean、两同步/owner保持。逐样本先max rank再median/MAD/原配对噪声；step原锚133.379621、上限140.04860205ms不重设，无新显存约束。每端点clean后原3次独立通信profile/rank，若全完成共24 trace，全部样本和尾值保留。

源证据只按精确hash复用此前真实当前设备controls/完整源比较/replica负例：source-check `cbf30d10062ab7a4bce98e8003d0829ed91268e2f4ea0c9685e385bad268611c`，不是新CPU全tensor比较或源/model/control运行。先前梯度/参数/loss有微小容差内差异，不写零误差；手工源/通信metric映射辅助边界、未保存完整源raw的限制保留。共享只读约4GB controls，不复制reference。

本精确代码/当前设备的两份Nsys已完成实际SQLite/Graph/API/NCCL审计，hash复用，不再采baseline或candidate native。每rank4103 kernels/2981 Graph/2 Graph launches、13 eager AllReduce和497,759,232B逻辑payload说明计算调度改变，通信数量/负载未减少。NCCL sum不是wire/关键路径；通过完整step独立验证也不能自动称通信优化成功。搜索39.3258618%与独立结果分列，不跨期相加。

控制器为[scripts/experiments/v6_ddp_feedback_independent_20261030.py](../scripts/experiments/v6_ddp_feedback_independent_20261030.py)，生产core仍`ed24af7acc8caa15578e0b859c85ef8a901be51354448ed39094606ed9e10a2f`。预检483冻结文件、实际两卡UUID20e8003c/a24af2e6、GPU空闲及10,368,204,800B余量通过；8GiB告警上方预留1GiB输出预算，不清理、不并行GPU。实际启动及请求身份另记；新长工作仅一个显式隐藏完成/磁盘监听，旧12400已handled，不重启旧批次或投递测试。

通信优先级保持，Qwen关闭、FSDP拒绝不重测。必要独立收尾之后，依据本轮真实缺口决定下一有界通信步骤，不能将整个研究默认为停止，也不自动无限追加candidate链。raw/private记录留服务器，GitHub仅代码/hash/小摘要；所有后台子进程CREATE_NO_WINDOW。
