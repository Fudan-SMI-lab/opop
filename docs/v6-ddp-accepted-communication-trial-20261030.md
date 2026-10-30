# DDP 已接受版本重发现后的一轮普通通信续行

声明一轮有界试验，根目录 `/root/autodl-tmp/dev/v6-work/ddp-accepted-communication-trial-20261030`。上一[证据批次](v6-ddp-accepted-communication-evidence-20261030.md)已经刷新已接受执行的registry，并取得实际bucket、tuning、提交时序和校准材料；通信仍13 AllReduce、497,759,232B逻辑输入，计算调度接受不等于通信成功。下一步由现有普通Agent自行判断当前证据支持的范围和候选，不重复旧失败、不承诺一定选到通信组或得到收益，不开启无限候选链。

源为 `ddp-accepted-communication-evidence-20261030/ddp-evidence` 的1fc24545217087adefa1248704619fcdc882a19057336b231d8e02918bf4d0d8；原代码ccd90dc158788a164d6fda242572530e9cdc9ca50563af1a1d9d597f0ac12e0b、单bundle trial-d06aa22e2d-param-a72285d2、空参数、contract保持。原accepted source4541、证据source1fc、独立current7a49和历史Graph09b7保持；隔离新分支可按原接受规则演进。采用existing fork_run和CLI resume rounds1，默认scopes/parameter exploration，API observe-only/no caps。不重复初始GPU discovery/source/controls，不复制大参考。

原框架accepted trial-d06aa22e2d feedback逐字40,982,219B、SHA35bcf4de383882fee05ba7fe66e8afa6dbdd78a54999587872f3c4dbd0808494，零新增bytes、无supplemental。现有route针对新registry执行，历史candidate/原sidecars按普通接口打包；当前discovery的collector材料另按原common inputs交付。没有开发者planner错误表、独立终态表、因果答案或手写实现。现有selector输入本来包含search.installed_bundles/source；必须核验实际snapshot、读取、恢复/替换关系和工具行为，不能以交付代替消费证明。整组输入变化，不是单变量A/B或无辅助原入口。

完整原DDP任务、2连续质量/2warmup/5clean/rank、全梯度/参数/loss/finite/exact replica、1e-6/1e-4、原133.379621→140.04860205ms约束、rankmax先于median/MAD/noise保持；新设备20e8003c/a24af2e6、adapter b276a7ed、原controls只读，生产core ed24af7acc8caa15578e0b859c85ef8a901be51354448ed39094606ed9e10a2f不变。原始source/metric映射辅助边界保持，不新增FSDP式显存约束或跨设备anchor。

不手选组/bucket/算法/collector，不修改生产prompt/sampler/质量/接受，不并行GPU，不自动安排independent/native；真实planner可选已有collector。完成后审计完整工具输入输出、初始代码及编辑、所有参数/worker/raw trace/尾样本。仅真实新接受才声明独立整版B/C/C/B；通信结论另需实际通信机制，不能以Graph或较低NCCL sum替代。拒绝/异常原样保留，不重复到通过。Qwen关闭、FSDP最后拒绝保持。

空间依据：旧普通trial262,836,224B，feedback640,086,016B，本次刚完成证据125,440,000B，均为实际allocated du；现余约9.28GiB。预算1GiB增量、8GiB告警线，属于估算而非API/collector cap；本轮不清理/复制参考。低磁盘不等于完成，不能终止活跃实验或盲删。新批次仅一个隐藏普通completion/disk watcher，旧50252处理后不重启；所有后台/SSH/queue显式CREATE_NO_WINDOW。raw/private configs/history留服务器，代码/hash/小型metadata进GitHub。
