# DDP 已接受版本重发现后的一轮普通通信续行

实际启动：2026-10-30 12:03:42 UTC（香港20:03），冻结 `2a0bb13a7ab8f46b27c58d77f3c38f81521f5a1a`，controller145514，唯一隐藏监听13432；旧50252已退出并handled。真实local OpenCode start/health/stop通过，server_url=null，无模型测试prompt。首次部署launcher错误地校验旧脚本名，已在创建新root/controller/session/model/watcher前中止；`launch-preflight-attempt-1.json`及原脚本保留，随后只修正部署路径，精确冻结代码/生产core未变，没有实验重启。

12:05:30 UTC启动审计SHA `96f1586200fe20d45d99f5e6e68b504a8af906fa8a7fd15ddbf6b4686538f5bd`：647冻结文件、source/branch current1fc、原request/prepared/discovery/controls/core一致。真实selector generation-deeff48fed0b/session ses_e734df353ffew5IaE4RafG3kbI，41初始文件50,078,774B逐字等于snapshot，原route反馈15,504,256B/SHA84e77ac656c0a7b818be825c89653ca31c9807f91b1bf60d55e72ee7bb312b31，supplemental=null；历史ccd90与实际installed_bundles/current source已交付。与此前普通feedback selector的prompt/model/tools相同，14输入相同、17变化、10新增、1删除，不是整个角色单变量A/B。

早期12 assistant/16 tools，私有history SHA a457fc9df023df57ce90462944a5a02f3398d0dc7455e54ceb2d3f0819d50d4b。tool11实际打印旧accepted质量/配对噪声/约束及parent+Graph setup路线；tool12打印installed bundle身份但affected_calls截断；tool13打印当前三collector索引，tool15读取候选代码。最初16工具尚未打开当前collector JSON正文，不能将交付/索引当作消费，也不能据早期状态预判最终不读。早期只完整复核11/12/13输出，其余参数/状态及部分输出；终态必须完整审计。详见[启动记录](evidence/v6-ddp-accepted-communication-trial-20261030/startup-verification.json)及[早期读取范围](evidence/v6-ddp-accepted-communication-trial-20261030/startup-tool-review.json)。当时无GPU jobs、两卡空闲，150GiB额度实际余9,812,717,568B约9.14GiB；角色仍运行，没有新候选质量或接受结论。

声明一轮有界试验，根目录 `/root/autodl-tmp/dev/v6-work/ddp-accepted-communication-trial-20261030`。上一[证据批次](v6-ddp-accepted-communication-evidence-20261030.md)已经刷新已接受执行的registry，并取得实际bucket、tuning、提交时序和校准材料；通信仍13 AllReduce、497,759,232B逻辑输入，计算调度接受不等于通信成功。下一步由现有普通Agent自行判断当前证据支持的范围和候选，不重复旧失败、不承诺一定选到通信组或得到收益，不开启无限候选链。

源为 `ddp-accepted-communication-evidence-20261030/ddp-evidence` 的1fc24545217087adefa1248704619fcdc882a19057336b231d8e02918bf4d0d8；原代码ccd90dc158788a164d6fda242572530e9cdc9ca50563af1a1d9d597f0ac12e0b、单bundle trial-d06aa22e2d-param-a72285d2、空参数、contract保持。原accepted source4541、证据source1fc、独立current7a49和历史Graph09b7保持；隔离新分支可按原接受规则演进。采用existing fork_run和CLI resume rounds1，默认scopes/parameter exploration，API observe-only/no caps。不重复初始GPU discovery/source/controls，不复制大参考。

原框架accepted trial-d06aa22e2d feedback逐字40,982,219B、SHA35bcf4de383882fee05ba7fe66e8afa6dbdd78a54999587872f3c4dbd0808494，零新增bytes、无supplemental。现有route针对新registry执行，历史candidate/原sidecars按普通接口打包；当前discovery的collector材料另按原common inputs交付。没有开发者planner错误表、独立终态表、因果答案或手写实现。现有selector输入本来包含search.installed_bundles/source；必须核验实际snapshot、读取、恢复/替换关系和工具行为，不能以交付代替消费证明。整组输入变化，不是单变量A/B或无辅助原入口。

完整原DDP任务、2连续质量/2warmup/5clean/rank、全梯度/参数/loss/finite/exact replica、1e-6/1e-4、原133.379621→140.04860205ms约束、rankmax先于median/MAD/noise保持；新设备20e8003c/a24af2e6、adapter b276a7ed、原controls只读，生产core ed24af7acc8caa15578e0b859c85ef8a901be51354448ed39094606ed9e10a2f不变。原始source/metric映射辅助边界保持，不新增FSDP式显存约束或跨设备anchor。

不手选组/bucket/算法/collector，不修改生产prompt/sampler/质量/接受，不并行GPU，不自动安排independent/native；真实planner可选已有collector。完成后审计完整工具输入输出、初始代码及编辑、所有参数/worker/raw trace/尾样本。仅真实新接受才声明独立整版B/C/C/B；通信结论另需实际通信机制，不能以Graph或较低NCCL sum替代。拒绝/异常原样保留，不重复到通过。Qwen关闭、FSDP最后拒绝保持。

空间依据：旧普通trial262,836,224B，feedback640,086,016B，本次刚完成证据125,440,000B，均为实际allocated du；现余约9.28GiB。预算1GiB增量、8GiB告警线，属于估算而非API/collector cap；本轮不清理/复制参考。低磁盘不等于完成，不能终止活跃实验或盲删。新批次仅一个隐藏普通completion/disk watcher，旧50252处理后不重启；所有后台/SSH/queue显式CREATE_NO_WINDOW。raw/private configs/history留服务器，代码/hash/小型metadata进GitHub。
