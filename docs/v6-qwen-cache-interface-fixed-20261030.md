# Qwen缓存接口候选原字节固定执行

**已完成并审计：2026-10-29 22:30:20–22:31:35 UTC。** [启动](evidence/v6-qwen-cache-interface-fixed-20261030/launch.json)冻结 `b17c4a4f93823f3a74143614596d12df632755ea`、controller57868；[500身份预检](evidence/v6-qwen-cache-interface-fixed-20261030/frozen-preflight.json)通过。终态审计SHA `c16a1b85af24a69fb09a7ab437a7f33f73c20d8a89d462dcab52ddd62c5242e4`，[公开小摘要](evidence/v6-qwen-cache-interface-fixed-20261030/public-terminal-audit.json)保留全部10个计时样本。两worker退出、500冻结身份/原source/adapter/entry/core/当前UUID/实际request/contract与完整启动manifest全部核验，源与隔离current仍空parent2a42。

| 固定endpoint | 完整连续质量 | 全生成中位数ms | TTFT中位数ms | 原TTFT约束 |
| --- | --- | ---: | ---: | --- |
| B | 2次零误差，各4,861,988元素 | 770.550629 | 25.941852 | 失败 |
| C | 2次零误差，各4,861,988元素 | 380.435753 | 26.076953 | 失败 |

两者各5样本、原2warmup/全32token工作不改。B的910.177790/896.865075ms尾样本、30.249485ms TTFT完整保留；所有metric从原样本重算一致。原锚23.285755、上限24.45004275不重设；B质量通过所以C按预定门执行，B约束失败不被隐藏。C完整质量/coverage通过，但不满足原TTFT，未进行搜索、配对或独立接受，没有发布候选或声称新优化成功。

C revision `8ba195cf6213a897698c9825e3f1a4685fda233997cb01bf12aae962320e4533`，单bundle fixed-cache-interface/空参数。质量阶段记录60 fast_graph_replay+2 fast_graph_replay_absorb，reference计数与框架fallback均空；该代码use_graphs=False时也发Graph标签，因此标签不等于native路径证明。安装mapping及外层通过亦不证明每个setup gate成功：verify异常可能留下fast_enabled的代码风险仍存在，本批没有setup异常sidecar。没有Nsys/trace或新保存全输出tensor；普通评价器保存的是完整比较结果，不伪称新独立源重算。两worker日志只有权重加载。

C setup5596.321494ms在warm-model计分外；runtime peak由8,168,429,568增至8,239,691,776B（+71,262,208），lifecycle peak由8,188,023,808增至8,297,083,392B（+109,059,584），不是显存收益。内部缓存/所有权/并发与唯一历史原因限制不因本次通过而消失。

只读审计首次把空diagnostic_metrics误当list（实际dict），第二次误从启动manifest期待值排除files（worker保存完整manifest），两个错误日志保留；按实际schema修正并核对完整manifest，没有模型、worker或候选重跑。短批在[监听预检](evidence/v6-qwen-cache-interface-fixed-20261030/watcher-precheck.json)前2秒已结束，未创建新监听/队列通知；不是监控中断或补发测试。两卡空闲、实际df余约8.00GiB，无清理。本轮到此封存固定约束失败；后续若做普通续行，应提供实际质量/TTFT及路径限制，不能重测同版本、换锚或手填原因。本轮没有新增后台实验。

声明一次B后C固定诊断，使用[真实角色审计](v6-qwen-cache-interface-candidate-20261030.md)的精确代码 `7e78b82dd21391e3db3d6292e3798feff1908f53259aa732a3c011660c28fb92` / proposal `048d30b638809aad76ef99e757be41ea20345d1def2f72da5c332cc56aaf94b6`，空参数。无新Agent、搜索、手改、调参、观察器、Nsys或正式性能接受。生成自检的flag/所有权风险保留，由原外层评价器检验真实安装、派发、完整质量与约束。

根目录 `qwen-cache-interface-fixed-20261030`；既有fork_run/bundle_from_outputs/stage/execute从当前设备空parent `2a42b9b182b77d54a4028e779d8ba8f8bee5dabec4a83917015ed52e486e0dd1` 建隔离分支。原request/discovery/adapter/contract与controls只读复用，不重新发现或复制参考张量。源和隔离current均保持parent，stage不是publish。B仅以完整质量/coverage为C前置门，B的TTFT不重新定义约束；B无效则C不运行。

原始B1/prompt256/32 greedy、BF16/SDPA/noTF32、全logits .02/.02/exacttokens、两同步/状态生命期、2连续质量/2warmup/5clean（若到达）保持。当前20e8 GPU原TTFT锚23.285755，上限24.45004275；不采用旧设备锚。保留所有尾样本、内部fallback、安装错误/未到达阶段，不重试到通过。固定计时不等于配对或独立接受，不借旧f065/d5bd trace。

真实DynamicCache支持父方法接口，但31个cache复用且ownership主要查第一层keys指针；teacher框架stride不匹配仍可能走父路径。直接自检未覆盖真实框架两次run/observe，verify异常未清fast_enabled，eager回退后的标签可能仍写Graph。原字节执行检验这些风险，不据此宣称唯一原因或任意并发生命期安全。

启动前df约8.11GiB、GPU空闲；分支discovery约68.9MB、原固定C约20.2MB，预计仅小规模结果/元数据增长，无大型profile或参考复制。本轮不清理、raw/private会话不出服务器。接近8GiB监听线需按实际df判断，告警不是完成，不终止实验或盲删。新长工作使用一个隐藏普通完成/磁盘监听，旧生成监听已处理。生产core `ed24af7acc8caa15578e0b859c85ef8a901be51354448ed39094606ed9e10a2f`、prompt/sampler/allocator/接受规则不变；人工源映射/此前CPU辅助边界及历史接受拒绝保持。
