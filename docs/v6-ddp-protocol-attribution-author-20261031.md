# DDP 已有产物协议归属取证编写（2026-10-31）

这是一次明确有界的证据获取，源于[普通selector完整终审](v6-ddp-pipeline-communication-selection-20261031.md)后的具体缺口，不是自动候选链、prompt对照或reviewer堆叠。问题是：原始当前代码、采集、日志和安装库源码能否识别steady AllReduce的实际协议，以及配置/communicator创建的生效范围；无法证明的部分明确unknown。没有指定所需protocol、group、算法或优化答案。

现有 `ProjectAgents.generate(diagnostic_author)` 只生成 `artifact-reader.py` 和 `evidence-plan.json`，syntax/schema验证；controller不执行生成reader。新实验question/interface/body独立声明，生产role prompts/model/tools/core不变，没有feedback repair段。代码只读JSON/text/SQLite/相关已安装源码、输出有界报告；禁止CUDA初始化、模型/tensor运算、进程组、真实collective、profiler、live workload、参数试验或修改env/config。该接口不调用原evaluator，不是已关闭runtime observer的再次修复。工具实际行为和所有输出仍需终审，外部FS非hermetic。

只输入当前02b933原代码/原request、adapter、原训练源/common/README、原search/ownership、四个当前collectorJSON、原selection（作为假说）、question/interface/index和正常自动environment。共17文件3,037,071B；原selection c1e508b9字节不改。索引引用当前五个原始capture的requests/results/logs/SQLite/JSON及SHA，不复制raw traces/controls。**不输入终审、单位纠正、因果表、手写实现、旧repair答案**。既有source/metric/CPU事实辅助沿版本历史保留，不声称未经辅助或单变量AB。

源current0bc64a5a与singlefixed-fused-pipeline、空参数/自身fine8、代码02b93360…25781B不变。预检1947冻结身份，核心ed24af7…163文件；实际盘余9,400,045,568B约8.75GiB，GPU空闲。新256MiB输出/框架空间预测加8GiB警戒通过，非API cap；无清理/新controls/referencecopies。APIobserve-only、正常schema/provider行为原样，错误保留。private配置/history只在服务器，禁止未过滤metadata输出凭证。

生成后审计真实snapshot/工具/代码/错误及可答性，再决定有界CPU读取。若仍不能建立归属，保留unknown和具体缺失证据，不自动再次修复/选择/候选直到满意。没有新质量/clean/formal/independent/生产接受；02b933独立57.8545851%为已完成联合收益，不能解释为纯通信份额，也不与formal10.8304661%或旧收益相加。Qwen/最新拒绝FSDP/77ac/observer及eef828 correction边界保持。

启动记录随后追加。只为本新批创建一个隐藏普通终态/磁盘监听；旧44808已经handled/退出，无Goal、模型轮询、投递测试或旧批重启。low_disk_space不是完成；仍活跃时按既有授权安全处理且保留全部数据，不杀worker/盲删。
