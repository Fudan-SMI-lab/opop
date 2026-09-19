# v5-c3-qwen3-4b-operator-goals - Work Plan

## TL;DR (For humans)
**What you'll get:** 先证明C3-B：在手工定义并冻结的通用模型任务下，用户目标驱动真实算子SOURCE重写、Qwen3-4B集成与retuning，交付同卡三目标交叉矩阵、baseline和失败记录。用户已明确“好开始执行”；T1准备已通过父级核验，当前T2归档先于产品实现/GPU。

**Why this approach:** 先提供可信的手工任务/evaluator，复用已有provided-eval优化路径与状态保存，只补任务数据、常驻模型测量和算子绑定三项职责；先验证优化能力，不等待自动任务构建。

**What it will NOT do:** 本轮不证明自动自然语言任务构建；不建安全、权限、恢复或通用缓存平台，不把配置搜索当算子贡献，不反复生成直到获胜，不广泛升级依赖、变更模型variant或重跑旧实验；仅允许下述已批准的最小包组合及其必要依赖。

**Effort:** Medium；快速关键路径为并行接缝实现＋有界实验，主实验拟三小时排程；准备/实施/分析另计，无完成时长保证。
**Risk:** High — 两host模型资产及实际runtime组成已由T1准备回执核验并经父级接受；替换入口、模型显存/质量与性能仍待后续验证，CPU准备通过不是模型质量通过。
**Decisions confirmed:** 最新用户“允许”已批准此前列出的最小包和数值/58条synthetic corpus组合，记录于2026-09-19T15:05:23.292Z；这是owner确认，不是科学验证，不再重复权限仪式。

Your next move: T1已获父级验收，完成T2唯一协议归档并核验远端后再放行T3/T4；T2核验前不改产品或运行GPU，主实验时钟尚未创建。

---

> TL;DR (machine): owner=Atlas parent; intent=clear; review_required=false; status=active; implementation_authorized=true; scope=C3-B manual-task-first; started_at=2026-09-19T12:58:09.432Z; current_task=2；T1已由父级核验，T2及后续checkbox仍由父级管理。

## Scope
### Must have
- 核心C3-B：给定手工定义/审阅/冻结的通用模型任务＋goal → baseline phase/operator trace → 真实TaskRewriter作Triton/CUDA算子或子图SOURCE重写 → 模型绑定 → native目标评价/retune → heldout。
- 面向用户的未来输入仅“可运行推理脚本/项目＋目标文本”；资产/workload/quality须包含、可发现或一次确认。本pilot由人工提供可信任务，不声称仅靠这两项已自动构建测试。
- 不是配置开关搜索、孤立kernel计时、三个手工goal策略，亦不是给旧latency字段换TTFT名字。
- 可编辑PyTorch/Transformers `Qwen/Qwen3-4B`，BF16、non-thinking；复用两host已有checkpoint/tokenizer/model source的精确revision。
- T1先查已有host资产/配置/缓存与interpreter并优先复用；仅缺`Qwen/Qwen3-4B`权重时，用户现已授权从官方同variant取得固定revision的weights/tokenizer至`/root/autodl-tmp`下。
- 用户明确例外允许T1在T2前获取权重（不加载模型/不运行GPU）；两host checkpoint/tokenizer身份一致，BF16 non-thinking不变，实际revision/子路径由operator发现并记录，不能杜撰。
- AutoDL参考`https://www.autodl.com/docs/network_turbo/`：仅下载进程/子shell内`source /etc/network_turbo`；加速可能影响正常网络，结束执行`unset http_proxy https_proxy`并退出子shell；不写全局proxy/持久配置，不输出认证环境或凭据。
- 下载/传输成本仍单列准备账。T1回执已核对两host同pin `1cfa9a7208912126459214e8b04321603b3df60c` 的全部10文件；本次不重做资产校验。
- 最新“允许”（记录2026-09-19T15:05:23.292Z）授权A `kernel-opt-venv` 加 `Optuna==4.9.0`；B `orch-venv` 加 `Transformers==5.16.1`、`safetensors==0.8.0` 及必要依赖。先审查resolution，保留原Torch/CUDA/Triton，不广泛升级/换模型；冲突或访问失败仍须报告。
- 此包授权允许remote T1在T2前完成环境准备/非模型验证，不代表已安装，不放行产品编辑/GPU；operator独占raw contract/preflight/T1证据，元数据owner不修改它们。
- 两host四张独立物理GPU，不是96GB显存池；每卡一个4B常驻实例，reference与candidate顺序绑定，不并驻两份大模型或FP64整模型。
- 本计划唯一权威文件在`.omo/plans/`；匹配draft只保留决策状态。历史计划编写未改产品/boulder；本次仅激活执行元数据，旧记录/报告/raw不变，产品工作须T2之后另行执行。
- 未来执行使用已授权`D:/Pyhon_projects/opop/v5`工作树与`.venv-v5`，保留继承dirty；不要求新PR/worktree/stash/清理。
- T2先GitHub归档本协议再产品编辑/GPU；T6再发布测试过的精确source SHA，主实验只运行该revision。历史规划及本次元数据记录不运行Git；T1权重获取和此次已批准的最小环境准备可提前，其他边界不变。

### 事实入口与文件边界
- 引用缩写E=`.omo/notepads/v5-c2-helper-pilot/c3-operator-entry-readiness.md`；O=`c3-objective-readiness.md`；R=`c3-model-reuse.md`；B=`time-bottleneck-audit.md`（后三者同目录）。
- 已有可用：`evaluation/task_eval.py`模块状态、`tuning/objective.py` native方向、`control/direct_task.py`与`tuning/tpe.py`搜索；不改造成新引擎。
- 新责任1：`examples/c3_qwen3/manual_task.py`及冻结JSON数据——手工任务合同、共享`evaluate(candidate_path, params, context)`薄适配；优先已有context/简单typed task spec，不新建通用schema。
- 新责任2：`examples/c3_qwen3/model_runner.py`——常驻模型生命周期、共享手工测量primitives、quality与有界运行入口；不是通用服务。
- 新责任3：`examples/c3_qwen3/model_binding.py`——精确模型revision的site绑定、trace/extractor和小fixture；不注册任意模型。
- 只连接必要接缝：`agents/task_rewriter.py`、`control/task_rewrite.py`、direct_task/TPE小预算；`task_cli.py`/`wiring.py`仅在provided-eval或重写runner确需时最小修改，不动自动入口分支。
- 已读`src/kernel_optimizer/task_cli.py:59–123`：`--eval-file`需显式`--direction`，80–84行选择已有evaluator，105–120行保留TaskEvaluator/TaskSearch及goal/context到rewriter。
- 当前复用`--eval-file --direction --goal`组合，TTFT=minimize、single/multi=maximize；122行`execute_best`额外调用和初始baseline搜索预算必须在pilot接线中显式处理。
- 完整路径相对v5；`agents/`、`control/`、`evaluation/`、`models/`、`tuning/`、`task_cli.py`、`wiring.py`源码缩写均前缀`src/kernel_optimizer/`。新增路径是提案，不声称已存在；最多必要包标记，不加第四项通用责任。

### T1数值合同：owner已批准，科学有效性尚待验证
- T2归档前准备补注：父级已审阅接受当前T1回执并勾选T1；该接受更新了回执写作时的“待父级review”状态，不改其历史记录。未加载模型、运行GPU或创建主实验clock。
- 两host全部10份官方资产已核验，pin=`1cfa9a7208912126459214e8b04321603b3df60c`；共同路径=`/root/autodl-tmp/c3-qwen3-4b-operator-goals/assets/Qwen3-4B/1cfa9a7208912126459214e8b04321603b3df60c/`。
- 模型/search runtime：A=`/root/autodl-tmp/kernel-opt-venv/bin/python`，B=`/root/autodl-tmp/orch-venv/bin/python`；均为Torch2.13.0+cu129/CUDA12.9、Triton3.7.1、Transformers5.16.1、safetensors0.8.0、Optuna4.9.0、Pydantic2.13.5。A另有orch-venv未改且不用于此模型/search runtime。
- 本地权威准备材料根=`results/c3-qwen3-4b-operator-goals/`，远端handoff根=`/root/autodl-tmp/c3-qwen3-4b-operator-goals/`；raw/接口原件不随本计划提交。
- 已批准`contract.json` SHA-256=`9d1ef6a79bf99eb6ac0e639d82e894227b403124764ba4018531c9ddf179eaf8`；`interface.md`（c3-task-local-v1.0.0）SHA-256=`ffddc3eb03c181194b57128eee085946cd4f1f1b701ca4da14ffc97e8d495e7f`。
- 当前`preflight.json` SHA-256=`95f1fb5e453dc4839e72f5b6305616599d10ba5e79d714ba9e5d46f5f016bcfa`；完整准备证据见`.omo/evidence/v5-c3-qwen3-4b-operator-goals/T1.md`。两host实际CPU import/tokenizer重现58 prompts/84,096 input tokens，CUDA未初始化；不是GPU fit、binding、quality或性能证明。
- A既有pydra-config/dill的pip-check冲突在安装前后相同，B pip-check通过；此继承限制照录T1，不扩大本次包修复范围。
- 历史上这些值为提案；本次真实用户“允许”完成唯一紧凑确认（记录2026-09-19T15:05:23.292Z），之前的自动nudges不算批准。
- 批准已准备的58条synthetic prompts，corpus SHA `d60e5c108e79c8327a74199d5c093b117f9e64661cff4356e33a723dceb30a02`；不重生成/换样本，不改下面数字。实际asset/runtime/interface由operator完成验证与冻结。
- TTFT：4096 rendered input tokens、1 output token；暖模型提交到首token可用的wall毫秒，minimize，包含本次prefill/首token生产。
- Single decode：256 input、完整128 output；真实growing KV执行，maximize `127/(t_last-t_first)` tokens/s，另报TTFT/总时长，不能测一个decode step代替。
- Multi：8个独立请求、各512 input/128 output；同步有限batch，maximize `1024/shared_batch_wall_s`，包含全部prefill与completion。
- Multi每请求独立KV/position/mask/RNG状态，以共同起止wall和实际token事件计数；不是单forward×8、CUDA stream数或在线服务吞吐/SLO声明。
- 统一非thinking chat template、`enable_thinking=false`的实际可用配置、greedy/seed0、固定输出长度；EOS在达到指定长度前不终止，记录该策略。
- 固定rendered token IDs（含template特殊tokens）精确达到各列长度；T1保存文本与IDs/hash，padding不计有效input/output token。
- 固定文本不足/模板不支持non-thinking/计数不符→明确外部条件缺项，不静默截断目标、改模型或改变输出长度。
- Tokenization、checkpoint载入与预热在暖测量外；请求接受后的KV分配、prefill、decode与token-ready同步在各指标对应边界内。
- 每个timed请求fresh KV；decode计分虽排除首token耗时，仍实际执行prefill；TTFT禁止缓存KV/prefill答案以“加速”。
- Local operator quality已批准用`torch.testing.assert_close(rtol=0.02, atol=0.02)`，另查shape/dtype/finite及必要状态副作用；批准不代表该阈值已经科学验证。
- Model quality已批准用teacher-forced logits `||candidate-reference||2 / max(||reference||2,1e-12) <= 0.01`，每prompt分别达标。
- Paired mean NLL(candidate)-mean NLL(reference) <= 0.02 nat/token；参考target token序列固定，各prompt有效位置计数后作paired平均，原值完整保留。
- 已批准固定2条短calibration prompts（各64 input＋32 reference continuation）及不相交heldout内容；reference输出仍需后续实际生成，拒绝测试后挑prompt或放宽tol。
- 每次model quality沿baseline teacher-forced tokens，不要求自由生成文本bitwise一致；计时仍执行自己的greedy完整生成并验证长度/状态。
- 搜索每config至少覆盖两条calibration quality；final每cell使用该列heldout prompts及其冻结baseline continuation作teacher-forced质量，不能仅以短calibration通过替代heldout质量。
- T6先baseline A/A检查quality、reset与计数再冻结手工evaluator/oracle；A/A不达标即STOP，不自动松阈值。禁止套KernelBench整模型FP64 rescue。
- 已批准搜索重复：每trial TTFT4请求、single2请求、multi1完整wave；TTFT取4个wall中位数，single取2个完整decode rates中位数，multi取该wave rate。
- 每candidate/params先做一次不计分同shape预热并reset；预热、quality、失败皆占wall与调用账，不能伪装免费。
- Final每cell 3 blocks；每block重复数仍TTFT4/single2/multi1，内容与calibration/search不相交，形状/采样/quality同列完全一致。
- T1冻结4条TTFT search文本、2条single、8条multi，每trial使用同组；final冻结12条TTFT、6条single、24条multi，按块固定分组，所有rows复用同列组。
- 原生目标使用真实ms或tokens/s与min/max，不把throughput取负塞latency；整模型quality不合格无有效score。

### 手工任务、binding与科学预算
- 三份显式手工goal合同共用同一checkpoint/baseline可编辑算子能力、quality政策、预算和共享evaluator，仅指标与事先冻结的column workload不同。
- 手工evaluator组合load/bind/reset/measure/quality；已有provided-eval路径实际导入callable并返回native TaskEvaluation；QA检查语义而非仅compile。
- 三个measurement primitives是人工实验设置，不是三种优化策略或手工winner；baseline phase profile＋agent决定真实source change，禁止固定TTFT→norm/decode→attention映射。
- 先收每goal一次baseline phase trace（不在timed区域），提供source site、调用次数/shape/状态；重叠duration不能直接宣称critical path收益。
- 每goal最多一次selected补充profile，确有解释需要才用；不每trial全profile，资源记录不代替目标J。
- 三组必要binding控制而非安全平台：(a)正确非PARAMS计算重写在模型路径执行且被替换site旧dispatch为0；(b)故意错误或跳过替换均拒绝；(c)baseline→candidate→restore后质量/状态恢复。
- Coverage/sentinel在诊断pass，timed run只关联相同安装bundle身份；compiled graph旧引用做最小失效/重绑定，不绕开candidate。
- Bundle显式列operator及helpers；跨轮传播/最终执行保留全部实际依赖，helper-only计算变化支持，检查整个执行bundle而非只main.py。
- 每个accepted bundle是相对原baseline可完整重建的累积site→replacement映射；r2继承r1实际改动，不能只交最后一个patch导致heldout漏用前轮算子。
- 不把dead-code、PARAMS-only或运行配置改动当operator贡献；源改变也仍须runner观察到目标site执行。
- Candidate不可编辑冻结手工evaluator/reference/workload；执行前后简单hash检查够用，任一变更invalid；不建OS sandbox/RBAC/对抗安全系统。
- Reference logits/局部fixtures可复用作oracle，绝不复用candidate性能；一份resident权重顺序恢复baseline/绑定candidate，状态每请求重建。
- 3个目标各2次顺序结构机会；r2从自己accepted incumbent继续，r1拒绝则仍从原baseline，不能从别的goal赢家注入。
- 每结构机会最多8个native config slots，default＋最多2个本candidate推荐包含在8内，合计上限48；无扩域额外B40/第3轮。
- Empty nonparam space实际一次评价，不造dummy参数；有限域耗尽不补重复点，不称完成8个unique trials。
- TPE现有startup10不适合8：仅pilot显式startup2，T5核对已装API最小值/计数合法；不支持就报blocker，不静默改成全随机或改全局默认。
- 真实agent生成预算仅6次TaskRewriter（3goal×2顺序rounds），每机会至多1次code/correctness修复，不是替代draw；人工任务准备不消耗入口生成调用。
- 搜索self-test/正式验证共享该机会8 slots：任何agent触发的本candidate模型GPU试验先占slot，失败/repair前尝试也占，native只用剩余数。
- 每slot内syntax→fixture quality→model quality→目标重复的调用另记子计数；评价错误不返还slot、私有分数不作正式缓存、原生重复评价仍占新slot。
- 保留bash/read/write自主性，提示和小helper把GPU请求送当前runner预算；审计越过helper的私有GPU调用即记超预算/协议违例，不隐去或补draw。
- 此首轮不做C2 fresh endpoint probes；显式关闭`control/task_rewrite.py`默认响应探测，不借resource_metrics过滤假装零调用。
- 三手工任务baseline验证、binding controls和A/A是T6预备工作，单列实际模型调用；readiness修代码须重测/重新发布后再开始main，T7后无额外smoke或隐藏`execute_best`。
- 初始baseline每goal一次完整廉价协议作为父分数；新结构按有效native J严格改善才接受，平局retain；搜索数据只用于选优不作heldout。
- Final四rows=baseline＋TTFT/single/multi各自winner；三columns=三goal workload，3 blocks/cell，共36 cell blocks，包含baseline9与winner27。
- 相同winner也按独立row fresh测，无有效child标baseline fallback；无效cell保留原因/缺失，不填0、不偷换另一个候选、不heldout重选。
- 三goal同时各独占一卡：预定A0=TTFT、A1=single、B0=multi；B1不追加搜索，可空闲；T1实测可用性后冻结映射。
- 搜索全部terminal后final固定A0串行，所有rows/columns同物理卡；其他卡停止timing，不用跨卡native分数直接填矩阵。
- Final列顺序TTFT/single/multi；每列block1 rows按baseline,T,S,M；block2反序；block3 S,M,baseline,T，固定并输出原始数值。
- 新主clock S在模型就绪/源码已发布后设定：search admission截止S+9000，final截止S+10800；最后30min只准heldout。
- 每goal从自己的首次search准入起连续90min（含生成、自测/等待，不暂停计时），与global search cutoff取更早；不同GPU可并行，同GPU禁止timing/编译干扰。
- Final必须等三搜索terminal再同卡；in-flight可drain并单列真实时间，不重置clock；不足36块写censored，不降重复数。
- T6用预备实测耗时估计最慢goal与36块/质量成本，若90min/150min search或30min heldout明显不够，在候选结果前STOP请求改额度。
- 3h不是完成保证；setup/资产发现/部署/分析另计并报告total turnaround，不从旧kernel测量臆造4B载入速度。

### 后续C3-A：DEFERRED（非执行项）
- 以后另行把仅有推理脚本/项目＋goal＋资产的输入交给EvalBuilder，自动产生任务，再与本轮手工任务oracle/tests作独立对照；不能把手工fixture作为agent答案泄漏后宣称自动完成。
- C3-A不属于当前验收/前置、checkbox、commit、调用预算或blocker；本轮不改`agents/eval_builder.py`、其wiring或taskCLI自动分支，也不强制运行未改动的自动入口测试。

### Must NOT have (guardrails, anti-slop, scope boundaries)
- **禁止：通用安全/权限/恢复平台、universal cache/model registry、全项目engine或legacy orchestrator重写。**
- **禁止：旧C2修复/重跑、opaque vLLM-first封装、静默模型/精度替换、三个预写goal算子策略、配置收益冒充算子贡献。**
- **禁止：盲目B40×100整模型生成、每trial重载权重、无上限generate-until-win、自测隐藏GPU预算、heldout择优重排。**
- 仅安装此次已批准的最小包及必要依赖，resolution不得改变Torch/CUDA/Triton或静默换模型；其他依赖/访问问题仍需报告。保留原资产身份/下载账，不清理用户dirty/旧raw、不复核旧数千hash。
- 不声称配置字符串/agent代码返回即证明目标正确、算子执行或论文正结果；禁止强求对角线胜出、事后换workload。

## Verification strategy
> QA由执行agent运行；owner已确认数值合同/最小包，T1实际CPU环境/接口准备已由operator核验并经父级接受，但不是模型质量或性能通过。归档任务不运行安装、实现测试或GPU；T2及后续完成勾选仍由父级负责。
- 生产变化采用focused pytest TDD：记录具体test id的RED断言、GREEN及真实CLI/model证据；文档/归档任务只做文件/回执验证。
- CPU统一前缀P：`$env:PYTHONPATH='D:/Pyhon_projects/opop/v5/src;D:/Pyhon_projects/opop/v5'; $env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'; & ./.venv-v5/Scripts/python.exe -B -m pytest -p no:cacheprovider -q`。
- 下文`P tests/... -k ...`表示把上述命令逐字展开再接参数；所有新增test/CLI为拟实现接口，不冒称本轮运行通过。
- 证据根EVID=`.omo/evidence/v5-c3-qwen3-4b-operator-goals/`；raw根RAW=`results/c3-qwen3-4b-operator-goals/`，每task用`T<N>.md`串实际命令/结果。
- Runtime命令M：在T1选定host/源码目录用已有绝对interpreter运行`python -m examples.c3_qwen3.model_runner`；receipt保存展开后的完整命令。
- Happy：冻结手工evaluator驱动正确operator替换并返回模型native J，goal/context确实送rewriter；edge：wrong/skipped/陈旧helper或不完整128输出拒绝；regression：provided-eval/native min/max不变。
- 独立质量阈值、token计数和binding证明先于score；guard失败仍留失败记录，不让测试脚本exit0代替measurement有效。
- 收集仅新输出的一个`RAW/evidence-index.json`（path/hash/用途及receipt引用），不构建ledger平台、不重复哈希旧raw；清理owned进程/显存state写teardown回执。
- 最后一次focused集合覆盖新测试及`tests/test_v5_direct_{end_to_end,task_eval,rewrite_edges}.py`，不强制未触及子系统的测试；缺工具记录，不安装，不每task全仓大suite。

## Execution strategy
### Parallel execution waves
四波、9项；不为凑任务数拆细或增加review。
- Wave1：T1已有资产核验及owner确认 → remote operator按授权完成最小包安装/CPU环境与接口验证（已获父级接受）→ T2归档；后续真实模型就绪及质量仍在T6验证。
- Wave2：T3手工task/evaluator owner A独占`manual_task.py`/合同数据与tests，T4 owner B独占`model_runner.py`/`model_binding.py`与tests，按T1接口并行。
- 共同接口以现有context＋冻结JSON和简单类型表达；manual_task调用runner的`load/bind/reset/quality/measure/close`，runner不反向import manual_task，双方用同接口fake测试，无循环依赖。
- 合同仅含asset/source/tokenizer refs、goal/direction/unit、prompt IDs/split、completion/reset/repeats/quality、trace/site scope；bundle仅entry/site映射/files/helpers/params/space/parent refs。
- T1冻结样例与API签名；新增typed定义如必要仅放task-local必需字段，不建`models/operator_task.py`通用schema。接口修改由owner A协调，T6才整合真实两侧。
- Wave3：T5 rewrite/helpers/小TPE owner C依赖T3/T4；然后T6发布/三手工任务验证＋operator smoke owner B，C交接后停止修改共享文件。
- Wave4：T7三个goal设备owner并行搜索 → T8自动同卡矩阵 → T9分析/报告；T7/T8同一有界runner自动串接，无人工阶段等待。
- F1仅一个最终独立复核者；不五代理审计，不Momus/高精度流程。CPU和raw分析可并行于非冲突GPU工作，final同GPU严格串行。
- 快速关键路径：发现现有资产→手工任务合同→evaluator与resident绑定并行→三任务baseline/一个真实绑定→48 slots→36 blocks；不等待任务自动构建。
- 实施/准备未有4B实测保证；将每阶段起止及等待写receipt，优先复用模型，禁止凭历史C2耗时承诺数小时一定完成。

### Dependency matrix
| Todo | Depends on | Blocks | Can parallelize with |
| --- | --- | --- | --- |
| 1 | 开始执行/权重授权及最新最小包、数值合同批准 | 2,3,4 | remote runtime验证lane；原始开始时间不重置 |
| 2 | 1 | 3,4 | 无；归档先于产品/GPU |
| 3 | 2，T1公共接口 | 5 | 4，文件互斥 |
| 4 | 2，T1公共接口 | 5 | 3，文件互斥 |
| 5 | 3,4 | 6 | 无共享文件并编 |
| 6 | 5 | 7 | 三手工任务baseline验证，GPU控制不重叠 |
| 7 | 6全部准入/预算可行/发布 | 8 | 三goal独占A0/A1/B0 |
| 8 | 7三条轨迹terminal | 9 | 无；全部cell仅A0 |
| 9 | 8含失败/censored结果 | F1 | CPU分析可分工但报告单owner |
| F1 | 9 | 交付关闭 | 单独紧凑核验 |

## Todos
> Implementation + Test = ONE todo. Never separate.
<!-- Tasks appended to the official scaffold; proposed implementation only. -->
- [x] 1. T1：发现现有Qwen3资产并冻结一次最小合同
  Wave1；权限/合同及实际准备验证已完成，T1已获父级接受。Owner Atlas/remote T1 operator；仅已批最小包环境准备/CPU验证，未运行模型/GPU或产品编辑。
  References：O:104-126、R:87-107、E:185及T1资产回执；数值在2026-09-19T15:05:23.292Z记录owner批准，原始提案及等待过程保留历史。
  输出：`RAW/preflight.json`记录checkpoint/tokenizer/source精确revision/path、interpreter/packages、可用GPU/显存、完整rendered语料与合同来源。
  复用已核验pin/10文件与原58条语料；仅安装已列A/B包及必要依赖，检查resolution保留Torch/CUDA/Triton；实际版本/import/runner接口由operator记录，不因owner允许直接验收。
  验收：路径都真实存在且属于Qwen/Qwen3-4B BF16，三目标语义与T3/T4字段同一冻结版本；禁止用模型名替代revision。
  Happy QA：Bash在已找到host执行`<PY> -c "import importlib.metadata as m; print({n:m.version(n) for n in ('torch','transformers','triton')})"`，只读配置/文件证据存EVID/T1.md。
  Failure QA：依赖未成功安装、受保护stack将改变、访问失败或实际接口未验证时不放行模型执行，记录确切问题；`tests/test_c3_manual_task.py::test_missing_assets_block_execution`仍在T3锁定，授权/下载不等于T1完成。
  Commit：N；仅本地合同/发现receipt，T2负责唯一协议归档。

- [ ] 2. T2：实现/GPU前归档本权威协议
  Wave1；依赖T1，阻塞T3/T4。Owner publisher；本计划已有数值确认结果作为同一文件的执行前补注，不生成docs副本。
  References：本计划Scope；B:92-96，避免多轮准备/审核开销；现有已授权v5工作树保留dirty。
  仅提交`.omo/plans/v5-c3-qwen3-4b-operator-goals.md`并核验远端；若被ignore，未来明确force-add仅此文件，不把raw/draft/凭据带入。
  验收：archive SHA/URL、协议字节及T1确认状态回执齐全；成功后才放行产品编辑/GPU，不自动代表main已开始。
  Happy QA：未来`git show <archive-sha>:.omo/plans/v5-c3-qwen3-4b-operator-goals.md`与`git ls-remote origin refs/heads/v5`对应，EVID/T2.md。
  Failure QA：`git diff --cached --name-only`出现其他文件则停止发布并交还其owner，不reset/stash用户改动；不发布未确认合同。
  Commit：Y；仅protocol，建议`docs(c3): archive Qwen3 operator-goal protocol`，执行时匹配repo已有风格。

- [ ] 3. T3：手工通用任务与provided-eval接口CPU验证
  Wave2；依赖T2，与T4并行。Owner A独占`examples/c3_qwen3/manual_task.py`、三份合同数据和`tests/test_c3_manual_task.py`；不共编runner。
  References：实际`src/kernel_optimizer/task_cli.py:59–123`；O:28-33,52-56；复用`evaluation/task_eval.py`/Objective和现有context，不新写通用schema/score engine。
  提供同一手工`evaluate(candidate_path, params, context)`，三合同明确goal/direction/metric；仅组合T4接口，T1字段与CPU fake保证无需等待runner实现。
  实际provided-eval组合：`optimize-task --eval-file examples/c3_qwen3/manual_task.py --direction <minimize|maximize> --goal '<目标文本>'`，其余project/config/candidate/space/context按已有CLI传入。
  验收：提供eval-file直接使用人工callable，native goal/context仍传TaskRewriter；缺completion/quality阻塞，metric方向/单位正确，不手工指定优化site/winner。
  Happy QA：`P tests/test_c3_manual_task.py -k 'provided_eval_path or goal_context_reaches_rewriter or shared_manual_metrics'`，用真实`cmd_optimize_task`配外部CPU fake捕获调用，EVID/T3.md。
  Failure QA：同文件`-k 'missing_assets_block_execution or missing_quality or wrong_metric or missing_callable'`；adjacent用`P tests/test_v5_direct_task_eval.py`，不运行模型。
  Commit：Y；仅manual task/data/tests，建议`feat(c3): define shared manual Qwen goal tasks`；若CLI确需小接线，交T5单owner处理。

- [ ] 4. T4：常驻Qwen runner、trace binding与小fixture
  Wave2；依赖T2，与T3并行。Owner B独占`examples/c3_qwen3/model_runner.py`、`model_binding.py`、必要包标记及tests。
  References：R:16-29,87-107,157-165；E:77-99,121-156；采用本计划三组binding控制，不扩展安全平台。
  实现单卡单常驻模型，load/bind/reset/quality/measure/close；operator fixture只用于local验证，真实TTFT仍含prefill，decode完整128，multi独立8请求。
  原baseline goldens顺序取得后恢复，不并驻reference clone；candidate变化重载小bundle/失效相应graph，权重不每trial重载。
  验收：CPU fixture证明load_once、fresh KV、full-token events；指定site replacement真实调用证据接口与restore-on-failure，错误/skip拒绝。
  Happy QA：`P tests/test_c3_model_runner.py tests/test_c3_model_binding.py -k 'resident_reset or goal_windows or bind_restore'`，EVID/T4.md。
  Failure QA：同文件`-k 'wrong_operator or skipped_binding or stale_graph or short_decode or shared_kv'`；故障后baseline输出恢复。
  Commit：Y；仅model runner/binding及tests，建议`feat(c3): add resident Qwen operator evaluation`；本任务CPU验证，真实GPU统一T6。

- [ ] 5. T5：真实operator bundle、推荐点与8-slot native搜索
  Wave3；依赖T3/T4。Owner C独占`agents/task_rewriter.py`、`control/task_rewrite.py`、`control/direct_task.py`、`tuning/tpe.py`、`wiring.py`必要接线。
  References：E:32-40,91-99,164-183；O:70-81；R:141-155，现有startup10与空space限制明确处理。
  传实际trace/site/interface及helper closure；整个bundle非PARAMS计算结构比较，helper-only改写可执行/跨轮保存；默认关闭C2 response probes。
  复用现有TPE，pilot startup2显式参数，保留legacy10；default/recommended≤2/agent self-tests共享8 slots，empty space一次评价；无跨candidate分数缓存。
  六结构机会按本goal incumbent递推，修复最多一次且不返还slot；仅pilot接线把初始baseline定为一次单列评价，并禁用额外execute_best；不能照搬原CLI初始整预算而漏记。
  Happy QA：`P tests/test_c3_operator_search.py -k 'helper_only_roundtrip or eight_slots or startup_two or empty_space'`，EVID/T5.md。
  Failure QA：同文件`-k 'params_only or budget_exhausted or frozen_evaluator_changed or hidden_probe or extra_baseline_search'`；再`P tests/test_v5_direct_rewrite_edges.py tests/test_v5_direct_end_to_end.py`。
  验收：实际fake evaluator调用总数断言≤48、repair/自测计数可追溯，未执行的建议不能当TRIAL_DONE；只三项新增责任，无orchestrator改写。
  Commit：Y；仅列出接缝及tests，建议`feat(c3): bound operator search and preserve helper bundles`。

- [ ] 6. T6：发布tested source并验证三手工任务及单operator smoke
  Wave3；依赖T5。Owner B接管runner整合；A/C交还接口与文件后再修改，不并编。References：E:140-164、O:104-116、实际task_cli provided-eval分支及本计划quality/预算。
  跑一次focused合集/LSP，发布tested SHA并部署；加载三份人工合同与同一手工evaluator，验证三metric模型baseline/计数/quality及共享resident runner，不调用入口生成agent。
  实际baseline phase trace每goal一次供T7 agent选择source/site；本任务仅选一个trace实际执行site的确定性非PARAMS正确smoke，再wrong/skip/restore，不造三个优化策略。
  Smoke是明确诊断控制而非research candidate，真实rewriter仅T7；全部模型调用单列，A/A通过冻结manual evaluator/fixtures/goldens/语料；readiness源码修复需重测并重新发布。
  Happy QA：未来`M validate-manual --assets RAW/preflight.json --tasks RAW/manual-tasks --output RAW/readiness`及`M smoke --tasks RAW/manual-tasks --output RAW/smoke`，EVID/T6.md。
  Failure QA：`P tests/test_c3_model_binding.py -k 'wrong_operator or skipped_binding or restore'`；真实controls任一坏替换仍被接受、A/A超阈值或模型OOM即STOP，不松标准。
  测load/warmup/quality/search-repeat/heldout成本，冻结S+9000/S+10800与goal90min预算可行性；不够在任何正式candidate结果前请求变更，不删采样。
  验收：三手工任务真实模型执行通过、同baseline/checkpoint/quality/预算、正确binding与计时边界；manual tasks准备完、模型ready、budget-fit成立才创建main S，不等自动入口。
  Commit：Y（若有必要整合修复）；仅已测接线/fixture引用，建议`fix(c3): complete admitted model-goal integration`；main用最终tested SHA，raw不提交。

- [ ] 7. T7：三goal各两次结构机会，设备间并行
  Wave4；依赖T6。Owner分别A0/A1/B0，共用冻结手工evaluator/模型revision，各轨迹仅自己的incumbent；References：R:149-155、O:74-81及本计划48slots规则。
  未来`M campaign --assets RAW/preflight.json --tasks RAW/manual-tasks --clock RAW/clock.json --output RAW/main`复用provided-eval核心自动launch/join三goal，再立即进入T8，无新controller平台。
  此入口传统一S/search/final与per-goal90min，actual source SHA落盘；baseline3目标计分先行，再6rewriter机会及bounded repairs，所有GPU动作显式归账。
  Happy QA：`P tests/test_c3_operator_search.py -k 'three_goals_two_rounds or incumbent_lineage or shared_clock'`；真实surface保留每slot模型质量/native J与调用时序，EVID/T7.md。
  Failure QA：同文件`-k 'repair_limit or deadline_censors or invalid_keeps_parent or selftest_consumes_slot'`；真实失败原位记录，不重开candidate或续时。
  验收：最多48 slots、default/建议/自测均在内，无C2 endpoints/expanded B40；三卡可并行但各卡无timing重叠，记录失败/未准入/active wall。
  Commit：N；已发布source运行，记录RAW/main，任何运行中产品修复不得偷偷替换固定SHA或覆盖旧结果。

- [ ] 8. T8：同一GPU的4×3×3独立heldout矩阵
  Wave4；依赖T7全部terminal，runner自动继续，不等待人工review/导出。Owner final-A0；References：E:130-138,158-166；O:98-116。
  冻结三goal selected bundle/params和baseline，传`M heldout --tasks RAW/manual-tasks --selection RAW/main/selection.json --clock RAW/clock.json --device A0 --output RAW/heldout`（由campaign内部同一流程调用）。
  4rows×3columns×3blocks=36，重复winner仍测，baseline fallback标明；所有rows用各column冻结prompt/quality/repeats，final不反向改变选择。
  Happy QA：`P tests/test_c3_goal_matrix.py -k 'thirty_six_blocks or duplicate_winners or native_direction'`，真实36格块值/quality/source/device证明，EVID/T8.md。
  Failure QA：同文件`-k 'invalid_cell_no_score or cutoff_preserves_rows or heldout_never_reselects'`；缺块censored、不借搜索分数补矩阵。
  验收：prefill真实执行、single128与multi1024计数可复算、旧targeted binding在另goal下质量仍有效；到final截止不新准入，in-flight drain单列。
  Commit：N；只新raw，final完成/截断后释放owned resident模型/进程并保存teardown，旧服务/文件不清理。

- [ ] 9. T9：论文结果、失败与复现材料交付
  Wave4；依赖T8（包括censored闭合）。Owner analysis/publisher单报告owner；References：B:9-39,77-96与本计划科学口径。
  新`docs/result-c3-qwen3-4b-operator-goals.md`列baseline＋3winner对3goal的矩阵/3block范围、native方向增益、质量、选中site/bundle和配置。
  RAW保留manual-tasks/evaluator/oracle、operatorbundles/metricsmatrix/sourceSHAs、参数trial、6次rewrite及repair/自测/失败、setup/main/drain/turnaround与一份新输出hash索引。
  Happy QA：`P tests/test_c3_goal_matrix.py -k 'report_recompute or all_rows_preserved'`；`M analyze --input RAW --output RAW/analysis.json`逐格重算，EVID/T9.md。
  Failure QA：同文件`-k 'missing_block_no_positive_claim or quality_failure_not_zero or config_only_not_operator_gain'`；报告同winner/负结果/条件不足，不事后改任务。
  验收/论文声明限定“手工定义任务下，目标驱动真实算子优化”；证明agent目标上下文、实际算子改写/绑定/quality与新数据，观察到才谈目标分化，不强求三个不同winner或对角线胜出。
  明确不证明自动自然语言任务构建；人工measurement primitives不是手工优化答案，配置变化/搜索分数也不充当SOURCE贡献或heldout结果。
  Commit：Y；仅最终报告和必要小复现说明，建议`docs(c3): publish Qwen3 operator-goal pilot results`；public不含checkpoint/凭据/raw巨目录。

## Final verification wave
> 仅一次compact independent F1，T9后运行；这是未来交付验收，不是本轮高精度计划评审或安全审计波次。
- [ ] F1. 核验手工任务下的目标驱动重写、binding、quality与36块矩阵
  References：EVID/T1–T9、唯一RAW/evidence-index.json、最终report/tested SHA；只核新输出，不重新哈希旧C2 raw。
  QA：`M analyze --input RAW --output <独立临时结果路径>`复算min/max与质量；核3手工合同共享baseline/能力/quality/预算、6rewrite机会/≤48slots/36blocks应有与实际状态，无隐瞒repair/private GPU。
  focused CPU：`P tests/test_c3_manual_task.py tests/test_c3_model_runner.py tests/test_c3_model_binding.py tests/test_c3_operator_search.py tests/test_c3_goal_matrix.py`，加改动文件LSP；不再全仓大测试。
  一票阻塞：实际替换未执行、wrong/skip可过、质量/metric边界错、预算/遗漏被掩盖、报告与raw不符；负收益/相同winner本身不阻塞诚实交付。
  证据EVID/F1.md含artifact/math/source/binding/cleanup与verdict；delivery APPROVE不等于C3-B普适效力或C3-A验证通过。Commit：N。

## Commit strategy
- 历史规划/本次元数据激活NO Git；执行已获授权，依序T2协议先归档、T3/T4/T5各独占小增量提交、T6主实验前发布tested SHA、T9报告发布；不做末尾混合巨commit。
- 跟随执行时仓库历史风格；只stage明确归属文件，继承dirty不stash/reset/清理，不强制PR/新worktree，不提交模型、私密路径内容或raw。
- 每task的Commit行是范围约束而非现在已提交；必要CPU测试附同提交，数据run不产生产品修复commit。新hash索引只覆盖本次输出。

## Success criteria
- 当前状态：既有权威计划＋draft已获执行授权；intent clear/review_required false/Atlas parent；status active，boulder选中新C3 work，T1已勾选，T2归档中，其余任务/F1未勾选，父级逐项验证。
- 未来软件完成：冻结人工task/evaluator通过实际provided-eval路径，goal/context送达真实TaskRewriter；非PARAMS算子/helpers实际进模型、wrong/skip拒绝/restore通过、resident reset及native方向/预算正确。
- 未来论文数据完成：3goal×2机会真实轨迹、baseline＋3winner的同卡三列36cell blocks（未完成就明确不足）、独立质量和完整失败/成本/calls证据。
- 科学胜负从矩阵如实报告；负收益/相同winner/无有效child不可删除，8B旧目标与C2旧结论不被此4B pilot替代。
- 当前科学范围仅C3-B“手工定义任务下，目标驱动真实算子优化”；自动自然语言任务构建不计本轮成果、前置条件或成功标准。
- 历史计划写作已完成；本次元数据owner在授权/ledger/状态验证后交还父级，不代替remote T1 operator或提前产品/GPU执行，不追加review。
- T1准备前提已由实际runtime/资产/CPU接口证据及父级验收满足；设备快照不代替执行时可用性。T6模型load/binding/质量与预算可行性仍须实测，发现确切缺项即报告，不把准备验收写成C3性能证明。
