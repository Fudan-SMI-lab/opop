# 找墙与测斜率：证据优先的条件化见证设计

> 日期：2026-09-14。状态：评审稿，不是已实现机制或效果报告。
> 交付边界：只新增本 Markdown；不修改运行代码、配置、旧文档或实验。没有访问 box4，没有 GPU 测量，没有远程事件重放。
> 本文的“提案阈值”均为待预注册的操作规则，不是观测到的性能规律。代码锚点以本次检出的文件行号为准，路径相对 `v3/`。

## 0. 结论与六问索引

**建议用“逐配置证据账本 + 条件化见证边”替代边际范围墙；把墙的存在、当前最优点的关联、实测延迟价值拆开。** 编译探针可以发现资源拒绝，不等于正确性验证，也不保证已完整遍历所有 kernel。斜率必须来自冻结搭档后的真实试验，不来自 TPE 的边际桶，也不把同一次 trial 的 20 个 timings 当成 20 次独立实验。

| brief 六问 | 本文答案 |
|---|---|
| 1. 新墙判据与非墙 | §3：同身份、同搭档的可行/拒绝见证；非单调图、非墙与未知分开 |
| 2. 斜率、数据、CI、n | §4–5：观测数据只发现线索；冻结后最多 8 对真实 trial；固定终点区间 |
| 3. 多 knob 回退、误标 | §6：输入解释、局部轴、交互射程三层，不互相冒充 |
| 4. 空间/run 成本 | §7：批启动、逐点、真实 trial 墙钟、机会成本、准入与停止 |
| 5. 接口与兼容 | §8–9：版本化 schema、状态机、分母、重放、开关、回滚 |
| 6. 预注册与可证伪门 | §10–11：仪器正确、信息送达、run 级结果分开验收 |

不缩空间、不删 choices、不加资源约束、不重新分配空间/族的 trial 配额；投递占用原有预算。原共享内存筛查及 PRUNED 行为保持不变。只让已经实现的资源读数与延迟证据进入机制判断；不用 surrogate、预测收益、全局资源模型或资源向量加权分数。候选之间不共享墙、斜率、资源缓存或延迟证据。

## 1. 独立设计记录：先形成候选，再看旧稿

本次阅读顺序是：先尝试 Codegraph（项目无索引，回退到限定目录读取）→ 完整 brief → `stats.py`、硬/软墙、`slope_guide.py`、`correctness.py`、worker 探针及真实 eval 路径 → **在会话中记录下列独立候选** → 才阅读 `design-conditioned-walls-and-slopes.md` 全文。不是从旧稿反向组织论证。

独立候选包含四项：

1. 以完整配置和执行身份为证据主键，保存点的已知状态，而不是保存一个会被边际成功擦除的上界。
2. 以“一对仅一轴不同的具体点”见证条件墙；与当前 incumbent 的关联是另外一个字段。远距多 knob 拒绝不硬归因给每个改变过的 knob。
3. 图中允许可行—拒绝—可行；不作二分搜索、不默认单调 frontier。
4. 延迟测量采用冻结端点、有限次数真实重复；墙先记录，延迟未知时不编造收益。20 个进程内采样不能替代跨进程重复。

迫使结论趋同的证据是：边际分桶确实未固定搭档；现成消融确实固定搭档；廉价批探针与昂贵 trial 的职责不同。迫使本稿偏离旧稿的证据是：探针部分遍历、资源字段可空、真实计时缓存与去重、非单调可行性，以及选择偏差与小样本置信度的成本。逐项比较见 §12。

## 2. 可核查证据与不能越过的边界

### 2.1 代码证据地图

| 锚点 | 已核对的实现/注释 | 设计后果 |
|---|---|---|
| `src/kernel_optimizer/tuning/stats.py:44–84,95–139` | `repr(value)` 分桶后取 trial `robust_ms` 的中位；`best_value` 与最快 trial 的 `best_trial_value` 分离 | 保留旧统计作描述，不能将换一个中位数叫作条件化；不顺便修改空间扩展逻辑 |
| `src/kernel_optimizer/evaluation/wall_attribution.py:240–308` | 范围外拒绝才成墙，严格递减且 gain>0 才探针 | 新路径不再用这两个边际门 |
| 同文件 `26–47,127–162,311–327,346–398` | top-K 条件性、三值 verdict、默认 origin 控制、prompt | 保留 origin 与未知，不把 top-K 当随机样本；旧 prompt 的“所以仍有斜率”必须在新 renderer 中拆掉 |
| `src/kernel_optimizer/evaluation/soft_wall.py:11–45,230–321` | winner spill 适用门；边际 spill/latency；没有独立确认 | 保留适用门，不保留边际单调门；文档里“不能便宜探 spill”并非版本无关定理 |
| `src/kernel_optimizer/tuning/slope_guide.py:166–225,230–254,256–403` | 完成数触发、整点去重、完整 incumbent、按拒绝数值挡住其后所有 choices | `_toward_wall` 并没有逐点编译证明 launchability；新判据不能原样复用该数值上界逻辑 |
| 同文件 `405–459` | 以最好 measured trial 锚定；快照混合重算/knob 单位 | 最快 trial 是选择结果，不是无偏基准；计数器不能相加 |
| `src/kernel_optimizer/evaluation/correctness.py:170–245,247–282,284–361` | batch 无返回值，填进程内缓存；只缓存 `ok` 答案；`backend:hash(source)`；None 放行 | 新账本需要独立保存原始响应与范围；不能以进程 hash 做持久身份，不能缓存超时为否定 |
| 同文件 `31–100,407–466` | 批/单点 deadline 不同；真实 quick/full eval 使用 exclusive lane 和完整 build+eval timeout | 新诊断不能改真实 trial timeout；full_eval 不是与 quick_test 同成本同 protocol 的样本 |
| `src/kernel_optimizer/gpu/worker_main.py:713–864` | 每变体独立模块名；拦截 JIT `warmup=True`；forward 异常吞掉；只要 seen 非空可 `ok=True` | 正向超限与负向“全部可行”证据不对称；批顶层不是所有变体的 verdict |
| 同文件 `519–552,777–790` | 正常 metadata 抽取读 `compiled.n_regs/n_spills`，compile probe 读 `meta.num_regs/num_spills`；可为 None | “字段在返回 schema 中”不等于“实测总能读到”；全资源向量可用性须逐字段验证 |
| 同文件 `35–39,213–230,1855–1961,2494–2584` | runtime OOR 未专门分类；compiled/correct/latency 分开；可疑快仍 accepted and flagged | 不能只看 `ok` 或 `complete` 就将值纳入可信斜率 |
| `src/kernel_optimizer/models/reports.py:16–59` | `latency_by_value: dict[str,float]` 没有配对、来源、重复字段 | 新证据另建 schema，不往旧浮点表塞 CI/未知字符串 |
| `src/kernel_optimizer/config.py:455–560,608–704` | attribution、soft、guide 独立开关，默认关闭，K 与剂量可配 | 新路径默认 off；开关须揭示采集、复测、投递和文本四种不同干预 |

### 2.2 原始输出与失败语义

本机可核对的是 worker 的**真实返回结构和失败分支**、调用方读取方式、测试 fixture；不是本次取得的 GPU 输出。brief 的 35 个拒绝、5/5 墙被过滤、16% 波动、7ms/点等都是**交接材料报告的历史测量**，未在本机复验。不要将这些数写成“本次实验发现”。

具体已读 fixture：`tests/test_improvements.py:10502–10548,10563–10615` 的 fake worker 首次返回 `{"ok":false,"failure_kind":"timeout","reason":"exceeded 600.0s"}`，之后返回 `{"ok":true,"max_shared":393216,"kernels":[{"name":"_k","shared":393216}]}`，批成功再包 `results`。测试断言失败批不填缓存、回答后不重复 worker 调用；`8739–8759` 另覆盖恰好等于 limit、below-limit、triton unavailable、`ok:true/max_shared:null/kernels:[]`。这些是**实际测试文件里的合成响应**，不是 GPU 运行记录；本次读取但没有执行。尤其当前实现可以缓存 `ok:true` 但数值缺失的 entry，故“只缓存答案”不能理解成“每个缓存都有足够资源结论”。

单变体回答为 `{"ok": true, "kernels": [...], "max_shared": N}`；批响应为 `{**primary, "results": {path: entry}}`。必须按请求 path/内容摘要逐个关联；primary 的失败不代表兄弟都失败，primary 成功也不代表全部成功。读取失败、无 Triton、初始化失败、没触达 kernel 是 `ok:false`；metadata 不可读可能有 kernel 行但 `shared:null`。`prescreen_batch` 超时后不保证拿到已执行的部分结果，不能把提交数当回答数。

更隐蔽的是 `model(*inputs)` 的异常被吞掉（worker `828–835`）：早期 kernel 编译后，后续 kernel 可能从未触达。还可能执行非 Triton 的 torch 运算；“Triton 不 launch”不等于整个任务不使用 GPU、不分配张量或不影响缓存。输入工厂每变体重新执行，随机输入、数据依赖控制流和同进程全局状态也可能改变触达范围。

因此新增观察必须保存 `coverage=unknown|partial|validated_workload`、每 kernel specialization/call-site、输入签名、字段来源和异常信息。对旧 probe 无法重建的字段一律 unknown。已见到并核实属于真实 workload 必经路径的 kernel 超限，可作硬拒绝证据；未超限只说明“已观察 kernel 的 shared 未超限”，不是全模型可启动。对未确认的 garbage-dependent 分支，即使看到超限也先记 scoped resource observation，不直接断言该 workload 必拒。

还须区分真实零与默认零：worker `783` 使用 `getattr(meta,"shared",0) or 0`，字段不存在会变成0。旧raw response不总能区分这两种情况；新采集需 `field_present`/读取来源，不能事后把所有0追认为有效零。static-check缓存又不同于screen缓存：`correctness.py:153–168` 缓存静态检查结果，包括失败；“失败不缓存”的规则只适用于这里讨论的compile-screen答案，不泛化到所有缓存。

### 2.3 四个不同的问题

| 证据层 | 足够条件 | 不能推出 |
|---|---|---|
| `compile_observed` | 该 specialization 返回编译结果 | 全 forward 编译成功、启动、正确、计时 |
| `resource_refused` | 必经 kernel 的实际 shared>该卡 opt-in limit，或可核验的 launch OOR | 所有搭档都失败、错误计算路径也值得优化 |
| `execution_validated` | 同 workload 真 trial launch 并通过既有 correctness protocol | 延迟有效、不同输入形状也正确 |
| `latency_validated` | 上项 + 完整有限正数 timings + 同 timing protocol + 无未解释计时/工作量疑点 | 全局最优、跨卡可迁移、解除资源约束必然更快 |

正向筛查仅复用已有精确共享内存判据，不扩大其拒绝权限。新诊断账本的更严格适用性不会反过来拒绝原 tuner 可采样的配置。

### 2.4 接线路径核查：不能照着 brief 的概括接线

以下由独立只读源码核查补充；不是在本次运行中触发过的缺陷。

| 精确锚点 | 实际行为与未来接线要求 |
|---|---|
| `src/kernel_optimizer/paramspace/materializer.py:65–96,108–130` | 要求唯一模块级普通 PARAMS 赋值、精确键集合，替换后复查边界外源码未变。新点必须走这个 materializer，不能只写空间子集；当前直接 bool 字面量被拒（`40–51`），所以 bool 设计仅定义类型语义，不能承诺目前能执行 bool trial |
| `src/kernel_optimizer/models/core.py:41–52,84–108`；`control/families.py:49–111,197–204` | source SHA 与 parameter key 均已存在；结构去重有意抹掉 PARAMS 值，绝不用于复测身份。`ParamSet.key()` 是 sorted repr 的 SHA-256 截断16位；新 canonical typed key 须保存与旧 key 的映射，不直接替换旧去重语义 |
| `src/kernel_optimizer/control/orchestrator.py:1204–1280,1400–1466` | ask→复用或物化→compile screen→quick_test→profile→TRIAL_DONE→tell；复用会改 trial_id/space_id，标 `reused_measurement=True`，仍影响 told/cadence。确认必须走非 reused 路径，并保留原 measurement 来源 |
| 同文件 `930–965,2296–2335` | resume/扩展构造 measured_cache，扩展复用 witness/旧 best；不是所有历史的通用缓存。复用处没有核验完整 source/workload 同一性，安全依赖上游保证；新确认不能沿用该隐含假设 |
| 同文件 `1356–1398,1572–1574`；`tuning/slope_guide.py:275–287` | **S7 直接用 find_walls/select_for_probing，不等待2e消融归因成功**。因此“2e 归因失败导致 S7 不能 enqueue”不是当前调用依赖，只有相关现象。新共享 ledger 要显式接到两消费者，不能只改归因出口 |
| `src/kernel_optimizer/tuning/tpe.py:83–157,178–220` | ask 内的 seen/guard/deweight 拒绝为内部 PRUNED；只有返回配置计 `_asked`。enqueue 无剩余预算检查，Optuna 跳过已等待重复时 wrapper 仍可返回 accepted。新的 planned/accepted/started 三层不可省略 |
| `control/orchestrator.py:1782–1823`；`tuning/tpe.py:95–122` | cache-only guard 知道的不可行点不都成为 TrialRecord；现硬墙输入主要来自 history 的 `infeasible_shared_memory`。新 ledger 须同时订阅 prescreen逐点回答、guard引用、post-materialize refusal 和 launch诊断，按配置去重，不能只重放 TRIAL_DONE 算全部拒绝 |
| `control/orchestrator.py:1655–1779` | top-K 按 trial 排，未按完整配置去重；default origin 是域首 choice，不必等于 PARAMS literal 默认。新 K 先按真实 measurement/config去重；default 的合法性/可运行性仍要验证 |
| `evaluation/wall_attribution.py:137–149,386–389` | `n_origins_probed` 包括 undecidable，而 prompt 的剩余 origin 被描述成“不触墙”。新 renderer 分开 attempted/answered/attributed/unknown，不能复用这句否定 |
| `control/orchestrator.py:1733–1738,1886–1904,3050–3072` | 物化失败会跳过；后续未发现墙不一定清除已存文本；rewriter 接 parent 存储 wall_text。必须对每次消费验证 parent source/epoch，显式清空无效文本，并写 payload 引用 |
| `src/kernel_optimizer/evaluation/profilerx.py:54–66,99–114`；`evaluation/statics.py:121–148` | profile 的 spills/regs/shared 分别取各 kernel 的 max，最大项可能来自不同 kernel；n_spills footprint 与静态 STL/LDL 次数不同。只保存 aggregate 无法证明 scoped spill 曲线 |
| `src/kernel_optimizer/gpu/worker_client.py:42–69,202,272–325` | job_wall_s 从取锁前开始，包含等待/cooldown/执行/timeout清理，排除更早 job文件/命令构造；subprocess timeout 不覆盖锁等待；锁按 jobs目录，不是跨run全局锁。两臂分目录不等于GPU隔离 |

本表中省略 `src/kernel_optimizer/` 的源码路径均以该目录为基准。

还有一个**静态路径契约风险**：worker_client `272–277` 转换三个标量路径，而不是 `extra_kernel_src_paths`；worker batch 结果按 worker-side path 返回，evaluator 按 host-side path 查询（`correctness.py:225–228`）。fake worker 原样回显路径的 fixture 不能证明 WSL roundtrip 正确。新 schema 应以 stable request_id 关联变体，保存 host/worker path 映射；未实机复现前称风险，不写成已发生的漏报。

这些接线要求属于后续实施面，不在本次改生产代码。`tests/test_s7_slope_guide.py:686–794`、`tests/test_reused_trial_artifact.py:124–179`、`tests/test_2e_wall_attribution.py:813–844,909–920` 提供可沿用的预算、复用和按物化源码识别 origin 的测试模式；不能拿其通过与否替代 GPU 可行性验证。

## 3. 问题一：墙是有条件的见证，不是参数固有上限

### 3.1 身份与点

定义执行身份 `I = (run, candidate, source_digest, reference_digest, workload_signature, backend, compiler/toolchain, device/limit, correctness_protocol, timing_protocol)`。这里 source_digest 是冻结候选原始源的完整摘要，不是抹掉 PARAMS 的结构指纹；每个点另有自己的 materialized_digest，不能要求 p/q 的物化摘要相同。其中 workload 包括 shape、dtype、stride、设备、输入/初始化工厂摘要与 seed 策略；不能只用 task 名称。配置 `x` 必须是**最终 materialized PARAMS 的完整映射**，包括不在当前空间的固定默认值，保留类型（`true`、`1`、`1.0`、`"1"` 不混同）。另存空间版本与声明域类型/顺序。源码路径不是身份，结构源摘要不能替代物化源摘要。

点状态不是一个 bool，而是 §2.3 的多个证据字段。缓存命中只引用原 `measurement_id`，不生成新测量。结构改写生成新 candidate/source 后禁止继承父的资源结论；只能把它当改写假设的来源链接。

### 3.2 墙的定义与强弱

给定身份 I、完整搭档 `c=x[-k]`，存在 p、q，且 `p[-k]=q[-k]=c`、`p[k] != q[k]`：

- q 有可核验的 `resource_refused`；p 有同 workload 的 `execution_validated`，则得到 `witnessed_conditional_wall(k,c,p,q)`。若 p 只有编译范围内未超限，降级为 `resource_transition_observed`，等待真 trial，不伪称可行端已证明。
- 若 p、q 是有序域相邻 choices，称 `adjacent_witness`；若中间未观测，称 `bracket_witness`，中间全部保留 unknown，不能定位“第一个失败”。
- `relevant_incumbent_adjacent` 还要求 p 的完整配置等于当前已验证 incumbent，q 为其相邻 choice。预先冻结的 top-K 非首位 origin 可另记 `relevant_topk_adjacent`，但不能混作 incumbent 一步触墙；top-K 的“高性能”只是实测排序，不是统计等价认证。只在其他搭档下成立的墙仍是历史见证，但不是“当前最优点被截断”。
- 与 incumbent 共搭档但隔了多个 choices 的拒绝，仅为 `incumbent_conditioned_bracket`。沿已知可行段发现相邻边，也须写清离 incumbent 的距离，不能与 incumbent 一步触墙混报。
- 只有远距拒绝且各单轴都未形成见证时为 `multi_knob_witness` 或 `unresolved_refusal`，不把拒绝里的所有 knob 都列成墙。

存在性不依赖延迟斜率。即使走向墙更慢，该资源见证仍存在，但其重写价值为 `worsening` 或 unknown。措辞是“在 c 这些搭档与 I 这个 workload 下，p 可运行而 q 因某资源拒绝”，不是“k 最大只能取 v”。

### 3.3 非单调、有序单位、类别和方向

保存有限域上的点图，不保存一个隐含单调性的数值上界。例如（**逻辑构造，不是 GPU 数据**）choices 为 `[1,2,3,4]`，状态为 `[valid,refused,valid,unknown]`：记录 1↔2、3↔2 两条局部边，4 未知；不删 3、4，不因 2 失败而二分或停止考察远端。即使完整扫描，也只能称“已扫描离散域内的多个可行分量”，不能外推域外 frontier。

候选扫描顺序：已有具体拒绝附近的单轴对 → 当前 origin 两侧相邻 choices → 在余下诊断 cap 内按固定轮转补点。顺序不按边际 gain 或拟合资源预测排；已测数字只作资源判定、实际延迟比较和成本准入。相同拒绝在多个采样事件出现只算一个配置见证，不能靠重复次数把轴顶到优先队列。

轮转实现提案：每个优先级内按稳定wall/axis ID字节序排列，持久化cursor；每次从上次位置之后取首个合格项。首次cursor、origin K（首轮提案1）、方向与所有ID算法写入manifest。run确认名额给按事件顺序最早满足准入的两个不同candidate；不在看到结果后回选“代表性候选”，并披露因此偏向早期、便宜候选的覆盖局限。

numeric 不自动意味着有序：数字编码的算法类别也是 categorical；没有明确 order metadata 的旧域，斜率为 `axis_order_unknown`。bool 允许做条件化开关对照、允许记录拒绝边，但无 high/low 或斜率；category 类似。不要把“类别无斜率”误写成“类别不能涉及资源墙”。

对明确有序域，保留 `index` 顺序及原始数值。方向为从可行内侧 p 朝 q 的 `lower|upper`；两侧分别处理。斜率单位默认 **每一个声明 choice 步长的 log 延迟改善**，不是每个原始整数单位；若另报数值割线，必须给 `ms/原始单位` 与有符号横坐标差，不跨不同 knob 比陡峭程度。倍增域 `[32,64,128,256]` 的每级可理解为每倍增，但只有确实等比时才这样命名。

### 3.4 stale anchor 与非墙

任何完整配置变更均产生新 `anchor_epoch`，即使新旧 latency 差小于 4%。墙随搭档而变，不随“最优改善是否显著”而变。旧证据不删除，仍属于旧 epoch；新 epoch 未补证据时为 stale，不借噪声门继续套用旧墙。源码、device、workload、compiler 改变更必须失效。

有效的否定包括：指定对的两端都 execution-valid（`not_wall_on_pair`）；完整指定轴所有 choices 都验证有效（`no_wall_in_scanned_domain`）；拒绝在其他搭档下成立而当前单轴变更有效（`not_attributed_at_origin`）。只扫一部分时不能宣称整轴无墙。

超时、materialize 失败、OOM、语法/编译错误、correctness mismatch、shared 缺失、路径不完整均不是硬资源墙，分别记失败或 unknown。全轴都拒绝但没有可行端是 `all_observed_refused/no_valid_anchor`；如果其中包含此前同身份验证有效的 incumbent，则是 `evidence_conflict`，优先查身份、非确定行为或 probe 路径，不能当作“最强墙”。

## 4. 问题二：测局部割线，不测采样史

### 4.1 数据来源及选择偏差

三类数据严格分流：

1. **历史观测配对**：同 I、相同完整搭档、只动一轴的两个真实 trial，可发现一个局部 contrast。TPE 自适应选择、相隔很久、最快点 winner's curse、复用缓存等均使它不自动成为确认样本。旧 log 缺 pair/time/provenance 时仅 `exploratory`。
2. **编译观察**：提供 scoped 资源事实，不提供 latency；零 launch 不能拿来算“走向墙更快”。
3. **前瞻确认**：在 freeze 事件后，用新的真实 trial 同时重测 A 与 B；历史 incumbent 的最佳一次 latency 不进入确认估计。选择哪个轴可以依赖此前已测证据，但不能用确认批中结果重新选择端点或增加样本。

两点割线不证明一条曲线单调，也不预测失败点 q 如果被“解墙”会有多快。若 A 已是墙前最后一个可行 choice，则选同搭档的一个内侧点作 A、墙前点作 B；这种测量用于解释重写，而不是投递一个新的极端配置。若没有第二个可验证内侧点，则无斜率；不得把失败点 latency 设为无穷大。

### 4.2 定义、每点 n、区间

冻结 A 为远墙侧、B 为近墙侧，二者只动有序 k。第 i 个独立进程配对的 trial 内中位数为 `a_i,b_i`，均经过原 correctness 与 timing 路径。定义：

`d_i = log(a_i / b_i)`，`D = median(d_i)`，`gain = 1-exp(-D)`。

`slope_step = D / abs(index(B)-index(A))`；D>0 总表示**向墙走更快**，因此 lower 墙也不翻错符号。若需绝对割线，另报 `median(b_i-a_i)/(value(B)-value(A))`，明确它不是上述无量纲改善率，默认不用于跨轴排序。

披露 `n_trials_A=n_trials_B=8, n_pairs=8, n_samples_per_trial=20`，并列出实际每 trial 的样本数、缺失、measurement_id 与 pair_id。20 次 CUDA timings 是进程内子样本；不能拼成 n=160、bootstrap 成 160 个“独立试验”，不能将 reused trial 加入 n。样本标准差 16% 不等于两端 medians 的标准误，历史 ±2–4% 也不是自动成立的 95% CI。

**主区间采用配对 log 比的保守、非参数次序统计区间**，不依赖 n=8 时无法检验的正态性。对 m 个独立、同分布或有共同中位数且满足所用尾界条件的 pair contrast，选最大整数 k≥1 使：

`2 * sum_{r=0}^{k-1} C(m,r) / 2^m <= alpha_local`，区间为 `[d_(k), d_(m-k+1)]`。

无合格 k 时为无界区间、结论 unknown。保留 ties，使用闭区间与保守覆盖，不把相等值随机抖动成方向；这是对 pair contrast 的中位数推断，**不是“两组总体中位数之差”定理**。NIST 次序统计/二项式构造为依据 [S2]；本文不用该网页的插值近似公式。

首轮提案每 run 最多两次确认 episode，每次只检验一条冻结边，`alpha_run=0.05`，`alpha_local=0.025`，family 包含 hard/soft、上下方向、所有候选和空间的确认 episode；选择后才生成全新的确认数据。用 Bonferroni 保守控制两次确认结论的 run 内家族错误 [S3]，不要求两 episode 互相独立，但每个区间自身的采样条件必须成立。未完成 episode 也消耗一个确认名额，不转让 alpha；不在反复重算时重开同一假设。

对 m=8，k=1，区间 `[min(d),max(d)]` 覆盖率为 99.21875%，比要求的 97.5% 保守。m=4 的极值区间仅 87.5%，m=6 为 96.875%，m=7 为 98.4375%。本方案仍固定收满 8 对才判决，不能看到第 7 对已满足就停。小样本 bootstrap 不是这个离散证据不足的补救。

### 4.3 噪声门、等价与 unknown

实用最小效应 `delta=log(1.05)` 是**提案阈值**：5% 对数对称倍率带，受 brief 的复测波动量级启发，尚未校准；换算成 gain，正侧门为 `1-1/1.05≈4.762%`，不能误称双侧都是相同的百分比改善。它表示实际关心的差异，不是假定噪声只有 5%。跑前用独立 A/A 测量验证测量协议与配对漂移；若 A/A 的预定噪声检查不能支持本 protocol，本轮不发 slope certificate，而不是调高/调低 delta 直到结论出现。重新校准需新预注册。

A/A 预检也是有界实验：另行批准的一次 protocol-control run 中，仅对一个预先指定、已正确的测试配置执行8对=16个真实trial，计入该run原B，沿用5%墙钟软cap，失败/不足即停止、不重做直到通过。通过门为全部有效且极值区间落入±delta，并通过§5.3质量规则。这只是仪器负对照，不为其他候选提供斜率或数值噪声估计；各候选仍只用自己的8对证据。本次没有做该预检，也不把它隐含成免费外部校准。即便预检通过，也不能证明新候选的pair独立性。

| 主区间 `[L,U]` | 判决 | 对消费者 |
|---|---|---|
| `L>delta` | `improving` | 允许报告该已测段改善及 CI；不外推 q |
| `U<-delta` | `worsening` | 该段向墙更慢，不据此要求解除该墙 |
| `L>=-delta && U<=delta` | `practically_equivalent` | 该 protocol 下没有超过预定尺度的效应，不等于数学斜率为零 |
| 其余或质量检查失败 | `unknown`，附具体原因 | 不给 agent/排序一个斜率数字；不叫 flat |

等价的区间包含规则比通常 TOST 更保守；等价界必须预先设定，非显著不能证明等价 [S4]。区间排除 0 但尚未越过 delta、跨越等价带、缺重复、漂移、错误端点，都是 unknown。审计事件保留原始读数、区间和阈值以便复查；消费者的 `certified_gain/slope` 必须为 null。这是“噪声内不报数字”的适用边界，不删除复验所需原始证据。

## 5. 固定 trial 配额下的可执行测量日程

### 5.1 不假装兼得高覆盖、低剂量与强推断

**首轮提案：每 candidate 最多一次确认（跨扩展空间累计），每 run 最多两次，每次固定 8 对=16 个真实 trial 槽。** 40-trial 空间最多消耗 40%，80-trial 空间最多消耗 20%。这很贵，故默认只开证据采集；确认单独开关、成本准入。不是每轴来 8 对，更不是每次重算都重复。原各空间/族的 B 和总预算不变；这些是投递剂量上限，不预留新的空间配额。

若评审坚持现有 `2/10` 小剂量且不增加确认调度能力，则选择**仅做观测型局部 contrast，斜率为 exploratory/unknown**。不能同时宣称沿用整配置去重、只投一两个新点，又取得多个真实重复的 CI。这个取舍必须在跑前选定。

普通新点建议与确认重复分开：首轮确认实验将普通新guidance关闭，以免混淆干预。后续单独开启普通guidance时，提案剂量为每10个told最多2个、每空间合计最多6个，仍从原B取槽，并与确认**共享**§7的5%新增测量墙钟软cap；不能在确认之外另领一份免费trial时间。无certificate的初次定向点仅标 `evidence_acquisition`，不按未知斜率数值排序。任何实测排序只能比较已有合法证据，未测点之间按固定轮转。所有上限只控制本机制建议，不阻止原TPE采到同样的合法配置。

### 5.2 时序与调度表（均为提案）

| 时点 | 行动 | 硬上限与退出 |
|---|---|---|
| 每累计 10 个 told 到达一次检查点 | 读取原始证据、检查 anchor epoch；不把 asked/pending 当 measured | 每空间最多两次新诊断批；无新身份/点则缓存读取，不发 worker |
| 第一个合格检查点 | 先合并 origin 邻点、拒绝邻点、控制点，物化并去重后编译 | 每批最多 48 个未回答点；最多两批/空间；跨空间/候选有 run 诊断墙钟 cap |
| 满足资源见证且找到两个候选端点 | 以固定轮转选唯一轴/方向，冻结完整 I、A、B、8 个 pair seed/order、所有阈值及事件前缀 | 同空间至少剩余 20 个可 ask 槽（16 测量 + 4 普通机会）；pending 也扣除；至少还有 §7 的墙钟余量 |
| freeze 后 | 每对 A/B 各发一次完整真实 trial；8 对的次序各自独立公平随机为 AB 或 BA，并记录生成 seed | 最多 16 次尝试；成功返回的 ask 计原预算（内部拒绝见下文）；无额外 correctness-only/免费计时旁路；不因有利中间结果提前判决 |
| 每对之间 | 允许原调度运行，但每对两次 timing 必须相邻，暂停本 run 的诊断 shared probe，复用 exclusive lane | 其他 job 插入、设备环境改变或 pair gap 超标则 episode invalid/unknown；不补样本、不删除坏对后继续取到显著 |
| 第 16 次完成或安全中止 | 固定终点计算一次判决，更新局部证据 | 未满 8 个有效 pair 就 unknown；不换 B、不加重复、不再占该 episode 名额 |
| 结束之后 | prompt 可消费有来源的资源事实与已确认割线；普通 advisory suggestion 仍整配置去重 | 本轮不对未测延迟作数值外推；再次行动须当前 epoch 匹配 |

确认的两端初次可以只有 scoped compile-fit，但一旦任一真实 trial correctness/launch/timing 失败即中止，资源观察保留，不能只对“成功存活的几对”报告斜率。现有 PRUNED 若拒绝了测量端点，也记 `blocked_by_existing_policy` 并结束，不绕开 PRUNED。已返回的真实 ask 消耗原trial槽；ask内部guard/deweight拒绝仍按原规则不增加 `_asked`，但计入本episode的16次尝试上限及实际墙钟，不补发替代点。原提前收敛/空间结束/run结束优先于确认计划；20槽准入检查不是锁定配额保证，剩余机会被原调度消费后本episode只能中止。

新 incumbent 在确认期间出现时，不改变已冻结 A/B。若 I 没变，完成后可出**历史冻结 contrast**，但当前 prompt/投递必须另检 anchor；若结构或 workload 变更则终止。不能把新最快端点换进去保留前半段样本。

### 5.3 pairing、漂移、工作量与重测缓存

“参数相同”不能证明是自然配对 [S1]。每 pair 采用同输入生成 seed/初始化策略，同设备、precision、计时方法、correctness 模式、warmup/L2/cache policy。随机 AB/BA 缓解一阶顺序偏差，但不消除热漂移、carryover、后台进程、共享锁外的 GPU 使用；记录实际 pair gap、作业启动结束、顺序、负载与可用温度/时钟 telemetry。

**提案质量规则**：pair gap 不超过 freeze 前同 candidate 已测 real-trial wallclock 中位数的 2 倍；已有有效实测基准不足则不准入。freeze 前后设备/workload/protocol 变化为 invalid；episode 内独立于效果方向的运行日志出现外部并发/错误/重试为 invalid。比较 A 在前 4 对与后 4 对的中位相对变化，若超过预设 5% 漂移界则整体 unknown。该检查不能证明独立性；即使通过，也须明确“CI 依赖残差稳定及跨 pair 独立假设，GPU A/A 验证未完成”。若长期漂移是共因，8 对并非有效 n=8，应降级而不是算漂亮 CI。

**必须有显式 `measurement_intent=confirm_repeat` 与唯一 `replicate_id`**，通过未来调度接线占用原 ask 槽，绕过的仅是“已有 latency 因而直接 reuse”的优化与普通 suggestion 去重；不绕过 correctness、资源筛查、PRUNED、合法域或预算。不能给 PARAMS 添假 knob 骗过 hash；不能使用缓存记录伪造重复。普通探索点去重仍按完整配置；确认任务的幂等键为 `(episode,pair,endpoint)`，同一个任务重放不再执行。

具体实现契约是给**新意图**提供预算计数的显式固定点入口，而不是改普通 `ask/enqueue` 对重复点的 PRUNED 分类；它仍受相同 guard/deweight 与失败到Optuna状态映射约束。`tpe.py:178–220` 中 shared/guard/materialize→PRUNED，runtime/correctness/OOM/timeout→FAIL 的现有映射及普通 duplicate→PRUNED 保持不变。若无法在不改这些行为的前提下接入，`slope_measurement.enabled` 不得开启；这正是“设计可提、当前执行器不能直接复用”的实施阻断项。

确认批不能同时按最小一次 latency 选赢家，再用该最小值证明效应。确认估计只看冻结 D；tuner 原 objective/选择不在本任务里重写，但原最小值赢家的偏差必须在报告中标为 selection。终局胜负使用预注册的最终验证路径与独立 run 配对，不以“本次投点暂时刷新 incumbent”替代。

## 6. 分层组合：拒绝恢复、交互回退与软墙

### 6.1 L0：恢复遗漏的 launch 资源证据

`_classify_exception` 只区分 OOM/runtime_error；strict 路径 `_classify_eval_failure` 另有自己的分类分支（worker `35–39,213–230`）。后续修复必须覆盖两条路径，不能只改一个函数。新增旁路解释字段 `resource_failure`：异常类型/模块、发生阶段、原始消息摘要、resource_kind、required、limit、kernel、workload 身份。只有明确 OOR 且资源种类/数值可核验才入硬拒绝账本；普通 OOM、非法地址、timeout、编译失败都不得猜成 shared 墙。

首轮可只增诊断规范化字段，保留 legacy `failure_kind` 与 Optuna 状态，避免把“墙输入修复”混成采样策略改变。缺原始消息的旧 runtime_error 保持 unknown。brief 的“约 2%”是历史材料口径，本机未复验，不能作为当前漏报率。

### 6.2 L1：局部轴见证；L2：多 knob 射程

单轴扫描不要求先有自然拒绝，可以对 incumbent 邻点发诊断；它也不会自动覆盖所有距 incumbent 5–6 维的拒绝。多 knob 回退单独开关、共用批处理与身份账本：从真实拒绝 r 出发，以已验证 incumbent a 为回退目标，令 D 为两者不同的维集合。

维护“当前仍拒绝点”r'；每层把 `r'[j]=a[j]` 的所有单维回退点跨墙合为一批。若某回退仍明确拒绝，可按固定维序选择它，删去该维继续；若每个单维回退均经真实可行验证，得到 **1-minimal refusing subset relative to this a/r path**；只有编译未超限时称 `screen_1_minimal`。若某点未知，不能声称 minimal。没有拒绝可删但多维同时回退可能可删，说明 1-minimal 不等于最小基数，更不等于全局唯一绑定集。

非单调情况下不做“失败越多 knob 越贵”的假设。也可记录回退路径中仅一轴不同的局部跨越边，但其搭档是 r'，不偷换成 a。不要求穷举所有子集；**提案上限两层、共享剩余点/批 cap，超过记 `interaction_unresolved_budget`**。这会牺牲最小化完整性，明确披露；若未来选择更深层数，是独立剂量实验。

与待办账对齐：`docs/next-round-changes.md:706–747` 已撤回“从最优点消融 over_ratio<1 就说明原拒绝不是资源墙”的解释；`758–804` 的回退量价要求跨墙按依赖层批处理，brief `183–187` 引述每层合批123s、每墙一批1.21h。后两数仅是历史材料报告，不能作本方案两层cap的实测价格。`749–755,835–841` 还记录过分组键让 crossing 结构性为0、正则不匹配真实消息的错误；新验证必须先查 raw payload，再谈机制覆盖。

真实二阶交互证据可来自同身份四角 `(a,b),(a',b),(a,b'),(a',b')` 的状态模式；如三个有效、合改拒绝，记录具体四角，不给两个轴各贴独立硬墙。延迟交互若无四角受控真实重复，只能称资源交互，不能给交互收益数字。

### 6.3 L3：软墙是 scoped spill 证据，不是独立硬限制

先按当前 source/workload 已测最优 trial 查 spill：明确 0 为 `not_applicable`；缺字段为 `applicability_unknown`；明确 >0 才 `applicable`。这个门保留，但“最优 trial 有 spill”不证明 spill 导致延迟损失。top-K 的适用情况另外列，不拿次优点 spill 为零/非零替换 winner 的分母。

只比较**相同逻辑 kernel/call-site、相同 workload 作用域与量纲**的 n_spills，并记录两端各自的 specialization（knob 改变本来就可能生成不同 specialization，不能强求编译实例相同）；保留每 kernel 的 `{shared_bytes,n_regs,n_spills,num_warps,num_stages}` 向量，不加权合成。若配置改变导致 kernel 集/作用域变化，标 `kernel_scope_changed`，先解释结构差异；不能将两个不同 kernel 的 max-spill 拼成同一条曲线。CUDA/CUBIN 的 STACK/LOCAL 推导值与 Triton n_spills 不能无转换校验直接混用。

compile-only 路径可能提供 scoped spill，也可能为 null；正常 profile 可作为同 scope 实测来源。Triton 官方 softmax 示例在 warmup 后 `_init_handles()` 再读 `kernel.n_regs` 和 `kernel.metadata.shared` [L1]，不能据此保证本仓库 `meta.num_spills` 每版可用。**不在本次实现 `_init_handles` 补丁**：加载句柄是否引发资源检查、成本如何、与正常 profile 是否一致均须 GPU 验证。

新软墙输出分为 `spill_observed_at_anchor`、`conditional_spill_change`、`measured_latency_contrast`。仅要求局部对有真实 spill 变化，不要求整轴单调；低侧/高侧都可升 spill。编译反复读到相同值不是独立性能确认。即使“spill 增加且向该端更快”成立，也只提示资源权衡；不写“降低 spill 必然更快”或通用“local memory 慢 100 倍”。资源转移到 shared 或全局内存的建议须作为待验证假设，各维实测变化分开报告。

## 7. 成本：真正要付的是 run 墙钟

### 7.1 核算式与历史价格的边界

令 S 为实际进入诊断的空间数，b_s 为新启动批数，u_sb 为去重且未命中缓存的点数。成本应记录：

`C_probe = sum_batches(startup + sum_point_compile + materialize/import/transfer + queue/lock_wait + cleanup)`。

`C_confirm = sum_actual_jobs(job_wall_s) + scheduler/materialization/event overhead`，包括失败、超时、PRUNED 前消耗、重试及终止清理。**不能用 kernel latency_ms ×20 来代替，也不能因为 16 次来自原 B 就填 0。**

这里 `sum_actual_jobs` 必须涵盖 static-check、compile-screen、eval 等**去重 job ID 的整条链**，不是仅相加 `TrialRecord.job_wall_s`：`orchestrator.py:1420–1465` 通常只把返回 eval job 的 stamp 写到 trial，screen-refused trial 也不自动带整链墙钟；复用 trial 还可能复制历史 stamp。新成本账应补主机 monotonic 全链 start/end 和子job链接，使用并发区间的并集统计run占用，另列作业总服务秒，不能重复累加重叠锁等待。缺字段时报告 unaccounted，不将事件间隔当单job用时。

历史参考仅用于敏感性：`correctness.py:170–186` 报告 48 点 11.02s、约 7ms 边际；`31–62` 同时记录多 kernel 的 40 点批为 76–260s、另有 490s 回答批和 1200s 无回答尾部。`config.py:469–506` 的 18 点 8.8s、0.49s/点是批平均而非可识别的边际价格。不能把这些不同卡/候选/冷热缓存的数拼成普适常数。

按“11s 启动 + 0.007s/点”的**参考代入算术，不是本次观测**：24 点 11.168s、36 点 11.252s、48 点 11.336s；两批各 48 点是 22.672s，不是一批 96 点的 11.672s。12 个轴、每轴 4 choices 的完整单 origin 去重上限为 `1+sum_k(|D_k|-1)=37` 点，而非默认 24；K 个 origin 再按完整配置合并，不能想当然线性免费。

### 7.2 run 级敏感性

若每空间两批各 48 点，参考 C_probe 为 `22.672*S` 秒：S=10/30/60 时分别 226.72/680.16/1360.32 秒，占 12h 的约 0.52%/1.57%/3.15%。若实际每批 150 秒，则同样三种 S 为 3000/9000/18000 秒，即 6.94%/20.83%/41.67%。这是说明尾部能吞掉 run 的情景算术，不是预计发生率。

| 假设每次真实 trial 墙钟 T（非观测） | 一次 16 槽确认 | 两次 run 上限确认 | 占 12h 墙钟 |
|---|---:|---:|---:|
| 20s | 320s | 640s | 1.48% |
| 60s | 960s | 1920s | 4.44% |
| 180s | 2880s | 5760s | 13.33% |

这些是确认的**总占用**，不是因果增量：它替代了本会发生的 TPE draws。若其中重复 incumbent 本来会命中 measured_cache，增量可能更接近总占用；若替代昂贵失败点，增量也可能负。没有反事实实测就不宣称净节省。报告 `C_new_probe`、`C_real_confirmation`、完成普通 trial 数、unique 完整配置数、完成空间/族数、最终质量；分配公平不能只看名义 B 相同。探针预热编译缓存也会改变后续 trial 价格，作为 treatment 效应计入，不能“扣掉”后宣称免费。

### 7.3 操作性 admission/stop（提案，不是预测 gate）

- 每空间最多两批、每批 48 点；诊断总墙钟 cap 为 `min(0.03*run_budget_s, 900s)`。含回退、软墙重探、终扫及控制，不能每个模块各领一份。12h 时为 900s（2.08%）。
- 每次诊断 deadline 不超过现有 `prescreen_timeout_s(cfg,n)`，也不超过剩余诊断 cap 和 run remaining；批用批 timeout，不误用 `screen_timeout_s`。预计耗时不是准入事实；直接为当前允许 deadline 留足余额。无余额即 skip，unknown，不改搜索域。
- 确认总实际墙钟 cap 为 `0.05*run_budget_s`（12h 为 2160s），包括两次 episode。需已完成同 candidate 至少 5 个真实、非 reused trial，使用已观测的最大全链 trial wallclock（不能仅取eval stamp）作为参考 T_ref；只有 `16*T_ref` 能放进剩余 cap，且 run remaining 留出既有最终验证/关闭所需预算才准入。T_ref 是观测成本参照，**不是最坏耗时保证**，实际每次启动还检查 remaining。该规则只控制新增诊断准入，不改原候选排名或空间/族预算分配。
- 原 trial build/eval timeout 不缩短。本机制每run最多一个在途新增测量job、一个active episode；若一个已启动 trial 越过确认软 cap，等它按既有规则结束，之后不再启动新确认；这会造成至多一个在途真实 trial 的超额，另计清理/锁等待。要硬 cap 必须安全取消且改变 trial 语义，本稿不采用。probe 的 deadline 也须计清理尾部，不能假装数学绝对上界。
- 无本地同 candidate 成本样本、unknown workload identity、缺有效 anchor、剩余槽不足、两 episode 已使用，均不准入。没有 GPU 遥测也不能假装已确认隔离；先 shadow 采集后再开确认。
- 不因“斜率差一点显著”再投；不因当前 treatment 落后提前结束对照。安全停止、成本停止与统计终点分开，停止的 episode 为 unknown，保留预算损失。

## 8. 数据契约与状态机

### 8.1 新 schema（设计字段，不是已添加模型）

所有新增事件使用 `schema_version=2, criterion="conditional_witness_v1", protocol_version, config_digest, event_id, parent_evidence_ids, source_seq_cutoff`。事件名中不编码成功/失败，判决在 payload。

稳定key提案：typed_params按键名排序，每值编码显式type与无损值，以UTF-8规范序列化后取完整SHA-256；禁止NaN/Infinity；保留原literal与legacy ParamSet.key映射。`wall_id`由I、partner_key、knob、p/q配置key、resource_kind生成，不因多次观察换id；`observation_id`与`attempt_id`另行唯一。若域允许数值等价但类型不同，须先证明物化语义一致才去重；默认不合并。schema版本变化不能悄悄更换key算法。

| 对象/中性事件名 | 必需字段与空值语义 |
|---|---|
| `POINT_OBSERVED` | I、space_version、完整 typed_params、config_key、materialized_digest、measurement_id、job_id、origin、phase、raw_result_ref、cache_hit/source_measurement_id、coverage、每 kernel 资源字段及 units/provenance；None 不是零 |
| `WALL_EVALUATED` | wall_id、anchor_epoch、origin_kind、axis_kind、knob、direction、p/q evidence IDs、partner_key、adjacent/bracket、witness_kind、relevance、resource_kind、required/limit、coverage、verdict/reason、unknown choices |
| `MEASUREMENT_PLANNED` | episode_id、freeze_seq、I/A/B、8 个 pair 计划与 order/seed、预算槽/墙钟额度、alpha_local、delta、质量规则、选择依据、ordinary/confirm 意图 |
| `MEASUREMENT_EVALUATED` | episode_id、所有 planned/started/completed/valid pair IDs、实际 n、per-trial sample n、CIs/算法/实际覆盖率、quality、判决/reason、raw contrast（仅审计）、certified contrast（unknown 时 null）、consumed_wall_s |
| `GUIDANCE_EVALUATED` | proposal_id、origin evidence、全配置、是否当前 epoch、intent、accepted/rejected/started/completed/reused、exact refusal reason、trial_id |
| `DIAGNOSTIC_CHECKPOINT` | run/space/candidate/epoch 作用域、点/轴/批/episode 的独立计数块、预算 used/remaining、snapshot_scope、counter_epoch、last_seq |

每个 kernel 字段保存 scope；`max_shared` 可以用来判断“任一必经 kernel 超限”，不能把它和另一个 kernel 的最大 n_regs 拼成一个物理上不存在的 kernel。资源向量仅用于审计/解释，不跨维求和排名。prompt 继续遵守既有原始资源可见性限制，不借新 schema 泄露候选绝对 latency 排名或另造综合资源分数。

上游依据为 `docs/v3-design-resource-ratio-and-conversion-efficiency.md:3–11,95–101,117–143,160–172,215–220`；该文部分已被后续设计替代，不拿旧“兑换率”当新要求。`docs/v3-revised-design-and-implementation.md:15–33,84–106` 禁止显式资源变化表/资源对兑换率送给agent。本稿每kernel向量与变化保留在审计侧；prompt只给获准的条件墙事实、过门的局部延迟趋势及行动假设，后续以实际改写对账，不制作新的资源兑换表。

### 8.2 状态转换

点：`unseen → planned → observed|attempt_failed`；observed 带编译/资源/执行/计时多个字段，attempt_failed 不写入答案缓存。任何同身份相互矛盾证据 → `conflict`，只停诊断使用，原 trial 流程照旧。

墙：`unresolved → resource_transition_observed → witnessed_conditional → relevant_at_epoch`；epoch 改变 → `stale_for_current_anchor`，历史 witness 不撤销；具体对被验证双可行 → `not_wall_on_pair`。资源墙与 latency state 正交，不能让 worsening 删除资源事实。

确认：`eligible → frozen → queued → collecting → completed|aborted_budget|aborted_quality|aborted_policy`；仅 completed 且满 8 有效对计算一次 certificate；任何 aborted 为 unknown。重启后 interrupted episode 不用剩余样本接着拼成“连续配对”；只重放已有证据，不补试验。

建议：`proposed → accepted|declined → started → completed|failed|reused`。accepted 不是执行，执行不是正确，correct 不是有效 latency。confirm-repeat 若落入 reused 为仪器缺陷，不能计入 n；普通 suggestion reused 是可观察结果，不新记测量。

### 8.3 分母纪律

| 读数 | 分子/分母单位 |
|---|---|
| 探针回答率 | 有可用 scoped 回答的 unique 点 / 实际提交 unique 点；另报 cache hits/请求点 |
| 非墙率 | 明确否定的轴/对 / 已获得足够证据的指定轴/对；unknown 单列，不塞分母当否定 |
| incumbent relevance | 当前 epoch 相关见证边 / 所有历史 witnessed 边；另报有相关墙的 unique candidate/eligible candidate |
| 软墙适用率 | winner 明确 spill>0 的 candidate-epoch / winner spill 有测量的 candidate-epoch；缺测/所有 epoch 另报 |
| 斜率可确认率 | completed 且非 unknown episode / planned episode；另报 completed/planned 与各失败原因 |
| 投递漏斗 | accepted/proposed、started/accepted、有效 latency/started；每层均为 proposal/job ID，不混 knob |
| prompt 覆盖 | 收到有证据文本的 unique family / 实际有改写请求的 family；另报 /全部 eligible family |
| 改写转化 | 子代通过独立终局比较 / 有完整 parent→wall→prompt→rewrite→child 链的改写；未送达不叫转化失败 |

累计快照按 `(run,space,instance,counter_epoch)` 取最新值或同实例差分，绝不跨事件相加；跨空间对 candidate 率去重。top-K origin 不是 K 个独立候选，重复编译不是 K 个独立资源事实。具体统计同时报告“计划/尝试/回答/有效”四个分母，避免将 missing 当零。

## 9. 消费、兼容、单变量开关与回滚

### 9.1 对消费者的最小改变

rewriter 示意模板（非实测数字）：

> 在候选 C、源码摘要 S、workload W、冻结 epoch E 的这些搭档下，p 的真实 trial 通过验证；仅把 k 从 p[k] 改为 q[k]，必经 kernel K 的 shared 读数超过设备 limit。此结论只属于该对，其他搭档下的成功不抵消它。向墙的 p0→p1 段：斜率 unknown（未取得足够独立配对）；不据此预测解除墙能带来多少收益。

若有 improving certificate，替换最后一句为已测端点、每点真实 trial n、每 trial samples n、单位、区间与阈值；不写未运行的 q 的收益。多 knob 或 soft 证据有各自标题，不用相同“已归因硬墙”措辞。

采样器只消费域内完整配置。扫描得到的 compile-fit 点是**待验证建议**，不是已可启动真理；普通建议选一个已观察局部可行分量内的点，未知外侧不默认不可行，仍可由原 TPE 采样。按完整 config_key 对 queued/running/asked 去重，hard/soft 同点合并来源。确认重复另有明确 intent，不能将它伪装成普通去重豁免。

### 9.2 replay 与迁移

- 旧事件原样读、原 renderer 原样再生，不把 `RESOURCE_WALL_ATTRIBUTED` 的事件名当肯定。没有 version/identity/pair 信息的 legacy 行默认 `legacy_marginal`，不能回填为新 criterion 的 confirmation。
- 旧 `robust_ms` 是 property 而非序列化字段；legacy 读 `median`，缺时 `mean`，但确认要求当前 protocol 的 raw samples 与实际 n。不要重写旧 `latency_by_value` 或旧 run。
- 原 `backend:hash(materialized_source)` 是 evaluator 内部缓存而非持久契约。新账本使用稳定内容摘要并绑定 I；重启后只有满足版本/身份/完整回答条件的资源证据可重用。失败只存尝试日志，后续可在 cap 内重探，但每 checkpoint 同失败点至多一次，不能忙等重试。
- replay 必须按事件前缀、freeze_seq、source_seq_cutoff 计算，不读未来最优点或最终 latency 来决定历史建议。离线 replay 无新实测数据时只能重算证据，不模拟新建议“会获胜”。
- 新统计与旧统计并列、标明 denominator/version；不把不兼容的计数曲线接成一条。未知 enum/schema 保留 raw 并报 unsupported，不默认成功。

读数脚本也不是语义真理：`scripts/probes/read_slope_guide_counters.py:1–26,60–82,92–102` 说明累计快照的三角数事故，但“zero steps就说明关闭”“accepted就说明drawn”仍比源码保证更强；`scripts/probes/did_the_enqueued_point_win.py:20–40,72–100` 明说它按knob值比较最好trial、未固定搭档，故不能用来确认本稿斜率。其 `main:379–405` 并未强制检查 RUN_FINISHED，新报告须显式加完成状态门；brief `309–315` 的跑完再判是操作纪律，不应误认为现脚本已强制执行。`scripts/analyze_s7_pair.py:221–227,250–272,327–350` 及 `tests/test_s7_pair_reader.py:143–164` 还残留把 sub-limit 消融解释成“非资源墙”的读法，且不能代表 top-K 归因与实际文本送达；重放新判据不能原样沿用这些结论。

### 9.3 开关矩阵（拟议配置；本次不修改 config）

| 开关 | 只改变什么 | 单变量对照 |
|---|---|---|
| `resource_failure_normalization.enabled` | 新资源失败解释字段，不改 Optuna 状态 | off/on 检查漏报及误报 |
| `conditional_evidence.collect` | 新点/边采集，无 prompt/投递 | off/on 量 instrumentation 成本；不是零成本 shadow |
| `conditional_evidence.criterion` | 相同已收集点上的 legacy/新判据解释 | 统一采集后比较判据，不将采集量差误当判据效果 |
| `conditional_evidence.interactions` | 同一诊断 cap 内是否做多 knob 回退 | 固定其他开关/点数规则 |
| `conditional_evidence.soft` | 是否解释 scoped spill | 不暗开 soft guide |
| `slope_measurement.enabled` | 是否运行固定 8 对确认、显式 repeat 接线 | 两臂同 wall collector，其余采样/文本关闭 |
| `slope_guide.evidence_consumer` | 允许新建议消费者；cadence/dose 独立记录 | 相同采集/确认条件下 off/on |
| `wall_evidence.in_prompt` | 有证据文本是否真正进入 analyst/rewriter | 同一采集/确认/投递，只改文本可见性 |

默认全 off 走旧路径；开关依赖不满足时配置校验失败，不静默降级。改 cadence、K、delta、alpha、max points、run cap 都是不同实验，不能多键一起改后称单变量。若做“全系统有效性”比较可以一起开，但称组合效应，不归因给单个构件。

回滚：先关 prompt 与新 guidance，再关尚未启动的确认任务，最后关采集；在途真实 trial 按旧终止规则结束，已产生事件不删除、已占预算不返还。新 epoch stale 后不发布旧建议；不取消原 sampler 的普通合法 trial。下一次运行全 off 必须通过旧行为黄金 fixture，不要求已被新机制改变过的当前 study 恢复成反事实旧序列。

## 10. 预注册：成功、失败、没有能力回答

### 10.1 跑前需冻结的 manifest

后续实施时将 manifest 摘要写入 config 头部及 run-start 事件，正文独立归档；本稿不改任何配置。必含：代码/文档版本、设备与依赖版本、任务/workload 集、两臂唯一差异、B 与 run wall budget、全部 §5/7 提案阈值、episode 选择/轮转规则、随机种子与顺序、alpha family、主终点/分母、delta、invalid/stop 规则、重放版本、最终验证 protocol，以及计划 run 数与不追加规则。

### 10.2 分阶段可证伪验收

| 编号 | 主问题与提案门 | 失败或不能回答时如何解释 |
|---|---|---|
| P0 仪器 | §11 所有确定性控制应输出预定状态；重复确认必须有不同真实 job/measurement IDs；状态/预算/PRUNED 兼容差异为 0 | 任一失败停止上线；不能用更多墙掩盖错判 |
| P1 墙的真实性 | GPU 留出的必经超限、双可行、部分遍历、错误输入各控制均分类正确；自然数据逐条能追溯 scoped bytes/limit/有效端；误用其他搭档、缺测作墙为 0 | 属于实现/证据问题；全无自然墙只说明覆盖/邻接少，不自动判仪器失效 |
| P2 相关性与送达 | 报 current-relevant/历史墙、首证据进度、accepted→valid、prompt provenance；至少 1 个有可核验相关见证的 episode 实际执行，才称机制被施用 | 0 施用是未检验斜率用途；不能断言“改写转换率差” |
| P3 斜率 | 每个 certified 结论符合固定 pair/区间/阈值规则；报告 planned/completed/improving/worsening/equivalent/unknown 全分母 | 不要求自然数据必须四态齐全；全 unknown 是成本/功效不足或无效应尚不可分，不是平坦证据 |
| P4 成本 | 实际 probe/确认占用、超额在途尾部与完成普通 unique 配置数全披露；probe 900s/3%规则与确认5%软 cap 按注册执行 | 成本导致停止是机制可用性负结果，不能从结果表排除 |
| P5 端到端 | 固定 run 墙钟和 B 下，独立最终验证的 delivered latency 优于对照，并超过注册实用界；比较资源报告不能替代这个主终点 | P0–P3 通过但 P5 无改善，仅说明本剂量/任务组合未获益；若 CI 宽，结论 inconclusive |

**端到端首轮提案为同一预注册workload的 8 对完整 run**，硬件/任务/seed/预算匹配，每对 treatment 顺序随机、非同时争抢同卡；只有所有 run 按预定结束规则完成后分析。主标量为每个 run 最终交付配置按既有相同 full-eval protocol 的 latency，pair log ratio 定义同 §4。选择结果的最后一次验证不能用于再挑一个候选后仍沿用原 CI；验证失败为交付失败，单列且不得从分母删掉。不存在有效最终 latency 的 run 不补造值，主 latency 结论不宣告成功，并报告两臂有效交付率。跨任务结论另作分层研究，不假定不同任务的pair contrast具有共同中位数。若每run上限12h，这一提案名义上限是16×12=192 GPU小时，另有一次有界protocol-control run；不是一对24小时实验的免费统计升级。

run 级主检验只有一个，alpha=0.05，实用界同样提案为 log(1.05)；可用预注册次序统计区间，代价是 n=8 的区间很宽。等价须整个 CI 落入等价带；跨带就是功效不足。其他 P1–P4 是机制/质量与成本指标，非另挑一个显著主终点。分任务、分 wall-kind、斜率与后续收益相关性均预标 exploratory，不用多次检验挑“斜率有效”的子群。

这里的 8 对不是功效计算结果，更不保证能检测 2–4% 增益。既有单对 S7/step-4 不能当 8 个独立 run；候选/空间/20 timings 也不能充当 run 重复。若资源只能跑一对，仍按同 protocol 报效应和成本，但将 P5 明确标“案例对照，不能完成总体确认”；不能跑到显著为止。

## 11. 控制矩阵、尚需验证的 GPU 工作与证据等级

### 11.1 最小控制矩阵（逻辑构造，尚非执行测试）

| 输入/干预 | 必须输出 |
|---|---|
| 同搭档有效 p、必经 shared 超限 q；另搭档 q 成功 | 原墙保留；另搭档不是其反证 |
| 同搭档 p/q 都正确运行 | `not_wall_on_pair` |
| `[valid,refused,valid,unknown]` | 两条局部边、远端 unknown，不造单调 frontier |
| 全部观察点拒绝、无有效 anchor | `no_valid_anchor`，不发相关墙建议 |
| 部分 kernel 触达且 observed shared 都未超限 | compile scoped fit，launch/correctness unknown |
| 无 kernel、timeout、OOM、编译错误、wrong answer、missing spills | 各自 reason，不生成硬墙；missing 不填 0 |
| bool/category 改动导致资源拒绝 | 分类见证可有，slope=null |
| lower-side 已测延迟向墙改善 | D>0，方向 lower；不错误套 high-side 数值界 |
| 换一个搭档但延迟差<4% | 新 epoch；旧墙不适用于新 anchor |
| hard/soft 指向同完整配置、同 episode 重放 | 普通 suggestion 只一份；重复任务不重复执行 |
| A/A 同配置的新真实 jobs；8 对 d 全在实用带内 | 若质量有效，equivalent；非“所有轴都有斜率” |
| 8 对稳定正/负大效应，与跨带波动 | improving / worsening / unknown；20 samples 不增加 pair n |
| 只有4对但每 trial有20个样本；cache reuse 8次 | unknown；真实重复分母分别为4与原测量数 |
| 三个四角有效、联合角拒绝 | multi-knob interaction，不给每轴单独硬上限 |
| 旧 schema、截断事件、missing median/pair/I | legacy 或 unsupported/unknown，不伪升级证据 |

此矩阵足以规定“仍有非墙/无可确认斜率”，但不能证明真实分布中墙的覆盖率，更不证明 C2 有效。真实自然样本全有墙或全无墙并非逻辑矛盾；须由受控正负样例排除仪器常数读数，再讨论自然分布。

### 11.2 GPU 验证缺口与下一步顺序

1. 在实验全部结束后、另行授权的空闲 GPU 上，验证 compile-only 每字段来源与 null 比例，核对多 kernel/data-dependent workload 的触达集合；比较 single/batch、冷/热、同配置重编是否一致。不得叫“零机时验证”。
2. 测量完整批启动/逐点/失败尾部和真实 trial job wallclock；确定隔离是否覆盖外部 GPU 使用、shared probe 与 exclusive timing 是否互斥。没有这些实测，§7 只是带停止规则（包括在途超额）的预算提案。
3. 在固定 B 内执行 A/A 配对控制，检查时间漂移、顺序效应、独立 pair 假设与 cache bypass；不足以支持 protocol 时停在资源见证阶段。
4. 只读、按前缀重放完成的真实 run，核对 refusals、source/workload 对齐与既有 proposal 落地；这一步不会产生缺失的反事实延迟。之后才运行单变量新实验。
5. 将信息准确送达与改写有效性分开：需要 wall_id→prompt artifact→agent request→child source→最终测量的来源链。agent 文本提到“资源”不等于 C2 生效。

## 12. 与旧设计的明确比较

旧稿指 `docs/design-conditioned-walls-and-slopes.md`，本节是在 §1 独立候选之后形成。

| 旧稿锚点/主张 | 本稿判断与替代 |
|---|---|
| `11–16,78–82`：探针从验证者变成入口定义者 | **证据强制趋同**：条件化是修掉搭档混淆的必要条件。但定义入口先保存 scoped observation，再组合有效端/拒绝端，不让 ok 单字段代替可行性 |
| `39–45,84–86`：全资源向量、零噪声、与 launch 数字一致 | **收紧**：六配置 shared 的历史核对不能外推所有字段/版本；当前两路径读取 regs/spills 的对象不同，null 和部分路径必须保留；无本次 GPU 复核 |
| `78–81,108–110`：第一个不可行/最后可行，复用 `_toward_wall` | **不采用**：当前 `_toward_wall` 会排除拒绝值之后的所有 choices，且未逐点证明 launchable；改为非单调见证图，区别相邻边与 bracket |
| `89–97,153`：incumbent 改善超过噪声才重扫、失败不缓存 | 后半保留；**失效条件修正**为完整配置身份变化，是否重探受成本 cap 控制。小 latency 差不代表相同搭档 |
| `111–127`：两次 trial 的20+20 bootstrap CI，unknown/flat | **不采用**：样本嵌套、历史基准选择偏差、漂移、多重比较没有被解决。改为冻结后真实重复，equivalent 与 unknown 分开 |
| `120–122`：拒绝点名次数排序 | **不采用为默认**：采样频率本身是自适应选择的结果，还可能把所有 knob 都点名；按 unique 见证与轮转，避免事件频率自我强化 |
| `131–139`：误标、局部轴、多维回退分层 | **基本趋同**：三个层面回答不同问题。不能承诺 L1+L2 必然解除零投递；可行邻接、功效、缓存和成本仍可能阻断 |
| `143–154`：24点、每空间≤67s；trial额外墙钟0；单点timeout | **否定免费论**：完整轴计数应从 choices 算；16真重复占实际run时间；批调用用batch deadline；多 kernel/尾部不是7ms线性模型 |
| `158–167`：新事件/version/多开关 | 保留版本与开关思想；补足 identity、pair、cache-source、scope、stale、幂等、分母与真正单变量臂 |
| `171–180`：自然数据必须有墙/无墙且各 slope 态都出现 | 控制 fixture 必须区分；自然数据不能规定答案。0投递或全unknown可证伪可用性，不自动判仪器坏；有投递也不等于有效 |
| `184–190`：“零机时”离线 frontier scan、四臂<20min | 没有本地数据/实测点数与runtime，不能接受此成本承诺；编译探针需要真实GPU上下文与进程成本，须另获授权实测 |

本稿不是声称比旧稿一定更快；它以更低可能覆盖、更昂贵确认换取可审计的证据等级。若预算拒绝这种代价，合理结果是“条件资源见证值得保留，斜率确认暂不可负担”，而不是降低证据标准后宣布机制成立。

## 13. 外部依据（本次已访问核对）

只引用支撑具体规则的统计与官方 API 来源，不把文献动机当本项目效果证据。

- **[S1] NIST/SEMATECH，Analysis of paired observations**：https://www.itl.nist.gov/div898/handbook/prc/section3/prc311.htm 。配对须有实验上的对应关系；本文采用 pair-level contrast，不把两个独立时间序列随便按下标配起来。未采用该页小样本 t 检验为默认。
- **[S2] NIST Dataplot，Median Confidence Limits**：https://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/mediancl.htm 。支持二项式与次序统计的中位数区间构造；本文取不插值的保守区间，并公开小 n 的覆盖率与假设。
- **[S3] NIST/SEMATECH，Bonferroni's method**：https://www.itl.nist.gov/div898/handbook/prc/section4/prc473.htm 。使用一般概率不等式分配有限 family 的覆盖误差，不搬用 ANOVA 的具体方差模型；冻结后新数据和固定终点是额外必要条件。
- **[S4] Lakens (2017)，Equivalence Tests: A Practical Primer，DOI 10.1177/1948550617697177**：https://pmc.ncbi.nlm.nih.gov/articles/PMC5502906/ 。支持预先设实用界、等价与未显著分开；不把该文的学科样本量建议套到 GPU。
- **[L1] Triton 官方 Fused Softmax tutorial**：https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html 。已通过 Context7 检索 `/triton-lang/triton` 并核对官方页面：warmup 后显式初始化句柄，再读 kernel.n_regs 与 metadata.shared。这里只用来警示属性来源；没有运行示例，也没有借用其性能数字。
- **[L2] Optuna Study API**：https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html 。已通过 Context7 `/websites/optuna_readthedocs_io_en_stable` 核对 enqueue_trial 的 params/user_attrs/skip_if_exists；框架 API 可以排队不代表本 harness 会接受重复或真正执行。适配必须以本仓库 tuner/cache 实现为准。

## 14. 本次验证范围

本次完成源码/注释/旧稿阅读、官方统计与库文档核对及 PowerShell 纯算术检查；无依赖安装、无 git 操作、无生产 import/monkeypatch、无 GPU/远程事件访问。§11 是待实现测试矩阵，不是“已通过 synthetic suite”。没有创建额外验证脚本，避免把文档任务变成生产原型。

算术检查使用 PowerShell 的 `1-2/[math]::Pow(2,n)`（n=4/6/7/8）、`11+0.007*n`（n=24/36/48/96）、`16*T` 与 `32*T/43200`（T=20/60/180），以 `ConvertTo-Json -Depth 4` 完整输出；退出码 0，输出已逐项核对。它只验证覆盖率/成本示例的计算，不验证实验独立性、仪器正确性或设计疗效。

本稿的所有运行门均等待 GPU protocol 验证。交接 brief 的 CPU 测试历史基线不作为本次测试结果；由于本次只新增 Markdown，没有运行生产测试、typecheck 或 build，也不声称旧基线已复现。

对本文件调用 `lsp_diagnostics(..., severity="all")` 返回“未配置 .md LSP”，故无LSP通过结论；没有安装或修改配置。采用完整分段读取校对、限定文件的标题/占位符搜索和算术复核替代文档可用的静态检查，不把它称为编译验证。

最终复核：文件路径存在；`grep("^## ", path=本文件)` 找到0–14共15个主章节；占位符搜索命中的省略号是probe schema示例，另一个命中是本节描述搜索的方法，没有未完成的TODO/TBD/FIXME。7项显式算术断言全部通过，退出码0；完整敏感性输出与§7一致。实际执行命令如下（工作目录 `D:/Pyhon_projects/opop/v3`，PowerShell 5.1，只做算术与路径存在检查）：

```powershell
$checks = @(@{name='coverage_n8'; actual=1-2/[math]::Pow(2,8); expected=0.9921875}, @{name='coverage_n4'; actual=1-2/[math]::Pow(2,4); expected=0.875}, @{name='two_batches_48'; actual=2*(11+0.007*48); expected=22.672}, @{name='run_30_spaces'; actual=30*2*(11+0.007*48); expected=680.16}, @{name='slow_run_60_spaces'; actual=60*2*150; expected=18000}, @{name='two_episodes_60s'; actual=32*60; expected=1920}, @{name='axis_unique_points'; actual=1+12*(4-1); expected=37}); foreach ($c in $checks) { if ([math]::Abs($c.actual-$c.expected) -gt 0.00000001) { throw ('FAILED: '+$c.name) } }; @{document_exists=Test-Path -LiteralPath 'D:/Pyhon_projects/opop/v3/docs/design-wall-slope-evidence-first.md'; checks_passed=$checks.Count; checks=$checks; run_sensitivity=@(@(10,30,60) | ForEach-Object { @{spaces=$_; reference_seconds=22.672*$_; reference_percent=100*22.672*$_/43200; slow_seconds=300*$_; slow_percent=100*300*$_/43200} })} | ConvertTo-Json -Depth 5
```

唯一交付变更为 `v3/docs/design-wall-slope-evidence-first.md`；没有新增 `wall-slope-design-tests/` 工件。剩余调查优先级：资源字段/触达范围真实性 → 全链墙钟与跨run隔离 → 真重复及A/A协议 → 完成run的只读重放 → 单变量GPU对照。
