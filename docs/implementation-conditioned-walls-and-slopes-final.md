# 条件化资源墙与斜率：最终实施规格

> 日期：2026-09-14。状态：待实施的最终合并规格，不是实现或效果报告。
> 本文是冲突规则的唯一实施依据；两份旧设计保留为历史草稿，不修改其正文。
> 本次仅交付本 Markdown。代码、配置、现有文档、运行实验均不改动。
> 所有数值默认值均为拟预注册操作值，不是实测最优值；GPU 工作须等现有实验全部完成并获得单独明确授权。

## 1. 决策摘要与范围

采用“逐完整配置观测账本、条件化点图、限额消费者”。资源事实的存在、对当前配置的适用性、延迟证据等级分别判断。
默认新实验走 exploratory 路径，certification OFF；工程配置默认仍为 off，以保持旧行为。普通 TPE、原 trial 配额 B、空间/族分配、完整验证和 FAIL/PRUNED 映射不变。
诊断可以提供资源事实和缺测位置，不能生成约束、删除 choices、建立隐式单调上限或自动扩充生产 guard 缓存。
探索投点是原 B 内的新完整配置，既不是免费 trial，也不是确认重复。未知斜率不能挡住中性假设检验，更不能被叫作平坦。
不使用边际延迟桶、surrogate、资源加权分数或预测收益作墙门、投点排序及收益承诺。禁止跨 candidate 复用资源或延迟证据。

### 1.1 输入与源码依据

已读取 `docs/handoff-wall-slope-design-brief.md`、`docs/design-wall-slope-evidence-first.md`、`docs/design-conditioned-walls-and-slopes.md`。下列路径相对 `v3/`。
本次源码核对先调用 Codegraph，确认项目无索引后，按限定路径读取；未建立索引。非下表直接核对项的详细历史锚点见证据优先稿 §2.1、§2.4。

| 本次直接核对的锚点 | 对实施的约束 |
|---|---|
| `src/kernel_optimizer/evaluation/correctness.py:31-66,69-100` | 批用 `prescreen_timeout_s(cfg,n)`，单点生产 screen 用 `screen_timeout_s(cfg)`，不是同一预算 |
| 同文件 `170-245,247-282,284-361` | `prescreen_batch` 无返回值并填 live `_screen_cache`；guard 读取此缓存；诊断不能直接调用它 |
| `src/kernel_optimizer/gpu/worker_client.py:259-325` | job 构造早于 wall stamp；锁在 `communicate(timeout)` 前，超时后 kill/reap/release；不能承诺整 job 硬 deadline |
| `src/kernel_optimizer/gpu/worker_main.py:713-864` | warmup 拦截仅编译所触达 kernel；forward 异常可被吞；逐变体结果与批顶层不同；元数据字段可缺失 |
| 同文件 `777-790` | shared 的旧默认 0 可能来自缺字段；regs/spills 取 metadata 属性，不保证与正常 profile 来源等价 |

brief 中 11s/批、7ms/点、自然拒绝数和既有噪声量级是历史材料，本次未复测。compile-only 可能建立 GPU 上下文、分配张量或执行非 Triton 运算，不是零 GPU 成本。

## 2. 身份、适用性与证据等级

### 2.1 执行身份与 typed 完整配置

每个执行身份只写一次不可变 `IdentityManifest`，记录 run/candidate、冻结原始 source SHA-256、reference SHA-256、backend、compiler/toolchain、device/limit、correctness/timing protocol。
workload 签名包含 shape/dtype/stride/device、输入及初始化工厂摘要、seed 策略和影响控制流的输入约束。文件路径、任务名、去掉 PARAMS 的结构指纹均不能替代身份。
`identity_id` 指向 manifest。兼容规则首版采用这些执行字段精确相等且 candidate 相同；字段未知不能证明兼容。跨 run 默认不复用，后续协议不得默默放宽。
每点保存最终物化的全部 PARAMS，包括非当前可调轴的固定值；键排序、显式类型、无损值编码为 canonical typed mapping，完整 SHA-256 为 `config_key`。
`bool:true`、`int:1`、`float:1.0`、`str:"1"` 不混同；浮点按无损表示保存，拒绝 NaN/Infinity。保留原 literal、legacy `ParamSet.key()` 映射，不替换旧 sampler 的键算法。
每点另存 `materialized_digest`，轴两端物化摘要当然不同。空间实例、版本、轴类型和 choices 顺序单独记录，不把空间扩展误当执行事实失效。

### 2.2 epoch 仅作溯源

`anchor_epoch` 记录 live incumbent 更新历史，`batch_anchor_id` 指向冻结诊断 origin。二者都不是资源缓存的失效主键。
对轴 k，当前适用条件为：执行身份兼容，且 typed 搭档投影 `current_x[-k] == evidence_x[-k]`。改变 k 自身保留该轴资源事实，重新计算当前位置、邻接与距离。
改动另一轴时，即使 latency 完全相同或变化小于噪声，旧 k 证据也不能当作当前事实；拒绝 `stale_within_noise` 和“改善不足所以沿用墙”。
精确回到兼容完整配置时重新适用并引用原证据，无需重编；缓存读取不增加独立 measurement 数，不补发预算，不重置轮转。
冻结 origin 在整个诊断批内不跟随 live incumbent；批结束后的消费者每次重新计算适用性，不因“属于旧 epoch”一律扔掉事实。

| 消费者状态 | 条件与权限 |
|---|---|
| `current_applicable` | 同身份且同轴搭档；可描述当前条件资源事实，当前邻接另算 |
| `historical_conditioned` | 同 candidate/source 的兼容执行身份、搭档不同；仅明确标为历史具体条件，不传递为当前上限 |
| `incompatible_or_conflicted` | source/workload 等不兼容、关键身份缺失或同范围证据冲突；不进入事实文本、指导或认证 |

墙判定与状态正交：同搭档资源事实不保证当前 incumbent 正好是有效端，也不保证一步触墙。历史证据不因延迟相近升级；冲突不以“最新一次覆盖”消失。
候选结构改写后可以保留 parent hypothesis 来源链接，但子代不得继承父代资源事实。每次消费均验证 candidate、source 和 cutoff。

### 2.3 点图与见证

点的编译、资源、执行、正确性、延迟是分离字段：`compile_observed` 不等于完整 forward 编译，不等于 launch，不等于 correctness，也不等于有效 latency。
有效 latency 还要求同 timing protocol、完整且有限正数的原始 samples、实际样本数可核对、原 correctness 通过且无未解释的工作量/异常快计时标记；`complete` 或 accepted 单字段不足。
硬墙见证需要同 I、同 `x[-k]`、仅 k 不同的 p/q：p 是同 workload 通过原完整 trial 正确性验证的有效端；q 有可核验、作用域明确的精确资源拒绝。
q 可来自已证实真实 workload 必经 kernel 的 required>limit，或明确 launch OOR。未知/垃圾数据依赖路径上的超限只记 scoped observation；普通编译失败不能冒充资源拒绝。
p 只有已观察编译范围内未超限时，输出 `resource_transition_observed`，不是 `witnessed_conditional_wall`。p 的 latency 未有效不阻止资源见证，但不能用于延迟比较。
有序域相邻 choices 上的 p/q 是 `adjacent_witness`；中间仍有 choices 是 `bracket_witness`，全部中间未知保留，不能声称第一个失败或最后可行。
保存离散非单调图。例如 `[valid,refused,valid,unknown]` 有两条局部见证，末点未知；禁止二分、失败后轴向截断和沿用旧 `_toward_wall` 的数值上界逻辑。
双有效端是 `not_wall_on_pair`；指定域全部经真实验证有效才是 `no_wall_in_scanned_domain`。只见拒绝无有效端是 `no_valid_anchor`；远距多维拒绝未归因为 `unresolved_refusal`。
同身份同范围出现有效与精确拒绝矛盾，标 `evidence_conflict`，保留双方 raw；停止该范围诊断消费，不改变原 tuner 决策。
categorical 和 bool 可以有具体资源见证，但没有斜率、数值方向或大小排序。数字编码类别仍是 categorical；旧域缺顺序元数据时不推断有序。
当前 materializer 不支持直接 bool literal 的限制保持不变；账本可以解释 bool 类型，不承诺新 bool trial 可执行，不在本项目内顺带扩展支持。

## 3. 最小记录族与接口契约

采用一次身份 manifest 加三类记录，不要求六事件框架。记录可携带状态更新，写入既有 append-only 流；事件名字不编码成功结论。
统一 envelope：`schema_version=1`（新命名空间）、`criterion=conditional_witness_v1`、`protocol_version`、`event_id`、`seq`、`run_id`、`identity_id`、`source_seq_cutoff`、`parent_evidence_ids`、`config_digest`。
所有判定引用的事实必须满足 `fact.seq <= source_seq_cutoff < evaluation.seq`；当前消费 cutoff 也冻结。更高版本不得静默按 v1 解释。

| 记录族 | 核心字段与语义 |
|---|---|
| `CW_POINT_OBSERVED` | observation/measurement/attempt/request/job IDs、space/version、typed config/key、materialized digest、phase、origin、raw reference/hash、cache source、覆盖与逐字段存在性、每 kernel 资源向量、执行/正确性/latency 状态 |
| `CW_CONDITION_EVALUATED` | evaluation/wall IDs、axis/type/partner key、p/q observation IDs、batch anchor/epoch、当前 incumbent 引用、见证类型、adjacency/distance/unknown choices、消费者状态、资源 kind/required/limit、verdict/reason、审计 contrast 与可发布 certificate |
| `CW_ACTION_RECORDED` | action/request/batch/checkpoint/proposal/episode IDs、intent、阶段与状态、资源 kind、依据 IDs/raw reference、coverage/field presence、cache source、全链成本、预算账、cursor、planned/accepted/started/completed 与拒绝原因 |

`resource_kind` 为 `shared_memory|registers|spills|threads|other_explicit|unknown`，保留原始名称和单位；OOM 单独记录 failure class，不强映射到其中一种墙。
`phase` 为 `materialize|static_check|compile_probe|production_screen|launch|correctness|timing|cleanup|unknown`。原 legacy `failure_kind` 原样保留，诊断解释是 additive 字段。
每 kernel 保存逻辑 name/call-site、specialization、workload scope；资源字段分别有 value/unit/field_present/read_source。null 或字段未出现不是 0，也不是 false。
coverage 使用 `unknown|partial|validated_workload`，另记 reached kernels、forward exception、必经路径依据。最后一种只在有可追溯覆盖验证时使用，不能由 probe `ok` 推出。
真实测量产生唯一 `measurement_id`；缓存引用 `source_measurement_id`。重复事件、TRIAL_DONE reuse、screen cache 命中不增加测量 n；真实重编也不能变成独立性能重复。
raw reference 为可定位且带摘要的工件引用；缺 raw/身份的旧数据降级，不能靠补默认值升级。manifest 中固定 key 算法与 raw 保留策略。
`wall_id` 由身份、轴、typed partner、p/q key、resource kind 和 scope 生成；不同观察共用同一见证身份但保留所有 observation IDs。
答案缓存仅收足够明确的 scoped 字段，失败尝试保留在账本而不缓存为否定；部分回答仅复用已知字段，不能用顶层 `ok` 满足所有缺测请求。
成本子对象固定包含 `chain_start/end`、`child_job_ids`、`materialize_s`、`static_s`、`screen_s`、`lock_wait_s`、`eval_s`、`cleanup_s`、`occupancy_union_s`、`service_sum_s`、`unaccounted_reason` 和预算前后值；未知分段为 null。动作账引用此对象，逐点不平摊虚构的批启动成本。

### 3.1 编译 worker 复用，生产缓存隔离

新增独立 `DiagnosticCollector.collect_batch(requests, allowance)` 与 `DiagnosticCache.lookup(identity, config, scope, required_fields)`；接口名称为拟议，不是现有 API。
collector 复用 `make_compile_probe_job` 与 worker 批执行能力，返回按 stable `request_id` 关联的逐点响应，不能调用 live evaluator 的 `prescreen_batch()` 或 `compile_screen()` 来做新诊断。
不能“先写 live cache 再删回去”，也不能仅换 tag 后仍写 `_screen_cache`。采用独立诊断 cache owner；生产 evaluator 没有指向它的 guard/read/write 引用。
如抽取底层 transport，共用部分必须无 evaluator cache 副作用，生产 wrapper 的调用、缓存、异常和映射须保持黄金 fixture 行为；本次不做此补丁。
新诊断命中拒绝只影响诊断建议选择，不发布 `CONFIG_SCREENED_INFEASIBLE` 生产拒绝，不调用 tell(PRUNED)，不建立空间约束，不把答案注入 live `_screen_cache`。
以后可在普通生产 screen 原本发生后，单向、只读地复制其响应引用到 diagnostic ledger；guard 引用已有生产答案也可记录，不增加生产探针、不反写生产缓存。
逐点绑定 host/worker path、request ID 与物化 digest。当前批路径转换/返回查询存在静态契约风险，不能以 fake worker 原样回显证明 WSL 正确；实现需原生及转换路径 fixture。
批 primary 成功/失败不代表兄弟变体；漏项记 unanswered。worker 异常、timeout 或部分返回不能让所有提交点成为 refused。

## 4. 确定性调度与共享诊断预算

### 4.1 拟预注册默认值

| 参数 | 默认与作用域 |
|---|---|
| checkpoint | 每空间累计每 10 个完成 told；按既有 told 语义，reuse 可推进 cadence，但不推进独立测量数 |
| pre-rewrite checkpoint | 可选，默认开启；复用同一预算，仅刷新证据，不发新 guidance、不新增指导额度 |
| frozen origins | 每个新批 1 个，取 cutoff 前当前有效 incumbent；无有效 origin 则跳过主动扫描 |
| unique point cap | 每新批最多 48 个新提交完整点，跨 hard/soft/control/interaction 合并 |
| started batch cap | 每空间实例最多 6 个已开始批；在取锁前持久记 started，失败或崩溃不退额度 |
| diagnostic run ceiling | `D = min(0.03 * run_wall_budget_s, 900s)`，全 run 共享软上限 |
| concurrency | 全 run 最多 1 个 active diagnostic batch；不与本 run 新 guidance/confirmation trial 重叠 |
| guidance dose | 每 10-told checkpoint 最多 2 个、每空间最多 6 个，未用 checkpoint 额度不结转 |
| guidance/confirmation ceiling | `G = 0.05 * run_wall_budget_s`，两者全 run 共用实际占用软上限 |
| new trial concurrency | 全 run 最多 1 个 active 新指导或确认 trial，原 trial timeout 不变 |
| optional interactions | 默认关闭；最多 2 个依赖层，使用相同 48 点、6 批、D 上限 |
| certification | 默认关闭；核心路径没有强制 16 次确认 trial 或 192 GPUh 门槛 |

空间 cap 绑定持久 `space_instance_id`；resume、epoch 变化、扩展同一空间版本不能清零。原外环真正创建新空间才有新空间 cap，但 run D/G 不补充，不为诊断人为创建空间。
batch=物理诊断作业批，checkpoint=评估机会，不能混算。缓存-only checkpoint 不启动批、不消耗批额度，但主机账本开销计入 D。
控制点、软墙、交互和所谓 terminal scan 均在这一个调度器内；无单独终扫特权。pre-rewrite 若落在相同 cutoff 的 10-told checkpoint，合并成同一请求。

### 4.2 检查点算法

1. 原 ask/tell 完成后持久化 told 水位。只处理尚未处理的 10-told 水位；重启按前缀恢复，禁止因为重算而重发。同一空间繁忙时合并请求，保留触发 ID，不追赶发出一串旧批。
2. 以当前事件前缀冻结消费快照。先读取兼容 ledger/cache 并重算见证、适用性与邻接，不按 incumbent flip 立即发 GPU 工作。
3. 若无 active batch 且预算允许，冻结一个完整 origin、identity、域版本与 source cutoff。批内 origin 不替换，新 incumbent 只影响下一次消费快照。
4. 合并所需点与缺失字段，以 `(identity_id,config_key)` 去重；同点多轴、多资源、多控制用途保留 source IDs 集合，只提交一次。已 known-refused 且范围明确的点无需重探。
5. 分优先级选择：已有兼容证据复用；当前已知拒绝对应的缺失搭档配对点；当前 origin 的两侧相邻点；剩余有序轴点；可选交互层。类别点的已知拒绝配对可进入第二级，不生成类别斜率。
6. 第二级只在拒绝 r 与 origin 仅差一轴时补当前 pair；远距拒绝不能让每个变化轴获“已知墙”优先权。远距回退只进最后的可选层。
7. 各级按稳定 axis/wall ID 字节序与声明 choice index 排序，持久化 round-robin cursor；同层从上次位置之后开始。初始 cursor 为序首前位，方向先 lower 后 upper；拒绝事件重复次数不改变优先级。
8. 合并 origin/control 引用到对应优先项；已缓存 origin 不为了“控制”重复运行。需主动补测的控制占同一队列及点额，无预算时明确 control_missing，不开专批。
9. 选择最多 48 个未回答完整点，物化失败记 attempt；不在同 checkpoint 忙等重试。批结果按 request ID 逐点入 ledger，评估图后持久化 cursor、水位、batch 与 D 成本。
10. 所有失败重试只可在后续正常 checkpoint、同 cap 内进行；同 checkpoint 每点至多一次。没有新点就结束；预算拒绝为 unknown，不改变任何 choices。

每 checkpoint 至多启动一个批，依赖第二层交互留待后续 checkpoint。按 request ID 合并同一水位的 hard/soft 请求；过期水位不累积 guidance 剂量。
旧批结果可以成为历史条件事实；发布时重新核验 live incumbent，不能以冻结 origin 的旧文本冒充当前 incumbent。

## 5. 成本、准入与停止

### 5.1 全链计费

每个 action 从主机 materialize 前到最后 cleanup 后记录 monotonic 区间，包含排队、物化、static check、screen、锁等待、cooldown、eval、结果解析、事件写入及清理。
子 job 使用唯一 job ID，单独记录 lock/execute/cleanup 和已知缺口。动作占用按区间并集计，job 服务秒另列，不能将 trial 全链与其子 job 再相加。
缓存命中只收本次读取/调度成本，复制历史 `job_wall_s` 或 source stamp 不再记账。失败、timeout、被原政策拒绝及未得到 latency 的 action 仍计实际花费。
`D_used` 是新诊断动作占用并集；`G_used` 是新 guidance 与 optional confirmation 的实际 trial 全链占用并集，不是相对普通 TPE 的因果增量。
被动 ledger 捕获不新增 GPU job，但有主机写入成本，单列 instrumentation 成本；主动 shadow 编译是真成本计 D。观察普通生产 screen 不把原 screen 服务秒重复算成新增诊断。
缺分段 stamp 标 unknown/unaccounted，不能填 0 或用相邻事件间隔猜单 job 成本。全链总区间存在时仍可计总额，只有分解未知。

### 5.2 诊断准入

计算 `d_allow = min(prescreen_timeout_s(cfg,n), D_remaining, run_remaining_after_existing_reserve)`；非正就拒绝启动。这里只限制新诊断的 communicate timeout，不使用单点 `screen_timeout_s`，不改真实 trial timeout。
started 在进入取锁阶段前持久化。运行中按全链已过时间占用预算；到 D/run 软界后禁止后续 dispatch，当前批按既有 timeout/清理路径结束。
锁等待发生在 communicate 前，清理又在其后，现实现不能提供整 job 硬时限；记录 `inflight_overrun_s`、`lock_overrun_s`、`cleanup_overrun_s` 与未归属部分，不承诺有限固定的整个 run 超额上界。
至多一个 active 批约束的是并发数量，不是其等待时长。若未来需要硬 deadline，必须另审安全取消、取锁超时与 trial 语义，不能在此偷偷缩短生产 build/eval。

### 5.3 新指导 trial 准入

Stage A/C0 先从自然发生的普通 trial 获取完整成本 stamps，不强制重复、不为取得 5 个样本专开任务。缺成本可能导致早期无 guidance，这是诚实的冷启动限制。
首版准入需同兼容身份至少 5 个非 reused、完整成功且已结算全链成本的真实 trial；取其最大实际全链占用 `T_ref`，每次准入按当前前缀重算。
单个指导点仅在剩余原 B 足够、域合法、全配置去重通过、`T_ref <= G_remaining` 且 run 余量覆盖 T_ref 与原最终验证/关闭保留预算时准入。无可核验保留预算或成本数据则 abstain。
原 B 的可用槽必须扣除已 asked 与尚未计入 asked 的全部 accepted/pending/reserved 槽，按 trial/proposal ID 去重；先原子预占再 enqueue，拒绝或取消可释放未使用 B 槽，但不返还本机制已用指导剂量。
T_ref 是已实现成本参考，不是最坏成本保证，不是 kernel latency×20。剩余 G 还扣除已接纳未启动任务的 T_ref reservation；开始前释放该 reservation 并转实际计费，再检查一次余额。
原 trial 的 static/screen/eval/timeout 路径不改。已开始 trial 越过 G 后正常结束，此后不再发新指导/确认；在途及锁/cleanup 超额必须披露，失败不退已用剂量或实际成本。
不因 G 耗尽禁止普通 TPE 工作，不偷改空间 B。原 run 结束、收敛与调度终止优先；未启动指导在空间结束时取消并记原因，不挪到下一空间领新额度。

### 5.4 历史价格只作规划

参考代入 `11 + 0.007*n` 秒：48 点为 11.336s，6 批为 68.016s；这是理想参考算术，不是每空间成本上界。12 轴各 4 choices 的单 origin 全图最多 `1+12*(4-1)=37` 个唯一点，不默认 24 点。
多 kernel、冷热状态和病态编译可使批耗时远大于上述数值。12h run 的 D 为 900s，G 为 2160s，两项独立计价且都不能被 epoch 刷新。
完整 trial 占原 B 仍会挤占普通探索。报告普通完成数、unique 点数、空间/族数及最终质量；编译预热也属于干预效应，不从 treatment 成本中扣掉以宣称免费。

## 6. 默认探索消费者

### 6.1 Guidance

只在正常 10-told checkpoint 提议，最多 2 个、每空间累计 6 个；quota 在 proposal 被批准提交前原子预占。一次 dispatch 尝试消耗剂量，即使 enqueue/原政策拒绝也不在本 checkpoint 补发替代点。
完整配置在 asked、pending/accepted、running、measured 与本机制 reserved/proposed 集合中原子去重；hard/soft 同点合并来源，拒绝重复工作。真实已测点不为“测斜率”再次投，失败过的 asked 点也无重复豁免。
enqueue wrapper 的 accepted 不保证真正入队或被 ask；必须记录 proposed、admitted、accepted、started、completed、correct、valid latency，关联真实 trial/job IDs。
默认挑当前兼容 origin 条件下尚未完整测量的点，按 §4 稳定资源/缺测优先序；允许 compile-fit 或证据不足的合法邻点做中性假设检验，明确其 launch/correctness 仍待原 trial 验证。
已有该配置精确拒绝时不建议重复；这只限制新增 advisory，不阻止原 TPE 自己 ask，也不写 guard。不同搭档的历史墙不得直接构造“当前最后可行值”。
exploratory latency 差仅留审计：保存端点、raw samples、真实 n、时间与选择偏差，但 `certified_slope=null`，不产生 improving/worsening/flat 标签或按差值排序。
未经认证，即使观测差看起来很大也不进入 prompt 的斜率数字/方向；噪声门以下尤其不能排序。未知既不等价于 flat，也不触发 axis-wide stop；局部证据不能封禁整轴。
普通 TPE 的 objective、采样、guard、PRUNED 和探索机会分配算法保持原样；指导点仍走原完整 trial 验证，不设快速 correctness 或免费计时旁路。
每个 proposal 发布与开始前均复查 identity、轴搭档和 parent source；若失去当前适用性则取消本机制 pending proposal，保留剂量已占记录，不取消无关普通 trial。

### 6.2 Prompt

只渲染当前请求 cutoff 可见的条件事实。新 renderer 每次重新生成并清空不适用存储文本，不能简单复用 parent 上次 `wall_text`；prompt artifact 绑定 parent/source/evaluation/request IDs 与内容摘要。
拟定上限每请求 4 条、每条至多 300 个 Unicode 字符，其中历史最多 2 条，总体最多 1200 字符；超额先删低优先整条事实，不截断条件或否定词。值过长则用 typed partner 摘要加证据引用。
先当前后历史，各组按见证强度（真实有效端硬墙、scoped 软证据、未解决观察）与稳定 wall ID 排序；不用 latency 相近、拒绝事件频次或看过最终结果筛选历史。
历史条目必须 source/workload 兼容、无冲突、有原始来源，明确写“历史搭档条件，并非当前限制”。不能混入父 candidate 的事实，也不能把历史模式当当前证明。
模板：“在搭档摘要 C 与 workload W 下，p 经原验证通过；仅改变 k 到 q 后，必经 kernel K 出现已核验资源拒绝。当前关系：同搭档 bracket，非一步触墙。斜率未认证。”
软证据另写“当前 winner 在 kernel K 观测到 spill，条件点资源变化不证明性能因果”；unknown origins 必须写未回答，不能称其不触墙。
每 kernel raw 向量、绝对量、资源变化表仍遵守既有 agent 可见性限制；审计可存全量，prompt 只发布原政策允许的事实/数值。不新增资源兑换表、综合分数或未认证斜率数字与方向。
文本中行动只作可验证假设，不保证解墙更快。来源链为 evidence→evaluation→prompt artifact→agent request→parent→child source→最终验证；仅提到“资源”不算送达或生效。

## 7. 分层增强与未来认证

### 7.1 L0 与软墙

worker 的 `_classify_exception` 和 strict `_classify_eval_failure` 两条异常路径都新增旁路 `resource_failure` 规范化；保留原 `failure_kind` 及 Optuna FAIL/PRUNED 映射，不能用改分类来改变 TPE 学习数据。
规范化保存异常模块/类型、原消息引用、phase、kernel、resource kind、required/limit 与单位。shared/register/thread 等明确 OOR 分别解释；资源类型或范围不足就是 unknown，不能把所有 runtime_error 升级。
OOM、非法访问、编译/语法错误、物化失败、timeout、worker crash、correctness mismatch、无效 timing 各自保留原因，不算硬墙。明确资源拒绝存在也不证明有效端或整个轴限制。
软墙 gate 只看当前 winner：同 scope `n_spills>0` 才适用，明确 0 是不适用，missing 是适用性未知。不能拿其他 top-K 点替代 winner 门。
按逻辑 kernel/call-site 比较局部资源，保留两端 specialization；不同 kernel 的 max 不能拼成曲线。scope 变化记 `kernel_scope_changed`，缺字段不补 0。
不要求整轴 spill 单调，也不要求边际延迟改善。spill>0 不证明变慢，conditional spill change 不证明降低 spill 能提速。

### 7.2 可选多 knob 回退

从真实拒绝 r 向冻结有效 a 回退，D 为差异维集合；每层合批所有 `r'[j]=a[j]`，按固定维序选仍明确拒绝的回退点作为下一 r'，最多 2 个依赖层。
同层跨拒绝合并点；没有足够批/点/D 预算则 unresolved。新层不越过下一个 checkpoint，不从 soft/control/终扫另领额度，live incumbent 变更也不换本路径的 a。
仅各单维回退都在可核验完整编译范围内未超限，称 `screen_1_minimal`；部分 coverage 连该名字也不能用。各回退都经真实完整运行验证才称相对此 a/r 路径的 `true_1_minimal`。
这些都不等于 `smallest_cardinality`，后者需要额外子集搜索证明，本规格不做。两层耗尽或任何未知都保留 `interaction_unresolved`，不谎称最小绑定集。
回退沿途单轴见证的搭档是 r' 而不是 a；四角资源交互只描述四角，不给每轴独立硬上限。没有受控四角性能证据不提供交互收益。

### 7.3 可选认证协议，非核心前置条件

认证模式需单独协议授权、A/A 通过、显式 `intent=confirm_repeat` 与预算计数固定点入口；普通 guidance 永远没有重复豁免。缓存重复不能计 n，旧 20 个 timings 只是同 trial 子样本。
可选保守协议 `paired8_v1`：每 candidate 至多 1 episode、每 run 至多 2，按最早合格事件序选择不同 candidate；每 episode 固定 8 fresh pairs，A/B 只变一个有序轴，freeze 后不换端点。
每对共享输入/初始化 seed，8 个 AB/BA 独立公平随机顺序预先存 manifest；两次 timing 相邻，暂停本 run 探针，记录 pair gap/外部并发/设备状态。完整重复占原 B，不走 measured-cache reuse。
准入至少剩余 20 个原 trial 槽（16 尝试与 4 普通机会，扣除所有已承诺 pending），且 `16*T_ref` 可容纳于剩余共享 G 和 run 保留预算。不是为确认预留新 B，原调度终止优先。
最多 16 次尝试，不补失败、不换点、不看中间效果停或加样本；未完成/原政策拒绝/无效 pair/重启中断均 unknown，消耗 episode 名额。仅重复去重和 latency reuse 在新意图入口被明确绕开，原 guard、域、验证与状态映射不绕开。
定义 `d_i=log(a_i/b_i)`，A 为远墙端、B 为近墙端；`D=median(d_i)`、`gain=1-exp(-D)`、`slope_step=D/abs(index(B)-index(A))`。不外推拒绝点 latency 或解墙收益。
报告实际 fresh trials A/B、有效 pairs、每 trial samples、measurement/job IDs。满 8 有效对才计算一次固定终点区间；20 timings 不可拼成 160 次独立重复。
run family 包含全部候选/空间/hard/soft/方向的最多 2 episodes，`alpha_run=0.05`、每 episode `alpha_local=0.025`，未完成不转让 alpha。取最大 k 满足 `2*sum(C(n,r),r=0..k-1)/2^n <= alpha_local`，CI 为 `[d_(k),d_(n-k+1)]`。
无合格 k 为无界 CI；n=8 时极值 CI 覆盖 99.21875%，不等于有高功效。固定 `delta=log(1.05)`，CI 全高于 delta 为 improving，全低于负 delta 为 worsening，全在带内为 practically_equivalent，其余 unknown。
认证前 A/A 为单独授权的有界 control run，预指定一个有效配置执行 8 pairs，计原 B/G，不重做到通过；全有效且极值 CI 落在 ±delta，并满足以下质量规则才通过。它不提供其他 candidate 的资源证据。
质量规则固定：pair gap 不超过 freeze 前兼容真实全链成本中位数 2 倍；设备/workload/protocol 改变、外部并发或重试使 episode invalid；A 前4/后4对中位相对漂移超过5%则 unknown。
CI 仍依赖跨 pair 独立及共同中位数/稳定残差假设，A/A 不能证明所有 candidate 独立。认证结果对当前使用仍须按轴搭档检查，不要求 epoch 相同。
8 pairs 只是可选保守协议，不是通用样本量。改样本数、区间、质量门或 alpha 必须提交一个新的完整冻结协议，不能交给实现者临场选择，也不能成为首个探索 pilot 的隐含要求。

## 8. 模式、模块变更图与 rollout

### 8.1 具体模式与依赖

拟议统一入口 `conditional_walls.mode`；禁止用彼此独立的 collect/prompt/guide/confirm 布尔键拼出未定义组合。每个运行 manifest 固定 mode 和预算值。

| mode | 采集/主动工作 | 新消费者与旧代处理 |
|---|---|---|
| `off` | 无新 ledger/诊断 | 按原 legacy 配置走旧路径，黄金行为不变 |
| `passive` | 只观察原生产事件，0 新 GPU job | 新消费者关闭；旧代仍按原配置，用于加性捕获验证 |
| `shadow` | 被动捕获加有界主动诊断 | 新消费者关闭；旧代按原配置。主动成本不得叫 passive |
| `prompt` | 与 shadow 相同采集政策 | 仅新 prompt；旧 hard/soft prompt 与旧 slope guidance 全部关闭 |
| `guidance` | 与 shadow 相同采集政策 | 仅新探索 guidance；旧墙/prompt/guidance 全部关闭 |
| `combined` | 与 shadow 相同采集政策 | 新 prompt+guidance，单调度器单剂量；仅后续组合效应实验 |
| `certify` | 有界诊断及 paired8_v1 | 新普通 guidance/prompt 关闭，证书先审计；旧代全关，单独确认实验 |

另有 `conditional_walls.extension=none|interactions2`，默认 none；仅 shadow/prompt/guidance/combined 可取 interactions2，certify 首版禁止叠加交互。soft/L0 是统一账本解释能力，不是隐藏的额外消费者。
`certification_protocol=none|paired8_v1`：只有 certify 必须为 paired8_v1，其余必须 none。非法组合或新消费模式仍开启 legacy consumer 时配置校验报错，不静默双发。
`pre_rewrite_checkpoint=true|false` 只在主动模式有效，passive/off 设 false；它永远不提供 guidance quota。修改任何默认参数都是新预注册条件。
prompt 单变量比较采用 legacy 全关的 shadow vs prompt；guidance 单变量比较采用同样 legacy 全关的 shadow vs guidance。两臂采用相同的采集触发、排序、限额与预算政策；不要求实际 checkpoint 时刻、origin、采集点、观测结果或实际成本相同，因为消费者干预可改变后续搜索轨迹。单变量指唯一改变消费者开关，不指实现后的观测相同。禁止 prompt arm 顺带开 guidance。
同一次请求 wall renderer 只能选择一代；共享 ledger 不得让旧 2e 与新 prompt 各写一份。combined 也只有一份 2/10、6/space、G 配额，不是两个消费者各领一次剂量。

### 8.2 模块实施图，均为后续拟修改

| 位置（`src/kernel_optimizer/` 下） | 责任和不变边界 |
|---|---|
| 新 `models/conditional_evidence.py` | manifest、三记录族、typed identity、unknown/coverage；不塞入旧浮点表 |
| 新 `evaluation/conditional_ledger.py` | 点/范围缓存、证据引用、冲突、前缀 reducer；没有 guard 写权限 |
| 新 `evaluation/diagnostic_collector.py` | 独立批接口、request ID 关联、独立 cache owner |
| `evaluation/correctness.py` | 仅抽取无缓存 transport 或只读生产观察钩子；原 prescreen/compile_screen/guard 行为保持 |
| `gpu/jobs.py`, `gpu/worker_main.py`, `gpu/worker_client.py` | 新诊断 request IDs、字段存在性/coverage、双异常解释、全链子 job stamps；旧 payload 与 timeout 语义保留 |
| 新 `evaluation/conditional_witness.py` | 非单调图、axis partner 适用性、adjacency/bracket、soft scope |
| 新 `control/diagnostic_scheduler.py` | cadence、origin freeze、RR、去重、唯一批队列与 D/G budget owner |
| `control/orchestrator.py` | told/pre-rewrite 接线、生产只读 capture、action 漏斗、成本链、parent 文本失效检查 |
| `evaluation/wall_attribution.py`, `evaluation/soft_wall.py` | 保留 legacy 路径；新模式通过条件证据 renderer，不复用旧单调门 |
| `tuning/slope_guide.py`, `tuning/tpe.py` | 新 advisory 适配与原 B 准入；普通 TPE 不变，不复用旧范围截断；确认固定点入口仅 E 阶段 |
| `tuning/stats.py`, `models/reports.py` | 旧 latency_by_value 保留描述用途；新 report 使用独立 schema 与分母 |
| `config.py` | mode/extension/protocol 枚举校验、预算默认、manifest 固定与 legacy 互斥 |
| 新 `v3/scripts/probes/replay_conditional_walls.py` 与 `v3/tests/test_conditional_*.py`（本行相对仓库根，不在 src 下） | 未来前缀重放与下节 fixture；本次不创建脚本或测试 |

### 8.3 阶段与依赖

A 输入与账本：实现 L0 双路径 additive normalization、身份、原始字段与成本 stamps，生产只读订阅和 cache 隔离 fixture；off 黄金比较先通过。
B 见证与适用性：依赖 A，纯 CPU 逻辑图、typed partners、冲突、epoch-only provenance、确定性 renderer 和预算状态机；不需要 GPU 证明逻辑。
C 被动捕获与重放：依赖 A/B；C0 passive 只读原 jobs、0 新 GPU 工作，收集自然全链成本；C1 active diagnostic shadow 必须另授权，付真实 D 成本并验证覆盖/字段/路径。
D 有界消费者：依赖 C0 的成本可核验及 C1 的诊断安全验证，分别运行 prompt-only 与 guidance-only 单变量臂；先检查机制送达和机会成本，不把探索差值叫斜率认证。
E 可选交互/确认：依赖 D 的预算与幂等验证，interactions2 和 certify 各自单变量评审；paired8_v1 另依赖 A/A 与显式真重复入口，不能阻塞 A-D 的核心交付。
回滚先停止新消费者及未启动新任务，再停止主动诊断，保留 ledger 与实际成本；在途按原规则清理。下一 run off 应恢复黄金行为，不能声称已受干预 study 恢复了反事实 TPE 序列。

## 9. 重放、测试与验收

### 9.1 版本与重启

replay 仅用 event prefixes，恢复 told 水位、冻结 origin、RR cursor、已用/预占预算、seen/pending 集和唯一 action IDs。不得读取保存的最终 incumbent 决定历史排队或历史 prompt。
无终结记录的 started job 在恢复时先核查现有工件/运行状态，不盲目重发；无法核实时记 interrupted/成本未知并禁止新的成本依赖 dispatch。批额度、trial 剂量及 episode 名额不退还。
legacy 事件按旧 schema/renderer 重放，缺 identity/coverage/raw/pair/cost 不补造；旧 marginal wall 不升级为新 witness/certificate。未知新 enum/schema 保留 raw、报 unsupported，跳过语义消费而非崩溃或默认成功。
不能保证未修改的老 reader 已理解新事件；实施需为当前 reader 增加未知事件容错 fixture，报表明确 unsupported 数。off 不发新事件且原文本/队列/状态保持 golden 输出。
重放只重算曾经可见的证据，不补出未执行反事实 latency。重扫保存的最终 incumbent 是新的 GPU 实验，不是历史重放；终点选中点的覆盖率不代表在线因果效用。

### 9.2 预规定测试矩阵（待实现、未在本次运行）

| 测试输入/动作 | 必须断言 |
|---|---|
| 只改 k，epoch 增加 | k 的同搭档资源事实仍 current_applicable；位置/邻接重算，不发冗余探针 |
| 改 partner 而 latency 相同 | 原 k 事实 historical_conditioned，不能以 noise 续用；其他轴分别判断 |
| 精确回到旧配置、重复 cache event | 复用同 measurement IDs，n 不增、cap 不重置 |
| valid/refused/valid/unknown 非单调域 | 两条见证、远端未知保留，不生成全轴上限 |
| categorical/bool 与 typed 1/1.0/true | 资源事实可有、slope null、键不混；bool 物化仍按原限制 |
| 空 kernels、partial forward、字段缺失或旧默认0 | 无完整可行推断；unknown 不变0/false；假数据分支超限仅 scoped |
| 同身份有效与精确拒绝冲突 | 隔离相同范围消费，保留 raw 双方，不覆盖、不新增 PRUNED |
| hard/soft/control 同点、拒绝事件重复、timeout 重试 | 一批一次点提交；优先级不被重复刷高；失败无忙等、refused 无重复工作 |
| diagnostic 返回 shared 超限 | live `_screen_cache` 前后相同，guard/tell/PRUNED 与不采诊断一致；普通生产 screen 仍正常拒绝 |
| 10-told 与 pre-rewrite 同时/多次触发 | 一个 checkpoint，guidance 至多2、space至多6；pre-rewrite 无额外剂量 |
| 双异常路径的每个资源类与非资源失败类 | shared/register/thread/other/unknown、OOM/非法访问/compile/timeout/crash/correctness/timing 分类准确，legacy 状态不变 |
| 批 primary/兄弟不同结果、host/worker 路径转换 | 按 request ID/digest 匹配，漏项 unknown，不把 primary 传播给全批 |
| 重启于计划/started/结果写入边界，附加未来事件 | 幂等无重复 dispatch；相同前缀输出相同；未来 winner/latency 不泄漏 |
| 8次缓存命中 vs 8对 fresh jobs | 前者 n 不增；后者实际 n 可查；普通 guidance 无 repeat 漏洞 |
| 48点/6批/D/G 边界、慢锁与清理、screen拒绝 | 不再发新任务；在途超额及失败全链计费，cache stamp 不双计，trial timeout 不变 |
| winner spills 为0/缺失/>0，跨 kernel 最大值变化 | not_applicable/unknown/applicable 分开，不拼伪曲线、不要求单调 |
| 回退两层仍未知、只有编译fit、真实全验证 | unresolved/screen_1_minimal/true_1_minimal 区分，均不等于最小基数 |
| uncertified contrast 任意数值、旧 parent 文本 | prompt 不输出斜率数值/方向，来源失效清空；unknown 无全轴 stop |
| off/非法 mode 组合、未知 schema、截断 legacy | golden 不变、非法组合拒绝、unsupported 温和处理、缺字段不伪造 |
| optional paired8 的固定正负/等价/跨带与不足8对 | 满额质量门/CI/alpha正确；缺对、漂移、中断 unknown，无可选停止 |

受控正负 fixture 必须有墙、非墙和未知，才能排除常数判据；自然数据无需凑齐所有标签。未来 CPU fixture 通过不替代 GPU coverage、元数据真实性、外部锁隔离或 A/A 验证。

## 10. 预注册读数、局限与同伴处置

跑前 manifest 冻结代码/文档/schema、mode、唯一臂差异、任务/设备/seed、原 B/run wall budget、全部剂量/成本/排序/文本规则、最终验证方法、停止原因、计划 run 数及不追加规则。
首个 GPU pilot 可另授权固定一对匹配 run，使用探索路径，主目标是证据正确送达与预算可用性；只报告案例效应，不宣称总体疗效。无需先投入 16 次确认 trial、8 对完整 run 或 192 GPUh。
最终 latency 为 run 完成后原统一 full-eval 路径的最终交付结果；所有预注册 run 按既定终止规则结束后读胜负，不能以中途 incumbent 胜率代替。
交付失败/缺有效 latency 的 run 保留分母、单列失败率，不补数、不择优删除。需要总体确认时另定完整冻结 run 级协议及功效目标，不能事后追加直到显著。

| 读数 | 预定义分母与解释 |
|---|---|
| 证据可用性 | 每次实际 eligible consumer request：先有 current-applicable 则 current；否则有历史则 historical-only；否则 no-evidence，三类互斥且穷尽 |
| eligible request | 新模式中正常发生的10-told guidance请求或真实改写prompt请求，身份/域结构可解析即可；缺证据、缺成本、预算拒绝仍纳入，不能事后只留成功请求 |
| 条件细分 | 同请求可另报轴/边数量及 incompatible/conflict 原因；不以 current墙/不断增长的all-history墙数作主适用率 |
| cache与回答 | 有兼容答案复用的unique点请求/unique点请求；提交回答率用实际提交unique点，字段覆盖另报，cache 不算 fresh |
| budget denial与延迟 | denied eligible requests/eligible requests；首证据延迟从候选首个eligible请求起计秒与told，未获得者作为未到达/删失，不填0 |
| soft gate | 实际消费者请求的winner spill>0/有明确winner spill读数请求，另报missing/全部请求；不拿历史epoch累计当稳定分母 |
| action funnel | request→proposal→admitted→accepted→started→completed→correct→valid latency，以对应唯一ID为单位，每层失败/复用分开 |
| prompt送达 | 有可核验artifact与agent request链的请求/实际eligible改写请求；parent/source过期和未送达单列 |
| 机会成本 | D/G占用、锁/清理超额、instrumentation、普通TPE完成与unique点数、空间/族数、原B消耗；不是无反事实依据的净增量 |
| 终局 | 完成run的最终delivered latency及有效交付率；机制覆盖/暂时刷新最优只能作中间读数 |

累计快照只取同 run/space/counter 实例最新值或前后差分，不跨事件求和；combined 按 consumer kind 分层统计请求，但合并 action ID 后计算总剂量，不能双算同一点。
真实性失败、诊断改写 guard、重复发工作或成本漏记是上线阻断；自然无墙、全 unknown、0 投递可能是覆盖/成本/机会不足，不自动等于仪器坏，也不证明机制无效。
局限包括历史身份不全、编译 partial coverage、字段版本差异、winner 选择偏差、低剂量低覆盖、成本门偏向早期便宜候选、外部 GPU 争用和探索数据无因果认证；本稿没有已证明疗效的结论。

### 10.1 同伴意见处置与精确替代范围

| 来源与被替代规则 | 最终处置 |
|---|---|
| frontier 稿 §3.1/§3.4：latency 超噪声才重扫，`rescan_min_improve_pct` | 删除该触发规则；10-told调度、单批冻结origin、按轴typed partners适用，无 stale_within_noise |
| evidence-first 稿 §3.4/§8.2：任意完整配置变化都令旧epoch stale；§8.3 current/all-history分母 | epoch仅溯源；轴自身变化保留资源事实；改用实际请求三分母 |
| frontier 稿 §3.1/§3.2：first-infeasible/frontier、沿用 `_toward_wall`、compile等于可行、确定性全向量 | 非单调点图、有效/精确拒绝见证、bracket与adjacency、coverage与field presence分别验证 |
| frontier 稿 §3.2：20+20 bootstrap、unknown/flat、worsening整轴停止、拒绝次数排序 | 探索差值仅审计，无斜率标签/排序/轴停止；稳定RR，不将缓存或timings充当独立trial |
| evidence-first 稿 §5.1/§5.2/§7.3：核心确认16槽、每空间2批；§10.2首轮8对run/192 GPUh | 默认exploratory与认证OFF，6批/space；paired8仅可选未来协议，首pilot不强制重型规模 |
| 两稿关于缓存复用未明确隔离的部分 | 编译worker可复用，live prescreen_batch不可直接用于新诊断；独立collector/cache，生产→ledger仅只读 |
| frontier 稿 §3.4：trial额外墙钟0、screen_timeout作批deadline、每空间≤67s | 原B不等于免费；全链实际占用，批timeout独立，D/G均为披露超额的软cap |
| 两稿多开关矩阵与终扫描述 | mode/extension/protocol合法组合固定；新旧消费者互斥、终扫和pre-rewrite无额外点/批/指导配额 |
| frontier 稿 §3.1/§3.7：自然标签必须齐全、最终incumbent“零机时”重扫 | fixture控制区分状态；自然标签不限；历史replay与新GPU扫描分开，终点覆盖非在线效用 |
| 两稿软墙/多维射程意见 | 保留winner spills>0、逐kernel局部事实；去掉全局单调要求；两层共享cap，三种minimal含义分开 |

未决工程验证项只有实机字段/覆盖真实性、路径往返、锁隔离与完整成本、可选认证测量质量；它们是对应上线阶段的门，不允许实现者自行放宽已冻结的默认语义。

## 11. 本次交付与验证边界

本次仅新建 `D:/Pyhon_projects/opop/v3/docs/implementation-conditioned-walls-and-slopes-final.md`；创建前已确认同名文件不存在，三份输入文档完整读取。
验证范围为源码边界只读核对、文档内容/标题/占位与规则一致性检查；这些不是未来 §9 测试通过，也不是历史实验结果复验。
未运行 git、GPU、SSH、生产测试套件/import、typecheck/build 或任何实验；未安装依赖，未创建其他格式、脚本、计划或配置。实际调用 Markdown LSP diagnostics 返回未配置 `.md` server，因此没有 LSP 通过结论；未为此安装或修改配置。
