# 面向论文写作者的C2框架完整流程与技术细节

本文依据 **2026-09-21 当前本地工作树** `D:/Pyhon_projects/opop/v5` 的源码，不是某次历史运行的 pinned 版本说明，也不是一次新实验报告。C3 已暂停，不在本文讨论范围。
目标读者是不熟悉项目、需要撰写论文 Methods 的作者。下文先解释系统如何工作，再给公式、配置、实验变体和证据边界；理解基本机制不需要阅读会话或内部笔记。
本文只做静态文档整理，没有启动 GPU、provider、SSH、实验或测试，没有修改源码、配置、旧报告或计划。
引用采用 `文件:起行至止行，符号`。除明确写出的 `scripts/`、`docs/` 和 `../KernelBench/` 外，代码简写路径均相对 `src/kernel_optimizer/`；行号可能随工作树后续编辑漂移。

## 1. 一句话理解框架与 C2

框架反复做两件不同的事：先为一种程序结构找较好的参数，再根据测到的性能与资源现象改写程序结构。前者类似给同一台机器调档位，后者类似重新设计机器，但两者都必须重新验证输出。
C2 关注其中的**资源反馈如何帮助结构优化**：不只告诉模型“这个程序慢”，还尽可能给出哪个局部参数方向仍有收益、在哪些伙伴参数下遇到编译资源限制、哪些资源成本随参数变化，以及这些观察支持什么改写假设。
这些信息不会直接成为性能分数。正确性通过后的实际 latency 才决定 KernelBench 候选的优劣；节省寄存器、减少 spill 或提高估算 occupancy，本身都不能让候选晋级。

需要始终分开三个层面：
1. **完整主框架**：从 KernelBench task 开始，生成种子，运行参数化、调参、多 family 改写、novelty 和最终复评。
2. **可选诊断机制**：旧 hard/soft walls、S7、v4.1 conditional scan、向量诊断与 ledger 等，很多默认关闭，不是必经阶段。
3. **C2 固定父代实验**：人为冻结父源码、最佳配置与历史，比较 G0/G1/G2 获得不同信息后的结构提案。这是局部信息增量实验，不等于从头运行完整框架。

当前能支持的算法目标是“预算内访问到的有效程序与配置中的最佳观测值”，不是全局最优，不是硬件极限证明，也不是保证资源反馈一定提高速度。
来源：`docs/contributions-current.md:73至110`；`control/orchestrator.py:648至754，_run`；`tuning/objective.py:31至52，objective_value/rank_record`；`scripts/experiments/c2_information.py:93至159，run_information`。

## 2. 全流程图与四层循环

```text
KernelBench level:id + reference factories + resolved config
                         |
                         v
任务/语义准备 -> 硬件校准与 reference cost -> eager/compile baselines
                         |
                         v
生成或导入 seeds -> 结构去重 -> 独立 families
                         |
                +--------v---------+
                | A 参数化/有限修复 | <--- 空间拒绝或 witness 错误
                +--------+---------+
                         v
               PARAMS + space + 两个有效 witnesses
                         |
                +--------v---------+
                | B TPE 调参        | <--- 可选同预算 enqueue
                | 编译/正确性/计时  |      可选 U/W -> C4 -> E1
                +--------+---------+
                         v
          stats / profile / 可选 walls / analyst report
                  |                    |
          可选扩域，新 space、新 B      v
                  +------------ C family 结构改写
                                  |
                           children 各自回到 A/B
                                  |
                  无本轮进展时，D novelty 新 family
                                  |
                    预算/历史规则停止，汇总 incumbents
                                  v
                导出已测 trial 源码 -> 一次 full 复评 -> report
```

下列 Loop A/B/C/D 是对应主控制器注释与函数职责的解释性命名，不是四个同名公开 API，也不是四个并行 GPU 进程。
主流程中的具体函数是 `_parameterize_with_repair`、`_tune`、`_rewrite_round`、`_novelty_round`；扩域由 `_maybe_expand_space` 实现。

```text
prepare(task, config): 校准；探测语义与 cost；测 baselines；注册 seeds

Loop A(candidate):
    最多 repair_attempts + 1 次参数化/验证，默认 3 次
    发布空间前检查 literal PARAMS、choices、约束可行率及两个 witnesses
    witness 执行失败 -> 修当前 candidate；其他空间拒绝 -> 反馈 parameterizer
    成功交给 B；用尽则 dropped

Loop B(candidate, accepted_space):
    新建预算为 B 的 TPE study；排入 witness/合格推荐 anchors
    while ask 返回点:
        materialize；筛查；复用合格测量或 quick correctness + timing
        保存 trial；tell objective 或失败状态
        若配置启用，在 checkpoint 执行诊断/受限 scan enqueue
    单调保留 candidate/family incumbent；形成统计与允许的分析反馈
    若扩域启用且符合条件：验证新空间，再给新 B；改善不足则不再扩

Loop C(families):
    从有有效 incumbent 的 active families 按固定规则选 slate
    每个 family 固定本轮 incumbent 为 parent；生成并注册 children
    每个 child 独立运行 A/B；按 family 结算一次历史及可选账本

Loop D(families):
    仅在本轮 C 没有 progressed 且未到墙钟时尝试 novelty
    通过上限/签名/相似度门后，以空 parents 注册新 family，运行 A/B

outer:
    交替调度 C 与有条件的 D，检查全局停止及连续无进展防空转
    在所有 family incumbents 中选最小已测 latency
    导出 selected trial；独立 full 复评；保存成功或失败状态
```

伪代码中的“结算已评估轮”须按实际返回值理解：注册了 child 并不保证它完成了有效测量，相关限制集中见附录 A。
来源：`control/orchestrator.py:648至754、1180至1408、2424至2580、3035至3172、3333至3586`。

## 3. 任务输入、参考语义与目标

### 3.1 KernelBench task 从哪里来

入口为 `kernel-opt [--config FILE] [-o key.path=value] run --task levelN:ID`。
adapter 在 `kernelbench_root/KernelBench/levelN/ID_*.py` 找唯一参考文件，读取源码、计算 SHA256，构造 `TaskSpec(kind="legacy", level, problem_id, name, ref_path, ref_src_sha)`；缺文件或匹配多个均报错。
它不是自动生成 workload 的模块，也没有从自然语言 goal 自动推导 shape 或质量条件。

| 输入 | 实际来源 | 写 Methods 时的边界 |
|---|---|---|
| 参考程序 | 文件中的 `Model` | 是整份 forward 的参考语义，不只某一个自定义 kernel |
| 初始化 | reference 的 `get_init_inputs()` | 提供 `Model` 与 `ModelNew` 的构造参数 |
| 输入与 shape | reference 的 `get_inputs()` 及其常量/逻辑 | 没有独立的主 CLI shape sweep；有限测试不覆盖任意 shape |
| 权重 | 相同 seed 下分别构造两个模型 | 不是自动 `state_dict` 复制；构造顺序变化可能改变对应关系 |
| dtype | evaluator 按 `evaluation.precision` 转换 tensor 和模型 | 默认 fp32 I/O 不代表内部乘法、累加或存储全部严格 fp32 |
| device | worker 使用进程可见的 `cuda:0` | 物理卡由环境决定；`cfg.device` 是限额/提示信息，不是选卡器 |
| quality | EvalConfig 的 strict/relaxed、次数和可选 rescue | 不存在 task 自带的独立质量标量优化目标 |
| objective | 有效整任务 latency，默认单位 ms | 不优化编译时间，也不最大化硬件使用率 |
| budget | 每空间次数、family 轮数、family 数、墙钟 | 不等于一个全局 GPU trial 硬帽 |

主 wiring 使用 `run.seed=0` 给 evaluator、validator 和 sampler；reference baseline helper 却固定 seed=42。两者必须分别披露。
参考和候选以相同 base seed 各自初始化，但没有无条件权重复制；每个 correctness input 的 seed 也不等于每次 forward 的 RNG 状态完全配对。
框架探测 reference 的 training 与 norm-layer flags，并交给 agent；评估路径没有统一强制 `.eval()`。探测失败是 unknown，不是默认推理模式。
例如当前 `level1:19` 的 `19_ReLU.py` 输入为 `torch.rand(4096, 393216)`、初始化参数为空。此处仅说明 shape 由文件决定，没有运行这个大张量任务。
来源：`tasks/kernelbench.py:10至42，KernelBenchAdapter.load/parse_task_arg`；`cli.py:408至419`；`wiring.py:115至140`；`../KernelBench/src/kernelbench/eval.py:124至148、488至565、727至783`；`../KernelBench/KernelBench/level1/19_ReLU.py:4至31`。

### 3.2 baseline、校准与 advisory cost

控制器先加载或测量设备校准，再 probe reference 语义与 task cost，之后测 eager 和 torch.compile baselines。
strict 默认各测 IEEE 一份；dual-witness-relaxed 才额外测各自 TF32 版本。eager IEEE 无有效 latency 会终止；其他 baseline 失败可保存负 sentinel、n=0 与 note，不能作为加速比分母。
baseline 不是 seed、family 或自动 fallback 候选。它用于比较和合理性诊断，不提供一份默认可改写的 incumbent。
校准失败可以降级为 unknown；根据校准、cost 和 baseline 推导的 plausibility flag 是提示，不是把所有“异常快但正确”的候选直接拒绝。
baseline helper 使用 seed=42、warmup=5、discard_first=1；compile backend 为 inductor、mode 为 default。不要把 candidate 的 warmup=3 写成全系统统一值。
来源：`control/orchestrator.py:756至881，_calibrate/_baseline/_install_plausibility`；`evaluation/benchmark.py:76至163`；`../KernelBench/src/kernelbench/timing.py:9至103，measure_ref_program_time`。

## 4. 对象词汇表与身份地图

| 术语 | 通俗定义与实际对象 |
|---|---|
| candidate，S | 一份结构候选的身份；`Candidate` 保存 candidate/family/parent IDs、backend、source hash、signature |
| family | 一个 anchor 出发的谱系容器；保存成员、incumbent、历史和改写次数，不是数学等价类 |
| space，Θ(S) | 一个 candidate 的某版离散 domains 与 constraints；有独立 space ID/version |
| params，θ | 一份完整键值配置 `ParamSet`，不是源码或权重 |
| trial | 一次被接纳 ask 对应的记录；可成功、失败，也可复用旧测量，不必然是 fresh GPU job |
| default | 源码 literal `PARAMS` 的默认值，不是当前最优 |
| incumbent | family 或 candidate 当前保留的最佳有效观测点；可能来自旧 space |
| baseline | reference eager/compile 的独立计时，不自动参与 family 搜索 |
| witness | 为有限准入而实测的配置；不要与 TF32/IEEE 两种 reference 见证混淆 |
| endpoint | 一个局部对比的完整参数点，不只是轴上的一个数字 |
| partners | 改轴 a 时固定的其他全部参数；换 partners 就换了局部问题 |
| ceiling | 容量上限或校准性能 roof，二者来源与单位不同 |
| floor | 某资源的模型下界，如 spills=0；不是证明所有程序都达到或受限于它 |
| hard wall | 在指定伙伴参数下，编译器给出的资源超限拒绝，例如 shared 超 per-block limit |
| soft wall | 仍可运行但 spill 等成本增大的观测，不是生产拒绝 |
| hypothesis / expectation / ledger | 改写设想、事前资源方向声明、事后逐 child 对账，均不是性能证明 |
| BestKernel | 论文可定义的导出概念；当前源码没有要求存在同名类 |

注册规则决定 family 边界：全局先按结构签名去重；`origin="seed"` 或无 parents 创建新 family，其余继承**第一个 parent** 的 family。
因此大幅改写、换 CUDA/Triton backend 也可能仍属原 family。novelty 通过空 parents 建新 family，不靠“修改超过多少行”判定。
结构签名规范化 AST：去 docstring，把 `PARAMS` 字典值置零，可加 backend 前缀，再 hash；不会规范化所有变量名，更不是程序语义等价判定。
纯 PARAMS 值改变通常不会产生新结构；无用 import 等 AST 改动却可能改变签名，不能用签名不同证明有效计算已经变化。
novelty 还要求与每个 anchor 的去注释文本相似度小于 0.85；普通 seed 注册没有这个额外相似度门。
来源：`models/core.py:49至120、328至406`；`control/families.py:30至119、182至253，structural_signature/register_candidate/accept_novel_seed`。

身份关系可简写为 `task -> family -> candidate -> space(version) -> trial(params, materialized source)`，但 profile、source 和缓存还必须按实际测量归属连接。
`ParamSet.key()` 是排序键值表示的 SHA256 前16位，不含 source、device、space 或协议；candidate UUID、AST signature、源码 SHA、内容寻址 artifact SHA 都不是它的同义词。
真正支持某个 latency 的是当时的 materialized trial 文件及其测量协议，而不是今天的 `source.py` 配上历史 params。附录 A 说明当前注册 SHA 可能陈旧的限制。
来源：`models/core.py:114至120`；`control/accepted_candidate.py:12至36`；`store/run_store.py:114至130，put_artifact/get_artifact`。

## 5. 默认配置与启用后行为

`load_config` 只读用户指定的一个 YAML，再应用 dotted overrides，缺省由 AppConfig 填充；不会先读 `configs/default.yaml` 再叠加实验 YAML。未知字段被拒绝。
下表是当前类默认与主 wiring，不是历史实验配置声明；具体运行必须使用其 manifest 中解析后的配置。

| 项目 | 默认 | 启用后或计数解释 |
|---|---|---|
| run.seed | 0 | baseline seed=42 是独立例外 |
| trials_per_space，B | 40 | 每个新 study 的 admitted asks 上限，不是全局40次 GPU调用 |
| max_seed_candidates | 4 | generator 请求还受自身 batch 大小限制 |
| generator/rewriter/novelty batch | 4 / 2 / 2 | 输出可失败或重复，不保证注册数 |
| repair_attempts | 2 | 最多初次加两次参数化尝试 |
| rewrite_rounds_per_family | 5 | 优先于 plateau 标记预算耗尽 |
| no_improve_rounds / min_improvement_pct | 2 / 2.0 | 最后两次相邻改善都严格小于2%才收敛 |
| max_families_active | 3 | slate 大小，不是 GPU 并行度 |
| productive / hard family cap | 6 / 6 | 前者排除无 best 的 frozen dead family，后者计所有已建 family |
| wall_clock_hours | 12 | 调度边界检查，不是中断每个在途 trial 的硬截止 |
| prefetch_parameterization | 1 | 单后台线程预取下一 candidate 参数化；0关闭 |
| space_expansions_per_candidate | 0 | 开启才请求新域、新版 space 和新 B |
| expansion idle / edge failure | 0.8 / 0.30 | 余量准入及失败偏好，不是可行性定理 |
| quick / full | 3正确性+20计时 / 5正确性+100计时 | 性能 sample 不等于独立结构实验 |
| precision / correctness_mode | fp32 / strict | relaxed 和 FP64 rescue 不是默认 |
| compile screen | True | unknown 放行真实评估 |
| eval / build timeout | 600 / 1200秒 | timed job 合计1800秒，非两个独立阶段倒计时 |
| GPU concurrency | enabled，shared jobs=2 | timing 独占，主 TPE 循环仍串行 |
| v3 diagnosis | label | vector 可选，记录 dimension 不等于投递 vector |
| wall attribution / in_prompt | False / False | 旧硬墙检测与提示各有开关 |
| soft wall / in_prompt | False / False | 旧 spill 墙，默认只在启用后运行 |
| S7 slope guide / S8 ordered categoricals | False / False | 前者 enqueue 规则点，后者提供分类值距离 |
| S1b failure deweight | False | correctness 失败值概率降权，不是删除可行域 |
| expectation ledger / reserve reconciled round | False / False | 记录、prompt投递和预留槽是不同条件 |
| access-pattern walls | False | 不作为默认完整反馈链的一部分 |
| v4.1 mode / f_frac / pcap | off / 0.1 / 48 | observe诊断；active加scan与brief；Q受B约束 |
| agent request_timeout_s | 3600 | 当前值，不得替换历史报告的1500秒 |
| strip_cuda_env_from_server | False | opt-in环境过滤，不据此追溯历史执行 |

`DeviceLimits` 缺省为 unknown、16 GB、255 registers/thread、49152 B static shared、101376 B opt-in shared、1024 threads/block。它们不是自动发现真实 GPU 的承诺。
作为**选定启用样本而非项目默认**，`configs/experiments_v41_m2b_b_box1gpu1.yaml:35至70、105至139` 使用 active、f=0.125、一次扩域、vector、relaxed+FP64，并关闭旧墙和 S7/S8。
配置加载器禁止 active 与 S7 同开，但没有自动禁用所有旧 wall prompt；直接构造 AppConfig 也不会执行这个额外 loader 检查。
S1 出域声明、可配置 probe attempts、OLS `for_prompt` 与 E4 类型存在，并不说明对应生产功能已接好，见附录 B。
来源：`config.py:14至51、101至207、225至381、384至810`；`wiring.py:130至139`；`models/core.py:409至434，DeviceLimits`。

## 6. Loop A：参数化、guard、witness 与修复

### 6.1 从程序到有限搜索空间

parameterizer 输出一份带唯一模块级 literal `PARAMS={...}` 的程序，以及 `ProposedSpace(params, constraints)`。它不是只发参数 JSON，也不是修改模型权重。
每个 domain 有 name、kind∈{int,float,str}、离散 choices 和 description；没有连续 low/high/step/log API。即使 choices 为 `[16,32,64,128]`，调参仍是分类选择。
例如概念程序可声明 `PARAMS={"BLOCK":32,"WARPS":4}`，域为 BLOCK∈{16,32,64}、WARPS∈{4,8}，并以表达式限制组合。例子不代表某真实 kernel 的合法 launch 公式。
`materialize` 通过 AST 定位 PARAMS span，只替换该字典，要求完整相同 keys，重新解析并检查 span 外文本未变。
键必须是 string，值必须是 int/float/str literal；bool、运行时表达式、多份 PARAMS 或缺失 PARAMS 均不满足契约。
来源：`models/core.py:79至120`；`models/reports.py:130至150`；`paramspace/materializer.py:40至55、65至130，find_params_span/materialize`。

`guard.check_config` 检查缺键/多键、类型、choices 成员和 constraints；float kind 接受 int 或 float，但不接受 bool。
约束环境含参数和 `MAX_REGS_PER_THREAD/MAX_SHARED_BYTES/MAX_SHARED_BYTES_OPTIN/MAX_THREADS_PER_BLOCK`。
允许算术、比较、and/or/not；不允许函数调用、索引、in/is、三元表达式。依赖关系可写逻辑约束，TPE 仍建议全部维度，不自动创建 conditional-knob DAG。
guard 是作者声明的组合合法性，不是编译器资源证明。比如估算 tile bytes 的表达式不能保证与实际 shared allocation 一致。
来源：`paramspace/guard.py:26至136，check_config`；`models/core.py:409至434，DeviceLimits.as_env`。

### 6.2 发布前的有限验证

validator 要求源码默认键与域键完全一致，每域至少两个不重复 choice，默认值属于域，约束语法合法。
空间总组合≤512时枚举，否则以 seed 抽512个组合；满足声明约束的比例须至少25%。这不是GPU正确率，也不是512次随机 forward。
之后测两个不同源码 witness：默认 literal 配置，以及各域 `choices[0]` 拼出的 minimal 配置。
minimal 按声明顺序，不是自动找数值最小或实际资源最省。若该组合不满足 guard，则寻找另一个非默认可行组合。
两个配置物化成相同源码会被判 `inert_space`。这只说明不能用同一文件充当两份见证，不证明每个 knob 都影响当前执行分支。
每个 witness 都调用 quick_test，默认3个 correctness inputs和20个性能样本；通过才发布空间。
默认 witness 失败直接拒绝。minimal 失败仅在非有限输出或多个正 ref_absmax 比值≥10⁴等保守越界指纹下，最多再试两个不同可行配置。
替代点按相对已排除配置的 Hamming 距离优先；普通有限小误差不会自动走回退。没有常规 max、mid、random GPU witness 集。
默认点准入没有单独调用完整 `check_config(default)`，不能把前面的成员检查说成已验证默认点全部约束。
提示词的2至8 choices、cheap-to-expensive、至少两个 knobs等要求，与实际 schema/validator 强制条件不同；不能把提示要求升级为保证。
来源：`paramspace/validation.py:35至66、98至262、264至371，SpaceValidator/_looks_out_of_range`；`agents/modules.py:878至918、1011至1023`。

### 6.3 retry、repair、rewrite 的分工

| 行为 | 触发 | 实际身份与默认预算 |
|---|---|---|
| agent output retry | schema、文件、lint/check_output失败 | 同一调用的纠错；通常最多3 attempts，analyst最多2 |
| transport retry/rescue | AgentCallError | 先从已有文件恢复，否则新session重seed；不在总attempt外无限追加 |
| parameterization retry | 空间契约拒绝 | 将拒绝反馈parameterizer，仍是当前candidate |
| artifact repair | default/minimal witness执行失败 | RepairAgent修当前source；默认主Loop A最多两次中间repair |
| structural rewrite | 需要新的计算结构/性能假设 | 注册新child，重新参数化、见证与调参 |

主 Loop A 默认最多3次参数化/验证，成功修复通常覆盖当前 candidate 的 source，不另建 repair child；用尽则 dropped。
普通 tuning trial 的 compile/runtime/correctness失败会记录并 tell，不会为每个失败点启动 repair campaign。
witness 路径传给 repair 的 kind 是 `witness_default_failed` 或 `witness_minimal_failed`，具体 compile/numerical错误在 detail；专用 guidance 表未按这两个外层名字匹配，所以通常走默认 guidance。
repair 仍收到 reference、semantics、失败配置和先前修复历史。`inert_space`、语法/键错误、数值错误、编译错误和环境依赖缺失不应混记为同一种失败。
当前 C2 native retune 的 witness 拒绝直接返回 rejected，不自动继承完整主框架的最多两次 repair。
来源：`agents/base.py:121至301、322至358`；`agents/modules.py:436至512、1580至1661`；`control/orchestrator.py:1180至1299、1602至1668`；`scripts/experiments/c2_retune.py:78至100`。

## 7. Loop B：TPE、目标、预算和缓存

### 7.1 项目实际设置与库背景分开说

每个新 space 新建 `TPESampler(seed=run.seed, multivariate=True, group=True, n_startup_trials=10, constant_liar=...)` 和 minimize study。
主路径的 constant_liar 来自 concurrency.enabled，但真实控制流是串行 ask、evaluate、tell；这个 flag 不会自动产生多个在飞的 GPU trials。
所有参数都执行 `suggest_categorical(name, choices)`，包括整数和浮点数。项目明确设置 startup=10；不要从不同版本官方页面的默认数字反推项目配置。
项目未显式传入 prior_weight、gamma、weights、n_ei_candidates 等；这些由安装的 Optuna 版本决定，依赖声明 `optuna~=4.9` 不等于历史环境锁定证明。
来源：`tuning/tpe.py:24至77、102至154，OptunaTPETuner`；`control/orchestrator.py:1315至1408，_tune`；`pyproject.toml:6至10`。

**TPE 的算法背景**：将已有目标观测按较好与其余部分划分，在参数空间分别估计 l(θ) 与 g(θ)，从较好侧模型采样候选，再偏向较大的 `log l(θ)-log g(θ)`。
对最小化延迟问题，l对应较低延迟的观测区域；这是一种基于观测分布的概率搜索，不是沿某个硬件梯度做下降。
multivariate/group 开启库的联合建模及分组机制；categorical 维度使用分类概率质量，不能把所有离散 choices 描述为字面上的连续 Gaussian 分布。
constant-liar 的库语义涉及避免靠近 RUNNING trials，但串行主循环不因此获得异步 BO。项目没有固定的“百分之几探索、百分之几利用”比例。
这些是 **Optuna 库算法背景**，不是项目自己实现的 wall acquisition，也不是把资源利用率加入 l/g 的新目标。
官方资料：[TPESampler API](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html)；[TPE sampler implementation](https://optuna.readthedocs.io/en/stable/_modules/optuna/samplers/_tpe/sampler.html)。stable页面会变化，论文复现仍需运行时版本。

### 7.2 优化标量与 ask/tell

定义 `L(θ)=LatencyStats.robust_ms`：存在正 median 时用 median，否则回退 mean；不是使用最小样本 min。
只有 complete 且目标 finite 的记录参与正常 objective 排名。通用 TaskSearch 的自定义 J 是另一接口，不能把其最大化/绝对gain语义套给本节。
每次 ask 内部最多尝试64次：抽全配置，检查完整 key 去重、guard与可选降权；拒绝时 tell PRUNED 并继续，尚不增加 admitted `_asked`。
点被返回给调用方才计一次 B，之后即使 GPU失败或复用已有测量也占该槽。达到 B 或64次拒绝用尽可提前返回 None，不证明整个空间穷尽。
tell时，可用目标提交数值；`infeasible_shared_memory/guard_rejected/materialize_error` 对应 PRUNED，其他 correctness/runtime等失败对应 FAIL。
没有把资源失败编码成虚构的大 latency。封装没有提交中间目标值，不能仅凭PRUNED存在声称TPE已学习精确约束边界。
来源：`models/core.py:123至156，LatencyStats`；`tuning/objective.py:31至52`；`tuning/tpe.py:102至155、232至275`。

### 7.3 anchors、fresh intent 与可选引导

witness 被转为 enqueue anchors，带 latency 的 witness 同时形成测量 cache；抽到后复用测量、写 `reused_measurement=True`、占 B、正常 tell。
rewriter/parameterizer 推荐的最多两个完整配置，需要 source身份匹配、guard合法、不与anchors重复；它们是未计分建议，不是免费成功点。
普通 enqueue 不允许 already-drawn完整点；`enqueue_fresh` 允许重复并计 fresh intent。C4同端点两次要有两个intent，调用方绕过测量cache，才是真正 fresh replicate。
static cache、compile-screen facts cache与timed measurement cache用途不同。compile timeout或non-answer不缓存为不可行；不能把它们统称correctness cache。
S8启用才为至少3个合格数值choices提供 rung distance：`d(a,b)=|index(a)-index(b)|`，不是原始差或log差。
S1b启用时，同candidate某(knob,value)至少7次correctness mismatch且零complete才降权，接受概率1/4；任一次complete永久撤销该规则。多个坏值不连乘，也不删除choices。
S7另按旧墙提出同预算点，见第10节；这些规则改变候选队列，不改变TPE的latency目标。
来源：`control/accepted_candidate.py:12至36`；`control/orchestrator.py:1043至1067、1343至1404`；`tuning/tpe.py:165至211`；`tuning/ordered_domains.py:68至170`；`tuning/deweight.py:71至157`。

## 8. 可选扩域：先说明为何扩，再生成新 choices

### 8.1 observed boundary 与 effect

对某轴值x，analyzer把所有可用trial按x分组，取边际目标中位数m(x)。伙伴参数可能不同，因此这不是固定其他参数后的单轴响应。
`effect_pct = 100 * (max_x m(x) - min_x m(x)) / min_x m(x)`。
`best_value` 是边际m最小的轴值；`best_trial_value` 是最快完整trial的实际轴值。主边界判据优先后者，不可用才回退前者。
在域声明顺序下，anchor位于已成功观测范围一端，端部最多3个中位数向边缘非严格改善，且至少2点、范围外无成功值，才得到at_boundary及min/max方向。
未测与失败都可能造成观测边缘；这不是物理上限，更不是完整域中每个值都测过。
来源：`tuning/stats.py:81至156，TuningStatsAnalyzer._param_stat`。

### 8.2 请求、验证与停止

扩域默认关闭；开启时，至少一个已知register/shared使用比例小于0.8才提供余量证据。没有profile或没有已知比例则宽松放行，含义是unknown allowed，不是实测空闲。
数值轴需effect达到配置门槛，默认2%，并符合边界方向；按名字后缀有下限WARPS=1、STAGES=1、BLOCK_K=16，没有对称通用上限表。
边缘失败率小于0.30的请求优先：存在健康请求就只留健康项，否则保留原非空列表；缺失败率按0处理是不降优先级，不是证明成功率100%。
主winner-anchored请求为空时，才用边际最优在端点的fallback，后者不要求同样的尾部单调检验。
parameterizer收到当前域、原约束、selected source/params、统计和历史，被要求给指定轴增加1至2个合法值；没有自动翻倍或固定步长外推算法。
每次扩域最多两次proposal/validation；choices字典完全相同会拒绝。请求要求保留旧值与结构，但host没有完整验证严格超集，顺序变化也能令tuple比较不同。
遗漏旧约束时，host逐条尝试恢复；对每个新增choice最多随机2000次寻找在恢复约束下可达的组合，未找到不等于不可达证明。
恢复发生在witness validation后，不承诺最终约束又完整跑过witness。当前扩域prompt也未渲染所有prior_feedback，不能保证第二次模型看到具体拒绝原因。
接受后同candidate/family，新space ID、version+1、新TPE与新B；source body可能改变，并记录source_changed。
旧winner仍合法可作anchor，只有新物化源码与旧trial字节一致才能复用旧latency。候选历史best不倒退，不能把旧分数直接归给变过的source body。
若 `new_best > old_best*(1-0.02)`，停止后续扩域；等于2%门槛不触发该停止分支。改善也可能来自重搜旧组合，不一定由新增值造成。
来源：`control/orchestrator.py:267至430，boundary_knobs_to_expand`；`:2424至2727，_maybe_expand_space/_restore_dropped_constraints/_choice_is_reachable`；`agents/modules.py:940至1009，_render_expand_prompt`。

## 9. 资源证据：测到了什么，没有测到什么

### 9.1 编译资源、运行成本与单位

| 字段 | 来源与单位 | 不代表什么 |
|---|---|---|
| n_regs | compiled metadata，registers/thread | runtime平均活跃寄存器 |
| Triton n_spills | compiled属性，local-memory 32-bit slots | 动态spill load/store次数或DRAM bytes |
| CUDA/cubin n_spills | 当前parser把STACK+LOCAL bytes写入同名字段 | 不能与Triton slots直接比较；非零stack/local也不必来自spill |
| shared_bytes | compiled shared bytes/block | 每SM带宽或所有block总分配 |
| SASS counts | 静态指令出现次数 | 执行次数、循环频率、cache hit |
| occupancy | 资源容量的解析估计，fraction | profiler measured achieved occupancy |
| peak_alloc/reserved | caching allocator peak，bytes | 候选独占或全卡used memory |
| candidate_aten_bytes/ops | dispatch逻辑tensor bytes / op次数 | fused kernel内部DRAM traffic或实际GPU launch数 |
| threads_launched | 可解析grid×warps×32的求和 | 未必覆盖callable grid，也无普遍优劣方向 |
| cpu_issue/overhead_gpu/wall_ms | 独立overhead probe，ms/call | 与cold-cache headline latency相同的量 |

LightProfiler优先采用有kernel记录的Triton数据，否则用cubin；资源字段分别取max、SASS求和、occupancy取min，可能来自不同kernel，不能假定有共同witness或时间权重。
Triton metadata提取会重新load、forward并检查JIT caches，没有为每个缓存项保证本次launch筛选；compile_s还含load/初始化/forward/synchronize，不是纯编译时间。
candidate ATen计数看不到fused Triton内部访问；reference与candidate计数器的去重规则也不同。不能声称 `candidate_aten_bytes <= actual_DRAM <= reference_bytes` 是经过证明的不等式。
unknown通常用None，不能改成0；但TaskCost失败0+notes、occupancy默认容量、classifier内部0 fallback和旧stats truthiness确有例外。论文应保留notes/provenance而非假装整条链无缺测。
来源：`evaluation/profilerx.py:40至118，LightProfiler.extract`；`gpu/worker_main.py:348至541、589至626、1099至1147`；`models/core.py:181至296`。

### 9.2 occupancy 的实际估算

令每线程寄存器r、每block warps数w、shared bytes为s；SM容量为R_SM、S_SM、T_SM和最大block数b_max：
```text
W_max = floor(T_SM/32)              R_block = max(1,r)*32*w
b_r = floor(R_SM/R_block)           b_w = floor(W_max/w)
b_s = floor(S_SM/s) if s>0 else b_max
b = max(0, min(b_r,b_s,b_w,b_max))   occupancy = b*w/W_max
```
limiter是这些block数约束中的最小项，不是独立occupancy硬墙。公式不模拟真实寄存器分配粒度，可能偏乐观。
worker缺属性时有65536 registers/SM、102400 shared bytes/SM、16 blocks/SM fallback，缺shared也可能按0传入；因此这项估算需要同时报告输入来源。
来源：`evaluation/statics.py:328至381，compute_occupancy`；`gpu/worker_main.py:296至309，_tier1_statics`。

### 9.3 reference cost、roof 与 pressure

reference cost model记F为FlopCounterMode得到的FLOPs，Bc为输入/参数/buffer/首个最终输出的逻辑compulsory bytes，Br为reference ATen materialization bytes。
`AI_max=F/Bc`、`AI_ref=F/Br`，单位FLOP/byte；`fusion_headroom=Br/Bc`是逻辑流量比，不是必然speedup。
view/alias按首次tensor大小而非storage区间并集计量，多输出、未用参数、未覆盖op或算法化简均限制其物理下界解释；分母非正时属性可返回0。
校准roof由event median t测得：streaming-add带宽 `8N/(t_ms*1e-3*1e12)` TB/s，GEMM吞吐 `2n^3/(t_ms*1e-3*1e12)` TFLOP/s。
cuBLAS分别测fp32/tf32/fp16/bf16；Triton reachable GEMM每精度测六配置，过相对FP64 correctness gate后取最高吞吐，不是任意Triton程序的不可超越上限。
所谓empty launch floor实际是一元素torch.add。DRAM探针尺寸由free memory限制，GEMM从8192按显存减半，不是注释所暗示的自动L2尺寸适配。
来源：`evaluation/task_cost.py:50至138`；`gpu/worker_main.py:1272至1435、1579至1689`；`gpu/tritonmm.py:83至193`。

用候选headline latency t形成 `A_BW=Bc/(t_ms*1e-3*1e12)`、`A_F=F/(t_ms*1e-3*1e12)`，再除相应roof得到p_BW和p_F。
对同一task，F和Bc固定，因此两项pressure都是**1/t的缩放**，不是独立测得的带宽利用率或执行FLOP counter。变快与pressure变高不能算两份独立成功证据。
通常精度roof取cuBLAS/Triton较大值，但classifier仅在静态SASS判断有tensor core时走该选择，否则用cuBLAS fp32；缺精度roof还有fallback。
候选precision来自params/source启发式，不能代替实际指令精度审计。peak、cost或precision未知时分类也可能使用fallback，单个label不证明硬件空闲。
来源：`evaluation/bottleneck.py:148至210、398至497、650至707`；`control/orchestrator.py:79至135，_detect_candidate_precision`。

### 9.4 ceiling、floor 与诊断标签

capacity headroom为 `max(0,limit-used)`，使用比例为used/limit；与用latency作分母的pressure不是同一概念。
八维DimensionState是regs、spills、shared、allocation、ATen bytes、ATen ops、threads、occupancy，不包含DRAM/compute pressure。
一般容量比例≥0.90为binding、≥0.70为near-binding；occupancy反向，小于0.30为binding、小于0.50为near-binding；spills>0为binding、0为slack。threads无优劣极性，ATen计数没有ceiling。
另一个bottleneck classifier的resource-near-limit阈值为0.80，不能与八维band或硬编译拒绝混用。classifier先检查时间/launch/issue/memory/compute，之后才可能resource_limited、latency residual或mixed。
模型floor包括spills=0、ATen ops=1、ATen bytes=Bc、allocation=Bc且confidence=0.5；regs/shared/occupancy无可推导floor。
`room=m-floor`、`room_frac=(m-floor)/m`；m=0时frac未知，低于floor保留负room并标below_floor。逻辑floor不是普适硬件下界。
这些维度是解释和反馈，不参与ConvergencePolicy的停止公式；binding不等于已经找到最优。
来源：`evaluation/dimensions.py:136至187、230至373`；`evaluation/bounds.py:140至224，lower_bound`；`evaluation/bottleneck.py:352至707`；`control/convergence.py:11至65`。

## 10. 六种“斜率”必须分开

| 种类 | 数据/固定条件 | 实际量与用途 |
|---|---|---|
| 边际effect/尾部gain | ordinary trials，partners可变 | 分组中位数百分比，用于扩域或旧墙筛选 |
| legacy hard/soft wall | 边际趋势；hard另有单轴compile消融 | 条件拒绝归因或spill趋势，不是统一资源导数 |
| v4.1 C4 contrast | frozen partners，F/N各两次fresh | log-latency contrast、相对gain与noise gate |
| raw TaskResponse | 两个有效端点，固定partners | ΔJ/Δx、ΔJ/Δr，有真实量纲 |
| spill OLS conversion | 同candidate多trial，可混partners | latency对spills回归，只写日志 |
| family recent gain | 相邻两次family历史best | 百分比改善，供family调度 |

固定父C2 information使用acquisition产生的raw TaskResponse及其掩码，不应自动改写成v4.1 C4扫描器；其ordinary history仍含边际统计和旧资源信息。
generic task rewriter也可构造raw响应，但不是主KernelBench `_tune` 自动执行的两点导数机制。以下各公式不能合成一个从未实现的统一slope目标。
来源：`tuning/stats.py:81至156`；`conditional/task_response.py:15至74`；`conditional/gates.py:74至141`；`evaluation/conversion_rate.py:139至265`；`scripts/experiments/c2_information_inputs.py:72至87`。

### 10.1 legacy hard wall、soft wall 与 S7

生产shared预筛以compiled required bytes严格大于per-block opt-in limit判拒绝，等于允许；unknown不拒绝。预筛默认抽至多min(64,max(16,B))个合法未重复配置，不能当全空间证明。
旧hard-wall finder要求numeric轴至少三个成功值，拒绝值在整个成功范围外；两侧都有拒绝时优先高侧，取该侧最近拒绝值。
尾部从远离墙到靠墙的latencies为l0,l1,l2，`G_tail=100*(l0-l2)/l0`；hard要求l1<l0、l2<l1且G>0。
它没有除参数距离或资源变化。比如16/32/64对应10/9/8ms，effect=25%，tail gain=20%；这两个百分比连分母都不同。
hard归因在最快K个trial上仅把该轴改成拒绝值、冻结partners再compile，默认K=1，可加第二origin。所谓default origin实际为每域choices[0]，不是源码literal default。
fit/refused/unknown得到not_attributed/attributed/undecidable；只证明这些partners下单改该轴仍超限，不证明解除限制后会更快。
旧soft-wall先要求candidate最快trial已知spills>0；轴上至少三个有spill测量的值，spills非递减且端点上升，尾部端点G>0。它不要求latency逐点严格下降，也不新做独立probe。
S7默认关闭；启用后每10个told重算，按G降序、同轴去重，最多建议2点。在当前最快params上只改轴，选朝墙最远的已声明合法choice，不跨同向known refusal。
S7不扩域、不加B、不产生UCB；soft引导还要同时开启use_soft_wall和soft_wall.enabled。
来源：`control/orchestrator.py:1531至1584、1670至1756、1796至2025`；`evaluation/wall_attribution.py:240至398`；`evaluation/soft_wall.py:230至358`；`tuning/slope_guide.py:166至428`。

### 10.2 raw TaskResponse：有量纲的有限差分

两个endpoint evaluation均valid且score非空，差值finite，才形成 `ΔJ=J_B-J_A`；数值轴且非零finite距离才有 `parameter_slope=ΔJ/(x_B-x_A)`。
资源r存在且差值finite时记 `Δr=r_B-r_A`；Δr非零才有 `resource_slope=ΔJ/Δr`。零变化保留delta=0但无slope，缺测进入unknown，不能填零。
Objective.gain按最小化/最大化方向给绝对改善，derivative仍保留ΔJ的符号，不改成正收益、不做log归一化。名义字符串轴只有contrast，没有参数数值斜率。
例子仅用于计算：J为最小化ms，x从32到64，J从10到9，r从40到50，则ΔJ=-1ms、gain=+1ms、parameter_slope=-1/32 ms/x单位、resource_slope=-0.1 ms/r单位。
负斜率仅表示这次对比中资源上升伴随J下降，不是资源变化造成加速的因果证明，也不能外推x=128；资源无unit metadata时不能猜其单位。
generic `task_rewrite.probe_responses`按轴首尾合法值锁定parent其他参数，新评估端点，消耗独立probe budget，不tell TPE也不加入incumbent trial list；它没有C4的replicate gate。
来源：`conditional/task_response.py:15至74，complete_response`；`tuning/objective.py:18至28`；`control/task_rewrite.py:29至95，probe_responses/run_rewrites`。

### 10.3 spill OLS：只日志，不喂给模型或 sampler

同candidate至少8个complete且有latency/resource的trial，spills至少3个distinct值，才拟合x=spills、y=latency_ms：
`β=Σ(x-x̄)(y-ȳ)/Σ(x-x̄)^2`，`α=ȳ-βx̄`，`R²=1-Σ[y-(α+βx)]²/Σ(y-ȳ)²`。
需R²≥0.4且slope/R² finite；常量x/y不输出。混合零/非零spill时，合格非零子集若出现相反符号，整条结果被抑制。
比例版为100β/base_latency，base取最小spill值对应latency的上中位数。它未控制partners混杂，CUDA/Triton spill单位问题也仍适用。
主 `_stats_and_analysis` 只记 `CONVERSION_RATES`。`for_prompt`虽存在但该路径不调用；OLS不进入analyst、rewriter、TPE、family排名或资源奖励。
来源：`evaluation/conversion_rate.py:71至86、116至265`；`control/orchestrator.py:2034至2050`。

## 11. v4.1 可选路径：U/W 探测、C4 与 E1

### 11.1 编译探测和条件墙

每candidate一个ConditionalScanBridge，tokens跨其spaces保留；每space新建planner/scanner。off不建，observe做probe/墙/事件，active再enqueue扫描并投递brief。
每10个told checkpoint触发，失败与复用trial计told，ask内部PRUNED重抽不计。planner围绕candidate累计历史最快complete trial的partners，不是全run winner。
U按回答最少的轴优先，tie按轴名；轴内从incumbent出发，按声明阶梯距离由近到远，同距先lower。
W轮转inward control、fit/refused转折邻点、2×2 corner和remote refusal walk-back；corner需要四角都有回答且三个fit、组合refused才记交互证据。
探测只有reached kernels的compiler facts，不含latency或correctness认证。shared超过limit为refused，已知不超为fit，答不出为unknown。
hard wall保留相邻fit→refused局部边，还要fit端内侧有另一个相邻fit，才有两个可C4端点；否则F=N，只能记墙。
v4.1 soft wall取已知fit且有spills的最后两点，要求spill_N>spill_F且spill_N>0；可跨unknown/refused缺口，端点最大spill也可能来自不同kernel。
因此它与旧soft wall“incumbent已经spill+边际latency趋势”不是同一准入条件，也不是二分搜索或全域单调可行性证明。
来源：`conditional/bridge.py:63至133、272至281`；`conditional/probe.py:179至511，ProbePlanner`；`control/orchestrator.py:1461至1527`。

### 11.2 probe秒预算与scan槽分开

编译诊断独立计秒：bootstrap envelope=600秒，第一次非timeout且非failed batch的实耗m冻结 `D=min(2m,600)`；以后仅在spent+projected≤D时放行。
bridge用projected=30+3n秒；first zero-answer视为timeout，可按冻结顺序取前ceil(n/2)重试一次，此后停止该space诊断，不用retry成本重冻D。
pcap默认48，attempt cap实际使用常量MAX_ATTEMPTS=4；配置max_probe_attempts没有传到账本，不能当有效可调参数宣传。
scan另从既有B切出 `Q=floor(B*f_frac)` 个槽，没有额外赠送TPE预算。B=40、f=.1时Q=4；启用样本f=.125时Q=5。
C4原子准入要求B剩余和Q剩余都至少4，扣4；E1扣1。有继承token时也可能先用E，不能说Q=4时任何情况下都不可能E。
来源：`conditional/probe.py:40至42、95至150、521至525`；`conditional/bridge.py:137至212`；`conditional/scanner.py:119至179、183至227、293至319`。

### 11.3 C4 的全部公式

事先固定F为内侧、N为墙侧，固定partners；不能看速度后把较快点重新命名N。
每端两次fresh trials，共4次；一对discovery、一对validation，各对内部随机顺序，不强制ABBA。每trial又有自身quick计时样本，旧incumbent不充fresh replicate。
对两端正latencies集合F、N，GM为几何均值，定义：
```text
c = mean(log F) - mean(log N)
g_out = 1-exp(-c) = 1-GM(N)/GM(F)
I=[L,U] = [1-max(N)/min(F), 1-min(N)/max(F)]
rho_e = (max(e)-min(e))/GM(e)
tau = max(0.04, rho_F, rho_N)
outward passes iff L > tau
I_rev = [-U/(1-U), -L/(1-L)]
inward passes iff lower(I_rev) > tau
```
反向区间是交换端点后的精确比例变换，不是简单取负；实现对U趋近1作1-10^-12 cap。
discovery gain为1-N_d/F_d，validation gain为1-N_v/F_v；最终gate用全部四次replicates，不只discovery。
每端不足2 fresh或有非正latency则quality不过、noise未知、不mint token；函数本身没有对全部异常浮点做显式isfinite过滤，应依赖上游有效性并保留此边界。
这是log contrast/相对收益，不含Δparameter或Δresource分母；choice_span只用于排序。I是经验replicate envelope，**不是统计置信区间**，4%是固定floor而非拟合出的置信水平。
例：F=[10,10]ms，N=[9,9]ms，c≈.10536、g=10%、I=[.10,.10]、rho=0、tau=.04，outward通过；反向为[-.11111,-.11111]。
在B40/f=.1下，这个C4已耗尽Q4，即使mint成功也无本space余槽执行E1；f=.125才可能余一槽，还需有合法可测点。
来源：`conditional/gates.py:34至49、74至141，evaluate_gates/reversed_interval`；`conditional/scanner.py:183至227、395至428`。

### 11.4 token 与 E1 的真实作用

通过方向gate的完整C4创建一次性token，生产默认mint E1；admission即消费，失败不返，E结果不递归mint。
同comparable_key新C4撤销旧pending token，不同F/N不是同一contrast，不能简单按“轴最新结果”覆盖全部。
E在token冻结partners下，向所支持端最近的未测screen-fit declared choice寻找，之后可找区间内点；unknown/refused gap不在同次方向遍历跨过，outward另有F之前一档fallback。
排序为 `(choice_span,-direction_aware_log_gain,axis_E_admissions,queue_entry,token_id)`，不是UCB，也没有实现注释中的strata rotation；round-robin counterfactual只日志。
生产bridge传fits map；standalone scanner在fits_screen=None时有放行fallback。E仍占B/Q，不扩大域，也不保证比incumbent快。
E2/E4有类型并不等于完整实现：生产只mint默认E1，E4虽扣4槽却只生成一个E_new，不能声称已实现四点E4协议。更多身份/brief限制见附录B。
来源：`conditional/tokens.py:57至158，TokenStore`；`conditional/scanner.py:152至179、249至373`；`conditional/bridge.py:216至268、359至392`。

## 12. 谁看到什么反馈，谁决定接受

### 12.1 测量、派生量、解释和账本四层

| 角色/层 | 输入或产生内容 | 不拥有的权限 |
|---|---|---|
| worker/profiler | correctness、latency、compiled resource、cost记录 | 不自动证明结构因果机制 |
| harness | stats、walls、DimensionState、局部contrast、账本对账 | 不能把unknown补成真实0 |
| analyst | source、selected params、trials CSV、stats、reference/semantics、profile；主路径可加cost/calibration/verdict与启用的brief | report是LLM解释；suggested_action不是停止授权 |
| rewriter | selected materialized parent、report、失败假设、允许的wall/response/ledger、reference与设备 | 不能用expectations直接晋级 |
| parameterizer | candidate source、任务/设备、prior feedback、rewrite intent、推荐配置；扩域另给原域约束、selected source/history | 不把旧source测量自动转给新body |
| repair | broken source、reference、semantics、失败配置/detail和前次修复结果 | 不保证修好即比baseline快 |
| generator/novelty | reference、设备、语义、校准；novelty另看family anchors/摘要 | 输出仍需注册、见证和tuning |

主分析在调参后先形成stats、profile/launch-overhead诊断、允许的wall/dimension/conditional材料，再调用analyst。rewriter还可直接获得这些文本，不只依赖analyst转述。
固定父C2 wrapper不是这个输入表的全集：其analyst传profile、evaluation digest、ordinary历史/reference，但未自动传主框架calibration/task_cost/bottleneck_verdict；rewriter也未自动拿family ledger或wall_text。
provided acquisition路径先有responses，再运行对应组的legacy分析与改写；targeted路径则先用ordinary信息让analyst规划最多六个请求，再采集responses，直接交rewriter。
targeted的report明确早于responses，当前不会再多跑一个分析agent解释新数据；rewriter需处理原假设与新观察冲突。
来源：`agents/modules.py:604至805、1029至1362、1472至1661`；`control/orchestrator.py:2027至2150、3333至3446`；`scripts/experiments/c2_local_agents.py:83至148`；`scripts/experiments/c2_information.py:93至106`。

### 12.2 事前expectations与事后reconciliation

rewriter可为每个child声明资源方向up/down/unchanged/unknown和why，合法维度由schema限制，不接受随意增加的百分比预测字段。
声明在REWRITE_PRODUCED中先于child测量保存，后续按**该child自己的retuned best**与共同parent best逐维比较，不借用sibling winner的profile。
资源变化delta=after-before，通常相对幅度严格超过5%才视为material；before=0且变化时rel=1，不变为0。reconcile使用round后的rel，5%边界有舍入差异。
结果包括hit、miss、vacuous、unpredicted、unmeasured；资源改善但latency未达到改善门槛可为no_conversion，不能推广成“该资源永远不是瓶颈”。
regs/spills/shared/allocation/ATen量使用lower-is-better，occupancy使用higher-is-better，threads只记changed；这是报告极性，不保证occupancy升高必快。
ledger记录与下一轮prompt投递分开，后者默认关闭。expectation准确率不进入acceptance、TPE或family分数；预留槽至多读取“有无ledger”。
parent best对child retuned best同时改变结构与配置，所以方向miss不必全是结构推理错误。固定父legacy wrapper的LegacyProposal未保存expectations，不能声称该路径也有完整逐child账本闭环。
来源：`models/reports.py:153至206`；`evaluation/conversion.py:90至149`；`evaluation/reconcile.py:105至274`；`control/orchestrator.py:3175至3331、3367至3371、3428至3446`；`scripts/experiments/c2_local_agents.py:42至49`。

### 12.3 工具与formal helper

agent调用建立新sandbox/session，seed上下文文件并提交结构化schema；可用bash及读写工具，sandbox路径检查只限制host封装的相对路径，不是OS隔离。
主wiring给六个模块安装可选formal self-test context；direct/generic task rewriter入口没有同样注入。工具可用、提示建议使用、实际采用必须靠不同证据说明。
formal helper复用真实evaluator配置、seed、reference hash、worker路径和语义，返回source/params/quality/latency/cost等；有效需ok、compiled、correct及有限正score。
helper结果标agent_self_test，不是原生B40 trial，不自动导入TPE score/cache；之后parameterizer若改body，之前自测不覆盖新产物。
private profiling或ad-hoc测试能力不等于所有调用都遵循formal协议；helper cutoff也不拦截agent全部bash命令。写工具采用率必须看实际调用记录。
来源：`agents/base.py:101至146`；`agents/sandbox.py:55至95、125至137`；`agents/self_test_context.py:40至122`；`agents/self_test.py:59至169`；`wiring.py:144至163、203至208`。

## 13. 正确性与计时：实际执行的质量合约

### 13.1 strict 与有限覆盖

strict先静态检查，再在worker运行reference与candidate，要求输出shape一致，所有规定correctness trials通过后才计时。
当前调用精确为 `torch.allclose(output, output_new, atol=tolerance, rtol=tolerance)`，output是reference r，output_new是candidate y。
allclose对第二个参数使用相对项，因此对有限元素实际公式是 `|r_i-y_i| <= atol + rtol*|y_i|`，并非经常口头写出的以reference为相对分母的版本。
fp32实际atol=rtol=1e-4；fp16/bf16为1e-2。EvalConfig声明的atol/rtol=1e-2未传入这条strict调用，不能据配置字段名误报阈值。
这也不是独立形式证明或全输出域有限性证明，具体特殊浮点行为仍由allclose决定；有限input覆盖不推导任意shape、任意状态序列、反向传播或整个参数域正确。
来源：`../KernelBench/src/kernelbench/eval.py:83至102、784至815，get_tolerance_for_precision/run_and_check_correctness`；`gpu/jobs.py:109至145`；`evaluation/correctness.py:407至466`。

### 13.2 dual-witness-relaxed 与 FP64 rescue

此处“双见证”指TF32和IEEE两种reference，不是参数空间default/minimal。对任一reference r与candidate y，先shape相同，再转float32比较：
`e_i=|r_i-y_i|/(|r_i|+1e-7)`，`f=count(e_i<0.01)/N`，默认要求 `f>0.99` 且cosine≥0.99985；两个reference任一个通过即可。
两个不等式都是严格边界：误差严格小于1%，比例严格大于99%。cosine使用float64和最大绝对值缩放，全零返回1；NaN cosine在比例门后有candidate全有限fallback。
仅两个reference都失败、FP64 gate显式启用且golden可构造时，尝试额外rescue：
`RMSE(a,b)=sqrt(mean((double(a)-double(b))^2))`，接受条件为 `RMSE(g,y) <= m*RMSE(g,r_tf32)+0.001`。
g为FP64 reference，m通常2，识别为低精度计算或fp16/bf16输出时为3；误差地板比较的是TF32 reference，不是IEEE。
低精度识别优先用PARAMS dtype字面值，否则文本token启发式，不是机器指令证明。FP64 unavailable不自动通过，NaN RMSE拒绝。
rescue通过后不再要求原relaxed fraction/cosine通过，所以必须单列rescue次数；不能将其计为strict allclose通过或逐元素统一误差界。
来源：`gpu/worker_main.py:1987至2138、2290至2383，_relaxed_close/_computes_low_precision/run_relaxed_correctness`；`config.py:325至347`。

### 13.3 RNG、training状态与计时

reference factories产生init_inputs/inputs；相同seed分别构造reference/custom，不复制state_dict。每trial生成input seed，但顺序forward之间不保证重新seed、clone输入或恢复模型状态。
relaxed同一reference model先TF32再IEEE，再custom；BN running state、in-place输入和Dropout RNG因此需要单独解释，不能声称随机掩码和状态逐forward严格配对。
默认保留reference的train/eval状态，no_grad不等于eval。FP32输入或IEEE全局matmul设置也不能约束自定义kernel内部精度。
quick默认3 correctness+20 perf，full默认5+100；correctness输入次数、计时sample、重复block和独立生成次数是不同单位。
默认CUDA-event路径先warmup，再逐样本synchronize、在start event前用256MiB dummy fill执行L2 thrashing，再记录当前stream的forward时间；默认discard_first=1。
candidate通常warmup=3，relaxed明确如此；baseline helper为5。这个cache操作是实现中的cold-cache意图，不是证明任意设备完全清空所有cache。
当前stream events也不是对任意自建stream行为的完整防护。独立overhead probe的warm-cache时间不要与headline latency混用。
最终选优使用robust_ms，但headline speedup用mean/mean；两者职责不同，不能交叉混分母。
来源：`gpu/worker_main.py:2241至2248、2275至2288、2327至2342、2442至2465`；`../KernelBench/src/kernelbench/eval.py:727至783`；`../KernelBench/src/kernelbench/timing.py:126至142、201至281`。

## 14. Loop C/D：family迭代、调度与收敛

### 14.1 exploration与performance是三个控制层

第一层TPE在给定结构的有限空间中按目标分布采样；第二层family规则分配结构改写机会；第三层probe/scan预算限制诊断与局部确认。没有统一UCB或“资源潜力总分”。
对active且有有效incumbent的family，legacy排序精确为 `(unproven, -recent_gain, incumbent_latency)` 升序。
`unproven=0`若rewrite_rounds_used=0，否则1；历史不足两点recent_gain=0，否则 `max(0,100*(h[-2]-h[-1])/h[-2])`。
优先给未改写路线机会，再偏向最近仍在改善者，最后才比较绝对latency；默认取前3个family，不代表3路GPU并行。
可选reserve需reserve_round_for_reconciled与expectation_ledger同时开启；slate至少有两个未改写family且外面有ledger family时，可替换最后一个未改写项。
预留按既有rank取waiting[0]，只读ledger存在性，不看hit/miss或预测准确率，也仍保留至少一个未改写family。
来源：`control/families.py:263至379，active_families/_improvement_pct`；`wiring.py:130至139`。

### 14.2 parent、siblings与历史

每轮parent是family.best的candidate及其selected point，不固定为seed或上次最后输出。优先读取对应materialized trial文件，否则尝试当前source物化，失败可退回source并标source_materialized。
同次rewriter输出的children共享固定parent，是siblings；无论backend或改动大小，继承第一parent family，每份独立走A/B。
family.best与candidate best按有效latency单调保优；较慢child不抹掉旧winner。历史每个结算轮追加一次family当前best，不按每个sibling追加。
初始有效family通过FAMILY_SEEDED建立h[0]；rewrite_rounds_used计机会，rounds_not_evaluated单独记无可用输出的轮，避免把所有agent失败当结构plateau。
只有本轮被计evaluated且best_after≥best_before，才将rewriter实际声明实施、且与analyst匹配的hypothesis IDs记失败。小于2%的真实改善不会自动进入“无改善假设失败”。
注册即被上层视为evaluated、novelty seed history缺口、同latency selected匹配等当前边界见附录A，不用理想化状态机掩盖。
来源：`control/orchestrator.py:1069至1074、2826至2993、3060至3172、3333至3457`；`control/families.py:381至407`。

### 14.3 精确停止条件与novelty

family先判rewrite_rounds_used≥5为budget_exhausted；否则历史至少3点，最后两次 `100*(h_i-h_(i+1))/h_i` 都严格小于2%才converged。
恰好2%不触发该plateau分支，同时满足轮数与plateau时预算标签优先。binding维度、墙数和analyst suggested_action不进入停止公式。
global先检查elapsed达到wall clock；或非空families全部不active时结束，仅全部frozen_converged才标converged，否则budget_exhausted。
novelty仅在本轮rewrite没有progressed且尚未到墙钟时尝试；先查productive cap，再查hard cap/结构签名/文本相似度。
productive定义为active或已有best；无best的frozen dead family不占productive但仍占hard总数，默认都为6。无有效parent的active family会被冻结，不能堵住rewrite slate。
外循环连续3次rewrite/novelty均无进展退出防空转；若所有family已frozen，global verdict可先结束，不保证最后再调用一次novelty。
这些是预算与观测历史的停止规则，不能称已证明不存在更好结构。
来源：`control/convergence.py:11至65，ConvergencePolicy`；`control/orchestrator.py:684至746、2995至3059、3492至3586`；`control/families.py:171至178、222至253`。

## 15. 固定父 C2 实验与完整框架的对应关系

### 15.1 冻结状态与 G0/G1/G2

Selection显式指定历史run、parent/trial/source/space/cutoff；Shared含reference、selected parent、ordinary trials、semantics/device/evaluation/backend/protocol和失败反馈。
准备要求parent是cutoff前成功ordinary trial，排除scan/reused历史点，源码hash和PARAMS匹配当时trial，space对应历史发布。off与conditional_on_treated_state标签不能抹掉。
原information stage每个parent只acquisition一次，先validate_for(shared)，再将同一fresh envelope分组mask；不是为G1/G2分别任意找不同端点。

| 组 | ordinary source/history/profile/reference | fresh endpoint objective/contrast | fresh resource |
|---|---|---|---|
| G0 | 保留 | 不提供fresh responses | 不提供 |
| G1 | 保留，包括旧resource与静态约束 | 保留与G2相同params、validity、J、delta/gain | 清端点metrics/detail和resource deltas/slopes等，标unknown |
| G2 | 保留 | 保留 | 完整保留 |

G1不是完全无资源知识，它只去掉fresh resource增量；空dict表示unknown，不是0。每opportunity重新调用对应analyst，不跨组复用report，也不把组名加进agent输入。
每组随后一次legacy结构提案，要求恰好一个child，再parameterize和native retune B40。native retune调用validator与原生调参/cache，不是完整多family `_run`。
targeted变体由planning analyst先请求contrast再采集，不宜拿早期固定selected-partner协议推断所有后续请求都共享同一组partners。
来源：`scripts/experiments/c2_local_inputs.py:32至42、64至111、132至215`；`scripts/experiments/c2_information_inputs.py:72至87，group_responses`；`scripts/experiments/c2_local_agents.py:65至184`；`scripts/experiments/c2_retune.py:78至145`。

### 15.2 不能合并的实验入口

早期local legacy曾用wrapper手写tune_legacy；direct兼容路径用单个task rewriter加TaskSearch/GpuAdapter，不经过legacy analyst→rewriter→parameterizer，也绕过native两witness准入。
direct仍逐trial调用正式worker检查数值，绕过两witness不等于没有correctness；反过来也不能因有逐trial验证而声称准入/缓存制度相同。
当前native retune可选扩域，默认cap0；`run_analysis=False`不必然禁止扩域，扩域仍是新space/new B。它不自动执行主Loop A repair campaign。
短闭环明确只迭代两轮；保留parent时带失败反馈，晋级后换成child的selected materialized source、trial、space和本space历史，并清旧失败假设。
helper pilot两臂都有bash/read/write、formal helper、general guidance、intent、推荐配置与合资格cap1扩域；C2增加targeted planning/acquisition，G0是可自行profiling的强对照。
所以pilot不是“有工具对无工具”，也不能把整个方法包的结果单独归因于helper或某一个新参数；允许使用与实际采用需分别报告。
历史report中的wall_text、Runtime在retune前关闭、retune绝不扩域等描述与今日源码已有差别，应按各自运行版本写，不能用当前功能补写历史。
来源：`scripts/experiments/c2_local_execution.py:54至73、104至128`；`scripts/experiments/c2_local_agents.py:52至56、151至189`；`scripts/experiments/c2_closed_loop.py:37至103、121至179`；`scripts/experiments/c2_information.py:93至131`；`docs/result-c2-helper-pilot.md:24至37、106至147`。

### 15.3 native promotion、full promotion与主final不同

native选择比较child tuning-selected latency与预声明historical parent robust latency，严格更低才选child；fresh finals不参与该排名。选了child但final失败，要报告失败，不偷偷重选。
后续full-promotion先固定native winner，再给parent/child各三full blocks，要求complete、100 samples、正finite真实median。
范围重叠/相触或quick/full严格方向冲突时最多加两对confirmation；无冲突比较三块median。第一对确认后，parent在新增块及整体都不差可早停，否则做第二对；最终child须两对逐一更快且整体median更低。
full-promotion内部没有2%门槛；pilot外层描述性的2%pair gate不是其接受公式。冻结选择后的heldout也不应反过来选赢家。
主框架则只有winner的一次full复评，不是三块比较/confirmation。fixed-parent保留父artifact也不是主框架自动fallback到reference baseline。
来源：`scripts/experiments/c2_promotion_full.py:38至78`；`scripts/experiments/c2_information_promotion.py:18至43、83至100`；`scripts/experiments/c2_information.py:141至159`；`control/orchestrator.py:3590至3692`。

## 16. 一个贯穿对象与反馈的概念例子

以下数字和结构均为说明，不是新实验或历史结果。设reference完成矩阵运算与后处理，generator提出路线A与B，各注册一个family。
A参数化出BLOCK/WARPS等离散域，default/minimal均通过，发布A-space-v1；B独立完成相同准入。两个tile值不等于两个family。
A在B40内得到10ms有效trial T，保存其实际源码。若普通边际统计向大BLOCK改善，只能说混合partners下有趋势，不能直接称资源导数。
若启用v4.1，编译probe在固定partners下找到两个fit端与拒绝边，C4得到F=[10,10]、N=[9,9]ms，方向gate通过；是否还有E1由Q与B剩余决定。
analyst可据此提出“换数据布局或分块方式，降低墙侧成本”的假设，rewriter输出A2、A3。两者仍继承A-family，即使A3换backend。
A2/A3各自重新参数化、见证和调参，不能继承T的分数。假设A2实测8.8ms，A3更慢或无效，family incumbent更新到A2；账本各自对共同parent对账。
若只扩A2的choices，仍是A2但space版本变；旧winner只有字节匹配才复用。8.8ms也可能来自旧域组合的再发现，不自动证明扩域有用。
下一轮parent为A2的selected trial，不是原A seed。两次history改善都低于2%或达到5轮时按规则冻结；资源binding本身不停止。
最终在所有family中选最小已测latency，导出其trial源码，做一次full。若full失败，即使best_kernel.py已存在，也不能作为有效性能交付。
此例的作用是解释身份与流程，没有证明资源下降造成加速，也没有预测具体task的收益。
对应来源：第6至15节各阶段；`control/orchestrator.py:1009至1028、2424至2580、3333至3692`。

## 17. 时间、最终交付、记录与恢复

### 17.1 三类预算和实际墙钟边界

B40限制每space admitted asks；witness、repair、compile probes、helper、overhead、agent与final不是同一计数单位，不能按40次GPU执行估算总成本。
12h计时从Orchestrator构造时开始，含其后的校准、baseline、agent/GPU等待，不含更早Runtime启动。resume构造新Orchestrator，时钟重置，不累计前进程elapsed。
外循环、family开始及batch后续candidate检查墙钟；`_tune`没有逐trial全局时间检查，已启动candidate可超过剩余时限。
`_pipeline_batch`实际以`if i and elapsed>=budget`检查，因此每个batch第0项都豁免，不仅seed batch；不能引用注释缩小这个例外。
正常退出先drain预取，再finalize；cancel不能强停正在运行的agent。summary.elapsed_hours在final复评前取得，不含随后final评估与report生成。
worker job_wall_s从GPU锁等待前开始，含cooldown；communicate timeout则在锁与cooldown后开始，kill/reap还可能耗时，二者不是同一时长。
当前request_timeout_s=3600；module timeout_s字段无消费者，不是第二个有效独立时限。idle/cgroup watchdog依环境与活动信号，不能说每次调用必然在3600秒准点退出。
来源：`control/orchestrator.py:545至550、612至626、1338至1408、3459至3488、3598至3618`；`gpu/worker_client.py:259至355`；`config.py:101至196`；`agents/runtime.py:627至702`。

### 17.2 final best与有效性检查

`_finalize`遍历所有family.best，取已测latency最小者，不要求family已converged，也不用资源节省打破平局；严格小于更新，平局保留先遇者。
无任何best返回best=None，不自动生成torch baseline作为BestKernel。有winner优先读其selected trial artifact，否则当前source+params物化，写`report/best_kernel.py`。
之后fresh worker process执行一次full correctness/perf复评，默认5/100；主路径不是三个full blocks。
summary保存candidate/family/trial/space IDs、params、tuned_ms、final_reeval_ok、full mean/median、precision与plausibility flag。
headline speedup为有效baseline.mean/final.mean；median ratio仅双方有真实正median才算，不把baseline mean/candidate median标成median speedup。
final无效**不自动fallback第二名或reference**，文件已在复评前导出；正常CLI也不因final_reeval_ok=False自动返回失败exit code。
使用者必须检查final_reeval_ok、quality协议与对应artifact，不只看文件存在、正speedup字段或CLI退出0。源码生成speedup的分支仅检查latency正值，不额外把reeval.ok合入条件。
中断路径可写RUN_INTERRUPTED与已有report，但不会补一次finalize；provisional report不等于full-verified交付。
来源：`control/orchestrator.py:3590至3692，_finalize`；`evaluation/benchmark.py:159至163`；`cli.py:376至405`；`reporting/report.py:39至107、607至661`。

### 17.3 records与identity对照表

| 记录/文件 | 谁在何时写 | 连接键与不能混淆的内容 |
|---|---|---|
| TaskSpec / manifest.json | adapter/CLI创建run | reference path/hash、resolved config，不保证外部文件以后不变 |
| task/ref.py、semantics/device等inputs | agent调用seed阶段 | 本次prompt上下文，不是所有历史调用同样输入 |
| state.json | started snapshot | task恢复入口，不是完整Optuna/family最终快照 |
| events.jsonl | RunStore各阶段append | seq/ts/type/payload；事件时间不是GPU时间 |
| CANDIDATE_REGISTERED / source.py | 注册，后续参数化/repair可覆盖文件 | candidate/family/parents与注册signature；后续字节可能不同 |
| SPACE_PUBLISHED / SPACE_EXPANDED | validator/控制器 | space/version/domains/constraints，扩域不是新family |
| trials/<tid>.py / TRIAL_DONE | fresh或cache trial | candidate/space/params、latency/profile/failure、reused标志 |
| QUICKTEST_DONE / SPACE_REJECTED / REPAIR_PRODUCED | witness与修复 | witness摘要不等于所有TrialRecord字段，repair不等于child |
| TUNING_DONE / STATS_DONE | study结束及分析 | sampler snapshot与统计范围可能不同，须按space/source归属看 |
| DIMENSION_STATE / BOTTLENECK_CLASSIFIED | harness分析 | derived诊断，不是LLM实测或通用硬件证据 |
| walls / scan blocks / tokens | 可选probe/scanner | axis、typed partners、F/N、fresh角色、admission与消费，不算免费预算 |
| CONVERSION_RATES | harness分析 | spill OLS只日志，未投prompt |
| BOTTLENECK_REPORTED / REWRITE_PRODUCED | analyst与rewriter | report是假设；hypothesis/expectations为测量前声明 |
| family history / HYPOTHESES_FAILED / ledger | round结算 | 每轮历史、逐child对账与未测，不能用sibling winner代填 |
| jobs/*.json / *.out.json | dispatch/worker | request/raw output；超时可能无out，host wall stamp另计 |
| sandboxes/<call_id> / agent events | 每次invoke | named call、attempt、provider request不相等 |
| report/best_kernel.py / RUN_FINISHED | finalize | selected measured source与full validity才构成交付依据 |
| report/report.md / trials.csv | reporter | 聚合视图，不能补造缺失final或raw samples |

来源：`store/run_store.py:14至33、53至95、114至172`；`models/core.py:328至406`；`agents/base.py:101至122、280至299`；`gpu/worker_client.py:268至325`；`reporting/report.py:562至645、1085至1102`。

### 17.4 resume与成本

恢复重放events重建candidate/space/trials/baselines、family.best、轮数、失败假设和ledger；partial pipeline用既有space与cache继续。
这不是保存/恢复完整持久Optuna study内部状态，而是新建tuner并复用证据，不能保证逐采样一致checkpoint resume。恢复也不保证reference、依赖、device或provider未变。
统计分析可将candidate跨space累计trials与当前space一起分析，不能默认每行曲线都是最新space/source的native subset。
usage来自provider返回metadata，缺省tokens={}、cost=0可能是未报告，不是免费；当前report total_cost主要汇总AGENT_CALL_FINISHED，不覆盖全部失败/外部自测/基础设施。
成本应分别报告calls、attempts、submissions、admitted trials、fresh jobs、samples、blocks及wall time；helper已包含在agent elapsed时不要重复相加。
来源：`store/run_store.py:139至172，replay`；`control/orchestrator.py:966至1106、1315至1332、2027至2031、2826至2993`；`agents/runtime.py:27至38、762至763`；`reporting/report.py:588至590、647至649`。

## 18. 论文写法、所需实验材料与主张边界

Methods可按“任务与质量协议，程序/空间表示，A/B/C/D搜索，资源证据与所选反馈变体，预算/停止，最终选择与记录”六段组织，第2节伪算法可作为基础。
必须另列实际config、硬件/工具链、reference与candidate身份、sampler/evaluation seed、correctness mode和rescue、fresh/cache计数、final/promotion规则。
如果写C2 G0/G1/G2实验，重点是冻结父状态和相同acquisition的mask，不要用完整多family图暗示每臂都从零搜索；targeted/helper变体另列方法包和对照工具能力。

| 可由实现说明的Methods主张 | 仍需运行证据/不能直接推出 |
|---|---|
| 在访问到的有效结构与配置中维护best observed | 全局最优、普适收敛或稳定端到端胜率 |
| 条件端点响应可交给结构rewriter | agent实际阅读/采用、具体资源导致速度改善 |
| native B40与可选Q内scan有明确计数 | 全campaign只有40次GPU调用或严格等总成本 |
| 用compiled资源、逻辑cost和估算occupancy诊断 | 实测DRAM利用率、achieved occupancy或不可逾越物理墙 |
| C4用fresh端点与经验noise gate | 95%CI、因果梯度或越墙后的收益预测 |
| 每child事前方向声明、事后对账可记录 | 声明正确获额外接受bonus或ledger已改善性能 |
| quick/full输出检查与独立复评 | 任意shape/state/precision形式等价，或export文件必有效 |
| 固定父信息实验比较不同fresh信息 | 完整12h框架端到端优势、helper单因素效应 |

历史数据入口为 [论文数据资产总目录](paper-data-inventory.md)，本文不复制其中的性能表，也不新增任何正负性能结论。
该目录区分固定父信息、短闭环、native-retune、helper与配置诊断等C2材料，并列原始artifact、协议和限制；引用时只取与本文C2范围及论文主张相符的条目。
性能表应同时给有效性、精度/rescue、selected source/params、baseline同口径统计、失败与censored数；不能把同一结构复测、多sample或resume当独立生成重复。
机制论证还需从enqueue到fresh测量、gate、token、prompt投递、改写采用、child资源变化、最终latency逐步对上证据；中间缺一步就把相应归因降为假设。
来源：`docs/paper-data-inventory.md:1至44、80至90`；本文第7、11、13、15、17节的实现引用。

## 附录 A. 源码、历史与轮次身份的关键限制

这些限制影响复现和归因，不表示本次观察到运行失败，也不是要求修复源码。
1. reference加载时有SHA，但主流程后续仍从外部path读source；hash记录不等于文件被不可变固定。
2. candidate source_sha/signature在注册时建立，parameterizer/repair覆盖source后未同步重算；validator space.source_sha又来自candidate，不能单靠它证明最终字节。
3. `_selected_trial`按candidate、complete、latency及robust_ms==best_ms取首个匹配，不额外用params匹配；同分trial的artifact选择不是更强联合主键查找。
4. `_do_rewrite`返回bool(registered)，不是至少一个child测量成功；注册但witness全失败或因墙钟未调参，仍可能被上层算evaluated，影响history解释。
5. fresh run的FAMILY_SEEDED初始化在Loop C前一次；后来novelty新建family未在该函数即时补seed history，不能保证每个新family当场有h[0]。
6. 扩域可能改body、未严格验证superset，统计可混跨space/source历史；selected measured artifact比最新source+旧params更可靠。
7. final artifact读取失败没有rewriter上下文同样的OSError fallback；复评失败也没有自动second-best fallback。报告/退出码不是有效性交付证明。
来源：`tasks/kernelbench.py:23至34`；`control/orchestrator.py:677至679、1069至1074、1200至1212、2027至2031、2536至2572、3453至3457、3564至3586、3608至3618`；`paramspace/validation.py:154至161`。

## 附录 B. conditional身份、缓存与未接线项

typed config key区分int/float/str/bool，endpoint身份需带完整partners；但v4.1 execution_identity实际仅candidate_id|backend|protocol，主动忽略space_id，也未含source hash/workload/toolchain/device。
tokens跨同candidate spaces保留，E沿用token旧partners并过新space guard，没有再与live incumbent partners匹配；source变更后不能自动断言旧contrast仍可迁移。
diagnostic bridge实际共享evaluator的production compile事实cache。observe虽不做scan latency/prompt干预，后续production guard仍可能受新compiler facts影响，所以不是绝对只日志、零干预对照。
E成功未把latency写进block.lat.by_role的E_new路径，brief读取该值的行可能缺失；不能假定每次E执行都在prompt里有完整速度反馈。
生产mint只有默认E1；E4扣4槽但未生成四点协议；max_probe_attempts配置未接常量账本；S1 declare_infeasible_out_of_space只见声明无消费者。
OLS for_prompt存在但主路径未用；`control/targeted_probes.py`的request-first helper在本次src检索无生产调用者，不等于scripts中C2 targeted acquisition不存在。
另外，statics_from_worker无生产消费者，resource_near_limit配置阈值未替代classifier .80常量，precision同义字符串也可能产生mismatch提示；不能由类/字段名营销未交付能力。
来源：`conditional/identity.py:17至50`；`conditional/bridge.py:73至88、169至188、216至245、384至392`；`conditional/scanner.py:293至319、395至409`；`config.py:394至428、731至761`；`control/targeted_probes.py:60至143`；`evaluation/conversion_rate.py:116至136`。

## 附录 C. 本次核对的冲突处理与紧凑源码索引

本次完整阅读三份内部事实记录共1135行，截断尾部已补读；正文引用直接指向实现，读者不必依赖这些内部材料。
补核strict发现笔记常见公式把relative项写成reference，但实际allclose(reference,candidate)使用第二实参candidate；第13节按当前调用修正，并限定为有限元素公式。
当前config确认request_timeout_s=3600、环境过滤默认False，不用旧报告1500秒替代；当前information在Runtime内retune，并支持可选扩域，不沿用旧文档“先关闭Runtime/绝不扩域”。
旧hard/soft slope单调性、probe独立cache、E strata rotation、仅seed第0项预算豁免等注释与可执行语句不完全相符，正文按实现表达，详细边界集中在第10、11、17节和附录B。
CUDA-event warmup、discard与L2 thrashing额外核对了相邻KernelBench timing.py；未验证其他timing后端，不把该行为推广到全部可选计时器。

| 主题 | 主要文件/函数定位 |
|---|---|
| task/CLI/wiring | `tasks/kernelbench.py:10至42`；`cli.py:376至443`；`wiring.py:115至170` |
| 对象与身份 | `models/core.py:49至156、328至434`；`control/families.py:30至253` |
| 参数空间准入 | `paramspace/materializer.py:65至130`；`paramspace/guard.py:26至136`；`paramspace/validation.py:98至371` |
| TPE与统计 | `tuning/tpe.py:24至275`；`tuning/objective.py:11至52`；`tuning/stats.py:81至175` |
| A/B/扩域 | `control/orchestrator.py:994至1459、2424至2727` |
| C/D/停止/final | `control/orchestrator.py:2826至3692`；`control/convergence.py:11至65` |
| 资源与旧墙 | `evaluation/profilerx.py:54至118`；`evaluation/statics.py:328至381`；`evaluation/wall_attribution.py:240至398`；`evaluation/soft_wall.py:230至358` |
| 条件扫描与响应 | `conditional/probe.py:95至525`；`conditional/scanner.py:119至428`；`conditional/gates.py:74至141`；`conditional/task_response.py:15至74` |
| 模型角色与账本 | `agents/modules.py:604至1661`；`evaluation/reconcile.py:105至274`；`agents/self_test.py:59至169` |
| quality与timing | `gpu/worker_main.py:1987至2465`；`../KernelBench/src/kernelbench/eval.py:784至815`；`../KernelBench/src/kernelbench/timing.py:126至281` |
| C2 variants | `scripts/experiments/c2_information.py:93至159`；`scripts/experiments/c2_retune.py:78至203`；`scripts/experiments/c2_closed_loop.py:121至179`；`scripts/experiments/c2_promotion_full.py:38至78` |
| 存储与报告 | `store/run_store.py:53至172`；`reporting/report.py:39至107、562至661`；[数据目录](paper-data-inventory.md) |

本文是当前实现的技术参考，不是历史部署一致性证明。论文的性能、采用率、资源因果和泛化结论仍须回到相应固定版本、实际配置及有效测量材料。
