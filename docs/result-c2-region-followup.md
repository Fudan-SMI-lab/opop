# C2 region follow-up：补测、自测协议差异与已实现的方法改进

**T7分析完成；四个正式helper集成job独立核验PASS，最终F1核验回执单独交付。** 用户要求的数据先行已完成，随后T4推荐配置交接、T5扩域证据/定向探测及自愿正式自测helper已实现、CPU验证并发布，helper真实worker路线也已验证。**已有补测四个pair仍为2通过、2失败，不满足3/4描述性门槛，尚未建立稳定C2收益。** 这些数据先于新方法修改，不是新代码的效果测试；也不构成统计显著失败或C2普遍无效。

原campaign的deadline `1789750472.9209218`、两个censored heldout记录和 **INCONCLUSIVE** 结论保持不变。新补测是另行授权、另目录、截止后的固定产物批次；不是把旧实验追认成四小时内完成，也不是重新生成或再调优候选。本文只归档当前follow-up报告，不改旧报告/raw，不发起新实验。

## 1. 批次身份、rep含义与冻结产物

- 原始生成/调优来源仍为已发布实验源码 [96c8bec3](https://github.com/Fudan-SMI-lab/opop/commit/96c8bec3f91071ab668550ff2510f3cd5a545f18)。本批0模型调用、0调优、0扩域、0晋级/重选、0替换；只测原rep1已完成的parent/G0/C2。
- **rep0和rep1是同一任务原始P1上的独立fresh生成重复**，sampler分别0/2；不是旧两轮闭环中“第二轮接第一轮winner”。provider的模型随机性并未因此完全受控。补测evaluation seed仍0。
- 原rep0：task43在HOST-A生成（G0 GPU0、C2 GPU1）并在A GPU0 heldout；task21在HOST-B同样配对、B GPU0 heldout。
- 原rep1：task21在HOST-A生成（C2 GPU0、G0 GPU1），task43在HOST-B生成（C2 GPU0、G0 GPU1）。本次分别在 **A GPU0、B GPU0** 对冻结三角色独立fresh测量，不跨卡直接比raw latency。
- 新计划19:05:44.6769983 UTC冻结，SHA256 `ecb36ce868e1b26ca28998e5b05871f6b5740321428550d8e37623e84b10b336`。每任务顺序parent,G0,C2 / C2,G0,parent / G0,C2,parent，各角色3块，共18 full。

| 本次角色 | 原始冻结来源 | selected space | source SHA256 |
|---|---|---|---|
| task21 parent | 原P1 | sp-e3acfbaf | e72351fde23fc8a11fe06fe9ecc5de9e8de0da52065b591edd0dbe999196be29 |
| task21 G0 | wave2/A1 incumbent | sp-83869f74 | 86c024ebdb4b7f3eaa6588c141dd0d2aa9586ffde0a80a51330404a9b3e2c4f9 |
| task21 C2 | wave2/A0 incumbent | sp-6af980fc | 07e436d072899e7d845941e8ee81de7171056278b4a81c0972a0c3588d6b3ac5 |
| task43 parent | 原P1 | sp-2001d847 | f97300db4bbecaa80392bc3dfcbf03355513f2d82571dbb6c360685682cef236 |
| task43 G0 | wave2/B1 incumbent | sp-560b108e | 134625bc1e853eddc0554e9c04019059f51c8b4b081dfb4b3c3b5b4bdd9bee5e |
| task43 C2 | wave2/B0 incumbent | sp-2710c4b9 | 7d89af3b9d4a809858c4f734598a2ff5644371dcd328c18abd78b301051870f0 |

四个G0/C2源码及copied opportunity JSON与原rep1产物字节一致，六组PARAMS与实际worker材料一致；helper隔离/哈希和两个reference也核验通过。这里没有为了新结果重新选择config。

## 2. 新rep1：全部18个block与质量

单位ms。每full block为100性能样本、5 correctness，seed0、fp32 reference、原dual-witness-relaxed/FP64契约，阈值未改。真实worker扫描确认 **18 eval full＋6 static_check**，不是24次full测量。

| Task | Role | Block1 | Block2 | Block3 | Median |
|---|---|---:|---:|---:|---:|
| task21 | parent | 8.494079590 | 8.500224113 | 8.499119759 | 8.499119759 |
| task21 | G0 | 6.178864002 | 6.174719810 | 6.180351973 | 6.178864002 |
| task21 | C2 | 5.155839920 | 5.152768135 | 5.169151783 | 5.155839920 |
| task43 | parent | 3.434495926 | 3.447296023 | 3.446784019 | 3.446784019 |
| task43 | G0 | 2.945024014 | 2.951168060 | 2.953216076 | 2.951168060 |
| task43 | C2 | 2.967583895 | 2.969599962 | 2.974720001 | 2.969599962 |

18/18有效，共1800性能样本、90/90 correctness passes。**task21所有block rescue=0；task43所有block rescue=5，共45次**，task43依赖授权FP64 witness，不宣称仅relaxed witness或严格数学等价通过。

独立从全部原始100样本重算median；sample导出精度有限，最大误差 **0.0000041134643549156635ms**，在容差 **0.000005000001ms** 内；reported worker median与结果记录一致。112条新manifest hash通过，计划与batch在两host字节相同，逐次role/index顺序匹配；未重新审计旧cohort数千文件，旧整体未改由operator回执背书，两个旧censored结果另按原hash独立复核。

### 2%与三块范围规则

沿用规则作**描述性比较**：`M=median(三块median)`，`gain=100×(1−M_C2/M_G0)`；需gain≥2%且`max(C2)<min(G0)`。每pair用本次共同fresh parent归一化`R=M_arm/M_parent`，不跨任务平均raw毫秒。

| 新rep1任务 | R_G0 | R_C2 | C2相对G0改善 | 范围检查 | 结果 |
|---|---:|---:|---:|---|---|
| task21 | 0.7270004633093031 | 0.6066322238633353 | **+16.556831187981924%** | max C2=5.169151782989502 < min G0=6.17471981048584 | PASS |
| task43 | 0.8562091629855999 | 0.8615567280861819 | **−0.6245629376278927%** | min C2=2.9675838947296143 > max G0=2.953216075897217，反向分离 | FAIL |

生产`compare_blocks`与独立NumPy算术完全一致。范围分离不是置信区间；task43的小幅负向不被说成显著普遍损害。

## 3. 原rep0＋补测rep1：跨批次描述性合表

rep0来自原campaign实际heldout，rep1来自本次新batch。当前四组固定产物的结果都已在各自同卡条件下被观察，**但不是原campaign在同一deadline内获得的四组数据**。

| Task / rep | 测量批次 | Parent median ms | G0 median ms | C2 median ms | C2改善 | 规则结果 |
|---|---|---:|---:|---:|---:|---|
| task43 / 0 | 原wave1 heldout | 3.492415905 | 3.098608017 | 2.992127895 | +3.436386% | PASS |
| task21 / 0 | 原wave1 heldout | 8.362495899 | 6.047263861 | 8.360960007 | −38.260215% | FAIL；C2为保留父 |
| task21 / 1 | 新post-deadline batch | 8.499119759 | 6.178864002 | 5.155839920 | +16.556831% | PASS |
| task43 / 1 | 新post-deadline batch | 3.446784019 | 2.951168060 | 2.969599962 | −0.624563% | FAIL |

**2通过、2失败，不是3/4；没有稳定一致的C2效果。** 这不是原global gate从INCONCLUSIVE改为FAIL，也不构成统计显著性检验。旧wave2 A0/B0原记录仍为censored、parent/G0/C2数组为空，旧报告和raw没有覆盖。本稿不计算跨任务raw平均，也不另造几何平均作为注册主指标。

task21 rep0的quadplane布局child虽有效但formal full很慢，原父被保留；新补测没有改变这一事实。task21 rep1的grid-mapping/SK1候选得到较好的完整任务结果，但同时包含结构、伙伴配置与生成轨迹差异，**不能归因于grid-order单独作用**。task43双方都能生成head拆分/物化改变，两个rep的相对次序并不一致。两任务、各一次额外固定产物比较不能证明泛化或C2普遍无效，也不验证prompt因果性。

## 4. Agent到底有没有主动测试：真实做了，但协议和覆盖不等价

依据最新248行[自测审计](../.omo/notepads/v5-c2-region-followup/agent-self-tests.md)及[脱敏证据JSON](../results/c2-agent-self-tests/evidence.json)。本报告读取这些已完成审计事实，不重做23会话检索、不重放命令。必须区分“工具可用→调用过→实际forward返回→某标准通过→同协议性能可比较”，shell exit0不够。

| 类别 | 实际候选/runtime测试 | 已恢复会话 / bash调用 | 覆盖与限制 |
|---|---:|---|---|
| rewriter | **8/8** | 8会话 / 78次 | 均至少有一次完整reference shape forward；含失败、修复、比较、profile和timing，不等于每个中间版本都经过正式gate |
| initial parameterizer | **6/8** | 8会话 / 35次 | 常用缩小shape或不同参数；另外2个仅AST/bytecode/import检查，没有forward证据 |
| expansion parameterizer | **2/7** | 7会话 / 24次 | 有限小shape/finiteness或compiler-floor探测；其余5个为静态/复制检查 |
| 合计 | 不能把类别比例当benchmark数 | **23会话 / 137次bash** | 137不是137次benchmark；没有用缺失store推断“未测试” |

rewriter完整shape为task21 `(10,112,224,224)`、task43 `(128,512,768)`。后续parameterizer经常只测小batch/小shape；四个有body变化的initial parameterizer都运行过post-edit候选，但仍未覆盖所有最终selected配置。Import、`py_compile`和`finite=True`都不是正式正确性/性能结论。

实际反例包括：一次脚本打印cosine0.011315仍输出`ok`；一次shell pipeline退出0但打印Triton CompilationError；quadplane坏中间版也曾exit0但cosine仅0.526883。后来有真实修复与成功输出，所以既不能因exit0误报通过，也不能把失败中间版当作最终返回产物。

### 4.1 Quadplane：真实profile、parity修复与返回源码对应

固定案例仍为task21/C2/rep0/wave1 B1，不换好案例。真实工具输出记录parent SK1约17.029ms，其中DW **11.1208ms、65.3%**；修复后的quadplane DW约1.8877ms、expand约5.3280ms。它们是私有profiler协议内的实测，不是编造的“瓶颈故事”，也不是正式harness对每个kernel的独立测量。

坏中间版之后的两处parity-address修复被历史write/edit记录追溯；成功validation与timer用的SHA256为 **`c61900bbda0c5474194c611b3bb027957fb1eb09ac19d233306b6e3f900dc863`**，匹配返回rewrite、初始/扩域parameterizer模板。修复没有在handoff中撤回；不能用“agent没测试”或“返回了修复前源码”解释已有证据。

### 4.2 关键新增证据：同源码、同全部默认参数，仍存在私有/正式反转

这不是仅将私有default与TPE另选的配置相比。审计找到了正式initial **default witness `tr-08e7ee2c`**：实际执行源码也是上述`c619…dc863`，全部18个参数与成功私有child timer一致，delta=`{}`：

```json
{"BLOCK_M":32,"BLOCK_N":128,"BLOCK_K":16,"SPLIT_M":16,"SPLIT_K":1,"STATS_BLOCK_M":128,"STATS_BLOCK_C":16,"DW_BLOCK_H":16,"DW_BLOCK_W":128,"EW_BLOCK":1024,"FIN_BLOCK_C":16,"COMPUTE_DTYPE":"fp16","DOT_MODE":"plain","GEMM_WARPS":4,"GEMM_STAGES":2,"STATS_WARPS":2,"DW_WARPS":4,"EW_WARPS":4}
```

| 固定产物 | 私有输出 | 正式已有输出 | 已控制 / 尚未控制 |
|---|---:|---:|---|
| quadplane child，c619…，同18默认参数 | **7.854ms**，30-forward循环平均 | **16.911423683166504ms**，default **quick 20样本/3 correctness**，3/3通过、0 rescue | 源码和params相同；runtime/timer/状态等仍不同。不是100/5 full |
| 原parent，e723…，同18参数（相对子仅SK4） | **17.464ms** | **8.4618239402771ms**，full100/5、5/5通过、0 rescue；另两块8.459343910/8.466943741ms | 私有parent源码字节也与正式materialized parent一致，不是偷换父配置 |

Parent完整SHA256为`e72351fde23fc8a11fe06fe9ecc5de9e8de0da52065b591edd0dbe999196be29`。child witness raw在旧机会`wave1/B1/information/retune/jobs/cand-48089a01-wit-default-eval-a6a7c793{,.out}.json`；对应`candidates/cand-48089a01/witness_default.py`。Parent证据见同机会`parent/jobs/final-reeval-eval-20b697c7.json`及其实际materialized文件。

**因此可排除“TPE选中参数变化”作为反转的唯一解释：差异在TPE最终选点前、相同default处已经存在。** 后来的selected child确有M/N、stage、EW/FIN等live参数变化，full约15.323ms；这是另一层差异，不能抹掉同配置证据。profile汇总的`num_stages=3`也不能替换源码中`PARAMS.GEMM_STAGES=2`。

### 4.3 仍未隔离的上下文差异

| 维度 | 私有B1 | 正式路径及解释边界 |
|---|---|---|
| Python/toolchain | shell默认Miniconda `python3`；历史输出Torch2.8.0+cu128、Triton3.4.0 | 显式配置worker解释器；审计时METADATA为Torch2.13.0+cu129、Triton3.7.1。当前metadata不是不可变历史package快照，不能单独确立compiler致因 |
| timer/cache | 5 warmups，单event pair包30次连续forward、平均值，脚本无L2 clear | formal3 warmups、逐forward同步/L2清理、quick20或full100样本median；还包括allocator/worker排他上下文差异 |
| 输入/质量/状态 | 私有seed0模型、seed1输入；fraction/cosine和单输入检查，未用完整FP64 gate | formal seed0、多trial dual-witness-relaxed与FP64 rescue；正确性阶段后模型/BN状态也未隔离 |
| 已相同条件 | full shape、训练模式、reference、Triton backend、fp32 I/O与fp16/plain compute | **B1没有eval/train BN错配证据**；整模型timer不应误说成只测DW。reference字节也相同 |

同配置证据加强了**协议/环境迁移失败**这一诊断，但没有将compiler、cache、warmup、seed或state的单独因果贡献量化。私有加速是真实的私有结果，formal结果也是真实的正式结果；不是假数据，不应只靠一个推测消解反转。

另一个覆盖边界：task21 rep1私有FP16中间存储fraction约0.951–0.957而被放弃，但没有走正式FP64 fallback，故不能称“已证明完整gate拒绝该variant”。最终候选用fp16 **compute**、fp32 **storage**，二者不要混写。task43 C2的stage3机会亦不能由fresh stages1/4响应、其他G0小shape S3测试或注释替代实际目标区域验证。

审计副作用脚注：最初CLI路径发现意外初始化了一个非实际campaign store、无session的staged DB及WAL/SHM（含schema/migration，不是空字节文件），故不能称审计字面意义“零文件系统写入”。实际campaign DB以read-only方式读取，无既有campaign记录被修改。后续无法证明创建时身份/清理所有权，按only-if-proven-safe条件保留这些文件，未删除；本次报告更新没有重复CLI发现或清理。

## 5. 数据先交付、先归档，再实现与发布

已核验的先后关系是：固定产物补测 **19:09:54.550283–19:13:44.904437 UTC** → T2数据/分析先交付 → 方法范围在产品编辑前归档 → T4、T5及helper实现/CPU集成 → 测试后发布 → 四job部署集成。没有为缺乏回执的步骤杜撰精确开始/结束时间。

- 预修改范围：[plan-c2-region-followup.md](plan-c2-region-followup.md)，归档 [0bd0c3d842ea5d2a6d8c8cfa866392732cf97a6a](https://github.com/Fudan-SMI-lab/opop/commit/0bd0c3d842ea5d2a6d8c8cfa866392732cf97a6a)。它不是把已经看到的数据重写成新注册主实验。
- 实现已发布：[8eae0eb4828826ab99c9948ea8160e66e34c3498](https://github.com/Fudan-SMI-lab/opop/commit/8eae0eb4828826ab99c9948ea8160e66e34c3498)，[范围归档到实现的diff](https://github.com/Fudan-SMI-lab/opop/compare/0bd0c3d842ea5d2a6d8c8cfa866392732cf97a6a...8eae0eb4828826ab99c9948ea8160e66e34c3498)。**旧96c8bec3产物的数据不能记成8eae0eb效果。**
- 父级最终整合验证为 **232 tests passed / 145.47s**，关键LSP clean；这是读取的回执证据，不是本次文档编辑重跑。各worker的382/407/282/85等suite互有重叠，不相加为独立测试数。

## 6. T4：推荐联合配置已进入实际消费者，但建议不等于测量

已实现可选`recommended_configs: list[ParamSet]`，默认空、最多2项；旧JSON/调用兼容。Rewriter可以给本candidate的1–2个具体联合区域配置；parameterizer在既有调用中按自己发出的source/key/default/伙伴解析，沿LegacyProposal→Child→RetuneInputs→CandidateRun到初始与expanded空间。

使用时在`RewriteCandidate`和`ParameterizationResult`的`recommended_configs`字段给出最多两个完整`{"values": {...}}`配置；省略等价于`[]`。具体key必须属于该candidate，不是只写轴名或从另一臂复制配置。是否正式覆盖仍看下述TRIAL_DONE，不看建议数量。

- 只承认parameterizer为当前源码解析的输出；mapping未知清空未解析建议并记`mapping_unknown`，不从父defaults或历史winner猜填。参数source不符、缺/多key、域外、类型或constraint不合法均有明确skip原因；**语义无效建议不使原本有效candidate整体硬拒绝**。结构格式错误仍走既有schema/repair。
- 原default/第二witness、合法兼容prior-best anchors和原复用规则保持。随后追加合法、候选内去重的推荐；重复witness/prior不额外占点，**消耗既有B40，不变成42**。不跨candidate/竞争臂搬运建议，不注入事后赢家。
- private“跑过/很快”的叙述不转成`measured_cache`分数。初始/扩域接受新的resolved输出替换旧推荐，不把不确定映射一直累积。
- `RECOMMENDED_CONFIG_REQUESTED/QUEUED/SKIPPED`记录source/candidate/space/params及原因。**QUEUED≠测到该点**；必须在真实`TRIAL_DONE`核对身份/配置/有效结果，才能宣称推荐区域被测量。prescreen、失败、有限空间耗尽等仍可能影响覆盖。

这实现了“把想实现的联合区域变成有限可测配置”的接线，而不要求新结构在父的旧固定点获胜、不禁止必要body/dtype修复，也不证明未来agent一定提出好配置。

## 7. T5：扩域事实输入与12端点内定向分配均已实现

### 7.1 扩域看到实际best，不再把fallback写成单调证据

现有parameterizer收到candidate-local实际selected TrialRecord、measured source、params/profile、numeric stats、原始complete/failed trial表、reference以及既有domain/constraints/intent/推荐配置。当前可调源与selected measured源明确区分，best可来自旧space；缺文件保持unknown，不用最后space/defaults冒充。

Directive标`statistics_scope="marginal_nonconditional"`、`source_equivalence="not_assumed"`；每值保留median/失败率/观测数、伙伴是否变化和space出处。`winner_anchored_marginal`、`median_edge_fallback`、`marginal_uncertain`只是证据标签；`conditional_trend`和`active_branch_effect`保持unknown，不能从边际表推固定伙伴单调性或因果。

**Native eligibility、effect/headroom/失败边界策略、cap、B40及no-op/retry规则未改变。** 没有硬编码ban某轴或自动过滤inactive-at-best轴；输入让agent依据实际source/shape判断其意义。Stats可能混合不同space/source的原限制被显式披露，而不是凭新标签声称算法已修正全部统计问题。

### 7.2 Opt-in targeted顺序，不加agent调用或probe预算

显式`probe_strategy="targeted"`时：普通analyst一次（source/selected/ordinary history，**没有未来responses**）→合法请求优先的fresh probes→rewriter一次（同一preliminary report＋实际responses）→一次初始parameterization及原native流程。该路线已接到`c2_information`和`c2_opportunity_program`，不只是无人调用的helper。

- 最多6个request；每个包含既有axis两个值及可选伙伴，按source/domain/constraints校验。总计仍 **最多12个唯一fresh endpoint attempts**；配置去重，失败也占预算，不补到12成功。剩余容量才用原有fallback，不新增域迁就请求。
- 未测、拒绝、超预算和中断点保留unknown；nominal参数不造数值slope。preliminary report明确早于response，实际测量事实优先；截止/在途记录不伪装成完整响应。
- 默认`provided`保留原response-first兼容路径。G0保持普通analyst/rewriter、0 probes；targeted C2也没有额外analyst或扩域agent调用。推荐点与定向probe是不同用途、不同既有预算，不混作免费新实验。

## 8. 自愿正式self-test helper：已发布的能力，不等于agent已使用

相关已配置sandbox可收到非秘密`task/self_test.json`、reference/dependency快照和usage说明；使用**绝对configured worker Python**、当前live evaluator seed/quality、真实reference/shape及既有timer。Quick/full复用原CorrectnessEvaluator/Benchmarker、WslGpuWorker和native job目录的设备仲裁/锁，不写第二套质量gate或计时器。

每次请求fresh materialization，记录source/reference hash、完整PARAMS、mode/seed、quality/FP64 rescue、score、raw worker结果与实际调用证据。Helper结果标`agent_self_test`，不自动进入TPE/private cache或B40；缺版本输出时保持unknown，不伪造环境metadata。它不是任意依赖项目的通用打包器。

**bash/Python/read/write和私有探索继续允许；未改全局PATH、权限或新建服务。** Agent可自愿借helper使测试可比；“说明、CPU路线及operator实调通过”不等于历史agent使用了它，更不等于未来模型必然采用它。需要新的实际session/helper调用记录才能声称模型利用率，不能从prompt中有路径推断执行。调用方式与quick/full边界见[正式helper使用指南](T6-selftest-helper.md)，以sandbox生成的configured-worker说明为准。

已解决一个集成问题：optional context曾因cwd-relative reference缺失而在正常agent调用前报错。修复绑定reference基准、核对可用来源并允许seeded reference；若不可用则标`available=false`及原因，**不伪造reference、不硬拒有效candidate**。显式helper调用的reference不符仍无score失败。此修复已包含在最终整合验证，不是尚未解决的阻塞，也不增加性能证据。

## 9. 四个固定full helper集成job：已完成，独立核验PASS

预声明并实际完成恰好4个full：审计过的parent默认配置P与quadplane child默认配置C，**P→C→C→P**，在HOST-B物理GPU0顺序执行，每个100性能样本/5 correctness；无TPE、新candidate、LLM draw、重选、重试或质量放宽。两host部署元数据为已发布8eae0eb且clean，HOST-A只做部署/保全、没有GPU job。这是有限集成验证，不是再跑18次heldout或新8候选campaign。

Plan于21:26:01.978743 UTC冻结，SHA256 `33a97b6ba7c2ee283c2c2294479d1446e72fda4cfa742d766c6bbe08f9786313`。P源码`e723…be29`、C源码`c619…dc863`与已审计原件字节相等；reference为`24e555726c4856120cceaa8dad2dbbca798a56c0a99313c77fbe84c16e37c20b`。P用同一18默认参数SK4，C用第4.2节同18参数SK1，角色间只此参数不同；未换成formal selected配置。CLI省略`--params`提取原defaults，四份fresh物化源/返回参数均核对匹配。

| 固定次序 | 角色 | Full median ms | compiled / correct / formal OK | correctness / rescue | Helper wall s |
|---|---|---|---|---|---|
| 1 | P默认 | 8.365056037902832 | true / true / true | 5/5 / 0 | 33.099368 |
| 2 | C默认 | 16.514127731323242 | true / true / true | 5/5 / 0 | 51.345417 |
| 3 | C默认 | 16.494080543518066 | true / true / true | 5/5 / 0 | 62.389494 |
| 4 | P默认 | 8.366080284118652 | true / true / true | 5/5 / 0 | 60.186813 |

每次actual `commands.jsonl`记录一个static＋一个full：均使用 `/root/autodl-tmp/orch-venv/bin/python` 和部署入口 `/root/autodl-tmp/c2-local-experiment/code/v5/src/kernel_optimizer/gpu/worker_main.py`，不是shell默认Python。角色context及raw job一致为seed0、fp32 I/O、100/5、原dual-witness-relaxed与启用FP64 multipliers2/3；共用jobs目录保留native设备仲裁。没有新timer或质量gate。

独立核验**83条export hashes、4 CLI exits0、4 full＋4 static＝8 worker calls、400原始sample、20/20 correctness、零rescue**。四块worker reported median与helper score精确相等；五位小数sample重算最大差0.000003962097167686807ms，小于0.000005000001ms容差。初始export verifier曾错误要求两种精度bit-exact，后修正仅核验条件，没有改分数、产品源码或重复job。

当前formal-default child约16.50ms、parent约8.366ms，方向与旧formal路径一致；历史私有同默认child7.854ms、parent17.464ms仍是不同上下文的记录。**同源同默认已经显示正式child慢，不能仅归因于TPE换参数。** 这四次没有重放私有runtime/timer、没有cache版本ablation，不识别Torch/Triton/cache各自因果贡献，也不证明新T4/T5改善优化性能。它证明的是**operator**能经helper到达正式worker；没有新的模型session表明agent已自愿使用helper。

## 10. 时间、交付状态与结论边界

| 已完成数据步骤 | 记录 | 口径 |
|---|---|---|
| rep1补测新额度 | 19:09:54.550283–19:39:54.550283 UTC，1800s | 新30min cap，不重置旧deadline |
| 并行measurement span | **230.35415410995483s**，最后19:13:44.904437完成 | task21=230.353991s、task43=167.721289s，不相加，不等于GPU busy |
| setup完成 | A19:05:43.348951；B19:05:44.638818 | 完成时刻，不是setup duration |
| 核验/导出准备/cleanup | A19:19:17.198848；B19:19:18.712638 | 测量后另计；export/download总duration未独立记录 |
| 四job helper集成 | 21:26:57.448069–21:30:25.710013 UTC，**208.2619788646698s** | 新1200s额度内完成；准备/后续export另计，不是GPU busy |

**本用户请求的两批新增正式工作：22 full＝18 rep1补测＋4 helper，10 static＝6＋4；2200性能样本、110/110 correctness。** 其中helper无rescue，补测task43有45次rescue；这不是新的22候选优化实验，也不包含历史agent私有脚本。230.35s与208.26s属于分开的批次，不能当作总体人工周转或GPU busy。

任务上下文中operator约30分钟整体周转包含准备/等待/核验/导出，不是230.35秒测量span或kernel time；也不同于30分钟调度上限。Helper cleanup观察为21:33:40.399435 UTC，部署最终clean元数据约21:34:32。旧结果保全由T6回执及本地原记录核验支撑；软件测试耗时、人力工时、GPU busy与账单不能互相替代，未知项不补零。

| 用户关心的步骤 | 现状 | 尚不能宣称 |
|---|---|---|
| Q1：rep与数据 | 独立生成rep定义已澄清，固定rep1补测完成，跨批次2/4 | 原四小时campaign被补成complete；稳定C2效力 |
| Q2：方法步骤 | T4配置交接、T5事实扩域输入/targeted probes已实现并发布；232项整合测试通过，helper真实worker路线通过 | 新优化campaign有效；模型已实际采用新推荐/targeted/helper |
| Q3：agent实际自测 | 23会话真实记录、同默认配置差异与源码链已审计 | 编造私有结果；已隔离compiler/cache致因 |
| 四job集成 | **完成，独立核验PASS**；仅固定P,C,C,P，无追加 | 新优化收益、私有runtime/cache因果隔离或自主模型采用 |
| 报告/F1 | T7分析完成，本文件随报告提交归档；最终F1核验回执单独交付 | 整个follow-up已获最终F1验收 |

本轮限定的方法接线已得到CPU功能验证、helper已得到四次真实正式路线验证，T7数据分析收尾；最终F1核验单独交付，**不自动追加实验**。本请求没有运行新方法的大型优化campaign；其科学效果仍需未来另行授权、预先固定协议的有界比较。保留自主实验并对齐正式协议，不用工具禁令代替诊断；先于代码取得的2/4仍不能承诺稳定效果。

## 证据入口

- [补测operator记录](../.omo/notepads/v5-c2-region-followup/heldout.md)、[T2先行分析回执](../.omo/evidence/v5-c2-region-followup/T2-combined-analysis.md)、[补测分析JSON](../results/c2-rep1-heldout-followup/analysis.json)保留当时的数据/状态，不因后续实现改写。
- [T4合同](../.omo/evidence/v5-c2-region-followup/T4-contract.md)、[T4消费者](../.omo/evidence/v5-c2-region-followup/T4-region-configs.md)、[T5输入](../.omo/evidence/v5-c2-region-followup/T5-inputs.md)、[扩域事实](../.omo/evidence/v5-c2-region-followup/T5-expansion-evidence.md)、[定向探测](../.omo/evidence/v5-c2-region-followup/T5-targeted-probes.md)。
- [Helper指南](T6-selftest-helper.md)、[reference兼容修复](../.omo/evidence/v5-c2-region-followup/T6-selftest-helper.md)、[T6发布与最终232测试回执](../.omo/evidence/v5-c2-region-followup/T6-publish.md)。较早helper指南中的旧审计路径搜索/集成状态由后续修复回执补充，不当作当前未解决事项。
- [T6四job operator回执](../.omo/evidence/v5-c2-region-followup/T6-integration.md)、[T7独立分析](../.omo/evidence/v5-c2-region-followup/T7-analysis.md)、[集成分析JSON](../results/c2-formal-selftest-integration/analysis.json)。新raw根`results/c2-formal-selftest-integration/`保留plan/context/commands/jobs/result及部署元数据。
- 新heldout raw在`results/c2-rep1-heldout-followup/task{21,43}/`；原rep0在`results/c2-opportunity-driven/heldout/wave1/`，旧rep1空记录仍在`heldout/wave2/`，没有覆盖。[旧campaign报告](result-c2-opportunity-driven-improvements.md)不改写。
- 自测证据与上述raw/.omo链接是脱敏后的**本地材料，不是公开数据集**；未公布完整会话、数据库或provider配置/凭据。范围/源码与本报告的GitHub归档不包含raw或本地核验回执。
