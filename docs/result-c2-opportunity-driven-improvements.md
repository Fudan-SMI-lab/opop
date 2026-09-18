# C2 条件响应驱动结构机会：两波实验结果与未完成的 heldout

日期：2026-09-18。**全局结论：INCONCLUSIVE。** 八个候选机会均完成，但规定的四个独立同卡 heldout pair 只测了两个；wave2 两组为真实截止后的 `censored` 空记录。不能用screen补齐、把缺失记零，或把已完成子集的胜负当作全局判定。

已测wave1：task43 C2相对G0 **改善3.436386%并满足严格范围分离**；task21 C2因有效但慢的child被拒晋级、保留原父，最终 **比G0慢38.260215%**。新接线确实交付了条件响应、selected上下文、机会意图、原生扩域和full晋级；但“推理输入被用上”“实现了结构动作”“搜到新增区域”“获得更优完整任务表现”是四个不同问题。本报告逐层拆开，不给未测的wave2或机制点补结果。

## 1. 协议、公开实现与证据边界

- [获批计划](plan-c2-opportunity-driven-improvements.md)在产品修改前单独归档：[f4a2d405](https://github.com/Fudan-SMI-lab/opop/commit/f4a2d405051d1784880547276220e5227a533f32)。实验统一使用已发布并部署的 [96c8bec3](https://github.com/Fudan-SMI-lab/opop/commit/96c8bec3f91071ab668550ff2510f3cd5a545f18)，[完整实现diff](https://github.com/Fudan-SMI-lab/opop/compare/f4a2d405051d1784880547276220e5227a533f32...96c8bec3f91071ab668550ff2510f3cd5a545f18)。本报告归档科学结果与缺失覆盖；最终核验回执单独交付。
- 两任务×两fresh reps×G0/C2，共8个固定机会。每次从各任务exact original P1重启，wave2不接wave1 winner；rep0/1 sampler=0/2，evaluation/validator=0。无补抽、替代rep、第三波或新实验。
- 两臂同为whole-task、传rewrite intent，初始B40、native expansion cap1、同样full晋级。C2额外得到当次fresh conditional responses及机会推理指令；G0不采。**比较是信息＋guidance package，不能隔离guidance单独作用**。同seed在不同域不意味着相同搜索轨迹，fresh rep不保证LLM随机性受控。
- 主性能只来自incumbent冻结后的全新heldout，parent/G0/C2各三块，固定正/反/轮转顺序。每块100性能样本、5次correctness，原dual-witness-relaxed、FP64 multipliers2/3、pass fraction0.99、cosine0.99985。screen/confirmation属于选择数据，native quick分数只用于描述和冲突检查。
- **本次源码/测量完整性核验PASS，主比较覆盖不完整。** 未复用旧污染core/A作controls，也未重审所有历史运行。新campaign无原primary被helper覆盖的证据；不同层次的parameterizer body变化另见第6节。

### 实际软件改动

| 优先级/类别 | 代表实现 | 实际作用与边界 |
|---|---|---|
| P0机会推理与路由 | `agents/method_prompts.py`、`modules.py` | responses存在时启用条件objective/parameter/resource→限制/耦合→结构动作→联合区域→净收益链；真实compiler wall分开，非一般fusion菜单 |
| P0 selected上下文 | `c2_local_agents.py` | analyst保留可调源＋显式selected params；rewriter接物化父＋相同params，不由文件名推定best |
| P0 intent与native continuation | `orchestrator.py`、retune/信息适配 | 同一机会意图贯穿初始与扩域；caller-owned Runtime覆盖retune；cap1实传；base-best/before/after refs、跨space实测winner归属可追溯 |
| P1 full晋级 | `c2_promotion_full.py`、`c2_information_promotion.py` | 复用parent3/最终child3筛选；重叠/方向冲突才触发至多两对确认；不修改TPE内部排序 |
| 两波与heldout | `c2_opportunity_*` | 固定8机会、同一绝对deadline、独立heldout和完整缺失行；没有将旧A终测协议套用 |

发布前父任务记录213项focused/integration tests通过；不把重叠suite相加。T9 smoke另行检查staging/mask：恰好3 full＋3 static，O/D各5/5有效、P预期0/5拒绝，115.837424s；**不是3个有效科学结果，也不计入以下66 full**。本次T13只做CPU数据/源码核验，没有新GPU/model/SSH或源码修改。

## 2. 实际机会、13个study与晋级

`complete`表示机会流程完成，不表示child优于父。两波固定映射如下，所有8机会均complete，drain=0。

| 波/槽 | task/arm/rep/seed | B40数/asked | native扩域 | full-policy incumbent | elapsed s |
|---|---|---:|---|---|---:|
| 1/A0 | 43/G0/0/0 | 1/40 | eligible，no_new_choices | child | 4992.359 |
| 1/A1 | 43/C2/0/0 | 2/80 | accepted1 | child | 5528.993 |
| 1/B0 | 21/G0/0/0 | 1/40 | ineligible | child | 3836.651 |
| 1/B1 | 21/C2/0/0 | 2/80 | accepted1 | **原parent** | 6481.510 |
| 2/A0 | 21/C2/1/2 | 2/80 | accepted1；base winner保留 | child | 6275.129 |
| 2/A1 | 21/G0/1/2 | 2/80 | accepted1；base winner保留 | child | 6327.800 |
| 2/B0 | 43/C2/1/2 | 1/40 | eligible，no_new_choices | child | 4340.568 |
| 2/B1 | 43/G0/1/2 | 2/80 | accepted1 | child | 4806.558 |

逐study核对`TUNING_DONE`与TrialRecord，不能仅计最后snapshot。下表quick为该space记录中的最佳median，**包含允许的witness/prior测量复用**，不是该space独立full结果。

| 机会 | study/space | asked | complete | fail | best quick ms |
|---|---|---:|---:|---:|---:|
| 1/A0 | initial sp-2c10b493 | 40 | 30 | 10 | 3.134464025 |
| 1/A1 | initial sp-27bdd648 | 40 | 36 | 4 | 3.005439997 |
| 1/A1 | expanded sp-d1c21d32 | 40 | 40 | 0 | 3.003376007 |
| 1/B0 | initial sp-e720b5a3 | 40 | 30 | 10 | 6.408191919 |
| 1/B1 | initial sp-5e1ab221 | 40 | 32 | 8 | 16.265727997 |
| 1/B1 | expanded sp-cc2e54f3 | 40 | 34 | 6 | 15.259135723 |
| 2/A0 | initial sp-6af980fc | 40 | 40 | 0 | 5.230576038 |
| 2/A0 | expanded sp-7edf1bc7 | 40 | 39 | 1 | 5.230576038 |
| 2/A1 | initial sp-83869f74 | 40 | 35 | 5 | 6.200831890 |
| 2/A1 | expanded sp-93dc7f1c | 40 | 30 | 10 | 6.200831890 |
| 2/B0 | initial sp-2710c4b9 | 40 | 34 | 6 | 3.030480027 |
| 2/B1 | initial sp-58141286 | 40 | 36 | 4 | 3.056639910 |
| 2/B1 | expanded sp-560b108e | 40 | 34 | 6 | 3.045423985 |

**合计13 B40、520 asked=450 complete+70 fail**；fail为48 infeasible_shared_memory、19 correctness_mismatch、3 runtime_error，30个TRIAL_DONE显式reuse。不能从520推成520个新worker/full jobs。两臂政策相同，但实际G0为6 studies/240 asked、C2为7/280，实际预算和成本并不相等。

### Screen晋级与计数

- 48 endpoint attempts=46 complete+2 fail，均来自四个C2机会，各12次；两次失败均在task43批次，失败不补采。G0=0。
- 31个logical模型调用started/finished：8 analyst＋8 rewriter＋8 initial parameterizer＋7 native expansion requests；记录attempts32（含既有有界格式/调用尝试），不作隐藏模型工作归零推断。
- **48 selection screen full、0 confirmation full、18独立heldout full=66 scientific full**，6600样本、330/330 correctness。116是screen48＋最多confirmation32＋heldout36的上限，**不是必须全部测116**；本次明确缺少的是wave2规定的18 heldout块。3 smoke另计。
- 8机会的parent/child screen范围均不相触，native quick与full方向亦不冲突；实际production `decide_promotion`和独立检查重现7 child/1 parent，故0 confirmation是符合规则，而非漏执行。内部晋级无2%最低门槛；2%只属于外部投入gate。
- wave1/B1的child是有效但慢：expanded quick15.259ms，child full median15.323ms，对应parent full8.462ms。扩域在它初始慢于父时仍实际执行，最终full规则保留原父，没有乐观quick晋级。它仍进入heldout且有自己的三块新父测量。

## 3. 四个主比较：只用独立同卡 heldout

`M=median(三块median)`；同pair共同fresh parent为分母，`R=M_arm/M_parent`。C2相对G0改善`100×(1−M_C2/M_G0)`，需≥2%且`max(C2)<min(G0)`。全局需至少3/4、覆盖两任务、且无missing/censored pair。范围只是描述性信息，不是置信区间/显著性。

实际production `c2_opportunity_summary.summarize`在8机会＋4份真实heldout记录上返回 **inconclusive，wins=1**；独立NumPy算术一致。

| pair/比较卡 | heldout状态 | M_parent ms | M_G0 ms | M_C2 ms | R_G0 / R_C2 | C2改善 | 范围门槛/结果 |
|---|---|---:|---:|---:|---|---:|---|
| task43 rep0/wave1 A0 | complete | 3.492415905 | 3.098608017 | 2.992127895 | 0.887239121 / 0.856750163 | **+3.436386%** | 严格有利分离；PASS |
| task21 rep0/wave1 B0 | complete | 8.362495899 | 6.047263861 | 8.360960007 | 0.723141026 / 0.999816336 | **−38.260215%** | C2更慢；FAIL |
| task21 rep1/wave2 A0 | **censored，未测** | — | — | — | null / null | — | INCONCLUSIVE |
| task43 rep1/wave2 B0 | **censored，未测** | — | — | — | null / null | — | INCONCLUSIVE |

wave1 task21 C2≈原父只是该保留策略的表现，不是新结构实现了零收益；其新结构的screen更慢，未成为incumbent。wave2的screens虽然存在，也不是主比较；本报告不跨GPU直接拿它们比较G0/C2，不为两个空pair生成“趋势gate”。因此既不宣称全局通过，也不以已完成一胜一负把全局改成FAIL。

## 4. 扩域究竟做了什么

本次与旧OFF运行不同：两臂实际cap1，**7次资格为true并请求、5次SPACE_EXPANDED、2次no-op、1次不合资格**。S7/scanner仍OFF。initial parameterizer定义新结构的域，与同一结构native扩域是不同阶段；SPACE_PUBLISHED也不等于SPACE_EXPANDED。

| 机会 | eligibility方向 | 实际新增choices/拒绝 | 最终native winner与新增值使用 |
|---|---|---|---|
| 1/A0 G0 | BLOCK_N min | identical domains；no_new_choices | initial sp-2c10b493 |
| 1/A1 C2 | BLOCK_M min；NUM_WARPS min | M+8,16；warps+1 | expanded sp-d1c21d32，但M64/N64/W4/S2均已在initial域，**无新增值** |
| 1/B0 G0 | 无资格 | 不扩域 | initial sp-e720b5a3 |
| 1/B1 C2 | SPLIT_M max；STATS_BLOCK_M max；DW_BLOCK_W max；FIN_BLOCK_C min | +32,64；+256,512；+512,1024；+4,8 | expanded sp-cc2e54f3，用STATS_M512、FIN_C8；前者在SK1不活跃；**child未晋级** |
| 2/A0 C2 | M max；N min；K max；SPLIT_M max；STATS_M min；STATS_C max；DW_H min；FIN_C min；EW_WARPS max | M+128；N+16,32；K+64,128；SPLIT_M+64,128；STATS_M+16,32；STATS_C+64,128；DW_H+2,4；FIN_C+4,8；EW_WARPS+16 | **base sp-6af980fc保留**，不是最后sp-7edf1bc7；无新增值 |
| 2/A1 G0 | BLOCK_M max；SPLIT_M max；FIN_BLOCK_C min | M+128,256；SPLIT_M+32,64；FIN_C+4,8 | **base sp-83869f74保留**，不是最后sp-93dc7f1c；无新增值 |
| 2/B0 C2 | BLOCK_M min；BLOCK_N min | identical domains；no_new_choices | initial sp-2710c4b9 |
| 2/B1 G0 | BLOCK_M max；BLOCK_N min | 只增加M256；N未加新值 | expanded sp-560b108e，M64/N32/W4/S4仍在initial域，**无新增值** |

**源码核验而非读取flag就下结论：** 五次accepted expansion的before/after完整可执行AST及PARAMS默认值都相同。1/A1的`source_changed=true`是**仅注释变化**：before `d3fd0f7f1db74099d8b8b7e1b9a6402c2ce514c61a9701d8a79ab5a6cf0d718a`，after `98fe77bf90391244248ac65b2a97d92ec29ea5c2046c5abd0354d6733269be15`；其`prior_best_source_matches=false`仍按实际保守字节策略记录。其余四次字节也完全相同，prior-match=true。内容实体在`waveN/<slot>/information/retune/artifacts/<sha256>`，所有base/best/before/after引用共26次核验匹配，13个space snapshot与study对应。

实际含义有限而具体：

1. **接通并执行了扩域，不再是只改配置。** 例如1/B1初始quick16.266ms仍获扩域机会，说明未先按旧父分数筛掉；no-op没有伪造第二B40。
2. **三次expanded-space winner不等于三次新区域收益。** 两次winner仍在旧choices；另外一次1/B1使用新值但结构未晋级，且STATS轴在所选路径不生效。
3. **两次扩域后base winner被正确保留。** 2/A0、2/A1的expanded最佳数值与base相同，包含复用的prior-best anchor，未建立更好新点；报告按真实selected_trial.space_id而非最后published space归属。
4. 缺少“同域额外B40”配对control。新域＋额外搜索/选择、测量波动及cache/anchor复用共同存在，**不能从两列quick下降推出纯domain扩展因果效应**。本次accepted expansion没有可执行body变化，不表示一般扩域必然保持body，也不补上该预算对照。

## 5. 固定 C2 rep0 机制链：输入、主张、实现、测量分别看

代表固定为wave1/A1 task43与wave1/B1 task21，后者即使保留父也不换好案例。[机制预声明](../.omo/notepads/v5-c2-opportunity-driven/mechanism-predeclared.md)记录测量前可选点选择。以下局部response是本次采集；proposal引用的内部timing/profiler是**生成者声称**，不当作独立复现实验。

### task43：资源权衡线索落成head拆分，但新域收益未被隔离

| 环节 | 证据及解释 |
|---|---|
| 实测conditional方向 | bf16/plain、其他伙伴固定：M16→128 latency3.697152→3.495936ms，同时register79→255、spill0→16；warps1→8为8.263680→3.615232ms，spill1262→0；N16→128反而3.528704→4.036592ms，spill0→274。不是“资源越少必然越快”或全局梯度。 |
| 声称的限制/耦合 | 预调优proposal认为HS96 padding到D128抬高tile/warp/pipeline成本，想改变M、warps、stages联合区域。其“stage3之前compile-refused”没有被本次fresh stage1/4端点直接证实，必须保留主张标签。 |
| 实际结构动作 | initial child将D128 QK/PV改为64+32贡献及两个PV accumulator/store，同时y直接以compute dtype物化，减少fp32-y往返。causal/softmax组织保留；QK求和次序改变，不据注释宣称严格等价。 |
| 目标及表达 | proposal强调M128×stage3以及M{64,128}×warps{4,8}、N32；initial domain已覆盖。扩域新增M8/16、warps1，不是首次开放前述目标。 |
| 真正选中/观测 | expanded winner为bf16/plain/M64/N64/W4/S2，未用新值，也不是完整复现所述目标点。child screen约2.940ms；独立同卡heldout C2=2.992128ms、G0=3.098608ms，pair通过。结果支持该次package的较好incumbent，不隔离head拆分、物化、intent或新域的单独效应。 |

source/proposal入口为`wave1/A1/information/generation/events.jsonl`及`generation/sandboxes/parameterizer-5d4443c2/candidate/{source,parameterized}.py`，实际winner见附录。原父在`information/common/parent.py`；response与selected伙伴上下文在同机会的导出sandbox中。

额外四点检查记 **N/A/0 jobs**：预调优目标有多个轴/伙伴边，结构还同时改变head与物化，不能唯一截取两点而强行作机制图；不构造新消融源、不换rep。这不等于机制失败。

### task21：真正实现了布局机会，净成本却未兑现

| 环节 | 证据及解释 |
|---|---|
| 实测conditional方向 | selected伙伴下SPLIT_K1→8为7.325696→10.253824ms，N16→128为12.230144→8.782848ms，M16→128为10.304960→9.300480ms。支持研究SK1及GEMM伙伴；DW H/W/warps受`probe_budget`限制未测，没有实测DW-width slope。 |
| 声称的限制/耦合 | H3 proposal把重点放到stride2 depthwise gather/raw1重复读取、sector overfetch；其“DW占65%”“布局后大幅变快”等来自生成者自述profiler/timing，非本分析独立证据。条件response到DW瓶颈主张之间因此有推断跳跃。 |
| 实际结构动作 | expand GEMM把raw1写入四个奇偶plane，地址`((h%2)*2+w%2)*HS*WS+(h//2)*WS+w//2`；DW按plane读，在输出列形成unit-stride逻辑地址并静态展开tap。buffer仍fp32、BN1在消费者read后normalize；同时存在散写/索引、代码量和live-state成本。连续逻辑地址不自动证明实际向量化或带宽主导。 |
| 目标联合区域 | SK1、plain fp16 compute、布局相关DW tile/warp和必要GEMM retune。SK1去掉raw3累加与独立stats pass，**不是去掉所有atomic**，融合的BN统计仍atomic_add。initial域已表达该目标；新DW512/1024在W2=112时不增加有效输出列，主要扩大masked区域。 |
| 扩域及实测断点 | 初始best quick16.266ms，扩域后15.259ms，最终child screens约15.323ms，仍远慢于parent8.462ms；因此原父被保留。新增STATS_M512在SK1无效，FIN_C8不是核心布局假设。heldout C2重新测原父8.360960ms，G0=6.047264ms。**结构动作实现与域扩展发生都未保证更好的可达调优结果。** |

来源为`wave1/B1/information/generation/events.jsonl`、`generation/sandboxes/parameterizer-e3b1215c/candidate/{source,parameterized}.py`、`responses.json`和native artifact refs。该initial parameterizer输入/输出字节相同，本轮这个负机会不能归到parameterizer改坏body；也不能未经配对ablation认定布局或tap展开哪一项单独造成变慢。

预声明曾允许原父/initial child各测DW_W32/128、伙伴相同且SK1，共4 QUICK；**实际NOT RUN/0 jobs**，因为mandatory heldout未完成且deadline过期。没有挑替代case、domain或赢家；缺少的是声称width权衡的额外实测，不是缺少整个证据链。后续诊断应针对这条“未测DW响应→瓶颈假设→scatter/unroll成本→净收益”的链，而非泛泛增加更多guardrails或自动扩大实验。

## 6. Rep1全部轨迹与parameterizer层次

Rep1也回到原P1，不接rep0。本节只解释源码动作与本轮探索轨迹，**screen不是跨GPU主结果**，没有择优加做机制点。

| rep1机会 | 声称与实际源码动作 | 目标区域/实际配置及限制 |
|---|---|---|
| task21 C2，2/A0 | response支持SK1、较宽N等固定伙伴方向；`_gemm_fullk_atomic`与`_proj_splitk`交换grid/program-id轴，尝试N相邻tile的L2局部性；保留fp32中间buffer | 预调优目标SK1、fp16/plain、N64/128、stages1/2、BK16；实际选SK1/N64/stage1/**BK32**，只是部分命中。DW轴仍缺response；cache收益没有独立counter证据。 |
| task21 G0，2/A1 | 同样选择既有SK1路径；`ModelNew.forward`增加per-shape持久workspace，`_zero_stats`逐次清理，`_bn_finalize`合并更新launch | 实际SK1/N128/stage1/DW64；workspace驻留、reset、allocator/launch权衡不同。说明SK1并非C2独有发现。两臂关于低精度存储的试验说法仍属生成者报告。 |
| task43 C2，2/B0 | 从warp/spill及N响应推向64+32 head拆分＋低精度y物化，vendor projection保留 | proposal强调重新考察W8与更深pipeline，实际仍M128/N32/W4/S2；dtype ieee端点invalid，不能造连续斜率；stages测的是1/4不是3。 |
| task43 G0，2/B1 | 无fresh response也独立得到head拆分＋低精度y物化的相近package | proposal强调stage3/更宽N，最终M64/N32/W4/S4；结构相似不证明C2信息是发现它的必要条件。两臂的head predicate细节不同，不能只看结构名字认定源码等价。 |

**输入真的送到了吗？** 对全部8机会导出文件检查：16份analyst/rewriter `tuning/selected_params.json`与实际parent参数匹配；四个C2的8份response交付与其raw数组一致，G0 sandbox无conditional-response文件；8份初始parameterizer intent与预调优proposal `change_summary`一致。可确认请求输入与外显proposal/source联系，**不能确认模型内部如何思考、是否逐条遵从**；没有完整会话transcript导出，不虚构其推理过程。

### 初始parameterizer前后AST（不是扩域AST）

| 机会 | input source.py → intended parameterized.py |
|---|---|
| 1/A0 G0 | 完整AST/defaults不变，仅文本改变 |
| 1/A1 C2 | defaults不变；去掉import时_DT/_MODE，forward按PARAMS每次派生dtype/mode并绑定launch块参数 |
| 1/B0 G0 | 新增PROJ_STAGES=2，projection launch从GEMM_STAGES改读PROJ_STAGES，拆分两个pipeline参数 |
| 1/B1 C2 | 字节与AST相同 |
| 2/A0 C2 | 字节与AST相同 |
| 2/A1 G0 | 字节与AST相同 |
| 2/B0 C2 | `_dotx`的bf16/split3分支`bh=a32.to(...)`改为`bh=b32.to(...)`；改变第二operand高位，实际selected bf16/plain不走该分支 |
| 2/B1 G0 | defaults不变；去掉import时_DT/_MODE，forward在launch时派生 |

共4个body变化、4个AST不变，未发现这8次parameterizer引入load/store/causal mask变化；这不是所有源码安全或性能中性的证明。不能把变化自动当坏事或自动解释收益；旧mask诊断是不同cohort，不证明本轮发生同样问题。初始参数化body修改、参数choices/defaults、TPE选点、full promotion、heldout五层应分开，不能统称“调参效果”。

## 7. 缺失heldout：必须承认的调度损失

唯一时钟：start **12:54:32.920922 UTC**，deadline **16:54:32.920922 UTC / 1789750472.9209218**。未重置、未追加许可，8机会均在截止前完成，但这不足以完成campaign。

| 节点 | 时间/时长 | 意义 |
|---|---|---|
| wave1最后机会结束 | 14:42:36；共享窗口6484s=1h48m04s | 不是各GPU时长相加 |
| wave1 heldout | 14:58:12/13启动；最后15:03:15完成 | 18新full块，后才启动wave2 |
| wave2启动 | 15:06:35–36 | 共用原deadline，剩约6477s |
| task43 pair可用 | B0 16:18:57、B1 16:26:43；pair **16:26:43**齐 | **尚余27m49.920922s** |
| 已观察到该pair齐 | 16:27:20.541355 | 尚余27m12.379567s；B-host导出16:32:20.534032已写完 |
| task21 pair齐 | A0 16:51:11、A1 16:52:04 | 最后pair尚余2m28.920922s |
| 最后完成首次观察 | 16:53:59.280988 | 115.280988s观察延迟，只剩33.639934s |
| A-host导出 | 16:55:56.461642 | 已超deadline83.540720s；收集不伪装成预算内科学工作 |

**task43较早的pair-local窗口被我们自己的全波交接边界浪费了。** T11等待HOST-A/全wave2完成，将两组heldout都保留给下一T12 dispatch，而没有在B0/B1已完成时及时派发task43。不是科学协议要求所有卡必须等齐，也不是当时所有GPU始终忙碌，更不能说“两个heldout都因为调优不可避免耗完四小时而无法做”。这证明存在未利用的准入窗口；不保证假设中的heldout一定会完成或得出什么成绩。

T12没有用延时补测补救。真实record-only CLI在17:08:19/20左右运行，late_start分别826.383900s、827.745056s；原deadline守卫立即拒绝，数组全空，事件仅RUN_CREATED/HELDOUT_CENSORED/RUN_FINISHED、job/model/evaluation数全0。late record creation不是scientific drain；原有科学drain均0。两组heldout各缺9块；机制task43=N/A，task21=预声明4点但未准入，actual QUICK=0。

## 8. 时间与成本：实际窗口、软件字段与未知

- 跟踪初始化08:53:55.445到首次科学启动12:54:32.920922，**4h00m37.475922s**，含实现/测试/发布/部署/smoke/等待，不能拆造成精确人工工时，也不能声称实现已经按2–3h名义预测完成。setup smoke115.837424s单列，部署完成A12:23:41、B12:26:26。
- 最后科学测量事件为wave2/A1 `RETUNE_FINAL_DONE`，**16:52:03.813684 UTC**，距first CLI **14250.892762s=3h57m30.893s**，离cap尚149.107238s。16:52:04是秒级机会完成marker。4h cap被遵守，却没有达到完整heldout覆盖；后续导出/closure/分析均另计，最终cleanup回执17:13:32。
- wave2共享窗口6329s=1h45m29s。不要把每臂elapsed或各阶段重叠子项相加冒充共享窗口、GPU busy、严格四卡加速或独立部署时长。模型并行/流水和host争用都存在，没有串行counterfactual测量。

已有事件可支持的部分wall账（秒）：retune inclusive包含validation、编译、初始/扩域study、expanded模型及child screen；标为子项的列**不要重复相加**。

| 机会 | generation agent | acquisition worker | parent screen | child screen（retune内） | retune inclusive | expansion agent（retune内） |
|---|---:|---:|---:|---:|---:|---:|
| 1/A0 | 996.662 | 0 | 91.451 | 68.353 | 3901.114 | 158.303 |
| 1/A1 | 1276.600 | 399.355 | 98.950 | 91.682 | 3751.224 | 132.559 |
| 1/B0 | 1731.560 | 0 | 102.680 | 105.592 | 1996.646 | 0 |
| 1/B1 | 1643.037 | 399.346 | 75.054 | 109.246 | 4358.207 | 246.938 |
| 2/A0 | 1613.487 | 300.246 | 90.306 | 88.299 | 4267.528 | 180.394 |
| 2/A1 | 2090.032 | 0 | 91.901 | 83.807 | 4142.701 | 131.491 |
| 2/B0 | 1299.954 | 293.839 | 131.871 | 51.950 | 2608.860 | 337.806 |
| 2/B1 | 1091.419 | 0 | 77.780 | 68.839 | 3631.504 | 110.041 |

每个`SPACE_PUBLISHED→TUNING_DONE`窗口包含prescreen/tuning等，不是纯GPU测量：1/A0=3581.233s；1/A1=1691.924/1729.787；1/B0=1848.509；1/B1=1953.006/1943.633；2/A0=2039.588/1854.753；2/A1=2050.660/1761.207；2/B0=2067.143；2/B1=1799.098/1567.745（斜杠为初始/expanded）。确认0是实际未触发，不是缺失字段补零。

wave1 heldout executor为task43=302.608260s、task21=240.712069s，记录transfer/wait=1890.196684s、937.375192s；该字段包含等待准入等，不解释成纯网络拷贝耗时。未执行的wave2 heldout不能套这些时长补成实测。软件/provider-report字段总和 **0.83235592**，只作为记录值，不是独立核实账单、租卡成本或已验证币种费用；完整计费、GPU busy、人力工时未知，不作零成本声明。

## 9. 完整full块表与质量/保真

下面22行×3块=**66个实际full median**，单位ms、显示9位小数。前16行为screen，后6行为wave1 heldout；wave2 heldout为空，不复制screen或wave1块。完整精度见[analysis.json](../results/c2-opportunity-driven/analysis.json)。

| 阶段/波/槽 | 角色 | block1 | block2 | block3 |
|---|---|---:|---:|---:|
| screen/1/A0 | parent | 3.489792109 | 3.495935917 | 3.493359923 |
| screen/1/A0 | child | 3.099647999 | 3.098623991 | 3.102720022 |
| screen/1/A1 | parent | 3.432447910 | 3.432447910 | 3.432447910 |
| screen/1/A1 | child | 2.940464020 | 2.941951990 | 2.939903975 |
| screen/1/B0 | parent | 8.363007545 | 8.361472130 | 8.364543915 |
| screen/1/B0 | child | 6.047231913 | 6.059008121 | 6.052271843 |
| screen/1/B1 | parent | 8.459343910 | 8.461823940 | 8.466943741 |
| screen/1/B1 | child | 15.296000004 | 15.323136330 | 15.323647976 |
| screen/2/A0 | parent | 8.497727871 | 8.503808022 | 8.507391930 |
| screen/2/A0 | child | 5.161504030 | 5.158400059 | 5.166543961 |
| screen/2/A1 | parent | 8.359935760 | 8.369152069 | 8.362991810 |
| screen/2/A1 | child | 6.068223953 | 6.061568022 | 6.065711975 |
| screen/2/B0 | parent | 3.434495926 | 3.430399895 | 3.428895950 |
| screen/2/B0 | child | 2.966527939 | 2.967551947 | 2.969087958 |
| screen/2/B1 | parent | 3.470335960 | 3.479552031 | 3.473407984 |
| screen/2/B1 | child | 2.982912064 | 2.985984087 | 2.985952020 |
| heldout/1/A0 | parent | 3.491344094 | 3.494911909 | 3.492415905 |
| heldout/1/A0 | G0 | 3.098623991 | 3.098608017 | 3.097599983 |
| heldout/1/A0 | C2 | 2.991103888 | 2.992127895 | 2.992159963 |
| heldout/1/B0 | parent | 8.361999989 | 8.362495899 | 8.371200085 |
| heldout/1/B0 | G0 | 6.047263861 | 6.049279928 | 6.045695782 |
| heldout/1/B0 | C2（保留父新测） | 8.360960007 | 8.360447884 | 8.366032124 |
| heldout/2/A0 | parent/G0/C2各缺三块 | — | — | — |
| heldout/2/B0 | parent/G0/C2各缺三块 | — | — | — |

原始worker数组与record一一对应，66份实际job源码均与冻结parent/selected在测量参数下materialize结果相同。五位小数导出sample重算median最大差 **0.000004239501953051672ms**，低于容差 **0.000005000001ms**。Raw FP64 rescue总165：wave1 A0/A1各30、wave2 B0/B1各30、heldout wave1 A0为45，其余组0。rescue通过不是严格数学等价；normalized null字段不据此填零。

所有8个initial primary/reference与staged字节一致；8个实际native winner与measured trial对应、13个空间正确计数；incumbent/helper哈希一致。wave1/wave2/heldout1/closure分别 **1686/1876/100/10条**manifest hashes通过，另本次smoke38条；是条目数，不冒称唯一文件数。旧cohort的原始不变由已有receipt背书，本次未重新扫描旧数千文件。

## 10. 结论、未解决事项与停止

1. **“是否用上C2机会链”：输入与外显实现层面是。** 实际selected context、fresh response、intent送达；固定rep0案例确有从条件线索到结构/伙伴的主张，并有代码动作。但task21的DW主张越过了实际endpoint覆盖，task43的stage3说法也比fresh证据更强；不能把生成文字当测量。
2. **“是否得到更优结构”：目前只支持有限混合结果。** 已测task43 pair支持更好incumbent；task21固定C2案例实现了四plane布局却在retune/full后失利并保留父。wave2的新动作有诊断价值但无独立heldout，不能宣布改善。C2的目标仍是改善结构在自身retune后的可达区域，不是要求新结构在旧固定参数点上赢。
3. **“是否搜到新域收益”：执行覆盖已建立，独立效应未建立。** 5实际扩域，2 base保留，2 expanded winner仍用旧choices，1用了新增值却未晋级；没有同域额外B40配对control。不能从配置ON、空间数增加或quick下降得出论文阳性。
4. **campaign全局INCONCLUSIVE，且有可改的执行失误。** 四pair只有两pair有效，不能通过放宽2%、减少分母、改时间预算、复用screen或截止后catchup来补结论。task43较早完成后可准入的窗口被全波交接耽误，应承担这项调度责任。
5. **后续仅建议，不执行。** 若要补证据，另行预声明follow-up与新时间预算，不回填当前campaign；优先实现pair-local及时heldout派发/明确时间预留。方法研究重点应是检验response到瓶颈/布局假设的证据质量、必要伙伴与新增代价，而不是泛化的防护平台或自动追加更多试验。

本轮是discovery-biased的两个已知benchmark、每任务两次fresh生成的pilot，且主比较少一半；无法证明通用有效性、独立guidance/intent因果效应或全局结构最优。实验程序至此停止，未授权补测、第三波或扩大样本。

## 附录A：实际native winner配置及incumbent区别

路径根为`results/c2-opportunity-driven/waveN/<slot>/information/retune/`。每个native winner的`report/selected.py`对应`candidates/<candidate>/trials/<trial>.py`，完整domain/constraints在`SPACE_PUBLISHED`与result空间快照。**只有8个最终native winners，不是13个**。1/B1表中是未晋级child，真正incumbent另列。

| 机会 | candidate / space / trial | native selected PARAMS |
|---|---|---|
| 1/A0 | cand-88a16232 / sp-2c10b493 / tr-dc0f40a3 | `{"COMPUTE_DTYPE":"bf16","DOT_MODE":"plain","BLOCK_M":128,"BLOCK_N":32,"NUM_WARPS":4,"NUM_STAGES":2}` |
| 1/A1 | cand-74ad95d9 / sp-d1c21d32 / tr-bbec8629 | `{"COMPUTE_DTYPE":"bf16","DOT_MODE":"plain","BLOCK_M":64,"BLOCK_N":64,"NUM_WARPS":4,"NUM_STAGES":2}` |
| 1/B0 | cand-6b58050e / sp-e720b5a3 / tr-ac63d235 | `{"BLOCK_M":32,"BLOCK_N":128,"BLOCK_NP":128,"BLOCK_K":64,"SPLIT_M":16,"SPLIT_K":1,"STATS_BLOCK_M":128,"STATS_BLOCK_C":16,"DW_BLOCK_H":16,"DW_BLOCK_W":128,"EW_BLOCK":256,"FIN_BLOCK_C":32,"COMPUTE_DTYPE":"fp16","DOT_MODE":"plain","GEMM_WARPS":4,"GEMM_STAGES":2,"PROJ_WARPS":4,"PROJ_STAGES":2,"STATS_WARPS":4,"DW_WARPS":4,"EW_WARPS":4}` |
| 1/B1（非incumbent） | cand-48089a01 / sp-cc2e54f3 / tr-5e14a39d | `{"BLOCK_M":64,"BLOCK_N":64,"BLOCK_K":16,"SPLIT_M":16,"SPLIT_K":1,"STATS_BLOCK_M":512,"STATS_BLOCK_C":32,"DW_BLOCK_H":16,"DW_BLOCK_W":128,"EW_BLOCK":2048,"FIN_BLOCK_C":8,"COMPUTE_DTYPE":"fp16","DOT_MODE":"plain","GEMM_WARPS":4,"GEMM_STAGES":3,"STATS_WARPS":1,"DW_WARPS":4,"EW_WARPS":8}` |
| 2/A0 | cand-c2be6230 / sp-6af980fc / tr-0a5332e4 | `{"BLOCK_M":64,"BLOCK_N":64,"BLOCK_K":32,"SPLIT_M":32,"SPLIT_K":1,"STATS_BLOCK_M":64,"STATS_BLOCK_C":32,"DW_BLOCK_H":8,"DW_BLOCK_W":64,"EW_BLOCK":512,"FIN_BLOCK_C":16,"COMPUTE_DTYPE":"fp16","DOT_MODE":"plain","GEMM_WARPS":4,"GEMM_STAGES":1,"STATS_WARPS":2,"DW_WARPS":8,"EW_WARPS":8}` |
| 2/A1 | cand-e0b3ce5f / sp-83869f74 / tr-fd5e0eae | `{"BLOCK_M":64,"BLOCK_N":128,"BLOCK_K":16,"SPLIT_M":16,"SPLIT_K":1,"STATS_BLOCK_M":128,"STATS_BLOCK_C":32,"DW_BLOCK_H":8,"DW_BLOCK_W":64,"EW_BLOCK":1024,"FIN_BLOCK_C":16,"ZERO_BLOCK_C":64,"COMPUTE_DTYPE":"fp16","DOT_MODE":"plain","GEMM_WARPS":8,"GEMM_STAGES":1,"STATS_WARPS":2,"DW_WARPS":4,"EW_WARPS":8}` |
| 2/B0 | cand-1c5b1ef0 / sp-2710c4b9 / tr-cf6e3465 | `{"COMPUTE_DTYPE":"bf16","DOT_MODE":"plain","BLOCK_M":128,"BLOCK_N":32,"NUM_WARPS":4,"NUM_STAGES":2}` |
| 2/B1 | cand-8098f763 / sp-560b108e / tr-cc4ee62c | `{"COMPUTE_DTYPE":"bf16","DOT_MODE":"plain","BLOCK_M":64,"BLOCK_N":32,"NUM_WARPS":4,"NUM_STAGES":4}` |

1/B1实际incumbent为原P1 `cand-4c96b8c4 / sp-e3acfbaf`，导出`wave1/B1/export/selected.py`；PARAMS为：

```json
{"BLOCK_M":32,"BLOCK_N":128,"BLOCK_K":16,"SPLIT_M":16,"SPLIT_K":4,"STATS_BLOCK_M":128,"STATS_BLOCK_C":16,"DW_BLOCK_H":16,"DW_BLOCK_W":128,"EW_BLOCK":1024,"FIN_BLOCK_C":16,"COMPUTE_DTYPE":"fp16","DOT_MODE":"plain","GEMM_WARPS":4,"GEMM_STAGES":2,"STATS_WARPS":2,"DW_WARPS":4,"EW_WARPS":4}
```

## 附录B：可复核材料与交付状态

- [T10 wave1](../.omo/evidence/v5-c2-opportunity-driven/T10-wave1.md)、[T11 heldout1/wave2](../.omo/evidence/v5-c2-opportunity-driven/T11-wave1-heldout-wave2.md)、[T12零job闭合与调度时序](../.omo/evidence/v5-c2-opportunity-driven/T12-deadline-closure.md)。
- [T13独立核验](../.omo/evidence/v5-c2-opportunity-driven/T13-analysis.md)、[完整精度analysis.json](../results/c2-opportunity-driven/analysis.json)、[机制零job机器记录](../results/c2-opportunity-driven/mechanism-deadline-closure.json)。
- [代码发布回执](../.omo/evidence/v5-c2-opportunity-driven/T9-publish.md)、[部署/smoke](../.omo/evidence/v5-c2-opportunity-driven/T9-deploy-smoke.md)。本地raw根`D:/Pyhon_projects/opop/v5/results/c2-opportunity-driven`，远端绝对路径根`/root/autodl-tmp/c2-opportunity-driven`通过各export manifest映射。

这些raw/.omo链接是本地工作区证据，受gitignore/本地归档边界限制，**不是已公开数据集**；计划/实现与本报告的GitHub归档不包含raw或本地核验回执。当前交付：科学分析完成、coverage明确不完整、全局INCONCLUSIVE；本报告随提交归档，最终核验回执单独交付。无实现/raw/旧报告/计划checkbox修改，无新实验。
