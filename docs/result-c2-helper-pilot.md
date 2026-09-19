# C2 formal-helper pilot：采用已发生，竞争力门槛未通过

日期：2026-09-19。**科学描述性投入gate：FAIL，1/4 pair通过，四个pair全部resolved。执行完整：8/8候选机会、36/36独立heldout full blocks完成。** 本次没有缺失heldout可作解释，也没有删掉负向pair或调低门槛。相对强G0，C2仅在task21 rep1通过；其余三个pair更慢。

另一方面，正式helper的主实验采用已有真实工具证据：**8/8 rewriters、5/8初始parameterizers调用过helper**。推荐配置确实进入原B40，定向请求确实转为条件响应，扩域agent确实读取实际selected证据。这些说明接线与采用成立，**不等于C2效力成立**。两任务、两reps的discovery-biased pilot不支持统计显著性、普适结论或单组件因果归因。

本报告汇合T6数值与uptake两条已完成核验lane；**发布时最终独立F1仍待审阅，未宣称F1 APPROVE**。所有执行已停止，不为阳性追加实验。旧campaign的INCONCLUSIVE、deadline与历史报告保持不变，未作跨批次pooling。

## 1. 协议、实现与此前helper证据不是一回事

| 阶段 | 精确出处 | 本次可证明的事 |
|---|---|---|
| 先归档协议 | [2bb14bf7035755e7e0cf42a3cd5861e49787e867](https://github.com/Fudan-SMI-lab/opop/commit/2bb14bf7035755e7e0cf42a3cd5861e49787e867)；[不可变协议](https://github.com/Fudan-SMI-lab/opop/blob/2bb14bf7035755e7e0cf42a3cd5861e49787e867/docs/plan-c2-helper-pilot.md) | 在pilot产品改动、真实agent gate和主实验前固定预算、时钟、四pair规则与私有工作边界 |
| 此前helper实现 | [8eae0eb4828826ab99c9948ea8160e66e34c3498](https://github.com/Fudan-SMI-lab/opop/commit/8eae0eb4828826ab99c9948ea8160e66e34c3498) | 既有operator固定点集成，不是本pilot自愿采用或新方法效力 |
| T2真实agent readiness | 在8eae0eb上，单一GLM-5.3/max build会话，中性A/B、明确要求用helper | 实际2次QUICK20/3、均有效且正确解释；无重开会话或替换。**显式指令gate不计入主实验voluntary分母** |
| 测试、发布、部署 | [cda113070c8ba3a91f480f42b68d196187796cda](https://github.com/Fudan-SMI-lab/opop/commit/cda113070c8ba3a91f480f42b68d196187796cda)；[相对归档的实现diff](https://github.com/Fudan-SMI-lab/opop/compare/2bb14bf7035755e7e0cf42a3cd5861e49787e867...cda113070c8ba3a91f480f42b68d196187796cda) | 本次两host运行的精确源码；主实验中未修代码、模型、质量或阈值 |

本轮新增/修改范围为12个Python文件，职责是host-local pair controller、显式PilotClock与result/summary身份、formal-helper可选cutoff、配套CPU回归；没有重写TPE、evaluator或晋级策略。代表文件为`c2_helper_pilot.py`、`c2_helper_pilot_records.py`、`c2_opportunity_{inputs,program,heldout,records,summary}.py`和`agents/self_test{,_context}.py`。

继承的推荐配置、实际best证据和targeted路径已在此前实现中提供：推荐最多2个candidate-local合法完整配置、见证/prior anchors保留、占用既有B40；C2显式targeted，G0 provided/empty。正式helper自愿调用，bash/Python/read/write及私有实验保持可用，没有全局PATH/权限禁令。

源码发布前76项focused CPU tests通过，12个改动文件LSP检查清晰；父级另有相关回归复核。重叠suite不相加，CPU验证也不冒充GPU性能结果。T2 readiness本身为2 quick＋2 static、session-through-cleanup272.321083s，独立于主时钟；它验证“agent能用”，主实验再核验“实际是否用”。

## 2. 固定设计、强G0与预声明gate

两任务(task43/task21)×两fresh reps×G0/C2，共8个固定机会；每次从各任务exact original P1/off-state开始，不接前一wave winner，也不拿旧候选或旧成绩当control。初始/expanded sampler均为rep0=0、rep1=2，evaluation/validator=0；不声称LLM seed完全可复现。

| 波/rep | HOST-A GPU0 | HOST-A GPU1 | HOST-B GPU0 | HOST-B GPU1 |
|---|---|---|---|---|
| wave1/0 | A0 task43 G0 | A1 task43 C2 | B0 task21 G0 | B1 task21 C2 |
| wave2/1 | A0 task21 C2 | A1 task21 G0 | B0 task43 C2 | B1 task43 G0 |

两臂共享general guidance、formal helper、intent、推荐配置、原生B40/合资格cap1扩域、工具权限与full晋级。C2增加ordinary planning analyst→原12端点内targeted acquisition→实际response→rewriter；G0没有正式fresh C2端点，但**普通历史、资源、私有profiling与自主实验均允许**，不是弱化对照。比较的是信息与方法package，不能隔离guidance/helper/推荐/扩域单独效力。

每个native空间B40；所有空间结束后，用父3块和最终子3块作selection screen。范围重叠/相触或quick/full严格反向才触发最多两对确认；内部晋级无2%最低门槛。冻结incumbent后，两臂及共同fresh parent在本host GPU0独立heldout，顺序`P,G0,C2 / C2,G0,P / G0,C2,P`，每角色三块。Heldout不重选或继续调优。

主指标`M=median(三块median)`，共同fresh parent归一化`R_arm=M_arm/M_parent`；改善`100×(1−M_C2/M_G0)`。**Pair PASS须改善≥2%且max(C2三块)<min(G0三块)；global须≥3/4、覆盖两任务、四pair全部resolved。** 三块范围是描述性信息，不是置信区间或统计显著性。

## 3. 主结果：完整覆盖下FAIL 1/4

所有值为同卡独立heldout，单位ms；下表保留实际median精度，改善显示至10位小数。生产`summarize`/`compare_blocks`与独立NumPy算术一致。

| Pair | M_parent | M_G0 | M_C2 | R_G0 | R_C2 | C2改善% | 结果 |
|---|---:|---:|---:|---:|---:|---:|---|
| task43 rep0 / A0 | 3.492863893508911 | 2.975200057029724 | 2.9921278953552246 | 0.8517938710863581 | 0.8566402776002102 | -0.5689647083 | FAIL |
| task21 rep0 / B0 | 8.363007545471191 | 3.675136089324951 | 5.568000078201294 | 0.43945148552629765 | 0.6657891969996519 | -51.5045958264 | FAIL |
| task21 rep1 / A0 | 8.50227165222168 | 4.205567836761475 | 3.787775993347168 | 0.49464049242210956 | 0.44550164335873765 | +9.9342552452 | PASS |
| task43 rep1 / B0 | 3.4355199337005615 | 2.944048047065735 | 3.0597119331359863 | 0.8569439572119033 | 0.8906110260405986 | -3.9287363596 | FAIL |

三个失败pair的C2整段范围均高于G0范围，不是有利改善仅差一点未过2%。唯一通过者task21 rep1的max C2=3.7898720502853394低于min G0=4.205567836761475。C2并未覆盖两任务的获胜条件；所有方向保留，不按成功pair缩分母。

**八个child都晋级，不是八个C2胜利。** 它们是在各自screen中优于原父，G0也能得到很强的候选；例如task21 rep0中G0 R=0.439451、C2 R=0.665789，两者均优于共同父，但C2更慢的数值差距在本次描述性比较中保留，不据此宣称统计显著性。

### 全部36个heldout block median与范围

共12角色×3块；以下ms显示9位小数，计算使用上表及本地JSON的未舍入值。没有使用最快块、screen或跨卡raw分数替代。

| Host/wave | Role | Block1 | Block2 | Block3 | min / max |
|---|---|---:|---:|---:|---|
| A/1 | parent | 3.491328001 | 3.492863894 | 3.492863894 | 3.491328001 / 3.492863894 |
| A/1 | G0 | 2.975200057 | 2.974720001 | 2.975744009 | 2.974720001 / 2.975744009 |
| A/1 | C2 | 2.992127895 | 2.992640018 | 2.991103888 | 2.991103888 / 2.992640018 |
| B/1 | parent | 8.360447884 | 8.363007545 | 8.371199608 | 8.360447884 / 8.371199608 |
| B/1 | G0 | 3.671040058 | 3.677184105 | 3.675136089 | 3.671040058 / 3.677184105 |
| B/1 | C2 | 5.565952063 | 5.570064068 | 5.568000078 | 5.565952063 / 5.570064068 |
| A/2 | parent | 8.502271652 | 8.504784107 | 8.497120380 | 8.497120380 / 8.504784107 |
| A/2 | G0 | 4.205567837 | 4.205567837 | 4.206592083 | 4.205567837 / 4.206592083 |
| A/2 | C2 | 3.789872050 | 3.787775993 | 3.784703970 | 3.784703970 / 3.789872050 |
| B/2 | parent | 3.433471918 | 3.436543941 | 3.435519934 | 3.433471918 / 3.436543941 |
| B/2 | G0 | 2.944000006 | 2.944048047 | 2.945024014 | 2.944000006 / 2.945024014 |
| B/2 | C2 | 3.059711933 | 3.059711933 | 3.061728001 | 3.059711933 / 3.061728001 |

## 4. 科学预算、质量与native空间

| 类别 | 实际计数 | 解释 |
|---|---:|---|
| 固定机会/晋级 | 8 complete / 8 child | 无替换、重跑或因结果取消rep |
| Native | **13 spaces，520 asked=415 complete+105 fail** | 每已准入space完整40；包含29个显式reused-measurement事件，不是520新full jobs |
| Native失败 | 87 shared-memory、6 runtime、12 correctness | 全保留，不以最终完整覆盖掩盖搜索失败 |
| C2端点 | 48 attempts=46 complete+2 fail | 四批各12，G0正式acquisition0；失败不补采 |
| 科学full | **48 screen＋0 confirmation＋36 heldout=84** | 116是上限，不要求填满；0 confirmation因各screen范围与quick/full方向无冲突 |
| 模型调用元数据 | 29 named calls、33 recorded attempts | 不等于provider请求数、机会数或benchmark数 |

全部84科学full采用100性能样本/5 correctness、seed0、fp32 reference，原dual-witness-relaxed及FP64 multipliers2/3、fraction0.99、cosine0.99985不变。**420/420 correctness passes中330次依赖FP64 rescue；36 heldout的180 passes中150次rescued。** Heldout task43所有角色每块5次rescue；task21父每块0、两个child角色每块5。不能表述为strict-allclose或严格数学等价。

Native逐space表如下，best quick为该space的最低有效record median，包括合法witness/prior reuse；不是独立full结果。最后发布的space不自动成为selected space。

| Wave/slot | Space | Asked | Complete/fail | Best quick ms | 最终native winner |
|---|---|---:|---|---:|---|
| 1/A0 G0 | sp-c44e8285 | 40 | 30/10 | 3.000831962 | 是 |
| 1/A1 C2 initial | sp-4c958b16 | 40 | 34/6 | 2.999776006 | 是 |
| 1/A1 expanded | sp-5d273a28 | 40 | 37/3 | 2.999776006 | 否 |
| 1/B0 G0 initial | sp-a1d16865 | 40 | 28/12 | 3.686399937 | 是 |
| 1/B0 expanded | sp-a36f39ee | 40 | 26/14 | 3.686399937 | 否 |
| 1/B1 C2 | sp-f3f4954d | 40 | 30/10 | 5.749248028 | 是 |
| 2/A0 C2 initial | sp-c56e791e | 40 | 35/5 | 3.819455981 | 否 |
| 2/A0 expanded | sp-19d6f963 | 40 | 35/5 | 3.813887954 | 是 |
| 2/A1 G0 | sp-19ffd1a8 | 40 | 26/14 | 4.088832140 | 是 |
| 2/B0 C2 initial | sp-fb8cb12c | 40 | 34/6 | 3.099135995 | 是 |
| 2/B0 expanded | sp-249cd004 | 40 | 31/9 | 3.104256034 | 否 |
| 2/B1 G0 initial | sp-b4095469 | 40 | 36/4 | 3.059711933 | 否 |
| 2/B1 expanded | sp-85e0d97d | 40 | 33/7 | 3.041792035 | 是 |

G0实际6 studies/240 asked，C2为7/280；同样cap1/eligibility政策不等于相同实现预算或成本。科学full、native quick/witness、static、readiness gate与agent私有测试分账，不能互相当免费额外trial或重复计数。

## 5. 主实验agent是否真的采用helper

Uptake核验从29个实际named sessions及522条tool records连接到result、commands、worker input和源码/完整PARAMS；不是只看usage文件或文字声称。T2显式readiness会话不计入下面分母。主实验两臂均可选择helper，未因不用它而重做候选。

| Module | 实际helper采用session / 本类session |
|---|---:|
| rewriter | **8/8** |
| initial parameterizer | **5/8** |
| analyst | **0/8** |
| expansion parameterizer | **0/5** |

27条包含helper模块的bash记录经语义核对，都是**CLI调用尝试**：7次因output目录已存在而在worker前失败，20次产生QUICK结果。七次中六次shell因pipeline仍exit0，所以exit0不等于测量成功。**不是27个benchmark，也不是27个有效结果。**

20个helper结果均为20性能样本/3 correctness、seed0：**16 valid、4 invalid，共40次worker submissions（20 static＋20 eval）**。4个invalid为3个worker标runtime_error的实际Triton compile诊断、1个correctness mismatch，都无有效score。没有用“很快”改判失败。

16个valid中 **15个通过3/3 FP64 rescue**，另一个是原父control、0 rescue；这与科学full的330/150 rescue账分开，不能加成独立候选成功率。20件产物按用途分为4无效中间版、1有效parent control、2有效fp16 override、13有效emitted-default测试（含跨module重复），不是20个新生成候选。

| 机会 | Rewriter / initial-param helper结果数 | 有效/无效结果 | pre-worker错误 | ad-hoc GPU命令尝试 |
|---|---|---|---:|---:|
| W1 A0 G0/43 | 1/1 | 2/0 | 1 | 4 |
| W1 A1 C2/43 | 1/1 | 2/0 | 1 | 3 |
| W1 B0 G0/21 | 1/1 | 2/0 | 0 | 14 |
| W1 B1 C2/21 | 1/0 | 1/0 | 1 | 0 |
| W2 A0 C2/21 | 2/1 | 2/1 | 1 | 0 |
| W2 A1 G0/21 | 6/2 | 5/3 | 2 | 3 |
| W2 B0 C2/43 | 1/0 | 1/0 | 0 | 1 |
| W2 B1 G0/43 | 1/0 | 1/0 | 1 | 1 |

实际helper使用绝对configured worker：A为kernel-opt-venv，B为orch-venv；20条source/reference hashes及完整参数都与执行快照对应。结果的runtime-version字段仍为unknown；部分版本命令提供旁证，不冒充逐结果不可变环境指纹。六个initial-param结果context的semantics标签为unknown，reference/worker仍为准，不能推断用了eval mode。

### 测试、修改、再测链条及不能泛化的覆盖

- W2 A0 C2的helper先报告mask维度compile错误；记录中的修改将K mask改到列向量轴，后续helper通过，initial parameterizer又测同一最终源。这是直接可追溯的feedback→fix→retest→submission，不是性能效力证明。
- W2 A1 G0保留六次rewriter结果：arange错误、mask错误、含FP64失败的数值错误、parent control、修复后default、fp16 override；parameterizer重复default/override。不能把前三个中间失败和parent control都叫submitted-child成功。
- W1 A1 C2的parameterizer增加BLOCK_D1、修复bf16/split3 operand和live precision lookup，并测试default；但bf16/plain测试不覆盖split3或所有D1值。W1 A0也有parameterizer body修改后的helper证据；“rewriter已测”不能代替post-edit覆盖。
- W1 B0的小shape ad-hoc测试曾打印cosine0.709114/maxdiff4.231却无条件`OK`；后续formal shape结果通过FP64 rescue。两种观察都保留，不混成同一正确性标准。

### 私有工具并未消失，也没有全协议同化

160条bash中有 **26个command-level ad-hoc GPU执行尝试，7个非零exit；6个含timer的命令尝试，4个返回timing输出，2个未到timer就失败**。一命令可测多个配置/stage，不能当kernel-launch计数。4个tool-state errors又是另一类（读路径/编辑错误），不能与4 invalid helper、7 pre-worker错误、105 native失败直接相加。

私有timer仍有warm-loop平均、不同warmups和输入clone方式；G0 W2 B1 analyst还用plain Python记录Torch2.8.0/Triton3.4.0做stage/整模型计时。**Helper可用且被采用，没有消除全部非正式或runtime不匹配测试。** G0 analyst的helper使用0也不等于没实验；强G0私有profiling本来就允许。这里不推断compiler/cache导致主结果，也不禁止自主工具。

## 6. 推荐配置与定向探测：有测量覆盖，但不是因果收益

### 推荐链：15个emitted点到24个space请求

每个rewriter都发出了candidate-local建议：除W2 B0为1个，其余各2个，共 **15个emitted点**。初始parameterizer收到并解析，跨13 spaces形成 **24 REQUESTED、15 QUEUED、9 duplicate-anchor SKIPPED**。24个请求按candidate ID＋space ID＋完整ParamSet都匹配到complete TRIAL_DONE，含witness/prior复用；不是24个额外GPU测量，仍在B40内。

仅W1 A1需把新key `BLOCK_D1=64`解析进两个建议；其余initial输入值保留。只有W2 B0最终selected完整配置等于建议，且该点也是现有default witness，不能称建议额外贡献了选中结果。其余建议虽被测量，未直接成为最终winner；QUEUED、正式覆盖与效果是不同层级。

### Targeted链：21个请求，48次有界attempts

四条C2实际顺序均为preliminary planning report→request-led acquisition→实际conditional responses→rewriter→parameterizer/native。G0为provided/empty、0正式fresh C2端点，但不限制私有探索。

| C2机会 | 请求contrasts | request-serving / fallback新端点 | complete/fail | observed / invalid / budget-unknown响应记录 |
|---|---:|---|---|---|
| W1 A1 | 4 | 6/6 | 11/1 | 7/1/2 |
| W1 B1 | 6 | 7/5 | 12/0 | 10/0/14 |
| W2 A0 | 6 | 8/4 | 12/0 | 9/0/15 |
| W2 B0 | 5 | 7/5 | 11/1 | 7/1/3 |
| 合计 | **21** | **28/20** | **46/2** | **33/2/34** |

请求共享端点、响应可引用同一观察，不能把重复contrast副本当额外job。两次失败为task43的IEEE fallback端点，消耗预算且不补采；W2 B0虽尝试N16，N128仍budget-unknown。response记录数不是worker调用数，也不是所有轴均被完整测量。

### 四条C2 response→source→joint-config链与推理缺口

| C2机会 | 真实response线索与实际源码动作 | 联合区域/选中点 | 证据缺口 |
|---|---|---|---|
| W1 A1 / task43 | W8将spill16→0却未使该response更优；N64/S1升至132 spills。实现compute-dtype y存储及64+32 head拆分、acc1/acc2 | 两建议为bf16/plain/D1=64下M128/N32/W4/S2与M128/N64/W8/S1；最终M64/N64/D1=64/W4/S2 | prose称每个a端点都是selected，实际M的selected在b端，N请求固定S1。资源下降不自动等于净收益，无per-launch分解证明流量归因 |
| W1 B1 / task21 | GEMM W8将spill108→2；小N低资源并不对应更好整任务。实现RAW1_DTYPE、FP32 DW loads和straight-line masked expand loop | 两建议在fp16/raw1-fp16、GEMM/DW W8下对比stage1/2；最终改为SK1/DWwidth32/GEMM W4/S2 | prose夸大stage成绩相等；相同资源metadata不证明pipeline inactive；没有接受扩域 |
| W2 A0 / task21 | 请求SK1/4改变task response而所报资源不变；映射到raw3 zeroing/atomics/stats成本。实现`_proj_fused`、三处STORE_DTYPE分配、w1f/wpf预转换，移除split-K knob | K16/64在共同fp16/storage-fp16/W8伙伴下被测；最终expanded winner使用新EW1024 | preliminary DW-staging想法不是实际主要动作；bf16 compute失败不能证明storage失败；父结构负向配置也不能自动剪掉新结构区域 |
| W2 B0 / task43 | W8去spill未占优，N64使spill16→50；只改dtype-native y store，保留padded attention/cuBLAS | 单个default建议与selected同为bf16/plain/M128/N32/W4/S2 | 无per-launch计时隔离cast成本，不能从源码出现该动作断言因果贡献 |

这不是把C2泛称“融合优化”：有实际partner、response、source和后续配置对应，也有可指出的推理偏差。结构目标是改善**新结构在自身retune后的可达区域**，不是要求它在父旧参数点获胜。采用已建立，但信息是否独特地导致更好结构仍未证明；G0也能独立profiling并发现相关数据流动作。

## 7. 扩域读取和实际搜索：literal增加不等于有效区域收益

全部5个accepted expansion调用实际读取了selected source、selected TrialRecord/params、numeric stats和raw trials。部分读取是bounded preview；尤其W1 A1、W2 A0不能由一次read推出完整历史已被消费或理解。边际统计仍非固定伙伴因果梯度，原numeric eligibility/cap未改。

| 扩域机会 | 新literal值 | 涉及至少一个新值的trial rows（complete/fail） | 最终结果 |
|---|---|---|---|
| W1 A1 C2 | M16；D1=256 | 11（10/1） | base winner保留；没有新值。HS96时D1=256经clamp与128语义重复，域宽可能被literal夸大 |
| W1 B0 G0 | DWwidth256；FIN_C64 | 6（3/3） | base winner保留；两新值未选中 |
| W2 A0 C2 | N256；SPLIT_M64/128；EW1024；DW/EW warps16 | 21（17/4） | expanded winner选中**EW1024**，其余selected值属于旧域 |
| W2 B0 C2 | M8 | 5（5/0） | base M128 winner保留 |
| W2 B1 G0 | warps16 | 4（3/1） | expanded winner仍为旧域W4/S3，不是W16 |

共 **5 accepted、3 ineligible、0 no-op**；47条new-literal native记录＝38 complete＋9 fail，包含prelaunch拒绝/复用，不是47个额外benchmark。五次中 **3次保留base，2次选expanded winner，只有1次selected使用新增值**。

五次before/after完整AST与defaults相同：W2 B0/B1的source_changed=true仅因注释/header变化、prior-source-match=false，不能叫算法变化；其余字节相同。导出未包含extensionless内容blob时，T6按ref的SHA找到实际导出的Python字节副本核对，没有假装读到缺失文件或修补raw。

W2 A0从base best到selected同时改了EW_BLOCK512→1024、FIN_BLOCK_C16→32、GEMM_STAGES2→3、EW_WARPS8→4。没有同域额外B40 control，新增域、额外搜索/选择、伙伴组合与噪声混在一起，**不能把该pair的+9.934255%归因于EW1024或纯扩域**。W2 B1在expanded study选旧域点同样不能证明warps16有益。

## 8. 自动pair-local调度、预算与真实成本

| 主时钟 | Unix秒 | UTC |
|---|---:|---|
| S | 1789783068.7650688 | 2026-09-19 01:57:48.765069 |
| work cutoff=S+16800 | 1789799868.7650688 | 06:37:48.765069 |
| final=S+18000 | 1789801068.7650688 | 06:57:48.765069 |
| 最后controller结束 | 1789795309.5170684 | 05:21:49.517068 |

主实验共享跨度 **12240.751999616623s＝3h24m00.752s**。机会工作用较早work cutoff准入，heldout用final，身份记录仍统一final；未按host/wave重置。全部在cutoff前结束，drain0，没有延时、补抽、第三wave或optional机制GPU任务；这次没有压力测试过期拒绝行为，不作普遍时限保证。

| Host/wave | pair-ready→heldout runner-start s | Heldout elapsed s |
|---|---:|---:|
| A/1 | 0.9439284801483154 | 333.7537668887526 |
| B/1 | 0.7401893138885498 | 246.13464779034257 |
| A/2 | 0.9283957481384277 | 354.58184985630214 |
| B/2 | 0.7034623622894287 | 355.999731361866 |

0.703–0.944s是**runner启动**延迟，不是首个kernel开始；staging/static/worker setup随后发生。Host B完成本地heldout并推进wave2时，A仍在wave1 expansion。没有全局wave barrier、人工交接或bulk export阻塞pair；不据四卡设置宣称4×加速。

### 逐机会wall（秒；阶段嵌套，不全部相加）

| 机会/arm/task | elapsed | generation agent | acquisition worker | parent / child screen | native retune inclusive | expansion agent子项 |
|---|---:|---:|---:|---|---:|---:|
| 1/A0 G0/43 | 3012.448 | 1091.343 | 0 | 96.905 / 61.974 | 1821.241 | 0 |
| 1/A1 C2/43 | 6233.964 | 1170.394 | 383.619 | 97.121 / 85.884 | 4579.628 | 265.623 |
| 1/B0 G0/21 | 4291.060 | 1648.528 | 0 | 61.302 / 100.353 | 2575.456 | 136.104 |
| 1/B1 C2/21 | 2727.107 | 985.251 | 201.592 | 59.753 / 52.897 | 1474.027 | 0 |
| 2/A0 C2/21 | 5307.449 | 1363.454 | 339.355 | 123.698 / 68.925 | 3477.406 | 202.596 |
| 2/A1 G0/21 | 3446.261 | 1837.424 | 0 | 124.782 / 103.935 | 1480.691 | 0 |
| 2/B0 C2/43 | 4834.861 | 601.313 | 335.405 | 63.107 / 92.214 | 3828.727 | 208.687 |
| 2/B1 G0/43 | 4968.587 | 692.186 | 0 | 58.606 / 59.071 | 4212.068 | 160.879 |

elapsed包含模型、编译、私有工作、调度/等待；retune包括expanded模型和child screen，generation agent wall也含工具执行/等待。G0 acquisition0、未扩域agent0与confirmation0是实际无调用；其他未知成本不能补零。各臂elapsed不是独立部署counterfactual，跨GPU并行和host争用未被隔离。

| 私有/模型账 | 已记录值 | 不可作的合并 |
|---|---:|---|
| 20 helper result wall之和 | 766.350109583s，其中invalid139.449804345s | 已包含于CLI区间和agent elapsed，不额外加到main span |
| 27 helper CLI tool区间之和 | 777.217s，pre-worker错误2.623s | 不是27个benchmark的总GPU time，也不再加result wall |
| 26 ad-hoc执行命令区间 | 108.514s | 含启动/编译/检查；可跨GPU重叠，非GPU busy |
| Named AGENT_CALL_FINISHED cost之和 | **1.07155984** | 较窄的named-call字段账，不是whole-session spend |
| Assistant metadata cost之和 | **8.93249084** | 另一套账，与上一行**不可相加**；币种、invoice和隐含重试账单未知 |

有 **323个distinct assistant-response metadata records**（356条messages内），不是已证明的323个provider HTTP请求。29 named calls与33 attempts又是不同计数层。Token/cost字段存在不意味着provider侧全部请求或租卡账单已闭合，缺失仍unknown。

Readiness gate、setup/发布/部署在S前；主实验后核验/export/分析另计。T5 export在两条科学chain完成后约05:31:12创建，未延迟heldout。05:27:06–08 cleanup检查四GPU均0MiB/0%、无owned进程/服务、相关端口释放；没有杀无关进程。记录无late framework admission/drain，formal helper已传cutoff的提交也受守卫，但**任意已准入agent turn内的ad-hoc命令没有普遍拦截保证**。

## 9. 可复核性、限制与最终判断

数值lane核对4322条导出hash、8组initial primary/reference、8份实测native winner、36份冻结heldout source/params及84份科学full materialization。原始sample数组完整，report medians与record一致；五位小数sample重算最大差0.000004253997802905474ms，小于0.000005000001ms容差，不改变gate。Uptake lane按真实tool→result→source/params→worker链、完整配置TRIAL_DONE及expansion read记录核验，而非只看exit或叙述。

本报告只公布汇总、必要配置/空间标识与可复核定位，不发布原始源码产物、完整会话、数据库、provider配置或secret。以下`results/`和`.omo`是**本地证据位置，不是已公开数据集**；有相应allowlisted导出副本时可按路径与manifest复核。公开读者无需raw即可读取全部36块和四pair结果。

| 复核内容 | 本地证据定位 |
|---|---|
| 精确数字/范围/归一化/selected params与成本 | `results/c2-helper-pilot/analysis.json`；`.omo/evidence/v5-c2-helper-pilot/T6-analysis-usage.md` |
| helper/私有命令/推荐/请求/扩域读取链接 | `results/c2-helper-pilot/usage-audit.json`；`.omo/evidence/v5-c2-helper-pilot/T6-uptake.md` |
| 四pair raw | `results/c2-helper-pilot/main/{A,B}/heldout/wave{1,2}/{result.json,events.jsonl,jobs/}` |
| 八机会及13spaces | `main/HOST/waveN/SLOT/information/retune/{result.json,events.jsonl,candidates/,report/selected.py}`（相对上述pilot根） |
| 条件response/前置假设 | 对应`information/acquisition/{preliminary-report.json,responses.json}`；generation events及sandbox输入 |
| 实际tools与messages | `main/HOST/_agent-evidence/waveN-SLOT/`；audit JSON保留精确ID、结果目录与配置链接 |
| 时钟、调度、完整性 | `main/{clock.json,campaign.json,execution-summary.json,export-manifest-A.json,export-manifest-B.json}`及各host `controller.json`；T1–T5本地回执 |

Raw根映射为本地`results/c2-helper-pilot/main/{A,B}`对应远端`/root/autodl-tmp/c2-helper-pilot/main/{A,B}`，不把远端绝对路径当本地路径。导出排除了数据库、缓存、provider body/凭据和无关完整会话；read预览/长输出可能有界，不能据一次bounded read声称消化全部历史。源码和精确公开协议通过第1节的不可变GitHub链接可追溯。

**最终诊断：** 自动pair-local调度完成了全部主比较；formal helper的真实主实验采用、推荐覆盖和事实输入读取已建立，并观察到测试反馈修复源码的实际链条。然而，信息/方法package在强G0下仅1/4通过：**采用成立，竞争力门槛不成立**。C2链中仍有固定伙伴识别、资源与性能关系、active branch、目标区域和新代价判断的缺口，不能用“接线正常”或“agent测试了”替代全任务heldout。

局限包括两个已知benchmark、每任务两次生成、provider随机性未完全控制、不同域/不同实际study数(G0 6/C2 7)、没有helper-OFF/推荐-OFF或同域额外B40消融、私有测试协议仍不统一。因此既不宣传普遍正效应，也不推断C2普遍无效或某compiler/cache是根因。本轮不与旧结果pool，不把旧INCONCLUSIVE改成complete，不以调阈值/择优块/预算变化制造阳性。

至此停止本次实验与owned服务；不自动启动下一轮或追加测量。公开报告后的独立F1尚待进行，结果与发布回执由父任务保留。若未来另行授权研究，问题应围绕上述证据链的具体推理与配置覆盖质量，而不是把失败转化成无限试验或泛化防护平台。
