# Task21配置诊断：能追回部分差距，K加宽不是主要解法

**结论：独立核验PASS；有限配置改善成立，但尚未追回G0。** C的BLOCK_K16→32/64仅改善0.258068%/0.460605%；预声明联合点T_SK1改善8.359685%，仍比新测G慢39.061188%。它追回原差距24.514684%，剩1.431552ms，尚需相对自身再降28.089209%才能追平G。

这不是新候选campaign、正式winner更新或C2效力阳性。18个固定full全部保留、无自适应补点；六点中最快C-source配置仍是探索性诊断点，**没有挑选后的独立heldout**。不改旧报告、旧deadline或旧1/4结果，也不把不同批次2/4与1/4作因果比较。最终F1审阅已APPROVE，限于本诊断证据与表述，不是C2效力通过。

## 1. 冻结协议与六个比较

协议先归档于[69d74a307992f5e58c0ff6e74307d141e1c85a12](https://github.com/Fudan-SMI-lab/opop/blob/69d74a307992f5e58c0ff6e74307d141e1c85a12/docs/plan-c2-task21-config-diagnostic.md)；正式evaluator保持[cda113070c8ba3a91f480f42b68d196187796cda](https://github.com/Fudan-SMI-lab/opop/commit/cda113070c8ba3a91f480f42b68d196187796cda)。HOST-B GPU0串行执行，不用历史quick当当前控制。

| Cell | 固定结构/配置 | 可以回答什么 |
|---|---|---|
| G | G0 rep0 selected，14个完整参数 | 新测同卡参照 |
| C | C2 rep0 selected，19个完整参数 | 新测C2基准；已经FP16 compute/raw1、SK1 |
| K32 | C只改BLOCK_K=32 | 所选伙伴下projection K16→32 |
| K64 | C只改BLOCK_K=64 | 所选伙伴下projection K16→64 |
| T_SK1 | 历史trial完整T参数，只改SK1 | 另一个预声明联合点；与C是多变量比较 |
| T_SK4 | 历史T参数原样，SK4 | 与T_SK1构成新测、真正匹配的SK对照 |

T为`tr-5dfdde92 / cand-3a36b550 / sp-f3f4954d`；T_SK4物化hash重现该历史trial，但不用其旧分数。C→T_SK1同时改M32→64、SPLIT_M16→32、STATS_M64→256、DW_W32→128、EW1024→512、GEMM_W4→8、STAGES2→1；其中STATS项在SK1路径不活跃。

固定三轮：`G,C,K32,K64,T_SK1,T_SK4`；反序`T_SK4,T_SK1,K64,K32,C,G`；再`K32,C,G,K64,T_SK4,T_SK1`。每cell一次/轮，不宣称完美位置平衡；K64不以K32结果决定是否运行。

## 2. 全部18块、范围与质量

单位ms；M为三块median的中位数，不是最快块。数值表显示9位小数，未舍入数据在analysis.json。

| Cell | Block1 | Block2 | Block3 | M | min / max |
|---|---:|---:|---:|---:|---|
| G | 3.664896011 | 3.668032050 | 3.664896011 | 3.664896011 | 3.664896011 / 3.668032050 |
| C | 5.561344147 | 5.564415932 | 5.561360121 | 5.561360121 | 5.561344147 / 5.564415932 |
| K32 | 5.545983791 | 5.547008038 | 5.550176144 | 5.547008038 | 5.545983791 / 5.550176144 |
| K64 | 5.535744190 | 5.531584024 | 5.539824009 | 5.535744190 | 5.531584024 / 5.539824009 |
| T_SK1 | 5.098495960 | 5.096447945 | 5.093343973 | 5.096447945 | 5.093343973 / 5.098495960 |
| T_SK4 | 6.282752037 | 6.284255981 | 6.288383961 | 6.284255981 | 6.282752037 / 6.288383961 |

18 full＋18 static＝36 worker submissions，1800性能样本；每full为100样本/5 correctness、seed0、fp32 reference及原dual-witness-relaxed/FP64 multipliers2/3。**90/90 accepted correctness全部依赖授权FP64 rescue**，不是strict-allclose或仅primary-relaxed通过。全部compiled/correct/formal OK，无替换或缺块。

独立核验303条限定export hashes、六组完整参数与物化hash、源码除PARAMS外AST相同；所有worker source/reference/参数/模式和冻结顺序对应。原始样本重算median最大误差0.000004190216064792196ms，在五位小数导出容差0.0000051ms内；没有重扫旧4322条或重审历史cohort。

## 3. 哪些差距被追回，哪些没有

本次fresh差距 `D=M_C−M_G=1.8964641094207764ms`。以下都是**由现有测量计算的描述量**，不是新增科学胜利：

```text
相对C改善 = 100*(1-M_X/M_C)
相对G慢多少 = 100*(M_X/M_G-1)
原差距追回比例 = 100*(M_C-M_X)/(M_C-M_G)
还需下降才能追平G = 100*(1-M_G/M_X)
```

| 配置 | 相对C改善% | 比G慢% | 原差距追回% | 剩余gap ms | 尚需相对自身再降% |
|---|---:|---:|---:|---:|---:|
| C | 0.000000 | 51.746737 | 0.000000 | 1.896464109 | 34.100725 |
| K32 | 0.258068 | 51.355128 | 0.756781 | 1.882112026 | 33.930220 |
| K64 | 0.460605 | 51.047783 | 1.350721 | 1.870848179 | 33.795785 |
| T_SK1 | **8.359685** | **39.061188** | **24.514684** | **1.431551933** | **28.089209** |
| T_SK4 | -12.998544 | 71.471604 | -38.118088 | 2.619359970 | 41.681306 |

匹配的T_SK4→T_SK1降低 **18.9013312047008%**；它是T伙伴下的SK条件效应。**C本来已经SK1**，不能把T_SK1相对C的8.359685%称为“打开SK1的收益”。T_SK1还留原gap的75.485316%；39.06% slower和28.09% further reduction分母不同，不矛盾。

观察支持“冻结C结构存在更好的有限配置”，不支持“已找全局最优”或“多给TPE预算就一定补齐”。K-only两个点虽范围低于C，幅度均不足0.5%，只追回原gap的0.76–1.35%；没有主要解决问题。未在这些诊断点上新设论文gate或显著性标准。

## 4. 两份真实profile：定位expansion/projection，不能精确分摊formal差距

仅在formal18结束后，各对selected G/C作3次同步warmup＋1次instrumented forward；每profile进程约12.43/12.48s、限60s，无额外点、安装或retry。实际trace kernel事件与`cuda-events.json`一致：G15、C13。

| Source/kernel组 | G us | C us | C−G us |
|---|---:|---:|---:|
| expansion：`_expand_gemm` / `_gemm_fullk_atomic` | 1242.602 | 2731.439 | 1488.837 |
| DW：`_dw_atomic` | 2089.788 | 2098.202 | 8.414 |
| projection：`_proj_gemm` / `_proj_splitk` | 510.262 | 901.872 | 391.610 |
| final：`_apply_buf` | 130.302 | 175.101 | 44.799 |
| 三次running updates合计 | 4.768 | 4.799 | 0.031 |
| 六个ATen fills合计 | 6.240 | 6.751 | 0.511 |
| FP16 weight-copy | 3.680（2次） | 无 | -3.680 |
| kernel duration合计 | 3987.642 | 5918.164 | 1930.522 |

初始state hash均为`e1f4b31e62998276ca69d8a7ca40c6518881ad1976f416d2fe89c0c10b6b63a1`，input hash均为`61cbe509056898d0f718f3ef3a0c371d63c9bf6dd66f3306c1b7b71ff8128291`；source/reference/config对应所选G/C，shape10×112×224×224、TRAIN、seed0一致，初始reference/candidate状态逐tensor相等。

**这不是formal的L2-flushed逐sample条件**：没有重放5 correctness，running buffers已受3 warmups更新，又有profiler扰动。两个sum之差1.930522ms不精确等于formal gap1.896464109ms；不能把“其中百分之多少由精度/存储造成”算成因果账。只profile了G和selected C，不能替T_SK1解释其内部加速。

CPU/ATen事件保留copy/fill/allocation；同起始分配225703424B下，peak allocated为G1214704128B、C1431580160B，**不是实测DRAM traffic**。stderr还有transitive LiteLLM cost-map加载超时后本地fallback及Kineto消息；是metadata/startup，不是模型推理，已含profile进程耗时，不隐去或追加安装重试。

## 5. 源码说明为什么“K加大/全部FP16”不是现成答案

- 所有C派生配置已是 **COMPUTE_DTYPE=fp16、RAW1_DTYPE=fp16**。`sources/C.py:369,391–394`的raw2/raw3明确hardcoded FP32，无相应可选参数；K32/K64/T只改PARAMS，没有改storage body。**本次未测试raw2/raw3压缩，不能保证all-FP16会赢。**
- C的`BLOCK_K`只传入projection K-loop；expansion用host推导的`K_PAD`（`C.py:336–343,396–403`）。因此只调BK不能直接处理profile中较大的expansion差异。BLOCK_M/N、GEMM_WARPS/STAGES同时传给两个dot阶段，是耦合控制，不是独立projection旋钮。
- G `_expand_gemm`（`G.py:64–93`）按M-tile发program，A加载在N-block内循环之外；C `_gemm_fullk_atomic`（`C.py:69–107`）按SPLIT_M×N-block作persistent M循环，复用/并行度/stat-atomic组织不同。G另有w1/wp预转换。结构、调度、M与伙伴并不匹配，trace定位不等于单因素解释。
- G raw1/2/3由STORE_DTYPE选中fp16，C只有raw1为fp16；但这份trace里DW约2.09ms、几乎相等，即便其输出storage不同。不能据“G多压缩raw2”就断言DW带宽是全部根因；projection读取、存储和dot伙伴也都改变。两者expansion均写FP16 raw1仍有大差异，尤其不宜仅用后续raw2/3解释它。

## 6. 面向用户的决定与G0比较范围

**现在应停止继续盲开候选campaign，不承诺prompt格式或扩大搜索预算能解决差距。** 手头最强证据是：K单轴收益很小；联合点能改善但只追回约四分之一；selected点trace把大差异定位到expansion/projection。若之后另行授权工程调查，候选级expansion实现及M/N/warp/stage/SPLIT_M耦合有具体机制动机，但这不是已证明的fix，更不是C2独有收益。此任务不实施它，不增加profile/storage/candidate作业。

[G0 overlap audit](../.omo/notepads/v5-c2-helper-pilot/g0-overlap-audit.md)已核对当前G0共享whole-task/伙伴指导、selected上下文、推荐配置、扩域证据/helper及自主工具；**未观察到fresh C2 response泄漏**。因此旧1/4负结果仍是强共享支持下“增量targeted-conditional workflow/package”的有效比较，不是C2全部概念与一个无能力baseline的比较。本次不重做该工具审计。

若未来要测更广决策package，可事先单独命名generic-agent arm，声明匹配的预算、工具和质量；不得在看到负结果后削弱/替换仍叫G0，或假定换弱baseline就会成功。旧2/4、旧1/4和本诊断不作跨批次因果比较。

## 7. 时间、身份与交付状态

S=`1789806364.916481`（08:26:04.916481 UTC），截止=`1789809964.916481`（S+3600）；formal结束08:37:41.323575，**696.4070937633514s**。到profile phase结束08:40:13.263765，**848.3472764492035s**，含formal→profile的operator转换/准备间隙，不是GPU busy。setup及后续export/笔记另计；无延时准入或drain。

G/C冻结源SHA分别为`fdd95ca58e696ba081a0f742c82fd49a600952ad88e29f3c7e83faeb56ce1f26`、`c4982b91b5d0456c409ccdfda193b1b6931436f07862939702df404c471c3849`；reference为`24e555726c4856120cceaa8dad2dbbca798a56c0a99313c77fbe84c16e37c20b`。完整六组参数/物化hash、历史T对应及全部数组见[analysis.json](../results/c2-task21-config-diagnostic/analysis.json)。

Raw根`results/c2-task21-config-diagnostic/`中`runs/`、`jobs/`、`profiles/{G,C}/{trace.json,cuda-events.json,key-averages.json,result.json}`可复核；[T2执行](../.omo/evidence/v5-c2-task21-config-diagnostic/T2-execution.md)、[T3独立核验](../.omo/evidence/v5-c2-task21-config-diagnostic/T3-analysis.md)记录边界。均为本地证据，不是已公开数据集；[F1回执](../.omo/evidence/v5-c2-task21-config-diagnostic/F1-verification.md)记录审阅范围，精确发布修订与报告URL见[publication回执](../.omo/evidence/v5-c2-task21-config-diagnostic/publication.md)。

本轮到此停止：结果未用于更新官方C2 incumbent，没有独立postselection验证、没有存储因果消融、没有泛化或显著性结论。F1已APPROVE本次有限诊断；按发布回执收尾，不自动续跑。
