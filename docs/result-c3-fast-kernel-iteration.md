# C3 fast-kernel iteration：有界开发失败与未启动的正式验证

**NO_DEV_GO；formal / heldout NOT_RUN；efficacy NOT_ESTABLISHED。** 两次全新agent probe均以480秒transport timeout结束，局部证据未满足CUDA trace准入；没有官方模型目标值、选定F*、正式S、三goal winners或36块矩阵。行政闭合不是科学目标完成。

本报告归档已结束的开发过程及限制。成功的人工setup control、成功的同产物诊断重放和两次失败的新生成是不同证据，不能合并为一次成功agent优化。最终一次紧凑独立核验尚待单独交付；本报告不授权新实验、修订窗口或补测。

## 1. 目标、输入与公开源码

本轮仍为C3-B manual-task-first，目标是让agent编写真实Triton/CUDA计算kernel，并证明compiled、trace-confirmed launched、output-used和数值正确，之后才讨论完整Qwen任务收益。Python dispatch重写、静态声明或局部速度本身均不满足该目标。C3-A自动任务构建仍未执行。

| 阶段 | 精确公开revision | 用途及边界 |
|---|---|---|
| 执行前协议 | [54ca2a98c6a34fd5a4727d518e1a253ecf53c43e](https://github.com/Fudan-SMI-lab/opop/commit/54ca2a98c6a34fd5a4727d518e1a253ecf53c43e) | 新tests、产品修改和GPU前归档 |
| Agent/device foundations | [ad0694acdadd9d272783773055d6c93818f472bf](https://github.com/Fudan-SMI-lab/opop/commit/ad0694acdadd9d272783773055d6c93818f472bf) | 显式profiles、phase输入与device/helper；初次setup在此遇到IR归属失败 |
| 有界开发入口 | [092e09febf91169dde0d2fe1727f90d85ec2a580](https://github.com/Fudan-SMI-lab/opop/commit/092e09febf91169dde0d2fe1727f90d85ec2a580) | 一次epoch后STOP、持久D/配额；不是完整正式矩阵交付 |
| IR修复、setup复核与F0 | [996318df6c7e2e9fcdfac3a8012d4e813b7e732b](https://github.com/Fudan-SMI-lab/opop/commit/996318df6c7e2e9fcdfac3a8012d4e813b7e732b) | 修正实际compiled IR参数归属；未改人工control算法 |
| Owner-thread修复、诊断与F1 | [38e6e10de2e9770692f39ebb8eed86e11eb33df5](https://github.com/Fudan-SMI-lab/opop/commit/38e6e10de2e9770692f39ebb8eed86e11eb33df5) | 仅fast_generation及线程/RPC回归；GPU callback由resident owner执行，gate不放宽 |

F0 framework ID为`36c96bb19771aa171ac3c0c94849c2634c2383f345ed722011a5c482408b38be`；F1为`d278370b2ff8b9c0ec551dbc4b21c2d873b0f9f5ef755f3fdcfa182dc6c21b67`。F1的schema、prompt和agent-config hashes与F0相同，仅实现/revision改变。这里F1指第二个开发framework，不是最终独立review。

模型固定`Qwen/Qwen3-4B@1cfa9a7208912126459214e8b04321603b3df60c`，BF16/eager/SDPA；A使用kernel-opt-venv，B使用orch-venv，沿既有Torch2.13.0+cu129、Triton3.7.1及Transformers5.16.1准备证据。没有下载新模型或降低质量规则。

新contract SHA256为`2c0e2fc70c76d4b780cb1301addcf2d723ec5f19f106af7ac3919fbc86996bd8`。开发复用明确声明的16条calibration/search记录；42条新heldout于候选前另行冻结，与旧58条不交，开发/本次分析未读取sealed内容。正式阶段没启动，因此这些新heldout没有模型测量结果。

## 2. 实际执行：control、诊断和新probe不能混同

### 人工setup control：验证harness，不是研究winner

初始ad0694a control已有编译、launch-counter和数值匹配，但IR输入/输出归属为空，gate正确拒绝并保留失败charge。996318d以同一control复核，IR、CUDA trace、正常数值、output participation和skip/suppression证明通过；wrong control被拒绝，三次模型A/A/control检查通过。

这证明该人工control在该次setup环境下的harness路线可工作，不证明agent生成成功、通用kernel证明能力或优化收益。三个setup profile窗口是诊断，不是三个goal分数。父级修改的是框架接缝；未把手写control或修好的候选算法当作agent答案。

### Fresh F0：有部分native证据，但trace为空且调用超时

996318d在setup通过后复用resident，从原baseline发一个GLM5.3/max primary；480秒timeout、generic/transport retry均0。开发goal为TTFT；既有native profile选择module33 post-attention RMSNorm并形成73个兼容RMSNorm leaves的canonical组，一份4096-token representative。未扩展成decoder wrapper或给agent预写优化答案。

Target site为`pure-727f184a931d6918df621c6385fe123d7dd248cb470c207fd46b715a8c7a1fa0`。关联CUDA工作记录14041.134us，但同名/重叠归属不明，exclusive fraction保持unknown；参考A/A噪声为0.192001462us，不能据关联时长编造Amdahl收益。

| Fresh probe局部观察 | F0（996318d） | F1（38e6e10） |
|---|---:|---:|
| entry / compiled | rmsnorm_kernel / true | rmsnorm_kernel / true |
| JIT / launch-counted calls | 2 / 1 | 2 / 1 |
| compile wall ms | 156.21 | 162.7 |
| registers / shared bytes | 35 / 32 | 33 / 32 |
| output-used；IR output / input args | true；[0] / [1,2] | true；[2] / [0,1] |
| 正常局部数值 / skip rejected | true / true | true / true |
| **CUDA profiler names** | **[]** | **[]** |
| local admission / latency | invalid / null | invalid / null |
| official model J | null | null |

两者均报`declared device kernel was not compiled and CUDA-trace-confirmed launched`。上表compiled=true说明这是组合gate错误文本；不能把它简写成“没有编译”，也不能把launch counter当trace-confirmed PASS。F0的helper在另一线程采集；其primary随后ReadTimeout，无完成返回artifact，summary为PARENT_TRIAGE/transport_failed。独立helper证据仍保留，即使候选summary的locals为空。

### Owner-thread诊断PASS，不等于fresh F1已恢复

38e6e10把整个agent invocation放到带复制ContextVars的transport线程，resident创建线程泵送helper callbacks；共享C2/default和device gate未改。一次**原F0不可变产物**经实际HelperService RPC及owner pump重放：owner/callback TID776849，socket helper776937，diagnostic transport776938；CUDA/IR/output/数值/suppression证明通过，局部latency为89.088us，`official_model_score=null`。

该89.088us仅属诊断local报告，不是端到端J、比较增益或新候选结果；F0没有被追认晋级。之后F1使用新session、新source、原baseline和相同target/goal，而非把F0重放结果作为起点。

Fresh F1的profiler实际在owner PID:TID **776849:776849**运行，仍返回`cuda_names=[]`。它同样480秒ReadTimeout、PARENT_TRIAGE/transport_failed，无repair/redraw、完成返回artifact或模型评价。因而“把callback移到owner线程就彻底解决问题”被真实F1结果否定。

**实际generation生命周期下剩余的CUDA collection/lifecycle/correlation根因尚未确定；两个primary未在480秒内完成的原因，以及局部反馈失败与transport timeout之间的因果关系，也未确定。** 没有证据证明改一个timeout、再换线程或修改kernel算法就会成功；不能把当前失败认定为已证明的数值kernel缺陷。

## 3. 源码与bundle身份

| Artifact | Source SHA256 | Bundle SHA256 |
|---|---|---|
| Fresh F0及其后续原样诊断重放 | `b6b0b447aef27031d0954de3595811e46f405ae81a8e1f8d9885425deefafedf` | `aba30792ba8a9fb488259ba26cac52a619f10df6fa837b39326738b7fb4c9c71` |
| Fresh F1 | `08363d2cb383451d5ffbd98a79410a1c5c5cfa8edfa766b3c25d0d0d19fbbdcb` | `68a03a7c2df2d6fff4cbd2ea41cde58f78b1aa98336d1621ae5203e5987e6321` |

F0诊断使用num_warps8，params hash`b342691f40029e88b20fb28828198d9741a332a75a484d422a005db7dc99d705`；F1为空params，hash`44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a`。失败workspace和helper-admitted snapshots保持原件；框架revision、候选identity与诊断用途分开，不pool成一个成功epoch。

## 4. 原始D与实际配额

**D=2026-09-20T10:30:17.689003Z /1789900217.689003；cutoff=12:00:17.689003Z /1789905617.689003。** 连续90分钟从工具/模型/control已就绪后起算，重载、框架修订、CPU测试、诊断、父级/操作等待均消耗D；没有暂停、重置或延期。

| 开发组成 | Agent charges | Local charges | Model-evaluation charges |
|---|---:|---:|---:|
| Fresh F0：capture＋reference A/A＋helper | 1 | 3 | 0 |
| 原F0诊断：fresh capture＋helper replay | 0 | 2 | 0 |
| Fresh F1：capture＋reference A/A＋helper | 1 | 3 | 0 |
| **合计** | **2** | **8** | **0** |

Local8包含**5 local operations＋3 captures**，不是8再加3；开发profile0。Model-evaluation0不是“无GPU/整模工作”：capture需要模型forward，F1进程单列2 capture-related forwards，局部kernel也实际执行。未使用的数量额度不授权F2、新D或免费setup。

Setup独立累计3 local /3 model /3 profiles，另有4 capture/preparation operations；首次失败local没有退回。初始setup261与recheck275 model forwards属于各自receipt，不算开发model J或可复用candidate分数。

F1执行11:42:43.787504–11:50:45.014001Z，481.226497s包含外围工作，transport上限480s，drain0。进程于11:50:45.545898Z关闭时距cutoff572.143s，稍后essential snapshot记录剩500.868s；二者是不同观察时刻，不是时钟变化。没有用剩余计数启动F2或正式阶段。

框架准备/实现始于D之前：新工作跟踪激活为05:33:08.8110247Z，输入冻结为05:44:58.193583Z；后续实现、发布、部署、setup及等待另属工程周转。**不能声称整个工作在90分钟内完成**；D也不是GPU busy、各agent时间之和或总人工工时。本报告写作/发布发生在开发之后。

## 5. C2隔离与输入边界：有限比较，不是历史通用等价

显式execution profile与goal/objective分开。T1曾发现cda1130-direct和8526a13-generic的effective prompt相同，但完整schema/seed不同；不能仅凭正文不变称兼容。T2在同一有限CPU fixture、recording-provider边界得到以下规范化结果：

| 当前route | 精确声明anchor | Effective prompt /完整schema /全部seed /whole request |
|---|---|---|
| `c2_direct_compat` | `cda113070c8ba3a91f480f42b68d196187796cda` direct | 全部equal |
| default `existing_generic` | `8526a1390c78346af1a5be112685b7fbe4defa59` generic | 全部equal |
| legacy StructureRewriter | 同8526a13的legacy | 全部equal |

比较保留实际字段/类型，仅规范化记录的非确定路径/IDs。保留的immutable-parent修复`82daf15b460004707e55dbdebb789e2788921527`相对CDA是明确的validation行为变化；model-facing equality不等于所有验证逻辑bit-for-bit相同、随机LLM输出相同或历史性能复现。每次旧C2实验仍属于自己的真实SHA，不统一改标cda1130。

C3采用`model_operator`/`model_project_operator`专用合同；新device声明只作metadata，local adapter必须实际证明执行。F1 owner-thread修改局限fast-generation路径。CPU兼容/回归证明软件路由边界，不证明未运行的模型结果；本报告仅引用已有证据，不重复新测试或旧C2 GPU实验。

## 6. 结论、停止与证据可用性

1. **Setup有条件通过，真实agent开发未通过。** 人工control和原F0诊断PASS有各自用途，不能替代两次fresh生成的完整device准入、model quality和正式选择。
2. **NO_DEV_GO。** 没有F*或正式S，三goal新生成与36块heldout均NOT_RUN，不是零分、负收益或baseline winners。方法效力NOT_ESTABLISHED；旧Python-wrapper增益不是本轮device kernel证据。
3. **框架问题仍未解决。** 空CUDA trace在owner-thread fresh F1重现，残余原因与timeout因果未知。保存失败及partial evidence，不放宽gate、不追加预算、不人工修候选迎合成功。
4. **有界闭合，不伪装科学完成。** T1–T6行政收尾包括T5条件分支未启动；没有新experiment或额外review infrastructure。下一步仅既定一次compact independent review，未获该review批准的状态不写成已通过。

历史cleanup记录runner已关闭、PID776849不在、无owned runtime survivors，port42710可绑定，A卡0MiB/无compute应用。GPU快照另有A0瞬时10%利用率，不能把它改写成全零利用率，也不能单凭它推断有owned残留。本报告未重新SSH查询或宣称当前设备预留。

主要证据是本地`.omo/evidence/v5-c3-fast-kernel-iteration/`下的`T1-inputs.md`、`T1-compatibility.md`、`T2-agent.md`、`T3-native-first.md`、`T3-native-recheck.md`、`T3-F0.md`、`T4-helper-owner-thread.md`、`T4-native-F1.md`及父级核验的`T6.md`。它们包含命令、计数和更细的阶段边界；本文已内嵌主要结果及精确源码/框架/候选身份，不要求公开读者取得raw才能理解结论。

本地essential raw在`results/c3-fast-kernel-iteration/setup-native-recheck-996318d/host-A/`及`T4-native-38e6e10/{terminal.json,F1-essential.json,owner-rpc-diagnostic.json,gpu-after.txt}`；完整原件仍在既有远端run目录。初次bulk profile拷贝并未全部核验，不能冒充完整公开profile dataset。本报告基于已核验receipt和essential记录，不重新做全session/source审计。

**这些raw/.omo材料不是公开数据集**；本次只公开报告，未发布完整会话、provider配置、数据库、凭据或模型。既有raw、失败、时间、旧计划和历史报告全部保留；没有新测试、模型/GPU调用、性能测量或科学数值推断。
