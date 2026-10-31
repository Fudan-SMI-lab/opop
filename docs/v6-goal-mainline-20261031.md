# v6 通用优化主线：持续目标

用户于2026-10-31明确要求使用 `/goal` 持续推进。当前 Goal 已激活；一次路线关闭、拒绝或辅助工作完成不代表整个项目完成。

目标是继承 C1 资源画像、C2 资源响应引导改写、C3 完整项目验收，形成从原项目、入口和目标出发的通用端到端优化能力。以原任务数值正确性、实际执行路径、原配对性能接受规则、反馈进入下一轮为依据，区分自主结果与开发者辅助。

先复用已有跨项目证据，仅补实际缺口。2026-10-27三项目原入口批次已跑通，但它的核心版本及之后的辅助实验不能直接证明当前核心的同等自主效果。当前先补一个非 DDP 原入口项目；不整批重跑旧实验，也不把单个 DDP 的持续压分当作主目标。

## 当前执行

当前同一生产核心e893的三个原入口有限两轮均已结束：ResNet实际折叠优化通过正式和独立验收；DDP两轮回退且性能拒绝；Qwen第二轮被现有规则正式接受，但实际全部回退，不能记作声明优化的收益。三项目均核实了真实反馈进入第二轮，未追加第三轮。最新完成时间为2026-10-31 18:51:10 UTC／香港11月1日02:51:10。主线剩余问题是如何处理“全部回退仍获接受”的通用验收缺口；用户明确禁止改变接受规则，已提出具体授权问题，尚未答复，生产规则保持。Goal仍active，不将这个阶段或辅助记录当成整个项目完成。

- **项目**：ResNet 原始离线 batch-32 推理；保留预训练模型、真实图片、FP32、完整 logits 和精确类别。目标沿用旧原入口批次的完整推理延迟目标。
- **实际启动**：香港2026-10-31 22:46:12，box4-3，controller `329695`。根目录 `/root/autodl-tmp/dev/v6-work/goal-resnet-original-20261031`。
- **入口**：现有生产 `optimize-project --project … --command … --goal … --devices 0 --prepare-only`。没有提供旧 request、adapter、候选、人工算法或审计答案。生产核心仍为 `e893f850…`。
- **有限后续**：检查实际生成的任务、评测及原源码语义；准备成功且容量满足后，进入普通 discovery 和两轮优化。结果按原规则接受或拒绝，不重试到通过；仅实际暴露的通用缺陷才考虑最小修正，生产 prompt 修改仍需明确授权。
- **启动核实**：22:46:56 controller 存活，实际 intake snapshot `7fec5ffc4745` 已生成；尚未生成评测制品，GPU 空闲属于角色生成阶段。此记录不是准备通过或优化接受。
- **自动续接**：实际 waiting 预检后只启动一个隐藏完成监听 `49068`，同一 Goal 保持 active；完成或低盘注意事件续接同一线程，不重启旧批次。后续仍须区分准备完成和优化接受。

### 准备完成与优化启动

准备于22:52:40完成，adapter `8d656e90…`；原reset/identity/错误输出/缺失覆盖控制通过，五个完整推理样本为4.012611、3.994588、4.008986、3.988195、3.991729ms，中位数 **3.994588ms**。实际 adapter 保留原始 `images` 和 batch `pixels`，FP32/noTF32；框架计时保留两端同步，输出生命周期按原 invocation 处理。`observe` 的重复前向在计时之外。

22:55对新controls与已有原源码参考作只读CPU比较：两次各32,000个logits最大绝对误差0、32个类别精确相同；4,816,896个输入像素精确一致。没有新模型前向/GPU/原源码重跑或参考复制。这是开发者对实际字段的核验，不称通用自动语义证明；核验结果未送入优化角色。角色把原offline内存目标的吞吐≥0.95baseline约束带入当前latency目标，保留原输出未改。该约束实际使首轮在screen阶段被拒绝，因此没有无此约束时的配对结果；不能声称它对搜索过程没有影响。

优化根目录 `/root/autodl-tmp/dev/v6-work/goal-resnet-optimization-20261031`，22:56:36启动 controller **331628**，执行现有生产CLI `--request 原request --resume --rounds 2`，继续同一run。核心仍为e893，未指定算法、分组或候选，也未提供上面的核验答案。22:58:57实际discovery worker通过，current `c68d8902…`，普通planner snapshot `5f1faa72f1c2`已出现，尚无trial结果，不构成优化接受。

准备完成事件49068已标记handled，迟到事件去重。新优化只设一个隐藏监听 **32596**。两轮结束后评估实际正确性、收益、执行路径与反馈消费，再决定有依据的下一步；不自动追加第三轮或修复直到通过。

### 已发生的反馈迭代

第一轮 `trial-2a2fef5824` 自主选择 channels_last＋cuDNN autotune。实际代码每次执行完整前向，trace出现NHWC卷积/BN及布局转换；不是缓存输出。两次完整质量检查各32,037元素，最大误差9.059906e-6，类别通过。五个screen样本中位数 **4.369144ms**，吞吐 **7324.089** 低于合同下限 **7610.297**，故 `no_valid_parameter_candidate` 拒绝；仅一次空参数观测，没有正式ABBA，未调参重试。旧基线3.994588ms与本次screen不是正式配对收益。

第二轮 selector `1b3399f80167` 实际读到了拒绝原因、上述耗时/吞吐/阈值及质量数据，最终理由明确引用这些反馈，转向 Conv–BatchNorm 折叠。它额外自主执行了CPU完整模型探测：早期属性错误、误折叠最大误差332.715及类别失败均保留，后续修正探测得到20对折叠、最大误差7.629395e-6和精确类别。这不是GPU验证，也不是开发者提供优化代码。跨profile推导的0.08ms上限、预计3.55–3.65ms及卷积kernel必不变等说法未获证实。候选 `d32ff2821c0d` 在所选模块范围内返回20个BN模块替换；没有第三轮或额外修复角色。

### 第二轮与独立完整版本结果

两轮控制器于23:45:38正常结束。第二轮代码 `8045114abbc2dc6b384c8e74c9f77e3445c455df51ddbacaa84bea56d1502cab`，proposal `38e3b9c7…`；按实际模块类型及 sibling 结构折叠，无新案例路径、哈希或模型名生产分支，每次执行完整模型，不缓存输出。模型结构匹配仍是任务范围内的实现，不宣称任意图通用。普通CLI正式提交版本 `d54c6182…`。

| 验收 | 原始基线→候选 | 实际判定 |
|---|---|---|
| 第二轮正式ABBA | 3.987046→3.912039ms，**1.8813%** | 两对收益0.070279/0.079735ms，均超过噪声0.012538/0.006636ms；原吞吐约束通过 |
| 独立完整版本ABBA | 4.009106→3.9134355ms，**2.3863%** | 两对收益0.092219/0.099122ms，均超过噪声0.013760/0.018792ms；原吞吐约束通过 |

每次ABBA各20个clean样本、8次完整检查，每次32,037元素；候选最大误差8.583069e-6，完整logits、类别及原数值策略通过，全部样本保留。两批分数不合并、不相加。独立根目录 `goal-resnet-independent-20261031`，controller347011，23:48:30—23:48:59；只调用一次现有 `promote_version(publish=False)`，复用原只读controls，无新角色/搜索/collector/源模型或参考重跑。独立版本 `b13ffa1e…` 为exploration，独立current保持原EMPTY `98aa5653…`，源current保持d54。短批次在监听预检前结束，未创建新监听；旧32596完成事件已handled。

各自原始profile的opop::0窗口内，基线20个BN kernel、候选0个，总kernel均88；它证实实际消除BN调用，不证明0.425ms全部可回收。完整模型仍有其他工作。实际候选峰值分配283.438MB，略高于基线281.976MB，不称显存收益。参数总数增加4800个bias，而adapter的parameter_count是安装前库存值，不是优化后的实际计数；该值不参与本次接受。成功安装后折叠参数持续保留，uninstall/fallback并非恢复原始权重；原全量质量仅证明当前推理范围。

第二轮普通planner选择candidate_diagnostics与binding_cost；诊断不是质量验收，绑定对照的两对parent/passthrough为3.983398/3.996711及3.998625/3.995930ms，不能直接分解正式收益。正式记录SHA `a9117297d673e1d107de1f1a042f7f735b046f67bc3046e46ef4f975cbc6fa65`，配对重算 `fc5c5547…`，独立核验 `19b1f311decbebae3f75c8d3ffb343ba887e922ca00ba6eae4be939fe6f2ae0b`，原始记录均在服务器。一次只读reader把profile coverage整数当bundle字典而报错，保留原reader并只改读取范围；没有重跑实验。

下一主线缺口是当前核心的DDP原入口流程，不继续ResNet压分，也不重开旧DDP候选链。容量处理纠正：最新用户指令称8GiB为warning，既有监听也明确只投递注意事件、不停止实验；先前把它解释成“预留8GiB再加forecast”的启动硬门槛并要求答复并不必要。没有把未答复当作授权，也没有降低告警阈值；按已授权的实际容量预算推进，保留原8GiB告警与低盘检查。Goal未宣告完成。

### 当前DDP原入口准备

2026-10-31 15:59:49 UTC／香港23:59:49，controller **347816**，根目录 `/root/autodl-tmp/dev/v6-work/goal-ddp-original-20261031`。现有生产CLI `optimize-project --project examples/v6_validation --command 'torchrun --standalone --nproc_per_node=2 gpt2_ddp_training.py --output /tmp/original-ddp' --goal … --devices 0 1 --prepare-only`。自然语言目标为完整训练step延迟，保留源模型、global batch/accumulation、FP32、完整梯度/更新参数、finite loss及精确梯度replicas；通信时长只作诊断。没有输入旧request/evaluator/候选/分组/算法/核验答案；复用原普通私有配置，生产核心仍e893，无新测试或框架改动。

启动available **8,619,958,272B**。准备阶段5GiB增长预测以既有两份完整参考各1,991,214,515B、合计3,982,429,030B为基础，额外约1.39GB覆盖任务产物和控制记录；不是API/collector限额。8GiB仍告警，预计跨线不当作完成或盲目终止理由；检查剩余工作量和真实df后处理，去重同一告警。无新冷存或参考复制。准备返回后先审实际生成语义；有效才按同request进入有限两轮普通优化，搜索容量另算，保留任何失败，不自动修复到通过。

11月1日00:02:42实际核实controller存活，普通intake snapshot `0f2f0dcc5cff`／session `ses_e6d4faa9affe3gjRkr37QRUHNa` 已出现，5个初始文件；GPU空闲、df available8,619,528,192B。这仅是身份/活动证据，不是准备正确性。单个隐藏监听 **5392** 经waiting预检启动；无模型健康调用、历史重审或额外生产改动。

### DDP准备通过，普通两轮优化已启动

香港11月1日00:15:51准备正常完成，prepared `caa5257d45cb871d2285d511ca11c1d2ed8c2a23eadc3d7432e6a2e1bdf10610`；adapter `3368992ca5b2497ed910ba005fdf90b3db4159c0b142facaad5581d07a07ac40`（13,907B）、request `3619e6a57c2f12f2cf81be5e16f236b61b5d9ddae95143cd61f8e35118ac6399`。实际代码保留原模型/种子/FP32/noTF32、CPU初始权重、两卡B1/S128/accum4、前三次no_sync、完整zero_grad/FWBW/loss除4/SGD及末端同步，计时前Gloo协调。生成的observe自主执行精确梯度副本检查，并在rank0计时外重放1/2步串行global batch参考；因此本次确有普通evaluator的新参考运算，不能称旧source哈希复用。开发者没有重跑模型/GPU或复制参考。

两卡reset各两次及identity比较全部为0，wrong-output/missing-coverage控制通过。每步均有148份完整梯度、124,439,808梯度元素和同数量更新参数；rank0两步串行参考梯度最大误差7.4505806e-9/6.7986548e-8、参数3.7252903e-9，原容差及精确replica通过。只读CPU mmap核对保存的检查值，不再算模型。五个逐样本rank-max为108.314193、107.011910、107.629041、106.525478、106.809146ms，中位数 **107.011910ms**；这是新原入口基线，不混用旧133.379621ms或旧DDP分数。rank0生命周期峰值约3.944GB包含串行参考，rank1约2.205GB；计时阶段两卡峰值约1.98GB。检查记录SHA `5a3bcda23b17612baba00c79d89250470d76ea33898e3e501dfcd6a77bcca2e3`。

实际intake13工具、evaluator19工具。原源码/配置接口读取及小模型CPU探测已核对；intake一次GPT2Config属性名错误保留。evaluator另自主执行两进程Gloo小模型fixture，首次误在rank1断言rank0专有字段而KeyError，修正fixture断言后通过；不是原两卡GPU质量证明，也没有修改交付adapter（一次写入、无编辑）。末尾检查ABI/语法并删除自身pycache；这些由普通角色发生，开发者未追加测试或修复链。记录范围仍非隔离文件系统下的绝对访问证明。

**当前优化根目录** `/root/autodl-tmp/dev/v6-work/goal-ddp-optimization-20261031`；2026-10-31 16:18:42.461565 UTC／香港00:18:42，controller **351511**。现有CLI对原run执行 `--request 原request --output 同一run --resume --rounds 2`；核心e893、普通prompt/model/tools及质量/噪声规则不变，没有输入旧候选、手工算法或上述审查答案。00:20:13已核实原discovery完成/current `8125e24d…`，普通collection-planner `43685c0420c3`／session `ses_e6d3e1edeffexIGvQvw0LrjJUh` 六初始输入已出现；尚无trial结果。新根只设置一个隐藏完成监听 **16040**，实际waiting预检后启动。

准备实际增加约4.08GB，结束可用约4.54GB，低于8GiB告警但在已声明5GiB预测内。旧5392已投递低盘注意事件；核对后准备已终态，已handled，不补旧监听。新两轮启动可用 **4,537,098,240B**，预测 **3GiB**：当前六张诊断trace约97MB，默认至多3个screen加4个正式endpoint、两轮约1.36GB；另约1.86GB覆盖发现、正常自主采集、角色产物及波动。原约3.98GB controls在同一run复用、不复制，无新清理。预测不是API/collector限额；8GiB仍告警，已确认的同一跨线不重复投递，完成监听只处理新终态，后续按实际df/剩余工作处理。生产框架、质量、接受规则均未更改。两轮后检查实际路径/数值/配对接受与反馈消费；不自动追加第三轮，也不以这一个项目闭环标记整个Goal完成。

### DDP首轮拒绝，第二轮已消费真实反馈

香港11月1日01:04，原两轮控制器351511仍在运行。C1普通规划器自主选择六项既有采集器；首轮selector据实际资源材料选择完整step执行计划，candidate `b3a004ef4784c04db03279a87a7f8f49bab11679594698adefe59fa57a7695c4`（8,975B）尝试前三次no_sync微批次Graph重放、最后一次原生DDP同步及foreach更新。无手写优化、生产核心或prompt修改。

首轮 `trial-ff10c5d978` 正式拒绝：B两对107.729532/107.880377ms，C117.743657/113.330675ms；收益−10.014125/−5.450298ms，原噪声界限5.779976/1.411752ms。整体 **107.8049545→115.537166ms，慢7.1724%**，`paired_gain_not_above_noise`及`objective_not_improved`。40个正式样本、16次完整检查均保留；rank0每次248,879,629元素、rank1每次248,879,623元素，最大误差0。但代码在setup捕获RuntimeError后永久回退，两卡质量路径均为parent passthrough/fallback；24张正式原始trace均3999kernels、13NCCL、0Graph启动，所以这是回退质量通过，不能称Graph优化成功。另有一次空参数screen的10样本/4检查/6trace，未合并分数。

普通反馈规划器自行选择candidate_diagnostics、binding_cost和python_runtime。独立诊断捕获到Transformers `eager_mask` 创建标量时的CPU→CUDA复制不允许进入Graph；这是既有collector实际发现的异常，不是开发者给出的修复答案。候选CPU小模型/单进程Gloo探测不证明原两卡Graph正确性；其异常回退中未赋值的updated局部变量及多参数组lr等条件性限制已记录，未人工修改。资源选择中的单位误标、rank颠倒及跨profile因果推断不作为性能证明。

第二轮普通selector `53d0f0db039a`／session `ses_e6d15d95effeEAqE6Y73URkxz7` 已启动，41项输入。实际工具14/16/18读到拒绝原因、107.8049545/115.537166ms、两项噪声和回退标签；工具23已展示捕获异常消息及binding_cost摘要。当前仅确认实际信息暴露，最终选择理由及第二轮结果仍待返回。没有额外轮次、人工候选修复或第三轮。服务器记录：`first-paired-review.json` SHA `84ab5eea7bc492ca0133cf65b00c8640464b0dc6b2376a3b5113693799ee7992`；`first-trial-review.json` SHA `19e87fae643f7b780afa3090d4e62df32f442f0274dc780267b4d9541a4bccdf`。只读reader的profile coverage类型及CPU/GPU同名窗口读取错误保留，未重跑实验。

01:04实际df可用3,271,634,944B；原8GiB已确认告警不重复投递，没有新清理。唯一完成监听16040在01:05:58核验waiting、controller存活。当前真实任务继续执行；Goal保持active，后续依据第二轮实际结果选择整版独立验收或剩余跨项目缺口，不以首轮拒绝或这份记录结束项目。

### DDP两轮关闭，转入Qwen离线原入口

第二轮selector实际读完首轮拒绝、配对数值、回退路径与异常位置，提出显式mask/loss和稳定梯度缓冲改写。普通candidate `d17ca345035c` 最终代码 `e13c616884b38814661f7ccb730e75a9718dfee54585f3afb0f8264c9206bcfa`（15,746B），proposal `6292d6b8…`；44实际输入36,837,487B、69工具，代码/proposal各一次写入无编辑。小模型CPU mask/logits/loss/grads与foreach探测通过不代表原GPU正确性；一次错误CPU API断言保留。无人工改代码或额外角色，普通两轮范围不变。

第二轮 `trial-0aa9491baf` 的B为113.069022/114.667429ms，C为112.386166/114.180735ms，收益0.682856/0.486694ms均低于原噪声2.450892/6.358922ms；整体113.8682255→113.2834505ms的0.51355%仅是未接受观察。正式40rank样本、16次全检查全部保留，每次rank0比较248,879,629元素/rank1比较248,879,623元素，误差0。但质量路径再次是首次parent、第二次setup失败后永久parent fallback；24正式加6screen自身trace均3999kernels/13NCCL/0Graph。不能称实际Graph优化或Graph质量通过。

普通反馈planner `0883452ed9ef` 的8输入271,377B、22工具实际核对，选择candidate_diagnostics/binding_cost/nccl_tuning，路径错误及git exit128保留。诊断在两rank的捕获backward line172定位到AccumulateGrad陈旧default-stream引用；line227为同一异常传播，不是第二个独立故障。绑定对照两对112.804938/115.530283、126.985549/111.760303ms不能直接分解或抵扣候选性能；tuning是另一次setup/warmup记录，不是timed-kernel协议join。第二轮共12jobs/24rank结果已核对。数值记录SHA `e77aa0eca14c1a02c93e768fe282a7f69ea8106fd3358879da8a087914fdc706`，终态记录 `7056dbba21448a73e042b30b88243e46ba6551c5903eccf0b494aba64e6cf15a`。

控制器351511于17:39:19 UTC正常结束，第二轮正式`paired_gain_not_above_noise`拒绝，原EMPTY current `8125e24d…`保持。原质量证明回退正确，C1资源和实际异常确实进入了C2第二轮，但没有得到新DDP优化接受；C3正确拒绝。不会启动独立候选验收或第三轮修复，也不把候选错误当作修改生产框架的理由。旧16040事件已handled，迟到去重。

**下一步已执行**：17:41:53 UTC／香港01:41:53启动 `/root/autodl-tmp/dev/v6-work/goal-qwen-offline-original-20261031`，controller378140；现有CLI从原`qwen_inference.py`、本地预训练资产及自然语言offline目标进入`--prepare-only`。目标为原B8/prompt256/new32完整同步prefill+decode吞吐，保留BF16、完整逐步logits、精确greedy tokens及原teacher-forced关系；不是旧B1交互式TTFT修复链。没有旧request/adapter/candidate、手工算法或上述DDP审查答案输入。生产核心仍e893、普通配置/model/tools不变，通用supervisor按protocol执行，无新生产代码或测试。

实际启动可用2,242,461,696B，准备预测512MiB：原offline参考77,811,899B×2加约381MB产物余量；不是APIcap，8GiB仍已确认warning，不作预留硬门槛。本次无清理、参考复制或并行GPU。原参考保留供准备后只读CPU语义比对，未额外执行源模型。普通intake `ce55934436c4`／session `ses_e6cf22ba5ffekOX0DmT0G9wvTm` 实际5输入18,963B与snapshot相同，core一致；这仅是启动证据，不是准备正确性。首次reader误查尚未建立的output目录，实际intake在同root的output.intake；FileNotFoundError保留，仅改读取位置，无重跑。唯一隐藏完成监听45912经waiting预检启动。准备有效后再按实际容量进入同request两轮普通优化，Goal保持active。

### Qwen离线准备通过，普通两轮优化运行中

17:48:51 UTC准备完成，prepared `0db575bc…`，adapter `24ea41d3129ae1cb3d1e73f83f36fe3156e07fa6b22616b08a6a0fd6b6580c86`（8,175B）。实际保留B8/256→32、预训练BF16/SDPA/noTF32、每请求空KV cache、固定32步greedy及原三处同步；完整logits和teacher-forced比较在计时外，整数tokens由既有通用比较器执行精确比较。原reset/identity/wrong-output/missing-coverage通过，基线314.384585tokens/s。框架目标计时包含调用返回/清理，原函数内部计时仅作诊断，不称原源性能同一计时。

只读CPU核对原offline参考和当前两份controls：每份38,897,920个input/token/logit元素全部一致，完整BF16 logits最大误差0、tokens精确相等。未重跑源模型/GPU或复制参考；开发者显式字段映射是辅助核验，不是通用自动语义证明，也未输入后续角色。intake13工具、evaluator20工具已核对，资产路径错误保留；角色末尾有实际tokenizer/CPU输入tensor fixture，不能称完全未做tensor操作。源语义记录SHA `9d453308179e9a814ca39cf9854079950e6ebe0a78d441251c45b01c62ada817`。

**17:53:37 UTC／香港02时前，已启动同request普通两轮优化**：根目录 `goal-qwen-offline-optimization-20261031`，controller380707；现有CLI `--resume --rounds 2`，生产核心e893及普通prompt/model/tools/原接受规则保持。唯一隐藏完成监听33744；旧准备45912、DDP16040已handled。没有第三轮、人工候选或旧交互式Qwen重试。

C1 discovery实际26,911 calls／17,853 groups，66,333 CUDA kernels全部有归属。普通initial planner `eba08066351a` 的6输入14,524B、10工具已核对，302显示行中request一行仅前2000字符暴露；一次plan写入、无编辑，自主选择python_runtime。两份worker结果通过，Python诊断的perturbed host时间不用于clean接受；全部归属也不证明因果关键路径。首轮selector已运行，后续实际改写与配对结果待返回。记录 `328e01881a189167e023be8046c3c6fa0d68d562d72bfeb91c0a3624e6fc4d4c`；只读reader的过大摘要和未考虑单行截断断言已记录，无实验重跑。

**按新实际占盘处理容量**：启动可用2,084,302,848B、1.5GiB预测；本次discovery原始trace/规范化记录合计约501MB，不能沿用旧小批量运行作为保证。18:00:04 UTC在GPU空闲时，仅将已关闭且正式拒绝的两轮DDP候选screen/pair-C共36张raw metric trace无损冷存：583,783,968→49,668,352B，逐文件恢复哈希通过，6项请求/决定/controls元数据保持，释放约534MB，可用1,894,416,384B。未触碰基线、源、接受版本、当前Qwen、controls、模型和依赖；不暂停/重启优化。RESTORE在 `storage-audit-20261031/goal-qwen-offline-optimization/RESTORE.md`，journal `1b960c93dde3f48b2352acb7466847db3693c53514feaf8b5452133efcf5692a`。8GiB仍已确认warning，不重复告警、不作预留硬门槛；继续依据实际剩余工作检查容量。

### Qwen首轮拒绝，原定第二轮消费反馈

首轮候选 `93b8d355…` 用静态KV缓冲和展开的decode计算替换原模块，但返回自定义 `_DecodeState`。下一步不匹配原 `DynamicCache` 调用签名，绑定回到原模型后因缺少 `get_query_offset` 失败。唯一空参数screen未完成全量质量检查或clean计时，正式配对未开始；普通诊断重复记录同一错误，0条候选内部事件不代表没有故障。`trial-a5eadc233c` 按原规则 `no_valid_parameter_candidate` 拒绝，原current保留。角色的小CPU直接替换测试绕过了生产绑定，不能代替原GPU集成结果。未人工修复候选或绑定规则。

第二轮selector `b0c970c751ef` 的28输入87,000,599B与快照相同；40个实际工具中，17读取拒绝和失败screen，19读取诊断堆栈，21读取完整旧候选，22/26读取实际绑定和类型描述源码，27读取原目标签名。最终选择明确连接这些证据，提出保留真实 `DynamicCache` 接口，继续同一decode范围。工具30空cache层索引错误保留；没有成功的缓存CPU证明。跨profile差值当host stall、所有cat当KV拼接、吞吐预测及通用fallback保证均未采信。记录 `second-selection-review.json` SHA `3229f2dd4bf40a3929646032359c744615c45a03d8ddb479a25204b81067e577`。

普通第二候选 `259b793b561a` 已收到31项实际输入129,065,077B；生产核心、模型、工具及接受规则不变。这是已声明两轮中的第二轮，不追加第三轮或修复到通过。最新可用磁盘约1.43GB，GPU空闲于角色生成阶段，唯一完成监听33744继续等待；没有新清理、额外模型测试或生产代码变更。待实际候选和质量/性能结果决定下一步，Goal保持active。

### Qwen两轮结束：正式接受与实际优化路径不一致

第二轮代码 `7e1cfaa2498f873184c02c31800ab2ce37500efd98665d05778935a5edf78846`（19,992B），proposal `889657bb…`（23,127B），空参数。普通candidate的31实际输入129,065,077B、58工具、代码一次写入两次编辑及proposal一次写入已核对。角色修复了真实DynamicCache接口，但小CPU复制计算fixture绕过了最终`_prepare`及生产绑定；其形状、长度和导入错误均保留，未由开发者手修。

实际资产`attention_bias=false`，候选却要求Q/K/V bias均存在，故`_prepare`直接返回None；后续o_proj尺寸条件也与实际2560/4096形状不符。每个候选正式endpoint的62次已记录调用均声明`fallback:setup-failure`并调用原reference；代码永久回退，声明的展开decode未执行。框架自身signature fallback计数为空只说明包装入口签名匹配。静态KV缓冲在失败setup中分配后释放，不能称它在计时路径中生效。未修候选、未新增GPU探测或第三轮。

| 原正式ABBA | Pair 0 | Pair 1 |
|---|---:|---:|
| 原模型中位数，tokens/s | 312.274981 | 300.546955 |
| 候选中位数，tokens/s | 318.446238 | 320.197246 |
| 收益 / 原噪声界限 | 6.171257 / 6.155472 | 19.650291 / 5.864949 |

合并端点中位数306.410968→319.321742tokens/s，观测 **4.2135482%**，两对确实按原规则通过，不能改写成性能拒绝；也不能将此差值归因于未执行的优化。20个正式clean样本、8次完整检查每次38,897,924元素，全部最大误差0；这是原模型回退路径的质量。另一次screen和binding_cost对照独立保留，不混入正式分数。所有尾部保留，无新独立版本比较。

生产CLI实际提交current `7456056b01a52e27dde11673260e9146a76eb4bd39c91ec2ef1fed6dc7a9fd4a`，历史accepted决定保持；原prepared EMPTY `0db575bc…`仍保留。本次14个新job的实际请求/adapter与原入口一致，核心e893未变。候选profile仍66,333kernels、0Graph启动；与原C1相同总数本身不是完整图等价证明，结合代码及回退路径才可确认本次未执行声明改写。timeline和cProfile为各自扰动诊断，不能抵扣clean耗时。候选诊断events为空是静默返回None，不代表优化成功。

普通反馈planner `251c44991e56`实际读取了接受数值、62次reference和fallback，明确识别矛盾并自主选择candidate_diagnostics、execution_timeline、python_runtime、binding_cost；8输入20,242B、14工具、一次plan写入，错误路径保留。没有注入开发者答案。记录：终态 `second-trial-review.json` SHA `2e6c17ea86be4ee03d0c3e0fae95f6f7937aa9bd252214d99ec0820e983a43a6`，数值 `760693e3652c35814be868b1e991a5f8895e61848f8536f317442e7ca5714500`，候选 `f825eb14eb34b247362de4ea876ad1dc36a80b3b2ccfba60eaf931bb6714ac75`，planner `a68ce8098f0592dbb4c9e68a450ae3e0c1c22d5fb71c378111b757e892861d35`；原始角色历史、配置和trace仅在服务器。

**下一步决策**：不把Qwen的4.21%计作展开优化成功，不追加候选/参数/独立GPU来寻求通过。已询问是否允许一个通用条件：新候选全部观测调用明确报告fallback且全部调用原reference时，不接受为优化；不因reference调用本身拒绝合法组合实现，不改变质量/噪声阈值或历史结果。代码确认现有计数来自连续质量检查，clean前关闭tracking；不能冒充计时路径证明，也不能仅凭这些计数排除后续预热才生效的实现。本次永久回退另有实际代码依据。拟议条件属于新接受政策，实施前必须明确这一范围并获得用户授权；尚未修改生产代码、prompt或测试。当前C1→C2反馈链已实测，C3对接受结果与声明优化不一致的处理尚待决策，Goal未标完成。

controller380707已退出，GPU空闲；完成监听33744已handled，迟到事件去重，不重启。终态核实时150GiB盘可用约400MB，8GiB仍已确认warning；所有当前工作已完成，无新清理或后台实验。此前18:00的必要冷存及RESTORE记录继续保留。

## 执行边界与容量

无并行 GPU、手写优化或 collector、案例路径/哈希/模型名生产分支；不重开已关闭的 Qwen/FSDP/77ac/observer/eef828 修复链。原接受版本、失败、质量与性能阈值保留。私有配置、角色历史和快照只留服务器，不输出原始配置或工具元数据内容。

必要容量处理仅使用现有无损冷存工具：65份已关闭、未采用角色的重复输入及拒绝候选派生记录，64,203,807B→10,172,362B；roundtrip hash 全部通过，21份保留对象不变。接受、原始源码、参考、controls、原生采集、模型及依赖保持在线。恢复说明：`/root/autodl-tmp/dev/storage-audit-20261031/goal-resnet-original/RESTORE.md`；journal SHA `7df73e30ce27585caffde46dbf71c19cfb51155bc12d566fea17d2efcb7920bf`。

启动实际150GiB盘 available 8,661,217,280B。64MiB增长预测只覆盖准备阶段，原8GiB警戒线不变，预测不是限制。后续优化须按实际容量重新安排；低盘事件不是实验完成。

优化启动available 8,656,666,624B，无新清理。单独60MiB增长预测来自旧同workload完整三轮ResNet实际45,412,352B＋intake270,336B，占用之上留37%以上余量，当前仅两轮；不是collector/API限制。原8GiB告警保留，超预测按实际事件处理，不盲目终止正在运行的工作。

独立验收启动available8,621,301,760B，16MiB预测基于实际四个正式配对job合计999,424B，包含共享artifact分支元数据余量；原8GiB线未降低。实际完成后available8,619,982,848B，无新增冷存、参考复制或并行GPU。
