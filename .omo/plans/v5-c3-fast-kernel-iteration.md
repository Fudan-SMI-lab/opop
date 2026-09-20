# C3 真设备 kernel：框架快速修订、C2隔离与冻结验证计划

## 1. 状态、目标与交付边界
状态：execution-authorized-active；最新用户已明确“开始执行”。2026-09-20T05:33:08.8110247Z激活新work，当前T1仅元数据与协议归档；尚无新tests/产品/GPU。
本文为唯一权威协议，不生成draft/公共副本；保留已完成的`.omo/plans/v5-c3-qwen3-4b-operator-goals.md`、全部旧work和raw。
用户澄清：“修复框架为什么会产生不符合要求的算子，而不是直接修复算子本身”；父级迭代的是任务流程/agent/schema/context/反馈等框架责任。
核心两环：开发阶段F_r框架固定→agent probe→父级快速定位框架原因→最小框架patch＋回归/C2兼容→F_r+1重新生成；之后F*冻结→全新正式生成/测量。
父级不编写/修改候选kernel算法、不把具体优化代码喂给agent照抄；失败候选原件是不可变诊断/测试artifact，不被人工改成主结果。
首要科学标准：agent实际编写的Triton/CUDA计算kernel，真实compiled、launched，并参与Qwen目标算子的输出；之后才讨论端到端收益。
Python decoder内联、F.linear封装、只import triton、无关dummy launch、config-only切换均不能获得该标准的PASS。
旧host-dispatch artifact的single约12.879%/multi约9.095%观测仍保留；旧严格C3设备kernel目标未达成，不回填旧winner或失败。
继续C3-B manual-task-first，复用人工任务/evaluator、Qwen resident runner及测量/质量定义；新协议另冻结独立heldout内容，C3-A自动任务构建仍DEFERRED。
在已授权`D:/Pyhon_projects/opop/v5`工作树执行，direct commit/push无需新PR/worktree；保留dirty，不重建环境/下载模型、不建安全/权限/cache/恢复/调度平台。
先单独GitHub归档本文并核验remote SHA，再写新characterization tests；狭窄实现/CPU验证后发布tested SHA再GPU。本激活片段仅元数据/协议发布，无SSH/model/GPU操作。
预期交付：框架缺陷与修订证据、C2兼容比较、固定F*下agent真实kernel和同卡结果/失败；不保证正收益、三个winner或对角线胜出。
开发中人工帮助属于framework engineering，必须披露；正式验证不允许逐候选人工优化、选择新prompt或挑选开发最好artifact冒充新生成。

## 2. 继承事实：只修已知接缝，不重做审计
引用K=`.omo/notepads/v5-c3-qwen3-4b-operator-goals/kernel-failure-diagnosis.md`；W=同目录`workflow-failure-diagnosis.md`。
K:7–16,24–37：仅TTFT尝试含自编Triton，接受的multi为Torch composition；被false-reject的三个final bundle也不是隐藏的设备kernel胜者。
K:74–86：RMS cast-before-weight、SiLU后BF16舍入等边界被改变；代数等价不等于冻结BF16实现等价；同时改多个算子使原因难隔离。
K:124–147：TTFT首次helper约21–22min，CPU脚本执行只占几十秒；不可把全部等待归因于CPU计算，更不能据CPU fallback通过认定GPU正确。
W:17,58–66：schema/compile/postprocessor失败会跳出旧repair路径；helper只有整模路径，缺少显式局部replay；过早耗尽slots挤掉正式对照。
W:40–52：旧profile是嵌套inclusive CPU dispatch，decode trace约60s；不能相加当CUDA瓶颈占比或指示重写完整36层。
W:70–76：短64-token fixture不覆盖4096 prefill、later decode或batch8；目标计时公式本身未发现新缺陷，不重写其定义。
直接复用已发布immutable-parent修复`82daf15b460004707e55dbdebb789e2788921527`、grouping修复`1870e8e612a1febfec3b77a63b2e091cbc7a6539`。
已有报告树`8526a1390c78346af1a5be112685b7fbe4defa59`为本激活实际核对的HEAD/generic基线；后续只核实际运行revision/改动范围，不重审历史数据。
不重复旧11小时资产/owner等待/实现/审计过程；不重哈希旧raw，不重放旧六会话，不重新调查已经修复的两个缺陷。
实际检查`src/kernel_optimizer/wiring.py:122–163,182–190`：legacy C2 `build_orchestrator`→`StructureRewriterAgent`；默认`build_task_rewriter`返回共享`TaskRewriterAgent`。
`scripts/experiments/c2_local_agents.py:52–56,83–92`同时使用direct/shared factory与legacy orchestrator；早期task43 direct的实际符号是`direct_child`，不是`generate_direct_proposal`。
`src/kernel_optimizer/task_cli.py:110–120`与`examples/c3_qwen3/operator_cli.py:131–134`亦走共享factory；当前C3隔离**未保证**，以下是拟实现分支，不是已存在能力。
父级提供的并行只读审计已确认：cda1130→8526a13期间，6745d90把共享`TaskRewriteResult`从`candidate_file,space`两字段扩为五字段，新增`bundle_file,site_groups,recommended_configs`。
同次`TaskRewriteInputs`新增`bundle_sources,bundle_document`；non-bundle prompt正文措辞未变，但seed序列化多了这两个default字段，实际AgentModule还把output schema发给模型。
因此早先C3已经改变未来direct C2的model-facing合同；尚无该变化造成历史生成输出差异的观测，不追溯否定任何旧实验。cda1130 helper pilot走legacy StructureRewriter，未受这组TaskRewriter改动影响。

## 3. 按任务kind隔离流程/agent，保护C2行为
显式声明两个独立轴：`task_kind`（`kernelbench`、`model_project_operator`、`existing_generic`）与native objective（TTFT/single/multi等的值/方向）。
任务kind/执行profile由调用者明确给定，不按goal regex、实验名、模型大小或“含Qwen”字符串猜测；目标细节仍是运行时context，不是三个预写优化策略。
Legacy C2的factory/StructureRewriter/prompt/schema/default flow全部保持不变；`kernelbench`任务的legacy/direct流程还需显式execution profile，不能仅看goal决定。
无selector的existing-generic factory保持当前documented default（含当前共享字段），避免破坏未知caller；不得把新增严格C3条件塞入默认或静默全局回滚。
Direct KernelBench/C2的`c2_local_agents.direct_child`显式选择`c2_direct_compat`：model-facing输出仅`candidate_file,space`，seed/input不含`bundle_sources,bundle_document`，比较锚为cda1130声明合同。
该cda1130锚只用于兼容合同对照；更早direct runs仍属于其各自真实旧SHA，不得统一改标cda1130或声称完全复现历史随机输出。
只有显式`model_project_operator`且该profile要求authored device kernel时启用严格kernel流程；其他任务不被强加Triton/CUDA条件。
最小方案：新`src/kernel_optimizer/agents/model_operator_rewriter.py`封装专用agent/subclass、task-specific guidance/result schema/context builder与小factory。
`examples/c3_qwen3/operator_cli.py`显式opt-in该factory/profile；未指定新profile时仍走既有默认路线，旧C3 SOURCE重现也声明其原profile。
最小factory政策：少量明确profile/classes或keyword-only selector；默认保持当前generic，direct C2显式c2_direct_compat，C3显式专用model_project_operator。不是registry/plugin平台。
C3新增字段只在专用输出model中出现；input/output JSON schema本身会展示给agent，只改render_prompt不足以隔离。
保护`agents/task_rewriter.py`默认prompt/schema、`agents/modules.py`、`agents/method_prompts.py`与C2 seeded files，不把device要求写入共享默认。
共享resident/evaluator/TPE可复用；所有新mode/budget/repair须opt-in，默认C2 probe、repair、TPE startup、推荐/default处理与控制流保持原样。
允许先只读检查/捕获已有行为；必须先归档本协议，才写新的characterization tests、改产品或运行GPU；归档后在第一份生产patch之前跑CPU characterization。
比较actual effective prompt、完整model-facing JSON schema、seed文件、API字段名/解析类型、factory/class/callroute、默认budget/repair/probe/TPE与native方向；仅规范化非确定IDs/临时根。
每次F_r→F_r+1只跑这一个focused C2兼容suite及受影响回归；不重跑C2 GPU实验、不建立新的测试/权限平台。
已确认的6745d90共享合同扩展须作为既存diff记录；T2显式direct兼容profile恢复其声明的model-facing两字段合同，同时保持generic当前默认与legacy路径，不做全局rollback。
82daf immutable-parent正确性修复可保留于共享/基类验证（需兼容测试），但明确它相对旧pin是validation行为变化；model-facing兼容不等于所有旧验证行为bit-for-bit一致。
要求精确历史重跑时使用该实验archived SHA；CPU snapshot只证明所比较有效输入/输出合同，不证明随机LLM会产生完全相同答案。
旧C2结果对其原SHA仍有效；今后C2执行/重现记录task_kind/profile/source SHA，只有比较证据支持才称prompt/schema相同。
C3修订如需改变共享行为，必须明确版本/回执和默认兼容证明；不能把F_r的产物重标成F_r+1，不能因此改旧科学失败记录。

## 4. 数据与测量合同：新heldout在开发前冻结
Qwen/Qwen3-4B pin=`1cfa9a7208912126459214e8b04321603b3df60c`、tokenizer/source与BF16 non-thinking不变；原58条corpus保持历史原件，不覆盖。
原42条heldout内容已经查看/报告，不能再称新自适应协议的fresh heldout；旧search/calibration可声明复用作开发输入，旧heldout只作历史/regression，不作新selection数据。
未来T1在任何开发/正式candidate之前，用同synthetic构造family/non-thinking template、新seed20260920冻结42条新heldout内容：TTFT12、single6、multi24，质量/长度/重复不变。
核验新heldout内部唯一、不交全部旧58条及全部新dev/search输入（文本/rendered IDs）；开发输入先声明，后续新输入也不得侵入heldout。evaluation仍seed0。
Heldout内容/参考与结果封存在final输入空间，开发分析/agent上下文不使用它们给反馈；只核长度/身份/不相交，不把新heldout变成prompt修订样例。
新raw记录corpus/contract新版本、内容hash及旧来源；旧9d1e…合同/旧corpus不改。此为前瞻新heldout约定，不声称已生成或属于旧数据的原批准状态。
沿用既有资产路径及A kernel-opt-venv/B orch-venv；Torch2.13.0+cu129、Triton3.7.1等以原准备回执和轻量运行检查为准，不安装升级。
一份模型/物理GPU常驻，重新绑定小bundle、每请求fresh KV；两host四卡不是显存池，不能同卡并行计时/编译。
TTFT仍4096→1，每次整模search evaluation为4请求；首tokenwall包含实际prefill，不从KV缓存取答案。
Single仍256→128，每call两请求，`127/(last-first)`；Multi仍8×512→128，一完整同步wave，`1024/shared_wall`含prefill。
Greedy、seed0、fixed EOS/output长度、warmup/reset及三个heldout blocks沿原测量合同；只前瞻更换heldout内容，不减少输出token、改变column形状或只测一个decode step。
Local rtol/atol=`0.02/0.02`，shape/dtype/device/finite及必要副作用不变；logits relative L2≤0.01，paired mean NLL delta≤0.02nat/token不变。
先局部正确，再两条64-input/32-reference-continuation短模型quality，随后该次完整目标测量；final按冻结heldout目标shape做quality。
局部representatives来自真实baseline call：4096 prefill、256 prefill＋接近末尾decode、8×512 prefill＋batch8 decode；仅取与所选算子有关的实际phase。
同一候选在其框架revision下经过所需代表shape集，不以64-token toy通过代替；动态mask/position/stride与捕获call一致，记录framework/candidate各自身份。
每次只保留一个小pure-tensor fixture及必要state，沿既有64MiB单call/128MiB总cap；不复制所有层KV或任意DynamicCache对象。
超过cap则换真正更小且可解释的目标边界，或记录unsupported；不能截掉必要状态、扩大cap后称同一验证。
精确保留原计算的中间BF16 cast、FP32 reduction、epsilon、weight dtype/alias与副作用；数值失败后不放宽阈值。

## 5. 开发环：框架固定、诊断、最小修订、重新生成
F_r（r=0,1,2）表示framework revision；候选另记C_r,0及最多一次artifact repair C_r,1，不能混用框架版本数和候选版本数。
状态：FIX_FRAMEWORK → AGENT_PROBE → PARENT_TRIAGE → ORDINARY_FAILURE 或 FRAMEWORK_DEFECT → TEST/PATCH/C2_COMPAT → F_r+1；达到DEV_GO则冻结F*。
F_0含T2初始C3隔离/helper接线；F_1/F_2至多两次证据驱动最小框架patch，不为凑满版本而改prompt。
每epoch固定源码SHA、agent prompt、agent-visible schema、context生成逻辑、measurement contract及profile；改动任一项即新framework身份并明确diff。
先用各goal一次短untimed CUDA phase窗口识别自身site/shape/kernel时间与CPU/GPU重叠；不以CPU inclusive总和或长60s trace代替device依据。
挑可观察headroom最高的一个goal及其ONE窄operator作开发probe任务，三个framework epochs保持此诊断任务，不靠换题找成功。
正式三goal各自依据自己的CUDA phase/shape选ONE算子，可同可异；不是TTFT→norm/decode→attention预写映射。
候选前记录匹配baseline/A/A局部噪声，用测得非重叠份额f解释`1/(1-f)`理想上界；重叠不明则限定假设，不设任意10%门槛。
若无可信headroom或无法捕获窄边界，则停止开发，不能靠调整测量定义创造收益。
任务上下文提供原完整算子计算、必要__init__/依赖、weights、shape/stride/dtype、cast阶段、epsilon、side effects、canonical groups与预算。
缺BF16 rounding上下文→修reference/context extractor；hostwrapper被当kernel→修C3专用agent合同＋真实device proof，而非修改返回kernel。
反馈太晚→修fixture-only helper/工作流调度；schema/postprocessor错误丢失→修有界错误路由；错误task agent→修显式task_kind/profile选择。
父级快判目标≤2min、≤10行，记录实际起止；字段为F_r/C_r hash、失败stage、框架组件、证据、一般化预期行为、regression test和预算。
快判仍检查scope/ABI/dtype/casts/weights/no-answer-cache及真实kernel意图，但输出不是逐行优化算法解答，也不能静态认证GPU正确性/速度。
只有可复现的data/schema/prompt/task-routing/profile/feedback/budget/control-flow/numerical-contract-context缺口才进入FRAMEWORK_DEFECT。
若必要上下文/runner已正确，普通候选错误或无性能收益不是框架bug；执行该F_r预声明的一次artifact repair或停止，不任意变prompt迎合该例。
普通repair仅由固定流程返回原始错误/失败artifact给同一agent，框架/prompt不变、计入call cap；父级不提供优化kernel代码。
框架patch前先写focused failing regression，保存原候选只读副本；patch针对一般化系统行为，不能把改过的测试candidate充当agent输出。
框架patch后跑受影响回归及C2 compatibility suite，记录通过/失败和新SHA；失败不发下一真实probe，不静默跳过兼容检查。
F_r+1使用新session/新生成，从同原baseline、相同开发任务出发；不把F_r最好的kernel拷贝成新epoch起点。
修复证据允许引用失败源码作fixture，正常reference源码仍需供agent理解；不能把父级写的优化答案嵌入新prompt。
DEV_GO需真实agent device kernel compiled/launched/output-used、局部正确、可信实测headroom且完整模型quality通过；局部收益/噪声如实记录，不宣称端到端已胜出。
DEV_GO产物只作机制附录；人写Triton control只证明harness，既不构成DEV_GO也不是正式winner。
达GO即可冻结当时F*，不用遍历全部epochs；无GO而无可证框架缺口时诚实停止，不强造patch。最多三个epochs仍无GO则不启动三goal正式验证。
所有失败、普通repair、framework原因、patch作者/时间、测试和新鲜生成记录保留；父级贡献明确为framework engineering。
通用/共享修复如确需行为变化必须单独version/receipt；旧F_r数据仍属于旧F_r，不重写其validity或假装历史框架已修复。

## 6. 最小C3实现与局部准入：慢default不被误剪枝
拟新增仅C3 agent小模块，修改C3 `operator_cli.py`/`operator_search.py`的显式route、开发/正式阶段控制；不更改共享C2默认prompt或输出model。
C3 `search_helper.py/search_session.py`新增opt-in `fixture_only`及阶段计数；借用原resident RPC，不另起模型registry/服务。
`model_runner.py/binding_runtime.py/torch_fixtures.py`只补所选pure-tensor的target-shape capture/真实CUDA反馈；不先兼容所有callback/DynamicCache形式。
Functional SiLU×up/RoPE如确被选中，仅task-local一个owned wrapper＋必要enclosing adapter；默认不让agent重写36层decoder。
C3 task-specific error adapter保留原sandbox/source/immutable parent及错误类型；artifact拒绝可按固定规则修一次，transport不generic restart。
沿用82daf不可变parent与1870grouping；给出canonical sites，不消耗GPU去猜组名，不用合成`{}`测试伪装合法default。
局部RPC输出compile、actual kernel launch/name、source/bundle/params、site/phase/shape/stride/dtype、cast阶段数值差和raw microbench。
Registers/shared-memory等只写真实metadata，缺失unknown；`fixture_only`不调用model quality/goal timing，`official_model_score=null`。
单RPC仅一source/config，可包含所需representatives/stages；3 warmups＋20 CUDA-event repeats/shape/side为拟定局部协议，配对顺序按call奇偶反转。
每个compile/local/default/TPE调用先占对应阶段local额度，失败不退；launch/forward/repeat子计数照录，不能批量隐藏config。
SOURCE_ELIGIBLE要求C3输出scope/ABI明确；实际compiled/launched/output participation与正确性由真实fixture证明，不是AST关键词、Triton import或dummy launch。
初筛default＋最多一个测前声明有意义alternative；default慢且实测tile利用/compiled资源/其他诊断指出具体可恢复配置问题时，允许该廉价备选再判断。
TPE仅在某通过点局部收益超出A/A噪声，或有上述狭窄实测可恢复情形且有余量时准入；resource counter自身不是收益证明。
每候选最多4 distinct configs总计含default、alternative、推荐；不是初筛2＋另外4。已实现pilot startup2/空space一次仍用，不造dummy参数。
局部trial仅可在同F/source/config/fixture/device身份下进入局部TPE目标并注明origin；不能变成模型J、旧private score或重复计样本。
整模候选必须先local通过，再短calibration quality及完整原goal repeats；local快不替代模型质量/有效native score。
正式接受还需匹配仪器epoch baseline、有效candidate严格native改善和一次确认，全部来自本阶段额度；平局/无效retain baseline。
首次setup需一个人写真实Triton control：baseline A/A、正确device输出参与、wrong/skip拒绝、restore；静态或Torch fallback通过不够。
Setup另封顶3local（reference/正确/错误）＋3model（A/A各1＋正确控制1），以及三个goal各1短profile窗口；失败即停，不反复control直到成功。
Fixture捕获forward及control/profile时间单列setup，不能当候选结果；后续任何framework GPU验证都计入dev额度，不补开免费setup。
Control前发布tested harness，每次开发patch通过CPU/C2兼容后标新revision再测；无新GPU实验或threshold放宽隐含在“验证”一词内。

## 7. 两套前瞻预算与时钟：开发不是正式实验
这些是待未来执行固定的新run上限，替代旧版本文的按goal连续三版预算，不回填历史48-slot实验；本次不消费任何额度。
开发最多3个framework revisions：F_0初始＋F_1/F_2两次针对性patch；每revision一个primary agent probe，在同选定初始goal/operator上生成。
每revision最多一次固定artifact repair，故dev总agent calls≤6；框架变更和candidate repair分开记，未用额度不构成必须生成的配额。
Dev所有epochs共享local GPU evaluations≤12、full-model calls≤4，含失败probe/repair验证、匹配baseline与复核；patch不重置这些计数。
Dev每candidate最多4 distinct configs（含default/alternative），模型调用前的local复核也占dev local12；一次问题已清楚就停，不机械烧完额度。
Dev D在工具/模型/control就绪、首次probe前记录；连续90min上限包括agent、父级诊断、F_1/F_2编码/CPU测试/等待/GPU，不能暂停时钟把工程当免费。
初始T2实现/setup在D前，另报实际工程elapsed/total turnaround；D之后的修补时间虽单列成本仍包含在90min内；无8–14h或3h完成保证。
每agent调用拟hard cap8min，generic/transport retry=0；artifact修复受阶段call上限，timeout不当新primary重复拉起直到成功。
目标首稿/局部反馈尽量3–5min、父级快判≤2min均为软目标；实际耗时写receipt，不从工具CPU时长推provider耗时。
DEV_GO之前不准入三goal正式生成；开发三epochs/预算/时间耗尽仍无GO则STOP，输出框架未能达标和未启动正式阶段，不填假失败分数。
达到DEV_GO后冻结F* source SHA、agent prompt/schema/context、task_kind/profile、数据/质量/计时/选择规则；保存C2 compatibility通过证据。
正式重新生成：三goal各从同一原baseline开始，一个primary＋最多一次固定artifact repair，即≤2 source versions/goal、正式agent calls总≤6。
不复制开发成功artifact，不让parent从各F_r挑最好kernel塞入正式结果；dev案例仅机制附录，正式代码必须来自F*的新session/tool输出。
正式每goal local GPU evaluations≤8，含全部compile/correctness/default/alternative/TPE/repair自测；每candidate最多4 distinct configs，不是初筛之外再给4。
正式每goal full-model calls≤4：通常baseline1＋至多2candidate点＋confirmation1，quality/warmup/完整goal repeats包含在call时间。
可在这4次内重分配baseline_alignment/candidate/confirmation；没有匹配baseline或确认的额度就不能接受，禁止导入dev或历史裸分数。
正式candidate model-call的local前置复核也计local8，admission同时检查两类额度；private/helper/人工要求测量不免费，不用合成空params伪造default。
正式总上限24local/12model、6agent calls，加独立heldout36 blocks；与dev12local/4model/6calls及setup3local/3model/profile3窗口分别列账。
Local/model可能含嵌套stage，报告原始job/forward/repeats及origin，不把配额向量相加冒充独立GPU jobs；所有失败已用额度保留。
F*、模型、接口、数据已冻结后才启动正式S；search admission=S+9000，final deadline=S+10800，最后30min只准heldout。
正式三goal可同时分配A0 TTFT、A1 single、B0 multi，B1不补候选；每goal从其首正式调用起连续90min，与global cutoff取更早，单GPU所有timing/编译串行。
正式parent quick triage只核预定协议/错误，不逐候选改算法、定制prompt、注入最佳config或人工cherry-pick；普通repair由F*固定规则处理。
若正式发现框架bug，停止相关准入、标记受影响rows/epoch无效或未验证；不inline patch、补draw、换F*或将修前修后结果pool。
未来开发F+1与新validation epoch须另行用户批准及新预算；本计划不自动授权二次正式实验，未受影响记录也保持其原epoch身份。
Setup实测决定dev/正式search及36-block预测是否可行；不足在各阶段候选结果前STOP请求调整，不缩长度/repeats/quality，不改变heldout内容。
两阶段各自截止不再准入新工作，在途按既有终止规则drain、真实结束单列；所有工程/等待计入总周转，不能只报正式S窗口。

## 8. 正式结果、失败与C2复现声明
失败两层分开：candidate的scope/ABI/compile/local/model/performance失败，及有证据的framework data/schema/routing/prompt/profile/feedback/budget/control缺陷。
同一kernel失败不是必然框架问题；错误分类、source hash、shape/阶段、证据、测试、预算与作者保留，不用长oracle波次拖延开发反馈。
F*正式选择只依据冻结规则下的新candidate/model证据；未接受则baseline fallback，不能换开发kernel或事后挑三个好结果。
Final只有一次：baseline＋3goal正式selected rows ×3metrics ×3blocks=36，同A0物理GPU串行，三个正式lane先terminal，其他卡不干扰。
每column用T1在dev前冻结的新heldout内容/reference与原shape/quality/repeats；不交旧58及全部dev/search，开发和正式选择均不得消费heldout。
固定列TTFT/single/multi；block1 baseline,T,S,M，block2反序，block3 S,M,baseline,T；重复winner/fallback也fresh测，null不填0，不heldout重选。
Dev无GO则不启动此矩阵，正式未完成则报告censored/invalid；未启动不算科学完成，不能为了勾选任务测假baseline赢家。
分别报告kernel authored/compiled/launched/output-used、local/model quality、native目标、heldout增益；单有AST/host composition收益仍不满足主device要求。
不强求不同算子/winner或对角线胜出；负收益是真实结果，不能修改quality/数据/目标后升级结论。
正式科学声明只归F*固定框架的新生成；人工框架开发过程、各F_r probe/ordinary repair作为独立开发证据，不宣称零人工工程或C3-A自动构建。
如存在人工算法改写的实验材料，不能作为本计划agent-generated主结果；只允许隔离为诊断控制并明确来源，不追加未授权测量。
报告C2 characterization基线SHA/profile、比较内容与差异，不凭未改文件名声称prompt/schema相同；过去C2结论不因新代码而被废除或重写。
新raw=`results/c3-fast-kernel-iteration/`按setup/dev/F_r/validation/epoch分目录，由数据/执行owner负责；report=`docs/result-c3-fast-kernel-iteration.md`由后续报告owner创建，本归档片段不写两者。
每epoch保留框架/agent/schema/contract/corpus/源hash、真实session与candidate作者；一份新输出索引，旧raw不复核，不做全会话大审计。

## 9. 实施任务、owner与验收
Wave1：T1只读捕获/合同与协议归档 → 新characterization tests及基线 → T2显式C2兼容/C3隔离与feedback；独立lane不共编文件，不每candidate重归档。
Wave2：T3真实control＋F_0首probe → T4有界framework修订/DEV_GO；父级负责框架诊断，不代写operator，不为每candidate开新review。
Wave3：T5冻结F*、正式三goal新生成及自动矩阵/cleanup → T6分析 → T7发布 → 唯一F1；dev无GO直接记录未运行正式阶段。
所有test/接口仍是待实施要求，本归档片段不运行；每次framework patch TDD RED→GREEN＋focused C2兼容，避免重复全仓或GPU C2 suite。
CPU前缀P：`$env:PYTHONPATH='D:/Pyhon_projects/opop/v5/src;D:/Pyhon_projects/opop/v5'; $env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'; & ./.venv-v5/Scripts/python.exe -B -m pytest -p no:cacheprovider -q`。
C2兼容C命令：`P tests/test_c3_task_routing.py tests/test_c2_agent_compatibility.py`，用真实factory/render/schema/seed逻辑和已有外部CPU fake，不调用模型。
EVID=`.omo/evidence/v5-c3-fast-kernel-iteration/`；RAW=`results/c3-fast-kernel-iteration/`；每task receipt记录实际source/profile/命令/证据。
裸`search_*.py`等路径均指`examples/c3_qwen3/`；`agents/`、`wiring.py`指`src/kernel_optimizer/`，新test均在`tests/`。

- [ ] 1. T1：冻结双阶段合同、C2行为基线、新heldout与协议归档
  最新用户执行授权已满足；parent/publisher owner负责元数据/归档，parallel CPU数据owner负责新corpus/runtime检查。引用本文当前路由、K/W及既有82daf/1870准备；只轻量核实际资产/revision，不下载或全审旧数据。
  顺序必须是只读捕获现有行为→单独归档协议→新增characterization tests→生产patch前基线GREEN；tests-first仍相对生产变化成立，但不能早于协议归档。
  未来用seed20260920在dev前冻结新42heldout，不交旧58/全部dev-search；新contract/corpus版本/hash，旧9d1e…及raw不改、不泄露给开发反馈。
  QA happy：归档后C命令在未改产品基线上GREEN，分别记录legacy/current-generic及cda1130 direct合同锚；failure：schema/seed/API解析类型差异不能当随机项忽略。证据EVID/T1.md。
  验收：协议先于新tests/产品/GPU，dev/正式预算不变；Commit：先仅协议、再单独必要characterization tests，绝不把测试写在归档前或每candidate新增归档。

- [ ] 2. T2：C3专用agent/schema/profile与局部反馈最小接线
  依赖T1；agent/routing owner独占`agents/model_operator_rewriter.py`、小factory selector及`operator_cli.py`，并将`c2_local_agents.direct_child`显式接c2_direct_compat；helper owner独占local RPC文件。
  引用`wiring.py:122–190`、`c2_local_agents.py:52–56`、W:80–106；`operator_search.py`由一个workflow owner在公共profile/预算接口确认后接线。
  新C3 task_kind/profile显式且native goal独立；legacy C2全部不动、无selector generic保持当前默认，direct兼容profile仅声明两字段输出/无bundle输入，C3使用专用agent/schema/context。
  QA：`P tests/test_c3_fixture_only.py tests/test_c3_framework_iteration.py` RED→GREEN，local不跑整模、实际C3 subclass/schema选择、无goal-regex路由、版本/repair分类正确；再C命令。
  Failure：仅正文prompt相同但direct schema/seed/API类型错、generic默认被回滚或CPU被强制kernel均拒绝；保留82daf snapshot回归并披露相对旧pin的validation差异。证据EVID/T2.md。
  Commit：狭窄显式profiles/feedback/tests，记录F_0、当前generic与cda1130 direct合同对照；不全局改schema/retry、动legacy agents/method_prompts或重写C2结果。

- [ ] 3. T3：真实harness控制与首个F_0 agent probe
  依赖T2；native owner做最多3profile窗口，选定最高headroom开发goal/operator；真实人写Triton control在setup3local/3model上限内，不当研究candidate。
  引用K:74–106与W:98–106；提供完整reference计算/cast/ABI/phase证据，先A/A/错误/skip/restore控制，不构造三种预设优化器。
  Tools ready后记录D并冻结F_0，发一个真实primary；局部/整模验证计dev12/4，不能因属于smoke/probe额外免费测量。
  QA：`P tests/test_c3_device_kernel_gate.py`拒绝dummy/unreached/import-only；真实control及agent各自launch/source/output证据，失败保持独立类别，再C命令。
  验收：人写control只证明harness，F_0 probe才检验生成框架；DEV_GO需agent质量/device证据。证据EVID/T3.md；Commit：仅必要C3控制/测试，probe前tested SHA。

- [ ] 4. T4：按框架原因做最多两次最小修订，不人工修kernel
  依赖T3；parent快判F_0及后续probe，列组件/证据/一般化预期/测试；原失败candidate只读保存，framework owner只修改有证据的C3接缝。
  普通候选错误走固定一次artifact repair或stop；缺上下文/路由/反馈等系统缺陷才写RED test→patch→GREEN＋C2兼容→F_r+1 fresh probe。
  QA：`P tests/test_c3_framework_iteration.py -k 'framework_revision or ordinary_error_no_prompt_patch or new_epoch_fresh_generation or dev_budget'`，并在每次patch后跑C命令。
  Failure：测试第二patch后仍缺GO停止、transport不重启、父级修改候选不计agent成功、测量合同/旧F产物不重标；保存patch/test/C2差异及实际90min时钟。
  验收：dev≤3framework revisions/6agent calls/12local/4model，人工工程时间计入D；无GO关闭开发、不发正式三goal；有GO冻结F*而非挑最好dev kernel。
  Commit：每个最小framework增量附tests和兼容receipt，EVID/T4.md；不global rollback C2，不修历史候选，不追加review wave。

- [ ] 5. T5：冻结F*后全新正式三goal生成、一次同卡矩阵与清理
  依赖T4 DEV_GO；无GO仅记录NOT_RUN。固定F*/agent schema/prompt/profile及新heldout，模型ready后设S；三goal从相同原baseline新session生成。
  每goal≤2source versions（primary＋artifact repair）/8local/4model，candidate≤4configs含default/一alternative；禁止拿dev kernel作正式初始解。
  正式parent只按冻结协议triage，不给逐候选算法反馈；发现framework bug即停相关准入并标受影响epoch，不inline patch后pool或自动延预算。
  QA：`P tests/test_c3_framework_iteration.py -k 'formal_fresh_after_freeze or profile_frozen or validation_bug_stops or split_budgets'`＋C命令确认default路径未漂移。
  Goal search terminal后自动在A0测一次36blocks；新heldout不交旧58/dev/search，源/params/quality/token一致，重复winner/fallback保留且不重选。
  截止/drain/无效/未测行完整；立即停止owned模型/RPC/agent进程并记录cleanup，旧资源不清理。证据EVID/T5.md；Commit：N，固定source运行。

- [ ] 6. T6：分别分析开发修订、C2兼容和正式科学结果
  依赖T5（含NOT_RUN/invalid闭合）；analysis owner不重做全会话审计，不生成新kernel、补测或改框架。
  分开F_r probe/人工framework patch与F*新生成；列每revision的prompt/schema/profile/hash、失败原因、回归证据、作者及实际时间。
  QA：`P tests/test_c3_framework_iteration.py -k 'dev_not_pooled or matrix_counts or invalid_no_score or c2_compat_claim'`，从新raw重算native方向/预算/矩阵。
  验收：C2 prompt相同声明有effective比较，不只看文件未编辑；36cell缺失、普通失败、framework失效不混为同一结论。EVID/T6.md。
  Commit：N，raw分析；旧SOURCE收益只是历史context，不能作为新F* control或证明设备kernel成功。

- [ ] 7. T7：发布固定框架验证与完整开发过程的有界报告
  依赖T6；report owner写`docs/result-c3-fast-kernel-iteration.md`，区分setup、dev、F*正式、heldout与工程周转，不混epoch或嵌套时间。
  QA happy：每结论链接framework/candidate/prompt/schema/contract source与真实launch/quality/model数据；failure：无DEV_GO或device proof标NOT_MET/NOT_RUN，不造赢家。
  引用EVID/T1–T6、C2默认兼容snapshot/diff、新raw索引；父级开发贡献如实披露，正式无逐candidate人工优化才能声明固定框架测试。
  验收：旧C2/旧C3 raw及结论不变；无新实验/修复/发布自动续作，仅剩一个compact F1。Commit：未来仅报告/必要复现说明，无模型/秘密/raw巨目录。

## 10. 唯一最终核验与停止
- [ ] F1. 核验框架修订/C2兼容与冻结F*的真实device结果
  依赖T7；单独context读取新证据，核task_kind显式选择、C3专用agent/schema/seed、default C2/generic有效行为比较与source SHA声明。
  核dev≤3epochs/6calls/12local/4model及formal每goal≤2versions/8local/4model、setup3＋3、final36分账，原失败和所有人工工程介入不隐藏。
  核F*后fresh生成、候选非人工改写、compiled/launch/output-used、heldout先冻结/未开发使用、质量/矩阵/时钟/cleanup；不凭AST或文字自报通过。
  QA：新focused tests＋C命令与snapshot/grouping必要回归、改动文件LSP；只新输出索引，不旧C2 GPU实验/大型审计。证据EVID/F1.md。
  无GO/负收益可形成诚实未达标交付；C2漂移未披露、dev/正式pool、人工kernel冒充agent、质量放宽或隐藏GPU工作则阻塞。只有一次F1。

当前T1第一片段仅激活新boulder记录/notes/ledger并发布这一份协议；旧记录不改。归档后才放行characterization tests；T1须另待兼容CPU及新corpus验收，不提前勾选，dev/formal时钟均未启动。
未来dev无GO在其3epochs/90min/调用额度边界停止；formal框架bug停止受影响epoch。进一步framework修订及新正式预算须用户另准，不能自动generate-until-win。
