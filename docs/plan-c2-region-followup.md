# C2 region follow-up：数据先交付，再顺序改进方法

日期：2026-09-18。本范围已由用户授权；先单独归档GitHub，再开始任何新产品修改，不另设批准仪式。
本文件是实施范围，不是效果报告；未来改进尚未实现或证明性能收益。
根目录：`D:/Pyhon_projects/opop/v5`；沿用现有环境、权限、工具和正常v5发布流程。

## 1. 已完成的数据步骤与不可改写的历史

用户顺序：①补齐固定产物数据并先交付分析；②推荐配置交接；③扩域输入与探测分配改进。
T1已完成独立新批次18个固定full blocks，无新候选、模型调用、TPE或重新选择。
每块100性能样本、5 correctness；新共享测量跨度230.354154s，约230.35s，不是GPU busy或总周转。
新rep1：task21 C2相对G0改善16.556831%；task43改善−0.624563%。
与原rep0描述性合表为四pair两胜两负，不满足3/4；跨批次不构成稳定C2效力或显著性证明。
T2数据及独立分析已先于本方法修改交付；当前结果草稿保留本地，最终阶段另行更新和发布。
旧campaign截止、两个censored heldout、raw和INCONCLUSIVE结论不变；新批次不回填旧时间窗口。

## 2. 自主自测事实与方法支持边界

真实记录显示8/8 rewriters、6/8初始parameterizers、2/7扩域parameterizers运行过候选或runtime probes。
23会话137条bash调用不是137次benchmark；工具可用、实际尝试、成功和协议等价必须分别报告。
quadplane child同源码、全部18个默认参数相同：私有timer约7.854ms，正式quick/default witness约16.911ms。
该正式child记录是20样本/3 correctness；不能误称100样本full，也不能只用TPE改参解释反转。
相同parent源码/参数：私有约17.464ms，正式full约8.462ms；两者方向相反且测量协议不同。
解释器/工具链、timer/cache/sync、seed和质量执行上下文不匹配，但没有隔离出某个因素的因果贡献。
保留bash、read、write、Python及自主实验；不把真实自测说成编造，不撤工具、不改成plain LLM。
旧审计意外初始化的空staged store按既有披露保留；本范围不清理远端文件或重做会话审计。

## 3. 第二步／T4：推荐联合配置进入既有B40

`RewriteCandidate`与`ParameterizationResult`新增可选`recommended_configs: list[ParamSet]`。
两者均默认空列表、最多2项，旧输出保持兼容；它们是待测建议，不是性能证据或硬接受门槛。
`ParameterizerInputs`携带rewriter的推荐；沿用现有parameterizer调用解释本次源码自己的key、defaults和伙伴。
parameterizer把可映射建议输出为其源码的具体配置；未知或缺失映射明确记录，不静默猜成已验证配置。
不增加LLM调用或证明schema；允许必要参数接线/body修复，不硬编码轴、dtype或特定历史赢家。
经`LegacyProposal`、`Child`、`RetuneInputs`、`CandidateRun`保留候选局部推荐，到初始及expanded空间。
必须按实际空间keys、choices、constraints和源码对应关系检查；不足以确定具体配置时记录原因、不入队。
非法、域外、约束不满足的建议记录为未准入，不使原本有效的候选整体遭硬拒绝。
保留原默认见证、第二见证、合法兼容prior-best anchor及原测量复用条件，不用建议替换这些点。
随后将合法建议按候选内去重追加为anchors；与见证/prior-best重合不另占重复推荐机会。
推荐消耗既有每空间B40，不在B40外赠送trial；有限空间耗尽等原语义不变。
私有速度或“已测通过”不转成measured_cache分数；需要测量时必须走既有正式evaluator。
仅本candidate自己的建议可流入其扩域；禁止跨候选、竞争臂或事后赢家注入和性能热启动。
先完成该合同、消费者和回归，再推进下一节，不以并行编辑绕过用户顺序。

## 4. 第三步／T5a：扩域看到实际证据而非虚构趋势

给现有扩域parameterizer实际selected `TrialRecord`、params/profile、numeric `TuningStats`。
同时给本candidate trial表、失败计数、task reference、既有domains/constraints、intent及recommended configs。
selected来自实测trial/space/artifact，不拿最后发布空间或当前defaults冒充真实best。
所有边际统计明确标为nonconditional；不能据单轴汇总声称固定伙伴的单调性或因果关系。
fallback若没有可靠证据，必须说未知；不得断言best位于边界、趋势单调或新区域必然更好。
先保持native eligibility、budget和threshold不变；不得以“补上下文”为名静默放宽策略或赠送预算。
要求按active branch、shape和伙伴解释轴是否生效；不是硬编码禁用某些axis或强制方向。
扩域建议仍受本空间合法性、同candidate来源和B40限制，源改变时不把旧分数当新源测量。
报告资格、接受/no-op/reject及实际执行，区分上下文可达、扩域发生和性能收益。

## 5. 第三步／T5b：现有12端点内的定向探测

新增显式模式：ordinary analyst一次→实际fresh C2 probes→rewriter一次；不增加agent调用数。
analyst用源码、selected、普通历史先提出最多6个有界probe requests：axis、两个existing values、可选partners。
这是基于探测前证据的初步建议，不能在报告或rewriter输入中伪装成analyst已看到未来response。
沿现有空间校验axis/value/partners及constraints；非法请求记原因，不执行，不扩域以迁就请求。
在原12 endpoint attempts内优先执行合法请求并去重；还有容量才沿现有端点fallback补覆盖。
实际失败消耗预算，不补采到成功数；不得把未测、拒绝或超预算点记零响应。
每份结果保留源码、selected、端点与固定伙伴对应；字符串/nominal值是contrast，不造数值slope。
rewriter接实际responses，实测事实优先于先前假设；无有效contrast时保留未知与普通源码推理。
原fresh-response-first路径保持兼容，由显式mode调用方选择新顺序；G0 ordinary analyst/rewriter不变。
必须接入实际`c2_information`／`c2_opportunity_program`消费者，不能只交付无人调用的helper。

## 6. 小型正式自测入口：支持探索，不管制探索

每个相关agent获得非秘密上下文：绝对configured worker Python、reference、shape/mode/seed及evaluation契约。
提供自愿调用的CLI helper，复用`CorrectnessEvaluator`、`WslGpuWorker`和既有benchmark/设备仲裁。
使用当前配置的seed、toolchain、timing与质量gate，包括FP64 rescue；不让shell默认解释器替代worker。
保留源码/参考hash、实际完整params、物化源码、协议/解释器、quality/rescue、score及有界结果记录。
fresh materialization及真实shape/reference必须对应本次请求；不写第二套timer或数值gate。
私有ad-hoc脚本继续允许，但结果与harness-equivalent调用明确区分；helper可用不等于必须使用。
不改全局PATH/env，不启动新服务，不改变bash/read/write权限，不建设安全或trace平台。

## 7. 文件所有权与顺序

| Owner／顺序 | 拟修改范围 | 边界 |
|---|---|---|
| 接口与提示，先行 | `src/kernel_optimizer/models/reports.py`、`agents/modules.py`、`agents/method_prompts.py` | max2可选合同、现有输入/输出与诚实说明 |
| T4 bridge/native，随后 | `control/orchestrator.py`、`scripts/experiments/c2_local_agents.py`、`c2_retune.py` | 配置交接、合法anchors、既有B40 |
| T5上下文/probes，T4后 | `control/task_rewrite.py`、orchestrator及`c2_information.py`／inputs／`c2_opportunity_program.py` | 显式顺序、实际响应与12端点 |
| 支持helper与测试 | 小型probe-config/self-test helpers；`wiring.py`／agent base仅必要时 | 复用调用链，不扩建平台 |

共享文件单一owner，前一owner交付接口后再顺序接线；独立测试可并行，不多人同时改modules/orchestrator。
不改旧计划、既有raw或历史报告；当前数据草稿在最后单独更新，新代码和数据效果分开解释。

## 8. TDD、最小集成验证与最终停止

先写失败用例，再最小实现：source exact/routing、max2、候选局部去重、推荐留在B40内。
覆盖未知映射/非法建议的明确理由及“不整体拒候选”，默认/第二见证/prior-best不丢失。
覆盖扩域真实best/profile/stats/trials交付与诚实fallback；不同空间/源码不能误贴测量标签。
覆盖定向探测12预算、去重/fallback、失败计数、非法请求原因、G0兼容及无额外模型调用。
覆盖helper使用configured worker/evaluation、FP64、shape/seed、fresh物化和记录，而非仅测试命令字符串。
跑focused CPU及直接相关旧回归、现有静态/LSP检查；不装依赖，不做全仓清理或大型审计。
发布测试过的精确source SHA后再部署，验证实际import及worker解释器；本次归档任务不执行这些步骤。
若需要GPU接线验证，最多4个固定full jobs：parent与quadplane child各在各自已审计默认配置测两次。
通过新正式helper，保持100性能样本/5 correctness；无TPE、新candidate、LLM draw或质量放宽。
执行前声明实际选择“最多4 jobs”或“仅已有CPU证明、不新增GPU”；不得看到结果后改验证选择或补跑。
这些检查只展示真实路线，不能隔离compiler/cache因果；不重复旧15次诊断或已交付18次heldout。
不启动新8候选campaign，不补造新论文阳性；缺失、失败、自测不等价及成本未知照实保留。
完成后仅一次紧凑独立核验；更新并发布新follow-up报告，区分已交付数据、代码接线与有限验证证据，然后停止。
