# 交接文档 — GPU 算子优化 Agent Harness (v2)

> 写给下一个 session 的 Claude Code。上一个 session 因 context 溢出中断。
> 日期:2026-09-02。工作目录:`D:\Pyhon_projects\opop\v2`(bash 路径 `/d/Pyhon_projects/opop/v2`)。

---

## 1. 任务总目标(用户原始需求)

在 `D:\Pyhon_projects\opop` 构建一个**基于 opencode 的 GPU 算子优化 agent harness**,对应论文
`paper\main_en.tex`("Guiding GPU Kernel Structure Search with Parameter Tuning Feedback",ICLR 2027
格式,method/experiments 目前是占位符)。9 条硬性要求:

1. 从任务生成一批候选 kernel
2. 每个候选的可调特征参数化
3. 每个候选做正确性测试 + 在参数空间内调参找性能天花板
4. 调参记录详细 profile(性能/资源占用);贝叶斯方法加速调参
5. 分析每候选调参结果:哪些参数有提升空间但被硬件/性能/资源限制顶住
6. 基于分析改写/调整候选结构 → 新候选
7. 分析每候选变更轨迹(含延迟数据)判断当前候选族是否到天花板(收敛/多轮无提升)
8. 生成显著不同的新候选作为新族种子
9. 每个部分是独立模块、可单独替换

**额外硬约束**:所有 LLM 步骤必须是真实 agent 调用(opencode session,agent 在沙箱内有自由度),
不允许硬编码 LLM API 调用。

**最终交付**:在 ≥2 个 KernelBench 真实任务上完成真实完整实验。用户已选定:
- `level3:21` (EfficientNetB2 MBConv) 和 `level3:43` (MinGPT CausalAttention)
- 开发期 smoke 用 L1 任务(level1:19 ReLU)
- agent 默认模型 `openai/gpt-5.6-sol`(每模块可 config 覆盖)
- agent 形态:plain "build" agent,每次调用一个全新 opencode session,沙箱为工作目录

**已批准的实施计划**在 `D:\ClaudeCode\data\plans\lively-cuddling-clarke.md`(架构、模块要点、
里程碑 M0–M5、单卡并行调研结论)。**先读它**——本文档只讲计划之外的状态与坑。

---

## 2. 当前进度:M0–M3 全部完成,M4 进行到 95%,M5 未开始

| 里程碑 | 状态 |
|---|---|
| M0 骨架/models/store/config/CLI | ✅ 完成 |
| M1 WSL GPU worker + 评估器 + baseline | ✅ 完成(真实 GPU 验证过) |
| M2 materializer + guard + TPE tuner + stats | ✅ 完成(真实 GPU tune-file 验证过) |
| M3 opencode agent runtime + 6 个 agent | ✅ 完成(generator/parameterizer live 冒烟通过) |
| M4 orchestrator 全环 L1 smoke + kill/resume 验证 | ✅ **完成(2026-09-02):resume 无重复 trial、loop C rewrite、loop D novelty、收敛 stop_kind、报告纯重放全部验证** |
| M5 实验1 level3:21 | ✅ **完成(2026-09-02):run-l3-21-20260902-113144,RUN_FINISHED,report 逐字节重放一致** |
| M5 实验2 level3:43 | 🟡 旧 run-l3-43-20260902-140823 完成(延迟数字可信,Loop C trace 被 resume bug 污染);干净重跑 run-l3-43-20260902-213608 因该批 agent 生成 4/4 seed 全灭而空(agent 随机性,非 bug)。**框架改进已实施(见 §10),待用改进后代码重跑验证** |

### 🔧 框架改进已实施(2026-09-03,见 §10 与 docs/analysis/)
基于对两个 L3 run 的磁盘诊断 + KernelFoundry 调研,已实施 6 项泛化改进(A/B/C/D/E/F,G 已移除)。
全套 **90 passed, 1 skipped**(skip 是 torch host 不可用的 relaxed_close,WSL 侧已单独验证)。
详见 `docs/analysis/improvement-implementation-plan.md` 与本文件 §10。

### 🔧 已修复的健壮性缺陷(2026-09-02,level3:43 途中发现)
**症状**:level3:43 run 在一个 repair agent 调用挂起后整个 orchestrator 进程崩溃退出
(`httpx.ReadTimeout: timed out` 未捕获,uv.exe 消失)。
**根因**:`runtime.py:OpencodeClient.prompt` 的 `self._http.post` 抛 `httpx.ReadTimeout`
(agent 调用超过 request_timeout_s=1200s 时),但 `base.py:invoke` 的重试循环只 `except
AgentCallError`,httpx 异常一路冒泡杀死多小时 run。这正是 plan 风险 #8("超时 abort +
从 events.jsonl 恢复")本应防住、但代码从未接线的情形。
**修复**:`runtime.py:prompt` 用 `try/except httpx.HTTPError` 包住 POST → 先 `abort(session_id)`
(杀掉卡死 session)→ 再 `raise AgentCallError(...)`,交由既有重试逻辑处理(重试→耗尽则
_parameterize_with_repair/_do_rewrite 里的 `except AgentCallError` 干净丢弃候选,run 继续)。
**回归测试**:`tests/test_agent_schemas.py::test_prompt_timeout_becomes_agent_error_and_aborts`
(FakeHttp 抛 ReadTimeout → 断言 AgentCallError + abort 被调用)。全套 78 测试绿。
**恢复**:`kernel-opt resume --run runs/run-l3-43-20260902-140823`,事件溯源保证不丢已花 GPU 时间。

### 🔧 已修复的健壮性缺陷 #2(2026-09-02,level3:43 旧 run 交付核验时发现)
**症状**:level3:43 旧 run(run-l3-43-20260902-140823)最终报告显示所有 4 个族都 `frozen_budget`、
`best history: []` 全空、全局 `budget_exhausted`——但全程只产生过 **1 次** REWRITE_PRODUCED(还被丢弃)。
即 Loop C(论文核心的"调参反馈→结构改写"机制)在 resume 后被**静默禁用**。延迟结果(cand-e4096974
@29.2ms,胜 eager 1.42×、compile 1.21×)是真实的,但族收敛/天花板 trace 不可信。
**根因(两处 in-memory-only 状态未随 resume 恢复,属"resume 丢失控制决策所依赖的内存态"同一 bug 类)**:
1. `_restore_pipeline` 重建 space/trials/best_ms/stats,但**从不恢复 `crun.report`**(BottleneckReport
   只在 live pipeline 的 `_stats_and_analysis` 里赋值)。resume 后所有族 best 候选 `report=None`,
   `_rewrite_round` 第 489 行 `if source_crun.report is None:` → `rewrite_rounds_used += 1; continue`
   静默烧掉改写轮次却不改写 → 伪 `budget_exhausted`。
2. `family.best_history` / `family.rewrite_rounds_used` 也是纯内存态(`record_round` 从不落盘),resume
   后为空,导致 `converged` stop_kind 在 resumed run 上永不可达,且改写预算不跨崩溃计数。
**修复(全部事件溯源,与 fix#1 同范式)**:
- `_restore_pipeline` 末尾:倒序扫 `BOTTLENECK_REPORTED` 事件恢复 `crun.report`。
- `_rewrite_round` 每完成一轮 append 新事件 `FAMILY_ROUND_RECORDED{family_id,best_ms,round}`。
- 新增 `_restore_family_control_state()`(在 run() 里 per-candidate pipeline 之后、Loop C 之前调用):
  从 `FAMILY_ROUND_RECORDED` 流重建 `best_history` 与 `rewrite_rounds_used`(len=轮数)。崩溃发生在
  首轮改写*内部*(尚未落 FAMILY_ROUND_RECORDED)时,轮数正确为 0 → 该轮被重试而非静默计数。
**回归测试**:`tests/test_resume_restore.py`(4 测:report 恢复/无事件时为 None/history+rounds 重建/
无轮次时为 0)。全套 82 测试绿。
**交付处置**:旧 run 保留存档;用修复后代码**从零重跑** level3:43 → run-l3-43-20260902-213608,
以取得可信的 Loop C + 收敛 trace。level3:21(run-l3-21-20260902-113144)无 resume 边界、3 轮真实改写、
真 `budget_exhausted`,完全可信,无需重跑。

### M5 实验1 结果(level3:21 EfficientNetMBConv)— 已归档
run `runs/run-l3-21-20260902-113144`(2.57h,283 事件,7 候选,152 trials):
- baseline eager 21.8ms / compile 16.3ms
- 4 seed 族,3 个空间因正确性未过双见证门被拒(SPACE_REJECTED, correctness_mismatch)
- 唯一存活 fam-3dacc96b:seed cand-ef4785e1 调参 25.2ms;3 轮结构改写(4edfa030/aac4d608/deb5eea7)best_history=[25.2,25.2,25.2] 均未突破 → budget_exhausted 冻结
- best-overall = cand-ef4785e1,θ_best={BLOCK_P:128,NUM_WARPS:2,NUM_STAGES:4},独立复测 PASS 25.0ms
- vs eager 0.872× / vs compile 0.652×(真实负向:MBConv 三卷积融合块 LLM triton 未跑赢 cuDNN)
- `kernel-opt report --run` 纯重放逐字节一致(sha256 ba95b626…,5342B)
- **诚实结论**:harness 完整演示两环搜索+参数反馈+家族收敛;此复杂算子上未超 baseline,是论文要展示的"调参撞天花板→结构改写→家族收敛"现象的真实实例。
- 报告位置:`runs/run-l3-21-20260902-113144/report/report.md`(注意在 report/ 子目录,不在 run 根)

测试:`uv run pytest` → 9 个文件 77 个测试全绿。
`uv run kernel-opt doctor` 全绿(WSL venv、torch 2.9.0+cu129、triton 3.5.0、CUDA、KernelBench、opencode)。

---

## 3. M4 状态(2026-09-02 更新)

### 已完成:kill/resume 验证 ✅

`runs/run-l1-19-20260902-011132` 在改写候选调参途中被中断(57 事件),resume 后:
- 未重复任何已完成 trial(新 5 个 trial 参数与旧 15 个零重叠,已程序化核验)
- 跑完 rewrite → 收敛(budget_exhausted)→ RUN_FINISHED(共 85 事件,20 trials)
- report.md 生成;`kernel-opt report` 纯重放再生,与原报告逐字节一致
- 最优:cand-ba2f4e04(rewrite 产物),251ms 调参值,独立复测 183ms,
  vs eager 1.082×,vs torch.compile 0.923×(L1 ReLU 是访存 bound,不指望大加速)
- 发现并修复:parameterizer 偶发只答 JSON 不写文件浪费重试 → prompt 已加显式警告
  (modules.py,"IMPORTANT: actually create ... on disk")

### 进行中:loop D(novelty)专项 smoke ✅ 完成

`configs/smoke_l1_novelty.yaml`(1 seed、rewrite 轮数 0、max 2 族)强制走 novelty。
run `runs/run-l1-19-20260902-093018` 已完成:novelty agent 产出 cand-16c51d16 通过新颖性门
(2D row-aware 布局,区别于种子的 1D flatten),成为第二族,完整 pipeline 跑通,RUN_FINISHED。
发现并修复第二个 agent 偶发问题:analyst 有时臆断沙箱文件缺失而放弃分析(经核验文件
确已正确写盘,是 agent 未实际读取)。因 analyst 是纯读取型、无产出文件校验兜底,已在其
prompt 加"文件已存在,先按路径读取,勿臆断缺失"。此问题不影响正确性/调参/收敛(BottleneckReport
仅建议),run 照常完成。generator/rewriter/novelty 有 _files_exist_check 兜底,无需改。

M4 全部验收通过。

---

## 4. 然后:M5 两个 L3 真实实验(核心交付)

M4 过了之后顺序跑(不要并行,单卡):

```bash
cd /d/Pyhon_projects/opop/v2
uv run kernel-opt run --task level3:21 --config configs/experiments_l3.yaml
# 完成后
uv run kernel-opt run --task level3:43 --config configs/experiments_l3.yaml
```

- 预算:40 trials/space、3 rewrite 轮/族、max 3 族、墙钟 12h/任务。真实耗时以 GPU 陪跑为主。
- 每份 report.md 必须含:θ_best 5/5 正确性、100 采样计时、vs eager 和 vs torch.compile 加速比、
  独立最终复测、完整 lineage + 收敛 trace(stop_kind)、agent token/成本。
- 中断就 `resume`,事件溯源保证不丢已花的 GPU 时间。
- L3 在 16GB VRAM 上可能 OOM:OOM 是一等 failure_kind,TPE 会规避;compile baseline OOM 会
  降级 eager-only 并在报告注明——这些都是设计内行为,不是 bug。
- 结果可用于回填论文的 experiments 部分(但用户没有明确要求本次写论文,先交付实验)。

---

## 5. 代码地图(27 模块,均已实现)

```
v2/src/kernel_optimizer/
├── cli.py            # doctor/baseline/tune-file/agent-smoke/run/resume/report
├── config.py ports.py wiring.py
├── models/{core,reports}.py       # Candidate/Family/ParameterSpace/TrialRecord/BottleneckReport...
├── store/run_store.py             # append-only events.jsonl + sha256 artifacts + replay()
├── agents/
│   ├── runtime.py    # OpencodeServer(spawn `opencode serve`, shell=True!) + OpencodeClient.prompt
│   ├── base.py       # AgentModule.invoke(): 沙箱→session→json_schema→pydantic→重试(带错误反馈)
│   ├── modules.py    # 6 agents: generator/parameterizer/analyst/rewriter/novelty/repair
│   ├── sandbox.py  prompts/*.md
├── tasks/kernelbench.py
├── gpu/
│   ├── worker_main.py    # WSL 侧一次性进程;litellm 用 sys.modules stub 绕过
│   ├── worker_client.py  # to_wsl_path, GpuRwLock(shared=正确性2路并发/exclusive=一切计时), WslGpuWorker
├── evaluation/{correctness,benchmark,profilerx}.py   # 静态检查缓存;单进程合并 eval
├── paramspace/{materializer,guard,validation}.py     # PARAMS 字节 span 替换;双见证发布门
├── tuning/{tpe,stats}.py          # Optuna TPE(multivariate,group,constant_liar);boundary 检测
├── control/{families,convergence,orchestrator.py}    # 收敛判定 harness 独占;novelty 门 <0.85 相似度
└── reporting/report.py
```

关键契约:候选文件恰好一个模块级 `PARAMS = {...}` 字面量 dict;materialize 只重写该字节 span,
其余字节必须不变;违规产生类型化 MaterializeError 原样喂给 repair agent(≤2 次)。

配置:`configs/default.yaml`(基准)、`configs/smoke_l1.yaml`(小预算 smoke)、
`configs/experiments_l3.yaml`(正式)。

---

## 6. 环境与已验证的事实

- **拓扑**:Windows host 跑编排器(uv venv,无 torch);一切 GPU 工作 =
  `wsl.exe -d Ubuntu -- bash -lc` 一次性子进程。WSL venv **复用**
  `/mnt/d/Pyhon_projects/opop/kernelfoundry/.venv-wsl`(torch 2.9.0+cu129、triton 3.5.0,验证过
  CUDA 可用,RTX 5080 Laptop sm_120)。不要新建 venv。
- **KernelBench** pin @423217d,源码 `/mnt/d/Pyhon_projects/opop/KernelBench/src`(经 PYTHONPATH)。
- **opencode**:harness 自动 spawn `opencode serve`(launch_cwd=`D:\Pyhon_projects\opop`,使
  `.opencode/opencode.jsonc` 的 provider/key 生效,版本 1.18.18)。Windows 上 `opencode` 是 .cmd
  shim → subprocess 必须 `shell=True`(runtime.py 已处理)。结构化输出走
  `format:{type:"json_schema"}` → 响应 `info.structured`,fallback 解析 fenced JSON。
- **计时纪律**:所有计时独占全卡(GpuRwLock exclusive);只有正确性/编译/静态检查可 2 路并发
  (shared)。这是单卡并行调研的结论,不要放宽。
- 每 trial ≈130s(L1),大头是 KernelBench 输入生成 + triton 编译,已经过一轮合并优化
  (静态检查缓存 + 单进程合并 eval),不必再优化。

### 观察一个 run 的进度

```bash
cd /d/Pyhon_projects/opop/v2
python -c "
import json
evs = [json.loads(l) for l in open('runs/<RUN_ID>/events.jsonl', encoding='utf-8')]
from collections import Counter
print(len(evs), Counter(e['type'] for e in evs))
for e in evs[-5:]: print(e['type'], str(e.get('payload'))[:120])
"
```

(事件的类型字段是 **`type`**,不是 `event_type`。)

---

## 7. 坑与教训(务必遵守)

1. **绝不在 workspace 根目录做无范围限定的递归搜索/grep**——根目录有一个 14GB 的 `.db` 文件。
   一切搜索限定在 `v2/`、`paper/` 等子目录。
2. Windows 上 python 读文件必须显式 `encoding='utf-8'`(默认 GBK 会炸)。
3. TaskOutput 的 timeout 上限是 600000ms。
4. 长命令用 `run_in_background: true`;agent 调用 + GPU trial 都以分钟计。
5. WSL worker 里 import kernelbench 会连带 import litellm(未装)——`worker_main.py` 的
   `_ensure_optional_deps()` 已用 sys.modules stub 解决,别动它。
6. 此前 Claude Code 本身的 "glm-5.3 classifier timeout" 报错已解决(settings.json 的 permissions
   allow 规则改成了 `"Bash"`)。若复发,根因是 `ANTHROPIC_DEFAULT_SONNET_MODEL: bigmodel/glm-5.3`
   路由不稳,与本项目代码无关。
7. **待办提醒(用户侧,勿代做)**:`.opencode/opencode.jsonc`、`opencode_backup.jsonc`、
   `kimi-provider.yaml`、全局 opencode.jsonc 里有明文 API key,建议用户轮换。交付时再提醒一次。
8. run/resume/report 的 CLI 全局参数在子命令**前**:`kernel-opt --config X run --task Y`
   (run 时);resume 不需要 --config(从 manifest 读)。
9. 用户沟通用中文。行动偏好:自主推进,只在破坏性动作或范围变更时询问。

---

## 8. 有用的历史产物(可参考)

- `runs/tunefile-l1-19-20260902-*` 附近的 tune-file run:手动调参验证,θ_best={BLOCK:512,NUM_WARPS:8}
  → 255ms vs 默认 464ms,profile(n_regs/n_spills/shared)记录完整
- `v2/examples/relu_triton.py` + `relu_space.json`:tune-file 用的手工示例
- `v2/examples/gen_cand_2.py`:generator agent 真实产出的候选(留档)
- 完整前史 transcript:`D:\ClaudeCode\data\projects\D--Pyhon-projects-opop\915ef857-b9f4-48a7-bc47-5e31ccbfa2ed.jsonl`
  (仅在需要精确历史细节时读,很大)

## 9. 一句话行动序列

```
读 plan 文件 → resume 中断的 smoke run(§3)→ 验证 M4 四条验收 → 顺序跑 level3:21、level3:43(§4)
→ 检查两份 report.md 达标 → 向用户交付(附 API key 轮换提醒)
```

---

## 10. 框架改进实施记录(2026-09-03)

诊断与调研文档:`docs/analysis/framework-diagnosis-and-improvements.md`、
`docs/analysis/improvement-implementation-plan.md`、`docs/research/kernelfoundry-findings.md`。
遗留决策:1(宽松容差)+ 2(双基线口径)采纳建议;3(参考 kernel 库 G)移除(偏离泛化增益)。

已实施 6 项(默认行为不变,由 config flag 控制;experiments_l3.yaml 已开启 A+F):

- **A 双精度见证正确性门**:`correctness_mode: dual_witness_relaxed`。新增 worker handler
  `run_relaxed_correctness`(gpu/worker_main.py)复刻 KernelBench 输入/种子生成,参考在 tf32+ieee
  各算一份,候选匹配任一即过(相对误差<1% 元素占比>99% + 余弦≥0.99985);计时内嵌该 job(ieee 下),
  不再走 KernelBench strict-perf(否则 tf32 候选会被 strict 复测再拒)。新 job 类型
  `eval_correctness_relaxed`(gpu/jobs.py::make_relaxed_correctness_job)。EvalConfig 加
  correctness_mode/relaxed_elem_tol/relaxed_pass_frac/cosine_min。
- **B Triton 硬约束 prompt**:新增 `agents/prompts/triton_pitfalls.md`(6 条硬约束 BAD/GOOD:
  mask/2幂BLOCK/constexpr/dot维度16/混精度fp32累加/next_power_of_2 host侧),注入
  generator/rewriter/novelty/repair 沙箱(仅 triton 相关时引用)。**只加硬约束,不加性能策略**(泛化原则)。
- **C 零成本静态门**:新增 `paramspace/triton_lint.py`,AST 检测 @triton.jit 体内
  tl.next_power_of_2(确定编译失败)→ 硬错误,挂在 AgentModule.check_output,GPU 前用既有重试机制反馈修复。
- **D 契约纠错 + (敏感性检查设计保留)**:`candidate_contract.md` 容差描述从错误的 1e-2 改为
  准确的 strict 1e-4 / dual-witness 语义。(D.2 任务敏感性 probe 已在计划中描述,本轮先落 D.1;
  宽容差安全性靠 attention 类任务本身对输入敏感 + strict 最终复测兜底。)
- **E novelty 名额修复**:`families.py::productive_family_count()` 只数 active/有 best 的族,
  dropped 族不占 max_families_total;加 max_families_total_hard(默认 6)防失控。seed 全灭时
  novelty 现在能产新族。
- **F repair 预算+分级**:experiments_l3.yaml repair_attempts 2→3;`modules.py::_repair_guidance`
  按 failure_kind 给分类修复指引(数值错→精度/fp32累加;编译错→constexpr/arange/dot;oom→减内存)。

回归测试:`tests/test_improvements.py`(lint×4、novelty 名额×3、repair 指引×1、relaxed_close×1[host skip])。
WSL 侧已验证 run_relaxed_correctness 依赖的 KernelBench 助手全部可导入、torch 2.9 CUDA 可用。

**下一步**:用改进后代码 + experiments_l3.yaml 重跑 level3:43(和 level3:21),
对比旧 run 的"4/4 seed 全灭",验证通过率与 Loop C trace 是否实质改善。

---

## 11. 改进后 L3 实验结果(2026-09-03)

用改进后代码 + experiments_l3.yaml(dual_witness_relaxed + repair 3)重跑两个 L3 任务,
均 RUN_FINISHED、report byte-identical replay OK、事件溯源完整。

### 双精度 baseline(决策 2,关键)
| task | eager(ieee) | eager(tf32) | compile(ieee) | compile(tf32) |
|---|---|---|---|---|
| L3:43 attention | 41.4ms | 28.8ms | 35.3ms | **17.9ms** |
| L3:21 MBConv | 25.3ms | 21.0ms | 22.1ms | **15.6ms** |
两个算子都是 matmul/conv-bound,tf32 tensor-core 把 torch.compile 提速近 2×。
**tf32 才是诚实对比基准。**

### 结果
- **L3:43**: best cand-52c895a9 @29.3ms(复测 PASS 31.1ms)。报告 vs_eager 1.33×/vs_compile 1.135×(对 ieee)。
  对 tf32:29.3 vs compile_tf32 17.9 = 0.61×,未超。5.16h,29 agent 调用 0 失败。
- **L3:21**: best cand-212fb80d @24.6ms(复测 PASS 25.3ms)。vs_eager 1.0×/vs_compile 0.87×(对 ieee)。
  对 tf32:24.6 vs compile_tf32 15.6 = 0.63×,未超。4.53h。4 个 seed 调参全部 24.6ms(同一 torch conv 回退瓶颈)。

### 改进前后对比(核心成果)
| 指标 | 旧 run | 改进后 |
|---|---|---|
| seed 通过率 | L3:43 clean-rerun 0/4;L3:21 old 1/4 | **两个都 4/4** |
| Loop C | L3:43 污染(1 rewrite/0 round) | **两个都 7 rewrites/6 family_rounds,两族各跑满 3 轮** |
| 收敛 | 伪 budget_exhausted | **真 budget_exhausted(best_history 三轮齐全)** |
| baseline 诚实度 | 只 ieee(旧 L3:43 "胜 1.42×"是假象) | **ieee+tf32 双记,揭穿假象** |

### 诚实结论
改进(A 双精度见证门/B Triton 反模式 prompt/C 静态门/D 契约纠错/E novelty 名额/F repair 预算+分级)
**决定性修好了框架流程**:通过率、多轮 Loop C、真收敛、诚实基线。但两个 L3 算子上 LLM triton 
调到天花板仍未超 tf32 torch.compile(attention/conv 是厂商库强项)——这与论文要展示的
"调参撞天花板→结构改写→家族收敛"现象一致,是真实的负向但完整的实验实例。

实施中引入并修复 3 个 bug(见 §10 下 fix commit ea8f379):Baseline note/kind、ModelNew 实例化、
torch 2.9 matmul API。运维教训:kill uv.exe 遗留 WSL worker + opencode serve 孤儿,需一并清理。

## 12. tensor-core / 精度维度改进(2026-09-03,H1/H2/H3)

深度取证(§11 + memory `opop-v2-root-cause-no-speedup`)定位根因:候选硬编码 ieee fp32
`tl.dot`,结构上用不了 tensor core;而 torch.compile 的近 2× 优势正来自 tf32 tensor core。
瓶颈分析只看寄存器/shared(微架构占用),看不到"精度路径/FLOP floor"这一真正的墙。改写反复
缓解寄存器压力却让延迟更差——证明诊断的瓶颈不是真限制器。据此实施 3 个方向:

- **H1 让 generator/rewriter/parameterizer 探索 tf32 tensor-core 路径**:
  - `candidate_contract.md` 新增 "Precision and the tensor-core path" 段:双精度门接受 tf32
    结果 → tf32 kernel 是合法候选;建议把 dot 精度做成 `PARAMS["DOT_PRECISION"]` 可调 knob;
    强调累加器必须 fp32(tf32 只降乘法输入尾数,不降累加)。
  - generator prompt:精度作为一等"计算方法轴",要求至少一个候选走 tf32 tensor-core 路径并把
    精度暴露为 PARAMS knob。
  - parameterizer prompt:matmul/conv 类要把 dot 精度提为 knob(choices ["tf32","ieee"])。
  - rewriter prompt:当 bottleneck 报告归因 `arithmetic_throughput`(或延迟 floor 跨资源画像走平)
    时,最高价值改写=把 dot 从 ieee 切 tf32/fp16-累加;明令不要在算术吞吐墙上继续做寄存器/shared 缓解。
  - `triton_pitfalls.md` #5 重写:区分"累加器 dtype(必须 fp32,是正确性硬约束)"与"输入精度
    (tf32 是性能 knob 不是 bug,双精度门接受)",避免把候选吓离 tf32 路径。
- **H2 瓶颈分析加入精度/资源平衡维度**(analyst prompt + `blocked_by` 枚举加 `arithmetic_throughput`):
  新增 "不要把'资源饱和'误当'该资源是性能限制器'" 段——(1) 寄存器满常是快配置的签名而非病,缓解=溢出更慢;
  (2) 闲置资源(shared 24% vs 寄存器满)只有在吞吐是墙时才值得换用;(3) 若用 ieee/scalar-FMA 且延迟 floor
  平坦像算术吞吐墙,应显式提出切 tf32 tensor-core(通常单点最高收益)。
- **H3 报告记录候选精度 + 4 路加速比 + 诚实同精度判定**(orchestrator `_finalize` + report.py):
  - `_detect_candidate_precision(src, params)`:优先看 DOT_PRECISION knob 实际取值,再看源码
    `input_precision=` 字面量 / fp16 / bf16 / 裸 tl.dot(→ tf32 默认路径),返回
    tf32|fp16|bf16|ieee_fp32|unknown(纯描述,不改运行)。
  - `speedups`:对每个记录的 baseline(dual 模式下 4 个)都算加速比;保留 speedup_vs_eager/compile 兼容字段。
  - `_honest_verdict(precision, speedups)`:tf32 候选对 `torch_compile_tf32` 比,ieee 候选对
    `torch_compile` 比(strict 模式无 tf32 baseline 时回退到 untagged),避免拿慢基线当稻草人。
    report.md "Best result" 段渲染候选精度、每 baseline 加速比、✅/❌ 同精度判定。

测试:`tests/test_improvements.py` 新增 H3 三组测试(精度检测 from-knob / from-source、同精度判定、
strict 回退)。全套 `uv run pytest -q` = 96 passed / 1 skipped(torch 依赖项,host 无 torch 跳过)。

**待验证**:用改进后代码重跑 L3(尤其 L3:43 attention),看 tf32 探索是否真能把候选拉到 tensor-core
路径并追平/接近 tf32 torch.compile —— 这是唯一有机会翻正的方向。
