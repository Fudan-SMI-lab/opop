# KernelFoundry 调研报告(供 v2 借鉴)

> 调研对象:`D:\Pyhon_projects\opop\kernelfoundry` = Intel ISL **KernelFoundry**(ICML 2026)。
> 日期:2026-09-02。方法:限定子目录只读源码调研(根目录有 14GB .db,禁止无范围递归搜索)。
> 目的:提炼 v2 harness 可借鉴的具体做法,尤其针对 level3:43 的 tf32 精度失败与 Triton 编译陷阱。

---

## 项目定位

进化式(**MAP-Elites 质量-多样性**)GPU kernel 生成框架,主战场 Intel GPU/SYCL,
完整支持 CUDA / Triton / OpenCL。**不复用** KernelBench 官方 `eval_kernel_against_ref`(全仓库无引用),
自建 pytest 测试 harness。对 v2 最有价值:双精度见证正确性门 + prompt 内置反模式代码对。

---

## Q1 整体流程:迭代进化,非单次生成

主循环 `kernelfoundry/algorithm/controller.py::_run_single()`(679-972 行):
```
for trial in range(max_iters):
    并行跑 branches_per_iteration 个分支(默认 2, ThreadPoolExecutor)
    每分支: 从 MAP-Elites 库采样 parent + inspirations → 构造 prompt → LLM 生成
    → AnswerProcessor 提取代码 → 并行评估(编译+正确性+性能+profiler)
    → 无论对错都 add 进 MAP-Elites 库(岛屿模型 + 迁移)
    → select_best_solution 选最优;stop_once_correct 可提前停
```
- **repair 是隐式的**:失败 kernel 不丢弃,以低分留在种群;被采样为 parent 时
  `template_manager.py::_determine_status()`(245-256)判 status="error",prompt 切修复模式。
- **多候选采样**:每分支独立采样 parent(exploration 20% / exploitation 70% / random 10%,
  `evolve_database_optimization_aware.py::_sample_parent()`:1743)。
- 可选 feedback LLM(`prompts/feedback_llm.py`):另一 LLM 把 console log 转"导师点评"再喂回(默认关)。

## Q2 Prompt 策略

**System prompt 仅一行**(`prompts/prompt_constructor.py`:22):
`"You are an expert CUDA engineer tasked with translating PyTorch code into performant CUDA kernel code."`

**User prompt** = Jinja2 模板 `prompts/templates/main_prompt.j2`,按 status 三态:
- `translate`(首轮):参考代码 + few-shot 向量加示例 + 硬件规格(`gpu_specs.py`,含 TF32 TFLOPS)+ 随机 2 条优化 tips
- `error`(上轮失败):上轮代码 + 蒸馏报错日志 + "分析错误→逐步解释→重写完整代码"
- `correct`(上轮正确):上轮代码 + top kernel + inspirations(带分数)+ 优化策略

**Triton 专属约束**(`main_prompt.j2`:216-219,原文):
```
1. tl.program_id(axis) 的 axis 必须是 0/1/2,不要用 tl.program_id(3)。
2. Triton 不支持 python/pytorch 全部功能,如嵌套函数、continue/break 不支持。
3. tanh/log1p/pow 等移到了 tl.extra.libdevice。
```

**Triton 常见陷阱清单**(`prompts/meta_prompting.py`:418-425,直接对应 v2 的坑):
```
1. Missing boundary mask: 始终用 mask=offsets < N
2. Non-power-of-2 BLOCK_SIZE: Triton 要求 2 的幂 block 尺寸,BLOCK_SIZE=100 报错
3. BLOCK_SIZE not constexpr: 所有编译期常量必须标注 tl.constexpr
4. Small tl.dot dimensions: tl.dot 要求两个输入维度都能被 16 整除,BLOCK_K<16 报错
5. Mixed float precision in tl.dot: 用 .to(tl.float16) 转换后再 tl.dot,累加用 float32
```

**BAD/GOOD 反模式代码对**(`prompts/optimization_aware_prompts.json` → `antipatterns.triton`):
```python
# BAD: BLOCK_SIZE 作为普通 python int 传入,非 constexpr
def kernel(x_ptr, N, block_size):
    offs = tl.arange(0, block_size)  # ERROR: 必须 constexpr
# GOOD: 声明 constexpr
def kernel(x_ptr, N, BLOCK_SIZE: tl.constexpr):
    offs = tl.arange(0, BLOCK_SIZE)  # OK: 静态已知
```
关于 `tl.next_power_of_2`:它的对策不是"别在设备代码里用",而是**所有示例/指令都把 BLOCK_SIZE
定为 host 侧字面量 constexpr**(JSON:430 "Choose BLOCK_SIZE as a power of 2 (128,256,512,1024)
declared as tl.constexpr")——从模式上根除 kernel 内计算 block 尺寸的路径。

关于 tf32(JSON `performance_hints.triton`):prompt 层面**允许** tf32
(`tl.dot input_precision='tf32' 用 Tensor Cores`),精度问题交给测试门(见 Q3)——"Always accumulate in float32"。

## Q3 正确性判定:双精度见证 + 宽松容差(最重要发现)

`tasks/kernelbench/task.py`:123-145:
```python
def test_all_close(self, reference_model, kernel_model, input_tensors, iteration):
    set_fp32_precision("tf32");  out_ref      = reference_model(*input_tensors)  # tf32 参考
    set_fp32_precision("ieee"); out_ref_ieee = reference_model(*input_tensors)  # ieee 参考
    out_kernel = kernel_model(*input_tensors)
    assert all_close_with_slack(out_ref, out_kernel) or all_close_with_slack(out_ref_ieee, out_kernel)
```
- **同一参考跑两遍(tf32 / ieee),kernel 匹配任一即通过**。v2 的 tf32 候选在此门下直接过。
- 容差(`kernelfoundry/testing.py::all_close_with_slack`:33-62):**相对误差 <1% 的元素占比 >99%**
  (epsilon=1e-7 防除零),非严格逐元素 allclose。
- 附加 `test_cosine_similarity`:展平余弦相似度 ≥ **0.99985**(同样双精度见证)。
- 结构:先 `test_output_shapes_match`(pytest dependency 门),再 all_close 和 cosine 各**跑 5 次、
  每次新随机输入**(NUMBER_ITERS=5,input 是 function-scope fixture);模型 init seed=42 固定,输入不固定。
- **未过怎么处理**:`evaluator.py::convert_test_result_to_exec_result` 给梯度化分数:
  0=提取失败 / 1=编译失败 / 2=运行时错 / 3=shape 不匹配 / 4=数值不匹配 / 5=正确。
  失败 kernel 照样入库(低 fitness),被采样成 parent 时走 error-repair prompt,反馈带蒸馏日志。
  连续失败断路器 `max_failures=5`。
- 性能也在 tf32 下测(`test_benchmark` 先 `set_fp32_precision("tf32")`)——speedup 对 tf32 参考算。

**日志蒸馏**(`algorithm/utils/eval_helper.py`):编译输出经 `postprocess_compiler_output`——
过滤非任务源文件 warning、缩短 include 栈/编译命令/路径、**截断但强制保留 error 行**
(5000 正文 + 2000 错误);pytest 输出把大 tensor dump 替换成 `tensor(...)`。

## Q4 性能调优:四层,无 Optuna/贝叶斯

1. **模板化 kernel + 内建网格搜索**(最接近 v2 PARAMS):LLM 输出 `forward_templated<BLOCK_X,BLOCK_Y>`
   模板 + dispatch 枚举组合;harness 正则提取全组合(`eval_pipeline/utils/extract_template_parameters.py`),
   **同一评估里逐个实测**,取最优,**每组合分数/耗时写回下轮 prompt**(`schemas.py::format_for_prompt`:102-107)。
2. **MAP-Elites 结构搜索**:特征维度 memory_opt/compute_opt/parallelism_opt 各 0-3 级
   (~2000 行正则从代码静态判级,含 flash_attention_style 检测),4 岛屿 + 迁移,archive 100。
3. **优化感知引导**:选 underexplored 网格坐标,把对应技术指令注入 prompt。
4. **meta-prompting(可选)**:prompt 的 4 区块本身参与进化,按子代得分归因 fitness。
- profiler 反馈闭环:NCU/unitrace/VTune 结果转文字进 prompt。

## Q5 模型与 backend

- 推理:`LLMEnsemble`。实际配置 **Kimi k3**(reasoning high、max_tokens 65536、超时 600/720/1200s)
  + gpt-4o/gpt-5.2 ensemble;RAG 关键词分类用 claude-4-5-sonnet。
- Backend:SYCL / CUDA(inline)/ **Triton** / OpenCL 四路,每路独立 prompt 约束/示例/反模式库。
- **无按算子(attention/conv)分派的专门代码**。算子知识全在 prompt:LLM 先把参考代码分类到
  ~40 关键词据此检索 RAG 示例;Flash-Attention 式在线 softmax 作为 compute_opt level-3 技术指令注入。
- 任务黑名单 `kernelbench_dataset.py::FILTERED_OUT`(28-49):剔除 19 个"正确性测试不可靠"任务
  (输出方差低/对输入不敏感,如 23_Softmax、94_MSELoss)。

## Q6 参考实现库

- `algorithm/prompts/kernel_examples/`:每语言一对 vector-add few-shot + 模板化 kernel 范例。
  **无预置 FlashAttention 模板**。
- RAG 库(`eval_pipeline/database/tables.py::Rag`:249):kernel_code + keywords + embedding + profile,
  设计上支持按关键词/embedding 检索高质量参考,但**默认 `rag: []` 为空**,不随 repo 分发。

## Q7 → v2 可借鉴的 9 点(按预期收益排序)

1. **双精度见证正确性门**(直解 tf32 失败):参考算 tf32 + ieee 两份,匹配任一即过。
   实现:v2 双见证 gate 里切 `torch.backends.cuda.matmul.fp32_precision` 跑两遍参考。
   证据:`tasks/kernelbench/task.py:123-145`。比"禁 tf32"好——tf32 dot 往往正是性能来源。
2. **宽松容差语义**:相对误差 <1% 元素占比 >99% + 余弦 ≥0.99985,而非严格 allclose。
   证据:`testing.py:33-75`。须配套第 9 点黑名单,否则低敏感任务漏判。
3. **prompt 内置 Triton 反模式 BAD/GOOD 对**:搬 `meta_prompting.py:418-425` 的 7 条陷阱 +
   antipatterns.triton;并补一条它没写但 v2 需要的:"`tl.next_power_of_2` 只能 host 侧调用
   (`triton.next_power_of_2(D)`),结果作为 `tl.constexpr` 参数传入;`tl.arange(0,X)` 的 X 必须 constexpr 2 的幂"。
4. **生成后确定性静态修补/拒绝**:提取代码后 regex 检测设备代码里 `tl.next_power_of_2`、
   非 constexpr `tl.arange` 上界 → 拒绝或自动改写,GPU 评估前省配额。
5. **失败候选不丢弃 → 三态修复 prompt** + 蒸馏日志(截断但保留 error 行、tensor dump 缩 `tensor(...)`)。
   4/4 全挂时一次 error-repair 通常比重开新族便宜。证据:`main_prompt.j2:126-131`、`eval_helper.py`。
6. **梯度化失败分级**(0-5)回传 generator 做针对性修复(数值错→查精度/归约顺序;编译错→查 constexpr)。
7. **模板化 dispatch:一次评估扫多组合 + 结果反哺 prompt**。Triton 可直接生成 `@triton.autotune` 多 config 版对照。
8. **首轮 few-shot 锚定输出格式**:极简 vector-add 输入/输出对固定签名/grid/mask 形状。
9. **任务级正确性可测性黑名单**:宽容差前对每任务做敏感性检查(扰动输入看参考输出变化),避免假阳性。

低成本细节:硬件规格注入 prompt(RTX 5080 SM 数/带宽/TF32 峰值);正确性测试跑 5 次、每次新随机输入。

**基线口径警告**:kernelfoundry speedup 对 **tf32 使能的 torch 参考**测;v2 若基线是 ieee 参考,数字不可直接比。
