# C3 是怎么把「一个整模型 + 一个目标」变成 KernelBench 式算子优化任务的

> 面向后续在主框架中实现该能力的开发者。C3 的这套流程目前只存在于 `examples/c3_qwen3/`
> (实验代码,针对 Qwen3-4B),**主框架 `src/kernel_optimizer/` 尚未实现**。本文用尽量平实的
> 语言讲清两件事:①算子组是怎么分出来的;②分好之后怎么决定优化哪一组。
> 每一步都标注实现文件,细节可回源码查证。

## 0. 一句话概括

KernelBench 的任务是别人手工打包好的(一段 reference 代码 + 固定输入形状)。C3 回答的是:
**没人打包时怎么办**——给一个完整模型和一句"我想让 TTFT 更快",框架自己跑一遍模型、
录下每个子模块干了什么,把"长得一样的模块"归成组,给每组过五道资格门,再按**实测的优化空间**
排序,选出一组,打包成一份和 KernelBench 任务同构的「算子任务包」交给 agent 去写 kernel。

整条链是:**用户目标 → 目标合同 → 短窗口 profile → 按身份分组 → 资格门 → 排名选择 → 任务包**。

## 1. 用户目标 → 目标合同(GoalSpec)

用户的目标描述("prefill 快一点"/"单流生成吞吐"/"多并发吞吐")先被翻译成**可测量的合同**
(`runner_records.py:26` `GoalSpec`),每个目标一条:

| 字段 | 含义 | 例(C3 的三个目标) |
|---|---|---|
| `id` | 目标名 | `ttft` / `single` / `multi` |
| `direction` + `unit` | 优化方向和单位 | ttft: minimize ms;single/multi: maximize tokens/s |
| `input_tokens` / `output_tokens` / `requests` | 负载形状 | 各目标不同(multi 是多请求) |
| `search_prompt_ids` / `final_prompt_groups` | 搜索用与终评用的固定 prompt 集 | 冻结,保证可比 |

**要点**:目标不是一句自然语言,而是"在什么负载下测什么数、往哪个方向"。后面每一步
(profile、排名、验收)都挂在具体某个 goal 上——因为**同一个模块在不同目标下的重要性完全不同**
(实测:MLP 复合站点在 ttft 下承载 146,793 μs CUDA 时间,在 single 下只有 17,258 μs)。

## 2. 短窗口 profile:模型自己说出它有哪些模块

`device_profile.py` 的 `profile_goal`:按该 goal 的合同 prompt 真跑一遍模型
(先预热一轮),在 `torch.profiler` 下给**每个子模块的 forward 包一层记录**,录下:

- 每次模块调用:模块路径(`model.layers.3.mlp.up_proj`)、**该模块类的源码**、权重形状、
  属性、输入张量的形状/dtype/strides、处于 prefill 还是 decode、第几个 decode step;
- 每个 CUDA kernel 的时间区间,并归属到发起它的模块 scope。

产物是一份 4.86 MB 的 JSON profile(每 goal 一份)。之后所有分组、排名都**只读这份文件**,
纯 CPU 操作,不再花 GPU 时间。每 goal 最多允许 3 个短窗口(admission 门,防 profile 吃预算)。

## 3. 算子分组:按「身份」不按「名字」

分组逻辑在 `fast_target.py`:

1. **family(家族)**:`family_key`(:61)= sha256(模块源码 + 权重形状 + 属性)。
   即:**代码一样、权重几何一样的模块就是同一种算子**——36 个 MLP 实例、73 个 RMSNorm 实例
   各自天然归为一组,不需要任何手写规则或类名匹配。换个模型照样成立,这是通用性的关键。
2. **site(站点/算子组)**:同一 family 的全部模块路径排序后取哈希 = `site_id`(:57)。
   一个 site 就是"一次替换要同时接管的那一组模块"——替换 RMSNorm 站点 = 73 个实例全换。
3. **复合站点(composite,可选)**:默认只收叶子模块;开 `allow_composite` 后,一个**包装模块**
   (如整个 `mlp` 块:gate/up/down 三个 Linear + 激活)也能当站点,children 的 CUDA 时间
   记入包装者(:179-194,否则复合站点会显得比它自己的叶子还小,排名就反了)。
   复合站点的意义是允许 agent **跨算子种类融合**。

## 4. 五道资格门:能被打包的才算候选

不是每个组都能当任务。判定复用同一个函数 `build_target`(`fast_sites.py` 只是逐个模块去问它,
被拒的组从不进入排名——**门只写一处**,排名不可能选出一个后面必然失败的站点):

| 门 | 判什么 | 实现 |
|---|---|---|
| ① 叶子/复合 | 默认只收叶子;包装模块须显式开 `allow_composite` | `fast_target.py:148` |
| ② 纯张量(KV cache) | forward 签名里带 cache 的模块无法抓成纯张量 fixture ⇒ 没有本地证明 ⇒ 拒 | `subtree_is_pure`(:83)——实测 `self_attn`/decoder 层/整模型都栽这道,MLP 干净 |
| ③ 几何代表 | ttft 需要恰好 1 个无歧义的代表调用(prefill);decode 目标需要 2 个(step 0 和 127)。形状对不上号 ⇒ 拒 | :174-177 |
| ④ 不嵌套/不重叠 | 复合站点成员不得互相嵌套;`register_site` 拒绝已属于别组的模块 | :167、`binding_runtime.py:45` |
| ⑤ fixture 容量 | 抓下来的输入/权重 fixture 不能超过显存预算——**按 `mem_get_info` 空闲量的 10% 实测推导**(box4 实测 2362 MiB),不是硬编码字面量 | `fixture_budget.py:75` |

> ⚠ 历史教训:门⑤曾是五个未测量的硬编码字面量(64/64/64/128/128 MiB),把排名第一的
> up_proj 等三个大站点**结构性挡死**,白烧了 3 次 agent 才发现。通用实现里**容量必须实测推导**,
> 且进程内只测一次并计入 `framework_id`。

另外还要求该组**有可测的关联 CUDA 时间**(`associated_cuda_us > 0`),否则替换它没有意义。

## 5. 排名与选择:实测优化空间 > 体量

这是最容易做错的一步,C3 的答案是三层排序(`fast_sites.py` `rank_key`:282),**每个 goal
独立排一次**:

```
第 0 层:短探针车道实测出正增益的站点,按增益从大到小   ← 最强证据
第 1 层:没探过的站点:先「已证可达」再「未知」,层内按 associated_cuda_us 从大到小
第 2 层:实测无空间的站点(探针增益=0,或等价性证据判「不可达」)——排最后但仍可手动强选
```

三个关键概念:

- **associated_cuda_us(体量)**:该站点的 kernel 在 profile 里实测占的 CUDA 时间。
  这是最直觉的排序依据,但**单独用它是错的**,实测过:
- **reachable(可达性,B20/B21 证据)**:在站点上绑一个**数值等价**的 Triton 替换,
  看生产 quality 门给多少分(旁边必须放一个读数恰为 0.0 的 torch 对照)。up_proj 体量是
  RMSNorm 站点的 6.3 倍(88,822 μs vs ~14,041 μs),但等价 GEMM 的门分 0.0139 > 限 0.01
  ——**任何正确的 GEMM 替换在这里都过不了门**,车道派过去只会烧完整个窗口重新发现这件事。
  而 RMSNorm 能以 0.0 过门,是因为它把 mean/rsqrt 留给 ATen、只写逐元素 kernel,舍入序列
  与 eager 逐位一致——GEMM 没有这条路(点积就是算子本身)。
  (注意:该判定所用的 0.01 是硬钉值;后续 B25 实测自校准 floor=0.0177,大 GEMM 站点可能翻案
  ——见 memory `gemm-unreachable-verdict-used-hardcoded-limit-not-self-calibrated-floor`。)
- **probed_gain_frac(实测增益)**:先派一个**短探针车道**到站点上真优化一小轮,记录
  `(baseline−best)/baseline`。这是"这个目标在这里有多少油水"的直接测量,按 goal 记录。
  实测正是它推翻了体量排序:norm 站点三目标全正(ttft +4.52% / single +12.16% / multi +5.29%),
  而体量最大的 up_proj 在 multi 下是 **−7%**——**承载 CUDA 时间 ≠ 可优化空间**。

证据以 JSON 文档形式落盘(`Reachability` 记录,按 `site_id` + `goal` 键),排名时读入。
**证据只改顺序,永远不改正确性门**;无证据时退化为按体量排,这正是没测之前的行为。

选择本身(`select_sites`:357)= 每个 goal 取自己排名第一的合格站点,外加两个善后:

1. **跨 goal 冲突解决**(`resolve_overlaps`:322):两个 goal 选的站点必须**相同或不相交**
   (一个选 `mlp`、另一个选 `mlp.down_proj` 会让绑定注册失败)。被包含的一方**提升**到
   包含它的站点(该站点本来就在它的 `considered` 名单里),并在 `promoted_from` 记录原因。
2. **强制指定**(`forced`):复现旧实验或回答"这个站点会怎样"时可以钉死站点,即便证据说
   它不可达也照用(这就是"强选"的含义),但记录里保留警告和它当时击败/未击败的完整排名。

每个 goal 的选择结果(`SiteChoice`)带上**完整的 considered 列表**——读者能看到这个选择
是在什么对手面前做出的,而不是只信一个赢家。

## 6. 产出:一份 KernelBench 式的任务包(OperatorBrief)

选定站点后,`build_target` 把它打包成 agent 可直接消费的任务(`fast_target.py:247`):

- **reference_source**:模块类源码 + 实录的 forward 源码;复合站点还附 children 的元数据
  清单和类定义(每个类只放一次——三个 `nn.Linear` 重复贴会浪费 200 行 prompt,实测导致过
  一次 1500 s 传输超时);再按 AST 递归补齐引用到的同文件定义,使 reference **自包含**。
- **representative_calls**:代表调用的真实张量形状/dtype(≈KernelBench 的 `get_inputs`)。
- **numerical_semantics**:必须保持的数值语义(逐个中间 cast/舍入/归约顺序、dtype/shape/
  strides/别名、不得改权重)——正确性门的口径随任务一起交付。
- **site_id / module_paths / associated_cuda_us / profile_sha256**:身份与证据指纹,
  保证后续任何测量能对回"当时选的是什么、依据是什么"。

campaign 启动时会**重建一次 target 并核对 site_id**(`build_selected_target`:414)——
排名和开跑各自调 `build_target`,若两者不一致就是在搜一个从没测过的站点,直接报错。

## 7. 移植到主框架时的注意点

1. **模型相关 vs 通用**:分组(family_key)、五道门、排名、冲突解决、任务包生成都是通用机制;
   与 Qwen3 绑定的只有目标合同的 prompt 集和 runner(怎么跑这个模型)。移植时输入接口
   = "一个能被逐模块 hook 的 forward + 一份 GoalSpec 合同";用户的自然语言目标 → GoalSpec
   的翻译是新增的一小步(方向/单位/负载形状三件事)。
2. **顺序上先便宜后贵**:profile(一次 GPU 短窗口)→ 分组+资格门(纯 CPU)→ 可达性探针
   (每站点一次等价替换绑定)→ 短探针车道(小额 agent 预算)→ 正式 campaign。每层证据
   只为下一层省钱,且全部落盘可追溯。
3. **别重蹈的三个坑**:容量门用实测不用字面量(§4);"不可达"判定的限值要用自校准 floor
   不要硬钉(§5);证据文档按 (site_id, goal) 双键——一个站点在不同目标下的证据不可互相污染。

## 相关文件与记忆

- 源码:`examples/c3_qwen3/fast_sites.py`(分组+排名+选择)、`fast_target.py`(资格门+任务包)、
  `device_profile.py`(profile 采集)、`fixture_budget.py`(容量门)、`binding_runtime.py`(站点注册)。
- memory:`hub-c3-floor-sites-and-device-proof`(五道门与站点排名的 24 条实证)、
  `norm-site-positive-gain-all-three-goals-2026-09-22`(体量≠空间的核心对照)、
  `four-hardcoded-capacity-literals-excluded-the-best-sites`、
  `gemm-unreachable-verdict-used-hardcoded-limit-not-self-calibrated-floor`、
  `d-site-ranking-by-measured-gain-and-ttft-two-site-probe`(probed_gain_frac 的由来)。
