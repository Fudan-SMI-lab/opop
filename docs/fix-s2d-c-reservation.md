# S2d(c) 现在可测:为「已有账本的族」保留一个改写槽位

**状态:已实现(`da1c39b`),默认关闭,未跑实验。** 这是把
`docs/result-s2d-c-never-administered.md` 的结论从「不可测」变成「可测」的那一处改动。

## 为什么之前不可测

账本按**族**存,所以 rewriter 只有在**同一个族**的第一轮被 reconcile 之后**再拿到一轮**时才会收到账本。
实测:本项目至今每一个 run —— 已完成 run 的 9 个族加上配对实验的两臂 —— **每个族都只拿到了 1 轮**。
所以传给 rewriter 的账本参数在**每一次调用**上都是 `[]`,**两臂都是**,与 `expectation_ledger` 开关无关。
**两臂在 S2d(c) 上完全相同,它们的延迟差不是关于它的证据。**

算术上每次都差**一个族**:4 个种子族、12 h 装得下 3 轮,而 `active_families` 规则 1 把每个
**从未改写过**的族排在最前,所以队列永远排不完。

## 改了什么

`v3.diagnosis.reserve_round_for_reconciled`(**默认 False**,且要求 `expectation_ledger` 同时为真):
`max_families_active` 里的**一个**槽位可以给已有账本的族 —— **但仅当同一轮里仍有一个未改写过的族拿到
槽位**。

**规则 1 的受保护场景被完整保留**:`run-l3-43-20260904-093730` 上排名第一的族在
[19.6, 19.6, 19.6] 卡住,而排名第二的走 [19.5, 17.9, 17.9] 并产出了那个 run 的冠军 ——
`max_families_active=1` 下按延迟排名会**直接删掉冠军**。所以保留槽位只让出**若干槽位中的一个**,
并且在候选名单里只剩一个未改写族时**主动放弃**(宁可把 S2d(c) 推迟一轮)。

**默认关闭是刻意的**:它改变族被改写的**顺序**,所以开了它的 run 与三个已完成 run、与配对两臂
**都不可比**。它是**实验的一个臂,不是修复**。

## J2d-8 没有被放宽

选择模块只知道**某个族有账本**,永远不知道账本**说了什么**。字段是
`families_with_a_ledger: set[str]` —— 按它**装的东西**命名,而不是按填它的事件命名(这也更准确:
`n_declared=0` 的条目 reconcile 不了任何东西,但仍然是下一轮 prompt 可以携带的账本)。

`test_j2d_8_...` 那条**文本禁令**照旧生效(`families.py` 里现在一个 "expectation" 都没有),
并且额外加了**行为**守卫:`test_selection_cannot_see_what_the_ledger_SAYS` 用「全命中」和「全未命中」
两种账本驱动 `active_families`,断言两次候选名单**逐项相同**。`test_the_set_is_only_ids` 断言类型是
`set`,这样后来若有人改成 `dict[str, ...]`(命中率就可从分配逻辑里读到了)会在这里失败。

## 构建过程中被既有测试抓到的两个缺陷

1. **集合更新写在了 `_record_reconciliation` 的 `except Exception` 里面**,而且在 ledger append
   **之后**。一个没有该属性的 `families` 协作者会抛 AttributeError,被 blanket except 吞掉,于是那一轮
   的**第二条 ledger 条目根本没被追加** —— `test_two_candidates_in_one_round_are_reconciled_SEPARATELY`
   从 2 条掉到 1 条,而**任何地方都没有报错**。**一个辅助字段绝不能损坏它所描述的东西。**
   与今天早些时候 prescreen 超时的缺陷**同形**(见 memory
   `a-prescreen-timeout-is-a-cost-cap-not-a-space-restriction`)。已把计算移出 try,并加测试钉住。
2. **一个 `max_families_active < 2` 的守卫是死代码**:它的 revert 变体**不改变任何行为**,因为
   「未改写族计数」守卫已经覆盖了它(1 个槽位 ⇒ `unproven_idx` 最多 1 项 ⇒ 必然放弃)。
   **删掉而不是留着** —— 那是 `a-variant-that-changes-no-behaviour-is-not-a-variant` 在说
   「这个分支是死的」,而不是「测试太弱」。

## 我自己写的两个「什么都没断言」的测试

- **wiring 测试在测试体里重算了那个 `and`**,于是它在**任何** `wiring.py` 上都通过 —— 它的 revert
  变体把真实的合取式删掉了而测试毫无反应。已改为 monkeypatch `FamilyManager` 后**驱动真实的
  `build_orchestrator`**,读它实际传出的 kwarg。
- **空族测试用的那个族本来就排在最后**(`best=None` 让 `_incumbent` 返回 inf),所以删掉
  `best is not None` 过滤器不会改变一个已有两个更好族的 2 槽名单。已补一个 `active=1` 的场景,
  那里排除是唯一挡住它的东西。

## 守卫

15 个测试;`scripts/revert_check_s2d_c_reservation.py` 的 **9 个变体覆盖两个方向** ——
**保留失效**(S2d(c) 仍不可测)与**规则 1 被饿死** —— 在 A800 上**全部 CAUGHT**。
变体由 `scripts/gen_revert_check_s2d_c.py` 生成,该生成器在**每个 anchor 都确认存在于目标文件**之前
**拒绝写出**(多行 anchor 经 shell heredoc 会被改写,而不生效的 patch 曾两次被读成 pass)。
A800 全套:**828 passed, 1 skipped**。

## 下一步(未做)

要真正**得到** S2d(c) 的证据,还需要一对开了这个开关的 run。注意它**不能**与现有三个 run 或配对两臂
合并比较 —— 搜索顺序不同。最省的形态是**再跑一对 12 h**(处理臂 = vector + ledger + reservation,
对照臂 = vector + ledger),这样自变量恰好是「账本是否到达下一轮 prompt」。
