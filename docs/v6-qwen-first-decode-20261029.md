# Qwen真实同输入首decode数值诊断

四种不同生成实现复现逐元素相同的错误greedy输出，既有事实不足以把原因归给单Graph/多Graph、pool共享、clone或某一种cache设计。本批缩小到第一个已出现差异的decode，直接检查共同执行路径，不继续让Agent根据同一猜测改写代码。

使用未修改的fresh adapter498c90f3准备真实Qwen3-4B，原BF16/SDPA/noTF32、模型资产和源参考不变。先跑一次原完整32-token baseline并与完整源参考核对，保存原始输出；再运行原prefill一次，保存完整36层prefix KV、实际token和prefill logits检查。共享同一模型和同一份token/prefix，以固定顺序运行 **DynamicCache eager、DynamicCache Graph、StaticCache eager、StaticCache Graph**，各一次首decode。

每个模式从同样数值的prefix副本开始，核对全量值SHA；DynamicCache保留实际257长度，StaticCache使用原候选长度287但仅前257位置有效。StaticCache使用库本身mask，无开发者重写attention。Graph使用3次独立warmup、单私有Graph捕获及一次replay；Static warmup后恢复完整prefix/清空tail/reset累计长度。所有工作使用当前token和KV，不缓存最终答案。记录完整首decode logits/token、更新后KV及逐层差异、prefix是否被改变、原失败greedy首decode是否复现。KV差异只作描述，不创造隐藏状态质量阈值。

同进程固定顺序有分配/库缓存和warmup状态影响，不能消除所有混杂；CPU读取和持有KV用于诊断，也不能代替未观测运行。该小型对照只覆盖首decode，不能建立完整32-token候选正确性、性能、lifetime或泛化接受。没有candidate安装/搜索、额外Agent角色、profile/Nsight、计时优化或生产修改。

另从实际token和teacher切片计算tensor_description，并与冻结binding signature比较，记录stride和匹配predicate；这是源构造与代码分派控制，不把它冒充旧运行逐调用事件。旧teacher缺少route本身不能证明回退。

控制器`scripts/experiments/v6_qwen_first_decode_20261029.py`；目录`/root/autodl-tmp/dev/v6-work/qwen-first-decode-20261029`。预检核对已完成B控制、27项冻结身份、adapter/参考、执行请求schema和core。实际CUDA执行再验证新增探针；接口预检不是数值通过。逐arm错误保留；不改旧candidate或重启已完成批次。

实际150GiB配额余约12.22GiB，预计raw输出约200MiB（完整baseline、prefix和4组更新后KV），留服务器。没有本地raw下载、引用复制或清理；只取小型摘要/索引至GitHub。启动前配置一个普通完成/8GiB监听，无Goal或模型轮询。后续仅根据真实对照结果决定下一步，不自动把诊断写入生产提示或接受候选。


## 已完成并复核（2026-10-28 21:26 UTC）

冻结9c78364d、控制器PID891621，21:26:03启动、21:26:19完成。该短计算在本会话内直接等待工具完成并审计，没有启用新的完成通知或模型轮询；上述监听安排用于后续长任务。见[启动](evidence/v6-qwen-first-decode-20261029/launch.json)、[完成](evidence/v6-qwen-first-decode-20261029/followup-completed.json)、[全量原始值重算审计](evidence/v6-qwen-first-decode-20261029/terminal-audit.json)。

完整baseline与源32-token logits/tokens零误差；重新prefill亦零误差。四模式token均为20772、全量36层prefix值SHA6337e260一致，全部执行后原256位置不变，实际有效长度257。结果如下：

| 首decode模式 | 对源最大logits误差 | token | 与旧失败首decode逐元素相同 |
| --- | --- | --- | --- |
| Dynamic eager | 0 | 相同 | 否 |
| Dynamic Graph | 0.1875，未通过原阈值 | 相同 | 是 |
| Static eager | 0.1875，未通过原阈值 | 相同 | 是 |
| Static Graph | 0.1875，未通过原阈值 | 相同 | 是 |

后三者完整logits逐元素相同；完整updated KV也留存并按层重算，layer0 KV与Dynamic eager相同，从layer1出现差异。该数据把范围缩小到共同执行路径，但不能唯一指定attention backend、mask、GQA布局或算术原因。Static eager本身复现差异，说明Graph replay并非该差异的必要条件；全部绕过framework binding仍复现，绑定亦非必要条件。没有据此认定某个完整候选可接受。

实际greedy输入stride[1,1]，teacher切片[32,1]；其余签名构造相同，冻结spec只匹配前者。结合binding按signature分派的源码，这解释了为什么不能假设两侧调用同一个replacement；这仍是当前实际输入/代码predicate控制，不是历史逐调用trace。旧teacher缺少route不是单独证据。

数学策略不变，所有raw哈希、full baseline、prefix、四模式logits/updated KV及交叉比较均CPU独立重算，未再次运行模型。实际剩12,913,614,848字节（12.03GiB）。下一步见[首层attention证据及一次真实候选续行](v6-qwen-attention-evidence-20261029.md)，只补新的局部证据，不重复既有角色A/B。
