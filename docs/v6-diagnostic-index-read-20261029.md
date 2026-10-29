# 针对未读取诊断索引的有限提示实验

前两次candidate重放都有索引文字可见，却没有打开索引或其中的观察文件。这是已经实际确认的输入消费缺口。本次从那两份**生成前初始快照**分别重放一次既有candidate角色，唯一变化是在prompt末尾追加同一通用段落（含分隔换行433 bytes）。32/37个输入文件、原失败feedback/候选/selection、model/tool配置和core全部保持，不输入新固定GPU结果、审阅报告、人工事实/原因或修复实现。CPU快照预检通过，122冻结文件核验。

> When prior feedback supplies an evidence index, open that index and inspect the relevant observation artifacts and their producing code before choosing a repair or asserting a cause. In the rationale, cite the inspected artifact and location, distinguish directly observed facts from hypotheses and instrumentation limits, and state any remaining evidence gap; an index pointer or a diagnostic status alone is not a quality result.

两任务各一次真实请求，不追加第三个reviewer、不重复核验到通过。这是与历史原样prompt的有限对照，单个输出差异不证明稳定因果优势；任务之间不是单变量A/B。此前accepted-feedback措辞研究和verifier研究仍关闭。本批段落仅为隔离实验，**production prompt、sampler、allocator、接受规则不变**，core仍 `ed24af7a…`。

Qwen初始快照generation-9e85d07b98af，历史失败a866；FSDP初始generation-433d566258dc，历史失败86e的df3d代码与原多参数feedback。原selection可能含错误推断，不手动修正；Agent需要自己把原反馈与已有观察范围核对。索引仅列原始来源/参考、Agent编写的诊断代码与固定B/C产物，没有采用过的解释/核验答案。FSDP新诊断只涵盖K6，不外推其他参数。环境文件系统可访问，终态必须审查完整工具路径/输出/脚本，不能只看初始文件名或关键词断言无污染。

判断重点是实际是否打开索引、阅读了哪些证据和产生证据的代码，是否引用准确、区分观察/假说/工具效应、保持工作量/精度/质量/约束，以及新代码和参数究竟改变什么。输出schema通过不是质量或推理正确；CPU/mock/语法探针和异常各自保存，pipeline exit0不掩盖内部失败。保持API observe-only，不设置新调用上限。控制器不安排GPU/model/search/接受；原candidate工具可自行探针，需审查实际行为后才能报告活动范围。

若未读取或仍有事实错误，保留结果，不重复直到成功；若有可用代码，也须先审查再决定是否需要声明的原样固定执行。两份既有固定代码已通过质量但未正式AB/BA，本批不以它们充当新优化接受，不追加独立重测来替代工作流缺口。没有自动生产collector/角色或固定失败路由。

控制器 `scripts/experiments/v6_diagnostic_index_read_20261029.py`；服务器目录 `dev/v6-work/diagnostic-index-read-20261029`。启动前核查真实150GiB容器df/GPU与冷维护结果。raw和private配置留服务器，只上传代码/哈希/小型脱敏回执到GitHub v6；新批配一条普通完成/8GiB监听，禁止Goal、模型轮询和额外投递测试。

启动前授权冷维护已完成：仅旧DDP/ResNet明确worker RUN JSON，229 payload/1,601保留文件/10当前版本/6参考校验，实际净释放2.63GiB，余11.11GiB。恢复journal独立保存在 `storage-audit-20261029/diagnostic-index-read/run/`；本批将访问的索引、观测、参考和源码均未压缩。见[上批审计及维护](v6-diagnostic-index-fixed-20261029.md)。
