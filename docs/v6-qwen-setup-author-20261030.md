# Qwen内部setup失败的诊断作者

这是[当前设备普通试验](v6-qwen-device-feedback-20261030.md)之后的一次独立真实diagnostic_author生成。外层评估在父路径质量通过，但新候选两种模式都内部自检失败；现有run-only sidecar只有异常位置/传播，没有完整比较值或输入状态，无法判定原.02/.02是否超差或比较是否等价。以这个具体缺口检验作者能否编写有界观察器，不是新增候选搜索、优化接受或重新开启已关闭的prompt/verifier研究。

复用既有ProjectAgents/Runtime/快照接口；沿用通用作者instructions/interface，增加一段通用的“内部捕获异常/回退/有界比较证据”说明。六份显式输入是原task、原始candidate_diagnostics sidecar、原adapter、确切04df候选、原请求/worker/source/只读controls索引和接口，另有自动worker-environment。没有开发者terminal/public audit、fast_enabled/chain E推断、因果答案、手写修复、其他作者输出或旧成功实现。实际文件大小/快照/完整工具读取须终态审计，外部FS可访问，不承诺隔离。

此controller只生成diagnostic.py与diagnostic-plan.json并验证syntax/schema，禁止作者会话GPU/model/collective执行。CPU/library/mock允许但必须保留失败，成功mock不等于原完整GPU质量。原entry应每worker只调用一次，保持完整连续质量、warmup/repeats、bindings/math/同步/返回值及内部catch/fallback；不得启用被禁快路径、放宽自检/外层容差、重放模型或人工构造替代结果。选择小规模比较值/必要状态元数据，不dump全部KV/model；缺失与非有限值须如实可序列化，原return/exception保持。观察带来的CPU拷贝、同步、owner和生命周期扰动必须明确。

生成后先审核原样代码、真实访问/CPU测试/体积，合适才另行声明固定观察执行；不自动安装生产collector或新增审批层。不把作者schema完成当执行正确、质量或性能接受，不反复生成到通过。源/当前/旧接受版本、全部旧失败冻结；共享controls不复制，未重新做源比较，既有人工源映射边界保持。

当前设备TTFT原anchor23.285755/limit24.45004275及完整B1/prompt256/32tokens、BF16/SDPA/noTF32/.02/.02/exact tokens不变。检查实际150GiB df与live GPU，约9.60GiB余量；仅小文件会话，未安排大trace/参考副本或清理。raw/private历史留服务器，GitHub只代码/哈希/小摘要。启动后设唯一普通完成/8GiB监听，旧44008已处理；没有模型等待轮询、Goal或投递测试。
