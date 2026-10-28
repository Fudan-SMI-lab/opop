# 首attention接口证据与一次候选事实续行

[首decode诊断](v6-qwen-first-decode-20261029.md)已证明，真实同token/完整prefix下Dynamic eager与源相同，而Dynamic Graph及Static eager/Graph均重现相同差异。下一步直接核查首个attention调用的输入/接口，不预设它由某个backend或GQA单独引起。

在新worker中准备同一原模型，加载经过哈希验证的完整prefix，按Dynamic eager、Static eager各做一次首decode。仅包装F.scaled_dot_product_attention，保留首调用的Q/K/V、选项、输出引用，记录36次调用结构；函数仍调用原实现。随后核对完整decode输出是否逐元素复现前批对应的无wrapper值；wrapper可能改变分配/生命期，不忽略其影响。保存完整首attention输入/输出，核查Q与按实际head数量展开、裁到有效长度的K/V值是否相同。不同padding、mask、布局和GQA选项是多个因素，不把两条接口差异自动解释为单一原因。

另外各调用一次保存的attention接口，恢复实际值、shape、stride和原选项，用小型torch.profiler采集实际CPU attention算子和CUDA kernel名，并核对输出是否与full-forward hook相同。这是两个独立的attention调用trace，不是完整模型native路径、clean计时、wire时间或关键路径证据。没有强制backend、放宽阈值、修改原candidate或原32-token任务。

只有前述worker正常且两条hook完整decode均复现原eager值，才开始一次新的真实candidate角色；否则保留错误并跳过角色。使用原处理组初始快照generation-713b51281927，原有事实字段保留，在同一个previous-feedback.txt追加一个新成员same_input_attention_diagnostic，包含本轮真实数值、输入比较、实际CPU算子和索引/限制。其他32输入、prompt、模型/工具配置、core不改；不提供手工候选、优化实现或唯一因果答案。记录新增字节量及实际初始快照，正常schema修复照旧、API observe-only。不是新的A/B优越性证明，不反复生成直到通过；生成代码本批不做独立GPU质量或性能接受。

控制器`scripts/experiments/v6_qwen_attention_evidence_20261029.py`；目录`/root/autodl-tmp/dev/v6-work/qwen-attention-evidence-20261029`。真实原始证据留服务器，GitHub只代码/小摘要/元数据。生产core仍ed24af7a，旧完整质量失败/独立拒绝/接受不覆盖。约12.03GiB可用，预计只新增几MB attention张量/小trace及会话文件；无需当前清理。长角色开始前启用唯一普通完成/8GiB监听，无Goal、模型进度轮询或投递测试。
