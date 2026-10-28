# Qwen事实反馈对照生成代码的固定GPU验证

[真实角色A/B审计](v6-qwen-relation-feedback-replay-20261029.md)表明新增事实被处理组读取，但两份实际生成代码都还没有真实完整GPU质量证据。此批仅验证原样代码，不改候选、不补实现答案、不开展新的自主搜索或正式性能接受。

固定顺序为 **B-observed、control-unobserved、control-observed、treatment-unobserved、treatment-observed**，每端点一次，不依结果重试。C/T使用两臂最终candidate.py及proposal.json的确切字节，两者自报参数空间都为空，故使用`{}`，没有开发者选参。使用既有bundle_from_outputs和stage，从同一已完成discovery、空bundle的`b61996e1`父版本建立隔离分支；不再次发现、不生成controls或参考，不发布任何current指针。

Control SHA `777488e03148618b7a56f2c56302d1e99c5abced778f0f56129040adaf0f4931`；treatment SHA `313aa7bc7818fa72269b00134d91096102692fb33b56c0c3bb5c96060141d562`。保留两份不同的plan声明，包括control空eliminated_calls和treatment23,225条声明，不人工补齐。审查执行错误、完整质量、覆盖和约束分别报告；不从route名推断native kernel。

完整batch1/prompt256/32 greedy tokens、BF16/SDPA/noTF32、全部logits0.02/0.02和exact tokens、两端同步、原state生命周期、2连续质量步、2warmup/5clean repeats均保留。原adapter498c90f3与controls共享只读。TTFT上限固定为此任务原controls24.051052×1.05=25.2536046ms，不能换成旧adapter变体的25.49943795。没有改变math、参考或阈值。

U使用原evaluate entry。O复用已验证的外部observer（SHA a06b6c0e），不修改生产core：保存完整greedy、teacher、observe后greedy CPU张量，比较各自对独立源及相互关系，并阶段标注候选route。它改变host工作/同步/CPU生命期；O计时仅诊断，不扣除开销。B观察fixture亦按固定协议执行，若失败原样保留。某候选可能在clean之前失败，controller completed不能写成quality pass。

此批不采大型profile或Nsight，不因schema有效、CPU toy通过、role完成或worker完成而接受候选。若质量通过，也只建立固定执行证据，后续完整独立评测/原生路径仍需单独决定。原始样本和失败全部保存；同输入CPU/CUDA比较器计数差与相对阈值方向须分开报告。

控制器`scripts/experiments/v6_qwen_feedback_fixed_validation_20261029.py`，目录`/root/autodl-tmp/dev/v6-work/qwen-feedback-fixed-validation-20261029`。预检已核对生成源码、初始输入、27项旧源码/controls/request身份、原执行请求和observer SHA；生产core仍ed24af7a，正式执行前再次核对。现约12.46GiB可用，预计观察张量约百余MB；不做本地raw转存。GPU结束后才允许授权无损冷压缩；当前无需清理。唯一普通终态/8GiB监听，不启用Goal或模型轮询。
