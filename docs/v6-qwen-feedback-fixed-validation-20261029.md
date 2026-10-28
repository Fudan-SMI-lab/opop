# Qwen事实反馈对照生成代码的固定GPU验证

[真实角色A/B审计](v6-qwen-relation-feedback-replay-20261029.md)表明新增事实被处理组读取，但两份实际生成代码都还没有真实完整GPU质量证据。此批仅验证原样代码，不改候选、不补实现答案、不开展新的自主搜索或正式性能接受。

固定顺序为 **B-observed、control-unobserved、control-observed、treatment-unobserved、treatment-observed**，每端点一次，不依结果重试。C/T使用两臂最终candidate.py及proposal.json的确切字节，两者自报参数空间都为空，故使用`{}`，没有开发者选参。使用既有bundle_from_outputs和stage，从同一已完成discovery、空bundle的`b61996e1`父版本建立隔离分支；不再次发现、不生成controls或参考，不发布任何current指针。

Control SHA `777488e03148618b7a56f2c56302d1e99c5abced778f0f56129040adaf0f4931`；treatment SHA `313aa7bc7818fa72269b00134d91096102692fb33b56c0c3bb5c96060141d562`。保留两份不同的plan声明，包括control空eliminated_calls和treatment23,225条声明，不人工补齐。审查执行错误、完整质量、覆盖和约束分别报告；不从route名推断native kernel。

完整batch1/prompt256/32 greedy tokens、BF16/SDPA/noTF32、全部logits0.02/0.02和exact tokens、两端同步、原state生命周期、2连续质量步、2warmup/5clean repeats均保留。原adapter498c90f3与controls共享只读。TTFT上限固定为此任务原controls24.051052×1.05=25.2536046ms，不能换成旧adapter变体的25.49943795。没有改变math、参考或阈值。

U使用原evaluate entry。O复用已验证的外部observer（SHA a06b6c0e），不修改生产core：保存完整greedy、teacher、observe后greedy CPU张量，比较各自对独立源及相互关系，并阶段标注候选route。它改变host工作/同步/CPU生命期；O计时仅诊断，不扣除开销。B观察fixture亦按固定协议执行，若失败原样保留。某候选可能在clean之前失败，controller completed不能写成quality pass。

此批不采大型profile或Nsight，不因schema有效、CPU toy通过、role完成或worker完成而接受候选。若质量通过，也只建立固定执行证据，后续完整独立评测/原生路径仍需单独决定。原始样本和失败全部保存；同输入CPU/CUDA比较器计数差与相对阈值方向须分开报告。

控制器`scripts/experiments/v6_qwen_feedback_fixed_validation_20261029.py`，目录`/root/autodl-tmp/dev/v6-work/qwen-feedback-fixed-validation-20261029`。预检已核对生成源码、初始输入、27项旧源码/controls/request身份、原执行请求和observer SHA；生产core仍ed24af7a，正式执行前再次核对。现约12.46GiB可用，预计观察张量约百余MB；不做本地raw转存。GPU结束后才允许授权无损冷压缩；当前无需清理。唯一普通终态/8GiB监听，不启用Goal或模型轮询。


已冻结`aa1ab0eda399612991f118990dfcf1658843bd25`，2026-10-28 21:13:43 UTC启动控制器PID889627（PPID1、独立session）。[启动及精确预检](evidence/v6-qwen-feedback-fixed-validation-20261029/launch.json)、[冻结协议](evidence/v6-qwen-feedback-fixed-validation-20261029/protocol.json)、[两份隔离候选身份](evidence/v6-qwen-feedback-fixed-validation-20261029/staged-identities.json)已存档，生产core不变。

部署前曾发现服务器临时上传副本缺少最后一条生成代码末尾SHA复核，版本检查在新批目录/worker创建前停止。同步精确已提交字节后才启动一次正式批次，记录见[部署修正](evidence/v6-qwen-feedback-fixed-validation-20261029/deployment-preflight-correction.json)；没有重新运行任何旧实验。监听先于launch记录启动，初次只记录launch_record_missing并等待，未发送通知；同一个监听现已跟踪真实controller。

21:14:07 UTC[启动验证](evidence/v6-qwen-feedback-fixed-validation-20261029/startup-verification.json)：两候选已成功stage，首B-observed真实GPU worker PID889699运行、占约8362MiB；此时尚无端点终态，不提前宣称质量通过。实际df余13,249,826,816字节（12.34GiB）。唯一普通完成/8GiB监听PID52576已核验waiting，见[设置](evidence/v6-qwen-feedback-fixed-validation-20261029/wakeup-setup.json)、[运行状态](evidence/v6-qwen-feedback-fixed-validation-20261029/wakeup-verification.json)。结束后自动审计全部实际结果；当前无新清理、raw本地转存或生产修改。


## 终态：五次尝试结束，四个候选端点均质量失败

2026-10-28 21:15:10 UTC完成。B-observed通过两次全量质量检查、6份完整输出与源零误差，保留5个clean样本；其TTFT24.285481ms在原25.2536046上限内，仅为观察baseline控制。Control-U/O、treatment-U/O全部在observe关系断言处失败，均没有clean样本、正式质量通过/覆盖结果或可接受性能。控制器completed不能解读为实验成功。见[终态审计](evidence/v6-qwen-feedback-fixed-validation-20261029/terminal-audit.json)。

四端点原GPU错误一致：1,132,088/4,861,952不匹配，max abs1.0，最大位置(0,6,50898)。两份观察greedy完整张量逐元素相同，且与更早C1/C2（31个Graph/shared pool版本）的完整greedy也逐元素相同；四种生成实现均token全对、logits有限，但首decode位置1已max0.1875、全程max1.0。两份新teacher与源逐元素相同，greedy在observe前后不变。CPU重算source方向/关系方向计数保持各自原值，未混同CUDA比较计数；12份新完整张量和旧两份greedy的哈希/值均复核。

两候选O各记录1次capture类回调+30次graph_replay；teacher没有该回调。这些是路径自报，不是原生kernel证据。实际源码审查发现binding按tensor stride等完整signature分派，teacher切片和greedy argmax构造可能产生不同stride；下一项记录实际输入与冻结predicate，而不从缺少route直接认定历史路径。四种代码的相同输出削弱了对某一种cache/pool机制的唯一归因，但仍不足以断定共同底层原因。

所有源码/adapter/contract/request/controls/entry SHA均核对；隔离branch和原source current仍为b61996e1、没有发布bundle。新反馈帮助treatment说明哪侧出错，但没有带来这次完整GPU质量通过，不采纳生产prompt修改，不再重复相同角色A/B。旧三轮零接受、旧同adapter46.9421%及其他历史结果均保持。

GPU均结束，实际剩13,123,629,056字节（12.22GiB），无需清理。下一步[同输入首decode诊断](v6-qwen-first-decode-20261029.md)直接使用原模型forward，对真实相同token/prefix KV做Dynamic/Static×eager/Graph四项控制，限定数值定位，不进行新搜索或接受。
