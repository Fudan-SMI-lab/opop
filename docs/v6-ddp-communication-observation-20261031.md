# DDP 精确 Agent 代码的一次固定通信观测

通信优先保持。终审后的92279探针允许在当前固定API/request上验证其可观测范围；不宣称通用observer正确性。下一步声明为一次串行B-observed→C-observed，使用现有LocalExecutor/device leases和原pair0实际request，每rank原evaluator一次。无新Agent/search/参数探索/Nsys/observer-off重复/优化代码/生产采纳。Qwen、最新拒绝FSDP和77ac修复链保持关闭。

精确probe SHA`92279e28dc1e548efe86ef44467a3b0d09f7d94586da3dc16c0a3f55b773db96`，46,532B；plan SHA`0a82e46248377b94a7ec21954eb34d2bdf556d65796557556923ebf9eeefcf13`，40,074B。[前置终审](v6-ddp-observation-mechanics-repair-20261031.md)及一次CPU夹具证明部分报告/恢复机制，不是GPU质量。代码不手改，plan过度承诺不采纳。

B/C原request为`ddp-communication-feedback-20261030/ddp-search/jobs/trial-d06aa22e2d-evaluation-f25a37c6-pair-0-{parent,candidate}/run/request.json`。只改入口/function/output/wait并增加diagnostic_context；原function/wait本来run/true。原materialized source、bundle、adapter、只读controls、设备、task/precision/bindings/quality/同步/owners保持。B空bundle，C单一ccd90/emptyparams，不复制source/controls/reference、不发布pointer；independent_reference=null。

两rank原GPT2 12层768hidden、B1/S128/accum4、FP32/eager/noTF32/dropout0、完整FW/BW/SGD、loss/fullgrad/updatedparams/finite/exactreplica及1e-6/1e-4保持；两连续quality、两warmup、五样本，原三独立通信profile/rank仍在样本后。B gate仅两rank原valid/mathtrue/2checks/5samples、entry returned、六patch恢复且无observer errors；时间和join完整性不是接受门。B失败则C not_run。实际失败/慢样本/unknown全部保留，不重跑到通过。

原锚点133.379621→140.04860205ms仍记录，但本次observer扰动下的时间只作诊断，不进行接受或observer成本扣除。CPU callback不等于GPU ready；Graph无Pythonzero_grad时允许unknown；重复index映射与bucket顺序只作代码假设。同一trace可验证submission/launch/kernel范围，但不能推wire time、GPU消费完成或关键路径。新观测不覆盖原接受/独立结论，也不把旧计算Graph收益称通信成功。

空间：原pair0 B/C实际du合计147,914,752B，加冻结框架/新元数据预估512MiB，不是输出/API cap。GPU空闲后18:41:39–18:42:32UTC服务器端压缩已结束拒绝FSDP trial的216份旧metric JSON：2,371,442,286→215,563,783B，逐压缩/解压SHA验证，1944保护文件及1565其他文件不变。实际净释放2,155,651,072B，余10,920,505,344B约10.17GiB。[小摘要](evidence/v6-ddp-communication-observation-20261031/storage-summary.json)。仅旧raw metric冷存，原拒绝/全部样本/源/controls/accepted/privatehistory不改；缺plain JSON不是丢失。恢复在服务器`storage-audit-20261031/ddp-observation-fixed/run/RESTORE.md`，journal SHA`22239c9c01dfab8c752251ea81ef3f2f264445acf85803ac46981623e91ed333`；旧journals各自独立。无raw本地传输。

启动前再次核实150GiB df、双卡空闲、冻结身份和现有API。新长任务只一个隐藏完成/磁盘监听，若短批次在预检前结束则不创建重复监听。终态需分开审计原质量、实际路径、观测错误/unknown、每条raw trace及完整身份；controller完成不是诊断正确性或通信接受。实际启动另记。
