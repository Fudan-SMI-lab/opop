# DDP 精确 Agent 代码的一次固定通信观测

后续[普通selector终审](v6-ddp-observation-feedback-selection-20261031.md)确认未读取新增观测文件，不能将其新选择归因于消费了本批结果。本批原质量、unknown关联、12 trace及观察扰动限制不改，不重复观测或延长修复链。当前声明一次原选择candidate生成，未新增GPU质量/接受。

已于18:47:43UTC（香港02:47:43）completed，父回合直接接续完成审计，监听14060排队事件已标handled/不重复处理。[终态摘要](evidence/v6-ddp-communication-observation-20261031/public-terminal-audit.json)，服务器完整audit SHA`2f976a911e241e2471548c7dfbd32ab82ae2a87f2ef2b37836c72af1d7f00439`。977冻结身份、真实request/core/原source/bundle/adapter/controls/current保持；4rank workers均passed/valid/mathtrue，8次完整比较各248,879,620元素零误差，20rank样本全部保留。

按各重复先取rank最大，观测B样本118.284116/116.387054/116.474292/125.634750/124.475389ms，中位118.284116；C73.659578/73.250891/73.128837/73.860802/72.125134ms，中位73.250891。均受observer扰动，只作描述；不算新收益或接受、不减observer成本。C两次quality仍为parent_eager_baseline和graph_setup_capture各一次，不是两次优化路径quality。setup约20ms安装不等于第二quality的Graph捕获成本。

每条原始trace独立重算，共12条、156个collective identity/同线程唯一innermost c10d/launch关联、kernel sum/union/overlap/sync数量核对。每窗口13AllReduceSumf32RING_LL、逻辑497,759,232B（12×28,351,488+157,541,376），B3999kernel/0GraphLaunch，C4106/2GraphLaunch。独立profile的rankmax通信中位B48.971275/C36.185936ms，不是wire time/关键路径或可从clean时间扣除的成本；没有新Nsys/物理transport日志证据，不把旧SHM记录当本次捕获。

关键范围差异：B每rank有12个Python锚点（2quality/2warmup/5measure/3profile）、7104 callback，3条结构关联；C每rank只有首个quality锚点、592 anchored/3256 unassigned callback，296广播中148标记unanchored precedent，quality_count_mismatch=true，3条profile关联全部unknown。原evaluator实际两次质量通过，与探针只识别一次不矛盾；不得将3256回调按序号强配、推导GPU ready或伪造Graph phase。raw callback全序列未保存，聚合摘要不能全独立重算。

每rank148 hooks移除、六patch报告restored、无observer_errors/截断丢弃，artifact约137KB(B)/94KB(C)。GPU恢复依据生成代码/报告及独立CPU身份检查，不是另加GPU身份logger。DDP弃用/barrier/loss/profiler及C的stale AccumulateGrad stream warning保留，不是此次原质量失败。teardown bucket分配和owner生命周期、通用存储/异常/并发保证仍有限。

此固定观测和mechanics修复链收尾，不再增加probe修复或observer-off重复。继续一次[原始观测反馈下的普通项目selector](v6-ddp-observation-feedback-selection-20261031.md)：只交付精确Agent probe/plan、新B/C原始request/result/diagnostic及trace索引；不交付上述审计表/因果结论，不手选算法。旧CPU事实辅助和original反馈保留；没有自动候选/GPU/优化接受。

实际18:46:26UTC（香港02:46）已启动，冻结`d9f57a815e16038be01d5bae598f3ffc8e263d0c`，controller190831，唯一隐藏完成/磁盘监听14060；旧48592已handled且进程不存在。启动余10,910,814,208B约10.16GiB。

18:47:29UTC[实际核验](evidence/v6-ddp-communication-observation-20261031/startup-verification.json)SHA`3307c3f3eed4754529047a8befd5e2f1663b38ff818ce7065759329d95600a9c`：977冻结身份/core保持，B/C两份实际worker request与原request仅声明的接线差异；entry为精确92279，原function/wait确为run/true。B通过既定gate，C两GPU worker191287/191288正在执行，余10,779,181,056B约10.04GiB。此为启动身份/活动，不是终态完整质量、观测正确性或接受证明，后续需审查全部实际输出与trace。

通信优先保持。终审后的92279探针允许在当前固定API/request上验证其可观测范围；不宣称通用observer正确性。下一步声明为一次串行B-observed→C-observed，使用现有LocalExecutor/device leases和原pair0实际request，每rank原evaluator一次。无新Agent/search/参数探索/Nsys/observer-off重复/优化代码/生产采纳。Qwen、最新拒绝FSDP和77ac修复链保持关闭。

精确probe SHA`92279e28dc1e548efe86ef44467a3b0d09f7d94586da3dc16c0a3f55b773db96`，46,532B；plan SHA`0a82e46248377b94a7ec21954eb34d2bdf556d65796557556923ebf9eeefcf13`，40,074B。[前置终审](v6-ddp-observation-mechanics-repair-20261031.md)及一次CPU夹具证明部分报告/恢复机制，不是GPU质量。代码不手改，plan过度承诺不采纳。

B/C原request为`ddp-communication-feedback-20261030/ddp-search/jobs/trial-d06aa22e2d-evaluation-f25a37c6-pair-0-{parent,candidate}/run/request.json`。只改入口/function/output/wait并增加diagnostic_context；原function/wait本来run/true。原materialized source、bundle、adapter、只读controls、设备、task/precision/bindings/quality/同步/owners保持。B空bundle，C单一ccd90/emptyparams，不复制source/controls/reference、不发布pointer；independent_reference=null。

两rank原GPT2 12层768hidden、B1/S128/accum4、FP32/eager/noTF32/dropout0、完整FW/BW/SGD、loss/fullgrad/updatedparams/finite/exactreplica及1e-6/1e-4保持；两连续quality、两warmup、五样本，原三独立通信profile/rank仍在样本后。B gate仅两rank原valid/mathtrue/2checks/5samples、entry returned、六patch恢复且无observer errors；时间和join完整性不是接受门。B失败则C not_run。实际失败/慢样本/unknown全部保留，不重跑到通过。

原锚点133.379621→140.04860205ms仍记录，但本次observer扰动下的时间只作诊断，不进行接受或observer成本扣除。CPU callback不等于GPU ready；Graph无Pythonzero_grad时允许unknown；重复index映射与bucket顺序只作代码假设。同一trace可验证submission/launch/kernel范围，但不能推wire time、GPU消费完成或关键路径。新观测不覆盖原接受/独立结论，也不把旧计算Graph收益称通信成功。

空间：原pair0 B/C实际du合计147,914,752B，加冻结框架/新元数据预估512MiB，不是输出/API cap。GPU空闲后18:41:39–18:42:32UTC服务器端压缩已结束拒绝FSDP trial的216份旧metric JSON：2,371,442,286→215,563,783B，逐压缩/解压SHA验证，1944保护文件及1565其他文件不变。实际净释放2,155,651,072B，余10,920,505,344B约10.17GiB。[小摘要](evidence/v6-ddp-communication-observation-20261031/storage-summary.json)。仅旧raw metric冷存，原拒绝/全部样本/源/controls/accepted/privatehistory不改；缺plain JSON不是丢失。恢复在服务器`storage-audit-20261031/ddp-observation-fixed/run/RESTORE.md`，journal SHA`22239c9c01dfab8c752251ea81ef3f2f264445acf85803ac46981623e91ed333`；旧journals各自独立。无raw本地传输。

启动前再次核实150GiB df、双卡空闲、冻结身份和现有API。新长任务只一个隐藏完成/磁盘监听，若短批次在预检前结束则不创建重复监听。终态需分开审计原质量、实际路径、观测错误/unknown、每条raw trace及完整身份；controller完成不是诊断正确性或通信接受。实际启动另记。
