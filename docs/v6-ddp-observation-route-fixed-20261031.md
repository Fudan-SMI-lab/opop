# DDP 合批与单全梯度规约候选的一次固定 B/C 验证

已实际启动并完成：2026-10-30 19:47:46–19:48:48UTC（香港03:47–03:48），controller202182、冻结`02aa174617f3a2d25984b30174ac06bb4d7c825c`。短批次在watcher预检前已经terminal，故没有建立监听或重复排队；直接完成终审，不能写成有活跃后台任务。新[配对正式验证](v6-ddp-observation-route-validation-20261031.md)另行声明。

[公开终审](evidence/v6-ddp-observation-route-fixed-20261031/public-terminal-audit.json) SHA`5cbf7d78a2ef892169f897400b04f27e0e448c1532e32585c442634030880e20`，服务器完整审计SHA`6110149004424c97a130b10a1fc46e391419c877bcc73c81d1e4c31eb2d458ce`。1028冻结身份及两真实request/全部source manifest/entry/adapter/controls/core/UUID、source和新branch current核对。C staged`bbe28c40885890ef6d2c1e69bb75f1a6934178494d9d704ab56b0887e76301e6`，单`fixed-observation-route`替换旧bundle；旧文件仍可留在版本files但未安装，非双bundle共存。无接受/current发布。

四rank worker均valid/mathtrue，8完整checks各248,879,620elements、20 clean rank样本全部保留。C第二连续检查max_abs_error分别9.5367431640625e-7、1.1920928955078125e-7，在原容差内；其余6检查零误差。原observe执行完整grads/params/exact replicas，未保存全量rawtensor，不声称本审计重算所有数值或新source proof。两端质量路径均first parent/second graph_setup_capture，一次reference，未内部fallback；不是两次优化路径检查。

aligned maxrank五样本B `[71.990687,71.477893,71.369506,71.233394,71.081868]`，C `[54.668082,53.951828,54.459087,53.997229,54.657885]`ms；median B71.369506/C54.459087，MAD0.136112/0.208995。两端原140.04860205ms约束通过，但只有一B/C，不形成paired-noise/search/independent接受，不与后续配对池化。B是当前acceptedccd90，不是空基线。

12份各自工作窗口raw trace重算：B每rank/每次4106 kernels、2984Graph、2cudaGraphLaunch、13 AllReduce；C1105/1103Graph/2launch/1 AllReduce，均原RING_LL名称，NCCL在Graph外。B逻辑payload12×28,351,488+157,541,376B，C单497,759,232B，总量相同。原trace preparation barrier在workload范围外，完整保留，未错计为稳态AR。实际合并机制有证据；计算合批同时改变，整步差值不能归为独立通信贡献。NCCL sum/union各自重算，不能作wire/critical path/跨profile减法；没有新Nsight或物理transport日志，旧SHM证据不冒充本次。

C runtime peak1,528,818,688B比B1,498,647,552B增加30,171,136B；lifecycle2,427,159,040比1,999,242,752增加427,916,288B；reserved同样更高，非内存收益/新内存约束。8–10ms install不是完整第二质量调用setup/capture成本。保留两端stale AccumulateGrad stream warning，无uncaught traceback。审计attempt1仅误期望原trace标签all_reduce（实际allreduce）；原错误保存、修正reader，无GPU重跑。

下一步为原样508a的一次现有`validate_bundle(publish=False)`，正式B/C/C/B相对当前accepted父版本检验原质量/约束/MAD噪声，正常resource-feedback planner自行选已有collectors；无新候选/参数/手选collector/独立阶段。失败不重试，只有真正新正式接受才另行考虑完整版本独立收口。此处固定结果保持、不预写接受。终审GPU空闲、实际剩10,484,969,472B约9.76GiB，1GiB新预算加8GiB警戒，不清理。

声明一次现有fork_run/bundle_from_outputs/stage/execute：B为当前已接受ccd90，C为Agent原样508a。不是空基线B，不重新发现/控制/source比较，不运行Agent/search/参数探索、手选collector或Nsight，不发布current，不自动接受或重试。既有3次postclean通信profile/rank来自原请求，保持。

来源`ddp-accepted-communication-trial-20261030/ddp-search` current`1fc24545217087adefa1248704619fcdc882a19057336b231d8e02918bf4d0d8`，单旧bundle；新候选replaces旧bundle、requires空、emptyparams。candidate SHA`508ad3823d3a7b980df7076023b3567242f12a2544cb9bbeae0ada66c6f52f0a`/14,898B，proposal`853ae1eb048e1d590e5b2ba03d75dfec378a4fc20bf4a7b0faa478ba03e1de89`/9,828B。代码终审SHA`74209119a32ada66f1d298a4f3c67bbe5e1c043a5c796529986ef77ad8d2c123`支持的只是固定原任务执行资格，不是通用安全或数值通过。oldaccepted4541/independent7a49/历史Graph09b7/source1fc不改。

原两rank GPT2-12层/768hidden/12heads/50257vocab/npos1024，B1/S128/accum4/effective8seq1024tokens、FP32/eager/noTF32/dropout0/usecacheFalse、seed20261020/input31000+rank、完整zero/FWBW/lossdiv4/SGD.001无momentum/weightdecay/foreachFalse/twosync/owners保持。新实现可改变内部合批/调度，原输出全梯度、更新参数、loss/finite/精确跨rank replicas仍由未改adapter验证。2连续全量检查1e-6/1e-4、2warmup、5clean/rank；aligned maxrank先于median/MAD。原current anchor133.379621→140.04860205ms，不重锚/混合时期/加内存约束。

B须两rank原valid/mathtrue/2完整checks/5samples才执行C；B计时不作门槛。C可能只在第二质量调用尝试新路径，setup失败后的parent质量不验证fast。每个失败/回退/尾样本保留；单B/C不足paired-noise或独立接受，不据此宣布通信成功。若后续确有新的正式接受，才另行声明必要独立完整版本验证。已有候选不因成功标签自动采用。

同当前两4090 UUID20e8003c-f531-d50c-084c-a9a97fe94004与a24af2e6-99bb-9729-5d14-50fa10a2b321/driver580.105.08，adapter`b276a7ed5524fc0c47d6080a6349a0a65b9e14bda8270cbaf87d6cd81bceb193`；core`ed24af7acc8caa15578e0b859c85ef8a901be51354448ed39094606ed9e10a2f`不变。prepared7a49及现有controls只读共享，无reference副本或新source proof。source/metricmapping/CPU事实辅助边界保留。

每端最多6原metric trace，各自重算NCCL名称/数量/逻辑payload/period，不以NCCL sum作wire/critical path或跨profile相减。代码的单规约是待测机制，不能由标签宣称；合批数学等价和模拟小模型不代替全量质量。有限输入/ptr guards、cat输入副本、update后回退、rank非对称错误、global override finallyFalse、内存与生命周期等审计局限不消失。

预计增长512MiB，8GiB警戒；当前约9.85GiB/GPU空闲，启动前再核对。不清理/复制大reference，不重启闭合链。新长任务仅一个隐藏完成/磁盘监听，低盘不是完成。启动身份、实际请求/结果及终审另记。原任务完成后根据真实结果选择下一有界步骤，不能将本次固定完成当整个项目结束。
