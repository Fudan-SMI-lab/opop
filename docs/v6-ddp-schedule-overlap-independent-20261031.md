# 重叠版本的原独立完整版本验证

香港2026-10-31 **21:19:49**原独立B/C/C/B正常完成，完整新结果终审 `d0c4bf0af5b06d94f88eb961d63c8c97b31f9e2e858fa90a3a420b8795368f25`，[公开终审](evidence/v6-ddp-schedule-overlap-independent-20261031/public-terminal-audit.json) SHA `3b3f92a1805813e062052d8c3fa56ef37e8df5b04445e77eabcf78c1a58d845f`。version-c5a4f0c3a7b5接受完整版本 `4e6bc78bd39550a754d66d04adc157b65bbec4b53ee245abda03bedbf3aa3158`，publication=exploration；独立CURRENT仍原EMPTY7a49，来源CURRENT88fa不变，没有发布指针。

|原独立配对|B 原始EMPTY ms|C 完整2166 ms|收益ms|噪声界限ms|
|---|---:|---:|---:|---:|
|B/C|109.687160|43.014729|66.672431|2.837348|
|C/B|115.921650|43.306667|72.614983|2.329440|

聚合 **112.804405→43.160698ms，改善61.7384640254%**，两对均超过各自原噪声界限。4jobs/8ranks/40clean/16fullchecks（每次248,879,620元素）/24自身metric traces全部核对，8当前关键身份/core/实际请求与物化文件/adapter/controls/devices保持。全部40样本保留，最大B135.103348/C43.384675；原anchor133.379621/limit140.04860205约束通过，没有改锚点或加内存门槛。12个checks误差0，4个C第二检查rank0最大9.5367431640625e-7、rank1 1.1920928955078125e-7在原容差内，精确副本检查通过。first parent/second graph_setup_lead2_pg2_tail2_perpiece_foldscale/reference1不是两个optimized checks，没有保存全部raw tensors或新增source比较。

全部自身trace：原始EMPTY B3999 kernels/0 Graph/0 launches/13AR；C1139 kernels/1127 Graph/22 launches/12AR。AR全部eager在Graph外，RING_LL入口。B载荷12×28351488+157541376B；C6144+2×28351488+5×56702976+50419200+3×35705344B，总量同为497,759,232B。C自身计算通信重叠14.956601—16.399222ms，原EMPTY B6.784588—13.281030ms；通信间重叠C26.495166—29.614966ms，B约0仅浮点roundoff。它们是各自capture内区间，不是wire/关键路径、跨capture的clean差值或可回收收益。完整版本收益包含计算批处理、通信和更新排程，不称纯通信贡献/通信量减少，也不与正式相对7e6的2.3751878%或历史59.69/58.46等相加。

原EMPTY Torch runtime peak1,981,029,376B/lifecycle2,479,585,792B/reserved2,589,982,720B；C4,619,939,328B/5,672,500,224B/runtime reserved5,888,802,816B/lifecycle reserved6,354,370,560B，内存增加。外部NCCL未全量测量，不能将相对parent的runtime reservation下降称此完整版本内存收益。全部deprecation/barrier/default loss/profiler/AccumulateGrad警告和尾部保留，无uncaught traceback；没有抑制警告或加入无关故障注入。

此批无Agent/新native/新source-control。只hash复用前一正式精确2166的native605d8220/46e34355/normalized585aa6db和旧真实source证据；不称新Nsys、新source tensor比较或全部所有者证明。主审首遍通过，没有实验/角色/GPU重跑。完整原日志、checks、请求、trace和服务器审计记录保留，私有配置未输出。

下一步依据：新接受2166仍继承7e6旧10call/9group registry，正式profile的12call/11group未附着为当前registry。已于香港 **21:26:34实际启动[一次正常discovery和普通collection-planner](v6-ddp-overlap-communication-evidence-20261031.md)**，显式从accepted4e6bc78b分支，只更新新代码的调用证据，不重新验证旧批、生成候选或选参数。旧33548已handled；以下为本批历史启动记录。

唯一33548已成功queue终态事件：batch_finished，完成UTC 2026-10-31T13:19:49.457966+00:00。终态不等于接受，完整独立审计由该事件续接，不重复监听。

香港2026-10-31 **21:18:24**实际启动一次existing `promote_version(publish=False)`，controller **320672**，唯一隐藏完成/磁盘监听 **33548**。[实际launch](evidence/v6-ddp-schedule-overlap-independent-20261031/launch.json)复用working runtime/core e893，现有通用v6_promote_accepted_version.py SHA b27c96c77ae87644e0bb76acce964e7d95ac1e23e075b5eeb0b39a1054541de3保持。8当前关键身份，无历史sweep/model-health/环境副本/部署门槛/新测试。依据[新正式接受终审](v6-ddp-schedule-overlap-validation-20261031.md)，不是根据固定分数自动提升。

来源ddp-schedule-overlap-validation-20261031/ddp-validation CURRENT88fa仍accepted7e6；显式采用正式accepted exploration `ea18e465a9b6570b779732346c63f94bd55be57c8c524a572fab29effbd17471`，current不等于探索接受版本不是失败。独立branch以原CURRENT-device prepared EMPTY `7a49c6370b52d881a52550766b279df22684adc16cf3c91be56275f8fe2e1df9`为B，即使接受publish=False也保持EMPTY指针。C singlefixed-schedule-overlap/params{}调用代码自身lead2/comm_groups2/tail_splits4；code2166f860330679a93f335985183675815b2e7e7053dc5be9453277aa68497a61/41127B、proposal b3fae015c2007ce75443c39fcceb8f5c51cb06403265bd99ced30c584b29e4c6/11239B。无其他lead/组数/tail参数尝试，无手工改代码。

这是原始EMPTY对完整2166的fresh B/C/C/B，不是7e6 parent。原GPT2/2rank/B1S128/accum4/FP32/noTF32/SGD.001、两连续全量grads/params/lossfinite/精确副本、atol1e-6/rtol1e-4、2warmup/5clean/rank、aligned rank-max-before-median-MAD，以及anchor133.379621/limit140.04860205保持，无新增内存约束。按每个新请求、rank、全检查、40clean、24自身metric trace审终态，保留尾部/失败，不池化fixed/formal数据。first parent/second setup不是两个optimized checks。没有Agent/search/parameters/collector/new native/source-control/reference副本/初始discovery/指针发布。

[source reuse](evidence/v6-ddp-schedule-overlap-independent-20261031/source-evidence-reuse.json)只是旧实际cbf30d10完整source对照/精确replica negative的hash复用，原微小非零误差在容差内，不是fresh model/source比较。只读controls共享，无副本；历史source/metric/CPU/user辅助范围保留，不称unaided A/B。[native reuse](evidence/v6-ddp-schedule-overlap-independent-20261031/native-candidate-reuse.json)仅复用本次正式精确2166的nsys-b6971ce879两SQLite605d8220/46e34355和normalized585aa6db，实际12AR/22launch/双stream/SHM，原受扰区间不作wire或关键路径。不再启动native capture。

香港 **21:20:28** [实际启动核对](evidence/v6-ddp-schedule-overlap-independent-20261031/startup-verification.json) SHAe131b25862554b87d6b3f878e9ab03b59ee61ddac37dca84061b9ff940e250a1：8当前关键身份/core/sourcecurrent88fa/独立EMPTY7a49保持。version-c5a4f0c3a7b5的四个实际请求核对原request、entry aed604ee、adapter b276a7ed、controls/devices；两B零bundle，两C精确2166/params{}。检查时已terminal、GPU为空、df8,669,392,896B；此记录只证明身份与活动，尚非完整独立数值/noise/接受审计。唯一33548按完成事件续接，不重复创建监听。local reader先找不存在schedule模板，后用已有parallel模板，仅本地读取路径纠正，无实验重跑。

容量是执行所需：GPU空闲13:17:41—51UTC，对7个已关闭未采用/拒绝旧角色的120份重复input JSON/TXT物化副本无损cold，**每份先与仍保留的canonical generation snapshot逐字节一致**；306,415,372→19,997,824B，释放286,417,548B，16原snapshot不变，9501其他文件metadata保持，全部roundtrip hash验证。旧accepted4541所在root明确排除；original snapshots/artifact objects/decisions/source/accepted versions/baselines/controls/provider configs/code/models/deps/current runs不动。RESTORE `/root/autodl-tmp/dev/storage-audit-20261031/ddp-overlap-independent/RESTORE.md`，journal302cfa502b7d3ecbb38040745439cc7d612e89402c43f000f08d14a72a39efcd，plan25b8f43080b11f9d0d0f4a514a74a9c7c500c648e303c6a75ffec292fb791243。冷存副本可按journal恢复，不是丢弃历史；旧journal独立保留，没有运行旧controller/角色。

[256MiB预测](evidence/v6-ddp-schedule-overlap-independent-20261031/disk-budget.json)依据最近三次原独立批实际quota增长223,232,000/222,515,200/222,822,400B，268,435,456B高于最大值20%以上，原8GiB警戒保持；不是API/collector cap。启动150GiB盘余8,893,091,840B。低空间不表示完成，不能盲杀/删除；必要无损cold仅GPU结束后保护当前/来源/接受/controls/模型依赖，仍活跃才替换单一监听。无raw本地搬运/模型轮询/投递测试，所有SSH/queue隐藏，不关闭VSCode。

旧34508正式已完整审核并handled，其余旧事件去重；短fixed无listener。原candidate通用尺寸/片/slot helper仍继承12block任务程序，不是通用模型框架。有限guards、复制输入、PG lifetime、shared pool、异常后未回滚weights等条件限制保持，不额外故障注入/评审/修复。原FP32完整数值要求不放宽，也不把代数缩放等价当普遍逐位证明。独立结果返回后先审实际新数值/质量/收益，再选择有依据的一步；不默认追加候选链或停止整个项目。
