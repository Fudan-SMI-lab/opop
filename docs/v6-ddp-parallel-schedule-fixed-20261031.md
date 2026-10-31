# 分片排程候选的原B/C验证

香港 **19:28:31** 正常完成，未创建监听（批次已在监听检查前结束）。[聚焦终审](evidence/v6-ddp-parallel-schedule-fixed-20261031/public-terminal-audit.json) `1159b6954480e6c56d0ea42b5b84680795aa16baf223e682ce4030b608fff67f`，public `65d2f94bb07bbe55be98bf45d382feab6f37b85f7c6303c398377f2cdd3dfb45`：9当前关键身份、4rank/20clean/8fullchecks（各248,879,620元素）/12自有raw traces核对，第一遍审计通过，无workload/role重跑。

| 原固定B/C | 先aligned rankmax后的五个样本 ms | median / MAD ms |
|---|---|---|
| B 当前9bdd | 45.420538 / 45.380782 / 45.879546 / 45.762014 / 45.521059 | 45.521059 / 0.140277 |
| C 精确7e6fb7 | 44.281763 / 44.425940 / 44.647315 / 44.334140 / 44.135465 | 44.334140 / 0.091800 |

观察改善 **2.6074064%**，只支持进入正式配对检验，不是接受结论。所有原质量/精确副本/约束通过；4个第一检查误差0，4个第二检查rank0最大9.5367431640625e-7、rank1最大1.1920928955078125e-7在原容差内。C route first parent / second graph_setup_capture_parallel_pg2_tail4_deferred_small，reference1；不是两个optimized-only检查，也没有全raw tensors持久化或新source比较。警告deprecation/barrier/AccumulateGrad stream完整保留，无未捕获Traceback；8.55–11.52ms安装耗时不是完整setup。Torch峰值B4,613,118,464B/C最大4,612,856,320B、lifecycle5,656,254,464/5,655,992,320B、reserved6,354,370,560B；微小allocator波动不称内存优化，外部NCCL分配未被Torch完整计量。

12份各自workload窗口中，B1138kernels/1130Graph/17launch/8AR；C1141/1130Graph/17launch/**11AR**。B通信stream21/25分3/5次，C分6/5次；全部NCCL在Graph外，RING_LL入口。C实际载荷6144 + 六个56,702,976 + 四个39,383,808 =497,759,232B，与B总量相同。B通信互相重叠18.242075–19.653816ms，C28.223369–29.909001ms；C compute/comm重叠14.327434–16.095945ms，B13.631895–14.065864ms。这是各自capture的区间描述，不能当wire、关键路径或可回收收益；未新采Nsys/物理transport，不挪用旧9bdd SHM作新证明。

明确决定并已于香港 **19:32:38** 启动[一次原AB/BA正式验证](v6-ddp-parallel-schedule-validation-20261031.md)，精确staged `7401459ffc71e929f71ed98e643774d7c23902c3dc2f6abb3ede455990b479ce`，空参数/同code，原噪声规则不变，不混合fixed样本、不扫描参数。以下为启动历史。

香港2026-10-31 **19:27:50** 已实际启动，controller **296856**。采用现有通用 `v6_fixed_candidate_validation.py`（331073e41abe3205d46f44484727daa53445aab656f623429459bc2d9ee46d73），工作runtime/core e893保持；没有新生产框架、prompt、模型、质量或接受改动。

B为新discovery current eb8cd905的已接受9bdd；C精确7e6fb7 /35595B、proposal442ac3f3 /11159B，单bundle fixed-parallel-schedule替换fixed-pipeline-parallel，requires[]，参数{}调用自身fine8/comm_groups2/tail_splits4。现有fork共享request/prepared EMPTY7a49/discovery/只读controls；不复制reference、不新跑source或discovery、不发布current。9当前关键文件核验，未重扫历史。

执行原LocalExecutor/device lease顺序B后C。B必须通过原数值/覆盖门槛才运行C；2连续full gradients/updated params/loss finite/exact replicas，atol1e-6/rtol1e-4，2warmup/5clean、先aligned rankmax后median/MAD，原anchor133.379621/limit140.04860205保持。每rank/arm三次postclean通信profile，最多12自有raw traces。first parent/second setup并非两个optimized-only checks；没有新增source比较。单B/C不是正式噪声判断或独立接受。

启动盘为150GiB、空闲8,823,558,144B，GPU空闲。实际完整同类fixed目录124,432,384B/32,223,232B，因此选择现有参数192MiB增长预算、同一8GiB警戒线；不是API cap，不清理当前或历史数据。此批无Agent/参数搜索/collector/native/手改实现；有失败保留，不自动换参数或修复。结果按实际质量、收益及真实通信机制决定下一步。
