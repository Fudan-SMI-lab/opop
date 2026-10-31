# 重叠候选的原固定B/C验证

香港2026-10-31 **20:56:14—20:56:57**完成一次原固定B/C。终审 `c885f93ec131d2c0a3979f681de486002b16ac0b536c7acb85d54dc06b943bfc`，[公开结果](evidence/v6-ddp-schedule-overlap-fixed-20261031/public-terminal-audit.json) SHA `3e5e0877862fec50c6f9c64179bf34a196aba6747f556e01f943276d4052f15d`。4 ranks/20 clean samples/8 full checks（每次248,879,620元素）/12自身raw metric traces，原质量、完整梯度/更新参数、精确副本、原约束全部通过。四个first checks误差0，四个second checks rank0最大9.5367431640625e-7、rank1 1.1920928955078125e-7，在原容差内；first parent/second graph setup不是两个optimized checks，没有新增source比较或保存全部raw tensors。

|原固定测量|B accepted7e6|C exact2166f860|
|---|---:|---:|
|aligned rank-max样本ms|44.409721 / 44.245637 / 44.458575 / 44.246453 / 44.527111|43.604853 / 43.188764 / 42.942287 / 43.264417 / 43.422932|
|median ms|44.409721|43.264417|
|MAD ms|0.117390|0.158515|

观察差异 **2.5789488747%**，仅固定B/C结果，不是正式配对noise或独立接受。原anchor133.379621/limit140.04860205、两连续全量质量检查、atol1e-6/rtol1e-4、2warmup/5clean/rank、先aligned rank-max后median/MAD均保持；保留全部尾部。质量route为graph_setup_lead2_pg2_tail2_perpiece_foldscale，tail2是两个tail slots，不是实际分片数。

每份自身trace B1141 kernels/1130 Graph/17 launches/11 eager AR；C1139/1127 Graph/22 launches/12 eager AR，全部AR在Graph外。B两组stream21:6/25:5；C stream21:5/25:7。B载荷6144+6×56702976+4×39383808B；C6144+2×28351488+5×56702976+50419200+3×35705344B，**总计仍497,759,232B**。这证明执行方式变化，不是通信量下降。C自身通信区间重叠26.808735—29.367229ms，计算通信重叠15.336048—16.578556ms，只作同capture描述，不作wire/关键路径或可回收收益。此固定批没有Nsys或物理transport日志，不借旧SHM当新证据。

C最大Torch runtime peak4,619,939,328B/lifecycle5,672,500,224B；runtime reserved5,888,802,816B，但lifecycle reserved仍6,354,370,560B，不能只选runtime reservation称全生命周期内存收益，外部NCCL内存未全量计入。原无新增内存约束。deprecation/barrier/AccumulateGrad stream警告保留，无uncaught traceback；installation约10ms不是完整setup成本。

[实际launch](evidence/v6-ddp-schedule-overlap-fixed-20261031/launch.json) controller314510；现有v6_fixed_candidate_validation.py SHA331073e41abe3205d46f44484727daa53445aab656f623429459bc2d9ee46d73，core e893与两arm相同，9当前关键身份保持。C staged ea18e465a9b6570b779732346c63f94bd55be57c8c524a572fab29effbd17471/singlefixed-schedule-overlap/params{}调用代码自身lead2/2/4；code2166f860330679a93f335985183675815b2e7e7053dc5be9453277aa68497a61，proposal b3fae015c2007ce75443c39fcceb8f5c51cb06403265bd99ced30c584b29e4c6。没有Agent、新collector/native/source-control、reference副本、参数搜索或接受/指针发布。

GPU空闲后仅对12份已关闭拒绝C的旧raw traces作授权服务器无损cold：53,789,629→4,303,662B，释放49,485,967B，roundtrip验证且2460其他文件metadata保持。RESTORE位于`/root/autodl-tmp/dev/storage-audit-20261031/ddp-overlap-fixed/RESTORE.md`，journal7904f6187fe2c039b270d262aa18f696ad6b7fca04d9e353dd15b14b3aa7fb59，planac075c0d150f4f965d4040bb31a90747f3383dac4a4ff71afff02b01712c5482；接受/来源/基线/controls/native/模型/依赖不动，cold不是丢失。[160MiB预测](evidence/v6-ddp-schedule-overlap-fixed-20261031/disk-budget.json)与8GiB警戒保持，实际本批仅约33MB增长。主审首遍通过，无实验重跑；短批在监听预检前结束，未创建listener，无queued fixed事件。根据原数值与整步观察，已继续[一次原正式AB/BA](v6-ddp-schedule-overlap-validation-20261031.md)，不混入固定样本。
