# DDP 原样合批与单规约版本的一次独立闭合

后续证据步骤已终审：原样accepted3cd9的正常发现生成current10992e0d、9calls/8groups，新Nsys/完整载荷校准和timeline已核对，source current7a49与本独立接受不变。详见[终审7cd9fa02…](v6-ddp-fused-communication-evidence-20261031.md)。继续一次普通项目选择，非重复独立验收或新生产采纳；53.91019995%不与其他期间相加。

已于20:13:13UTC完成，reason=batch_finished；处理旧正式配对延迟事件时主动发现并完成本批终审。唯一监听45316的排队完成事件已标记handled，后续投递须去重。审计SHA`bd738743223705f897d1db1154ea4c8f6819b6a0db6844cacb3c1966a623f620`，[公开审计](evidence/v6-ddp-observation-route-independent-20261031/public-terminal-audit.json) SHA`88926877127fb1ae398e0b287873bb0e71f51236b20e6a955518158824c2bcb5`。1235冻结身份、全部4请求/manifest/entry/adapter/controls/core/UUID、完整bundle与contract核对保持；sourcecurrent1fc及独立current7a49均未发布。

完整版本`3cd9fc79f28ac3398d4bdfc5af76b089d7d64ce00a43532f02a8a571671529ca`，trial version-aefa021e172f，quality=pass/coverage_errors空/accepted=true/publication=exploration。对原始当前设备基线，B124.313109/110.580173ms，C53.788142/54.473702ms；pair gain70.524967/56.106471分别大于noise5.158692/4.366202。聚合117.446641→54.130922ms，**53.91019995%**；原140.04860205ms约束通过。这是独立全版本期间收益，与正式对当前accepted父版本24.07796199%、旧ccd90独立36.1578233%分别记录，不能相加或跨期重锚定。

4jobs/8ranks、40个rank clean样本、16次全量检查每次248879620元素、24条各自postclean metric traces完整核验。B最大138.891162ms、C55.528206ms等全部尾样本保留。每个C端第二检查r0 max9.5367431640625e-7、r1 1.1920928955078125e-7在原容差，其他检查零误差，exact replica observe通过；质量仍首次parent、第二次优化setup，不是两次优化路径检查。原始全部输出张量未保存，没有新source tensor comparison。

24条原始trace分别确认B13 AR与C1 AR，总逻辑输入均497759232B；没有重采native，逐个复核正式508a两份SQLite及normalized hash（每rank1105/1103Graph/2versioned launch/1 eager AR）。原source/replica negative仅hash复用，微小非零source误差和manual mapping援助边界不变。支持一个通过完整独立验证、确有规约合并的**计算合批与通信合并联合案例**；不支持减少通信字节、纯通信收益比例、wire/criticalpath或任意输入/owner安全。所有原条件、guard/fallback限制和旧拒绝保留，生产未采用。

下一步[一次新接受版本的正常发现](v6-ddp-fused-communication-evidence-20261031.md)：从独立accepted exploration 3cd9显式fork，在独立branch运行现有discover及原collection-planner自主选择的collectors。当前完整版本继承旧ccd90 registry，之前post-evaluation profile没有附着新registry；先取得原样508a真实调用/通信材料，才决定后续项目方向。此步不生成候选、不手选group/collector、无新acceptance。终审时GPU空闲，实际150GiB剩10,031,603,712B约9.34GiB，1GiB预算加8GiB警戒，无清理。

已于2026-10-30 20:11:22UTC（香港10月31日04:11:22）启动controller208696，冻结`f81953b61a10955d7ca3e494793353120547f32e`；唯一隐藏监听45316。实际20:12:25[启动核验](evidence/v6-ddp-observation-route-independent-20261031/startup-verification.json) SHA`1db83668d3339b1b0907864e358ca6d566dec8d9820ee30e636063f453ea76eb`：1235冻结/sourcecurrent1fc/新current7a49/request/prepared/core一致，已出现B/C/C三个真实请求，entry/adapter/materialized source/bundle/controls/UUID核对通过；第二C两rank209536/209537仍运行，剩10,138,636,288B约9.44GiB。无Agent generation。仅启动身份与活动证据，不是独立质量/接受终审；全部四请求/结果/24traces仍须终局核验。旧54144已handled，排队重复不重启。

正式配对终审e431bb80…确认新接受后，声明一次现有`promote_version(publish=False)`，原样完整版本与原始当前设备空prepared基线7a49比较，新B/C/C/B workers。源为ddp-observation-route-validation-20261031/ddp-validation的已接受exploration revision `bbe28c40885890ef6d2c1e69bb75f1a6934178494d9d704ab56b0887e76301e6`，源current仍是1fc；不能因publishFalse就把接受版本误作未接受，或偷偷改current。

单bundle fixed-observation-route，emptyparams，code `508ad3823d3a7b980df7076023b3567242f12a2544cb9bbeae0ada66c6f52f0a` 14898B、proposal `853ae1eb048e1d590e5b2ba03d75dfec378a4fc20bf4a7b0faa478ba03e1de89` 9828B保持。新隔离branch current保留`7a49c6370b52d881a52550766b279df22684adc16cf3c91be56275f8fe2e1df9`，源1fc/原accepted4541/旧Graph09b7均不变。既有fork/import只共享制品和controls，不复制大型模型/reference或重跑source/controls/discovery。

原GPT2任务2rank/B1/S128/accum4/effective8seq、FP32/eager/noTF32/dropout0/usecacheFalse、全FWBW/global mean SGD.001、原状态owner/twosync保持。2连续全量grads/params/lossfinite/exact replica检查1e-6/1e-4、2warmup/5clean及每endpoint/rank3次独立通信profile保持。先aligned rankmax再median/MAD，原anchor133.379621→140.04860205ms不改。无新memory约束，任意失败/尾样本/独立拒绝原样保留，不重试或覆盖正式接受结果。

复用正式本批508a实际native，逐个hash核验：每rank1105 kernels/1103Graph/2versioned GraphLaunch、1 eager AllReduce在Graph外、497759232B不变、真实worker channels SHM/direct/direct。无新Nsys/Agent/search/parameter/collector阶段。source-evidence-reuse仅重验原真实当前设备source/controls及replica negative证据hash；不是新tensor comparison/source模型运行。原source误差微小非零但在原容差，manual source/metricmapping/CPU事实援助边界保持。

独立验证针对完整联合版本；计算合批也改变，不能把全部收益归因通信合并。NCCLsum包含等待，不是wire/criticalpath，不跨profile减clean wall。先前正式71.582732→54.347069ms/24.07796199%与此新original-baseline独立期间分开记录，不相加或重锚定。首次质量parent、第二次setup/Graph的范围、有限身份guard/复制tokens/update后fallback/rank非对称错误等限制保持，不宣称任意owner安全或自动生产采用。

预期新增1GiB profile/小结果预算与8GiB警戒，启动前检查150GiB实际df及GPU空闲；无清理。只在确有长运行时建立一个隐藏plain监听，短批次若在监听预检前结束则直接处理，无重复listener/排队。所有SSH/后台CREATE_NO_WINDOW。旧54144已handled，重复queued完成事件必须去重。原始配置/history/raw traces留服务器，仅代码/hash/小公开摘要入GitHub。Qwen、最新FSDP拒绝、77ac和observer链不重启；此为必要独立闭合，不是自动无限候选链或整个项目结束。
