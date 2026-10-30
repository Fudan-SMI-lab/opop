# DDP 原样合批与单规约版本的一次独立闭合

已于2026-10-30 20:11:22UTC（香港10月31日04:11:22）启动controller208696，冻结`f81953b61a10955d7ca3e494793353120547f32e`；唯一隐藏监听45316。实际20:12:25[启动核验](evidence/v6-ddp-observation-route-independent-20261031/startup-verification.json) SHA`1db83668d3339b1b0907864e358ca6d566dec8d9820ee30e636063f453ea76eb`：1235冻结/sourcecurrent1fc/新current7a49/request/prepared/core一致，已出现B/C/C三个真实请求，entry/adapter/materialized source/bundle/controls/UUID核对通过；第二C两rank209536/209537仍运行，剩10,138,636,288B约9.44GiB。无Agent generation。仅启动身份与活动证据，不是独立质量/接受终审；全部四请求/结果/24traces仍须终局核验。旧54144已handled，排队重复不重启。

正式配对终审e431bb80…确认新接受后，声明一次现有`promote_version(publish=False)`，原样完整版本与原始当前设备空prepared基线7a49比较，新B/C/C/B workers。源为ddp-observation-route-validation-20261031/ddp-validation的已接受exploration revision `bbe28c40885890ef6d2c1e69bb75f1a6934178494d9d704ab56b0887e76301e6`，源current仍是1fc；不能因publishFalse就把接受版本误作未接受，或偷偷改current。

单bundle fixed-observation-route，emptyparams，code `508ad3823d3a7b980df7076023b3567242f12a2544cb9bbeae0ada66c6f52f0a` 14898B、proposal `853ae1eb048e1d590e5b2ba03d75dfec378a4fc20bf4a7b0faa478ba03e1de89` 9828B保持。新隔离branch current保留`7a49c6370b52d881a52550766b279df22684adc16cf3c91be56275f8fe2e1df9`，源1fc/原accepted4541/旧Graph09b7均不变。既有fork/import只共享制品和controls，不复制大型模型/reference或重跑source/controls/discovery。

原GPT2任务2rank/B1/S128/accum4/effective8seq、FP32/eager/noTF32/dropout0/usecacheFalse、全FWBW/global mean SGD.001、原状态owner/twosync保持。2连续全量grads/params/lossfinite/exact replica检查1e-6/1e-4、2warmup/5clean及每endpoint/rank3次独立通信profile保持。先aligned rankmax再median/MAD，原anchor133.379621→140.04860205ms不改。无新memory约束，任意失败/尾样本/独立拒绝原样保留，不重试或覆盖正式接受结果。

复用正式本批508a实际native，逐个hash核验：每rank1105 kernels/1103Graph/2versioned GraphLaunch、1 eager AllReduce在Graph外、497759232B不变、真实worker channels SHM/direct/direct。无新Nsys/Agent/search/parameter/collector阶段。source-evidence-reuse仅重验原真实当前设备source/controls及replica negative证据hash；不是新tensor comparison/source模型运行。原source误差微小非零但在原容差，manual source/metricmapping/CPU事实援助边界保持。

独立验证针对完整联合版本；计算合批也改变，不能把全部收益归因通信合并。NCCLsum包含等待，不是wire/criticalpath，不跨profile减clean wall。先前正式71.582732→54.347069ms/24.07796199%与此新original-baseline独立期间分开记录，不相加或重锚定。首次质量parent、第二次setup/Graph的范围、有限身份guard/复制tokens/update后fallback/rank非对称错误等限制保持，不宣称任意owner安全或自动生产采用。

预期新增1GiB profile/小结果预算与8GiB警戒，启动前检查150GiB实际df及GPU空闲；无清理。只在确有长运行时建立一个隐藏plain监听，短批次若在监听预检前结束则直接处理，无重复listener/排队。所有SSH/后台CREATE_NO_WINDOW。旧54144已handled，重复queued完成事件必须去重。原始配置/history/raw traces留服务器，仅代码/hash/小公开摘要入GitHub。Qwen、最新FSDP拒绝、77ac和observer链不重启；此为必要独立闭合，不是自动无限候选链或整个项目结束。
