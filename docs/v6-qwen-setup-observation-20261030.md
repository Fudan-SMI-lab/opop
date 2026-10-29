# Qwen setup失败的一次原样固定观察

在[修复作者审计](v6-qwen-setup-diagnostic-repair-20261030.md)和独立CPU边界通过后，声明一次固定baseline B、原04df候选C观察。exact Agent probe `bc4751ca9b77f9edaf5269c10dd053298e287d50452f51d113f1555a33265ff5`、plan `33fdb9239bc1420293fd89fe74a139a081276e2c955aca411db4b9c1bdef63a6`，无手改。使用既有LocalExecutor/设备lease，每worker原entry一次；本批没有Agent、搜索、参数选择、性能接受、Nsight或生产采用。

B请求来自qwen-device-refresh-fixed-20261030的既有原baseline endpoint；C请求来自qwen-device-feedback-20261030 trial0db61e9938原screen。均保留原materialized source、adapter498c90f3、合同/request、shared reference、设备UUID20e8及原bundles，C仍原样04df、空参数。不复制source/controls/reference，不发布指针。仅entry/function/output/wait及diagnostic_context可改变；function/wait原来已是run/true，实际不变。independent_reference=null，不新增源比较；人工源字段映射边界保留。

顺序B-observed、C-observed各一次。B须原worker passed/valid、两次完整检查/math不变、原entry返回且install_windows为空；不以B的TTFT作接受门或重设anchor。若B不满足门，C记not_run。C即使worker passed仍须分别审核原valid、内部catch/fallback、setup比较、质量与覆盖，不按controller完成认定诊断正确。保留所有错误/意外通过，不重测到通过。

完整B1/prompt256/32greedy tokens、BF16/SDPA/noTF32、full logits .02/.02/exact tokens、两同步边界、原state生命周期、2continuous/2warmup/5clean若达到均不变。当前原anchor23.285755→TTFT limit24.45004275；旧25.2536046/25.49943795不适用。观察计时仅诊断，不做接受/差值扣除或跨设备收益。原04df父回退质量通过但TTFT24.679412拒绝保持，内部.012/.012失败不预判为原质量超差。

预期获得安装期比较摘要/有限超差项、原断言结果和部分reference输入/cache来源。新策略最多128详细记录/32失败、reference前2048，后续失败优先淘汰成功项且明确省略；超过范围不补造。失败后CPU取值改变同步/生命周期，存续reference闭包仍有额外调用，不承诺无扰动/任意并发安全。摘要不是完整raw operands，不能重算所有聚合；setup比较与全任务source质量分开，object-id/首层metadata不证明所有KV别名或唯一根因。计划中的未测overhead/内存/因果保证不采纳。

启动前核验真实150GiB df、GPU空闲、实际UUID与370冻结身份；当前约9.56GiB，预计小摘要/原评价结果，无大型trace/参考副本。本轮无清理/raw下载。server保存raw/private历史，GitHub仅代码、SHA、小摘要；如任务仍运行只设一个普通完成/8GiB监听，旧43664处理后不重启。历史接受/拒绝、关闭的研究和生产core全部保持。
