# DDP 合批与单全梯度规约候选的一次固定 B/C 验证

声明一次现有fork_run/bundle_from_outputs/stage/execute：B为当前已接受ccd90，C为Agent原样508a。不是空基线B，不重新发现/控制/source比较，不运行Agent/search/参数探索、手选collector或Nsight，不发布current，不自动接受或重试。既有3次postclean通信profile/rank来自原请求，保持。

来源`ddp-accepted-communication-trial-20261030/ddp-search` current`1fc24545217087adefa1248704619fcdc882a19057336b231d8e02918bf4d0d8`，单旧bundle；新候选replaces旧bundle、requires空、emptyparams。candidate SHA`508ad3823d3a7b980df7076023b3567242f12a2544cb9bbeae0ada66c6f52f0a`/14,898B，proposal`853ae1eb048e1d590e5b2ba03d75dfec378a4fc20bf4a7b0faa478ba03e1de89`/9,828B。代码终审SHA`74209119a32ada66f1d298a4f3c67bbe5e1c043a5c796529986ef77ad8d2c123`支持的只是固定原任务执行资格，不是通用安全或数值通过。oldaccepted4541/independent7a49/历史Graph09b7/source1fc不改。

原两rank GPT2-12层/768hidden/12heads/50257vocab/npos1024，B1/S128/accum4/effective8seq1024tokens、FP32/eager/noTF32/dropout0/usecacheFalse、seed20261020/input31000+rank、完整zero/FWBW/lossdiv4/SGD.001无momentum/weightdecay/foreachFalse/twosync/owners保持。新实现可改变内部合批/调度，原输出全梯度、更新参数、loss/finite/精确跨rank replicas仍由未改adapter验证。2连续全量检查1e-6/1e-4、2warmup、5clean/rank；aligned maxrank先于median/MAD。原current anchor133.379621→140.04860205ms，不重锚/混合时期/加内存约束。

B须两rank原valid/mathtrue/2完整checks/5samples才执行C；B计时不作门槛。C可能只在第二质量调用尝试新路径，setup失败后的parent质量不验证fast。每个失败/回退/尾样本保留；单B/C不足paired-noise或独立接受，不据此宣布通信成功。若后续确有新的正式接受，才另行声明必要独立完整版本验证。已有候选不因成功标签自动采用。

同当前两4090 UUID20e8003c-f531-d50c-084c-a9a97fe94004与a24af2e6-99bb-9729-5d14-50fa10a2b321/driver580.105.08，adapter`b276a7ed5524fc0c47d6080a6349a0a65b9e14bda8270cbaf87d6cd81bceb193`；core`ed24af7acc8caa15578e0b859c85ef8a901be51354448ed39094606ed9e10a2f`不变。prepared7a49及现有controls只读共享，无reference副本或新source proof。source/metricmapping/CPU事实辅助边界保留。

每端最多6原metric trace，各自重算NCCL名称/数量/逻辑payload/period，不以NCCL sum作wire/critical path或跨profile相减。代码的单规约是待测机制，不能由标签宣称；合批数学等价和模拟小模型不代替全量质量。有限输入/ptr guards、cat输入副本、update后回退、rank非对称错误、global override finallyFalse、内存与生命周期等审计局限不消失。

预计增长512MiB，8GiB警戒；当前约9.85GiB/GPU空闲，启动前再核对。不清理/复制大reference，不重启闭合链。新长任务仅一个隐藏完成/磁盘监听，低盘不是完成。启动身份、实际请求/结果及终审另记。原任务完成后根据真实结果选择下一有界步骤，不能将本次固定完成当整个项目结束。
