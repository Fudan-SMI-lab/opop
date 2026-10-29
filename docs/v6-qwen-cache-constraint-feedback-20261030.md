# Qwen缓存接口固定结果后的普通约束反馈续行

空间维护已于2026-10-29 22:48:52–22:49:50 UTC完成：[小摘要](evidence/v6-qwen-cache-constraint-feedback-20261030/storage-completed.json)。133个payload共1,758,344,779B压为118,171,812B，全部gzip/解压SHA验证；2856其余文件、500当前冻结文件、10个历史current源/bundle、旧e5ad版本与6个完整FSDP参考保持。实际df净释放1,640,067,072B（约1.53GiB），余10,229,653,504B（约9.53GiB），GPU始终空闲。初查旧targeted-isolation的16大文件实际为指向其他位置的symlink，已明确排除，未操作。没有raw下载/旧控制器重启，恢复journal SHA `7872ecc61cee947898ff1bd40efd67f61dfc627e4c5a253b339cf3fbee0e9086`。

用户询问是否持续推进时，明确告知上一轮已结束、没有后台实验；本轮恢复实际推进。当前20e8 GPU两卡空闲，150GiB数据盘约8.00GiB，先执行已有授权的旧RUN JSON服务器端无损冷压缩，再启动一轮普通续行。旧固定B/C、候选生成及其已处理监听均不重启。

新根目录 `qwen-cache-constraint-feedback-20261030`，既有CLI `--resume --rounds 1 --feedback`，默认selector/candidate/允许scope/参数探索不改，API observe-only。复用 `qwen-cache-interface-fixed-20261030/qwen-fixed` 的空current `2a42b9b1…`、原prepared009b、当前设备已完成discovery与controls，无新初始发现、参考复制或源验证重跑。没有手选group、候选实现、因果答案或生产prompt/sampler/allocator/接受规则修改。

外部固定7e78不是正式搜索trial，因此本轮反馈明确为开发者组装的诊断续行标记：accepted=false表示原TTFT约束失败，不是假称配对噪声拒绝。精确历史代码 `7e78b82dd21391e3db3d6292e3798feff1908f53259aa732a3c011660c28fb92`、原proposal/selection/bundle通过既有candidate_artifacts提供，未装入新parent。B/C原worker结果、各5样本、完整质量/coverage/原约束逐值携带；1508B索引只列原始请求/结果与通用路径证据限制，不提供外部setup flag审计或手写修复。

CPU实际registry/plan路由预检：536冻结身份；原始诊断feedback2,536,045B/SHA `3659c259e843d58a8d57346eea3509e268a3f893985b1bad2cca55416ff7d4d2`；selector routed3,057,250B、历史selected candidate routed2,521,920B，质量/约束和全部样本保持，历史代码逐字相同。只是接口验证；实际角色初始快照、读取和后续行为须另审计，不能以大小/指针交付宣称消费。原选择中不准确推断不手工改写，由角色核对实际新结果。外部文件系统非密闭。

原完整工作保持B1/prompt256/32 greedy BF16/SDPA/noTF32、全logits .02/.02/exacttokens、两同步/状态所有权、2连续质量/2warmup/5clean。当前设备原TTFT锚23.285755，上限24.45004275；不采用旧25.2536046/25.49943795、重设锚或跨设备合并。原固定B770.550629/C380.435753ms，TTFT25.941852/26.076953均失败；10样本/4完整零误差检查、尾样本及peak增加均保留。Graph标签非native证据；不复制旧版本trace作为新路径证明。

控制器没有独立整版或专门native阶段；真实planner可申请现有collector。若新版本依规则接受，必须终态审计后再声明独立整版验证；否则保留失败，不重复到通过。没有自动生产采用、重开已关闭prompt/interpreter/verifier/reading研究或追加reviewer。人工源映射/以前CPU事实辅助边界及所有历史结果保持。

空间维护仅选旧已完成RUN目录的普通JSON，排除symlink、共享inode、当前500冻结身份、source/store/controls/model/provider。计划133个payload/1,758,344,779B，另外2856文件、10个历史current源/bundle、旧e5ad探索版本、6个完整FSDP参考校验。初次执行预检将str传给signature而报错，发生在创建run目录/压缩/移除任何payload之前；原脚本/日志保留，仅改Path转换。恢复journal与RESTORE位于服务器 `storage-audit-20261030/qwen-cache-constraint/run`，旧journal不覆盖，无raw下载或旧存储控制器重启。缺失未压缩冷路径不等于内容丢失。

新长任务只设一个普通完成/磁盘监听；本地helper、SSH和通知子进程显式隐藏，避免再次弹PowerShell窗口。low_disk_space不是完成，不终止进行中实验或盲删。GitHub仅代码、哈希与小摘要，私有配置/完整会话和raw留服务器。
