# box4-2 磁盘占用、清理与防止重复落盘（2026-10-26）

本轮实测并清理。空间耗尽的主要原因是实验产物长期全量保留、重复序列化和跨运行的重复副本。
此前把部分实验转移到系统盘只是推迟问题，最终使两个磁盘都接近满载。不是新的模型权重持续增长，
也没有发现已删除却仍被进程持有的大文件或 inode 耗尽。

## 清理前后

| 文件系统 | 容量 | 清理前可用 | 清理后可用 |
|---|---:|---:|---:|
| 数据盘 `/root/autodl-tmp` | 122 GiB | 约 0.81 GiB（df 显示 100%） | 约 15.86 GiB |
| 系统盘 `/` | 30 GiB | 约 1.07 GiB（97%） | 约 3.75 GiB |

合计恢复约 17.7 GiB 可用空间。扫描及摘要见 [原始维护记录](evidence/v6-storage-20261026/summary.json)。
文件系统数值以 df/disk_usage 为准；overlay 的 du 会包含只读基础镜像，不能直接相加解释系统盘用量。

当时没有项目实验进程。`free` 显示宿主机约 850 GiB available，但容器 cgroup 的实际内存上限只有
**2 GiB**，不能把宿主机数字当作本容器额度；`/dev/shm` 也只有 2 GiB 且已清空。
`nvidia-smi` 返回 PermissionError，本轮不声称做过 GPU 验证，也未改变运行模式或临时目录。

## 谁在占用空间

清理前数据盘主要目录：

- `dev/v6-work`：约 68.54 GiB，其中 `evidence-followup-20261021` 约 23.97 GiB、
  `agent-repair-20261021` 约 21.33 GiB、`targeted-isolation-20261021` 约 5.31 GiB。
- 旧 Qwen 项目约 15.14 GiB，其中模型 assets 约 13.93 GiB。
- 三个已有 Python 环境合计约 20.52 GiB，属于原有依赖，没有删除。
- 系统盘上 `opop-v6-stage2`、`opop-v6-stage3` 的旧实验合计约 22 GiB，另外有此前迁入的
  `opop-targeted-cold-20261021` 约 2.23 GiB。

对四个实验根目录按实际 inode 去重后，主要产物为：

| 类别 | 清理前占用 | 原因 |
|---|---:|---|
| 完整参考张量 `.pt` | 26.82 GiB | DDP 连续步骤的完整梯度／参数，多个独立准备批次、两个 rank；单文件最大约 2.14 GiB |
| Chrome trace | 11.80 GiB | Qwen 单份约 348 MiB，多次发现／候选后 profiling 累计 |
| 通信 metric trace + timeline trace | 约 16.8 GiB | 多 rank、多重复、多个评测端点，每次保留原始事件 |
| worker 请求 + 启动记录 | 11.90 GiB | 每个请求内嵌大绑定说明，各 rank 启动时又完整抄写参数 |
| rank／合并 discovery | 7.98 GiB | 原始发现和合并结果，以及部分分支副本 |
| trial／反馈结果 | 3.58 GiB | 完整资源观测进入结果和后续反馈 |

表中 trace 子项来自 `other_json` 的细分，与 Chrome trace 分开，不与其重复计算。
其余还有冻结目标文件、Agent 声明输入与内容寻址附件；相同内容在不同运行目录仍有物理副本。
OpenCode 主会话数据库约 0.79 GiB，远小于这些实验产物，保留以支持会话追溯和续跑。

## 已执行的清理

1. 删除可重新下载的 npm `_cacache`，释放约 0.91 GiB。安装的 node/opencode 包未变。
2. 对 **3,081 份已结束作业**的 `request.json` 和 `rank-*-started.json` 做 gzip 归档：
   11.24 GiB → 0.70 GiB。每份在移除原始副本前都解压并校验 SHA-256，写入可恢复清单。
   有活跃 PID、未确认 worker 回收的作业跳过。receipt、结果、候选、参考张量、trace 均未归档或删除。
3. 对内容寻址附件、已冻结的 Agent 输入和已物化的作业源码中 **621 份相同内容**做硬链接去重，
   释放约 6.30 GiB。仅在同一文件系统、相同大小／SHA-256／权限／属主下合并；原路径与内容保留。
   不链接模型、参考张量、可变反馈指针或 provider 数据库；原 mtime 保存在清单。

扫描后的实验根目录未发现失效符号链接。历史框架 freeze、原始结果内容和接受／拒绝判定没有改写。
worker 请求／启动记录现在是同目录 `.json.gz`；若手工回放旧请求，需要先恢复原文件。

远端完整清单在 `/root/autodl-tmp/dev/storage-maintenance-20261026/`，本地和 GitHub 保存公开元数据。
本轮不依赖会随重启消失的 `/dev/shm` 保存证据。

恢复原始 worker 记录（恢复全部需要重新容纳约 11.24 GiB 原始文件；日常查看可直接用 gzip 读取）：

```bash
python scripts/v6_archive_worker_records.py --restore \
  /root/autodl-tmp/dev/storage-maintenance-20261026/archived-worker-records.jsonl
```

恢复会核对压缩件及还原内容的哈希，不覆盖后来被修改的同名文件。压缩件保留。

## 已修正的新作业行为

- `ProjectSearch.execute` 把 bundle 的绑定说明改成对已有冻结 `.targets.json` 的文件引用，
  不再嵌入每个 worker 请求。该文件仍属于原 revision manifest，worker 原有源码哈希校验继续执行。
- evaluator 兼容旧内联 `specs` 与新 `specs_path`；计算、绑定内容、数值门槛和验收逻辑不改。
- rank 启动记录保留设备／PID／revision 证明及原请求的路径、哈希和 JSON pointer，
  不再复制完整参数。worker 的内存上下文不变。
- 新增显式维护工具 `scripts/v6_archive_worker_records.py`，默认只列出可归档量，
  传 `--apply --manifest ...` 才执行上述可恢复归档；不限制 Agent/API 用量。

对真实历史 Qwen 请求做等值核对，两个引用文件与原内联值完全相同，序列化体积由
**53,100,393 字节降到 19,304 字节**。这是实际请求的离线序列化验证，不是新 GPU 性能结果。
真实 CPU 子进程、旧／新绑定输入、质量／覆盖语义、归档还原共 **17 项测试通过**；源码哈希和
命令保存在 [测试记录](evidence/v6-storage-20261026/tests.json)。
最后将引用文件读取放到安装计时开始之前，保持原计时边界；受影响的 6 项质量／覆盖测试再次通过，
最终源码哈希见 [最终复核](evidence/v6-storage-20261026/final-tests.json)。

## 后续保留策略

上述修改解决了已确认的元数据重复写入，**不会让原始 trace 和完整参考张量停止增长**。
应继续保持以下区分，而不是删模型、放宽质量检查或把所有缓存都清空：

- 活跃任务：保留可直接读取的源码、参考张量、接受版本及必要诊断。
- 已完成作业：定期运行维护工具，把重复请求／启动记录归档；旧源码和相同输入仍可去重。
- 已完成实验批次：按清单将冷 trace、旧参考张量移交持久归档，再释放服务器副本；
  必须先检查仍被其他分支共享的 references，不能直接按目录日期删除。
- 原生 trace、解析结果和 Agent 附件继续区分；后续改进应减少同一份大观测在多个反馈中的复制，
  通过索引按需读取，而不是减少必要观测或增大 Agent 上下文。
- 新一批大 profiling 前核对磁盘和当前容器模式。不要把 `/dev/shm` 当永久仓库，
  也不要靠修改 `TMPDIR` 迁移扩展构建而再次触发 noexec 加载失败。

本轮尚未引入全局自动淘汰或统一冷存储系统；清理后的空间可用于继续开发，但不能支持无限期全量留存。
