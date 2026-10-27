# box4-3 清理准备证据

执行方案见[文档](../../v6-storage-cleanup-20261027.md)。本轮未删除、压缩或搬走任何实验原始产物。

- `inventory.json`：远端只读元数据扫描、df/du、inode 去重后的分类、较大文件和符号链接。
- `files.jsonl.gz`：完整逐路径 stat 清单，压缩的只是新生成的元数据记录；不包含原文件内容，不能充当数据备份。
- `candidate-breakdown.json`：从清单派生的候选目录分类。current DDP 标记 active_protected，其他也仅为候选。
- `local-coverage.json`：本地归档存在/大小、旧验证记录、49份导出文件新 SHA 校验、两份现有冷归档的新完整压缩件 SHA，以及旧 tgz 的成员概况。
- `deferred-cleanup.json`：现有完成守候 prompt 的变更哈希和原样追加指令。没有额外队列消息或 Goal。

扫描发生在当前实验运行时，文件可能继续增长。删除前必须在终态重新核对数据完整覆盖和依赖；这些清单本身不授予按目录一键删除的依据。完成后另加实际归档/删除/恢复索引及空间前后对照。
