# box4-2 存储维护证据

说明见 [分析与恢复方法](../../v6-storage-audit-20261026.md)。
远端完整审计目录：`/root/autodl-tmp/dev/storage-maintenance-20261026/`。

- `summary.json`：清理前分类、完成后的文件系统空间、参考张量与模型保留范围。
- `archived-worker-records.jsonl`：3,081 份 worker 请求／启动记录的原始与压缩哈希、路径和还原信息。
- `archive_worker_records.py`、`archive-progress.log`：实际执行的归档脚本及进度。仓库正式脚本额外明确拒绝没有 Linux `/proc` 的归档环境；当时的实际执行环境为 Linux。
- `dedup.json`：621 份不可变内容的逐项硬链接清单、SHA-256 和原 mtime。
- `cache-cleanup.json`：npm 下载缓存的明确删除边界及前后空间。
- `real-request-size-check.json`：实际 Qwen 请求 53,100,393 → 19,304 字节，两个绑定文件与旧内联对象等值。
- `tests.json`／`tests.log`：17 项针对性测试；`final-tests.json`／`final-tests.log`：最终将读取移出安装计时后，6 项相关复核及最终源码哈希。
- `symlink-audit.json`：完成清理后，扫描的实验目录没有失效符号链接。
- `manifest.json`：第一批公开导出的逐文件字节大小／哈希；`local-verification.json`：本地核验。
- `credential-check.json`：第一批公开归档已检查已知 provider 凭据，未匹配；不包含凭据。

本目录只包含维护元数据和工具，不包含压缩后的私有 worker payload、模型、参考张量、provider 配置或会话数据库。
引用压缩件仍保留在 box4-2 的原目录中。查看旧请求可直接读取 gzip；恢复原始文件使用正式工具的 `--restore`。
