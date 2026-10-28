# Server-only diagnostic restoration

All original content remains in verified gzip payloads on this server. No raw download is required.
Look up the original path in cold-json-journal.jsonl, then run:

```sh
/root/autodl-tmp/dev-venv/bin/python /root/autodl-tmp/dev/opop-v6/scripts/maintenance/v6_server_storage.py restore-json --journal /root/autodl-tmp/dev/storage-audit-20261029/qwen-original-entry/run/cold-json-journal.jsonl --path /absolute/original/path
```
The restore verifies both archive and decompressed hashes and restores all recorded inode aliases.
