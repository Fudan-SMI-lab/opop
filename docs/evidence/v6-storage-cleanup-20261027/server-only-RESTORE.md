Server-only maintenance. No raw evidence was downloaded.

Environment: kernel-opt-venv paths remain, with only byte/mode/owner/mtime-identical files hardlinked to dev-venv. Different files remain independent. Before an intentional in-place edit to one shared path, copy that file apart; replacing a path normally leaves the other link intact. environment-plan.json.gz and environment-journal.jsonl record the originals and content hashes.

Old freezes: freeze-plan.json and per-root RESTORE_FROM_GIT.json identify original commits. Small git-metadata archives stay here. To rebuild a cold checkout, move its marker to a separate metadata directory, recreate the checkout from retained /root/autodl-tmp/dev/opop-v6, and checkout the recorded commit. All commits and ref targets were verified in retained Git before retirement.

Cold JSON: all original payloads, including failed samples, remain as verified gzip on this data disk. cold-json-journal.jsonl records SHA256, original path aliases, inode group, permissions, owner and timestamps. To restore one complete original hardlink group:

/root/autodl-tmp/dev-venv/bin/python /root/autodl-tmp/dev/opop-v6/scripts/maintenance/v6_server_storage.py restore-json --journal /root/autodl-tmp/dev/storage-audit-20261027/server-only-run/cold-json-journal.jsonl --path /absolute/original/path.json

Have enough free disk for the uncompressed group. Existing changed files are not overwritten. The gzip stays available after restoration. Current accepted versions, latest DDP/TP, provider state, models and references are protected. Older cold JSON paths require this explicit restoration before replay; result acceptance/rejection is unchanged.
