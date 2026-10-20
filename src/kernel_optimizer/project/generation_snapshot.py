"""Immutable inputs at a project-role boundary, without changing its prompt."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from uuid import uuid4

from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.runtime.provenance import framework_source
from kernel_optimizer.store.run_store import RunStore


def public_settings(value):
    """Keep model/tool settings while leaving authentication in deployment config."""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            normalized = key.lower().replace("_", "").replace("-", "")
            if normalized in {"apikey", "password", "secret", "authorization", "accesstoken", "token"}:
                result[key] = "<deployment credential>"
            else:
                result[key] = public_settings(item)
        return result
    if isinstance(value, list):
        return [public_settings(item) for item in value]
    return value


def capture(store, sandbox, *, role, session, prompt, outputs, config, fresh, history=None):
    files = {}
    paths = list((sandbox / "inputs").rglob("*"))
    paths.extend(sandbox / name for name in outputs)
    for path in sorted(set(paths)):
        if path.is_file():
            raw = path.read_bytes()
            files[path.relative_to(sandbox).as_posix()] = {
                "ref": store.put_artifact(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw),
            }
    settings = json.loads((sandbox / "opencode.json").read_text(encoding="utf-8"))
    manifest = {
        "schema_version": "project-generation-snapshot.v1",
        "id": "generation-" + uuid4().hex[:12], "role": role, "session": session,
        "store": str(store.run_dir.resolve()), "sandbox": str(sandbox),
        "fresh_session": fresh, "prompt_ref": store.put_artifact(prompt),
        "outputs": list(outputs), "files": files,
        "model": config.agents.default_model, "agent": config.opencode.agent,
        "tool_model_settings": public_settings(settings),
        "framework_source": framework_source(),
        "history_ref": store.put_artifact(json.dumps(history)) if history is not None else None,
        "history_status": "empty" if fresh else "recorded" if history is not None else "unavailable",
        "scope": "Input files and declared draft outputs before the request. External assets, tool-accessible filesystem and provider internals are not a complete environment snapshot. Initial-role replay uses a fresh session; continuation snapshots require their conversation history.",
    }
    path = store.run_dir / "generation-snapshots" / (manifest["id"] + ".json")
    write_json(path, manifest)
    store.append("PROJECT_GENERATION_SNAPSHOT", {"path": str(path), "role": role, "session": session,
                                               "fresh_session": fresh})
    return path


def load(path: Path):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    store = RunStore.open(Path(manifest["store"]))
    files = {}
    for name, entry in manifest["files"].items():
        raw = store.get_artifact(entry["ref"])
        if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise ValueError(f"generation input changed: {name}")
        files[name] = raw
    return manifest, store.get_artifact(manifest["prompt_ref"]).decode("utf-8"), files


def edit_one(prompt, files, edit):
    """One explicit experimental treatment, never a failure-dependent policy."""
    files = dict(files)
    if edit is None:
        return prompt, files
    name = edit["path"]
    original = prompt if name == "prompt" else files[name].decode("utf-8")
    old, new = edit["old"], edit["new"]
    if not old or original.count(old) != int(edit.get("occurrences", 1)):
        raise ValueError("treatment must match the declared number of occurrences")
    changed = original.replace(old, new)
    if name == "prompt":
        prompt = changed
    else:
        files[name] = changed.encode("utf-8")
    return prompt, files
