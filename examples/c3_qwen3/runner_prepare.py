"""CPU-only parsing of the approved immutable T1 artifacts."""

import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Final

from .runner_records import AssetSpec, Contract, FrozenRecord, PreparedTask, Prompt, RunnerError

CONTRACT_SHA: Final = "9d1ef6a79bf99eb6ac0e639d82e894227b403124764ba4018531c9ddf179eaf8"
REVISION: Final = "1cfa9a7208912126459214e8b04321603b3df60c"


class Corpus(FrozenRecord):
    repo_id: str
    revision: str
    enable_thinking: bool
    prompts: tuple[Prompt, ...]


class AssetFile(FrozenRecord):
    name: str
    size: int
    verified_sha256: str
    official_integrity_match: bool


class HostAssets(FrozenRecord):
    repo_id: str
    revision: str
    asset_path: str
    assets_complete: bool
    files: tuple[AssetFile, ...]


class Host(FrozenRecord):
    assets: HostAssets


class Preflight(FrozenRecord):
    contract_sha256: str
    hosts: dict[str, Host]


def prepare(contract_path: Path, assets_manifest: Path) -> PreparedTask:
    contract_path, assets_manifest = contract_path.resolve(), assets_manifest.resolve()
    raw = contract_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != CONTRACT_SHA:
        raise RunnerError("contract bytes differ from approved T1 identity")
    contract = Contract.model_validate_json(raw)
    manifest = Preflight.model_validate_json(assets_manifest.read_bytes())
    if manifest.contract_sha256 != digest or set(manifest.hosts) != {"A", "B"}:
        raise RunnerError("preflight contract/host identity mismatch")
    for host in manifest.hosts.values():
        assets = host.assets
        if (not assets.assets_complete or assets.revision != REVISION
                or assets.repo_id != "Qwen/Qwen3-4B"
                or assets.asset_path != contract.assets.local_path_both_hosts
                or len(assets.files) != 10 or not all(f.official_integrity_match for f in assets.files)):
            raise RunnerError("preflight assets are incomplete or incompatible")
    if manifest.hosts["A"].assets.files != manifest.hosts["B"].assets.files:
        raise RunnerError("host asset identities differ")
    corpus_bytes = (contract_path.parent / contract.assets.corpus_file).read_bytes()
    corpus_sha = hashlib.sha256(corpus_bytes).hexdigest()
    if corpus_sha != contract.assets.corpus_sha256:
        raise RunnerError("corpus hash mismatch")
    corpus = Corpus.model_validate_json(corpus_bytes)
    if corpus.revision != REVISION or corpus.enable_thinking:
        raise RunnerError("corpus revision/thinking mismatch")
    prompts = {p.id: p for p in corpus.prompts}
    if len(prompts) != 58:
        raise RunnerError("expected 58 unique frozen prompts")
    for prompt in prompts.values():
        if (len(prompt.input_ids) != prompt.input_tokens
                or hashlib.sha256(prompt.rendered_text.encode()).hexdigest() != prompt.rendered_sha256):
            raise RunnerError(f"prompt identity mismatch: {prompt.id}")
        # T1 stores the compact JSON token vector, not tokenizer-dependent text approximations.
        token_bytes = json.dumps(prompt.input_ids, separators=(",", ":")).encode()
        if hashlib.sha256(token_bytes).hexdigest() != prompt.token_ids_sha256:
            raise RunnerError(f"token ID hash mismatch: {prompt.id}")
    spec = AssetSpec(contract.assets.local_path_both_hosts, REVISION, contract_path, assets_manifest,
                     MappingProxyType({f.name: f.size for f in manifest.hosts["A"].assets.files}))
    return PreparedTask(contract, spec, MappingProxyType(prompts), digest, corpus_sha)
