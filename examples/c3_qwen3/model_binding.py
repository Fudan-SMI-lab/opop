"""Cumulative operator bundle identities and instance-local forward binding."""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal

from pydantic import ConfigDict, JsonValue, TypeAdapter

from .runner_records import BindingReceipt, FrozenRecord, RunnerError


class Site(FrozenRecord):
    model_config = ConfigDict(frozen=True, extra="forbid")
    site_id: str
    replacement_callable: str


class BundleDocument(FrozenRecord):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    entry: str
    sites: tuple[Site, ...]
    files: tuple[str, ...]
    helpers: tuple[str, ...]
    params: dict[str, JsonValue]
    space: dict[str, JsonValue]
    parent_bundle: str | None
    cumulative_from_original_baseline: Literal[True]
    source_body_rewrite_required: Literal[True]


@dataclass(frozen=True, slots=True)
class BundleSpec:
    path: Path
    document: BundleDocument
    sources: Mapping[str, bytes]
    params: Mapping[str, JsonValue]
    source_hashes: Mapping[str, str]
    bundle_sha256: str
    params_sha256: str


def load_bundle(path: Path, params: Mapping[str, JsonValue]) -> BundleSpec:
    path = path.resolve()
    document = BundleDocument.model_validate_json(path.read_bytes())
    names = (*document.files, *document.helpers)
    if document.entry not in names or len(set(names)) != len(names):
        raise RunnerError("entry missing or duplicate bundle source")
    ids = [s.site_id for s in document.sites]
    if len(set(ids)) != len(ids):
        raise RunnerError("duplicate site in cumulative bundle")
    sources: dict[str, bytes] = {}
    for name in names:
        source = (path.parent / name).resolve()
        if Path(name).is_absolute() or not source.is_relative_to(path.parent) or source.suffix != ".py":
            raise RunnerError(f"bundle source escapes root or is not Python: {name}")
        sources[name] = source.read_bytes()
    hashes = {name: hashlib.sha256(value).hexdigest() for name, value in sources.items()}
    encoded = json.dumps({"bundle": document.model_dump(mode="json"), "sources": hashes},
                         sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    effective = TypeAdapter(dict[str, JsonValue]).validate_python(dict(params))
    encoded_params = json.dumps(effective, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return BundleSpec(path, document, MappingProxyType(sources), MappingProxyType(effective),
                      MappingProxyType(hashes), hashlib.sha256(encoded).hexdigest(),
                      hashlib.sha256(encoded_params).hexdigest())


from .binding_runtime import ModelBinding

__all__ = ["BindingReceipt", "BundleSpec", "ModelBinding", "load_bundle"]
