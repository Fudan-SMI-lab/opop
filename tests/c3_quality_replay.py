from __future__ import annotations

import hashlib
import importlib
import json
import math
import sys
from array import array
from collections.abc import Callable, Iterator, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from types import FrameType
from typing import Final

from examples.c3_qwen3.model_binding import ModelBinding
from examples.c3_qwen3.runner_oracle import nll
from examples.c3_qwen3.runner_records import OracleManifest, RunnerError
from tests.c3_tiny_backend import NumericCodec

ORACLE: Final = (Path(__file__).resolve().parents[1] / "results/c3-qwen3-4b-operator-goals/"
                "readiness-foundation/host-A/native-raw/oracle-a1cbcc116f1f49eca9284ef5e0702603/oracle.json")


@dataclass(frozen=True, slots=True)
class Row:
    prompt_id: str
    target: int
    logits: array[float]


@dataclass(frozen=True, slots=True)
class Arithmetic:
    candidate_nll: float
    reference_nll: float
    difference2: float
    norm2: float


def legacy_nll(logits: tuple[float, ...], target: int) -> float:
    if not logits or not all(map(math.isfinite, logits)) or not 0 <= target < len(logits):
        raise RunnerError("invalid logits/target in teacher-forced quality")
    maximum = max(logits)
    return maximum + math.log(math.fsum(math.exp(v - maximum) for v in logits)) - logits[target]


def legacy_compare(candidate: tuple[float, ...], reference: Sequence[float], target: int) -> Arithmetic:
    ref = tuple(reference)
    if len(candidate) != len(ref):
        raise RunnerError("logit shape mismatch")
    return Arithmetic(legacy_nll(candidate, target), legacy_nll(ref, target),
                      math.fsum((a - b) ** 2 for a, b in zip(candidate, ref, strict=True)),
                      math.fsum(v * v for v in ref))


def current_compare(candidate: tuple[float, ...], reference: Sequence[float], target: int) -> Arithmetic:
    if importlib.util.find_spec("examples.c3_qwen3.quality_arithmetic") is not None:
        module = importlib.import_module("examples.c3_qwen3.quality_arithmetic")
        result = module.compare_logits(candidate, reference, target)
        return Arithmetic(result.candidate_nll, result.reference_nll, result.difference2, result.norm2)
    ref = tuple(reference)
    return Arithmetic(nll(candidate, target), nll(ref, target),
                      math.fsum((a - b) ** 2 for a, b in zip(candidate, ref, strict=True)),
                      math.fsum(v * v for v in ref))


def load_rows(path: Path = ORACLE) -> tuple[Row, ...]:
    manifest = OracleManifest.model_validate_json(path.read_bytes())
    rows: list[Row] = []
    for prompt in manifest.prompts:
        source = path.parent / prompt.logits_file
        with source.open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == prompt.logits_sha256
        assert prompt.vocab_size == 151936 and len(prompt.tokens) == 32
        with source.open("rb") as stream:
            for target in prompt.tokens:
                values = array("f")
                values.fromfile(stream, prompt.vocab_size)
                rows.append(Row(prompt.prompt_id, target, values))
            assert not stream.read(1)
    assert len(rows) == 64
    return tuple(rows)


class EmptyModel:
    def named_modules(self) -> Iterator:
        return iter(())


def replay(compare: Callable[[tuple[float, ...], Sequence[float], int], Arithmetic],
           rows: tuple[Row, ...], diagnostic: bool) -> tuple[float, tuple[Arithmetic, ...]]:
    binding = ModelBinding(EmptyModel(), NumericCodec())
    records: list[Arithmetic] = []
    started = perf_counter()
    with binding.diagnostic() if diagnostic else nullcontext():
        for row in rows:
            records.append(compare(tuple(row.logits), row.logits, row.target))
    return perf_counter() - started, tuple(records)


def python_calls(operation: Callable[[], float]) -> tuple[int, float]:
    calls = 0
    def count[T](frame: FrameType, event: str, arg: T) -> None:
        nonlocal calls
        if event == "call":
            calls += 1
    previous = sys.getprofile()
    sys.setprofile(count)
    try:
        value = operation()
    finally:
        sys.setprofile(previous)
    return calls, value


def main() -> None:
    rows = load_rows()
    results = {}
    measured: dict[str, tuple[Arithmetic, ...]] = {}
    for name, operation in (("legacy", legacy_compare), ("current", current_compare)):
        for diagnostic in (False, True):
            elapsed, values = replay(operation, rows, diagnostic)
            measured[f"{name}_{diagnostic}"] = values
            digest = hashlib.sha256(json.dumps([
                (r.candidate_nll, r.reference_nll, r.difference2, r.norm2) for r in values]).encode()).hexdigest()
            results[f"{name}_{'diagnostic' if diagnostic else 'plain'}"] = {
                "wall_s": elapsed, "rows": len(values), "nll_sum": math.fsum(r.candidate_nll for r in values),
                "difference2_sum": math.fsum(r.difference2 for r in values), "checksum": digest}
    row = rows[0]
    for name, operation in (("legacy", legacy_nll), ("current", nll)):
        count, value = python_calls(lambda: operation(tuple(row.logits), row.target))
        results[f"{name}_nll_python_calls"] = {"calls": count, "value": value}
    results["maximum_absolute_error"] = {field: max(abs(getattr(a, field) - getattr(b, field))
        for a, b in zip(measured["legacy_False"], measured["current_False"], strict=True))
        for field in ("candidate_nll", "reference_nll", "difference2", "norm2")}
    results["maximum_relative_norm2_error"] = max(abs(a.norm2 - b.norm2) / max(abs(a.norm2), 1e-12)
        for a, b in zip(measured["legacy_False"], measured["current_False"], strict=True))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
