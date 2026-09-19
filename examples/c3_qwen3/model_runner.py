"""Public resident-runner surface; importing and preparing never loads Torch."""

from collections.abc import Callable, Mapping
from pathlib import Path
from types import MappingProxyType
from uuid import uuid4

from pydantic import JsonValue

from kernel_optimizer.evaluation.task_eval import TaskEvaluation

from .model_binding import BundleSpec, ModelBinding
from .runner_backend import ModelBackend
from .runner_measure import WaveSpec, execute_wave, score_waves
from .runner_prepare import prepare
from .runner_records import (
    AssetSpec, BindingReceipt, GoalSpec, MeasurementReport, PreparedTask, Prompt,
    QualityReport, RawRecord, RunnerError, WaveRecord,
)


class ResidentRunner:
    def __init__(self, prepared: PreparedTask, backend: ModelBackend, output_dir: Path) -> None:
        self.prepared = prepared
        self.backend = backend
        self.binding = ModelBinding(backend.model, backend.codec)
        self.output_dir = output_dir.resolve()
        self.receipt = BindingReceipt("baseline", MappingProxyType({}), "baseline", (), True)
        self.warmed: set[tuple[str, str, int, int, int]] = set()
        self.quality_identity: tuple[str, str] | None = None
        self.closed = False

    def bind(self, bundle_spec: BundleSpec) -> BindingReceipt:
        self._open()
        self.quality_identity = None
        self.receipt = self.binding.bind(bundle_spec)
        return self.receipt

    def reset(self, request_specs: tuple[Prompt, ...] = ()) -> None:
        self._open()
        self.backend.reset()

    def _open(self) -> None:
        if self.closed:
            raise RunnerError("resident runner is closed")

    def _raw(self, kind: str, data: Mapping[str, JsonValue]) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"{kind}-{uuid4().hex}.json"
        identity: dict[str, JsonValue] = {
            "bundle_sha256": self.receipt.bundle_sha256, "params_sha256": self.receipt.params_sha256,
            "source_hashes": dict(self.receipt.source_hashes), "site_ids": list(self.receipt.site_ids),
            "contract_sha256": self.prepared.contract_sha256, "corpus_sha256": self.prepared.corpus_sha256,
        }
        path.write_text(RawRecord(kind=kind, binding=identity, data=dict(data)).model_dump_json(indent=2), encoding="utf-8")
        return path

    def measure(self, goal_spec: GoalSpec, prompt_ids: tuple[str, ...]) -> MeasurementReport:
        self._open()
        waves: list[WaveRecord] = []
        stamp = self.backend.weights_stamp()
        initial_calls = self.backend.forward_calls
        try:
            self.binding.verify_installed()
            if goal_spec not in self.prepared.contract.goals:
                raise RunnerError("measurement goal differs from prepared contract")
            if self.receipt.site_ids and self.quality_identity != (self.receipt.bundle_sha256, self.receipt.params_sha256):
                raise RunnerError("candidate requires passing quality and exclusive binding coverage")
            prompts = tuple(self.prepared.prompts_by_id[p] for p in prompt_ids)
            if (len(prompts) != goal_spec.requests or len(set(prompt_ids)) != len(prompt_ids)
                    or any(len(p.input_ids) != goal_spec.input_tokens for p in prompts)):
                raise RunnerError("incorrect prompt count, uniqueness, or length")
            batches = (prompts,) if goal_spec.id == "multi" else tuple((p,) for p in prompts)
            key = (self.receipt.bundle_sha256, self.receipt.params_sha256, goal_spec.input_tokens,
                   goal_spec.output_tokens, len(batches[0]))
            if key not in self.warmed:
                with self.binding.diagnostic():
                    waves.append(execute_wave(self.backend, self.binding, WaveSpec(batches[0], goal_spec.output_tokens, True)))
                if not waves[-1].complete:
                    raise RunnerError(waves[-1].detail or "warmup incomplete")
                self.binding.require_coverage()
                self.warmed.add(key)
            for batch in batches:
                waves.append(execute_wave(self.backend, self.binding, WaveSpec(batch, goal_spec.output_tokens)))
                if not waves[-1].complete:
                    raise RunnerError(waves[-1].detail or "measurement incomplete")
            self.binding.verify_installed()
            evaluation = score_waves(goal_spec, tuple(waves))
        except (RunnerError, RuntimeError, ValueError, KeyError) as exc:
            self.binding.restore()
            self.quality_identity = None
            evaluation = TaskEvaluation(valid=False, detail=f"{type(exc).__name__}: {exc}")
        finally:
            self.backend.reset()
            if self.backend.weights_stamp() != stamp:
                self.close()
                evaluation = TaskEvaluation(valid=False, detail="candidate mutated resident weights; runner closed")
        path = self._raw("measurement", {"goal": goal_spec.id, "waves": [w.model_dump(mode="json") for w in waves],
                                         "diagnostic_coverage": self.binding.coverage(),
                                         "forward_calls": self.backend.forward_calls - initial_calls,
                                         "evaluation": evaluation.model_dump(mode="json")})
        return MeasurementReport(evaluation, path)

    def quality(self, prompt_ids: tuple[str, ...], oracle_refs: Path) -> QualityReport:
        self._open()
        from .runner_quality import evaluate_quality

        report = evaluate_quality(self, prompt_ids, oracle_refs)
        self.quality_identity = (self.receipt.bundle_sha256, self.receipt.params_sha256) if report.valid else None
        return report

    def create_oracles(self, prompt_ids: tuple[str, ...]) -> Path:
        self._open()
        from .runner_oracle import create_oracles

        return create_oracles(self, prompt_ids)

    def capture_fixtures(self, prompt_ids: tuple[str, ...]) -> Path:
        self._open()
        if self.binding.active:
            raise RunnerError("fixtures must come from original baseline")
        with self.binding.diagnostic(capture=True):
            for prompt_id in prompt_ids:
                prompt = self.prepared.prompts_by_id[prompt_id]
                if len(prompt.input_ids) != 64:
                    raise RunnerError("local fixtures use the frozen 64-token calibration inputs")
                wave = execute_wave(self.backend, self.binding, WaveSpec((prompt,), 2))
                if not wave.complete:
                    raise RunnerError(wave.detail or "fixture capture failed")
        return self._raw("fixtures", {"trace": self.binding.trace, "coverage": self.binding.coverage(),
                                      "fixture_count": len(self.binding.fixtures)})

    def profile(self, goal_spec: GoalSpec, prompt_ids: tuple[str, ...]) -> Path:
        self._open()
        if self.binding.active:
            raise RunnerError("baseline profile requires original dispatch")
        self.binding.restore()
        tracer = ModelBinding(self.backend.model, self.backend.codec)
        for path in tracer.modules:
            if path:
                tracer.register_site(path, (path,))
        prompts = tuple(self.prepared.prompts_by_id[p] for p in prompt_ids)
        batch = prompts if goal_spec.id == "multi" else prompts[:1]
        try:
            with tracer.diagnostic(trace=True):
                wave = execute_wave(self.backend, tracer, WaveSpec(batch, goal_spec.output_tokens))
                if not wave.complete:
                    raise RunnerError(wave.detail or "baseline profile failed")
            return self._raw("profile", {"goal": goal_spec.id, "trace": tracer.trace,
                                         "coverage": tracer.coverage(), "sources": tracer.source_catalog(),
                                         "duration_semantics": "inclusive CPU dispatch, not GPU critical path",
                                         "wave": wave.model_dump(mode="json")})
        finally:
            tracer.restore()

    def close(self) -> None:
        if not self.closed:
            self.binding.restore()
            self.binding.fixtures.clear()
            self.binding.modules.clear()
            self.binding.originals.clear()
            self.backend.close()
            self.closed = True


def load(asset_spec: AssetSpec, device: str, *, backend_factory: Callable[[AssetSpec, str], ModelBackend] | None = None) -> ResidentRunner:
    prepared = prepare(asset_spec.contract_path, asset_spec.assets_manifest)
    if backend_factory is None:
        from .torch_backend import TorchBackend

        backend_factory = TorchBackend
    backend = backend_factory(asset_spec, device)
    return ResidentRunner(prepared, backend, asset_spec.contract_path.parent / "runner-output")


__all__ = ["AssetSpec", "BindingReceipt", "GoalSpec", "MeasurementReport", "PreparedTask",
           "QualityReport", "ResidentRunner", "load", "prepare"]


if __name__ == "__main__":
    from .runner_cli import main

    raise SystemExit(main())
