"""Final grouping must preserve row dispatch while retaining representative group fixtures."""

import json
from pathlib import Path
from time import time

import pytest

from examples.c3_qwen3 import search_matrix as matrix
from examples.c3_qwen3.manual_data import CALIBRATION_IDS, fingerprint
from examples.c3_qwen3.model_binding import BundleDocument, ModelBinding, Site, load_bundle
from examples.c3_qwen3.model_runner import ResidentRunner, prepare
from examples.c3_qwen3.runner_records import RunnerError
from examples.c3_qwen3.search_bundle import baseline_bundle, export_selection
from examples.c3_qwen3.search_readiness import prepare_readiness
from examples.c3_qwen3.search_records import SearchClock, Selection
from kernel_optimizer.models.core import ParamSet
from tests.c3_tiny_backend import NumericCodec, Scale, TinyBackend, TinyModel
from tests.test_c3_model_runner import RAW


class LayerStack(Scale):
    def __init__(self) -> None:
        self.layers = tuple(Scale() for _ in range(36))
        super().__init__()

    def original(self, value: float) -> float:
        for layer in self.layers:
            value = layer.forward(value) / 2
        return value * self.weight


class LayerModel(TinyModel):
    def __init__(self) -> None:
        self.scale = LayerStack()

    def named_modules(self):
        return ((f"model.layers.{i}", layer) for i, layer in enumerate(self.scale.layers))


class SizedCodec(NumericCodec):
    def __init__(self) -> None:
        self.local_checks = 0

    def size(self, call) -> int:
        return 4 * 1024 * 1024

    def check(self, expected, actual) -> None:
        self.local_checks += 1
        super().check(expected, actual)


def resident_at(output: Path) -> ResidentRunner:
    prepared = prepare(RAW / "contract.json", RAW / "preflight.json")
    backend = TinyBackend()
    backend.model = LayerModel()
    backend.codec = SizedCodec()
    return ResidentRunner(prepared, backend, output)


def chosen(root: Path, sites, ready) -> Selection:
    root.mkdir()
    (root / "operators.py").write_bytes(
        b"def f(site, call, params): return call.args[0] * site.weights['weight']\n"
        b"def g(site, call, params): return call.args[0] + call.args[0]\n")
    baseline = load_bundle(ready.baseline_bundle, {})
    document = baseline.document.model_copy(update={"sites": tuple(Site(site_id=sid, replacement_callable=fn)
        for sid, fn, _ in sites), "parent_bundle": baseline.bundle_sha256})
    path = root / "bundle.json"
    path.write_text(document.model_dump_json())
    bundle = load_bundle(path, {})
    return Selection(bundle=path, bundle_sha256=bundle.bundle_sha256, source_hashes=dict(bundle.source_hashes),
        params=ParamSet(values={}), params_sha256=bundle.params_sha256,
        site_groups={sid: paths for sid, _, paths in sites}, evaluation=ready.checks["multi"][0],
        contract_sha256=ready.contract_sha256, model_revision=ready.model_revision, goal_id=root.name)


@pytest.fixture(scope="module")
def prepared_matrix(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("matrix-groups")
    resident = resident_at(tmp_path / "initial")
    try:
        ready = prepare_readiness(resident, tmp_path / "ready")
    finally:
        resident.close()
    return ready


def test_grouped_36_module_matrix_stays_under_cap_and_checks_every_target(tmp_path, prepared_matrix, monkeypatch):
    # Given: one 36-member winner group, baseline and two independently measured fallbacks.
    ready = prepared_matrix
    paths = tuple(f"model.layers.{i}" for i in range(36))
    winner = chosen(tmp_path / "multi", (("decoder-layer", "f", paths),), ready)
    base = load_bundle(ready.baseline_bundle, {})
    fallback = winner.model_copy(update={"bundle": base.path, "bundle_sha256": base.bundle_sha256,
        "source_hashes": dict(base.source_hashes), "site_groups": {}, "baseline_fallback": True})
    selections = {}
    for name, selected in (("ttft", fallback), ("single", fallback), ("multi", winner)):
        directory = tmp_path / "frozen" / name
        export_selection(selected, directory)
        selections[name] = directory / "selection.json"
    originals = {p: p.read_bytes() for directory in (tmp_path / "frozen").iterdir() for p in directory.iterdir()}
    resident = resident_at(tmp_path / "final")
    capture, require = resident.capture_fixtures, resident.binding.require_coverage
    captures, coverage = [], []
    def capture_once(ids):
        captures.append(ids)
        return capture(ids)
    def check_coverage():
        require()
        if resident.binding.active:
            assert all(resident.binding.counts[p]["replacement_calls"] > 0 and
                       resident.binding.counts[p]["old_calls"] == 0 for p in paths)
            coverage.append(tuple(resident.binding.counts))
    monkeypatch.setattr(resident, "capture_fixtures", capture_once)
    monkeypatch.setattr(resident.binding, "require_coverage", check_coverage)
    started = time() - 1
    clock = SearchClock(campaign_started_unix_s=started, search_cutoff_unix_s=started + 9000,
                        final_deadline_unix_s=started + 10800)
    try:
        # When: the actual matrix, evaluator, fixture guards and quality/measurement loops execute.
        result = matrix.run_matrix(resident, ready, selections, clock=clock, output=tmp_path / "matrix")
        # Then: unchanged caps admit two group/phase fixtures, not 72 module/phase fixtures.
        assert result.preparation_error is None, (result.preparation_error, len(resident.binding.sites),
                                                 len(resident.binding.fixtures), resident.binding.fixture_bytes)
        assert len(result.cells) == result.slots_used == 36 and all(c.evaluation.valid for c in result.cells)
        assert len(resident.binding.sites) == 1 and len(resident.binding.fixtures) == 2
        assert resident.binding.fixture_bytes == 16 * 1024 * 1024
        assert captures == [CALIBRATION_IDS] and len(coverage) == 12
        assert resident.backend.codec.local_checks == 18
        assert all(p.read_bytes() == data for p, data in originals.items())
        executed = load_bundle(tmp_path / "matrix/rows/multi/execution-bundle.json", {})
        assert executed.source_hashes == winner.source_hashes and executed.params_sha256 == winner.params_sha256
        assert all(c.selection_sha256 == winner.bundle_sha256 and c.execution_bundle_sha256 == executed.bundle_sha256
                   for c in result.cells if c.row == "multi")
    finally:
        resident.close()


def test_original_group_capture_passes_while_per_path_capture_exceeds_same_cap(tmp_path):
    # Given: the same 36 numerical modules and codec accounting in both configurations.
    observations = []
    for per_path in (True, False):
        resident = resident_at(tmp_path / str(per_path))
        paths = tuple(resident.binding.modules)
        groups = {p: (p,) for p in paths} if per_path else {"decoder-layer": paths}
        for name, members in groups.items():
            resident.binding.register_site(name, members)
        resident.bind(load_bundle(baseline_bundle(tmp_path / f"base-{per_path}"), {}))
        try:
            # When / Then: toggling only the capture grouping toggles the real guard failure.
            if per_path:
                with pytest.raises(RunnerError, match="128 MiB"):
                    resident.capture_fixtures(CALIBRATION_IDS)
            else:
                resident.capture_fixtures(CALIBRATION_IDS)
            observations.append((len(resident.binding.fixtures), resident.binding.fixture_bytes))
        finally:
            resident.close()
    assert observations == [(15, 120 * 1024 * 1024), (2, 16 * 1024 * 1024)]


def test_partial_overlaps_partition_complete_row_mapping_and_keep_boundaries(tmp_path, prepared_matrix):
    # Given: overlap across rows is legal, but f and g assignments must not be conflated.
    a, b, c, d = (f"model.layers.{i}" for i in range(4))
    left = chosen(tmp_path / "ttft", (("left", "f", (a, b, d)),), prepared_matrix)
    right = chosen(tmp_path / "single", (("overlap", "g", (b, c, d)),), prepared_matrix)
    resident = resident_at(tmp_path / "runtime")
    try:
        # When
        plan = matrix.matrix_groups({"ttft": left, "single": right}, resident.binding)
        # Then: common b/d remain together; each row's original per-module callable survives exactly.
        assert set(plan.groups.values()) == {(a,), (b, d), (c,)}
        assert sum(map(len, plan.groups.values())) == len({p for members in plan.groups.values() for p in members})
        expected = {"ttft": {a: "f", b: "f", d: "f"}, "single": {b: "g", c: "g", d: "g"}}
        for row in expected:
            actual = {p: site.replacement_callable for site in plan.sites[row] for p in plan.groups[site.site_id]}
            assert actual == expected[row]
        assert plan == matrix.matrix_groups({"single": right, "ttft": left}, resident.binding)
    finally:
        resident.close()


@pytest.mark.parametrize("split", ["original_groups", "original_forward"])
def test_distinct_original_boundaries_are_not_fused(tmp_path, prepared_matrix, split):
    # Given: identical per-row callable names but distinct original grouping or forward interfaces.
    a, b = "model.layers.0", "model.layers.1"
    sites = (("one", "f", (a,)), ("two", "f", (b,))) if split == "original_groups" else (("one", "f", (a, b)),)
    selected = chosen(tmp_path / "multi", sites, prepared_matrix)
    model = LayerModel()
    if split == "original_forward":
        model.scale.layers[1].forward = lambda value: value + value
    binding = ModelBinding(model, SizedCodec())
    # When
    plan = matrix.matrix_groups({"multi": selected}, binding)
    # Then
    assert set(plan.groups.values()) == {(a,), (b,)}


@pytest.mark.parametrize("defect", ["overlap", "duplicate", "missing", "extra", "empty"])
def test_malformed_row_mapping_is_not_silently_compressed(tmp_path, prepared_matrix, defect):
    # Given: an ambiguous or incomplete frozen mapping.
    p = "model.layers.0"
    sites = (("first", "f", (p,)), ("second", "g", (p,))) if defect == "overlap" else (("first", "f", (p,)),)
    selected = chosen(tmp_path / "multi", sites, prepared_matrix)
    changes = {"duplicate": {"first": (p, p)}, "missing": {}, "extra": {"first": (p,), "unused": ("model.layers.1",)},
               "empty": {"first": ()}, "overlap": selected.site_groups}
    selected = selected.model_copy(update={"site_groups": changes[defect]})
    resident = resident_at(tmp_path / "runtime")
    try:
        # When / Then
        with pytest.raises(RunnerError):
            matrix.matrix_groups({"multi": selected}, resident.binding)
    finally:
        resident.close()


def test_actual_frozen_exports_plan_one_group_without_source_or_raw_changes():
    # Given: read-only actual T7 exports and the failed original T8 matrix.
    root = RAW / "main-af77534-host-A"
    if not root.is_dir():
        pytest.skip("local campaign export is not present")
    selected = {name: matrix.load_selection(root / "main-af77534" / name / "selected/selection.json")
                for name in ("ttft", "single", "multi")}
    files = [p for s in selected.values() for p in s.bundle.parent.iterdir()]
    raw = root / "heldout-af77534/matrix.json"
    hashes = {p: fingerprint(p) for p in [*files, raw]}
    binding = ModelBinding(LayerModel(), SizedCodec())
    # When
    plan = matrix.matrix_groups({"baseline": selected["ttft"], **selected}, binding)
    # Then: no import/execution of the Qwen operator; only its authenticated mapping is read.
    assert len(plan.groups) == 1 and len(next(iter(plan.groups.values()))) == 36
    assert [s.replacement_callable for s in plan.sites["multi"]] == ["fused_decoder_layer"]
    assert all(not plan.sites[row] for row in ("baseline", "ttft", "single"))
    assert selected["multi"].bundle_sha256 == "8e37a73572d19b4d6379be6fc2af38ef4cb00775142ed0380feb933c97fecbea"
    original = json.loads(raw.read_bytes())
    assert original["slots_used"] == 0 and len(original["cells"]) == 36
    assert all(c["evaluation"]["score"] is None for c in original["cells"])
    assert all(fingerprint(p) == sha for p, sha in hashes.items())
