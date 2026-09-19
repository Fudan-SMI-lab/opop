"""Behavioral replacement tests use small numerical named modules."""

import importlib
import json
from pathlib import Path

import pytest

from tests.c3_tiny_backend import NumericCodec, TinyModel


def test_bundle_hash_when_helper_body_changes(tmp_path: Path) -> None:
    # Given: one complete cumulative bundle with a declared relative helper.
    path = Path(__file__).resolve().parents[1] / "examples/c3_qwen3/model_binding.py"
    assert path.is_file(), "T4 operator binding implementation is absent"
    api = importlib.import_module("examples.c3_qwen3.model_binding")
    bundle = {"entry": "operators.py", "files": ["operators.py"], "helpers": ["helper.py"],
              "sites": [{"site_id": "scale", "replacement_callable": "replace"}],
              "params": {}, "space": {"params": [], "constraints": []}, "parent_bundle": None,
              "cumulative_from_original_baseline": True, "source_body_rewrite_required": True}
    (tmp_path / "bundle.json").write_text(json.dumps(bundle), encoding="utf-8")
    (tmp_path / "operators.py").write_text(
        "from .helper import factor\ndef replace(site, call, params):\n    return call.args[0] * factor\n",
        encoding="utf-8")
    helper = tmp_path / "helper.py"
    helper.write_text("factor = 2\n", encoding="utf-8")
    first = api.load_bundle(tmp_path / "bundle.json", {})
    # When: only a helper's numerical computation changes.
    helper.write_text("factor = 3\n", encoding="utf-8")
    second = api.load_bundle(tmp_path / "bundle.json", {})
    # Then: execution identity changes even though entry bytes do not.
    assert first.bundle_sha256 != second.bundle_sha256
    assert first.source_hashes["operators.py"] == second.source_hashes["operators.py"]


def make_bundle(root: Path, body: str, factor: int = 2):
    api = importlib.import_module("examples.c3_qwen3.model_binding")
    root.mkdir(exist_ok=True)
    (root / "helper.py").write_text(f"factor = {factor}\n", encoding="utf-8")
    (root / "operators.py").write_text("from .helper import factor\n" + body, encoding="utf-8")
    document = {"entry": "operators.py", "files": ["operators.py"], "helpers": ["helper.py"],
                "sites": [{"site_id": "scale", "replacement_callable": "replace"}], "params": {},
                "space": {"params": [], "constraints": []}, "parent_bundle": None,
                "cumulative_from_original_baseline": True, "source_body_rewrite_required": True}
    (root / "bundle.json").write_text(json.dumps(document), encoding="utf-8")
    return api.load_bundle(root / "bundle.json", {})


def binding(model):
    api = importlib.import_module("examples.c3_qwen3.model_binding")
    assert hasattr(api, "ModelBinding"), "instance-local binding is absent"
    return api.ModelBinding(model, NumericCodec())


def test_bind_restore_when_real_replacement_is_used(tmp_path: Path) -> None:
    # Given: an original call captured for the explicitly selected named site.
    model = TinyModel()
    binder = binding(model)
    binder.register_site("scale", ("scale",))
    with binder.diagnostic(capture=True):
        assert model.scale.forward(3.0) == 6.0
    bundle = make_bundle(tmp_path, "def replace(site, call, params):\n    return call.args[0] * factor\n")
    # When: bind a numerical rewrite and run its diagnostic/local check.
    binder.bind(bundle)
    with binder.diagnostic():
        value = model.scale.forward(3.0)
    binder.validate_local()
    # Then: the candidate computes the answer, old dispatch is unused, restoration works.
    assert value == 6.0
    assert binder.coverage()["scale"]["replacement_calls"] == 1
    assert binder.coverage()["scale"]["old_calls"] == 0
    binder.restore()
    assert model.scale.forward(4.0) == 8.0


def test_wrong_operator_when_numerical_body_changes(tmp_path: Path) -> None:
    # Given: one real baseline fixture.
    model = TinyModel()
    binder = binding(model)
    binder.register_site("scale", ("scale",))
    with binder.diagnostic(capture=True):
        model.scale.forward(3.0)
    bundle = make_bundle(tmp_path, "def replace(site, call, params):\n    return 0.0\n")
    binder.bind(bundle)
    # When / Then: wrong computation fails and baseline dispatch is restored.
    with pytest.raises(RuntimeError, match="numerical"):
        binder.validate_local()
    assert model.scale.forward(3.0) == 6.0


def test_skipped_binding_when_stale_graph_calls_old_reference(tmp_path: Path) -> None:
    # Given: a stale bound forward reference retained before candidate installation.
    model = TinyModel()
    stale = model.scale.forward
    binder = binding(model)
    binder.register_site("scale", ("scale",))
    binder.bind(make_bundle(tmp_path, "def replace(site, call, params):\n    return call.args[0] * factor\n"))
    # When: execution bypasses the installed instance forward.
    with binder.diagnostic():
        assert stale(3.0) == 6.0
    # Then: zero candidate coverage is not a valid binding proof.
    with pytest.raises(RuntimeError, match="not executed"):
        binder.require_coverage()
    assert model.scale.forward(3.0) == 6.0


def test_helper_variant_when_rebinding_same_path(tmp_path: Path) -> None:
    # Given: a loaded relative helper, then changed bytes at the same source path.
    model = TinyModel()
    binder = binding(model)
    binder.register_site("scale", ("scale",))
    body = "def replace(site, call, params):\n    return call.args[0] * factor\n"
    binder.bind(make_bundle(tmp_path, body, 2))
    assert model.scale.forward(3.0) == 6.0
    # When: helper-only new variant is installed in a fresh namespace.
    binder.bind(make_bundle(tmp_path, body, 3))
    # Then: no stale Python bytecode/function reference is executed.
    assert model.scale.forward(3.0) == 9.0
    binder.restore()
    assert model.scale.forward(3.0) == 6.0


def test_old_fallback_when_candidate_calls_original_body(tmp_path: Path) -> None:
    # Given: a numerically correct wrapper that secretly calls the old computation.
    model = TinyModel()
    binder = binding(model)
    binder.register_site("scale", ("scale",))
    bundle = make_bundle(tmp_path, "def replace(site, call, params):\n    return site.module.original(*call.args)\n")
    binder.bind(bundle)
    # When: the candidate wrapper is entered but the original executes underneath it.
    with binder.diagnostic():
        assert model.scale.forward(3.0) == 6.0
    # Then: numerical equality alone is not a source-replacement proof.
    with pytest.raises(RuntimeError, match="exclusively"):
        binder.require_coverage()


def test_failure_when_replacement_raises_restores_baseline(tmp_path: Path) -> None:
    # Given: a candidate failing on the model's normal forward path.
    model = TinyModel()
    binder = binding(model)
    binder.register_site("scale", ("scale",))
    bundle = make_bundle(tmp_path, "def replace(site, call, params):\n    raise ArithmeticError('broken operator')\n")
    binder.bind(bundle)
    # When / Then: propagate failure, but remove the broken dispatch.
    with pytest.raises(ArithmeticError):
        model.scale.forward(3.0)
    assert model.scale.forward(3.0) == 6.0
