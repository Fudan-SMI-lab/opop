import hashlib
from pathlib import Path

import pytest

from examples.c3_qwen3.device_ir import stored_dependencies

NATIVE = Path(__file__).parent / "fixtures/c3_device_ir/rms_norm_control.ttir"
SHA = "ec4ea2e295bfa43bbe6db7f22ad8caa223222b311280d364fd34e8e9c24f0c5c"


def test_native_saved_ir_preserves_input_and_weight_provenance() -> None:
    # Given: the exact native bytes, not renamed source or a reconstructed kernel.
    raw = NATIVE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SHA
    # When / Then: output position2 is derived from actual input0 and weight1.
    assert stored_dependencies(raw.decode()) == {2: frozenset({0, 1})}


@pytest.mark.parametrize("names", [("X", "Y"), ("arg0", "arg1"), ("arg9", "arg4")])
def test_signature_attributes_do_not_overwrite_positional_roots(names) -> None:
    # Given: function attributes contain '=', but are not SSA definitions.
    x, y = names
    ir = f"""tt.func public @copy(%{x}: !tt.ptr<f32> {{tt.divisibility = 16 : i32}}, %{y}: !tt.ptr<f32>) attributes {{noinline = false}} {{
    %v = tt.load %{x} : !tt.ptr<f32>
    tt.store %{y}, %v : !tt.ptr<f32>
    }}"""
    # When / Then: argument order, not spelling/suffix, defines runtime positions.
    assert stored_dependencies(ir) == {1: frozenset({0})}


def test_quoted_reduction_when_operand_list_spans_lines() -> None:
    # Given: a quoted reduction with multiline operands and a multiline combiner region.
    ir = """tt.func public @reduce(%X: !tt.ptr<f32>, %Y: !tt.ptr<f32>) {
    %v = tt.load %X : !tt.ptr<f32>
    %r = "tt.reduce"(
      %v
    ) <{axis = 0 : i32}> ({
      ^bb0(%a: f32, %b: f32):
        %sum = arith.addf %a, %b : f32
        tt.reduce.return %sum : f32
    }) : (tensor<4xf32>) -> f32
    tt.store %Y, %r : !tt.ptr<f32>
    }"""
    # When / Then: declared reduction operands carry the load dependency across the region.
    assert stored_dependencies(ir) == {1: frozenset({0})}


@pytest.mark.parametrize("body", ["%v = arith.constant 0.0 : f32", "%v = tt.splat %X : !tt.ptr<f32>",
    '%v = arith.constant 0.0 : f32 loc("tt.load %X = %arg0")',
    '%v = arith.constant 0.0 : f32\n#loc = loc("tt.store %Y, %v; tt.load %X")'])
def test_no_load_or_confusing_location_cannot_prove_output(body: str) -> None:
    # Given: constants, pointer-only values and location strings are not tensor loads.
    ir = f"tt.func public @dummy(%X: !tt.ptr<f32> {{tt.divisibility = 16 : i32}}, %Y: !tt.ptr<f32>) {{\n{body}\ntt.store %Y, %v : !tt.ptr<f32>\n}}"
    # When / Then: the IR gate stays fail-closed.
    assert stored_dependencies(ir) == {}


def test_named_signature_maps_retained_arguments_to_original_jit_slots() -> None:
    # Given: constexpr/specialized arguments between tensors were omitted from the native signature.
    ir = NATIVE.read_text()
    slots = {"X": 0, "specialized_scalar": 1, "W": 3, "Y": 5}
    # When / Then: exact names map retained arguments without shifting output/input attribution.
    assert stored_dependencies(ir, slots) == {5: frozenset({0, 3})}


def test_canonical_signature_mapping_requires_matching_retained_arity() -> None:
    # Given: canonical names carry no original JIT-name information.
    ir = "tt.func public @copy(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) {\n%v = tt.load %arg0 : !tt.ptr<f32>\ntt.store %arg1, %v : !tt.ptr<f32>\n}"
    # When / Then: known retained order maps correctly; additional unknown elision cannot be guessed.
    assert stored_dependencies(ir, {"X": 0, "Y": 2}) == {2: frozenset({0})}
    assert stored_dependencies(ir, {"X": 0, "N": 1, "Y": 2}) == {}
