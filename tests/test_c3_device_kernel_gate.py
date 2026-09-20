from examples.c3_qwen3.device_ir import stored_dependencies


def test_store_provenance_when_actual_input_load_reaches_output() -> None:
    # Given: compiled-style SSA, not candidate Python text or a claimed kernel name.
    ir = """
    %0 = tt.splat %arg0 : !tt.ptr<f32>
    %1 = tt.load %0 : tensor<4xf32>
    %2 = arith.mulf %1, %1 : tensor<4xf32>
    %3 = tt.splat %arg1 : !tt.ptr<f32>
    tt.store %3, %2 : tensor<4xf32>
    """
    # When / Then: the stored result is derived from a real input pointer load.
    assert stored_dependencies(ir) == {1: frozenset({0})}


def test_dummy_store_when_only_constant_is_written() -> None:
    # Given: a launched dummy writing constants does not prove target computation.
    ir = "%0 = arith.constant 0.0 : f32\n%1 = tt.splat %arg1 : !tt.ptr<f32>\ntt.store %1, %0 : f32"
    # When / Then: no input-derived output is claimed.
    assert stored_dependencies(ir) == {}
