"""Materializer tests: PARAMS span detection, substitution, error kinds."""

import pytest

from kernel_optimizer.models.core import ParamSet
from kernel_optimizer.paramspace.materializer import (
    MaterializeError,
    extract_defaults,
    materialize,
)

GOOD = '''import torch

PARAMS = {
    "BLOCK_M": 64,
    "BLOCK_N": 32,
    "MODE": "vec",
    "SCALE": 0.5,
}

def f():
    return PARAMS["BLOCK_M"] + PARAMS["BLOCK_N"]
'''


def test_extract_defaults():
    assert extract_defaults(GOOD) == {
        "BLOCK_M": 64, "BLOCK_N": 32, "MODE": "vec", "SCALE": 0.5,
    }


def test_materialize_roundtrip():
    params = ParamSet(values={"BLOCK_M": 128, "BLOCK_N": 64, "MODE": "scalar", "SCALE": 1.0})
    out = materialize(GOOD, params)
    assert extract_defaults(out) == params.values
    # everything outside PARAMS unchanged
    assert "def f():" in out
    assert 'return PARAMS["BLOCK_M"] + PARAMS["BLOCK_N"]' in out


def test_materialize_preserves_key_order():
    params = ParamSet(values={"SCALE": 2.0, "MODE": "x", "BLOCK_N": 16, "BLOCK_M": 32})
    out = materialize(GOOD, params)
    assert out.index("BLOCK_M") < out.index("BLOCK_N") < out.index("MODE")


def test_negative_literal_allowed():
    src = "PARAMS = {\n    'X': -3,\n}\n"
    assert extract_defaults(src) == {"X": -3}


def test_no_params():
    with pytest.raises(MaterializeError) as exc:
        extract_defaults("x = 1\n")
    assert exc.value.kind == "NO_PARAMS_DICT"


def test_multiple_params():
    src = GOOD + "\nPARAMS = {'A': 1}\n"
    with pytest.raises(MaterializeError) as exc:
        extract_defaults(src)
    assert exc.value.kind == "MULTIPLE_PARAMS"


def test_non_literal_value():
    src = "N = 4\nPARAMS = {'A': N * 2}\n"
    with pytest.raises(MaterializeError) as exc:
        extract_defaults(src)
    assert exc.value.kind == "NON_LITERAL_VALUE"


def test_bool_value_rejected():
    src = "PARAMS = {'A': True}\n"
    with pytest.raises(MaterializeError) as exc:
        extract_defaults(src)
    assert exc.value.kind == "NON_LITERAL_VALUE"


def test_non_string_key():
    src = "PARAMS = {1: 2}\n"
    with pytest.raises(MaterializeError) as exc:
        extract_defaults(src)
    assert exc.value.kind == "NON_STRING_KEY"


def test_key_mismatch():
    with pytest.raises(MaterializeError) as exc:
        materialize(GOOD, ParamSet(values={"BLOCK_M": 1}))
    assert exc.value.kind == "KEY_MISMATCH"


def test_syntax_error():
    with pytest.raises(MaterializeError) as exc:
        extract_defaults("def broken(:\n")
    assert exc.value.kind == "SYNTAX_ERROR"


def test_params_inside_function_ignored():
    src = "def g():\n    PARAMS = {'A': 1}\n"
    with pytest.raises(MaterializeError) as exc:
        extract_defaults(src)
    assert exc.value.kind == "NO_PARAMS_DICT"
