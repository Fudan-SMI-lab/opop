"""Guard tests: domain checks and restricted constraint evaluation."""

import pytest

from kernel_optimizer.models.core import (
    Constraint,
    DeviceLimits,
    ParamDomain,
    ParameterSpace,
    ParamSet,
)
from kernel_optimizer.paramspace.guard import ConstraintError, check_config, eval_constraint

DEVICE = DeviceLimits()

SPACE = ParameterSpace(
    space_id="sp-t", candidate_id="c-t", source_sha="x",
    domains=[
        ParamDomain(name="BLOCK_M", kind="int", choices=[32, 64, 128]),
        ParamDomain(name="NUM_WARPS", kind="int", choices=[2, 4, 8]),
        ParamDomain(name="MODE", kind="str", choices=["a", "b"]),
    ],
    constraints=[
        Constraint(expr="BLOCK_M * NUM_WARPS <= 1024"),
        Constraint(expr="BLOCK_M * 4 <= MAX_SHARED_BYTES"),
    ],
)


def ok(values):
    return check_config(SPACE, ParamSet(values=values), DEVICE)


def test_valid_config():
    assert ok({"BLOCK_M": 64, "NUM_WARPS": 4, "MODE": "a"}) is None


def test_unknown_param():
    r = ok({"BLOCK_M": 64, "NUM_WARPS": 4, "MODE": "a", "EXTRA": 1})
    assert r is not None and r.reason == "unknown_param"


def test_missing_param():
    r = ok({"BLOCK_M": 64, "NUM_WARPS": 4})
    assert r is not None and r.reason == "missing_param"


def test_not_in_choices():
    r = ok({"BLOCK_M": 65, "NUM_WARPS": 4, "MODE": "a"})
    assert r is not None and r.reason == "not_in_choices"


def test_type_mismatch():
    r = ok({"BLOCK_M": "64", "NUM_WARPS": 4, "MODE": "a"})
    assert r is not None and r.reason == "type_mismatch"


def test_bool_rejected_one_way_or_another():
    # Pydantic coerces True -> 1 at ParamSet validation, so the guard sees an
    # int that is not among the choices; either way the config is rejected.
    r = ok({"BLOCK_M": True, "NUM_WARPS": 4, "MODE": "a"})
    assert r is not None and r.reason in ("type_mismatch", "not_in_choices")


def test_constraint_violated():
    r = ok({"BLOCK_M": 128, "NUM_WARPS": 8, "MODE": "a"})
    assert r is None  # 128*8=1024 <= 1024 -> fine
    space2 = SPACE.model_copy(update={
        "constraints": [Constraint(expr="BLOCK_M * NUM_WARPS < 1024")]})
    r2 = check_config(space2, ParamSet(values={"BLOCK_M": 128, "NUM_WARPS": 8, "MODE": "a"}),
                      DEVICE)
    assert r2 is not None and r2.reason == "constraint_violated"


def test_device_constants_available():
    assert eval_constraint("MAX_REGS_PER_THREAD == 255", DEVICE.as_env())


def test_eval_constraint_arith():
    assert eval_constraint("2 ** 3 == 8 and 7 // 2 == 3 and 7 % 2 == 1", {})
    assert eval_constraint("A + B > 5", {"A": 3, "B": 4})
    assert not eval_constraint("A < 2 or B < 2", {"A": 3, "B": 4})


def test_eval_constraint_rejects_calls():
    with pytest.raises(ConstraintError):
        eval_constraint("__import__('os').system('x')", {})


def test_eval_constraint_rejects_attributes():
    with pytest.raises(ConstraintError):
        eval_constraint("A.bit_length", {"A": 3})


def test_eval_constraint_unknown_name():
    with pytest.raises(ConstraintError):
        eval_constraint("UNKNOWN > 1", {})


def test_constraint_invalid_reported():
    space = SPACE.model_copy(update={"constraints": [Constraint(expr="len(MODE) > 0")]})
    r = check_config(space, ParamSet(values={"BLOCK_M": 64, "NUM_WARPS": 4, "MODE": "a"}),
                     DEVICE)
    assert r is not None and r.reason == "constraint_invalid"
