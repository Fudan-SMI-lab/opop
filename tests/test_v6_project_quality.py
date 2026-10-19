import torch

from kernel_optimizer.project.evaluation import compare, corrupt, snapshot


def test_integer_tokens_stay_exact_with_float_tolerances():
    assert not compare(torch.tensor([100001]), torch.tensor([100000]), {"atol": 0.02, "rtol": 0.02})["passed"]


def test_full_nested_outputs_and_nan_are_checked():
    reference = {"logits": torch.arange(12.).reshape(3, 4), "tokens": (1, 2)}
    actual = snapshot(reference)
    assert compare(actual, reference, {"atol": 1e-4, "rtol": 1e-4})["passed"]
    actual["logits"][2, 3] = float("nan")
    assert not compare(actual, reference, {"atol": 1e-4, "rtol": 1e-4})["passed"]
    assert not compare({"logits": reference["logits"]}, reference, {})["passed"]


def test_negative_control_preserves_reference_and_exceeds_relative_tolerance():
    reference = {"state": [torch.tensor([1e8, 0.0])], "output": torch.tensor([5.])}
    wrong = corrupt(reference)
    assert reference["state"][0][0] == 1e8
    assert not compare(wrong, reference, {"atol": 0.02, "rtol": 0.02})["passed"]
