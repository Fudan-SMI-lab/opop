from importlib.util import find_spec

import pytest
from pydantic import ValidationError


def test_declaration_when_backend_and_kernel_are_supplied() -> None:
    # Given
    assert find_spec("kernel_optimizer.models.device_operator") is not None
    from kernel_optimizer.models.device_operator import DeviceKernelDeclaration

    # When
    declaration = DeviceKernelDeclaration(backend="triton", source_file="kernels.py", entry="compute")
    # Then
    assert declaration.model_dump() == {"backend": "triton", "source_file": "kernels.py",
                                         "entry": "compute", "output_arg_indices": ()}


@pytest.mark.parametrize("field,value", [("backend", "torch"), ("source_file", ""), ("entry", ""),
                                        ("output_arg_indices", [-1])])
def test_declaration_rejected_when_static_fields_are_invalid(field: str, value: str | list[int]) -> None:
    # Given
    assert find_spec("kernel_optimizer.models.device_operator") is not None
    from kernel_optimizer.models.device_operator import DeviceKernelDeclaration

    payload = {"backend": "cuda", "source_file": "kernels.py", "entry": "module.compute", field: value}
    # When / Then
    with pytest.raises(ValidationError):
        DeviceKernelDeclaration.model_validate(payload)
