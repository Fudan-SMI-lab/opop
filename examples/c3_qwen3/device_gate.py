from .device_records import KernelEvidence
from .runner_records import RunnerError


def require_device_proof(evidence: KernelEvidence) -> None:
    if (not evidence.compiled or evidence.launches < 1 or not evidence.kernel_name
            or evidence.kernel_name not in evidence.cuda_names):
        raise RunnerError("declared device kernel was not compiled and CUDA-trace-confirmed launched")
    if evidence.local_quality_passed is False:
        raise RunnerError(evidence.quality_detail or "device probe local correctness failed")
    if (not evidence.output_used or not evidence.stored_output_args or not evidence.loaded_input_args
            or not evidence.skip_control_rejected):
        raise RunnerError("device output participation absent: dummy/unrelated/fallback computation")
