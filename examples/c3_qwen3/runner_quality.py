import hashlib
import math
from array import array
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import JsonValue

from .quality_arithmetic import compare_logits
from .runner_oracle import continuation
from .runner_records import OracleManifest, QualityReport, RunnerError

if TYPE_CHECKING:
    from .model_runner import ResidentRunner


def evaluate_quality(runner: "ResidentRunner", prompt_ids: tuple[str, ...], oracle_refs: Path) -> QualityReport:
    metrics = {"local_checks_passed": 0.0}
    raw: list[dict[str, JsonValue]] = []
    detail: str | None = None
    stamp = runner.backend.weights_stamp()
    initial_calls = runner.backend.forward_calls
    valid = False
    try:
        runner.binding.verify_installed()
        manifest = OracleManifest.model_validate_json(oracle_refs.read_bytes())
        if (manifest.contract_sha256 != runner.prepared.contract_sha256
                or manifest.corpus_sha256 != runner.prepared.corpus_sha256 or not prompt_ids):
            raise RunnerError("oracle task identity mismatch or empty quality set")
        refs = {r.prompt_id: r for r in manifest.prompts}
        runner.binding.validate_local()
        metrics["local_checks_passed"] = 1.0
        with runner.binding.diagnostic():
            for prompt_id in prompt_ids:
                ref = refs[prompt_id]
                prompt = runner.prepared.prompts_by_id[prompt_id]
                if ref.input_ids != prompt.input_ids or len(ref.tokens) != 32 or len(ref.nll) != 32:
                    raise RunnerError("oracle prefix/continuation mismatch")
                source = (oracle_refs.parent / ref.logits_file).resolve()
                if not source.is_relative_to(oracle_refs.parent.resolve()):
                    raise RunnerError("oracle logits path escapes its manifest directory")
                with source.open("rb") as stream:
                    if hashlib.file_digest(stream, "sha256").hexdigest() != ref.logits_sha256:
                        raise RunnerError("oracle logits hash mismatch")
                candidate_nll: list[float] = []
                reference_nll: list[float] = []
                difference2, norm2 = 0.0, 0.0
                with source.open("rb") as stream:
                    for token, logits in continuation(runner, prompt, ref.tokens):
                        expected = array("f")
                        expected.fromfile(stream, ref.vocab_size)
                        compared = compare_logits(logits, expected, token)
                        candidate_nll.append(compared.candidate_nll)
                        reference_nll.append(compared.reference_nll)
                        difference2 += compared.difference2
                        norm2 += compared.norm2
                    if stream.read(1):
                        raise RunnerError("oracle has extra logit positions")
                if any(abs(a - b) > 1e-6 for a, b in zip(reference_nll, ref.nll, strict=True)):
                    raise RunnerError("oracle NLL disagrees with fixed logits/targets")
                error = math.sqrt(difference2) / max(math.sqrt(norm2), 1e-12)
                delta = math.fsum(a - b for a, b in zip(candidate_nll, reference_nll, strict=True)) / 32
                raw.append({"prompt_id": prompt_id, "logits_relative_l2": error,
                            "reference_nll": reference_nll, "candidate_nll": candidate_nll,
                            "paired_mean_nll_delta_nat_per_token": delta, "valid_tokens": 32})
        runner.binding.require_coverage()
        runner.binding.verify_installed()
        errors = [float(row["logits_relative_l2"]) for row in raw]
        deltas = [float(row["paired_mean_nll_delta_nat_per_token"]) for row in raw]
        metrics.update(logits_relative_l2_max=max(errors),
                       paired_mean_nll_delta_nat_per_token=math.fsum(deltas) / len(deltas))
        valid = max(errors) <= 0.01 and metrics["paired_mean_nll_delta_nat_per_token"] <= 0.02
        if not valid:
            detail = "frozen teacher-forced logits/NLL gate failed"
    except (OSError, EOFError, ValueError, RuntimeError, KeyError, AssertionError) as exc:
        detail = f"{type(exc).__name__}: {exc}"
    finally:
        runner.backend.reset()
        if runner.backend.weights_stamp() != stamp:
            runner.close()
            valid, detail = False, "candidate mutated resident weights; runner closed"
        if not valid:
            runner.binding.restore()
    path = runner._raw("quality", {"prompts": raw, "metrics": metrics, "valid": valid, "detail": detail,
                                   "forward_calls": runner.backend.forward_calls - initial_calls,
                                   "coverage": runner.binding.coverage(), "trace": runner.binding.trace})
    return QualityReport(valid, metrics, path, detail)
