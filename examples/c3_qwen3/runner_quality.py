import hashlib
import json
import math
from array import array
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import JsonValue

from .quality_arithmetic import compare_logits
from .runner_oracle import continuation
from .runner_records import OracleManifest, QualityPolicy, QualityReport, RunnerError

if TYPE_CHECKING:
    from .model_runner import ResidentRunner

# The criteria this gate enforces when the contract does not declare them, as
# (metric key, limit, what the metric measures). Declared as data rather than inline comparisons so a
# failure can report WHICH criterion failed, BY HOW MUCH, and against WHAT -- see `_verdict`. A bare
# "gate failed" is not actionable: a candidate 10% over the limit and one 10x over read identically,
# and a rewriter told only that it failed cannot tell whether to adjust its numerics or abandon the
# approach. Measured consequence: four independent candidates in the live F4 lane were rejected at
# 0.011155, 0.011807, 0.014172 and 0.011155, the first and last byte-identical because the agent,
# told only that it failed, re-derived the same kernel.
CRITERIA: tuple[tuple[str, float, str], ...] = (
    ("logits_relative_l2_max", 0.01, "worst-prompt relative L2 of the continuation logits"),
    ("paired_mean_nll_delta_nat_per_token", 0.02, "mean paired NLL delta in nats/token"),
)
# The criterion whose limit a measured floor can raise. See `equivalence_floor`.
FLOOR_METRIC = "logits_relative_l2_max"


def equivalence_floor(evidence_path: Path | None, contract_sha256: str, site_id: str,
                      module_count: int) -> float | None:
    """The measured logits error of an equivalence-preserving replacement of THIS site group.

    Returns None when no evidence is supplied, which leaves the declared absolute limit in force.

    The evidence is a JSON document produced by measuring a replacement that reproduces the original
    operator's arithmetic, bound to the same site group, through this same function against the same
    oracle. It is accepted only when it names the same contract, the same site and the same module
    count, because the floor is a property of that group: the same kernel measured 0.0 at one module
    and 0.010349 at 73. Evidence for a different group says nothing about this one.

    This mirrors `TargetSpec.reference_noise_us`, which measures timing noise from an A/A pair rather
    than assuming a number. A floor cannot be written by hand: it has to come from a measurement whose
    identity matches what is being judged.
    """
    if evidence_path is None:
        return None
    if not evidence_path.is_file():
        raise RunnerError(f"declared equivalence-floor evidence is missing: {evidence_path}")
    document = json.loads(evidence_path.read_text(encoding="utf-8"))
    if (document.get("contract_sha256") != contract_sha256 or document.get("site_id") != site_id
            or document.get("module_count") != module_count):
        raise RunnerError("equivalence-floor evidence belongs to a different contract/site/module count")
    floor = document.get("logits_relative_l2_max")
    if not isinstance(floor, (int, float)) or not math.isfinite(floor) or floor < 0:
        raise RunnerError("equivalence-floor evidence has no finite measured logits error")
    return float(floor)


def criteria(policy: QualityPolicy | None, floor: float | None = None) -> tuple[tuple[str, float, str], ...]:
    """The criteria to enforce: the contract's declared limits, with the floor applied when measured.

    Two separate corrections, both of which were defects:

    1. The limits were hardcoded here while the contract declared them in `shared_policy.quality`.
       Nothing compared the two, so the enforced limit and the published limit could disagree. When a
       policy is supplied its numbers are authoritative -- the contract bytes are hash-verified in
       `prepare`, so this enforces exactly what the approved artifact says.

    2. An ABSOLUTE logits limit is not a property of a candidate alone. It depends on how many modules
       the selected site group replaces, because each replaced module contributes its own
       reordering difference and they compound through the forward. Measured on the Qwen3 TTFT task,
       one equivalence-preserving Triton RMSNorm bound to a nested prefix of the 73-module group gave
       0.0 at 1 and 4 modules, 0.004489 at 16 and 36, and 0.010349 at all 73 -- the same kernel
       throughout. So a `floor` measured on the ACTUAL group raises the limit to
       `floor * policy.logits_relative_l2_floor_multiple`, never lowers it below the declared value.

    With `floor=None` -- no evidence measured -- the declared limits apply exactly as before, so a task
    that never measures a floor is unaffected.
    """
    declared = CRITERIA if policy is None else (
        (FLOOR_METRIC, policy.logits_relative_l2_max_per_prompt, CRITERIA[0][2]),
        ("paired_mean_nll_delta_nat_per_token", policy.paired_mean_nll_delta_max_nat_per_token, CRITERIA[1][2]),
    )
    if floor is None:
        return declared
    multiple = 1.0 if policy is None else policy.logits_relative_l2_floor_multiple
    return tuple((key, max(limit, floor * multiple) if key == FLOOR_METRIC else limit, what)
                 for key, limit, what in declared)


def _verdict(metrics: dict[str, float],
             active: tuple[tuple[str, float, str], ...] = CRITERIA) -> tuple[bool, str | None]:
    """Whether every criterion holds, and if not, the measured values against their limits."""
    failed = [(key, metrics[key], limit, what) for key, limit, what in active
              if key in metrics and metrics[key] > limit]
    if not failed:
        return True, None
    breaches = "; ".join(f"{what} {value:.6g} exceeds {key} limit {limit:.6g} "
                         f"(over by {(value / limit - 1) * 100:.1f}%)"
                         for key, value, limit, what in failed)
    passing = "; ".join(f"{key} {metrics[key]:.6g} within {limit:.6g}" for key, limit, _ in active
                        if key in metrics and metrics[key] <= limit)
    return False, f"frozen teacher-forced logits/NLL gate failed: {breaches}" + (
        f" | passing: {passing}" if passing else "")


def evaluate_quality(runner: "ResidentRunner", prompt_ids: tuple[str, ...], oracle_refs: Path,
                     floor_evidence: Path | None = None) -> QualityReport:
    metrics = {"local_checks_passed": 0.0}
    raw: list[dict[str, JsonValue]] = []
    detail: str | None = None
    stamp = runner.backend.weights_stamp()
    initial_calls = runner.backend.forward_calls
    valid = False
    # The criteria actually applied, recorded in the receipt so a reader never has to guess which
    # limits a verdict used. Resolved before the measurement so a malformed policy or mismatched floor
    # fails the evaluation rather than silently falling back to a different limit than intended.
    policy = runner.prepared.contract.quality_policy()
    floor = None
    active = criteria(policy)
    try:
        runner.binding.verify_installed()
        # A row that replaces NOTHING -- the baseline, measured with the original dispatch -- has no
        # site group, so no amplification and no floor to consult. Its own logits error is 0.0 by
        # construction. Consulting the floor here would compare the evidence's site/module count
        # against an empty binding and reject it, which would fail every baseline cell in the
        # published matrix the moment floor evidence exists.
        if runner.receipt.site_ids:
            # The floor is calibrated for ONE site group. With more than one active site the module
            # count would be a sum over groups and the recorded site_id only the first of them, so the
            # evidence would describe something other than what is being judged. Refusing is the only
            # honest option: silently using the first group's identity would apply a floor measured
            # for a different amount of amplification than the candidate actually has.
            if len(runner.receipt.site_ids) > 1 and floor_evidence is not None:
                raise RunnerError("equivalence-floor evidence covers one site group; this bundle "
                                  f"binds {len(runner.receipt.site_ids)}")
            sites = runner.binding.sites.get(runner.receipt.site_ids[0], ())
            floor = equivalence_floor(floor_evidence, runner.prepared.contract_sha256,
                                      runner.receipt.site_ids[0], len(sites))
            active = criteria(policy, floor)
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
        valid, detail = _verdict(metrics, active)
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
                                   "criteria": [{"metric": k, "limit": v, "measures": w} for k, v, w in active],
                                   "criteria_source": "contract" if policy is not None else "module_default",
                                   "equivalence_floor": floor,
                                   "floor_evidence": str(floor_evidence) if floor_evidence else None,
                                   "coverage": runner.binding.coverage(), "trace": runner.binding.trace})
    return QualityReport(valid, metrics, path, detail)
