"""The six concrete agent modules: generator, parameterizer, analyst, rewriter,
novelty, repair. Each is thin: sandbox seeding + prompt + output schema."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

from kernel_optimizer.agents.base import AgentModule
from kernel_optimizer.agents.sandbox import Sandbox
from kernel_optimizer.models.core import DeviceLimits, TaskSpec
from kernel_optimizer.models.reports import (
    BottleneckReport,
    GeneratedCandidate,
    GenerationResult,
    NoveltyCandidate,
    NoveltyResult,
    ParameterizationResult,
    RepairResult,
    RewriteCandidate,
    RewriteResult,
    TuningStats,
)
from kernel_optimizer.paramspace.triton_lint import (
    declares_no_custom_kernel,
    delegates_to_baseline_compiler,
    lint_triton_source,
)


def _contract_doc() -> str:
    return (
        resources.files("kernel_optimizer.agents.prompts")
        .joinpath("candidate_contract.md")
        .read_text(encoding="utf-8")
    )


def _triton_pitfalls_doc() -> str:
    return (
        resources.files("kernel_optimizer.agents.prompts")
        .joinpath("triton_pitfalls.md")
        .read_text(encoding="utf-8")
    )


def _measured_ceilings_doc(calibration) -> str:
    """This box's measured ceilings, as a block for any agent's prompt. Never a datasheet.

    Factored out of `_bottleneck_doc` because it was reachable ONLY from there, and
    `AnalystInputs` is the only Inputs dataclass carrying a calibration -- so the four agents
    that actually WRITE kernels (generator, parameterizer, rewriter, novelty) were told the
    card's name, VRAM, register and shared-memory limits, and nothing about what it can do.
    Verified on run-l3-21-20260908-232211: the generator's `docs/device.md` was 8 lines with no
    DRAM figure and no TFLOP figure, in a run whose own calibration had just measured
    fp16 at 158.6 and bf16 at 164.2 TFLOP/s.

    That gap defeats the guidance those agents are given. The contract tells them to treat
    dot-product precision as a first-class choice; the rewriter is now told a CUDA rewrite wins
    under strict IEEE fp32. Both
    are arguments about ratios between ceilings, and neither agent could see a single one of
    those ceilings for the box it was writing for.

    Empty string when there is no calibration, so a box that could not measure still runs -- the
    prompt then simply omits the section rather than asserting a default.
    """
    if calibration is None:
        return ""
    out: list[str] = [
        "## What this GPU can actually do (measured on THIS box, not a datasheet)\n",
        f"- DRAM: **{calibration.dram_tbs:.3f} TB/s**\n",
        f"- fp32 (no tensor cores): **{calibration.fp32_tflops:.1f} TFLOP/s**\n",
    ]
    if calibration.tf32_tflops > 0:
        out.append(
            f"- tensor cores (tf32): **{calibration.tf32_tflops:.1f} TFLOP/s** "
            f"= {calibration.tf32_tflops / max(calibration.fp32_tflops, 1e-9):.2f}x the fp32 "
            f"figure. A kernel not using them is limited by the lower number.\n")
    # P3: show the low-precision ceilings too. While these were unmeasured, an fp16 kernel
    # was scored against tf32 and read as >100% of "peak", so the agent was told a candidate
    # with real headroom was saturated. The ratio is COMPUTED from the two measured figures rather
    # than written in as a constant: fp16/tf32 is 1.80 on a 4090 and 2.05 on an A800, and tf32/fp32
    # moves far more (1.61 vs 5.88), so any hardcoded factor is badly wrong on some card. Naming the
    # computed ratio still makes the lever explicit.
    for label, value in (("fp16", calibration.fp16_tflops),
                         ("bf16", calibration.bf16_tflops)):
        if value > 0:
            out.append(
                f"- tensor cores ({label}): **{value:.1f} TFLOP/s** "
                f"= {value / max(calibration.tf32_tflops, 1e-9):.2f}x the tf32 figure. "
                f"Your candidate is measured against the ceiling for the precision it "
                f"actually computes in.\n")
    if calibration.empty_launch_floor_ms > 0:
        out.append(
            f"- smallest possible launch: **{calibration.empty_launch_floor_ms*1e3:.1f} us**. "
            f"Nothing on this box can be faster than this per launch.\n")
    # G10: the figures above are what cuBLAS achieves. Every candidate here is Triton, and on this
    # box a plain correctness-gated Triton matmul reaches only 84% of the cuBLAS fp32 figure while
    # EXCEEDING it at fp16/bf16. Handing an agent the cuBLAS number alone sets a target its own
    # backend cannot hit at fp32, and understates what it can hit at fp16 -- so state the reachable
    # figure wherever the two differ, with its provenance.
    for label, cublas, tri in (("fp32", calibration.fp32_tflops, calibration.fp32_triton_tflops),
                               ("tf32", calibration.tf32_tflops, calibration.tf32_triton_tflops),
                               ("fp16", calibration.fp16_tflops, calibration.fp16_triton_tflops),
                               ("bf16", calibration.bf16_tflops, calibration.bf16_triton_tflops)):
        if tri <= 0 or cublas <= 0:
            continue
        ratio = tri / cublas
        # A 5% band, and unlike the classifier's evidence this one IS a judgement call: the prompt
        # is prose an agent reads, and four near-identical lines bury the two that matter (the
        # failure mode KernelPro measured, where a wall of numbers made the model do worse than no
        # numbers at all). The classifier's evidence has no such cut-off, so nothing is lost -- a
        # 2% gap is still recorded where a decision is audited, just not restated here.
        if ratio < 0.95:
            out.append(
                f"- **{label} from Triton: {tri:.1f} TFLOP/s** ({ratio:.2f}x the "
                f"{cublas:.1f} cuBLAS figure above). Measured with a plain tiled Triton matmul, "
                f"checked against a fp64 reference. Triton is what you are writing in, so treat "
                f"**{tri:.1f}** as the target at {label} -- the remaining "
                f"{(1 - ratio) * 100:.0f}% is the code generator and tiling cannot reach it.\n")
        elif ratio > 1.05:
            out.append(
                f"- **{label} from Triton: {tri:.1f} TFLOP/s**, i.e. {ratio:.2f}x the "
                f"{cublas:.1f} cuBLAS figure -- Triton BEATS the library at this precision on this "
                f"box, so the ceiling for {label} is {tri:.1f}, not {cublas:.1f}.\n")
    for s in calibration.suspect:
        out.append(f"- ⚠ {s}\n")
    out.append("")
    return "".join(out)


def _bottleneck_doc(verdict, task_cost, calibration) -> str:
    """Render the harness's MEASURED bottleneck analysis for the agent (steps 6+7).

    DETECT -> ANALYZE -> RECOMMEND, deliberately not a metric dump. KernelPro measured that
    feeding an LLM raw hardware-counter output made it perform WORSE than feeding it nothing at
    all (NoFeedback beat raw ncu, p=0.0007): a wall of numbers invites the model to pattern-match
    on whichever one looks anomalous. So each section states a conclusion, then the numbers behind
    it so the agent can disagree, then what the conclusion implies is worth trying.

    Three things are stated that a metric dump would omit, and each has cost us a run:

      * WHICH CEILING the percentages are against, and that it was measured on this box rather
        than read off a datasheet. A container's achievable bandwidth is not the card's.
      * WHAT IS UNMEASURABLE here, explicitly. An agent told "nothing is wrong" reasons
        differently from one told "bank conflicts and stalls are unknown on this box".
      * WHETHER the two classification methods agreed. A verdict presented without its
        uncertainty gets trusted exactly where it is least reliable.
    """
    if verdict is None and task_cost is None:
        return ("# Measured bottleneck analysis\n\n"
                "Not available for this run: the harness could not measure this box's ceilings, "
                "so no throughput fraction can be computed. Reason from `tuning/stats.json` and "
                "`tuning/trials.csv` alone, and do NOT assume the absence of a verdict means the "
                "kernel is fine.\n")

    out = ["# Measured bottleneck analysis\n",
           "Produced by the harness from measurements, not by a model. Every number here is "
           "either a direct measurement or a ratio of two measurements; nothing is estimated. "
           "It is ADVISORY -- you may disagree with the verdict, and the evidence is given so "
           "that you can.\n"]

    if task_cost is not None and (task_cost.flop_count or task_cost.compulsory_bytes):
        out.append("## What this task requires (measured on the REFERENCE, so it applies to "
                   "every candidate)\n")
        out.append(f"- {task_cost.summary_line()}\n")
        if task_cost.fusion_headroom > 1.5:
            out.append(
                f"- **The reference materializes {task_cost.fusion_headroom:.1f}x the traffic it "
                f"cannot avoid**, across {task_cost.op_count} ops. That difference is "
                f"intermediates written and re-read. Fusing them away is the largest single lever "
                f"this task offers, and it is a property of the TASK -- no amount of tuning a "
                f"per-op kernel reaches it.\n")
        if calibration is not None and task_cost.compulsory_bytes:
            # NOT guarded on flop_count being nonzero. A 0-FLOP task is the CLEAREST case of
            # "cannot be compute-bound" -- a pure elementwise or pooling op has no
            # multiply-accumulate at all -- and an earlier version of this guard skipped exactly
            # that case, so the one task where the statement is certain was the one task that
            # never got told.
            ridge = calibration.ridge_flop_per_byte
            ai = task_cost.max_arithmetic_intensity
            if ridge > 0 and ai < ridge:
                zero = ("This task performs no multiply-accumulate arithmetic at all. "
                        if task_cost.flop_count == 0 else "")
                out.append(
                    f"- {zero}**This task cannot be compute-bound on this GPU** under any correct "
                    f"implementation: its highest possible intensity is {ai:.2f} FLOP/byte and "
                    f"this card's roofline ridge is {ridge:.1f}. Optimize for BANDWIDTH and for "
                    f"fewer launches; chasing arithmetic throughput cannot pay here.\n")
            elif ridge > 0:
                out.append(
                    f"- This task CAN be compute-bound here ({ai:.2f} FLOP/byte against a ridge "
                    f"of {ridge:.1f}), so arithmetic throughput is a legitimate target.\n")
        out.append("")

    if calibration is not None:
        out.append(_measured_ceilings_doc(calibration))

    if verdict is not None:
        out.append(f"## Verdict: **{verdict.kind}**\n")
        out.append(f"{verdict.suggests}\n")
        if verdict.disagreement:
            out.append(f"\n**Confidence caveat.** {verdict.disagreement}\n")
        out.append("\n### The numbers behind it\n")
        for key, value in verdict.evidence.items():
            if key == "thresholds":
                continue
            out.append(f"- `{key}` = {value}\n")
        th = (verdict.evidence.get("thresholds") or {})
        if th:
            out.append(
                f"\nThresholds used: DRAM saturated at >= {th.get('dram_saturated_frac')} of the "
                f"measured ceiling, compute at >= {th.get('compute_saturated_frac')}, nothing "
                f"saturated below {th.get('idle_frac')}, launch-bound at cpu/gpu >= "
                f"{th.get('launch_bound_cpu_ratio')}. "
                + ("These were DERIVED on this box from workloads whose bottleneck is known "
                   "analytically -- they are not constants.\n" if th.get("calibrated")
                   else "⚠ These are uncalibrated fallbacks; this box's own separation was not "
                        "measured.\n"))
        if verdict.unmeasured:
            out.append(
                "\n### What this analysis CANNOT see\n"
                "Hardware counters need a host-side permission that cannot be set from inside a "
                "container, so the following are **unknown** on this box -- not measured and "
                "found to be fine:\n")
            for item in verdict.unmeasured:
                out.append(f"- {item}\n")
            out.append(
                "\nDo not propose a change whose entire justification is one of these, and do "
                "not treat their absence as evidence that the kernel is clean.\n")
    return "".join(out)


def _tier1_doc(profile) -> str:
    """What the compiled kernel's own instructions say. Step 5's signals, for the agent.

    Separate from the verdict because it answers a different question: the verdict says what
    limits the kernel, this says what the kernel IS. Tool-affinity filtering (KernelPro's term)
    is applied here -- a signal is only shown when it is relevant to this kernel, since an
    irrelevant one costs attention and invites a change that addresses nothing.
    """
    if profile is None:
        return ""
    lines: list[str] = []
    occ_pct = profile.occupancy_pct
    if occ_pct is not None:
        limiter = profile.occupancy_limiter
        lever = {
            "registers": "reduce live values per thread -- a smaller BLOCK_M/BLOCK_N, or fewer "
                         "simultaneous accumulators",
            "shared_memory": "reduce shared usage -- a smaller tile, or fewer pipeline stages",
            "warps_per_block": "raise num_warps, or launch more blocks",
            "blocks_per_sm": "already at the per-SM block cap; occupancy is not the lever",
        }.get(limiter or "", "reduce the per-thread footprint")
        verdict_word = "LOW" if occ_pct < 50 else "adequate"
        lines.append(
            f"- **Theoretical occupancy {occ_pct:.0f}% ({verdict_word})**, limited by "
            f"`{limiter}`. {profile.occupancy.get('active_warps')} of "
            f"{profile.occupancy.get('max_warps_per_sm')} warp slots per SM are usable, at "
            f"{profile.occupancy.get('blocks_per_sm')} blocks/SM. To raise it: {lever}.\n")
        if occ_pct < 50:
            lines.append(
                "  Note this is THEORETICAL occupancy, computed from resource use. Low occupancy "
                "is not automatically bad -- a large-tile kernel can be fastest at low occupancy "
                "because each thread does more work. Treat it as a hypothesis to test, not a "
                "defect to fix.\n")
    tc = profile.uses_tensor_cores
    if tc is False:
        lines.append(
            "- **No tensor-core instructions** in the compiled kernel (checked by disassembling "
            "the cubin: no HMMA/IMMA/BMMA/OMMA). If this task's arithmetic is a matmul-like inner "
            "product, moving it onto tensor cores raises the ceiling rather than approaching it.\n")
    elif tc is True:
        lines.append(f"- Tensor cores ARE in use ({profile.sass.get('tensor_core')} instructions "
                     f"of {profile.sass.get('instructions')}).\n")
    if profile.n_spills:
        lines.append(
            f"- **Register spills: {profile.n_spills}**. Spilled values go to local memory, which "
            f"is DRAM-backed, so a spill inside the inner loop can cost more than the arithmetic "
            f"it was making room for.\n")
    sass = profile.sass or {}
    gl = (sass.get("global_load") or 0) + (sass.get("global_store") or 0)
    if gl and sass.get("instructions"):
        vec = sass.get("vec_128") or 0
        if vec == 0:
            lines.append(
                f"- All {gl} global memory accesses are narrower than 128-bit. Wider accesses "
                f"move the same bytes in fewer transactions, though this is usually a small win "
                f"compared with occupancy or fusion.\n")
    if not lines:
        return ""
    return ("# What the compiled kernel actually is\n\n"
            "Read from the compiled code itself (disassembly + compiler metadata), so these are "
            "facts about the binary rather than inferences:\n\n" + "".join(lines))


def _device_doc(device: DeviceLimits, calibration=None) -> str:
    """The target device: hard per-block limits, then what the box measurably achieves.

    `calibration` is optional so a box without one still renders a valid doc, and so this stays
    callable from tests that have only a DeviceLimits. But pass it wherever it exists: the limits
    alone are what a candidate must not EXCEED, while the ceilings are what it is measured
    AGAINST, and only the second kind makes "is this kernel fast" answerable.
    """
    return (
        f"# Target device\n\n"
        f"- {device.name}\n"
        f"- VRAM: {device.vram_gb} GB\n"
        f"- Max registers/thread: {device.max_regs_per_thread}\n"
        f"- Max static shared memory/block: {device.max_shared_bytes_static} B\n"
        f"- Max opt-in shared memory/block: {device.max_shared_bytes_optin} B\n"
        f"- Max threads/block: {device.max_threads_per_block}\n"
        + ("\n" + _measured_ceilings_doc(calibration) if calibration is not None else "")
    )


def _eval_semantics_doc(semantics: dict | None) -> str:
    """Improvement J: render the probed reference eval semantics (train/eval mode +
    norm-layer flags) as a task fact for the agent. Empty/None -> a neutral note so
    the contract degrades gracefully (no forced assumption)."""
    if not semantics:
        return (
            "# Reference evaluation semantics\n\n"
            "(Not probed for this run. Infer the reference's run mode from its source: "
            "if it does not call `.eval()`, an nn.Module defaults to TRAIN mode, which "
            "changes BatchNorm/Dropout behavior — match whatever the reference actually does.)\n"
        )
    mode = "TRAIN" if semantics.get("training") else "EVAL"
    lines = [
        "# Reference evaluation semantics\n",
        f"The harness ran the reference model and observed it in **{mode} mode** "
        f"(`model.training == {bool(semantics.get('training'))}`). Your kernel MUST "
        f"reproduce the semantics of THIS mode, not an assumed one.\n",
    ]
    norm = semantics.get("norm_layers") or []
    if norm:
        lines.append("Normalization layers detected (each with its own runtime state):\n")
        for n in norm:
            lines.append(
                f"- `{n.get('type')}`: training={n.get('training')}, "
                f"has_running_stats={n.get('has_running_stats')}, "
                f"track_running_stats={n.get('track_running_stats')}, "
                f"momentum={n.get('momentum')}"
            )
        lines.append(
            "\n**BatchNorm/InstanceNorm in TRAIN mode normalize with the CURRENT BATCH "
            "mean/var — NOT running_mean/running_var** (which are 0/1 on an untrained "
            "model and give a large systematic error). In EVAL mode they use "
            "running_mean/running_var. Match each layer's stated `training` flag."
        )
    else:
        lines.append("(No normalization layers with running-stat buffers were detected.)")
    return "\n".join(lines) + "\n"


def _rejected_repairs_doc(prior: list[dict]) -> str:
    """Same-candidate repair history, so the agent cannot silently re-try or invert a
    fix that was already rejected (L3:48 cand-0137895f oscillated between exp(A) and
    exp(-exp(A)) because each repair call saw only the current error)."""
    out = [
        "# Your earlier repairs of THIS candidate — all rejected",
        "",
        "Each entry is a fix you already produced and the failure it still hit. Treat",
        "every diagnosis below as DISPROVEN. Re-proposing one, or merely inverting one,",
        "wastes an attempt: if a claim and its opposite both appear here, neither is the",
        "cause and you must look elsewhere (indexing, masking, accumulation order,",
        "dtype, or a boundary/tail case).",
        "",
    ]
    for i, item in enumerate(prior, 1):
        out.append(f"## Attempt {i} — REJECTED")
        out.append(f"- Your diagnosis: {str(item.get('diagnosis', '')).strip()}")
        detail = str(item.get("failure_detail", "")).strip()
        if detail:
            out.append(f"- Still failed with: {detail}")
        out.append("")
    return "\n".join(out)


def _repair_guidance(failure_kind: str) -> str:
    """Failure-class-specific repair hints (improvement F): route the repair to the
    likely cause instead of a generic 'fix it' prompt. Purely additive — the agent
    still diagnoses from the actual failure detail in failure/detail.txt."""
    numeric = (
        "This is a NUMERICAL error: the kernel compiled and ran but its output did "
        "not match the reference within tolerance. First check the ERROR MAGNITUDE in "
        "failure/detail.txt: a LARGE, roughly constant systematic offset (not a few "
        "outliers) usually means a SEMANTIC mismatch, not a precision one — most often "
        "BatchNorm/normalization run in the wrong mode. Read `task/eval_semantics.md`: "
        "if the reference is in TRAIN mode, BatchNorm must use the CURRENT BATCH "
        "mean/var, NOT running_mean/running_var (which are 0/1 on an untrained model "
        "and cause exactly this kind of large offset). A SMALL error just over tolerance "
        "is a precision issue: accumulate dot products/reductions in fp32 (use "
        'input_precision="ieee" for tl.dot on fp32 refs), check reduction order and '
        "masking of padded lanes, and keep softmax/normalization numerically stable "
        "(subtract the row max before exp)."
    )
    compile_ = (
        "This is a COMPILE/RUNTIME error: the kernel failed to build or crashed. Focus "
        "on Triton language constraints — every tl.arange bound and tile size must be a "
        "compile-time tl.constexpr power of two, tl.dot input dims must be divisible by "
        "16, never call tl.next_power_of_2 inside device code (compute it on the host "
        "and pass it in as a tl.constexpr), and mask every out-of-bounds load/store."
    )
    oom = (
        "This is an OUT-OF-MEMORY error: reduce per-program memory — smaller tiles, "
        "fewer pipeline stages, or stream the reduction instead of materializing large "
        "intermediates."
    )
    excessive = (
        "The kernel was measured as MORE THAN 10x FASTER than the reference. On this "
        "hardware that is not a real optimization — it means the kernel is not doing "
        "the reference's work. Look for: an output that is allocated but never filled "
        "(or filled from a cached/stale buffer), a grid that covers only part of the "
        "output, a loop bound that skips most of the reduction, work moved outside the "
        "timed region, or an early return on a condition that is always true. Fix the "
        "kernel so it computes the full result; do NOT try to make the timing look "
        "more plausible."
    )
    mapping = {
        "correctness_mismatch": numeric,
        "compile_error": compile_,
        "runtime_error": compile_,
        "static_check_failed": compile_,
        "oom": oom,
        "excessive_speedup": excessive,
    }
    return mapping.get(failure_kind,
                       "Diagnose from the failure detail and fix the root cause.")


def _files_exist_check(files: list[str], sb: Sandbox) -> str | None:
    missing = [f for f in files if not sb.exists(f)]
    if missing:
        return (f"your JSON references files that do not exist in the workspace: "
                f"{missing}. Write each file to disk, then answer again.")
    return None


def _detect_backend(source: str) -> str:
    """Which backend a candidate file uses, read from the source rather than declared.

    Needed by the sandbox rescues: when a transport failure kills a call, the agent's JSON --
    which is where `backend` normally comes from -- never arrives, but the file it wrote is on
    disk. And `backend` is NOT narration that can be left blank: `structural_signature` hashes
    it (families.py:79-91), so guessing wrong would make a CUDA candidate collide with a Triton
    one or vice versa, and `accept_novel_seed` would reject a genuinely new structure as a
    duplicate.

    The markers are unambiguous in the direction that matters: `load_inline`/`cpp_extension` is
    the only way a candidate compiles CUDA C here, and `@triton.jit`/`tl.` is the only way it
    writes a Triton kernel. Defaults to "triton" for a file with neither, matching both prompts'
    stated default and the schema's own default -- a rescued file that has no kernel at all is
    rejected by `check_output` regardless of what this returns, so the default cannot smuggle
    anything past the gate.
    """
    if "load_inline" in source or "cpp_extension" in source or "CUDAExtension" in source:
        return "cuda"
    return "triton"


def _rescued_files(sb: Sandbox, rel_dir: str) -> list[str]:
    """Output files an agent wrote before a transport failure, or []. Shared by the rescues.

    A single place so a new producing module gets the same behaviour by calling it rather than
    by reimplementing the walk -- the defect this addresses was precisely that only ONE of the
    three file-producing modules had a rescue.
    """
    return sb.list_outputs(rel_dir)


def _triton_lint_check(files: list[str], sb: Sandbox) -> str | None:
    """Improvement C: reject certain Triton compile-failures before the GPU sees
    them, feeding the specific problem back into the agent's own retry loop. A
    no-op for non-Triton files (no @triton.jit body -> no findings).

    Also enforces the contract's Backend rule that a candidate must contain a
    kernel at all. That check has to live here rather than inside
    `lint_triton_source`, because the lint walks `@triton.jit` bodies and a file
    with none has nothing to walk — the absence is exactly what must be reported.
    """
    problems: list[str] = []
    for f in files:
        try:
            src = sb.read_output(f)
        except (OSError, ValueError):
            continue  # existence is checked separately; don't double-fault here
        no_kernel = declares_no_custom_kernel(src)
        if no_kernel:
            problems.append(f"{f}: {no_kernel}")
        delegated = delegates_to_baseline_compiler(src)
        if delegated:
            problems.append(f"{f}: {delegated}")
        hard_errors, _warnings = lint_triton_source(src)
        for err in hard_errors:
            problems.append(f"{f}: {err}")
    if problems:
        return ("Static Triton check rejected your kernel(s) before evaluation. "
                "Fix these and answer again:\n- " + "\n- ".join(problems))
    return None


def _triton_lint_warnings(files: list[str], sb: Sandbox) -> list[str]:
    """Improvement L: collect NON-BLOCKING lint warnings (e.g. hardcoded fp16 cast
    with no dtype knob). Surfaced as advisories; never rejects a candidate."""
    warns: list[str] = []
    for f in files:
        try:
            src = sb.read_output(f)
        except (OSError, ValueError):
            continue
        _hard, file_warnings = lint_triton_source(src)
        for w in file_warnings:
            warns.append(f"{f}: {w}")
    return warns


# --- 1. candidate generator ---------------------------------------------------


@dataclass
class GeneratorInputs:
    task: TaskSpec
    ref_source: str
    device: DeviceLimits
    n_candidates: int
    eval_semantics: dict | None = None
    # This box's measured ceilings, so the prompt can state what the card ACHIEVES and
    # not merely what it forbids. Optional: a box without a calibration still runs.
    calibration: object | None = None



class CandidateGeneratorAgent(AgentModule[GeneratorInputs, GenerationResult]):
    name = "generator"
    output_model = GenerationResult

    def seed_sandbox(self, inputs: GeneratorInputs, sb: Sandbox) -> None:
        sb.write_input("task/ref.py", inputs.ref_source)
        sb.write_input("docs/candidate_contract.md", _contract_doc())
        sb.write_input("docs/triton_pitfalls.md", _triton_pitfalls_doc())
        sb.write_input("docs/device.md", _device_doc(inputs.device, inputs.calibration))
        sb.write_input("task/eval_semantics.md", _eval_semantics_doc(inputs.eval_semantics))

    def render_prompt(self, inputs: GeneratorInputs, sb: Sandbox) -> str:
        return f"""You are optimizing a GPU operator from KernelBench.

The reference PyTorch implementation is in `task/ref.py` (task: {inputs.task.name},
level {inputs.task.level}). Read it carefully, then read
`docs/candidate_contract.md`, `docs/device.md`, and `task/eval_semantics.md` (which
tells you the run mode — train vs eval — the reference is evaluated in; match it,
especially for BatchNorm). If you write any Triton
(`@triton.jit`) kernel, you MUST also read `docs/triton_pitfalls.md` first and
obey every rule that applies — they are compiler/correctness hard constraints,
not style preferences.

Write {inputs.n_candidates} candidate kernel implementations, each in its own file
`candidates/cand_1.py`, `candidates/cand_2.py`, ... Each candidate must follow the
contract exactly (ModelNew + a PARAMS dict of tunable knobs).

CRITICAL: the candidates must differ in COMPUTATIONAL APPROACH, not just in
parameter defaults or code style. Vary along axes such as: work partitioning
(what each thread block owns), data placement (shared memory vs registers vs
recompute), fusion boundaries (which ops are fused into one kernel), thread
communication (warp shuffle vs shared memory reduction), or kernel organization
(single fused kernel vs a small pipeline of kernels).

PRECISION / TENSOR CORES: for matmul- or convolution-bound work, the dot-product
precision is a first-class approach axis — read the "Precision and the tensor-core
path" section of the contract. A kernel that runs `tl.dot(..., input_precision="ieee")`
(or scalar FMA loops) leaves the tensor cores idle; a tf32 tensor-core path
(`input_precision="tf32"`, or fp16/bf16 inputs with an fp32 accumulator) is often
materially faster on this class of card and is what torch.compile uses -- the MEASURED
ratio for this box is in the ceilings block above, so use that number rather than
assuming one. The dual-precision correctness gate
accepts a tf32-matching result, so at least one of your candidates SHOULD take the
tf32 tensor-core path (with an fp32 accumulator), and you should expose the dot
precision as a PARAMS knob (e.g. "DOT_PRECISION": "tf32") so the tuner can compare
it against "ieee" on real measurements.

You may run quick syntax checks (e.g. `python -c "import ast; ast.parse(open('candidates/cand_1.py').read())"`),
but you cannot run GPU code here — the harness evaluates on the GPU afterwards.

When done, answer with JSON:
{{"candidates": [{{"file": "candidates/cand_1.py", "backend": "triton",
  "approach_summary": "<1-2 sentences>", "structural_axes": ["<axis>", ...]}}, ...]}}
"""

    def check_output(self, output: GenerationResult, sb: Sandbox) -> str | None:
        if not output.candidates:
            return "empty candidate list; produce at least one candidate"
        files = [c.file for c in output.candidates]
        missing = _files_exist_check(files, sb)
        if missing:
            return missing
        # Every candidate is checked for "has a kernel at all", regardless of the
        # backend it declares — a `cuda`-declared file with neither a jit kernel nor
        # an inline extension is the same contract violation. The Triton-specific
        # compile-failure patterns are a no-op on a genuine CUDA file, so passing all
        # files here costs nothing and closes the backend-label loophole.
        return _triton_lint_check(files, sb)

    def soft_check(self, output: GenerationResult, sb: Sandbox) -> list[str]:
        triton_files = [c.file for c in output.candidates if c.backend == "triton"]
        return _triton_lint_warnings(triton_files, sb)

    def rescue_from_sandbox(self, sb: Sandbox) -> GenerationResult | None:
        """Rebuild the result from `candidates/*.py` the agent already wrote.

        The same rescue the rewriter has, in the module where a loss is most expensive: these are
        the SEED candidates, so a discarded generator call costs the run its entire starting
        population for that attempt, and every family that would have descended from it.

        `approach_summary` and `structural_axes` are narration -- nothing gates on them (they are
        recorded for the report and the analyst's context). `backend` is read from the source,
        because `structural_signature` hashes it. Whatever this returns still passes through
        `check_output`, so a half-written file is rejected exactly as it would be on the normal
        path.
        """
        files = _rescued_files(sb, "candidates")
        if not files:
            return None
        out = []
        for f in files:
            try:
                source = sb.read_output(f)
            except OSError:
                continue
            out.append(GeneratedCandidate(
                file=f, backend=_detect_backend(source),
                approach_summary="[recovered from sandbox after a transport failure; the "
                                 "agent's own summary never arrived]",
                structural_axes=[]))
        return GenerationResult(candidates=out) if out else None


# --- 2. parameterizer -----------------------------------------------------------


@dataclass
class ParameterizerInputs:
    task: TaskSpec
    candidate_source: str
    device: DeviceLimits
    prior_feedback: str = ""
    # Attribution only (see RepairInputs.candidate_id).
    candidate_id: str | None = None
    # Improvement K: when set, a focused space-EXPANSION request rather than a
    # fresh parameterization. Describes which knobs hit the tried-range boundary
    # (with direction) so the agent extends only those choices, structure unchanged.
    expand_directive: str = ""
    # The constraints of the space being expanded, as (expr, rationale) pairs.
    # An expansion re-declares the WHOLE space, but the constraints live in the
    # space object rather than in `candidate/source.py`, so without this the agent
    # is asked to reproduce them from memory of a file it cannot see. It reliably
    # cannot: measured over all 30 expansions on record, the replacement space
    # admitted configurations its predecessor had excluded in 21 of them
    # (15.7% of the shared sub-grid), and those newly-admitted configurations won
    # nothing on any candidate (0 of 21) while failing at 48.3% against 26.0% for
    # the doubly-legal region.
    prior_constraints: tuple[tuple[str, str], ...] = ()
    # This box's measured ceilings, so the prompt can state what the card ACHIEVES and
    # not merely what it forbids. Optional: a box without a calibration still runs.
    # The parameterizer needs these as much as the writers do -- it chooses the tile and
    # precision DOMAINS, and "is fp16 worth a choice here" is a question about the ratio
    # between two ceilings it otherwise cannot see.
    calibration: object | None = None


class ParameterizerAgent(AgentModule[ParameterizerInputs, ParameterizationResult]):
    name = "parameterizer"
    output_model = ParameterizationResult

    def seed_sandbox(self, inputs: ParameterizerInputs, sb: Sandbox) -> None:
        sb.write_input("candidate/source.py", inputs.candidate_source)
        sb.write_input("docs/candidate_contract.md", _contract_doc())
        # The parameterizer REWRITES the kernel body (and, on an expansion, chooses new
        # tile-dimension values), so every Triton rule that constrains a tile dimension
        # binds it exactly as it binds the generator/rewriter. It was the only
        # Triton-writing agent that never received this doc.
        sb.write_input("docs/triton_pitfalls.md", _triton_pitfalls_doc())
        sb.write_input("docs/device.md", _device_doc(inputs.device, inputs.calibration))

    def render_prompt(self, inputs: ParameterizerInputs, sb: Sandbox) -> str:
        if inputs.expand_directive:
            return self._render_expand_prompt(inputs)
        feedback = (
            f"\nPrevious attempt was rejected: {inputs.prior_feedback}\n"
            if inputs.prior_feedback
            else ""
        )
        return f"""A candidate kernel is in `candidate/source.py`. Read it, plus
`docs/candidate_contract.md` and `docs/device.md`. If the kernel is Triton
(`@triton.jit`), you MUST also read `docs/triton_pitfalls.md` and obey it — both in
the rewritten body AND when choosing each knob's choices: every value you offer for a
tile dimension must be legal there (e.g. the CONTRACTION dimension of a `tl.dot` must
be at least 16, so 8 is never a legal choice for a K tile, though it may be legal for
an M or N tile).{feedback}

Your job: parameterize every tunable feature of this kernel.

1. Rewrite the file as `candidate/parameterized.py` so ALL tunable knobs flow
   through one module-level `PARAMS = {{...}}` dict (contract section "PARAMS
   block"). Tunables typically include block/tile sizes, num_warps, num_stages,
   vectorization widths, split factors, group sizes. If the kernel does a matmul
   or convolution via `tl.dot`, expose the COMPUTE PRECISION as a SINGLE unified
   knob (e.g. "COMPUTE_DTYPE" with choices ["fp16", "bf16", "tf32", "ieee"]) that
   drives BOTH the input cast AND the tl.dot precision consistently:
   - "fp16"/"bf16" → cast the dot inputs to tl.float16/bfloat16 (tensor cores);
   - "tf32" → keep fp32 inputs with `input_precision="tf32"`;
   - "ieee" → keep fp32 inputs with `input_precision="ieee"`.
   The accumulator MUST stay fp32 for every choice. Do NOT hard-code a dtype cast
   in the kernel body and separately add a mismatched precision knob — the one knob
   must control the actual precision the kernel computes in, so the tuner can
   compare precisions on real measurements. On matmul/conv-bound work this is
   usually the highest-impact tunable.
2. For each PARAMS key, propose the list of values worth trying, ordered from
   cheapest (least resources) to most expensive. Keep each list to 2-8 values.
   The current PARAMS default must be included in its list.
3. Propose constraints that rule out illegal/doomed combinations, as boolean
   expressions over the PARAMS names (operators: + - * / // % ** comparisons
   and/or). You may reference device constants: MAX_REGS_PER_THREAD,
   MAX_SHARED_BYTES, MAX_SHARED_BYTES_OPTIN, MAX_THREADS_PER_BLOCK.
   **DO NOT write a shared-memory constraint.** Shared usage is decided by the Triton
   compiler, not by your arithmetic: `metadata.shared` includes multi-buffering,
   alignment padding and intermediates the compiler allocates, and it is not even
   monotonic in the tile dimensions (measured: BLOCK_M=128/BLOCK_N=32/stages=1 and
   BLOCK_M=128/BLOCK_N=64/stages=1 both compile to exactly 65536 bytes). Hand-written
   bounds of the form `stages * elements * bytes <= MAX_SHARED_BYTES_OPTIN` were
   audited on real candidates and came out at a MEDIAN 32% of the compiler's own
   figure -- one such constraint admitted all 36 configurations of its candidate,
   including the 17 that cannot launch, and rejected none. It was indistinguishable
   from having no constraint at all, while making it look as though the dimension were
   protected. **The harness screens this for you** by compiling the configuration
   before it spends a trial on it, using the compiler's number. Leave it to that, and
   spend your constraint budget on things arithmetic can actually decide:
   (a) ONE CONSTRAINT PER LAUNCHED KERNEL, for the limits you CAN compute: thread count
       (`NUM_WARPS * 32 <= MAX_THREADS_PER_BLOCK`) and any relation the kernel body
       requires between knobs (a tile that must divide another, a `tl.dot` contraction
       dimension that must be at least 16, a split factor that must not exceed a
       dimension). These are exact, and a violation is a compile error rather than a
       measurement.
   (b) DERIVE THEM FROM THIS KERNEL'S BODY, not from a template. Read the
       `tl.load`/`tl.dot` shapes and the launch grid and write down what they require.
       A plausible-looking formula copied from elsewhere is worse than none, because it
       reads as protection while admitting the same failures.
   GRAMMAR LIMIT: a constraint is evaluated by a restricted parser that allows ONLY
   names, numeric/string literals, `+ - * / // % **`, the comparisons
   `< <= > >= == !=`, and `and`/`or`/`not`.
   Conditional expressions (`A if C else B`), function calls (`min()`, `max()`, `abs()`),
   membership tests (`in`, `not in`), identity tests (`is`, `is not`), indexing, and
   comprehensions are REJECTED. Express a conditional rule as a
   disjunction instead — e.g. instead of
   `(4 if DTYPE == "fp16" else 2) * BLOCK_M <= X`, write
   `(DTYPE != "fp16" and 2 * BLOCK_M <= X) or (DTYPE == "fp16" and 4 * BLOCK_M <= X)`.
   Express a membership test as a disjunction of `==` too — instead of
   `DTYPE in ("fp16", "bf16")`, write `DTYPE == "fp16" or DTYPE == "bf16"`.
   ONE THING THE TILE DOMAIN MUST DO, since no constraint will do it for you: if you
   expose a precision knob, keep the tile choices small enough that EVERY precision you
   offer has at least one launchable configuration. A tile sized for 2 bytes/element
   often needs twice the shared memory at 4, and a precision whose every configuration
   is infeasible is a precision the tuner silently never measures -- on one real
   candidate that cost two of the four precisions it declared.

The rewritten kernel must be functionally identical to the original at the
default PARAMS values.

IMPORTANT: actually create `candidate/parameterized.py` on disk with your file
tools before answering. A JSON answer that references a file you did not write
is rejected and wastes an attempt.

Answer with JSON:
{{"file": "candidate/parameterized.py",
  "space": {{"params": [{{"name": "BLOCK_M", "kind": "int", "choices": [32, 64, 128],
             "description": "..."}}, ...],
            "constraints": [{{"expr": "...", "rationale": "..."}}, ...]}}}}
"""

    def _render_expand_prompt(self, inputs: ParameterizerInputs) -> str:
        """Improvement K: focused space EXPANSION — extend only the boundary knobs'
        choices toward the improving direction, keeping structure and other knobs."""
        if inputs.prior_constraints:
            prior = "\n".join(
                f"  - `{expr}`" + (f"  ({why})" if why else "")
                for expr, why in inputs.prior_constraints
            )
            prior_block = (
                "\nThe space you are expanding ALREADY HAS these constraints. They are part\n"
                "of the space, not of `candidate/source.py`, so this listing is the only\n"
                "place you can read them. Repeat every one that still applies verbatim in\n"
                "your response:\n"
                f"{prior}\n"
                "\nDropping one silently re-admits configurations that were deliberately\n"
                "excluded, and the tuner will spend trials launching them. If the kernel\n"
                "body no longer matches a constraint, replace it with the corrected form and\n"
                "say so in its rationale — but do not simply omit it.\n"
            )
        else:
            prior_block = ""
        return f"""A candidate kernel is in `candidate/source.py` (already parameterized
with a `PARAMS` dict). Read it plus `docs/candidate_contract.md` and `docs/device.md`.
If the kernel is Triton, also read `docs/triton_pitfalls.md`: every value you ADD must
be legal there. In particular the CONTRACTION dimension of a `tl.dot` (the shared K of
`(M,K) x (K,N)`) must be at least 16 — Triton rejects a smaller one outright with
`Input shapes should have M >= 1, N >= 1 and K >= 16`. So check which dot dimension a
knob actually feeds before expanding it downward: an M or N tile may legally go below
16, a K tile may not. When the improving direction is downward and the knob is already
at its legal floor, expand a different knob or leave that domain unchanged — an illegal
value makes the witness fail to compile and the ENTIRE expansion is rejected.

During tuning, some knobs reached the EDGE of the value range that was offered and
latency was still improving toward that edge, while hardware resources still had
headroom. Your job is a FOCUSED EXPANSION, not a redesign:

{inputs.expand_directive}
{prior_block}
Rules:
- Keep the kernel STRUCTURE and all other knobs' choices UNCHANGED. Rewrite the file
  to `candidate/parameterized.py` (it may be nearly identical to the source — only
  the PARAMS choices for the named knobs and any dependent constraints change).
- For each named knob, ADD 1-2 larger/smaller legal values in the improving
  direction (e.g. append 256 to [64,128,256]... keep values legal for tl.dot dims,
  powers of two where the kernel requires it, and within device limits).
- Keep the current default value in each list. Update constraints only as needed so
  the new values remain feasible under hardware limits (registers/shared/threads).
- CONSTRAINT GRAMMAR: constraints are evaluated by a restricted parser allowing ONLY
  names, numeric/string literals, `+ - * / // % **`, the comparisons
  `< <= > >= == !=`, and `and`/`or`/`not`.
  Conditional expressions (`A if C else B`), function calls (`min`/`max`/`abs`),
  membership tests (`in`, `not in`), identity tests (`is`), and indexing are REJECTED —
  express conditional rules as a disjunction of `and` clauses, and membership tests as
  a disjunction of `==` (`DTYPE == "fp16" or DTYPE == "bf16"`).
- Do NOT introduce new knobs or remove existing ones.
- **DECLARE EVERY KNOB**: `space.params` MUST list ALL keys of the `PARAMS` dict —
  not just the ones you expanded. The unexpanded knobs are repeated verbatim with
  their existing choices. A response whose declared names differ from the PARAMS keys
  is rejected outright (`key_mismatch`) and wastes the attempt.
- **NEVER SHRINK A KNOB**: expansion only ADDS values. Every knob must keep at least
  2 choices, and you must not drop values that were already offered — not even ones
  that measured poorly. A knob left with a single choice is rejected
  (`degenerate_domain`) and wastes the attempt. If you believe a value is useless,
  keep it in the list anyway; the tuner will avoid it on its own.

IMPORTANT: actually create `candidate/parameterized.py` before answering.

Answer with JSON:
{{"file": "candidate/parameterized.py",
  "space": {{"params": [{{"name": "...", "kind": "int", "choices": [...],
             "description": "..."}}, ...],
            "constraints": [{{"expr": "...", "rationale": "..."}}, ...]}}}}
"""

    def check_output(self, output: ParameterizationResult, sb: Sandbox) -> str | None:
        if not output.space.params:
            return "space.params is empty; declare at least two tunable parameters"
        missing = _files_exist_check([output.file], sb)
        if missing:
            return missing
        # The parameterizer REWRITES the kernel body, and its output is the source
        # that actually gets tuned and reported — yet it was the only Triton-writing
        # agent with no static gate. So a contract violation that reached it (or that
        # it introduced while rewriting) was never checked again before 40 GPU trials
        # were spent on it. Same reason it needed triton_pitfalls.md:
        # finding-parameterizer-lacks-triton-pitfalls-doc.md.
        return _triton_lint_check([output.file], sb)


# --- 3. bottleneck analyst -------------------------------------------------------


@dataclass
class AnalystInputs:
    task: TaskSpec
    candidate_source: str
    stats: TuningStats
    trials_csv: str
    device: DeviceLimits
    # Attribution only (see RepairInputs.candidate_id).
    candidate_id: str | None = None
    eval_semantics: dict | None = None
    # Improvement M: @triton.jit kernels defined in the source that NO trial launched.
    never_launched_kernels: list[str] = field(default_factory=list)
    # Steps 6+7: the harness's own MEASURED bottleneck analysis, the task's cost, this box's
    # ceilings, and what the compiled kernel is. All optional -- the prompt degrades to "not
    # available" rather than assuming, because a box that could not be calibrated must still run.
    bottleneck_verdict: object | None = None
    task_cost: object | None = None
    calibration: object | None = None
    profile: object | None = None


class BottleneckAnalystAgent(AgentModule[AnalystInputs, BottleneckReport]):
    name = "analyst"
    output_model = BottleneckReport

    def seed_sandbox(self, inputs: AnalystInputs, sb: Sandbox) -> None:
        sb.write_input("candidate/source.py", inputs.candidate_source)
        sb.write_input("tuning/stats.json", inputs.stats.model_dump_json(indent=2))
        sb.write_input("tuning/trials.csv", inputs.trials_csv)
        sb.write_input("docs/device.md", _device_doc(inputs.device))
        sb.write_input("task/eval_semantics.md", _eval_semantics_doc(inputs.eval_semantics))
        # Steps 6+7. Written unconditionally (the renderer says "not available" when it has
        # nothing) so the agent's reading list is the same shape on every box -- a file that
        # sometimes does not exist is a file the agent learns to stop opening.
        sb.write_input("analysis/bottleneck.md",
                       _bottleneck_doc(inputs.bottleneck_verdict, inputs.task_cost,
                                       inputs.calibration))
        tier1 = _tier1_doc(inputs.profile)
        if tier1:
            sb.write_input("analysis/compiled_kernel.md", tier1)
        if inputs.never_launched_kernels:
            sb.write_input(
                "tuning/never_launched_kernels.md",
                "# Kernels that NEVER ran\n\n"
                "These `@triton.jit` kernels are defined in `candidate/source.py` but were "
                "launched by **zero** trials in the entire tuning budget (measured from each "
                "trial's compiled-kernel metadata, not inferred):\n\n"
                + "".join(f"- `{k}`\n" for k in inputs.never_launched_kernels)
                + "\nThey are DEAD CODE, so every latency number in `tuning/trials.csv` "
                "measures the other path. If one of these is the candidate's advertised "
                "optimization, that optimization has never been evaluated and the tuning "
                "result says nothing about it. The usual cause is a guard the harness's "
                "fixed run mode never selects (see `task/eval_semantics.md`) — check the "
                "`kernels_launched` column in `tuning/trials.csv` to see what did run. "
                "Say so explicitly in your summary, and make your first hypothesis "
                "moving the optimization onto the live path.\n",
            )

    def render_prompt(self, inputs: AnalystInputs, sb: Sandbox) -> str:
        dead = ""
        if inputs.never_launched_kernels:
            dead = (
                "\n\nSTOP AND READ `tuning/never_launched_kernels.md` FIRST. The harness "
                "measured that "
                + ", ".join(f"`{k}`" for k in inputs.never_launched_kernels)
                + " was launched by ZERO trials, so it is dead code and every latency "
                "number below measures a different path. Address that before any "
                "resource analysis: an unreached kernel is not a slow kernel."
            )
        return """A kernel candidate was tuned over its parameter space. These files
already exist in your working directory — read them with your file tools
before answering; do NOT assume any are missing (a stale index may hide them,
so read by path):
- `analysis/bottleneck.md` — the HARNESS'S OWN measured analysis: what this task
  requires, what this GPU can actually do (measured on this box, not a datasheet),
  which resource limits this kernel, and explicitly what could NOT be measured
  here. Read this FIRST: it is measurement, and it tells you which of your
  hypotheses are already ruled out.
- `analysis/compiled_kernel.md` — what the compiled binary actually is: occupancy
  and its limiting resource, whether tensor cores are used, spills, access widths.
  Read from the disassembly, so these are facts about the binary. (Absent when the
  compiled code could not be read.)
- `candidate/source.py` — the kernel (PARAMS dict = tunable knobs)
- `tuning/stats.json` — per-parameter statistics: best value, whether the optimum
  sits at a boundary of the tried range (`at_boundary` + direction), effect size,
  failure rates per value, resource usage (registers/shared memory/spills) at the
  best config, and failure clusters
- `tuning/trials.csv` — the full trial log (params, status, latency, resources,
  and `kernels_launched`: which Triton kernels each trial actually ran)
- `docs/device.md` — hardware limits
- `task/eval_semantics.md` — the run mode the harness evaluates the reference in
  (train vs eval) and the state of each normalization layer

Analyze which parameters still have headroom but are BLOCKED — i.e. the latency
trend keeps improving toward a boundary value, and going further fails or is
prevented by a hardware/resource limit (registers, shared memory, threads, OOM,
compile failures). You may compute things (python is available) over trials.csv.

CRITICAL — do not confuse "a resource is saturated" with "that resource is the
performance limiter." Reason about the WHOLE resource balance before proposing a
change:
1. Resource balance: compare each resource at the best config against its device
   limit. High register use (even at the 255/thread max) is often the SIGNATURE of
   the fast configuration (large accumulator tiles live in registers), NOT a
   pathology to relieve — relieving it by spilling to shared memory or recomputing
   usually makes latency WORSE. Only call a saturated resource "blocking" if the
   trial data shows latency still wants to move toward a value that resource
   forbids AND a lower-usage config is not already just as fast.
2. Idle resources: if a resource is far below its limit (e.g. shared memory at 24%
   while registers are maxed), ask whether the kernel could trade the saturated
   resource for the idle one to raise arithmetic throughput — but only if the trial
   data suggests throughput (not that resource) is the wall.
3. Precision / tensor-core path: check how the kernel does its core math. If it
   uses full-IEEE fp32 matmul (e.g. tl.dot(..., input_precision="ieee")) or scalar
   FMA loops, it is NOT using the tensor cores, and a tf32/fp16-accumulate tensor-
   core path can be materially faster on matmul/conv-bound ops -- by the ratio between
   this box's MEASURED ceilings, not a fixed factor (this is how torch.compile
   wins). If the flat latency floor across many configs looks like an arithmetic-
   throughput wall rather than a memory/occupancy wall, say so and propose switching
   the dot path to tf32 (input_precision="tf32") or fp16 inputs with fp32
   accumulation — the harness's dual-precision correctness gate accepts a tf32-
   matching result, so this is allowed. This is frequently the single highest-impact
   change and must be considered explicitly, not omitted.

Then propose concrete structural-change hypotheses that address the REAL limiter
(e.g. "switch tl.dot to input_precision='tf32' to use tensor cores", "split K so
each block needs less shared memory", "two-stage reduction to allow larger tiles").
Prefer a precision/tensor-core hypothesis when the evidence points to an arithmetic
throughput floor.

Every hypothesis must be EXECUTABLE UNDER THE RUN MODE in
`task/eval_semantics.md`. A fusion that is only valid in the other mode is worth
nothing here: if the reference runs in TRAIN mode, BatchNorm uses the CURRENT
BATCH mean/var, so its scale/shift are not known until the batch has been reduced
and CANNOT be folded into preceding weights. Do not propose folding
`running_mean`/`running_var` (or any "inference batch-norm" fold) when the mode is
TRAIN — a rewriter that implements it will produce a branch the harness never
executes, and the whole rewrite is wasted. State the mode you assumed in the
hypothesis `risk` field.

Answer with JSON matching:
{"summary": "...",
 "parameter_limits": [{"param": "...", "headroom_direction": "increase|decrease",
   "blocked_by": "registers|shared_memory|threads|oom|compile_failure|arithmetic_throughput|none",
   "predicted_gain_pct": <number or null>, "evidence": "..."}],
 "hypotheses": [{"id": "H1", "change": "...", "expected_effect": "...", "risk": "..."}],
 "suggested_action": "tune_more|rewrite|stop"}
""" + dead


# --- 4. structure rewriter --------------------------------------------------------


@dataclass
class RewriterInputs:
    task: TaskSpec
    best_source: str  # best materialized source of the candidate
    report: BottleneckReport
    failed_hypotheses: list[dict]
    device: DeviceLimits
    n_candidates: int
    eval_semantics: dict | None = None
    # This box's measured ceilings, so the prompt can state what the card ACHIEVES and
    # not merely what it forbids. Optional: a box without a calibration still runs.
    calibration: object | None = None



class StructureRewriterAgent(AgentModule[RewriterInputs, RewriteResult]):
    name = "rewriter"
    output_model = RewriteResult

    def seed_sandbox(self, inputs: RewriterInputs, sb: Sandbox) -> None:
        sb.write_input("candidate/best.py", inputs.best_source)
        sb.write_input("analysis/bottleneck.json", inputs.report.model_dump_json(indent=2))
        sb.write_input(
            "history/failed_hypotheses.json", json.dumps(inputs.failed_hypotheses, indent=2)
        )
        sb.write_input("docs/candidate_contract.md", _contract_doc())
        sb.write_input("docs/triton_pitfalls.md", _triton_pitfalls_doc())
        sb.write_input("docs/device.md", _device_doc(inputs.device, inputs.calibration))
        sb.write_input("task/eval_semantics.md", _eval_semantics_doc(inputs.eval_semantics))

    def render_prompt(self, inputs: RewriterInputs, sb: Sandbox) -> str:
        return f"""`candidate/best.py` is the current best version of a kernel (already at
its best-known PARAMS). `analysis/bottleneck.json` explains what limits it —
which parameters wanted to go further and what resource blocked them.
`history/failed_hypotheses.json` lists changes already tried that did NOT help;
do not repeat them. Read `docs/candidate_contract.md`, `docs/device.md`, and
`task/eval_semantics.md` (the run mode the harness evaluates in). If
your rewrite uses Triton, also read `docs/triton_pitfalls.md` and obey it.

Produce up to {inputs.n_candidates} REWRITTEN kernel(s), each targeting a specific
hypothesis from the bottleneck report: change the structure so the blocked
parameter direction becomes reachable (less shared memory per element, fewer
registers, different work partitioning, etc.). This is a structural change, not a
parameter change — the new file may have different PARAMS keys.

If the bottleneck report blames `arithmetic_throughput` (or the latency floor is
flat across many resource profiles), the highest-value rewrite is to move the core
matmul/conv off the IEEE-fp32 scalar path onto the tensor cores: switch
`tl.dot(..., input_precision="ieee")` to `"tf32"`, or cast the dot inputs to
fp16/bf16 while keeping an fp32 accumulator. Read the "Precision and the tensor-core
path" section of the contract — the dual-precision gate accepts a tf32-matching
result, so this is a legal rewrite and is usually the only thing that moves an
arithmetic-throughput floor. Do NOT keep spending rewrites on register/shared-memory
relief when the report says the limiter is arithmetic throughput.

**A rewrite may also change BACKEND, and that is sometimes the only rewrite that can
work.** Declare it in the `backend` field. The bottleneck report tells you what limits
the kernel; it does not tell you which backend to use, because that inference is yours
to make. The one case measured on this hardware: when the task genuinely requires
**strict IEEE fp32** arithmetic, `tl.dot(..., input_precision="ieee")` has no fast path
here — a hand-written CUDA attention kernel reached 55–74% of this card's fp32 CUDA-core
roof where the best of 36 Triton tile configurations reached 18%, and no tile closed the
gap. So if the report says `arithmetic_throughput` AND the correctness gate has been
rejecting your tf32/fp16 attempts (i.e. the task really does need full fp32), a
`cuda` rewrite via `torch.utils.cpp_extension.load_inline` is the move — retrying tiles
inside Triton is not. The same applies if you need a warp primitive or a memory
instruction Triton does not expose (`__shfl_*`, a specific `cp.async` shape,
`__launch_bounds__`).

Be aware of the trade you are making, and say it in `change_summary`: a CUDA candidate
compiles in about a minute rather than seconds, and the harness reads less of a resource
profile from it (register/shared/spill figures come from the cubin, but `num_warps` and
pipelining depth are Triton launch properties that do not exist there). Do not switch
backend for style, for variety, or on a hunch — switch when you can name the thing
Triton cannot reach. CUTLASS/CuTe and TileLang are NOT installed; a candidate using them
fails to compile.

Write each rewrite to `rewrites/rw_1.py`, `rewrites/rw_2.py`, ... following the
contract (ModelNew + PARAMS dict). The rewrite does NOT need to be faster at the
old default parameters — it needs to unlock the blocked region (e.g. allow a
bigger tile that the parent could not compile/run).

THE OPTIMIZED PATH MUST BE THE PATH THAT ACTUALLY EXECUTES. The harness always
evaluates in the mode stated in `task/eval_semantics.md` — it never calls
`.eval()` or `.train()`. So if you write `if module.training: <fallback> else:
<your fast kernel>` and the mode is TRAIN, your kernel is dead code: the harness
measures the fallback, every trial times the same unoptimized path, and the
rewrite scores zero while looking correct. Do not guard your optimization behind
a mode check that the harness's mode does not select. In TRAIN mode BatchNorm
scale/shift depend on the current batch's mean/var, so they cannot be folded into
preceding weights — reduce the batch statistics first (a two-pass or
partial-reduction kernel) and fuse around that, or fuse something else.

Answer with JSON:
{{"candidates": [{{"file": "rewrites/rw_1.py", "backend": "triton", "hypothesis_id": "H1",
  "change_summary": "..."}}, ...]}}
"""

    def check_output(self, output: RewriteResult, sb: Sandbox) -> str | None:
        if not output.candidates:
            return "empty rewrite list; produce at least one rewrite"
        files = [c.file for c in output.candidates]
        missing = _files_exist_check(files, sb)
        if missing:
            return missing
        return _triton_lint_check(files, sb)

    def rescue_from_sandbox(self, sb: Sandbox) -> RewriteResult | None:
        """Rebuild the result from `rewrites/*.py` the agent already wrote.

        This module is where the loss was measured: on run-l1-42-20260907-193510 a rewriter
        wrote rewrites/rw_1.py (7152 bytes) at 21:47:00 and the attempt was killed by the read
        timeout at 21:50:20. The rewrite was finished and discarded, so its family recorded no
        improvement and was then declared `converged` without ever having evaluated a rewrite.

        Only the files are needed to proceed -- `hypothesis_id` and `change_summary` are
        narration the pipeline does not gate on -- so the artifact is self-describing enough to
        recover. The empty `change_summary` is deliberate and marked: a reader of the lineage
        must be able to tell a rescued candidate from one the agent described.

        `backend` is NOT narration and so is read from the source, the same way the generator
        and novelty rescues do it: a rescued CUDA rewrite defaulted to "triton" would be handed
        to the Triton loader and would fail as if the candidate were broken.
        """
        files = sb.list_outputs("rewrites")
        if not files:
            return None
        return RewriteResult(candidates=[
            RewriteCandidate(file=f, hypothesis_id="",
                             backend=_detect_backend(sb.read_output(f)),
                             change_summary="[recovered from sandbox after a transport "
                                            "failure; the agent's own summary never arrived]")
            for f in files
        ])

    def soft_check(self, output: RewriteResult, sb: Sandbox) -> list[str]:
        # Triton-specific WARNINGS only for files that are actually Triton. `check_output` still
        # lints every file for "has a kernel at all" (a hard contract violation either way);
        # this is the advisory pass, and running it over a CUDA rewrite would report absent
        # `tl.` idioms as defects. Read from the source rather than the declaration, since the
        # declaration is not what the loader will act on.
        triton_files = [c.file for c in output.candidates
                        if _detect_backend(sb.read_output(c.file)) == "triton"]
        return _triton_lint_warnings(triton_files, sb)


# --- 5. novelty generator -----------------------------------------------------------


@dataclass
class NoveltyInputs:
    task: TaskSpec
    ref_source: str
    family_summaries: list[dict]  # {family_id, approach_summary, best_ms, anchor_source}
    device: DeviceLimits
    n_candidates: int
    eval_semantics: dict | None = None
    # This box's measured ceilings, so the prompt can state what the card ACHIEVES and
    # not merely what it forbids. Optional: a box without a calibration still runs.
    calibration: object | None = None



class NoveltyGeneratorAgent(AgentModule[NoveltyInputs, NoveltyResult]):
    name = "novelty"
    output_model = NoveltyResult

    def seed_sandbox(self, inputs: NoveltyInputs, sb: Sandbox) -> None:
        sb.write_input("task/ref.py", inputs.ref_source)
        sb.write_input("docs/candidate_contract.md", _contract_doc())
        sb.write_input("docs/triton_pitfalls.md", _triton_pitfalls_doc())
        sb.write_input("docs/device.md", _device_doc(inputs.device, inputs.calibration))
        sb.write_input("task/eval_semantics.md", _eval_semantics_doc(inputs.eval_semantics))
        for i, fam in enumerate(inputs.family_summaries, 1):
            sb.write_input(f"families/family_{i}/anchor.py", fam.get("anchor_source", ""))
            sb.write_input(
                f"families/family_{i}/summary.json",
                json.dumps({k: v for k, v in fam.items() if k != "anchor_source"}, indent=2),
            )

    def render_prompt(self, inputs: NoveltyInputs, sb: Sandbox) -> str:
        return f"""We are optimizing the KernelBench task in `task/ref.py`. The approaches
tried so far are documented under `families/family_*/` (anchor source +
summary with measured performance). Read them, plus
`docs/candidate_contract.md`, `docs/device.md`, and `task/eval_semantics.md`
(the run mode the harness evaluates the reference in — your kernel must be
correct AND optimized for THAT mode, not a guessed one; the harness never calls
`.eval()`, so an optimization guarded behind a mode check the harness does not
select is dead code that will be timed as the fallback). If your candidate uses
Triton, also read `docs/triton_pitfalls.md` and obey it.

Produce up to {inputs.n_candidates} NEW candidate kernel(s) whose core computational
approach is CLEARLY DIFFERENT from every existing family — different work
decomposition, different data-flow strategy, different fusion structure, or a
different algorithmic formulation. A candidate that is a parameter tweak or a
minor variation of an existing family will be automatically rejected by a
structural-similarity gate, wasting the attempt.

Write each to `novel/nv_1.py`, `novel/nv_2.py`, ... following the contract
(ModelNew + PARAMS dict).

Answer with JSON:
{{"candidates": [{{"file": "novel/nv_1.py", "backend": "triton",
  "approach_summary": "...", "difference_claim": "how it differs from every
  existing family"}}, ...]}}
"""

    def check_output(self, output: NoveltyResult, sb: Sandbox) -> str | None:
        if not output.candidates:
            return "empty candidate list; produce at least one novel candidate"
        files = [c.file for c in output.candidates]
        missing = _files_exist_check(files, sb)
        if missing:
            return missing
        # See CandidateGeneratorAgent.check_output: the has-a-kernel rule is
        # backend-independent, so every produced file is checked.
        return _triton_lint_check([c.file for c in output.candidates], sb)

    def soft_check(self, output: NoveltyResult, sb: Sandbox) -> list[str]:
        triton_files = [c.file for c in output.candidates if c.backend == "triton"]
        return _triton_lint_warnings(triton_files, sb)

    def rescue_from_sandbox(self, sb: Sandbox) -> NoveltyResult | None:
        """Rebuild the result from `novel/*.py` the agent already wrote.

        Same loss the rewriter's rescue addresses, in the module that had no rescue. Measured on
        run-l1-42-20260908-023039: a novelty call started 03:24:14, wrote novel/nv_1.py (5907
        bytes) at 03:40:00, and the read timeout killed it at 03:49:15 -- a finished candidate
        discarded 9m15s after it was complete. That call then cost a second full attempt.

        `approach_summary` and `difference_claim` are narration: the novelty GATE is
        `structural_signature(source, backend)` plus a difflib similarity against every anchor
        (families.py:181-205), both computed from the file, so a rescued candidate faces exactly
        the same test as a described one. `backend` is not narration -- the signature hashes it --
        so it is read from the source instead of defaulted (see `_detect_backend`).
        """
        files = _rescued_files(sb, "novel")
        if not files:
            return None
        out = []
        for f in files:
            try:
                source = sb.read_output(f)
            except OSError:
                continue
            out.append(NoveltyCandidate(
                file=f, backend=_detect_backend(source),
                approach_summary="[recovered from sandbox after a transport failure; the "
                                 "agent's own summary never arrived]",
                difference_claim="[not stated: recovered from the sandbox. The structural "
                                 "signature and similarity gate still applied.]"))
        return NoveltyResult(candidates=out) if out else None


# --- 6. repair ------------------------------------------------------------------------


@dataclass
class RepairInputs:
    task: TaskSpec
    broken_source: str
    failure_kind: str
    failure_detail: str
    device: DeviceLimits
    eval_semantics: dict | None = None
    ref_source: str | None = None
    # Attribution only: lets AGENT_CALL_STARTED name the candidate this call is about,
    # so a transport timeout can be tied to a specific candidate and repeat-repair
    # attempt instead of guessed at by "nearest following REPAIR_PRODUCED".
    candidate_id: str | None = None
    # Rejected repairs of THIS candidate: [{"diagnosis", "failure_detail"}], oldest
    # first. Without it the agent cannot see that its previous fix was rejected, and
    # oscillates: on L3:48 cand-0137895f it first claimed the decay must be exp(-exp(A))
    # and then, after that was rejected, claimed the opposite -- reverting to a form
    # already known to fail. Strictly same-candidate history; nothing cross-candidate.
    prior_attempts: list[dict] | None = None


class RepairAgent(AgentModule[RepairInputs, RepairResult]):
    name = "repair"
    output_model = RepairResult

    def seed_sandbox(self, inputs: RepairInputs, sb: Sandbox) -> None:
        sb.write_input("candidate/broken.py", inputs.broken_source)
        sb.write_input("failure/kind.txt", inputs.failure_kind)
        sb.write_input("failure/detail.txt", inputs.failure_detail)
        sb.write_input("docs/candidate_contract.md", _contract_doc())
        sb.write_input("docs/triton_pitfalls.md", _triton_pitfalls_doc())
        sb.write_input("docs/device.md", _device_doc(inputs.device))
        sb.write_input("task/eval_semantics.md", _eval_semantics_doc(inputs.eval_semantics))
        # Ground truth for a correctness_mismatch. Without it the agent can only guess
        # at the reference's conventions from the broken kernel's own comments.
        if inputs.ref_source:
            sb.write_input("task/ref.py", inputs.ref_source)
        if inputs.prior_attempts:
            sb.write_input(
                "failure/rejected_repairs.md", _rejected_repairs_doc(inputs.prior_attempts)
            )

    def render_prompt(self, inputs: RepairInputs, sb: Sandbox) -> str:
        guidance = _repair_guidance(inputs.failure_kind)
        ref_line = (
            "`task/ref.py` is the REFERENCE implementation this kernel must match "
            "numerically -- read it and verify every convention you rely on (signs, "
            "parameterizations, transposes, accumulation order) against it rather than "
            "inferring them from the broken kernel.\n"
            if inputs.ref_source
            else ""
        )
        prior_line = (
            "`failure/rejected_repairs.md` lists YOUR earlier fixes to this same "
            "candidate and how each was rejected. Do NOT re-propose any of them, and do "
            "NOT simply invert a rejected claim -- if a previous diagnosis and its "
            "opposite were both rejected, the real cause is elsewhere.\n"
            if inputs.prior_attempts
            else ""
        )
        return f"""The kernel in `candidate/broken.py` failed with `{inputs.failure_kind}`.
The full failure detail is in `failure/detail.txt`. Read the contract in
`docs/candidate_contract.md`, `task/eval_semantics.md` (the reference's run mode —
train vs eval — which decides BatchNorm behavior), and if the kernel uses Triton
also read `docs/triton_pitfalls.md`.
{ref_line}{prior_line}
{guidance}

Diagnose the failure and write a fixed version to `candidate/fixed.py`. Keep the
computational approach the same — this is a repair, not a redesign. Preserve the
PARAMS structure (you may adjust default values or the dict's keys only if the
failure demands it).

Answer with JSON:
{{"file": "candidate/fixed.py", "diagnosis": "...", "change_summary": "..."}}
"""

    def check_output(self, output: RepairResult, sb: Sandbox) -> str | None:
        missing = _files_exist_check([output.file], sb)
        if missing:
            return missing
        return _triton_lint_check([output.file], sb)
