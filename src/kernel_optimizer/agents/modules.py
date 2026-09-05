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
    GenerationResult,
    NoveltyResult,
    ParameterizationResult,
    RepairResult,
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


def _device_doc(device: DeviceLimits) -> str:
    return (
        f"# Target device\n\n"
        f"- {device.name}\n"
        f"- VRAM: {device.vram_gb} GB\n"
        f"- Max registers/thread: {device.max_regs_per_thread}\n"
        f"- Max static shared memory/block: {device.max_shared_bytes_static} B\n"
        f"- Max opt-in shared memory/block: {device.max_shared_bytes_optin} B\n"
        f"- Max threads/block: {device.max_threads_per_block}\n"
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


class CandidateGeneratorAgent(AgentModule[GeneratorInputs, GenerationResult]):
    name = "generator"
    output_model = GenerationResult

    def seed_sandbox(self, inputs: GeneratorInputs, sb: Sandbox) -> None:
        sb.write_input("task/ref.py", inputs.ref_source)
        sb.write_input("docs/candidate_contract.md", _contract_doc())
        sb.write_input("docs/triton_pitfalls.md", _triton_pitfalls_doc())
        sb.write_input("docs/device.md", _device_doc(inputs.device))
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
~2x faster and is what torch.compile uses. The dual-precision correctness gate
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
        sb.write_input("docs/device.md", _device_doc(inputs.device))

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
   Example: "BLOCK_M * BLOCK_K * 4 + BLOCK_K * BLOCK_N * 4 <= MAX_SHARED_BYTES".
   That example is a single-kernel fp32 matmul. Do NOT copy its shape — three rules
   decide whether a shared-memory constraint actually prevents anything:
   (a) ONE CONSTRAINT PER LAUNCHED KERNEL. If the file launches several kernels, each
       has its own tile knobs and its own `num_stages`, and each needs its own
       shared-memory and thread bounds over ITS OWN knobs. A kernel with no bound is a
       kernel whose configurations will be sampled and will die at launch with
       `OutOfResources: shared memory`. An unbounded kernel is the single most common
       cause of wasted trials — do not leave one out.
   (b) THE BYTE WIDTH FOLLOWS THE PRECISION KNOB. If you expose COMPUTE_DTYPE (step 1),
       then fp16/bf16 stage 2 bytes per element and tf32/ieee stage 4. A hard-coded `* 4`
       is wrong twice over: it lets 4-byte configs through on the kernels it does cover,
       and it forbids 2-byte configs that would have fit. Write the bound as a
       disjunction over the precision knob, using the grammar below. The fp32 accumulator
       is always 4 bytes regardless — only the staged operands change width.
   (c) DERIVE THE ARITHMETIC FROM THIS KERNEL'S BODY. Count what is actually resident
       per stage: a matmul stages two operand tiles; a flash-attention kernel stages K
       and V tiles and holds an accumulator across the loop; a reduction may stage one
       tile and a running statistic. Read the `tl.load`/`tl.dot` shapes in the source and
       add up the bytes those imply. A plausible-looking formula copied from elsewhere is
       worse than none, because it reads as protection while admitting the same failures.
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
   Putting (b) and the grammar together, a per-kernel dtype-aware shared-memory bound
   has this SHAPE (replace <K> with the kernel's own knob prefix, and
   <elements staged per stage> with the count you derived from the kernel body per (c)):
     (COMPUTE_DTYPE == "fp16" or COMPUTE_DTYPE == "bf16") and
       <K>_NUM_STAGES * <elements staged per stage> * 2 <= MAX_SHARED_BYTES_OPTIN
      or (COMPUTE_DTYPE == "tf32" or COMPUTE_DTYPE == "ieee") and
       <K>_NUM_STAGES * <elements staged per stage> * 4 <= MAX_SHARED_BYTES_OPTIN
   Emit one such constraint per launched kernel. Use MAX_SHARED_BYTES_OPTIN (not
   MAX_SHARED_BYTES_STATIC) — Triton opts into the larger limit automatically.

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
        return _files_exist_check([output.file], sb)


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


class BottleneckAnalystAgent(AgentModule[AnalystInputs, BottleneckReport]):
    name = "analyst"
    output_model = BottleneckReport

    def seed_sandbox(self, inputs: AnalystInputs, sb: Sandbox) -> None:
        sb.write_input("candidate/source.py", inputs.candidate_source)
        sb.write_input("tuning/stats.json", inputs.stats.model_dump_json(indent=2))
        sb.write_input("tuning/trials.csv", inputs.trials_csv)
        sb.write_input("docs/device.md", _device_doc(inputs.device))
        sb.write_input("task/eval_semantics.md", _eval_semantics_doc(inputs.eval_semantics))
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
        return """A kernel candidate was tuned over its parameter space. These five
files already exist in your working directory — read them with your file tools
before answering; do NOT assume any are missing (a stale index may hide them,
so read by path):
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
   core path can be ~2x faster on matmul/conv-bound ops (this is how torch.compile
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
        sb.write_input("docs/device.md", _device_doc(inputs.device))
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
{{"candidates": [{{"file": "rewrites/rw_1.py", "hypothesis_id": "H1",
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

    def soft_check(self, output: RewriteResult, sb: Sandbox) -> list[str]:
        return _triton_lint_warnings([c.file for c in output.candidates], sb)


# --- 5. novelty generator -----------------------------------------------------------


@dataclass
class NoveltyInputs:
    task: TaskSpec
    ref_source: str
    family_summaries: list[dict]  # {family_id, approach_summary, best_ms, anchor_source}
    device: DeviceLimits
    n_candidates: int
    eval_semantics: dict | None = None


class NoveltyGeneratorAgent(AgentModule[NoveltyInputs, NoveltyResult]):
    name = "novelty"
    output_model = NoveltyResult

    def seed_sandbox(self, inputs: NoveltyInputs, sb: Sandbox) -> None:
        sb.write_input("task/ref.py", inputs.ref_source)
        sb.write_input("docs/candidate_contract.md", _contract_doc())
        sb.write_input("docs/triton_pitfalls.md", _triton_pitfalls_doc())
        sb.write_input("docs/device.md", _device_doc(inputs.device))
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
