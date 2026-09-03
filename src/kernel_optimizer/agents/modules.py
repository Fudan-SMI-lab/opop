"""The six concrete agent modules: generator, parameterizer, analyst, rewriter,
novelty, repair. Each is thin: sandbox seeding + prompt + output schema."""

from __future__ import annotations

import json
from dataclasses import dataclass
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
from kernel_optimizer.paramspace.triton_lint import lint_triton_source


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
    mapping = {
        "correctness_mismatch": numeric,
        "compile_error": compile_,
        "runtime_error": compile_,
        "static_check_failed": compile_,
        "oom": oom,
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
    no-op for non-Triton files (no @triton.jit body -> no findings)."""
    problems: list[str] = []
    for f in files:
        try:
            src = sb.read_output(f)
        except (OSError, ValueError):
            continue  # existence is checked separately; don't double-fault here
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
        triton_files = [c.file for c in output.candidates if c.backend == "triton"]
        return _triton_lint_check(triton_files, sb)

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
    # Improvement K: when set, a focused space-EXPANSION request rather than a
    # fresh parameterization. Describes which knobs hit the tried-range boundary
    # (with direction) so the agent extends only those choices, structure unchanged.
    expand_directive: str = ""


class ParameterizerAgent(AgentModule[ParameterizerInputs, ParameterizationResult]):
    name = "parameterizer"
    output_model = ParameterizationResult

    def seed_sandbox(self, inputs: ParameterizerInputs, sb: Sandbox) -> None:
        sb.write_input("candidate/source.py", inputs.candidate_source)
        sb.write_input("docs/candidate_contract.md", _contract_doc())
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
`docs/candidate_contract.md` and `docs/device.md`.{feedback}

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
   GRAMMAR LIMIT: a constraint is evaluated by a restricted parser that allows ONLY
   names, numeric/string literals, `+ - * / // % **`, comparisons, `and`/`or`/`not`.
   Conditional expressions (`A if C else B`), function calls (`min()`, `max()`, `abs()`),
   indexing, and comprehensions are REJECTED. Express a conditional rule as a
   disjunction instead — e.g. instead of
   `(4 if DTYPE == "fp16" else 2) * BLOCK_M <= X`, write
   `(DTYPE != "fp16" and 2 * BLOCK_M <= X) or (DTYPE == "fp16" and 4 * BLOCK_M <= X)`.

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
        return f"""A candidate kernel is in `candidate/source.py` (already parameterized
with a `PARAMS` dict). Read it plus `docs/candidate_contract.md` and `docs/device.md`.

During tuning, some knobs reached the EDGE of the value range that was offered and
latency was still improving toward that edge, while hardware resources still had
headroom. Your job is a FOCUSED EXPANSION, not a redesign:

{inputs.expand_directive}

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
  names, numeric/string literals, `+ - * / // % **`, comparisons, and `and`/`or`/`not`.
  Conditional expressions (`A if C else B`), function calls (`min`/`max`/`abs`), and
  indexing are REJECTED — express conditional rules as a disjunction of `and` clauses.
- Do NOT introduce new knobs or remove existing ones.
- **DECLARE EVERY KNOB**: `space.params` MUST list ALL keys of the `PARAMS` dict —
  not just the ones you expanded. The unexpanded knobs are repeated verbatim with
  their existing choices. A response whose declared names differ from the PARAMS keys
  is rejected outright (`key_mismatch`) and wastes the attempt.

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


class BottleneckAnalystAgent(AgentModule[AnalystInputs, BottleneckReport]):
    name = "analyst"
    output_model = BottleneckReport

    def seed_sandbox(self, inputs: AnalystInputs, sb: Sandbox) -> None:
        sb.write_input("candidate/source.py", inputs.candidate_source)
        sb.write_input("tuning/stats.json", inputs.stats.model_dump_json(indent=2))
        sb.write_input("tuning/trials.csv", inputs.trials_csv)
        sb.write_input("docs/device.md", _device_doc(inputs.device))

    def render_prompt(self, inputs: AnalystInputs, sb: Sandbox) -> str:
        return """A kernel candidate was tuned over its parameter space. These four
files already exist in your working directory — read them with your file tools
before answering; do NOT assume any are missing (a stale index may hide them,
so read by path):
- `candidate/source.py` — the kernel (PARAMS dict = tunable knobs)
- `tuning/stats.json` — per-parameter statistics: best value, whether the optimum
  sits at a boundary of the tried range (`at_boundary` + direction), effect size,
  failure rates per value, resource usage (registers/shared memory/spills) at the
  best config, and failure clusters
- `tuning/trials.csv` — the full trial log (params, status, latency, resources)
- `docs/device.md` — hardware limits

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

Answer with JSON matching:
{"summary": "...",
 "parameter_limits": [{"param": "...", "headroom_direction": "increase|decrease",
   "blocked_by": "registers|shared_memory|threads|oom|compile_failure|arithmetic_throughput|none",
   "predicted_gain_pct": <number or null>, "evidence": "..."}],
 "hypotheses": [{"id": "H1", "change": "...", "expected_effect": "...", "risk": "..."}],
 "suggested_action": "tune_more|rewrite|stop"}
"""


# --- 4. structure rewriter --------------------------------------------------------


@dataclass
class RewriterInputs:
    task: TaskSpec
    best_source: str  # best materialized source of the candidate
    report: BottleneckReport
    failed_hypotheses: list[dict]
    device: DeviceLimits
    n_candidates: int


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

    def render_prompt(self, inputs: RewriterInputs, sb: Sandbox) -> str:
        return f"""`candidate/best.py` is the current best version of a kernel (already at
its best-known PARAMS). `analysis/bottleneck.json` explains what limits it —
which parameters wanted to go further and what resource blocked them.
`history/failed_hypotheses.json` lists changes already tried that did NOT help;
do not repeat them. Read `docs/candidate_contract.md` and `docs/device.md`. If
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


class NoveltyGeneratorAgent(AgentModule[NoveltyInputs, NoveltyResult]):
    name = "novelty"
    output_model = NoveltyResult

    def seed_sandbox(self, inputs: NoveltyInputs, sb: Sandbox) -> None:
        sb.write_input("task/ref.py", inputs.ref_source)
        sb.write_input("docs/candidate_contract.md", _contract_doc())
        sb.write_input("docs/triton_pitfalls.md", _triton_pitfalls_doc())
        sb.write_input("docs/device.md", _device_doc(inputs.device))
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
`docs/candidate_contract.md` and `docs/device.md`. If your candidate uses Triton,
also read `docs/triton_pitfalls.md` and obey it.

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
        triton_files = [c.file for c in output.candidates if c.backend == "triton"]
        return _triton_lint_check(triton_files, sb)

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

    def render_prompt(self, inputs: RepairInputs, sb: Sandbox) -> str:
        guidance = _repair_guidance(inputs.failure_kind)
        return f"""The kernel in `candidate/broken.py` failed with `{inputs.failure_kind}`.
The full failure detail is in `failure/detail.txt`. Read the contract in
`docs/candidate_contract.md`, `task/eval_semantics.md` (the reference's run mode —
train vs eval — which decides BatchNorm behavior), and if the kernel uses Triton
also read `docs/triton_pitfalls.md`.

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
