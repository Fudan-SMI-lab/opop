# T1 model-facing anchors

Captured after protocol archive `54ca2a98c6a34fd5a4727d518e1a253ecf53c43e`, before any production patch.

| Anchor | Actual archived source | Purpose |
|---|---|---|
| `cda1130-direct.json` | `cda113070c8ba3a91f480f42b68d196187796cda` | Declared two-field direct-C2 model-facing compatibility target, not a relabeling of earlier runs |
| `cda1130-legacy.json` | same CDA source | Legacy StructureRewriter comparison |
| `8526a13-generic.json` | `8526a1390c78346af1a5be112685b7fbe4defa59` | Existing no-selector generic default, including bundle defaults |
| `8526a13-legacy.json` | same 8526 source | Unchanged legacy StructureRewriter contract |
| `8526a13-bundle.json` | same 8526 source | Current shared bundle behavior, not future model_operator behavior |

`tests/c2_contract_capture.py` supplies identical finite source/reference/TaskSpec/params/cases/history across pins. Real factories, AgentModule.invoke, render/seed and legacy self-test context execute. RecordingProvider captures the exact schema supplied to `prompt` and stops before any external provider call. Sandbox Git initialization is suppressed. No candidate code executes in these captures.

The only fixture configuration overrides are `/fixture/venv` and `/fixture/kernelbench/src`, to remove user-environment paths. Effective model/retry/seed/budget defaults are not overridden. Legacy uses the real formal-helper context, including its appended prompt and generated self-test files.

Normalizations replace only concrete sandbox/session roots, fixture roots, source roots and their exact Windows/JSON-escaped/WSL aliases. The exact shell-quoted PYTHONPATH argument is mapped as one argument because temporary source roots contain spaces; both its lexical quotes and path belong to this location mapping. Full mapping tables are recorded beside each RAW request. No schema titles, field names, description values, candidate contents, objective values or semantic defaults are removed.

Complete effective prompt and seeded prose are reviewable evidence. Tests compare schema, structured seeds, source files, file inventory, parser behavior and routes; they intentionally do not pin prose sentences or prompt snapshots. `comparisons.json` in RAW records whole-request and effective-prompt comparisons separately.

Tests load these small JSON artifacts directly and require neither Git nor the temporary archived source trees. Captures used Python 3.12.7, Pydantic 2.13.5 and Optuna 4.9.0, not a claim to reproduce a historical dependency environment or stochastic LLM output. Exact historical replay requires the experiment's actual archived SHA/environment.

The export utility is opt-in: `python -B -m tests.c2_capture_export --label <cda1130|8526a13|worktree> --output <RAW>`. `--write-fixtures` is required to overwrite anchors. For historical export, put the corresponding read-only archived `src` FIRST on PYTHONPATH and the current test-support root second. Normal test runs never regenerate anchors.

T2 changes the direct route test to the CDA anchor only after introducing explicit `c2_direct_compat`; generic remains on 8526. Dedicated `model_operator` expectations require new RED tests, not rewriting these anchors. The immutable-parent validation fix remains separately versioned behavior.
