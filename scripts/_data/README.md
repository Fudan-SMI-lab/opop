# Pre-extracted classification rows (evidence for the multi-binding recount)

Each file is one row per `BOTTLENECK_CLASSIFIED` candidate: the classifier's own `evidence`
dict plus that candidate's winning-trial `profile`. Extracted from `events.jsonl` on the box
that produced the run, because the run directories live on the two Linux boxes and the full
event logs are far too large to copy.

    rows_box1.json   run-l3-21-20260908-232211   box 1 (RTX 4090)   10 candidates
    rows_box2.json   run-l3-21-20260909-154359   box 2 (RTX 4090)   10 candidates
                     run-l3-43-20260909-015247                      13 candidates
                     run-l3-43-20260908-053708                      18 candidates

Consumed by `../multibind_latency_independent.py`, which also accepts run directories
directly (`--dump` regenerates these files). Committed so the 72.5% / 25.5% multi-binding
rates can be re-derived without access to either box.

Both fields are needed per candidate: `shared_bytes` and `n_spills` are missing from the
evidence dict on some runs but present in `TRIAL_DONE.profile`, and `occupancy` is the other
way round. Reading one source only reports the missing dimension as slack.
