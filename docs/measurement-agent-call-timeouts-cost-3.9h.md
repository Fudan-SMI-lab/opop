# Measurement: agent-call ReadTimeouts have cost 3.9h, and the timeout is 2.3× the slowest working call

Recorded because `run-l3-21-20260905-195615`'s first rewrite round lost 20 minutes to one, and
the pattern turned out to be the project's largest single source of wasted wall clock outside
GPU work.

## The cost

`python scripts/audit_agent_call_timeouts.py` over every run on disk:

```
failed attempts: 16   of which ReadTimeout: 12
wall lost to failed attempts: 233.7 min (3.9 h)
```

Distributed unevenly by module — `repair` accounts for 8 of the 12:

```
repair         x8    generator x1
rewriter       x2    (parameterizer's 4 failures are schema/502, not timeouts)
```

`opencode.request_timeout_s` is 1200 s. Each attempt gets a fresh timer, which is correct, but
means one call can burn 40 min across two attempts and still fail — observed twice
(`run-l3-21-20260903-210650` repair-0a74aa71, `run-l3-43-20260904-093730` repair-baff126a; the
latter succeeded on attempt 3 after 40 min lost).

## The measurement error I made first, since it changes the conclusion

My first pass timed each call from its **first** `AGENT_CALL_STARTED` to its
`AGENT_CALL_FINISHED`. But the finish event carries `attempt: 3`, so that span includes the two
failed attempts before it. That produced a "successful call took 59.5 min" row and a headline of
313.7 min lost, both wrong. Splitting each call at its intermediate `FAILED` events:

```
                  per-CALL (wrong)      per-ATTEMPT (right)
repair max            59.5 min               19.4 min
generator max         25.1 min                9.6 min
rewriter max          20.4 min                6.6 min
all p99               20.4 min                8.7 min
lost total           313.7 min              233.7 min
```

The corrected figures matter because the per-call numbers make 20 min look like a *necessary*
timeout (there is a 20.4-min success!) when no single attempt has ever taken longer than 19.4,
and only one has exceeded 10.

## Successful attempts, per attempt

```
module              n   median      p90      p99      max
analyst           162      1.3      2.2      4.1      6.6
parameterizer     301      1.7      2.8      4.4      6.5
rewriter           59      3.0      4.8      6.6      6.6
repair             90      2.3      5.6     19.4     19.4
generator          22      5.8      8.9      9.6      9.6
all               635      1.8              8.7     19.4
```

So the current 1200 s is **2.3× the slowest attempt that has ever worked**, and 6.7× the p99.

## The trade, stated as a trade

```
timeout 600s (10 min): would kill 1 of 635 successful attempts, save 110 min on the hangs
timeout 720s (12 min): would kill 1 of 635 successful attempts, save  88 min on the hangs
timeout 900s (15 min): would kill 1 of 635 successful attempts, save  55 min on the hangs
```

The single casualty at every threshold is the same 19.4-min `repair` success.

**What this cannot settle, and why I am not changing the value.** A killed slow-but-working call
is not free: it costs its own elapsed time *and* a retry, so the "saved" column is an upper
bound. More importantly the 12 timeouts are all-or-nothing hangs (the session's token counts
freeze and never move again — checked live on the current one via the opencode session endpoint:
`input 22847, output 792` identical 45 s apart, 0 files written). A hang costs the full timeout
whatever the timeout is; shortening it reduces the constant, it does not address the cause.

The cause is not diagnosed here. It could be the provider, the opencode server, or the
1.18.18/1.17.13 SDK version skew already noted in the plan. Retrying works — every hung call
eventually succeeded on attempt 2 or 3, and `AGENT_SESSION_RESET` (which starts a genuinely new
session rather than reusing the wedged one) is what makes that work.

So this note records the cost and the distribution. A timeout change is a config value with a
measured trade behind it, which is a decision for the run owner, not something to slip in
mid-experiment.

## One thing worth fixing regardless

`repair` carries 8 of 12 timeouts on 90 attempts (8.9%) against `rewriter`'s 2 on 59 (3.4%) and
`parameterizer`'s 0 on 301. Whether that is prompt size, prompt content, or chance at n=90 is not
established — but if a cause exists it is module-specific, and `repair` is where to look.

## Update: `parameterizer` then timed out too, and three hypotheses all fail

The paragraph above pointed at `repair` as module-specific. Within hours `parameterizer` timed out
on `run-l3-21-20260905-195615` — after **306 clean attempts** across every prior run. So whatever
causes this is not a property of one module's prompt. Cost is now **253.7 min (4.2 h)** over 13
ReadTimeouts, and this single run has lost 40 min of its own.

I tested the three explanations that were available. None of them stands:

**1. Module-specific cause — refuted.** `parameterizer`'s first timeout in 306 attempts is the
counter-example. Its prompt did not change.

**2. Time-clustered endpoint trouble — weak.** If the endpoint has bad windows and whichever
module calls during one takes the hit, timeouts should cluster. They barely do
(`scripts/audit_timeout_clustering.py`, 90-minute windows):

```
clusters: 7 for 12 timeouts, and 5 of the 7 are singletons
  09-03 21:11  n=3  span 85.3 min  generator, repair
  09-04 10:49  n=2  span 20.0 min  repair
  09-05 02:05  n=2  span 70.8 min  repair
  09-05 20:51  n=2  span 71.0 min  parameterizer, rewriter
gaps between timeouts: 20, 65, 71, 85, 330, 366, 382, 402, 674, 895 min
```

Two clusters do mix modules, which is what this hypothesis predicts, and the tightest gaps are
exactly 20.0 min — a retry hitting the wall immediately after its parent. But most timeouts are
isolated.

**3. In-window call share — refuted.** If clustering explained the module split, each module's
timeout share should track its share of calls *made during those windows*. It does not:

```
module            all calls   share  in-window   share  timeouts   share
analyst                 166   25.0%         57   25.1%         0    0.0%
parameterizer           311   46.9%        106   46.7%         1    8.3%
repair                   99   14.9%         37   16.3%         8   66.7%
rewriter                 63    9.5%         18    7.9%         2   16.7%
```

`repair` makes 16.3% of in-window calls and takes 66.7% of the timeouts; `parameterizer` makes
46.7% and takes 8.3%. The windows do not explain it.

**What does correlate: prompt size.** Summing each sandbox's non-`.git` bytes, the failure rate
rises monotonically with median prompt size across all five modules:

```
module            calls  timeouts    rate  median KB
analyst             166         0   0.00%       13.2
parameterizer       311         1   0.32%       23.3
generator            23         1   4.35%       30.1
rewriter             63         2   3.17%       30.2
repair               99         8   8.08%       31.9
```

Monotone in five points is suggestive, and it fits the mechanism (a larger prompt is a longer
request, more of it exposed to whatever drops). But it does not explain the *magnitude*: the three
largest modules sit within 6% of each other in size while their rates span 3.17–8.08%, and
`generator`'s 4.35% is one timeout in 23 calls. At n=13 total none of these splits is
statistically separable.

**Conclusion, stated as a null result.** The cause is not identified. It is not one module, not
obviously the clock, and prompt size correlates without accounting for the spread. What is
established is the cost (4.2 h), that every occurrence recovered on retry, and that
`AGENT_SESSION_RESET` is what makes the retry work. The earlier "look at `repair`" pointer should
be treated as retracted — it was a plausible reading of 12 events that the 13th contradicted.

Reproduce with `python scripts/audit_agent_call_timeouts.py`.
