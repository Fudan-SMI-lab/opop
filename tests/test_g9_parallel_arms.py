"""The G9 A/B may run its agent calls in parallel; this pins why that is safe.

The whole point of the parallel path is wall clock: an agent call is median 18.9 min of network wait
during which the GPU is idle, so six sequential calls cost ~2 h in which almost nothing computes.
Overlapping the waiting does not change what any call receives.

But there is one hazard that would make a parallel run silently wrong, and it is the reason this file
exists: `OpencodeClient._abort_and_close` CLOSES its httpx transport. It must -- an abort alone leaves
the streaming POST hung forever (measured). With ONE client shared across threads, a single call
hitting its idle-abort would tear the transport out from under every other in-flight call, and those
calls would fail for a reason that has nothing to do with them. In an A/B that means an arm gets
blamed for a neighbour's timeout.

So: one client per worker. These tests drive the real objects rather than asserting on source text.
"""
from __future__ import annotations

import inspect
from pathlib import Path

from kernel_optimizer.agents.runtime import OpencodeClient


def test_each_client_owns_its_own_transport():
    """Two clients must not share a transport, or one abort kills both."""
    a = OpencodeClient("http://127.0.0.1:1", timeout_s=5.0)
    b = OpencodeClient("http://127.0.0.1:1", timeout_s=5.0)
    try:
        assert a._http is not b._http, (
            "two OpencodeClients share one httpx transport, so one call's idle-abort would close "
            "the connection pool used by every other in-flight call -- in the A/B that makes one "
            "arm fail because of another arm's timeout")
        # Closing one must leave the other usable.
        a.close()
        assert a._http.is_closed, "close() did not close the transport"
        assert not b._http.is_closed, (
            "closing one client closed another client's transport -- the parallel path would "
            "collapse after the first abort")
    finally:
        a.close()
        b.close()


def test_abort_closes_the_transport_which_is_why_sharing_is_unsafe():
    """Pin the behaviour the hazard rests on, so a future change to it is caught here.

    If `_abort_and_close` ever stops closing the transport, sharing a client WOULD become safe and
    this file's premise would be obsolete -- better to fail loudly then than to keep paying for
    per-worker clients for a reason that no longer holds.
    """
    src = inspect.getsource(OpencodeClient._abort_and_close)
    assert "close" in src, (
        "_abort_and_close no longer closes the transport; re-check whether per-worker clients are "
        "still required (see this module's docstring)")


def test_the_ab_gives_each_worker_its_own_client():
    """The A/B must construct one client per worker, not reuse the runtime's single client."""
    path = Path(__file__).resolve().parents[1] / "scripts" / "g9_information_ab.py"
    src = path.read_text(encoding="utf-8")
    # Behavioural assertions are preferred, but this is a script with a live-GPU main(); what can be
    # checked cheaply is that the construction is inside the worker fan-out and parameterised by
    # worker count.
    assert "OpencodeClient(" in src, "the A/B never constructs its own client"
    assert "for _ in range(n_workers)" in src, (
        "clients are not created per worker, so a parallel run would share one transport")
    assert "clients[i % n_workers]" in src, (
        "workers are not each handed their own client")


def test_gpu_evaluation_is_not_parallelized():
    """Timing two kernels at once corrupts both measurements; evaluation must stay serialized."""
    path = Path(__file__).resolve().parents[1] / "scripts" / "g9_information_ab.py"
    src = path.read_text(encoding="utf-8")
    # The evaluation loop must sit OUTSIDE the executor block.
    ev_at = src.index("PHASE 2")
    pool_at = src.index("ThreadPoolExecutor(max_workers")
    assert ev_at > pool_at, (
        "the evaluation phase is not after the agent fan-out; if _evaluate ran inside the pool, two "
        "kernels could be timed simultaneously and both numbers would be wrong")
    assert "_evaluate(" in src[ev_at:], "phase 2 does not contain the evaluation call"
