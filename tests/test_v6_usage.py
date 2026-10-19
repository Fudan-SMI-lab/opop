from kernel_optimizer.store.usage import UsageLedger


def message(ident, cost=None, finished=True, **tokens):
    return {"info": {"id": ident, "role": "assistant", "cost": cost,
                     "finish": "stop" if finished else None, "tokens": tokens}}


def test_session_snapshots_resume_without_double_counting(tmp_path):
    ledger = UsageLedger(tmp_path / "usage.sqlite")
    ledger.observe("s", [message("a", 0.2, False, input=10)])
    ledger.observe("s", [message("a", 0.3, input=10, output=20, cache={"read": 30}), message("b", 0.4)])
    ledger = UsageLedger(ledger.path)
    result = ledger.observe("s", [message("a", 0.3, input=10, output=20, cache={"read": 30}), message("b", 0.4)])
    assert result["messages"] == 2 and result["reported_cost"] == 0.7
    assert result["tokens"] == {"input": 10, "output": 20, "cache.read": 30}
    ledger.observe("s", [message("a", 0, False, input=1)])
    assert ledger.summary()["reported_cost"] == 0.7


def test_missing_cost_and_observation_failure_are_not_zero_cost(tmp_path):
    ledger = UsageLedger(tmp_path / "usage.sqlite")
    result = ledger.observe("s", [message("a")])
    assert result["reported_cost"] is None and result["messages_without_cost"] == 1
    ledger.unavailable("s", "ReadTimeout")
    assert ledger.summary()["sessions"][0]["status"] == "unavailable"
    assert ledger.summary()["messages"] == 1


def test_session_identity_and_token_components_are_independent(tmp_path):
    ledger = UsageLedger(tmp_path / "usage.sqlite")
    ledger.observe("s1", [message("a", 0, total=50, input=20, output=30)])
    ledger.observe("s2", [message("a", 0.5, total=7, output=7)])
    result = ledger.summary()
    assert result["messages"] == 2 and result["tokens"]["total"] == 57
    assert result["reported_cost"] == 0.5
