"""Run store tests: event append/replay round-trip, artifacts, step idempotency."""

from kernel_optimizer.store.run_store import RunStore


def make_store(tmp_path):
    return RunStore.create(tmp_path, "run-test", {"task": "t"})


def test_create_and_reopen(tmp_path):
    store = make_store(tmp_path)
    store.append("STEP_DONE", {"step_key": "a"})
    reopened = RunStore.open(store.run_dir)
    events = reopened.iter_events()
    assert [e.type for e in events] == ["RUN_CREATED", "STEP_DONE"]
    assert events[0].seq == 0 and events[1].seq == 1


def test_append_seq_continues_after_reopen(tmp_path):
    store = make_store(tmp_path)
    store.append("STEP_DONE", {"step_key": "a"})
    reopened = RunStore.open(store.run_dir)
    ev = reopened.append("STEP_DONE", {"step_key": "b"})
    assert ev.seq == 2


def test_replay_steps_and_trials(tmp_path):
    store = make_store(tmp_path)
    store.append("STEP_DONE", {"step_key": "baseline"})
    store.append("CANDIDATE_REGISTERED", {"candidate": {"candidate_id": "c1"}})
    store.append("SPACE_PUBLISHED", {"space": {"space_id": "s1", "candidate_id": "c1"}})
    store.append("TRIAL_DONE", {"trial": {"trial_id": "t1", "space_id": "s1"}})
    store.append("TRIAL_DONE", {"trial": {"trial_id": "t2", "space_id": "s1"}})
    state = store.replay()
    assert "baseline" in state.steps_done
    assert "c1" in state.candidates
    assert "s1" in state.spaces
    assert [t["trial_id"] for t in state.trials["s1"]] == ["t1", "t2"]
    assert not state.finished


def test_replay_finished(tmp_path):
    store = make_store(tmp_path)
    store.append("RUN_FINISHED", {"summary": {}})
    assert store.replay().finished


def test_artifacts_content_addressed(tmp_path):
    store = make_store(tmp_path)
    ref1 = store.put_artifact("hello", "greeting")
    ref2 = store.put_artifact(b"hello", "greeting-again")
    assert ref1 == ref2
    assert store.get_artifact(ref1) == b"hello"


def test_candidate_dir(tmp_path):
    store = make_store(tmp_path)
    d = store.candidate_dir("cand-x")
    assert d.is_dir()
    assert d.name == "cand-x"


def test_create_refuses_existing(tmp_path):
    make_store(tmp_path)
    try:
        RunStore.create(tmp_path, "run-test", {})
        raise AssertionError("expected FileExistsError")
    except FileExistsError:
        pass
