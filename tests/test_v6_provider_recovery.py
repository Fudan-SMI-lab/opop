from datetime import datetime, timedelta, timezone
import json
from types import SimpleNamespace

from kernel_optimizer.agents.runtime import provider_failure
from kernel_optimizer.project.provider_queue import ProviderQueue, retry_deadline
from kernel_optimizer.runtime.execution import write_json


def test_actual_five_hour_provider_limit_is_not_a_short_concurrency_retry():
    # Sanitized fields from the actual box4-2 response. No credentials/headers dump.
    error = {"data": {"statusCode": 429, "message":
        "已达到 5 小时使用上限，2026-10-20 12:12:44 后可继续使用。",
        "responseHeaders": {"Date": "Tue, 20 Oct 2026 01:19:46 GMT", "Set-Cookie": "not-retained"},
        "responseBody": json.dumps({"error": {"code": "1308"}})}}
    facts = provider_failure(error)
    now = datetime(2026, 10, 20, 1, 35, tzinfo=timezone.utc).timestamp()
    advice = retry_deadline(facts, now=now, local_timezone=timezone(timedelta(hours=8)))
    assert advice["not_before"] == datetime(2026, 10, 20, 4, 12, 44, tzinfo=timezone.utc).timestamp()
    assert advice["timezone_assumption"]
    assert "not-retained" not in json.dumps(facts)


def test_standard_retry_after_precedes_text_and_unrelated_dates_do_not_schedule():
    now = 1792450000.0
    assert retry_deadline({"retry_after": "120"}, now=now)["not_before"] == now+120
    assert retry_deadline({"message": "request created at 2026-10-21 12:00:00"}, now=now) is None
    deadline = retry_deadline({"retry_after": "Tue, 20 Oct 2026 04:12:44 GMT"}, now=now)
    assert deadline["not_before"] == datetime(2026, 10, 20, 4, 12, 44, tzinfo=timezone.utc).timestamp()


def test_provider_declared_window_does_not_reduce_available_parallelism(tmp_path):
    config = SimpleNamespace(agents=SimpleNamespace(default_model="provider/model"),
                             opencode=SimpleNamespace(launch_cwd=tmp_path))
    queue = ProviderQueue(config)
    write_json(queue.state_path, {"capacity": 8, "healthy": 0, "failures": 0})
    receipt = queue.overloaded(429, {"retry_after": "9000"})
    assert receipt["temporary_parallel_capacity"] is None
    assert queue._read()["capacity"] == 8
    assert queue._read()["admission"] == "provider_availability"
