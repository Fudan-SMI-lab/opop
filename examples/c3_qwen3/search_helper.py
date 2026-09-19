"""One loopback request at a time, borrowing the live search session; never loads a model."""

import argparse
import json
import shlex
import socket
import socketserver
import sys
from pathlib import Path
from threading import Thread
from types import TracebackType
from typing import Self

from pydantic import JsonValue

from kernel_optimizer.agents.sandbox import Sandbox
from kernel_optimizer.evaluation.task_eval import TaskEvaluation
from .runner_records import FrozenRecord
from .search_records import EvaluationRequest
from .search_session import OperatorSession


class HelperReply(FrozenRecord):
    evaluation: TaskEvaluation
    slots_used: int
    slots_remaining: int
    receipt: Path


class HelperService:
    def __init__(self, session: OperatorSession) -> None:
        self.session = session
        class Handler(socketserver.StreamRequestHandler):
            def handle(self) -> None:
                try:
                    self.connection.settimeout(10)
                    payload = self.rfile.readline(1024 * 1024)
                    self.connection.settimeout(None)
                    request = EvaluationRequest.model_validate_json(payload)
                    result = session.self_test(request)
                except (ValueError, OSError) as exc:
                    with session.lock:
                        admitted = session.budget.admit()
                        result = TaskEvaluation(valid=False, detail=f"helper request rejected: {exc}")
                        row: dict[str, JsonValue] = {"purpose": "self_test", "admitted": admitted,
                            "opportunity": session.opportunity,
                            "slot": session.budget.used, "evaluation": result.model_dump(mode="json"), "forward_calls": 0}
                        session.attempts.append(row)
                        session._write("attempts.jsonl", row)
                reply = HelperReply(evaluation=result, slots_used=session.budget.used,
                    slots_remaining=max(0, session.budget.limit - session.budget.used), receipt=session.output / "attempts.jsonl")
                self.wfile.write(reply.model_dump_json().encode() + b"\n")
        self.server = socketserver.TCPServer(("127.0.0.1", 0), Handler)
        self.thread = Thread(target=self.server.serve_forever, name="c3-resident-self-test")

    def __enter__(self) -> Self:
        self.thread.start()
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None,
                 traceback: TracebackType | None) -> None:
        self.server.shutdown()
        self.thread.join()
        self.server.server_close()

    def seed_sandbox(self, sandbox: Sandbox) -> str:
        endpoint = {"host": "127.0.0.1", "port": self.server.server_address[1],
                    "deadline_unix_s": self.session.budget.deadline_unix_s,
                    "slots_remaining": self.session.budget.limit - self.session.budget.used}
        path = sandbox.write_input("task/operator-helper.json", json.dumps(endpoint, indent=2)).resolve()
        root = Path(__file__).resolve().parents[2]
        command = (f"PYTHONPATH={shlex.quote(str(root / 'src') + ':' + str(root))} {shlex.quote(sys.executable)} -B "
                   f"-m examples.c3_qwen3.search_helper --endpoint {shlex.quote(str(path))} "
                   "--request REQUEST.json --output RESULT.json")
        sandbox.write_input("task/operator-helper.md", command + "\n"
            "Request: absolute bundle path, params as {values: {...}}, optional site_groups mapping to traced module paths. "
            "Every attempt, including a failure, consumes one of the SAME eight opportunity slots. "
            "Use this existing resident runner; do not load a second model. Bash/read/write remain available. "
            "Private GPU work bypassing this helper is a protocol violation and must be reported, not hidden.\n")
        return "\nUse task/operator-helper.md for the actual absolute same-runner self-test command and budget.\n"


def request_test(endpoint: dict[str, JsonValue], payload: dict[str, JsonValue]) -> HelperReply:
    host, port = str(endpoint["host"]), int(str(endpoint["port"]))
    if host != "127.0.0.1":
        raise OSError("C3 helper only connects to its local resident process")
    with socket.create_connection((host, port), timeout=10) as connection:
        connection.settimeout(None)
        connection.sendall(json.dumps(payload).encode() + b"\n")
        with connection.makefile("rb") as stream:
            return HelperReply.model_validate_json(stream.readline(1024 * 1024))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("endpoint", "request", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        reply = request_test(json.loads(args.endpoint.read_text()), json.loads(args.request.read_text()))
        args.output.write_text(reply.model_dump_json(indent=2), encoding="utf-8")
        return 0 if reply.evaluation.valid else 1
    except (OSError, ValueError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
