"""KernelBench task adapter: resolve local pinned tasks by level + problem id."""

from __future__ import annotations

from pathlib import Path

from kernel_optimizer.models.core import TaskSpec, sha256_text


class KernelBenchAdapter:
    def __init__(self, kb_root: Path):
        self.kb_root = Path(kb_root)
        if not (self.kb_root / "KernelBench").is_dir():
            raise FileNotFoundError(f"KernelBench problems dir not found under {kb_root}")

    def load(self, level: int, problem_id: int) -> TaskSpec:
        level_dir = self.kb_root / "KernelBench" / f"level{level}"
        matches = sorted(level_dir.glob(f"{problem_id}_*.py"))
        if not matches:
            raise FileNotFoundError(f"no task {problem_id}_*.py in {level_dir}")
        if len(matches) > 1:
            raise ValueError(f"ambiguous task id {problem_id} in {level_dir}: {matches}")
        ref_path = matches[0]
        src = ref_path.read_text(encoding="utf-8")
        return TaskSpec(
            level=level,
            problem_id=problem_id,
            name=ref_path.stem,
            ref_path=ref_path,
            ref_src_sha=sha256_text(src),
        )

    def ref_source(self, task: TaskSpec) -> str:
        return Path(task.ref_path).read_text(encoding="utf-8")


def parse_task_arg(arg: str) -> tuple[int, int]:
    """Parse 'level1:19' or 'level3:21' into (level, problem_id)."""
    left, _, right = arg.partition(":")
    if not left.startswith("level") or not right:
        raise ValueError(f"task must look like 'level3:21', got {arg!r}")
    return int(left.removeprefix("level")), int(right)
