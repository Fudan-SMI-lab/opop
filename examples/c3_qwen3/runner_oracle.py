import hashlib
import sys
from array import array
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from .quality_arithmetic import nll
from .runner_records import OracleManifest, OraclePrompt, Prompt, RunnerError

if TYPE_CHECKING:
    from .model_runner import ResidentRunner


def continuation(runner: "ResidentRunner", prompt: Prompt, targets: tuple[int, ...] = ()) -> Iterator[tuple[int, tuple[float, ...]]]:
    backend = runner.backend
    backend.reset()
    with backend.inference():
        backend.begin((prompt.input_ids,))
        current = (prompt.input_ids,)
        try:
            for step in range(32):
                runner.binding.phase = "prefill" if step == 0 else "decode"
                logits = backend.forward(current)
                token = targets[step] if targets else backend.greedy(logits)[0]
                values = backend.cpu_logits(logits)
                if len(values) != 1 or backend.cache_length() != len(prompt.input_ids) + step:
                    raise RunnerError("quality request has wrong batch/cache position")
                yield token, values[0]
                current = ((token,),)
        finally:
            backend.reset()


def create_oracles(runner: "ResidentRunner", prompt_ids: tuple[str, ...]) -> Path:
    if runner.binding.active or runner.receipt.site_ids:
        raise RunnerError("oracle creation requires original baseline dispatch")
    if sys.byteorder != "little":
        raise RunnerError("oracle f32 format requires little endian")
    directory = runner.output_dir / ("oracle-" + uuid4().hex)
    directory.mkdir(parents=True)
    records: list[OraclePrompt] = []
    for prompt_id in prompt_ids:
        prompt = runner.prepared.prompts_by_id[prompt_id]
        path = directory / f"{prompt_id}.f32"
        tokens: list[int] = []
        losses: list[float] = []
        digest = hashlib.sha256()
        width = 0
        with path.open("wb") as stream:
            for token, logits in continuation(runner, prompt):
                values = array("f", logits)
                data = values.tobytes()
                stream.write(data)
                digest.update(data)
                width = len(logits)
                tokens.append(token)
                losses.append(nll(tuple(values), token))
        records.append(OraclePrompt(prompt_id=prompt_id, input_ids=prompt.input_ids, tokens=tuple(tokens),
                                    logits_file=path.name, logits_sha256=digest.hexdigest(),
                                    vocab_size=width, nll=tuple(losses)))
    manifest = OracleManifest(contract_sha256=runner.prepared.contract_sha256,
                              corpus_sha256=runner.prepared.corpus_sha256, baseline=True, prompts=tuple(records))
    path = directory / "oracle.json"
    path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return path
