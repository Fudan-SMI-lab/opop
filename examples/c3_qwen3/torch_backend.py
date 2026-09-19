import hashlib
import gc
import importlib
import importlib.metadata
import re
from collections.abc import Sequence
from contextlib import AbstractContextManager
from pathlib import Path
from time import perf_counter
from typing import Final

from .operator_types import NamedModel, Value
from .runner_records import AssetSpec, RunnerError
from .torch_fixtures import TorchFixtureCodec

QWEN_SOURCE_SHA: Final = "cbb7f2dc274c2f5592746c0dc6985ca50353efa07376f92cc922b77680a74f69"


class TorchBackend:
    """One native Qwen3 model; the only production boundary importing Torch/Transformers."""

    def __init__(self, assets: AssetSpec, device: str) -> None:
        if re.fullmatch(r"cuda:\d+", device) is None:
            raise RunnerError("native runner needs one explicit cuda:<index> device")
        versions = {"torch": "2.13.0+cu129", "triton": "3.7.1", "transformers": "5.16.1"}
        for name, expected in versions.items():
            if importlib.metadata.version(name) != expected:
                raise RunnerError(f"runtime version mismatch: {name} requires {expected}")
        root = Path(assets.local_path)
        for name, size in assets.file_sizes.items():
            path = root / name
            if not path.is_file() or path.stat().st_size != size:
                raise RunnerError(f"local pinned asset missing or wrong size: {name}")
        self.torch = importlib.import_module("torch")
        self.hf = importlib.import_module("transformers")
        source = importlib.import_module("transformers.models.qwen3.modeling_qwen3")
        if source.__file__ is None or hashlib.sha256(Path(source.__file__).read_bytes()).hexdigest() != QWEN_SOURCE_SHA:
            raise RunnerError("installed Qwen3 implementation differs from T1 source hash")
        self.device = self.torch.device(device)
        self.device_index = int(device.split(":")[1])
        with self.torch.cuda.device(self.device):
            self.engine = self.hf.Qwen3ForCausalLM.from_pretrained(
                str(root), dtype=self.torch.bfloat16, attn_implementation="sdpa",
                local_files_only=True, trust_remote_code=False,
            ).to(self.device)
            self.engine.eval()
            self.engine.requires_grad_(False)
        self.codec = TorchFixtureCodec()
        self.cache = None
        self.batch_size = 0
        self.forward_calls = 0

    @property
    def model(self) -> NamedModel:
        if self.engine is None:
            raise RunnerError("native model is closed")
        return self.engine

    def inference(self) -> AbstractContextManager:
        return self.torch.inference_mode()

    def clock(self) -> float:
        return perf_counter()

    def synchronize(self) -> None:
        self.torch.cuda.synchronize(self.device)

    def reset(self) -> None:
        self.cache = None
        self.batch_size = 0

    def begin(self, rows: Sequence[Sequence[int]]) -> None:
        self.torch.random.default_generator.manual_seed(0)
        with self.torch.cuda.device(self.device):
            self.torch.cuda.manual_seed(0)
        self.cache = self.hf.DynamicCache(config=self.engine.config)
        self.batch_size = len(rows)

    def forward(self, rows: Sequence[Sequence[int]]) -> Value:
        self.forward_calls += 1
        if self.cache is None or len(rows) != self.batch_size:
            raise RunnerError("forward requires fresh request-owned batch state")
        start = self.cache.get_seq_length()
        ids = self.torch.tensor(rows, dtype=self.torch.long, device=self.device)
        positions = self.torch.arange(start, start + ids.shape[1], device=self.device).unsqueeze(0)
        mask = self.torch.ones((self.batch_size, start + ids.shape[1]), dtype=self.torch.long, device=self.device)
        output = self.engine(input_ids=ids, attention_mask=mask, position_ids=positions,
                             past_key_values=self.cache, use_cache=True, logits_to_keep=1, return_dict=True)
        if output.past_key_values is not self.cache:
            raise RunnerError("model replaced or bypassed the request-owned KV cache")
        return output.logits[:, -1, :]

    def greedy(self, logits: Value) -> tuple[int, ...]:
        return tuple(self.torch.argmax(logits, dim=-1).tolist())

    def cpu_logits(self, logits: Value) -> tuple[tuple[float, ...], ...]:
        values = self.torch.as_tensor(logits).detach().float().cpu().tolist()
        return tuple(tuple(row) for row in values)

    def cache_length(self) -> int:
        if self.cache is None:
            raise RunnerError("request cache is absent")
        return int(self.cache.get_seq_length())

    def weights_stamp(self) -> tuple[int, ...]:
        if self.engine is None:
            return ()
        tensors = (*self.engine.parameters(), *self.engine.buffers())
        return tuple(part for p in tensors for part in (id(p), p.data_ptr(), p._version))

    def close(self) -> None:
        self.reset()
        self.engine = None
        gc.collect()
        with self.torch.cuda.device(self.device):
            self.torch.cuda.empty_cache()
