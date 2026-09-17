"""Real-layout source staging and fresh CPU imports; no GPU performance evidence."""

import ast
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from kernel_optimizer.config import AppConfig
from kernel_optimizer.gpu import worker_client
from kernel_optimizer.models.core import ParamDomain, ParameterSpace
from kernel_optimizer.paramspace.materializer import extract_defaults, materialize
from kernel_optimizer.paramspace.validation import SpaceValidator
from scripts.experiments.c2_retune import RetuneInputs, retune


@pytest.mark.parametrize("filename", ["parameterized.py", "source.py", "reference.py"])
def test_real_layout_keeps_primary_reference_and_importable_reserved_helpers(tmp_path, monkeypatch, filename):
    # Given: the actual parameterizer candidate/ layout, including input source.py.
    candidate = tmp_path / "generation-P/sandboxes/parameterizer-fixture/candidate"
    candidate.mkdir(parents=True)
    nested = candidate / "nested"
    nested.mkdir()
    (nested / "__init__.py").write_text("")
    (nested / "tool.py").write_text("VALUE = 11\n")
    source_helper = "PARAMS={'x': 1}\ndef payload(value): return value + 100\n"
    imports = "from nested.tool import VALUE\n"
    if filename != "source.py":
        (candidate / "source.py").write_text(source_helper)
        imports += "from source import payload\n"
    else:
        imports += "def payload(value): return value + 100\n"
    if filename != "reference.py":
        (candidate / "reference.py").write_text("OFFSET = 3\n")
        imports += "from reference import OFFSET\n"
    else:
        imports += "OFFSET = 3\n"
    primary = candidate / filename
    text = imports + "PARAMS={'x': 7, 'DW_STAGES': 2}\nRESULT = payload(PARAMS['x']) + VALUE * PARAMS['DW_STAGES'] + OFFSET\n"
    primary.write_bytes(text.replace("\n", "\r\n").encode())
    reference = tmp_path / "reference.py"
    reference.write_bytes(b"REFERENCE_SENTINEL = 99\r\n")
    space = ParameterSpace(space_id="published", candidate_id="child", source_sha="original", domains=[
        ParamDomain(name="x", kind="int", choices=list(range(80))),
        ParamDomain(name="DW_STAGES", kind="int", choices=[1, 2]),
    ])
    space_path = tmp_path / "space.json"
    space_path.write_text(space.model_dump_json())
    helpers = tuple(p for p in candidate.rglob("*.py") if p != primary)
    if filename == "source.py":
        helpers += (primary,)
    spec = RetuneInputs(task="level3:21", source=primary, reference=reference, space=space_path,
                        helpers=helpers, sampler_seed=0, evaluation_seed=0, output=tmp_path / "retune")
    cfg = AppConfig()
    cfg.gpu.compile_screen_enabled = False
    cfg.wsl.kernelbench_src = str(tmp_path / "kernelbench")
    before = {p: p.read_bytes() for p in (primary, reference, *helpers)}
    evaluated = []

    def worker(self, job, timeout_s, tag, **kwargs):
        path = Path(job["kernel_src_path"])
        assert set(extract_defaults(path.read_text())) == {"x", "DW_STAGES"}
        if job["job_type"] == "static_check":
            return {"ok": True}
        assert Path(job["ref_src_path"]).read_bytes() == before[reference]
        _, env = self._build_command(self.jobs_dir / "in.json", self.jobs_dir / "out.json")
        # The production builder uses POSIX separators; adapt only the Windows CPU transport.
        paths = re.split(r":(?=[A-Za-z]:[\\/])", env["PYTHONPATH"]) if os.name == "nt" else env["PYTHONPATH"].split(":")
        completed = subprocess.run([sys.executable, "-B", "-c",
            "from pathlib import Path; import sys; "
            "exec(compile(Path(sys.argv[1]).read_text(), sys.argv[1], 'exec')); print(RESULT)", str(path)],
            cwd=self.jobs_dir, env={**env, "PYTHONPATH": os.pathsep.join(paths)}, capture_output=True,
            text=True, timeout=30, check=False)
        assert completed.returncode == 0, completed.stderr
        params = extract_defaults(path.read_text())
        assert int(completed.stdout.strip()) == int(params["x"]) + 100 + 11 * int(params["DW_STAGES"]) + 3
        evaluated.append(path.read_text())
        value = 1.0 if params == {"x": 7, "DW_STAGES": 2} else 10.0
        return {"ok": True, "latency_ms": {"mean": value, "median": value, "min": value, "max": value,
                                          "std": 0.0, "n": job["num_perf_trials"]}}

    monkeypatch.setattr(worker_client, "_wsl_hop_needed", lambda: False)
    monkeypatch.setattr(worker_client.WslGpuWorker, "run_job", worker)
    # When: native validation/tuning/finals consume the actual staged primary.
    result = retune(spec, cfg)
    # Then: no input/helper overwrites another role, and actual imports work in fresh processes.
    assert result.status == "complete", result.rejection
    assert (spec.output / "inputs" / filename).read_bytes() == before[primary]
    assert (spec.output / "reference.py").read_bytes() == before[reference]
    for helper in helpers:
        assert (spec.output / "inputs" / helper.relative_to(candidate)).read_bytes() == before[helper]
    assert {p: p.read_bytes() for p in before} == before
    assert len(result.trials) == 40 and len(result.finals) == 3 and evaluated
    trial = result.selected_trial
    assert trial is not None
    selected = spec.output / "report/selected.py"
    measured = spec.output / "candidates" / trial.candidate_id / "trials" / f"{trial.trial_id}.py"
    assert selected.read_bytes() == measured.read_bytes()
    assert selected.read_text() == materialize(primary.read_text(), trial.params)
    manifest = json.loads((spec.output / "manifest.json").read_text())
    assert manifest["staged_source"] == f"inputs/{filename}"
    assert manifest["staged_reference"] == "reference.py"


def body_without_params(data: bytes) -> str:
    tree = ast.parse(data.decode("utf-8"))
    tree.body = [node for node in tree.body if not (isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "PARAMS" for t in node.targets))]
    return ast.dump(tree)


def test_archived_fifteen_sources_stage_exactly_without_changing_raw_exports(tmp_path, monkeypatch):
    # Given: the read-only exported T6 manifests and their original path maps.
    raw = Path(__file__).parents[1] / "results/c2-method-improvements"
    exports = [raw / f"export-manifest-{host}.json" for host in ("A", "B")]
    if not all(p.is_file() for p in exports):
        pytest.skip("T6 raw exports are not present; synthetic real-layout regressions still run")
    mappings, hashes = {}, []
    for export in exports:
        data = json.loads(export.read_text(encoding="utf-8"))
        mappings.update(data["remote_to_relative"])
        hashes.extend(data["files"].items())
    manifests = sorted({p for p, _ in hashes if p.startswith("core/") and p.endswith("/retune/manifest.json")})
    assert len(manifests) == 15
    proofs = []
    intended_different = body_different = helper_equal = 0

    class StagingCaptured(Exception):
        pass

    def stop_before_gpu(self, candidate, source, proposal, task, work_dir, **kwargs):
        captured.append((source, task.ref_path))
        raise StagingCaptured

    def forbidden(*args, **kwargs):
        pytest.fail("archived source staging must not execute a worker")

    monkeypatch.setattr(SpaceValidator, "validate_and_publish", stop_before_gpu)
    monkeypatch.setattr(worker_client.WslGpuWorker, "run_job", forbidden)
    # When: each original intended primary is staged into a new pytest-owned run only.
    for number, relative in enumerate(manifests):
        data = json.loads((raw / relative).read_text(encoding="utf-8"))["inputs"]
        primary = raw / mappings[data["source"]]
        reference = raw / mappings[data["reference"]]
        helpers = tuple(raw / mappings[p] for p in data["helpers"])
        old = (raw / relative).parent / "inputs/source.py"
        primary_bytes, old_bytes = primary.read_bytes(), old.read_bytes()
        helper_equal += int(old_bytes == helpers[0].read_bytes())
        intended_different += int(old_bytes != primary_bytes)
        body_different += int(body_without_params(old_bytes) != body_without_params(primary_bytes))
        spec = RetuneInputs.model_validate({**data, "source": primary, "reference": reference,
            "space": raw / mappings[data["space"]], "helpers": helpers, "output": tmp_path / f"case-{number}"})
        captured = []
        with pytest.raises(StagingCaptured):
            retune(spec, AppConfig())
        staged = spec.output / "inputs" / primary.name
        assert staged.read_bytes() == primary_bytes
        assert captured == [(primary.read_text(encoding="utf-8"), spec.output / "reference.py")]
        assert captured[0][1].read_bytes() == reference.read_bytes()
        published = ParameterSpace.model_validate_json(spec.space.read_text(encoding="utf-8"))
        assert set(extract_defaults(captured[0][0])) == set(published.param_names())
        proofs.append({"run": Path(relative).parent.as_posix(), "staged_sha256": hashlib.sha256(staged.read_bytes()).hexdigest(),
                       "restored_param_keys": sorted(set(extract_defaults(captured[0][0])) - set(extract_defaults(old_bytes.decode())))})
    # Then: the supplied failure counts are reproduced, exact new copies are proven, and raw files remain intact.
    assert (helper_equal, intended_different, body_different) == (15, 13, 10)
    assert all(hashlib.sha256((raw / p).read_bytes()).hexdigest() == digest for p, digest in hashes)
    print(json.dumps({"raw_manifest_entries_verified": len(hashes), "unique_raw_paths_verified": len({p for p, _ in hashes}),
                      "historical_helper_equal": helper_equal,
                      "historical_intended_different": intended_different, "historical_body_different": body_different,
                      "new_exact_staging": proofs}, indent=2))
