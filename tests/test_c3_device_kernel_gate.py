from examples.c3_qwen3.device_ir import stored_dependencies


def test_store_provenance_when_actual_input_load_reaches_output() -> None:
    # Given: compiled-style SSA, not candidate Python text or a claimed kernel name.
    ir = """
    %0 = tt.splat %arg0 : !tt.ptr<f32>
    %1 = tt.load %0 : tensor<4xf32>
    %2 = arith.mulf %1, %1 : tensor<4xf32>
    %3 = tt.splat %arg1 : !tt.ptr<f32>
    tt.store %3, %2 : tensor<4xf32>
    """
    # When / Then: the stored result is derived from a real input pointer load.
    assert stored_dependencies(ir) == {1: frozenset({0})}


def test_dummy_store_when_only_constant_is_written() -> None:
    # Given: a launched dummy writing constants does not prove target computation.
    ir = "%0 = arith.constant 0.0 : f32\n%1 = tt.splat %arg1 : !tt.ptr<f32>\ntt.store %1, %0 : f32"
    # When / Then: no input-derived output is claimed.
    assert stored_dependencies(ir) == {}


def test_opencode_server_does_not_inherit_the_orchestrators_cuda_profiler_env(monkeypatch) -> None:
    # Given: the orchestrator process holds CUDA/Triton/profiler settings, and opencode is a
    # Node server that uses none of them. Both C3 fast probes that spawned this subprocess came
    # back with an empty CUDA activity list from a kernel that had demonstrably compiled,
    # launched and computed correctly, while all four `prove()` calls without it recovered their
    # activity names -- so a C3 lane can withhold the inherited CUDA view from the server.
    # That is a 6/6 correlation and not a demonstrated mechanism, so the strip is OPT-IN: this
    # test pins BOTH states, because C2's published arms ran with the full inherited env and a
    # changed default would silently alter their launch environment.
    import subprocess
    from pathlib import Path
    from kernel_optimizer.agents import runtime as rt

    for key, value in {"CUDA_VISIBLE_DEVICES": "0", "CUDA_INJECTION64_PATH": "/opt/libcupti.so",
                       "TRITON_CACHE_DIR": "/inherited/triton", "TORCHINDUCTOR_CACHE_DIR": "/inherited/ti",
                       "PATH": "/usr/bin", "HOME": "/root", "HTTPS_PROXY": "http://proxy:1",
                       "NODE_OPTIONS": "--max-old-space-size=4096"}.items():
        monkeypatch.setenv(key, value)

    seen: dict = {}

    class _Proc:
        def poll(self):
            return None

    def fake_popen(argv, **kwargs):
        seen.update(kwargs)
        return _Proc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(rt, "resolve_opencode", lambda: "opencode")
    monkeypatch.setattr(rt.OpencodeServer, "_wait_healthy", lambda self: None)

    def launch(**overrides):
        seen.clear()
        cfg = rt.OpencodeConfig(launch_cwd=Path("."), server_env={
            "TRITON_CACHE_DIR": "/explicit/triton",
            "OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX": "131072"}, **overrides)
        rt.OpencodeServer(cfg).start()
        return seen["env"]

    # When: the default, which every existing C2 arm used.
    env = launch()
    # Then: the environment is inherited whole -- no behaviour change for published runs.
    assert env["CUDA_VISIBLE_DEVICES"] == "0"
    assert env["CUDA_INJECTION64_PATH"] == "/opt/libcupti.so"
    assert env["TORCHINDUCTOR_CACHE_DIR"] == "/inherited/ti"

    # When: a C3 lane opts in.
    env = launch(strip_cuda_env_from_server=True)
    # Then: every CUDA/profiler-only key is gone unless the caller asked for it by name.
    assert "CUDA_VISIBLE_DEVICES" not in env
    assert "CUDA_INJECTION64_PATH" not in env
    assert "TORCHINDUCTOR_CACHE_DIR" not in env
    # An explicit server_env entry still wins over the strip -- isolation stays expressible.
    assert env["TRITON_CACHE_DIR"] == "/explicit/triton"
    assert env["OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX"] == "131072"
    # ...and nothing the server actually needs was collateral damage.
    assert (env["PATH"], env["HOME"]) == ("/usr/bin", "/root")
    assert env["HTTPS_PROXY"] == "http://proxy:1"
    assert env["NODE_OPTIONS"] == "--max-old-space-size=4096"
