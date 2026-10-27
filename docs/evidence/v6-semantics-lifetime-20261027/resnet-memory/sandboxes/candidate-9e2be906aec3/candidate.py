"""Memory-lean chunked composite for the observed ResNet-18 encoder stage bundles.

The accepted chunked-stem parent (bundle trial-4bdb465d41-param-0bfb4c5a,
measured peak_allocated_bytes 204,905,472 vs original 281,975,808) moved the
allocator peak out of the stem and into stages.0.layers.1: during the second
basic layer the live set is the resident weights (46,758,048 B) plus the
GPU-resident pixels (19,267,584 B) plus the stem output (25,690,112 B) plus
the layers.0 output (25,690,112 B) plus up to three more 25,690,112 B
(32,64,56,56) temporaries inside conv2d -> batch_norm -> relu. conv2d,
eval-mode batch_norm with frozen running stats, relu and the residual add_ are
strictly per-sample operators, so this replacement streams the frozen batch
through the two observed stage boundaries (model.resnet.encoder.stages.0 and
stages.1) in fixed sample chunks and copies each chunk result into one
preallocated full-size output, mirroring the accepted stem pattern. Every
chunk still executes the observed parent aten operators through
context['references'], so the parent work runs on each invocation and
reference telemetry records it.
"""

from __future__ import annotations

import torch
from torch.nn.modules.utils import _pair

STAGE_0 = "ed609fcfe9f1b7538a7256aa"
STAGE_1 = "e4174ed71f25bb1aa98a4321"
DEFAULT_CHUNK_SAMPLES = 8

STAGES = (
    {
        "id": STAGE_0,
        "label": "chunked_stage0_composite",
        "layers": (
            {
                "conv0": ("0e645d228cc3d2af27a1ecc4", "e2b61f3af66883244f03b706"),
                "bn0": ("cea956bf9012c4410f4b30f8", "847c427a9031ddd09016a1ce"),
                "relu0": ("42421d27240c30720dcc0863", "3ae07038f2fc000186a5c19b"),
                "conv1": ("d0384cb46bb193e5d6c55933", "65b33ff07c19bdb1887ee9d7"),
                "bn1": ("4529cf594d1da89461bd9f90", "b5390518b6e0a73cb8e23943"),
                "tail_identity": "06238f4bdd375d834b686fe3",
                "shortcut_identity": "5b18f7c4e53ac5f48d01e34a",
                "shortcut_conv": None,
                "shortcut_bn": None,
                "add": "50e433b07d2a2088a810e4a9",
                "relu_final": ("1d4d69ca555d2a7f570c36fd", "d8abe6995c6c03cb0269951e"),
            },
            {
                "conv0": ("21d413dfed396ad249d2a4b7", "c94673fb7aee6ef9fefe41bf"),
                "bn0": ("042ac61d92422fdc4fc91834", "ac875b541e21e70dec32bf80"),
                "relu0": ("87ff5f11b9431aba2e7d2693", "812190ea0f5add1201bdc7b7"),
                "conv1": ("b69c034d4b53e55dc56edd12", "f972518449600a297b7496a4"),
                "bn1": ("b05973afdd3024b06d823eeb", "96a737ec20aa408fc1fa9d24"),
                "tail_identity": "917e6260e60b88847b4d08f5",
                "shortcut_identity": "185eab8ac3768e370752cd6f",
                "shortcut_conv": None,
                "shortcut_bn": None,
                "add": "0a55f75461e5d6a41c77fe90",
                "relu_final": ("26fe32de6f89e29e5848acd6", "a36942ee716e0ac61dfa654e"),
            },
        ),
    },
    {
        "id": STAGE_1,
        "label": "chunked_stage1_composite",
        "layers": (
            {
                "conv0": ("dbf7bc380a16d0c71b121ac2", "5ae8a4b9b9555cbcd7a9d78d"),
                "bn0": ("6073def15630ca83b0fa7416", "b7a51a5243382e818c397c12"),
                "relu0": ("0216674971a5b836a999945d", "3110a6443ac2d93761b3d5c6"),
                "conv1": ("579d44310e19bc10445ec286", "ccb680b6fa3d5a2f0df29400"),
                "bn1": ("2743e487182d96cef02239b9", "284a9f512e723ab5a1d83998"),
                "tail_identity": "55c254925f1e83d2147f16a1",
                "shortcut_identity": None,
                "shortcut_conv": ("4f55669c9904be049f1a615f", "bb72850c82c1f09031c003d3"),
                "shortcut_bn": ("013eeadd234965480748c13f", "54ac9fcdf984cdfde95b2863"),
                "add": "82be61c5097290efe5a3112f",
                "relu_final": ("307c22fdca28029f286c3b89", "6508d164f59b80fbc5ad7f1a"),
            },
            {
                "conv0": ("c8992c6d4ebe2ac8945deabf", "5d2e4a5bb1e53820dcf7d240"),
                "bn0": ("4cd47aeb2954d35601a9ee57", "8dbe0c017aabeb02a2604e82"),
                "relu0": ("36ef01a8b27e69d58d78a709", "6a40efa91c6207ccdc63b63e"),
                "conv1": ("e43fb15b41a43f2b6382346a", "f60ea87bffbf74ac7701bb6f"),
                "bn1": ("f6b0ccf5b7ae11352441d478", "d6cd4f9f3cfba9ab59bc4777"),
                "tail_identity": "4fa20af9ecbc083fa05d4168",
                "shortcut_identity": "1461f1c9b7b18f46ad294e95",
                "shortcut_conv": None,
                "shortcut_bn": None,
                "add": "2507a0eb57b6f0457f0fb388",
                "relu_final": ("b660c4114fa105b873dffbd9", "26a75606952bd85d7ca907a1"),
            },
        ),
    },
)


def _conv_call(ref, mod, x):
    return ref(
        x,
        mod.weight,
        mod.bias,
        _pair(mod.stride),
        _pair(mod.padding),
        _pair(mod.dilation),
        int(mod.groups),
    )


def _bn_call(ref, mod, x):
    return ref(
        x,
        mod.weight,
        mod.bias,
        mod.running_mean,
        mod.running_var,
        False,
        float(mod.momentum),
        float(mod.eps),
        True,
    )


def _check_conv(mod):
    return (
        isinstance(mod, torch.nn.Conv2d)
        and mod.weight is not None
        and mod.weight.dtype == torch.float32
    )


def _check_bn(mod):
    return (
        isinstance(mod, torch.nn.BatchNorm2d)
        and bool(mod.affine)
        and bool(mod.track_running_stats)
        and mod.weight is not None
        and mod.bias is not None
        and mod.running_mean is not None
        and mod.running_var is not None
    )


def _resolve_stage(cfg, references, modules):
    stage_id = cfg["id"]
    stage_module = modules.get(stage_id)
    parent = references.get(stage_id)
    if parent is None and stage_module is not None:
        parent = stage_module.forward
    ready = parent is not None and isinstance(stage_module, torch.nn.Module)
    layers = []
    if ready:
        for spec in cfg["layers"]:
            try:
                conv0 = (references[spec["conv0"][0]], modules[spec["conv0"][1]])
                bn0 = (references[spec["bn0"][0]], modules[spec["bn0"][1]])
                relu0 = references[spec["relu0"][0]]
                conv1 = (references[spec["conv1"][0]], modules[spec["conv1"][1]])
                bn1 = (references[spec["bn1"][0]], modules[spec["bn1"][1]])
                add = references[spec["add"]]
                relu_final = references[spec["relu_final"][0]]
                if spec["shortcut_conv"] is None:
                    shortcut = None
                    shortcut_identity = modules[spec["shortcut_identity"]]
                    if not isinstance(shortcut_identity, torch.nn.Identity):
                        ready = False
                else:
                    shortcut = {
                        "conv": (
                            references[spec["shortcut_conv"][0]],
                            modules[spec["shortcut_conv"][1]],
                        ),
                        "bn": (
                            references[spec["shortcut_bn"][0]],
                            modules[spec["shortcut_bn"][1]],
                        ),
                    }
                bn_modules = [bn0[1], bn1[1]]
                if shortcut is not None:
                    bn_modules.append(shortcut["bn"][1])
                relu0_module = modules[spec["relu0"][1]]
                relu_final_module = modules[spec["relu_final"][1]]
                tail_identity = modules[spec["tail_identity"]]
                if not (
                    _check_conv(conv0[1])
                    and _check_conv(conv1[1])
                    and _check_bn(bn0[1])
                    and _check_bn(bn1[1])
                    and all(_check_bn(m) for m in bn_modules[2:])
                    and (shortcut is None or (_check_conv(shortcut["conv"][1]) and _check_bn(shortcut["bn"][1])))
                    and isinstance(relu0_module, torch.nn.ReLU)
                    and not bool(relu0_module.inplace)
                    and isinstance(relu_final_module, torch.nn.ReLU)
                    and not bool(relu_final_module.inplace)
                    and isinstance(tail_identity, torch.nn.Identity)
                    and all(item is not None for item in (conv0[0], bn0[0], relu0, conv1[0], bn1[0], add, relu_final))
                    and (shortcut is None or all(item is not None for item in (shortcut["conv"][0], shortcut["bn"][0])))
                ):
                    ready = False
                layers.append(
                    {
                        "conv0": conv0,
                        "bn0": bn0,
                        "relu0": relu0,
                        "conv1": conv1,
                        "bn1": bn1,
                        "shortcut": shortcut,
                        "add": add,
                        "relu_final": relu_final,
                        "bn_modules": tuple(bn_modules),
                    }
                )
            except Exception:
                ready = False
                layers = []
                break
    if not ready:
        layers = []
    return {
        "id": stage_id,
        "label": cfg["label"],
        "parent": parent,
        "stage_module": stage_module,
        "layers": layers,
        "ready": ready,
    }


def _supported(plan, x):
    if not plan["ready"]:
        return False
    try:
        if not isinstance(x, torch.Tensor) or not x.is_cuda or x.dim() != 4:
            return False
        if x.dtype != torch.float32 or not x.is_contiguous() or int(x.shape[0]) <= 0:
            return False
        if bool(plan["stage_module"].training):
            return False
        first_conv = plan["layers"][0]["conv0"][1]
        if int(x.shape[1]) != int(first_conv.in_channels):
            return False
        for layer in plan["layers"]:
            convs = [layer["conv0"][1], layer["conv1"][1]]
            if layer["shortcut"] is not None:
                convs.append(layer["shortcut"]["conv"][1])
            for conv in convs:
                if bool(conv.training) or conv.weight is None:
                    return False
                if conv.weight.dtype != torch.float32 or not conv.weight.is_cuda:
                    return False
                if str(getattr(conv, "padding_mode", "zeros")) != "zeros":
                    return False
                if int(getattr(conv, "groups", 1)) != 1:
                    return False
            for bn in layer["bn_modules"]:
                if bool(bn.training) or not bool(bn.affine) or not bool(bn.track_running_stats):
                    return False
                if bn.weight is None or bn.bias is None or bn.running_mean is None or bn.running_var is None:
                    return False
        return True
    except Exception:
        return False


def _chunked_forward(plan, chunk, x):
    total = int(x.shape[0])
    step = chunk if total >= chunk else total
    out = None
    for start in range(0, total, step):
        stop = min(start + step, total)
        piece = x[start:stop]
        h = piece
        for layer in plan["layers"]:
            layer_in = h
            ref, mod = layer["conv0"]
            h = _conv_call(ref, mod, h)
            ref, mod = layer["bn0"]
            h = _bn_call(ref, mod, h)
            h = layer["relu0"](h)
            ref, mod = layer["conv1"]
            h = _conv_call(ref, mod, h)
            ref, mod = layer["bn1"]
            h = _bn_call(ref, mod, h)
            shortcut = layer["shortcut"]
            if shortcut is None:
                residual = layer_in
            else:
                ref, mod = shortcut["conv"]
                residual = _conv_call(ref, mod, layer_in)
                ref, mod = shortcut["bn"]
                residual = _bn_call(ref, mod, residual)
            h = layer["add"](h, residual)
            h = layer["relu_final"](h)
        if out is None:
            out = torch.empty((total, *h.shape[1:]), dtype=h.dtype, device=h.device)
        out[start:stop].copy_(h)
    return out


def _make_replacement(plan, chunk, record_route):
    stage_id = plan["id"]
    parent = plan["parent"]
    ready = bool(plan["ready"])

    def replacement(*args, **kwargs):
        if parent is None:
            raise RuntimeError(f"no parent callable for observed stage {stage_id}")
        x = args[0] if args else kwargs.get("input")
        if not _supported(plan, x):
            record_route(stage_id, "setup_failure_parent" if not ready else "unsupported_signature_parent")
            return parent(*args, **kwargs)
        try:
            record_route(stage_id, plan["label"])
            return _chunked_forward(plan, chunk, x)
        except Exception:
            record_route(stage_id, "runtime_failure_parent")
            return parent(*args, **kwargs)

    return replacement


def build(context):
    references = context.get("references", {}) or {}
    modules = context.get("modules", {}) or {}
    parameters = context.get("parameters", {}) or {}
    record_route = context.get("record_route", lambda *_: None)
    chunk = parameters.get("chunk_samples", DEFAULT_CHUNK_SAMPLES)
    if isinstance(chunk, bool) or not isinstance(chunk, int) or chunk < 1:
        chunk = DEFAULT_CHUNK_SAMPLES
    return {
        cfg["id"]: _make_replacement(_resolve_stage(cfg, references, modules), chunk, record_route)
        for cfg in STAGES
    }
