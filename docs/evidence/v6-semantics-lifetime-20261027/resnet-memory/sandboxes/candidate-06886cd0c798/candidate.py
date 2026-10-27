"""Fused streaming composite for the observed ResNet-18 model.resnet bundle.

The accepted chain (chunked stem trial-4bdb465d41-param-0bfb4c5a, then chunked
stages.0/1 trial-31442e4c3a-param-9aa9f111; paired peak_allocated_bytes
281,975,808 -> 204,905,472 -> 155,229,184 B) is now capped by two moments that
cannot be reduced from the inner boundaries separately: the stem moment (one
full (32,64,56,56) preallocation plus two concurrent chunk-sized (c,64,112,112)
temporaries and the 7x7 conv workspace) and the stage-0 moment (the full-size
stem output plus the full-size stage-0 preallocation plus chunk temporaries),
both on top of the resident weights (46,758,048 B) and GPU-resident pixels
(19,267,584 B). This replacement takes over the whole model.resnet boundary and
streams the frozen batch through the identical parent aten chain in nested
sample chunks: the stem in stem_samples sub-chunks assembled into a per
stage-chunk buffer, stages.0/1 in stage_samples chunks copied into one macro
stage-1 buffer, and stages.2/3 plus the adaptive pooler in tail_samples macro
chunks written into one preallocated (32,512,7,7) last_hidden_state and
(32,512,1,1) pooler_output, so no full (32,64,56,56) stem output and no
full-size stage intermediate ever coexists with the weights and pixels.
conv2d, eval batch_norm (training=False, frozen running stats), out-of-place
relu, max_pool2d, the in-place residual add_ on chunk-owned temporaries, the
1x1 stride-2 shortcut convs and adaptive_avg_pool2d are strictly per-sample, so
fixed-batch chunking is elementwise exact, and every operator still executes
through context['references'] on each invocation. The two installed bundles are
explicitly replaced and their chunked behavior is preserved inside this fused
loop; output_hidden_states=True and any unsupported signature fall back to the
parent callable.
"""

from __future__ import annotations

import torch
from torch.nn.modules.utils import _pair

try:
    from transformers.modeling_outputs import (
        BaseModelOutputWithPoolingAndNoAttention as _OutputCls,
    )
except Exception:
    _OutputCls = None

MODEL_CALL = "fb7277dba44d770ed1cbb22f"
STEM_CONV_OP = "c50eca3f56c05cd6d8ecf149"
STEM_BN_OP = "6c3507a909e1a0f2daab2f33"
STEM_RELU_OP = "eeec508cd856dffbdda57b58"
STEM_POOL_OP = "531d78ee274899b94802f19d"
POOL_OP = "2e115926c50254b6cf5d5c85"
STEM_SPEC = (3, 64, 7, 2)

LAYER_SITES = (
    (
        0,
        0,
        "99fe6dec5636794a4019bebd",
        "82513c2dce9b362d19cc035a",
        "f6823d9cc3007d6d8cd9fd4a",
        "99fe6dec5636794a4019bebd",
        "82513c2dce9b362d19cc035a",
        "b5ac2c7b81d3862899629c14",
        "f6823d9cc3007d6d8cd9fd4a",
        None,
        None,
        (64, 64, 3, 1),
        (64, 64, 3, 1),
        None,
    ),
    (
        0,
        1,
        "99fe6dec5636794a4019bebd",
        "82513c2dce9b362d19cc035a",
        "f6823d9cc3007d6d8cd9fd4a",
        "99fe6dec5636794a4019bebd",
        "82513c2dce9b362d19cc035a",
        "b5ac2c7b81d3862899629c14",
        "f6823d9cc3007d6d8cd9fd4a",
        None,
        None,
        (64, 64, 3, 1),
        (64, 64, 3, 1),
        None,
    ),
    (
        1,
        0,
        "05fcd4a411a24fe379bb76ee",
        "bb770908f49fbb0f57fb3580",
        "d037925e1587ba6d846f62c0",
        "deab6dcbfa607b371374fd55",
        "bb770908f49fbb0f57fb3580",
        "4a71b80e11097eed5c4cdb8a",
        "d037925e1587ba6d846f62c0",
        "fde109aa95151c175af4d48a",
        "bb770908f49fbb0f57fb3580",
        (64, 128, 3, 2),
        (128, 128, 3, 1),
        (64, 128, 1, 2),
    ),
    (
        1,
        1,
        "deab6dcbfa607b371374fd55",
        "bb770908f49fbb0f57fb3580",
        "d037925e1587ba6d846f62c0",
        "deab6dcbfa607b371374fd55",
        "bb770908f49fbb0f57fb3580",
        "4a71b80e11097eed5c4cdb8a",
        "d037925e1587ba6d846f62c0",
        None,
        None,
        (128, 128, 3, 1),
        (128, 128, 3, 1),
        None,
    ),
    (
        2,
        0,
        "5604ac692ad65621f937fa94",
        "3a2ad08e7ac21ad8f038eef4",
        "df2c27a4e6c2d0e6339d3fa1",
        "b21668f41d9fb06e481b7ac6",
        "6e0698b461722017c7b34b0f",
        "c276affd9a33331513ba1aee",
        "f6bba3ab0e51c8d06823a513",
        "d75d81e4ac3f3ef9a2cabccc",
        "676fbdb9f1b64a0d53921081",
        (128, 256, 3, 2),
        (256, 256, 3, 1),
        (128, 256, 1, 2),
    ),
    (
        2,
        1,
        "e0dec3276cb2ef8762cd83a4",
        "bc65d5b388369d3040d18c6f",
        "8c5e33d99697b235debc58a2",
        "f0693d88324f90d8cd1ba86a",
        "ecb7faf10f62f364a9a8ca02",
        "02ef170c1a1e583b19aa7e54",
        "5488005b7d0670c133ccd331",
        None,
        None,
        (256, 256, 3, 1),
        (256, 256, 3, 1),
        None,
    ),
    (
        3,
        0,
        "9558b170ac4859e4173c9e02",
        "d822faa7a06ed2c7aa54573f",
        "89ac1f0557ceebf94bd891da",
        "baabb2ce566f7cb0689f2eb2",
        "3ed371865874dd003973ef30",
        "610b19656896ef485b73b42c",
        "453f2a281d2e8754017a58e6",
        "46a59afd3ab1f67ecfc98b46",
        "eee99d07f250494d6cdb10f2",
        (256, 512, 3, 2),
        (512, 512, 3, 1),
        (256, 512, 1, 2),
    ),
    (
        3,
        1,
        "dbe179fcec7253437f3b2686",
        "29ee40e0947eeb9039c9fbe4",
        "77611ebca79b698454ee66d8",
        "79ed3fea6cbba028226317b0",
        "98b9a8eb1b7561f975610f02",
        "27aedd3d29c17aca7bf948c7",
        "eeb82a5ca68e6e78098137cb",
        None,
        None,
        (512, 512, 3, 1),
        (512, 512, 3, 1),
        None,
    ),
)

DEFAULT_STEM_SAMPLES = 4
DEFAULT_STAGE_SAMPLES = 8
DEFAULT_TAIL_SAMPLES = 32


def _conv_ok(conv, spec):
    if not isinstance(conv, torch.nn.Conv2d):
        return False
    if conv.weight is None or conv.weight.dtype != torch.float32 or not conv.weight.is_cuda:
        return False
    if conv.bias is not None and (conv.bias.dtype != torch.float32 or not conv.bias.is_cuda):
        return False
    if str(getattr(conv, "padding_mode", "zeros")) != "zeros":
        return False
    if int(getattr(conv, "groups", 1)) != 1:
        return False
    if bool(conv.training):
        return False
    if spec is not None:
        in_c, out_c, k, s = spec
        if int(conv.in_channels) != in_c or int(conv.out_channels) != out_c:
            return False
        if tuple(_pair(conv.kernel_size)) != (k, k) or tuple(_pair(conv.stride)) != (s, s):
            return False
    return True


def _bn_ok(bn, num_features=None):
    if not isinstance(bn, torch.nn.BatchNorm2d):
        return False
    if bool(bn.training) or not bool(bn.affine) or not bool(bn.track_running_stats):
        return False
    if bn.weight is None or bn.bias is None or bn.running_mean is None or bn.running_var is None:
        return False
    if num_features is not None and int(bn.num_features) != int(num_features):
        return False
    return True


def _relu_ok(module):
    return isinstance(module, torch.nn.ReLU) and not bool(module.inplace)


def _resolve(context):
    references = context.get("references", {}) or {}
    modules = context.get("modules", {}) or {}
    models = context.get("models", {}) or {}
    parent = references.get(MODEL_CALL)
    resnet = modules.get(MODEL_CALL)
    if not isinstance(resnet, torch.nn.Module):
        root = models.get("model")
        resnet = getattr(root, "resnet", None) if isinstance(root, torch.nn.Module) else None
    plan = {
        "parent": parent,
        "module": resnet,
        "stem": None,
        "stages": None,
        "pool": None,
        "pooler": None,
        "training_modules": (),
        "stem_in_channels": 0,
        "ready": False,
    }
    if _OutputCls is None or not callable(parent) or not isinstance(resnet, torch.nn.Module):
        return plan
    try:
        embedder = resnet.embedder
        conv_layer = embedder.embedder
        conv = conv_layer.convolution
        norm = conv_layer.normalization
        act = conv_layer.activation
        maxpool = embedder.pooler
        if not (_conv_ok(conv, STEM_SPEC) and _bn_ok(norm, conv.out_channels) and _relu_ok(act)):
            raise ValueError("stem structure mismatch")
        if not isinstance(maxpool, torch.nn.MaxPool2d):
            raise ValueError("stem pooler mismatch")
        if bool(maxpool.return_indices) or bool(maxpool.ceil_mode):
            raise ValueError("stem pooler options mismatch")
        stem = {
            "conv": (references[STEM_CONV_OP], conv),
            "bn": (references[STEM_BN_OP], norm),
            "relu": references[STEM_RELU_OP],
            "pool": (references[STEM_POOL_OP], maxpool),
        }
        if stem["relu"] is None or any(stem[k][0] is None for k in ("conv", "bn", "pool")):
            raise ValueError("stem references missing")
        stages_mods = list(resnet.encoder.stages)
        if len(stages_mods) != 4:
            raise ValueError("stage count mismatch")
        stages = ([], [], [], [])
        training_modules = [conv, norm]
        for row in LAYER_SITES:
            stage_idx, layer_idx = row[0], row[1]
            layers = list(stages_mods[stage_idx].layers)
            if len(layers) != 2:
                raise ValueError("layer count mismatch")
            layer = layers[layer_idx]
            seq = list(layer.layer)
            if len(seq) != 2:
                raise ValueError("conv layer count mismatch")
            conv0, bn0, act0 = seq[0].convolution, seq[0].normalization, seq[0].activation
            conv1, bn1, act1 = seq[1].convolution, seq[1].normalization, seq[1].activation
            if not (_conv_ok(conv0, row[11]) and _conv_ok(conv1, row[12])):
                raise ValueError("layer conv mismatch")
            if not (_bn_ok(bn0, conv0.out_channels) and _bn_ok(bn1, conv1.out_channels)):
                raise ValueError("layer norm mismatch")
            if not (_relu_ok(act0) and isinstance(act1, torch.nn.Identity) and _relu_ok(layer.activation)):
                raise ValueError("layer activation mismatch")
            shortcut = None
            shortcut_mod = layer.shortcut
            if row[9] is None:
                if not isinstance(shortcut_mod, torch.nn.Identity):
                    raise ValueError("shortcut mismatch")
            else:
                if isinstance(shortcut_mod, torch.nn.Identity):
                    raise ValueError("shortcut mismatch")
                sc_conv = shortcut_mod.convolution
                sc_bn = shortcut_mod.normalization
                if not (_conv_ok(sc_conv, row[13]) and _bn_ok(sc_bn, sc_conv.out_channels)):
                    raise ValueError("shortcut structure mismatch")
                sc_conv_ref = references[row[9]]
                sc_bn_ref = references[row[10]]
                if sc_conv_ref is None or sc_bn_ref is None:
                    raise ValueError("shortcut references missing")
                shortcut = {"conv": (sc_conv_ref, sc_conv), "bn": (sc_bn_ref, sc_bn)}
                training_modules.extend((sc_conv, sc_bn))
            entry = {
                "conv0": (references[row[2]], conv0),
                "bn0": (references[row[3]], bn0),
                "relu0": references[row[4]],
                "conv1": (references[row[5]], conv1),
                "bn1": (references[row[6]], bn1),
                "add": references[row[7]],
                "relu_final": references[row[8]],
                "shortcut": shortcut,
            }
            if any(
                entry[key] is None
                for key in ("relu0", "add", "relu_final")
            ) or any(
                entry[key][0] is None for key in ("conv0", "bn0", "conv1", "bn1")
            ):
                raise ValueError("layer references missing")
            stages[stage_idx].append(entry)
            training_modules.extend((conv0, bn0, conv1, bn1))
        if any(len(stage) != 2 for stage in stages):
            raise ValueError("stage layer mapping mismatch")
        pooler = resnet.pooler
        if not isinstance(pooler, torch.nn.AdaptiveAvgPool2d):
            raise ValueError("pooler mismatch")
        pool_ref = references[POOL_OP]
        if pool_ref is None:
            raise ValueError("pooler reference missing")
        output_size = pooler.output_size
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        output_size = tuple(int(v) for v in output_size)
        if len(output_size) != 2:
            raise ValueError("pooler output size mismatch")
        plan.update(
            {
                "stem": stem,
                "stages": tuple(tuple(stage) for stage in stages),
                "pool": pool_ref,
                "pooler": pooler,
                "training_modules": tuple(training_modules),
                "stem_in_channels": int(conv.in_channels),
                "output_size": output_size,
                "ready": True,
            }
        )
    except Exception:
        plan["ready"] = False
    return plan


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


def _stem_forward(stem, piece):
    ref, mod = stem["conv"]
    h = _conv_call(ref, mod, piece)
    ref, mod = stem["bn"]
    h = _bn_call(ref, mod, h)
    h = stem["relu"](h)
    ref, mod = stem["pool"]
    h = ref(
        h,
        _pair(mod.kernel_size),
        _pair(mod.stride),
        _pair(mod.padding),
        _pair(mod.dilation),
        bool(mod.ceil_mode),
    )
    return h


def _layer_forward(layer, x):
    ref, mod = layer["conv0"]
    h = _conv_call(ref, mod, x)
    ref, mod = layer["bn0"]
    h = _bn_call(ref, mod, h)
    h = layer["relu0"](h)
    ref, mod = layer["conv1"]
    h = _conv_call(ref, mod, h)
    ref, mod = layer["bn1"]
    h = _bn_call(ref, mod, h)
    shortcut = layer["shortcut"]
    if shortcut is None:
        residual = x
    else:
        ref, mod = shortcut["conv"]
        residual = _conv_call(ref, mod, x)
        ref, mod = shortcut["bn"]
        residual = _bn_call(ref, mod, residual)
    h = layer["add"](h, residual)
    h = layer["relu_final"](h)
    return h


def _stream(plan, x, stem_step, stage_step, tail_step):
    total = int(x.shape[0])
    stem = plan["stem"]
    stages = plan["stages"]
    pool_ref = plan["pool"]
    output_size = plan["output_size"]
    tail = tail_step if total > tail_step else total
    last_hidden = None
    pooled = None
    for m0 in range(0, total, tail):
        m1 = min(m0 + tail, total)
        macro_n = m1 - m0
        s1_buf = None
        for c0 in range(m0, m1, stage_step):
            c1 = min(c0 + stage_step, m1)
            chunk_n = c1 - c0
            stem_buf = None
            for s0 in range(c0, c1, stem_step):
                s1 = min(s0 + stem_step, c1)
                h = _stem_forward(stem, x[s0:s1])
                if stem_buf is None:
                    stem_buf = torch.empty((chunk_n, *h.shape[1:]), dtype=h.dtype, device=h.device)
                stem_buf[s0 - c0:s1 - c0].copy_(h)
                h = None
            h = stem_buf
            stem_buf = None
            for layer in stages[0]:
                h = _layer_forward(layer, h)
            for layer in stages[1]:
                h = _layer_forward(layer, h)
            if s1_buf is None:
                s1_buf = torch.empty((macro_n, *h.shape[1:]), dtype=h.dtype, device=h.device)
            s1_buf[c0 - m0:c1 - m0].copy_(h)
            h = None
        t = s1_buf
        s1_buf = None
        for layer in stages[2]:
            t = _layer_forward(layer, t)
        for layer in stages[3]:
            t = _layer_forward(layer, t)
        if last_hidden is None:
            last_hidden = torch.empty((total, *t.shape[1:]), dtype=t.dtype, device=t.device)
            pooled = torch.empty((total, t.shape[1], *output_size), dtype=t.dtype, device=t.device)
        last_hidden[m0:m1].copy_(t)
        p = pool_ref(t, output_size)
        t = None
        pooled[m0:m1].copy_(p)
        p = None
    return last_hidden, pooled


def _supported(plan, x, output_hidden_states):
    if not plan["ready"]:
        return False
    if output_hidden_states:
        return False
    try:
        if not isinstance(x, torch.Tensor) or not x.is_cuda or x.dim() != 4:
            return False
        if x.dtype != torch.float32 or not x.is_contiguous() or int(x.shape[0]) <= 0:
            return False
        if int(x.shape[1]) != int(plan["stem_in_channels"]):
            return False
        if bool(plan["module"].training):
            return False
        for module in plan["training_modules"]:
            if bool(module.training):
                return False
        return True
    except Exception:
        return False


def _make_replacement(plan, stem_step, stage_step, tail_step, record_route):
    parent = plan["parent"]
    ready = bool(plan["ready"])

    def replacement(*args, **kwargs):
        if parent is None:
            raise RuntimeError(f"no parent callable for observed call {MODEL_CALL}")
        x = args[0] if args else kwargs.get("pixel_values")
        output_hidden_states = args[1] if len(args) > 1 else kwargs.get("output_hidden_states")
        return_dict = args[2] if len(args) > 2 else kwargs.get("return_dict")
        if not _supported(plan, x, output_hidden_states):
            record_route(
                MODEL_CALL,
                "setup_failure_parent" if not ready else "unsupported_signature_parent",
            )
            return parent(*args, **kwargs)
        try:
            record_route(MODEL_CALL, "fused_streaming_resnet")
            last_hidden, pooled = _stream(plan, x, stem_step, stage_step, tail_step)
            if return_dict is None:
                config = getattr(plan["module"], "config", None)
                return_dict = True if config is None else bool(getattr(config, "return_dict", True))
            if return_dict:
                return _OutputCls(
                    last_hidden_state=last_hidden,
                    pooler_output=pooled,
                    hidden_states=None,
                )
            return (last_hidden, pooled)
        except Exception:
            record_route(MODEL_CALL, "runtime_failure_parent")
            return parent(*args, **kwargs)

    return replacement


def _int_param(parameters, name, default):
    value = parameters.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return default
    return value


def build(context):
    parameters = context.get("parameters", {}) or {}
    record_route = context.get("record_route", lambda *_: None)
    stem_step = _int_param(parameters, "stem_samples", DEFAULT_STEM_SAMPLES)
    stage_step = _int_param(parameters, "stage_samples", DEFAULT_STAGE_SAMPLES)
    tail_step = _int_param(parameters, "tail_samples", DEFAULT_TAIL_SAMPLES)
    plan = _resolve(context)
    return {
        MODEL_CALL: _make_replacement(plan, stem_step, stage_step, tail_step, record_route)
    }
