from __future__ import annotations

import torch
import torch.nn.functional as F


def _bias_free_linear(module):
    weight = getattr(module, "weight", None)
    if not isinstance(module, torch.nn.Module) or not isinstance(weight, torch.Tensor):
        return None
    if weight.dim() != 2 or getattr(module, "bias", None) is not None:
        return None
    return weight


def _packed_mlp(module):
    gate = _bias_free_linear(getattr(module, "gate_proj", None))
    up = _bias_free_linear(getattr(module, "up_proj", None))
    down = _bias_free_linear(getattr(module, "down_proj", None))
    if gate is None or up is None or down is None:
        return None
    if gate.shape != up.shape:
        return None
    if down.shape[1] != gate.shape[0]:
        return None
    if gate.dtype != up.dtype or gate.device != up.device:
        return None
    if gate.device != down.device:
        return None
    return gate, up


def _cat_weights(a, b):
    if torch.is_inference_mode_enabled() and not (a.is_inference() or b.is_inference()):
        with torch.inference_mode(False):
            return torch.cat((a, b), dim=0)
    return torch.cat((a, b), dim=0)


def _tensor_version(tensor):
    try:
        return tensor._version
    except RuntimeError:
        return None


def _make_forward(module, parent_forward, gate_weight, up_weight):
    packed = _cat_weights(gate_weight, up_weight)
    intermediate = gate_weight.shape[0]
    in_features = gate_weight.shape[1]
    dtype = packed.dtype
    device_type = packed.device.type
    gate_version = _tensor_version(gate_weight)
    up_version = _tensor_version(up_weight)
    act_fn = getattr(module, "act_fn", None)

    def forward(hidden_states, *args, **kwargs):
        try:
            if (
                not args
                and not kwargs
                and isinstance(hidden_states, torch.Tensor)
                and module.gate_proj.weight is gate_weight
                and module.up_proj.weight is up_weight
                and (gate_version is None or gate_weight._version == gate_version)
                and (up_version is None or up_weight._version == up_version)
                and hidden_states.dim() >= 1
                and hidden_states.shape[-1] == in_features
                and hidden_states.dtype == dtype
                and hidden_states.device.type == device_type
            ):
                proj = F.linear(hidden_states, packed)
                gate = proj[..., :intermediate]
                up = proj[..., intermediate:]
                activated = act_fn(gate) if act_fn is not None else F.silu(gate)
                return module.down_proj(activated * up)
        except Exception:
            pass
        return parent_forward(hidden_states, *args, **kwargs)

    return forward


def build(context):
    modules = (context or {}).get("modules") or {}
    references = (context or {}).get("references") or {}

    shared = {}
    replacements = {}

    for call_id, module in modules.items():
        pair = _packed_mlp(module)
        if pair is None:
            continue
        gate_weight, up_weight = pair
        parent_forward = references.get(call_id)
        if parent_forward is None:
            parent_forward = module.forward
        key = id(module)
        if key not in shared:
            shared[key] = _make_forward(module, parent_forward, gate_weight, up_weight)
        replacements[call_id] = shared[key]

    return replacements
