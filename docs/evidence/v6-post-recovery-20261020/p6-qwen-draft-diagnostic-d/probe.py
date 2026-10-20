"""Diagnostic of an unsubmitted Agent draft; never project acceptance evidence."""
from collections import Counter
from pathlib import Path
import hashlib
import importlib.util
import sys
import torch


def run(context):
    path = Path(context["parameters"]["candidate"])
    spec = importlib.util.spec_from_file_location("draft_composition", path)
    C = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = C
    spec.loader.exec_module(C)
    device = context["device"]
    dtype = torch.bfloat16
    refs = {}
    for cls, roles in C._CALLS.items():
        for role, cid in roles.items():
            if cid is None:
                continue
            if role in ("mul", "wmul", "mul1", "mul2"):
                refs[cid] = torch.ops.aten.mul.Tensor
            elif role in ("entry", "cast"):
                refs[cid] = torch.ops.aten.to.dtype
            elif role == "rsqrt":
                refs[cid] = torch.ops.aten.rsqrt.default
            elif role == "silu":
                refs[cid] = torch.ops.aten.silu.default
            elif role == "neg":
                refs[cid] = torch.ops.aten.neg.default
            elif role == "cat":
                refs[cid] = torch.ops.aten.cat.default
            elif role == "add":
                refs[cid] = torch.ops.aten.add.Tensor
            else:
                raise ValueError(role)
    routes = Counter()
    site_classes = {cid: cls for cls, roles in C._CALLS.items() for cid in roles.values() if cid}
    def route(cid, name):
        routes[(site_classes[cid], name)] += 1
    out = C.build({"references": refs, "record_route": route})
    torch.manual_seed(8127)
    examples = {}
    # Multiple current inputs and all six classes, without editing draft code,
    # signature guards, kernels or verification-window constants.
    for cls in C._CALLS:
        examples[cls] = []
        for _ in range(3):
            if cls.startswith("norm"):
                shape = {"normQ": (8, 1, 32, 128), "normK": (8, 1, 8, 128),
                         "normH": (8, 1, 2560)}[cls]
                examples[cls].append((torch.randn(shape, device=device, dtype=dtype),
                                      torch.randn((shape[-1],), device=device, dtype=dtype)))
            elif cls.startswith("rope"):
                heads = 32 if cls == "ropeQ" else 8
                q = torch.randn((8, 1, heads, 128), device=device, dtype=dtype).transpose(1, 2)
                table = torch.randn((1, 40, 128), device=device, dtype=dtype)
                examples[cls].append((q, table[:, 5:6].unsqueeze(1), table[:, 6:7].unsqueeze(1)))
            else:
                examples[cls].append(tuple(torch.randn((8, 1, 9728), device=device, dtype=dtype)
                                           for _ in range(2)))
    observations = {}
    for cls, samples in examples.items():
        ids = C._CALLS[cls]
        first_strides = {}
        checks = 0
        failure = None
        for iteration in range(C._VERIFY_N[cls] + 3):
            sample = samples[iteration % len(samples)]
            try:
                if cls.startswith("norm"):
                    x, w = sample
                    x32 = x.float() if ids["entry"] is None else out[ids["entry"]](x, torch.float32)
                    m = x32.pow(2).mean(-1, keepdim=True) + 1e-6
                    r = out[ids["rsqrt"]](m)
                    xn = out[ids["mul"]](x32, r)
                    tb = out[ids["cast"]](xn, dtype)
                    actual = out[ids["wmul"]](w, tb)
                    expected = w * (x32 * torch.rsqrt(m)).to(dtype)
                    intermediates = {"x32": x32, "m": m, "r": r, "xn": xn, "tb": tb}
                elif cls.startswith("rope"):
                    q, cos, sin = sample
                    qc = out[ids["mul1"]](q, cos)
                    n2 = out[ids["neg"]](q[..., 64:])
                    rot = out[ids["cat"]]([n2, q[..., :64]], -1)
                    rs = out[ids["mul2"]](rot, sin)
                    actual = out[ids["add"]](qc, rs)
                    expected = q * cos + torch.cat((-q[..., 64:], q[..., :64]), dim=-1) * sin
                    intermediates = {"q": q, "cos": cos, "qc": qc, "n2": n2,
                                     "rot": rot, "rs": rs, "actual": actual}
                else:
                    g, u = sample
                    s = out[ids["silu"]](g)
                    actual = out[ids["mul"]](s, u)
                    expected = torch.nn.functional.silu(g) * u
                    intermediates = {"s": s, "actual": actual}
                if not first_strides:
                    first_strides = {name: list(t.stride()) for name, t in intermediates.items()}
                if not torch.equal(actual, expected):
                    failure = {"iteration": iteration, "kind": "unequal_output",
                               "max_abs_error": float((actual.float() - expected.float()).abs().max()),
                               "mismatched_elements": int((actual != expected).sum())}
                    break
                checks += 1
            except Exception as exc:
                failure = {"iteration": iteration, "kind": type(exc).__name__, "error": str(exc)}
                break
        observations[cls] = {"checks_passed": checks, "failure": failure, "first_strides": first_strides}
    torch.cuda.synchronize()
    return {
        "candidate": str(path), "candidate_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "purpose": "Unsubmitted frozen Agent draft diagnostic; standalone synthetic pointwise chains, not full Qwen quality or performance acceptance",
        "candidate_code_unchanged": True, "verification_windows": C._VERIFY_N,
        "observations": observations,
        "routes": {cls: {name: count for (owner, name), count in routes.items() if owner == cls}
                   for cls in examples},
    }
