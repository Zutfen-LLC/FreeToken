import torch, json, math
s = torch.load("/tmp/la/captures-single-bisect1.pt", map_location="cpu", weights_only=False)
d = torch.load("/tmp/la/captures-first-bisect1-step8.pt", map_location="cpu", weights_only=False)
si = {(m["step"], m["checkpoint"]): t for m, t in zip(s["records"], s["tensors"])}
di = {(m["step"], m["checkpoint"]): t for m, t in zip(d["records"], d["tensors"])}
out = []
for (step, cp), st in sorted(si.items()):
    if not cp.startswith("after_layer_") or cp in ("after_layer_15","after_layer_31","after_layer_47"):
        continue
    dt = di.get((step, cp))
    if dt is None:
        out.append({"step": step, "checkpoint": cp, "missing": "D"}); continue
    exact = bool(torch.equal(st, dt))
    diff = (st.double() - dt.double()).abs()
    out.append({"step": step, "checkpoint": cp, "exact_equal": exact,
                "max_absdiff": float(diff.max()), "mean_absdiff": float(diff.mean()),
                "nonzero_frac": float((diff > 0).float().mean())})
print(json.dumps(out, indent=1))
