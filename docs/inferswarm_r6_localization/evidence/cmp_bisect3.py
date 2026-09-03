import torch, json
s = torch.load("/tmp/la/captures-single-bisect3.pt", map_location="cpu", weights_only=False)
d = torch.load("/tmp/la/captures-first-bisect3-step8.pt", map_location="cpu", weights_only=False)
si = {(m["step"], m["checkpoint"]): t for m, t in zip(s["records"], s["tensors"])}
di = {(m["step"], m["checkpoint"]): t for m, t in zip(d["records"], d["tensors"])}
out = []
for (step, cp), st in sorted(si.items()):
    if cp not in ("after_layer_0", "embedding_output"): continue
    dt = di.get((step, cp))
    exact = bool(torch.equal(st, dt))
    diff = (st.double() - dt.double()).abs()
    out.append({"step": step, "checkpoint": cp, "exact_equal": exact,
                "max_absdiff": float(diff.max()), "nonzero_count": int((diff>0).sum())})
print(json.dumps(out, indent=1))
