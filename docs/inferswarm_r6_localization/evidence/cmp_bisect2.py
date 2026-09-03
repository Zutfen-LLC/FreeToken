import torch, json
s = torch.load("/tmp/la/captures-single-bisect2.pt", map_location="cpu", weights_only=False)
d = torch.load("/tmp/la/captures-first-bisect2-step8.pt", map_location="cpu", weights_only=False)
si = {(m["step"], m["checkpoint"]): t for m, t in zip(s["records"], s["tensors"])}
di = {(m["step"], m["checkpoint"]): t for m, t in zip(d["records"], d["tensors"])}
out = []
for (step, cp), st in sorted(si.items()):
    if cp not in ("after_layer_1", "after_layer_2", "embedding_output"): continue
    dt = di.get((step, cp))
    if dt is None: out.append({"step": step, "checkpoint": cp, "missing": "D"}); continue
    exact = bool(torch.equal(st, dt))
    diff = (st.double() - dt.double()).abs()
    out.append({"step": step, "checkpoint": cp, "exact_equal": exact,
                "max_absdiff": float(diff.max()), "nonzero_frac": float((diff>0).float().mean()),
                "nonzero_count": int((diff>0).sum())})
print(json.dumps(out, indent=1))
