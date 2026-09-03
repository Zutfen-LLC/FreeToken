import torch, json
def load(p):
    b = torch.load(p, map_location="cpu", weights_only=False)
    return {(m["step"], m["checkpoint"]): t for m, t in zip(b["records"], b["tensors"])}
A = load("/tmp/la/captures-op-probe-3090.pt")
B = load("/tmp/la/captures-op-probe-3060.pt")
out=[]
for (step, cp), at in sorted(A.items()):
    bt = B[(step,cp)]
    exact = bool(torch.equal(at, bt))
    d = (at.double()-bt.double()).abs()
    out.append({"op": cp, "exact_equal": exact, "max_absdiff": float(d.max()),
                "nonzero": int((d>0).sum()), "total": at.numel()})
print(json.dumps(out, indent=1))
