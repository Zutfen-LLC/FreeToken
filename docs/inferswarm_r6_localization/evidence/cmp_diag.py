import torch, json
def load(p):
    b = torch.load(p, map_location="cpu", weights_only=False)
    return {(m["step"], m["checkpoint"]): t for m, t in zip(b["records"], b["tensors"])}
S   = load("/tmp/la/captures-single-bisect3.pt")        # S arm, 3090, all 48 layers
D1  = load("/tmp/la/captures-first-bisect3-step8.pt")   # D stage 1, 3060 (01 GPU0)
T30 = load("/tmp/la/captures-first-diag-t4-3090.pt")    # stage-1 code on 3090
M30 = load("/tmp/la/captures-first-diag-t4-3060.pt")    # stage-1 code on 3060 (01 GPU0)
rows=[]
for step in (0,1,7):
    for cp in ("embedding_output","after_layer_0","after_layer_15"):
        rows.append({
          "step": step, "checkpoint": cp,
          "diag3090_vs_S": bool(torch.equal(T30[(step,cp)], S[(step,cp)])),
          "diag3060_vs_D_stage1": bool(torch.equal(M30[(step,cp)], D1[(step,cp)])),
          "diag3090_vs_diag3060": bool(torch.equal(T30[(step,cp)], M30[(step,cp)])),
        })
print(json.dumps(rows, indent=1))
