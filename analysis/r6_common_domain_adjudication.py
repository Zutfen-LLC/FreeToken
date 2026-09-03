#!/usr/bin/env python3
"""Offline common-domain residual adjudication for InferSwarm R6.

Pure-stdlib (no numpy): reads retained float32-LE logits rows, selects the
frozen reference top-32 indices per declared step (0, 1, 7), and re-derives
the three comparisons T->S, T->D, S->D plus residual-structure diagnostics.

Retained evidence only:
  T = docs/inferswarm_r6/reference-generation.json  (Transformers ref)
  D = docs/inferswarm_r6/lifecycle/distributed-logits-0-1-7.f32.bin
  S = requal-inplace-softcap-b11f773/single-gpu-logits-0-1-7.f32.bin

No GPU/model execution, no threshold change, no historical result change.
Deterministic: identical inputs -> byte-identical outputs (verified by
running twice and hashing).

All arithmetic is IEEE-754 double on exact float32-promoted source values;
float32 source precision is preserved in the retained machine-readable table
(values are exactly representable; every stored residual is the exact double
difference of two float32 values).
"""
import argparse
import array
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent  # analysis/ -> repo root
STEPS = [0, 1, 7]
VOCAB = 262144
EXPECTED_T_TO_S = {0: 0.234375, 1: 0.34375, 7: 0.34375}
EXPECTED_T_TO_D = {0: 0.25, 1: 0.5, 7: 0.515625}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_rows(path):
    """Return list of 3 full-vocab rows (floats) from f32-LE binary."""
    data = Path(path).read_bytes()
    assert len(data) == VOCAB * 4 * len(STEPS), (
        f"unexpected size {len(data)} for {len(STEPS)} steps x {VOCAB} f32")
    a = array.array("f")
    a.frombytes(data)
    if sys.byteorder != "little":
        a.byteswap()
    assert a.itemsize == 4
    return [list(a[i * VOCAB:(i + 1) * VOCAB]) for i in range(len(STEPS))]


def mean(v):
    return math.fsum(v) / len(v)


def pct(vals, q):
    s = sorted(vals)
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * q
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    if lo == hi:
        return s[lo]
    return s[lo] * (hi - pos) + s[hi] * (pos - lo)


def dot(x, y):
    return math.fsum(a * b for a, b in zip(x, y))


def pearson(x, y):
    xm = mean(x)
    ym = mean(y)
    num = dot([a - xm for a in x], [b - ym for b in y])
    den = math.sqrt(dot([a - xm for a in x], [a - xm for a in x])
                    * dot([b - ym for b in y], [b - ym for b in y]))
    return None if den == 0.0 else num / den


def cosine(x, y):
    den = math.sqrt(dot(x, x)) * math.sqrt(dot(y, y))
    return None if den == 0.0 else dot(x, y) / den


def rankdata(v):
    """Average ranks (1-based), ties share the mean rank."""
    order = sorted(range(len(v)), key=lambda i: (v[i], i))
    ranks = [0.0] * len(v)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(x, y):
    return pearson(rankdata(x), rankdata(y))


def f32(x):
    """Round-trip through float32 (identity for our stored values)."""
    return float(struct_f32(x))


def struct_f32(x):
    import struct
    return struct.unpack("<f", struct.pack("<f", x))[0]


def residual_stats(T, S, D, res, idxs):
    a = [abs(r) for r in res]
    im = max(range(len(a)), key=lambda i: a[i])
    return {
        "max_abs_diff": a[im],
        "argmax_index": idxs[im],
        "argmax_T": T[im],
        "argmax_S": S[im],
        "argmax_D": D[im],
        "signed_residual_at_max": res[im],
        "mean_signed": mean(res),
        "mean_abs": mean(a),
        "rms": math.sqrt(mean([r * r for r in res])),
        "median_abs": pct(a, 0.5),
        "p90_abs": pct(a, 0.90),
        "min_signed": min(res),
        "max_signed": max(res),
    }


def ratios(num, den):
    """|num|/|den| entrywise; zero denominators counted, never divided."""
    out, zeros = [], 0
    for n, d in zip(num, den):
        if d == 0.0:
            zeros += 1
            out.append(None)
        else:
            out.append(abs(n / d))
    valid = [r for r in out if r is not None]
    res = {
        "entries": len(num),
        "zero_denominator_entries": zeros,
        "valid_entries": len(valid),
        "zero_denominator_positions": [
            i for i, r in enumerate(out) if r is None],
    }
    if valid:
        res.update({
            "median": pct(valid, 0.5),
            "mean": mean(valid),
            "max": max(valid),
            "min": min(valid),
            "q1": pct(valid, 0.25),
            "q3": pct(valid, 0.75),
            "p90": pct(valid, 0.90),
        })
    return res


def order_desc(v, idxs):
    return [idxs[i] for i in sorted(range(len(v)), key=lambda i: (-v[i], i))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--single-gpu-bin", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    ref_path = REPO / "docs/inferswarm_r6/reference-generation.json"
    dist_path = REPO / "docs/inferswarm_r6/lifecycle/distributed-logits-0-1-7.f32.bin"
    ref = json.loads(ref_path.read_text())

    inputs = {
        "reference_generation_json": {
            "path": str(ref_path), "sha256": sha256_file(ref_path)},
        "distributed_logits_bin": {
            "path": str(dist_path), "sha256": sha256_file(dist_path),
            "dtype": "float32 little-endian", "vocab": VOCAB,
            "rows": "steps 0,1,7 in order"},
        "single_gpu_logits_bin": {
            "path": args.single_gpu_bin, "sha256": sha256_file(args.single_gpu_bin),
            "dtype": "float32 little-endian", "vocab": VOCAB,
            "rows": "steps 0,1,7 in order"},
    }

    S_rows = load_rows(args.single_gpu_bin)
    D_rows = load_rows(dist_path)

    per_step = {}
    residual_rows = []
    for si, step in enumerate(STEPS):
        idxs = list(ref["step_top32_logits"][str(step)]["top_indices"])
        assert len(idxs) == 32 and len(set(idxs)) == 32, "bad frozen domain"
        T = [float(v) for v in ref["step_top32_logits"][str(step)]["top_values"]]
        assert len(T) == 32
        S = [S_rows[si][i] for i in idxs]
        D = [D_rows[si][i] for i in idxs]

        SmT = [s - t for s, t in zip(S, T)]
        DmT = [d - t for d, t in zip(D, T)]
        DmS = [d - s for d, s in zip(D, S)]

        # decomposition identity in exact float32 arithmetic
        recon = max(abs(struct_f32(a + b) - struct_f32(c))
                    for a, b, c in zip(SmT, DmS, DmT))

        A, B = SmT, DmT
        same = sum(1 for a, b in zip(A, B)
                   if (a > 0 and b > 0) or (a < 0 and b < 0) or (a == 0 and b == 0))
        gt = sum(1 for a, b in zip(A, B) if abs(b) > abs(a))
        lt = sum(1 for a, b in zip(A, B) if abs(b) < abs(a))

        per_step[str(step)] = {
            "indices": idxs,
            "stats": {
                "T_to_S": residual_stats(T, S, D, SmT, idxs),
                "T_to_D": residual_stats(T, S, D, DmT, idxs),
                "S_to_D": residual_stats(T, S, D, DmS, idxs),
            },
            "direction_A_ST_vs_B_DT": {
                "pearson": pearson(A, B),
                "cosine": cosine(A, B),
                "dot_product": dot(A, B),
                "frac_same_sign": same / 32.0,
                "frac_opposite_sign": (32 - same) / 32.0,
                "frac_absDT_gt_absST": gt / 32.0,
                "frac_absDT_lt_absST": lt / 32.0,
            },
            "decomposition_max_abs_reconstruction_error_f32": recon,
            "rank_diagnostics": {
                "T_top1_index": order_desc(T, idxs)[0],
                "S_top1_index": order_desc(S, idxs)[0],
                "D_top1_index": order_desc(D, idxs)[0],
                "spearman_T_vs_S": spearman(T, S),
                "spearman_T_vs_D": spearman(T, D),
                "spearman_S_vs_D": spearman(S, D),
                "T_order_desc": order_desc(T, idxs),
                "S_order_desc": order_desc(S, idxs),
                "D_order_desc": order_desc(D, idxs),
            },
            "relative_ratios": {
                "absDS_over_absDT": ratios(DmS, DmT),
                "absST_over_absDT": ratios(SmT, DmT),
            },
        }

        for k in range(32):
            residual_rows.append({
                "step": step, "index": idxs[k],
                "T": T[k], "S": S[k], "D": D[k],
                "S_minus_T": SmT[k], "D_minus_T": DmT[k], "D_minus_S": DmS[k],
                "abs_S_minus_T": abs(SmT[k]),
                "abs_D_minus_T": abs(DmT[k]),
                "abs_D_minus_S": abs(DmS[k]),
            })

    # generated-token positions
    gen = ref["generated_token_ids"]
    gen_tok = {}
    for si, step in enumerate(STEPS):
        tok = int(gen[step])
        idxs = per_step[str(step)]["indices"]
        entry = {
            "generated_token_id": tok,
            "in_reference_top32": tok in idxs,
            "single_argmax_full_vocab": max(range(VOCAB), key=lambda i: (S_rows[si][i], -i)),
            "distributed_argmax_full_vocab": max(range(VOCAB), key=lambda i: (D_rows[si][i], -i)),
        }
        if tok in idxs:
            row = next(r for r in residual_rows if r["step"] == step and r["index"] == tok)
            entry.update({k: row[k] for k in
                          ("T", "S", "D", "S_minus_T", "D_minus_T", "D_minus_S")})
        gen_tok[str(step)] = entry

    # hard reproduction gate
    for step in STEPS:
        got = per_step[str(step)]["stats"]["T_to_S"]["max_abs_diff"]
        assert got == EXPECTED_T_TO_S[step], f"T->S step {step}: {got}"
        got = per_step[str(step)]["stats"]["T_to_D"]["max_abs_diff"]
        assert got == EXPECTED_T_TO_D[step], f"T->D step {step}: {got}"

    def amax(key):
        return max(per_step[str(s)]["stats"][key]["max_abs_diff"] for s in STEPS)

    out = {
        "schema": "inferswarm.r6.single-vs-distributed-common-domain/1",
        "task": "offline common-domain residual adjudication (retained evidence only)",
        "source_head": "b11f7739b1377fa6109e8e3154b1188e11a4ffec (FreeToken branch inferswarm-r6)",
        "producers": {
            "historical_r6_physical": "44d6c94e4fd2ee967451cc959f930883ca3f4a25",
            "historical_distributed_evidence_arm": "3018acf721e897355e362f965f510471bf88d64c",
            "single_gpu_diagnostic": "b11f7739b1377fa6109e8e3154b1188e11a4ffec",
        },
        "domain": {
            "definition": "exact 32 reference top-32 indices per step from reference-generation.json",
            "steps": STEPS,
            "entries_per_step": 32,
            "substitutions_rejected": ["S top-32", "D top-32", "union top-k", "full vocabulary"],
        },
        "inputs": inputs,
        "per_step": per_step,
        "generated_token_positions": gen_tok,
        "cross_step_summary": {
            "T_to_S_max": {str(s): per_step[str(s)]["stats"]["T_to_S"]["max_abs_diff"] for s in STEPS},
            "T_to_D_max": {str(s): per_step[str(s)]["stats"]["T_to_D"]["max_abs_diff"] for s in STEPS},
            "S_to_D_max": {str(s): per_step[str(s)]["stats"]["S_to_D"]["max_abs_diff"] for s in STEPS},
            "aggregate_max": {"T_to_S": amax("T_to_S"), "T_to_D": amax("T_to_D"), "S_to_D": amax("S_to_D")},
            "reproduced_historical_exact": True,
        },
        "statements": [
            "no new physical execution occurred",
            "no comparator threshold changed",
            "no historical R6 result changed",
            "all numbers re-derived from already-retained logits",
        ],
    }

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    jpath = outdir / "single-vs-distributed-common-domain.json"
    jpath.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    cpath = outdir / "common-domain-residuals.csv"
    cols = ["step", "index", "T", "S", "D", "S_minus_T", "D_minus_T", "D_minus_S",
            "abs_S_minus_T", "abs_D_minus_T", "abs_D_minus_S"]
    with open(cpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(residual_rows)

    print(json.dumps({
        "json": str(jpath), "json_sha256": sha256_file(jpath),
        "csv": str(cpath), "csv_sha256": sha256_file(cpath),
        "rows": len(residual_rows),
    }))


if __name__ == "__main__":
    main()
