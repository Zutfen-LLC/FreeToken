"""Graph-captured, device-local A/B overlap control for D3 physical evidence."""
from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import time
from pathlib import Path

import torch

PRIMARY = "GPU-d5c05739-96c1-7e49-89b6-bf54c2121c55"
WORKER_A = "GPU-e1f2f90c-49ab-2689-0cf1-e5d9da520176"
WORKER_B = "GPU-1fc28f83-1d45-926e-54d0-ba1e835ef099"


def pct(v: list[float], q: float) -> float:
    v = sorted(v); p = (len(v)-1)*q; lo, hi = math.floor(p), math.ceil(p)
    return v[lo] if lo == hi else v[lo]*(hi-p)+v[hi]*(p-lo)


def dist(v: list[float]) -> dict:
    return {"n": len(v), "median_ms": statistics.median(v), "p95_ms": pct(v,.95), "max_ms": max(v)}


def ordinal(uuid: str) -> int:
    for i in range(torch.cuda.device_count()):
        if torch.cuda.get_device_properties(i).uuid.lower() == uuid.lower(): return i
    raise RuntimeError(f"required physical GPU absent: {uuid}")


def branch_graph(device: int, count: int, side: int):
    """Capture a fixed, allocation-free local GEMM chain on one worker."""
    d=torch.device("cuda",device); torch.cuda.set_device(d)
    g=torch.Generator(device=d).manual_seed(9100+side)
    x=torch.randn((2048,2048),device=d,dtype=torch.float32,generator=g)
    y=torch.randn((2048,2048),device=d,dtype=torch.float32,generator=g)
    out=torch.empty_like(x); stream=torch.cuda.Stream(device=d); graph=torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        for _ in range(count): torch.mm(x,y,out=out); x,out=out,x
    torch.cuda.synchronize(d)
    with torch.cuda.graph(graph,stream=stream):
        for _ in range(count): torch.mm(x,y,out=out); x,out=out,x
    return graph, d, (x,y,out)


def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument("--output",required=True); ap.add_argument("--iterations",type=int,default=100); ap.add_argument("--gemms",type=int,default=8); ns=ap.parse_args()
    p,a,b=ordinal(PRIMARY),ordinal(WORKER_A),ordinal(WORKER_B)
    # Independently captured calibration graphs; no cross-device clock arithmetic.
    ga, da, keep_a=branch_graph(a,ns.gemms,0); gb, db, keep_b=branch_graph(b,ns.gemms,1)
    for g,d in ((ga,da),(gb,db)):
        for _ in range(10): g.replay()
        torch.cuda.synchronize(d)
    def measure(g,d):
        values=[]
        for _ in range(ns.iterations):
            t=time.perf_counter_ns(); g.replay(); torch.cuda.synchronize(d); values.append((time.perf_counter_ns()-t)/1e6)
        return values
    va,vb=measure(ga,da),measure(gb,db)
    # One primary-captured graph with precisely the D3 dependency topology.
    dp=torch.device("cuda",p); torch.cuda.set_device(dp); sp=torch.cuda.Stream(device=dp); sa=torch.cuda.Stream(device=da); sb=torch.cuda.Stream(device=db)
    ready=torch.cuda.Event(); done_a=torch.cuda.Event(); done_b=torch.cuda.Event(); ab=torch.cuda.CUDAGraph()
    xa,ya,oa=keep_a; xb,yb,ob=keep_b
    with torch.cuda.graph(ab,stream=sp):
        ready.record(sp)
        with torch.cuda.stream(sa):
            ready.wait(sa)
            for _ in range(ns.gemms): torch.mm(xa,ya,out=oa); xa,oa=oa,xa
            done_a.record(sa)
        with torch.cuda.stream(sb):
            ready.wait(sb)
            for _ in range(ns.gemms): torch.mm(xb,yb,out=ob); xb,ob=ob,xb
            done_b.record(sb)
        done_a.wait(sp); done_b.wait(sp)
    for _ in range(10): ab.replay()
    torch.cuda.synchronize(dp); vab=measure(ab,dp)
    ma,mb,mab=statistics.median(va),statistics.median(vb),statistics.median(vab)
    result={"schema":"inferswarm.d3.overlap-control/1","physical_tested_freetoken_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),"topology":"gpu0_ready_then_a_and_b_then_gpu0_join","workers":{"a":{"uuid":WORKER_A,"cuda_index":a},"b":{"uuid":WORKER_B,"cuda_index":b}},"workload":{"kind":"preallocated_cuda_graph_captured_device_local_fp32_gemm_chain","gemms_per_branch":ns.gemms,"matrix":"2048x2048","host_activity_during_replay":False},"calibration":{"a":dist(va),"b":dist(vb)},"ab":dist(vab),"ratios":{"ab_over_max":mab/max(ma,mb),"ab_over_sum":mab/(ma+mb)},"overlap_target":{"ab_over_max_lt":1.4,"ab_over_sum_lt":.70},"overlap_confirmed":mab/max(ma,mb)<1.4 and mab/(ma+mb)<.70}
    Path(ns.output).write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); print(json.dumps(result,sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
