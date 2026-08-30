"""D4 placement-only physical correctness gate: local oracle plus weighted AB."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import inferswarm_d3.whole_model_correctness as d3
D4_SHA="283595b7559bb3aa46a08c7d00cfef1e0a77eb62967d6392c618a63f35d34cdf"
_base=d3.command
def command(root,model,port,shape,placement):
 cmd=_base(root,model,port,shape,placement)
 if shape!="local":
  cmd[cmd.index("--inferswarm-d3-placement")]="--inferswarm-d4-placement"
  cmd.insert(cmd.index("--inferswarm-experimental-d3-graph-multiworker")+1,"--inferswarm-experimental-d4-capability-weighted")
 return cmd
def main():
 p=argparse.ArgumentParser();p.add_argument("--repo",required=True);p.add_argument("--model",required=True);p.add_argument("--revision",required=True);p.add_argument("--manifest",required=True);p.add_argument("--placement",required=True);p.add_argument("--output",required=True);ns=p.parse_args()
 d3.command=command;d3.PLACEMENT_SHA=D4_SHA;out=Path(ns.output);out.parent.mkdir(parents=True,exist_ok=True)
 rows=[d3.run_shape(Path(ns.repo),ns.model,ns.revision,ns.manifest,ns.placement,shape,out.parent) for shape in ("local","ab")]
 local,weighted=rows;difference=d3.first_difference(local["token_ids"],weighted["token_ids"]);report=weighted["d3_counters"]
 checks={"parser_exact":weighted["runtime_contract"]["checks"]["placement_sha"],"resident_banks_exact":report["worker_a_resident_slots"]==report["worker_b_resident_slots"]==3000,
 "ownership_exhaustive":weighted["ownership"]["selection_arithmetic_exact"] and weighted["ownership"]["worker_ab_disjoint"],"whole_model_graph":report["graph_active"] and report["captured_batch_sizes"]==[1],
 "deterministic_w4_equal":difference is None,"zero_runtime_anomalies":all(report[k]==0 for k in ("fallback_count","failure_count","graph_recapture_count","steady_state_host_sync_count","steady_state_expert_weight_bytes_host_to_worker_a","steady_state_expert_weight_bytes_host_to_worker_b")),
 "paging_valid":weighted["paging_delta"]["pswpin"]==weighted["paging_delta"]["pswpout"]==0}
 result={"schema":"inferswarm.d4.physical-correctness/1","placement_sha256":D4_SHA,"shapes":rows,"checks":checks,"classification":"D4_PLACEMENT_PRIMITIVE_PASS" if all(checks.values()) else "D4_INVALID"}
 out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n");print(json.dumps({"classification":result["classification"],"checks":checks},indent=2));return 0 if all(checks.values()) else 2
if __name__=="__main__":raise SystemExit(main())
