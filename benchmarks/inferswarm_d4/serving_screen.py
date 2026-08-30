"""Matched D4 control then weighted W4 serving screen."""
from __future__ import annotations
import argparse,json,statistics,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import inferswarm_d3.serving_screen as d3
D3_SHA="6677fe1c506376a55aa8dcabb8d5761dc0373ced9d9b053209991059556d5887"
D4_SHA="283595b7559bb3aa46a08c7d00cfef1e0a77eb62967d6392c618a63f35d34cdf"
ACTIVE_D4=False;_base=d3.command
def command(root,model,port,shape,placement):
 cmd=_base(root,model,port,shape,placement)
 if ACTIVE_D4:
  cmd[cmd.index("--inferswarm-d3-placement")]="--inferswarm-d4-placement"
  cmd.insert(cmd.index("--inferswarm-experimental-d3-graph-multiworker")+1,"--inferswarm-experimental-d4-capability-weighted")
 return cmd
def main():
 global ACTIVE_D4
 p=argparse.ArgumentParser();p.add_argument("--repo",required=True);p.add_argument("--model",required=True);p.add_argument("--revision",required=True);p.add_argument("--manifest",required=True);p.add_argument("--d3-placement",required=True);p.add_argument("--d4-placement",required=True);p.add_argument("--output-dir",required=True);ns=p.parse_args()
 root,out=Path(ns.repo),Path(ns.output_dir);out.mkdir(parents=True,exist_ok=True);d3.command=command
 ACTIVE_D4=False;d3.PLACEMENT_SHA=D3_SHA;control=d3.run_arm(root,out,ns.model,ns.revision,ns.manifest,ns.d3_placement,"CONTROL","ab")
 ACTIVE_D4=True;d3.PLACEMENT_SHA=D4_SHA;weighted=d3.run_arm(root,out,ns.model,ns.revision,ns.manifest,ns.d4_placement,"WEIGHTED","ab")
 for name,row in (("control",control),("weighted",weighted)):
  row["schema"]="inferswarm.d4.serving-arm/1";row["arm"]="D4-"+name.upper();(out/f"d4-{name}.json").write_text(json.dumps(row,indent=2)+"\n")
 tc=control["analysis"]["decode_tok_s"]["median"];tw=weighted["analysis"]["decode_tok_s"]["median"];gain=tw/tc
 classification="D4_WEIGHTING_STRONG" if gain>=1.15 else "D4_WEIGHTING_PROMISING" if gain>=1.05 else "D4_WEIGHTING_NEUTRAL" if gain>=.97 else "D4_WEIGHTING_HARMFUL"
 result={"schema":"inferswarm.d4.serving-analysis/1","control_median_decode_tok_s":tc,"weighted_median_decode_tok_s":tw,"WEIGHTED_GAIN":gain,"WEIGHTED_VS_S2B":tw/67.8157,"WEIGHTED_VS_S1":tw/53.2212,"classification":classification,"order":["D4-CONTROL","D4-WEIGHTED"]}
 (out/"d4-analysis.json").write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result,indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
