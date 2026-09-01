#!/bin/bash
# R4 physical campaign driver — runs ON inferswarm01 (Node A) as zutfen.
# Usage: r4_campaign.sh <producer_sha> <arm: diagnostic|clean|preflight|characterize|microbench>
# Research-internal; retains all raw output under docs/inferswarm_r4/.
set -euo pipefail

SHA="$1"
PHASE="$2"
FT=/home/zutfen/FreeToken-r4
VENV=/home/zutfen/FreeToken/.venv/bin/python
MODEL=/srv/models/nvidia/Qwen3.6-35B-A3B-NVFP4/491c2f1ea524c639598bf8fa787a93fed5a6fbce
OUT=$FT/docs/inferswarm_r4
NODE_B=10.0.0.219
TMPDIR=/var/tmp

cd "$FT"
mkdir -p "$OUT/raw"

identity_check() {
  echo "== producer identity =="
  git -C "$FT" rev-parse HEAD
  git -C "$FT" status --short | wc -l
  ssh -i /home/zutfen/.ssh/id_r4_staging zutfen@$NODE_B "git -C /home/zutfen/FreeToken-r4 rev-parse HEAD; git -C /home/zutfen/FreeToken-r4 status --short | wc -l"
}

case "$PHASE" in
  preflight)
    identity_check | tee "$OUT/raw/identity-check.txt"
    # Node B profile (captured on Node B)
    ssh -i /home/zutfen/.ssh/id_r4_staging zutfen@$NODE_B "cd /home/zutfen/FreeToken-r4 && TMPDIR=/var/tmp /home/zutfen/FreeToken/.venv/bin/python -m benchmarks.inferswarm_r4.capture_profile --node node-b --out /tmp/r4-node-b-profile.json" | tee "$OUT/raw/node-b-preflight.log"
    scp -q -i /home/zutfen/.ssh/id_r4_staging zutfen@$NODE_B:/tmp/r4-node-b-profile.json "$OUT/node-b-hardware.json"
    # Node A profile
    TMPDIR=$TMPDIR $VENV -m benchmarks.inferswarm_r4.capture_profile --node node-a --out "$OUT/node-a-hardware.json" 2>&1 | tee "$OUT/raw/node-a-preflight.log"
    ;;
  freeze-plan)
    $VENV -m benchmarks.inferswarm_r4.freeze_r4_plan \
      --r2-plan docs/inferswarm_r2/frozen-plan.json \
      --node-a-profile "$OUT/node-a-hardware.json" \
      --node-b-profile "$OUT/node-b-hardware.json" \
      --implementation-commit "$SHA" \
      --out-dir "$OUT" 2>&1 | tee "$OUT/raw/freeze-plan.log"
    scp -q -i /home/zutfen/.ssh/id_r4_staging "$OUT/r4-frozen-plan.json" zutfen@$NODE_B:/home/zutfen/FreeToken-r4/docs/inferswarm_r4/
    scp -q -i /home/zutfen/.ssh/id_r4_staging "$OUT/r4-frozen-plan.json.sha256" zutfen@$NODE_B:/home/zutfen/FreeToken-r4/docs/inferswarm_r4/
    ;;
  characterize)
    # iperf3 server on Node B (one-off), client here; then reverse; then bidir
    ssh -i /home/zutfen/.ssh/id_r4_staging zutfen@$NODE_B "nohup /usr/bin/iperf3 -s -1 >/tmp/iperf-server.log 2>&1 &" 
    sleep 1
    $VENV -m benchmarks.inferswarm_r4.network_characterize --peer $NODE_B --interface eno1 --out-dir "$OUT/raw" 2>&1 | tee "$OUT/raw/characterize-a.log"
    ;;
  diagnostic|clean)
    ARM="$PHASE"
    # start Node B service
    ssh -i /home/zutfen/.ssh/id_r4_staging zutfen@$NODE_B "cd /home/zutfen/FreeToken-r4 && nohup /home/zutfen/FreeToken/.venv/bin/python -m benchmarks.inferswarm_r4.node_b_service --plan docs/inferswarm_r4/r4-frozen-plan.json --model $MODEL $([ $ARM = diagnostic ] && echo --diagnostic) --ready-file /tmp/r4-node-b-ready.json > /tmp/r4-node-b.log 2>&1 & echo \$!"
    for i in $(seq 1 120); do
      ssh -i /home/zutfen/.ssh/id_r4_staging zutfen@$NODE_B "test -f /tmp/r4-node-b-ready.json" 2>/dev/null && break
      sleep 10
    done
    scp -q -i /home/zutfen/.ssh/id_r4_staging zutfen@$NODE_B:/tmp/r4-node-b-ready.json "$OUT/raw/node-b-ready-$ARM.json"
    scp -q -i /home/zutfen/.ssh/id_r4_staging zutfen@$NODE_B:/tmp/r4-node-b.log "$OUT/raw/node-b-$ARM.log"
    # run the arm
    TMPDIR=$TMPDIR $VENV -m benchmarks.inferswarm_r4.run_experiment \
      --arm "$ARM" --plan "$OUT/r4-frozen-plan.json" --model "$MODEL" \
      --reference docs/inferswarm_r2/reference-v2-session-a.json \
      --peer-host $NODE_B --out "$OUT/arm-$ARM.json" 2>&1 | tee "$OUT/raw/arm-$ARM.log"
    # fetch Node B final report
    ssh -i /home/zutfen/.ssh/id_r4_staging zutfen@$NODE_B "kill \$(pgrep -f node_b_service) 2>/dev/null; cp /tmp/r4-node-b.log /tmp/r4-node-b-final-$ARM.log" || true
    scp -q -i /home/zutfen/.ssh/id_r4_staging zutfen@$NODE_B:/tmp/r4-node-b-final-$ARM.log "$OUT/raw/node-b-$ARM-final.log" || true
    ;;
  microbench)
    ssh -i /home/zutfen/.ssh/id_r4_staging zutfen@$NODE_B "cd /home/zutfen/FreeToken-r4 && nohup /home/zutfen/FreeToken/.venv/bin/python -m benchmarks.inferswarm_r4.transport_microbench --server --port 18490 >/tmp/r4-microbench-b.log 2>&1 &"
    sleep 2
    $VENV -m benchmarks.inferswarm_r4.transport_microbench --client --peer $NODE_B --port 18490 --out "$OUT/transport-microbenchmark.json" 2>&1 | tee "$OUT/raw/microbench.log"
    ;;
  *) echo "unknown phase $PHASE"; exit 1;;
esac
echo "phase $PHASE complete"
