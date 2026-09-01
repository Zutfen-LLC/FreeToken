#!/bin/bash
# R4 physical campaign driver — runs ON inferswarm01 (Node A) as zutfen.
# Usage: r4_campaign.sh <producer_sha> <phase>
#   preflight   capture hardware profiles + run the mechanical fail-closed gate
#   freeze-plan freeze R4 plan (carries producer SHA) + planner authorization
#   characterize iperf3/ping network characterization
#   microbench  transport-only microbenchmark
#   diagnostic  W2/W4 diagnostic arm (correctness)
#   clean       W2/W4 clean measurement arm
#   test-summary generate the immutable test summary from a pytest log
#   manifest    generate MANIFEST.sha256 over canonical artifacts
# Result composition is run manually (compose_result.py) after all phases.
# Research-internal; retains all raw output under docs/inferswarm_r4/.
set -euo pipefail

SHA="$1"
PHASE="$2"
FT=/home/zutfen/FreeToken-r4
VENV=/home/zutfen/FreeToken/.venv/bin/python
MODEL=/srv/models/nvidia/Qwen3.6-35B-A3B-NVFP4/491c2f1ea524c639598bf8fa787a93fed5a6fbce
OUT=$FT/docs/inferswarm_r4
NODE_B=10.0.0.219
SSH_B="ssh -i /home/zutfen/.ssh/id_r4_staging zutfen@$NODE_B"
TMPDIR=/var/tmp
export PYTHONPATH=$FT/python

cd "$FT"
mkdir -p "$OUT/raw"

case "$PHASE" in
  preflight)
    # Mechanical fail-closed gate: producer identity (both nodes, clean
    # collection, no safe.directory noise), checkpoint identity, GPU/BDF,
    # VRAM, Block-B RAM, canonical link.  Exits non-zero before any
    # heavyweight realization if anything drifts.
    TMPDIR=$TMPDIR $VENV -m benchmarks.inferswarm_r4.r4_gate_cli \
      --producer-sha "$SHA" \
      --node-a-repo "$FT" \
      --node-b-repo /home/zutfen/FreeToken-r4 \
      --ssh-node-b "$SSH_B" \
      --node-a-model "$MODEL" \
      --node-b-model "$MODEL" \
      --out-dir "$OUT" 2>&1 | tee "$OUT/raw/preflight-gate.log"
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
    $SSH_B "nohup /usr/bin/iperf3 -s -1 >/tmp/iperf-server.log 2>&1 &"
    sleep 1
    $VENV -m benchmarks.inferswarm_r4.network_characterize --peer $NODE_B --interface eno1 --out-dir "$OUT/raw" 2>&1 | tee "$OUT/raw/characterize-a.log"
    ;;
  diagnostic|clean)
    ARM="$PHASE"
    # start Node B service
    $SSH_B "cd /home/zutfen/FreeToken-r4 && PYTHONPATH=/home/zutfen/FreeToken-r4/python nohup /home/zutfen/FreeToken/.venv/bin/python -m benchmarks.inferswarm_r4.node_b_service --plan docs/inferswarm_r4/r4-frozen-plan.json --model $MODEL $([ $ARM = diagnostic ] && echo --diagnostic) --ready-file /tmp/r4-node-b-ready.json > /tmp/r4-node-b.log 2>&1 & echo \$!"
    for i in $(seq 1 120); do
      $SSH_B "test -f /tmp/r4-node-b-ready.json" 2>/dev/null && break
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
    $SSH_B "kill \$(pgrep -f node_b_service) 2>/dev/null; sleep 3; cp /tmp/r4-node-b-final.json /tmp/r4-node-b-final-$ARM.json 2>/dev/null || true" || true
    scp -q -i /home/zutfen/.ssh/id_r4_staging zutfen@$NODE_B:/tmp/r4-node-b-final-$ARM.json "$OUT/raw/node-b-final-$ARM.json" || true
    ;;
  microbench)
    $SSH_B "cd /home/zutfen/FreeToken-r4 && PYTHONPATH=/home/zutfen/FreeToken-r4/python nohup /home/zutfen/FreeToken/.venv/bin/python -m benchmarks.inferswarm_r4.transport_microbench --server --port 18490 >/tmp/r4-microbench-b.log 2>&1 &"
    sleep 2
    $VENV -m benchmarks.inferswarm_r4.transport_microbench --client --peer $NODE_B --port 18490 --out "$OUT/transport-microbenchmark.json" 2>&1 | tee "$OUT/raw/microbench.log"
    ;;
  test-summary)
    # usage: r4_campaign.sh <sha> test-summary <pytest-log> [predecessor-log]
    PYTEST_LOG="${3:-$OUT/raw/pytest-focused.log}"
    PRED_LOG="${4:-$OUT/raw/pytest-predecessor.log}"
    $VENV -m benchmarks.inferswarm_r4.test_summary \
      --producer-sha "$SHA" \
      --focused-log "$PYTEST_LOG" \
      --predecessor-log "$PRED_LOG" \
      --out "$OUT/test-summary.json" 2>&1 | tee "$OUT/raw/test-summary.log"
    ;;
  manifest)
    ( cd "$OUT" && find . -type f ! -name '*.sha256' ! -name 'MANIFEST.sha256' \
        ! -path './raw/*' -print0 | sort -z | xargs -0 sha256sum > MANIFEST.sha256 )
    sha256sum "$OUT/MANIFEST.sha256" | awk '{print $1"  MANIFEST.sha256"}' > "$OUT/MANIFEST.sha256.sha256"
    echo "manifest entries: $(wc -l < "$OUT/MANIFEST.sha256")"
    ;;
  *) echo "unknown phase $PHASE"; exit 1;;
esac
echo "phase $PHASE complete"
