"""Ordinary-serving bridge for the InferSwarm R5B epoch research gate."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from freetoken.message import AbortMsg, DetokenizeMsg, TokenizeMsg, UserReply
from freetoken.research.r3_planner import require_frozen
from freetoken.research.r5b_epochs import EpochServingController
from freetoken.tokenizer.detokenize import DetokenizeManager


class R5BFrontendDispatcher:
    """Route the existing TokenizeMsg waist through one epoch-aware controller."""

    def __init__(self, config_path: str, server_config: Any) -> None:
        from benchmarks.inferswarm_r4.r4_plan import load_r4_plan
        from benchmarks.inferswarm_r5a.runtime import (
            realize_local_split_plan,
            require_clean_exact_source,
        )
        from benchmarks.inferswarm_r5b.runtime import realize_isolated_network_plan
        from benchmarks.inferswarm_r5a.strategy import (
            evidence_catalog,
            objective,
        )
        from benchmarks.inferswarm_r5b.strategy import (
            QwenTokenBoundaryStrategy,
            compile_candidate,
            operator_policy,
            policy_evaluator,
            planning_problem,
            snapshot_with_availability,
            transition_policy,
        )

        path = Path(config_path).resolve()
        raw = json.loads(path.read_text())
        self.config_path = path
        self.raw = raw
        self.repository_root = Path(
            raw.get("repository_root") or path.parents[2]
        ).resolve()
        environment_path = self._resolve(path, raw["environment"])
        r4_plan_path = self._resolve(path, raw["participant_plan"])
        local_plan_path = self._resolve(
            path,
            raw.get(
                "local_participant_plan",
                str(self.repository_root / "docs/inferswarm_r2/frozen-plan.json"),
            ),
        )
        self.report_path = self._resolve(path, raw["report_out"])
        self.environment = json.loads(environment_path.read_text())
        require_frozen(self.environment, "R5B frozen environment")
        expected_sha = self.environment["implementation_commit"]
        require_clean_exact_source(self.repository_root, expected_sha)
        self.r4_plan = load_r4_plan(r4_plan_path)
        self.local_plan = json.loads(local_plan_path.read_text())
        if self.r4_plan.get("provenance", {}).get("r4", {}).get(
            "producer_sha"
        ) != expected_sha:
            raise RuntimeError("participant plan producer differs from frozen R5B source")
        if self.local_plan.get("digest") != self.environment.get(
            "compatibility", {}
        ).get("local_participant_plan_digest"):
            raise RuntimeError("local participant plan differs from frozen R5B environment")
        records_path = self._resolve(path, raw["serving_evidence"])
        records = json.loads(records_path.read_text())
        require_frozen(records, "R5B serving evidence catalog derivative")

        problem = planning_problem(expected_sha)
        initial_snapshot = snapshot_with_availability(
            self.environment, gpu_a1_available=False
        )
        policy = operator_policy(expected_sha)
        declared_objective = objective(expected_sha, raw.get("objective", "ttft_ms"))
        evidence = evidence_catalog(
            expected_sha,
            self.repository_root,
            serving_records=records.get("records", []),
        )
        frozen_transition_policy = transition_policy(expected_sha)

        def compiler(evaluation):
            return compile_candidate(
                dict(evaluation), r4_plan=self.r4_plan, local_plan=self.local_plan
            )

        def realizer(execution_plan):
            realization_path = execution_plan["strategy_realization"]["path"]
            if realization_path == "r2-local-split":
                return realize_local_split_plan(
                    dict(execution_plan),
                    local_plan=self.local_plan,
                    local_plan_path=local_plan_path,
                    model_path=server_config.model_path,
                    diagnostic=bool(raw.get("diagnostic", False)),
                    local_gate=self.environment["compatibility"][
                        "local_split_preflight"
                    ],
                )
            if realization_path == "r4-persistent-boundary":
                return realize_isolated_network_plan(
                    dict(execution_plan),
                    r4_plan=self.r4_plan,
                    model_path=server_config.model_path,
                    peer_host=raw["peer_host"],
                    peer_port=int(raw.get("peer_port", 18485)),
                    diagnostic=bool(raw.get("diagnostic", False)),
                )
            raise RuntimeError(
                f"R5B selected unsupported realization path {realization_path!r}"
            )

        self.controller = EpochServingController(
            problem=problem,
            initial_snapshot=initial_snapshot,
            policy=policy,
            objective=declared_objective,
            evidence_catalog=evidence,
            compiler=compiler,
            realizer=realizer,
            transition_strategy=QwenTokenBoundaryStrategy(),
            transition_policy=policy_evaluator(frozen_transition_policy),
        )
        self.transition_policy = frozen_transition_policy
        self._detokenizer = None
        self._detokenizer_lock = threading.Lock()
        self._tasks: set[asyncio.Task] = set()
        self._report_lock = threading.Lock()
        self._closed = False
        self._event_secret = str(raw["event_secret"]).encode()
        self._event_socket_path = self._resolve(path, raw["event_socket"])
        self._event_socket_path.parent.mkdir(parents=True, exist_ok=True)
        if self._event_socket_path.exists():
            self._event_socket_path.unlink()
        self._event_socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self._event_socket.bind(str(self._event_socket_path))
        os.chmod(self._event_socket_path, 0o600)
        self._event_thread = threading.Thread(
            target=self._event_loop, name="inferswarm-r5b-events", daemon=True
        )
        self._event_thread.start()
        self._write_report()

    @staticmethod
    def _resolve(config_path: Path, value: str) -> Path:
        candidate = Path(value)
        return (
            candidate.resolve()
            if candidate.is_absolute()
            else (config_path.parent / candidate).resolve()
        )

    def _ensure_detokenizer(self, state) -> DetokenizeManager:
        with self._detokenizer_lock:
            if self._detokenizer is None:
                self._detokenizer = DetokenizeManager(
                    state.frontend_tokenizer().tokenizer
                )
            return self._detokenizer

    def _current_gpu_a1(self) -> dict[str, Any]:
        expected = self.environment["node_a"]["gpus"][1]
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=uuid,pci.bus_id,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        rows = []
        for line in output.splitlines():
            uuid, bdf, total_mib = (item.strip() for item in line.split(","))
            rows.append(
                {
                    "uuid": uuid,
                    "pci_bdf": bdf,
                    "vram_total_bytes": int(total_mib) * 1024 * 1024,
                }
            )
        actual = next((item for item in rows if item["uuid"] == expected["uuid"]), None)
        if actual is None:
            raise RuntimeError("frozen GPU A1 is not discoverable")
        actual.update(
            {
                "integrity_eligible": True,
                "representation_backend_compatible": bool(
                    self.environment["compatibility"]["local_split_preflight"]
                    ["representation_backend"]["compatible"]
                ),
            }
        )
        return actual

    def _event_loop(self) -> None:
        from benchmarks.inferswarm_r5b.strategy import validate_gpu_a1_event

        while not self._closed:
            try:
                data = self._event_socket.recv(65536)
            except OSError:
                return
            if not data or data == b"CLOSE":
                continue
            try:
                envelope = json.loads(data)
                payload = dict(envelope["payload"])
                canonical = json.dumps(
                    payload, sort_keys=True, separators=(",", ":")
                ).encode()
                expected_mac = hmac.new(
                    self._event_secret, canonical, hashlib.sha256
                ).hexdigest()
                if not hmac.compare_digest(expected_mac, str(envelope.get("hmac", ""))):
                    raise ValueError("resource event HMAC is invalid")
                payload["authenticated"] = True
                payload["observed_at_ns"] = time.perf_counter_ns()
                payload["observed_identity"] = self._current_gpu_a1()
                validate_gpu_a1_event(payload, self.environment)
                available = payload["kind"] in ("AVAILABLE", "RETURNED")
                snapshot = __import__(
                    "benchmarks.inferswarm_r5b.strategy",
                    fromlist=["snapshot_with_availability"],
                ).snapshot_with_availability(
                    self.environment, gpu_a1_available=available
                )
                payload["resource_snapshot_digest"] = snapshot["digest"]
                active = self.controller.active_epoch
                payload["active_plan_executable"] = not (
                    not available
                    and payload["resource_id"]
                    in active.execution_plan["compute_units"]
                )
                self.controller.submit_resource_event(payload, snapshot)
            except Exception as exc:  # evidence must retain control-seam failures
                self.controller._event_audit.append(
                    {
                        "accepted": False,
                        "received_at_ns": time.perf_counter_ns(),
                        "failure_reason": f"{type(exc).__name__}: {exc}",
                    }
                )
            finally:
                self._write_report()

    async def submit(self, msg, state) -> None:
        if isinstance(msg, AbortMsg):
            return
        if not isinstance(msg, TokenizeMsg):
            raise RuntimeError("R5B epoch serving accepts generation requests only")
        loop = asyncio.get_running_loop()
        task = asyncio.create_task(asyncio.to_thread(self._run, msg, state, loop))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _deliver(self, state, reply: UserReply) -> None:
        state.stats.observe(reply)
        if reply.uid not in state.ack_map:
            return
        state.ack_map[reply.uid].append(reply)
        state.event_map[reply.uid].set()

    def _run(self, msg: TokenizeMsg, state, loop) -> None:
        try:
            manager = state.frontend_tokenizer()
            prompt_ids = manager.tokenize([msg])[0].tolist()
            detokenizer = self._ensure_detokenizer(state)
            maximum = int(msg.sampling_params.max_tokens)

            def on_token(step, token_id, commit):
                finished = step + 1 == maximum
                text = detokenizer.detokenize(
                    [
                        DetokenizeMsg(
                            uid=msg.uid,
                            next_token=int(token_id),
                            finished=finished,
                            finish_reason="length" if finished else None,
                            stop_strs=list(msg.sampling_params.stop_strs or []),
                        )
                    ]
                )[0]
                reply = UserReply(
                    uid=msg.uid,
                    incremental_output=text,
                    finished=finished,
                    prompt_tokens_delta=len(prompt_ids) if step == 0 else 0,
                    completion_tokens_delta=1,
                    finish_reason="length" if finished else None,
                    prefill=None,
                )
                loop.call_soon_threadsafe(self._deliver, state, reply)

            self.controller.serve_tokens(
                session_id=msg.uid,
                prompt_token_ids=prompt_ids,
                max_new_tokens=maximum,
                sampling_inputs={
                    "temperature": float(msg.sampling_params.temperature),
                    "top_p": float(msg.sampling_params.top_p),
                    "top_k": int(msg.sampling_params.top_k),
                    "seed": getattr(msg.sampling_params, "seed", None),
                },
                on_token=on_token,
            )
            self._write_report()
        except Exception as exc:  # request must receive a terminal fail-closed error
            reply = UserReply(
                uid=msg.uid,
                incremental_output="",
                finished=True,
                error=f"InferSwarm R5B request failed: {type(exc).__name__}: {exc}",
                error_code="inferswarm_r5b_failure",
            )
            loop.call_soon_threadsafe(self._deliver, state, reply)
            self._write_report()

    def _write_report(self) -> None:
        from freetoken.research.n0_model_block import write_json_with_sha

        with self._report_lock:
            self.report_path.parent.mkdir(parents=True, exist_ok=True)
            value = self.controller.report()
            value["transition_policy"] = self.transition_policy
            value["control_seam"] = {
                "kind": "authenticated-local-unix-datagram",
                "socket_path": str(self._event_socket_path),
                "public_api": False,
                "polling_scheduler": False,
            }
            write_json_with_sha(self.report_path, value)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            closer = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
            closer.sendto(b"CLOSE", str(self._event_socket_path))
            closer.close()
        except OSError:
            pass
        self._event_thread.join(5)
        self._event_socket.close()
        self.controller.close()
        self._write_report()
        if self._event_socket_path.exists():
            self._event_socket_path.unlink()


__all__ = ["R5BFrontendDispatcher"]
