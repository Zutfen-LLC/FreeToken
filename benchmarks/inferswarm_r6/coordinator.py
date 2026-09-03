"""R6 CPU-only Coordinator serving waist (dense Gemma variant, #65).

The accepted #67 external-Coordinator waist with exactly two changes:
the strategy seam binds the R6 dense Gemma strategy, and the compiler
embeds the frozen 3-stage chain participant plan.  Everything else —
request shape, tokenizer purity, epoch/commit semantics, fencing arm,
reporting — is the accepted path unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping

from freetoken.research.r5b_epochs import EpochServingController
from freetoken.research.xc_coordinator import make_remote_realizer

try:
    from benchmarks.inferswarm_xc.cpu_only import (
        load_json_with_sha,
        printable_increment,
        write_json_with_sha,
    )
except ModuleNotFoundError:
    from inferswarm_xc.cpu_only import (  # type: ignore
        load_json_with_sha,
        printable_increment,
        write_json_with_sha,
    )

DEFAULT_MAX_OUTPUT_TOKENS = 32768


def _require_clean_exact_source(repository_root: Path, expected_sha: str) -> None:
    import subprocess

    actual = subprocess.check_output(
        ["git", "-C", str(repository_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != expected_sha:
        raise RuntimeError(f"source SHA drift: {actual} != frozen {expected_sha}")
    status = subprocess.check_output(
        ["git", "-C", str(repository_root), "status", "--porcelain"], text=True
    )
    if status:
        raise RuntimeError("R6 Coordinator serving refuses a dirty source tree")


def _ensure_repo_on_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    for entry in (str(repo_root), str(repo_root / "benchmarks")):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    return repo_root


def _resolve(base: Path, value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else (base / candidate).resolve()


def build_controller(
    *,
    environment: Mapping[str, Any],
    chain_plan: Mapping[str, Any],
    serving_records: Mapping[str, Any],
    repo_root: Path,
    node_agent_host: str,
    node_agent_port: int,
    scope_id: str,
    objective_metric: str,
) -> EpochServingController:
    _ensure_repo_on_path()
    from benchmarks.inferswarm_r6.xc_strategy import (
        GemmaTokenBoundaryStrategy,
        compile_candidate,
        operator_policy,
        planning_problem,
    )
    from freetoken.research.r3_planner import freeze

    expected_sha = environment["implementation_commit"]

    def compiler(evaluation):
        return compile_candidate(dict(evaluation), chain_plan=dict(chain_plan))

    realizer = make_remote_realizer(
        host=node_agent_host, port=node_agent_port, scope_id=scope_id
    )

    # Evidence catalog: the R6 chain validation run is the only applicable
    # ranking evidence for this shape; recorded with EXACT context.  Where
    # absent, candidates surface FEASIBLE_UNRANKED (honest no-evidence).
    records = list(serving_records.get("records", []) or [])

    return EpochServingController(
        problem=planning_problem(expected_sha),
        initial_snapshot=_r6_snapshot(environment),
        policy=operator_policy(expected_sha),
        objective=_r6_objective(expected_sha, objective_metric),
        evidence_catalog=freeze(
            {
                "schema": "inferswarm.r6.evidence-catalog/1",
                "implementation_commit": expected_sha,
                "records": records,
            }
        ),
        compiler=compiler,
        realizer=realizer,
        transition_strategy=GemmaTokenBoundaryStrategy(),
        transition_policy=_r6_transition_policy(expected_sha),
    )


def _r6_snapshot(environment: Mapping[str, Any]) -> dict[str, Any]:
    """Resource snapshot for the frozen R6 topology (3 GPUs, 2 nodes)."""
    from freetoken.research.r3_planner import freeze

    def unit(unit_id, record, capabilities):
        return {
            "id": unit_id,
            "stable_device_id": record["uuid"],
            "pci_bdf": record["pci_bdf"],
            "memory_resource_id": f"{unit_id}.vram",
            "availability": record.get("availability", "AVAILABLE"),
            "integrity_eligible": record.get("integrity_eligible", True),
            "capabilities": capabilities,
        }

    a = environment["node_a"]
    b = environment["node_b"]
    a0, a1 = a["gpus"][0], a["gpus"][1]
    b0 = b["gpus"][0]
    return freeze(
        {
            "schema": "inferswarm.r6.resource-evidence-snapshot/1",
            "implementation_commit": environment["implementation_commit"],
            "environment_digest": environment.get("digest"),
            "evidence_context": {
                "runtime_context": environment["runtime_context"],
                "network_context": environment["network_context"],
            },
            "nodes": [
                {
                    "id": a["node_id"],
                    "compute_units": [
                        unit("gpu.node-a.0", a0,
                             ["freetoken-resident-stage-first-v1"]),
                        unit("gpu.node-a.1", a1,
                             ["freetoken-resident-stage-middle-v1"]),
                    ],
                    "memory_resources": [
                        {
                            "id": "gpu.node-a.0.vram",
                            "kind": "accelerator-vram-a0",
                            "capacity_bytes": a0["vram_total_bytes"],
                            "reservation_bytes": a0.get("reservation_bytes", 0),
                        },
                        {
                            "id": "gpu.node-a.1.vram",
                            "kind": "accelerator-vram-a1",
                            "capacity_bytes": a1["vram_total_bytes"],
                            "reservation_bytes": a1.get("reservation_bytes", 0),
                        },
                    ],
                },
                {
                    "id": b["node_id"],
                    "compute_units": [
                        unit("gpu.node-b.0", b0,
                             ["freetoken-resident-stage-last-v1"]),
                    ],
                    "memory_resources": [
                        {
                            "id": "gpu.node-b.0.vram",
                            "kind": "accelerator-vram-b0",
                            "capacity_bytes": b0["vram_total_bytes"],
                            "reservation_bytes": b0.get("reservation_bytes", 0),
                        },
                    ],
                },
            ],
            "links": [
                {
                    "id": "path.node-a.local-staging",
                    "source_memory_resource_id": "gpu.node-a.0.vram",
                    "target_memory_resource_id": "gpu.node-a.1.vram",
                    "available": True,
                    "capabilities": ["freetoken-static-boundary-v1"],
                },
                {
                    "id": environment["network"]["link_id"],
                    "source_memory_resource_id": "gpu.node-a.1.vram",
                    "target_memory_resource_id": "gpu.node-b.0.vram",
                    "available": environment["network"].get("available", True),
                    "capabilities": ["freetoken-static-boundary-v1"],
                    "negotiated_mbps": environment["network"]["negotiated_mbps"],
                },
            ],
        }
    )


def _r6_objective(implementation_commit: str, metric: str = "ttft_ms") -> dict:
    from freetoken.research.r3_planner import freeze

    if metric not in ("ttft_ms", "complete_request_wall_ms", "decode_tok_s"):
        raise ValueError(f"unsupported frozen R6 objective {metric!r}")
    maximize = metric == "decode_tok_s"
    return freeze(
        {
            "schema": "inferswarm.r6.objective/1",
            "implementation_commit": implementation_commit,
            "id": f"r6-dense-serving-{metric}",
            "metric": metric,
            "direction": "MAXIMIZE" if maximize else "MINIMIZE",
            "unit": "tok/s" if maximize else "ms",
            "statistic": "single-run",
            "evidence_context": {
                "model_revision": "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7",
                "workload_geometry": "r6-dense-3stage-chain",
            },
        }
    )


def _r6_transition_policy(implementation_commit: str) -> dict:
    from freetoken.research.r3_planner import freeze

    return freeze(
        {
            "schema": "inferswarm.r6.transition-policy/1",
            "implementation_commit": implementation_commit,
            "correctness_and_feasibility_first": True,
            "operator_policy": "automatic within this bounded physical campaign",
            "single_legal_shape": True,
            "reason": "R6 freezes one legal dense shape per fabric; no "
            "economically-ranked reconfiguration arm is exercised",
        }
    )


class _DecodeState:
    def __init__(self) -> None:
        self.token_count = 0
        self.sent_prefix = ""


class R6CoordinatorRuntime:
    def __init__(self, config: Mapping[str, Any]) -> None:
        base = Path(str(config["config_path"])).resolve().parent
        self.report_path = _resolve(base, str(config["report_out"]))
        self.scope_id = str(config["scope_id"])
        self.tokenizer_path = str(config["tokenizer_path"])
        self.node_agent_host = str(config["node_agent_host"])
        self.node_agent_port = int(config["node_agent_port"])
        self.repository_root = Path(str(config["repository_root"])).resolve()
        environment = load_json_with_sha(_resolve(base, str(config["environment"])))
        _require_clean_exact_source(
            self.repository_root, environment["implementation_commit"]
        )
        chain_plan = json.loads(
            _resolve(base, str(config["chain_plan"])).read_text()
        )
        serving_records = load_json_with_sha(
            _resolve(base, str(config.get("serving_evidence")))
        ) if config.get("serving_evidence") else {"records": []}
        self.controller = build_controller(
            environment=environment,
            chain_plan=chain_plan,
            serving_records=serving_records,
            repo_root=_ensure_repo_on_path(),
            node_agent_host=self.node_agent_host,
            node_agent_port=self.node_agent_port,
            scope_id=self.scope_id,
            objective_metric=str(config.get("objective", "ttft_ms")),
        )
        self._report_lock = threading.Lock()
        self._tokenizer = None
        self._tokenizer_lock = threading.Lock()
        self._decode_states: dict[int, _DecodeState] = {}
        self.instance_id = uuid.uuid4().hex[:12]
        self.request_log: list[dict[str, Any]] = []

    def _load_tokenizer(self):
        with self._tokenizer_lock:
            if self._tokenizer is None:
                from transformers import AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.tokenizer_path, trust_remote_code=False
                )
            return self._tokenizer

    def _render_and_tokenize(self, body: Mapping[str, Any]) -> list[int]:
        tokenizer = self._load_tokenizer()
        messages = body["messages"]
        ctk = dict(body.get("chat_template_kwargs") or {})
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **ctk
        )
        return tokenizer.encode(prompt, add_special_tokens=False)

    @staticmethod
    def _sampling_of(body: Mapping[str, Any]) -> dict[str, Any]:
        temperature = body.get("temperature")
        top_k = body.get("top_k")
        top_p = body.get("top_p")

        def pick(value, fallback):
            return value if value is not None else fallback

        return {
            "temperature": float(pick(temperature, 0.0)),
            "top_k": int(pick(top_k, -1)),
            "top_p": float(pick(top_p, 1.0)),
        }

    def _decode_incremental(
        self, session_id: int, token_ids: list[int], *, finished: bool
    ) -> str:
        state = self._decode_states.setdefault(session_id, _DecodeState())
        tokenizer = self._load_tokenizer()
        whole = printable_increment(tokenizer, token_ids, finished=finished)
        fresh = (
            whole[len(state.sent_prefix):] if whole.startswith(state.sent_prefix) else ""
        )
        state.sent_prefix = whole
        return fresh

    def handle_chat(self, body: Mapping[str, Any]) -> dict[str, Any]:
        started = time.time_ns()
        prompt_ids = self._render_and_tokenize(body)
        maximum = int(body.get("max_tokens") or DEFAULT_MAX_OUTPUT_TOKENS)
        if maximum < 1:
            raise ValueError("max_tokens must be at least 1")
        sampling = self._sampling_of(body)
        session_id = len(self.request_log) + 1
        events: list[dict[str, Any]] = []

        def on_token(step: int, token_id: int, commit: Mapping[str, Any]) -> None:
            ids = prompt_ids + [event["token_id"] for event in events] + [int(token_id)]
            text = self._decode_incremental(session_id, ids, finished=False)
            events.append(
                {
                    "step": step,
                    "token_id": int(token_id),
                    "epoch_id": commit.get("epoch_id"),
                    "generation": commit.get("generation"),
                    "plan_digest": commit.get("plan_digest"),
                    "position": commit.get("position"),
                    "committed_at_ns": commit.get("committed_at_ns"),
                    "text": text,
                }
            )

        inject_after_step = body.get("inferswarm_fencing_arm_after_step")
        injections: list[dict[str, Any]] = []

        def after_commit(step: int, ctrl: Any) -> None:
            if inject_after_step is None or step != int(inject_after_step):
                return
            ledger = ctrl._sessions.get(session_id)
            if ledger is None:
                return
            active = ctrl.active_epoch
            injections.append(
                {
                    "injection": "CONTROLLED_LATE_REAL_SERVING_RESULT",
                    "accepted": ctrl.inject_late_result(
                        epoch_id=active.epoch_id,
                        plan_digest=active.execution_plan["digest"],
                        session=ledger,
                        position=ledger.committed_position - 1,
                        token_id=int(events[-1]["token_id"]) if events else -1,
                    ),
                    "reason_attempted": "duplicate already-committed position",
                }
            )
            injections.append(
                {
                    "injection": "CONTROLLED_STALE_EPOCH_RESULT",
                    "accepted": ctrl.inject_late_result(
                        epoch_id=active.epoch_id + "-retired",
                        plan_digest=active.execution_plan["digest"],
                        session=ledger,
                        position=ledger.committed_position,
                        token_id=424242,
                    ),
                    "reason_attempted": "retired/stale epoch identity",
                }
            )

        completed = self.controller.serve_tokens(
            session_id=session_id,
            prompt_token_ids=prompt_ids,
            max_new_tokens=maximum,
            sampling_inputs=sampling,
            on_token=on_token,
            after_commit=after_commit if inject_after_step is not None else None,
        )
        all_ids = prompt_ids + [event["token_id"] for event in events]
        tail = self._decode_incremental(session_id, all_ids, finished=True)
        content = "".join(event["text"] for event in events) + tail
        record = {
            "schema": "inferswarm.r6.coordinator-request/1",
            "session_id": session_id,
            "prompt_token_ids": prompt_ids,
            "generated_token_ids": completed["generated_token_ids"],
            "committed_epoch_ids": completed["committed_epoch_ids"],
            "committed_plan_digests": completed["committed_plan_digests"],
            "request_wall_ns": time.time_ns() - started,
            "token_events": events,
            "fencing_arm_injections": injections,
        }
        self.request_log.append(record)
        self._decode_states.pop(session_id, None)
        self.write_report()
        return {
            "record": record,
            "content": content,
            "usage": {
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": maximum,
                "total_tokens": len(prompt_ids) + maximum,
            },
        }

    def write_report(self) -> None:
        with self._report_lock:
            value = self.controller.report()
            value["coordinator_scope"] = {
                "kind": "cpu-only-external-coordinator-r6-dense",
                "scope_id": self.scope_id,
                "instance_id": self.instance_id,
                "node_agent": f"{self.node_agent_host}:{self.node_agent_port}",
                "requests": list(self.request_log),
            }
            write_json_with_sha(self.report_path, value)

    def close(self) -> None:
        self.controller.close()
        self.write_report()


def make_handler(runtime: R6CoordinatorRuntime):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send_json(self, code: int, value: Any) -> None:
            data = json.dumps(value).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):  # noqa: N802
            if self.path == "/health":
                self._send_json(
                    200, {"status": "ok", "instance_id": runtime.instance_id}
                )
                return
            self._send_json(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802
            if self.path != "/v1/chat/completions":
                self._send_json(404, {"error": "not found"})
                return
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > 4 * 1024 * 1024:
                self._send_json(400, {"error": "invalid body length"})
                return
            body = json.loads(self.rfile.read(length))
            try:
                outcome = runtime.handle_chat(body)
            except Exception as exc:
                runtime.write_report()
                self._send_json(
                    500,
                    {
                        "error": (
                            f"R6 coordinator request failed: "
                            f"{type(exc).__name__}: {exc}"
                        )
                    },
                )
                return
            self._send_json(
                200,
                {
                    "id": f"chatcmpl-{runtime.instance_id}",
                    "object": "chat.completion",
                    "model": body.get("model"),
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant",
                                        "content": outcome["content"]},
                            "finish_reason": "length",
                        }
                    ],
                    "usage": outcome["usage"],
                },
            )

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            pass

    return Handler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18080)
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text())
    config["config_path"] = args.config
    runtime = R6CoordinatorRuntime(config)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(runtime))

    import signal

    def _graceful_shutdown(signum, frame):  # noqa: ARG001
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _graceful_shutdown)
    signal.signal(signal.SIGINT, _graceful_shutdown)
    print(
        json.dumps(
            {
                "coordinator_listening": True,
                "host": args.host,
                "port": args.port,
                "scope_id": runtime.scope_id,
                "instance_id": runtime.instance_id,
                "variant": "r6-dense",
            }
        ),
        flush=True,
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()
        runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
