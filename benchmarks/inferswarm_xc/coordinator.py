"""CPU-only Coordinator serving waist for the InferSwarm external-Coordinator
proof (inferswarm #67).

Runs on inferswarm00.  Accepts the ordinary OpenAI-compatible
``/v1/chat/completions`` request shape, tokenizes it with the checkpoint's
own tokenizer files (config/tokenizer metadata only — no model weights), and
drives the accepted ``EpochServingController`` whose realizer is the remote
``xc_wire`` seam.  This is the narrowest faithful adaptation of the accepted
R5B serving waist to a host with no Torch and no CUDA: same request path,
same sampling-field resolution, same epoch/commit semantics — but
``freetoken`` server/engine imports are absent by construction.

Research-internal: not a public daemon.  One Coordinator, one bounded proof.
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

from .cpu_only import load_json_with_sha, printable_increment, write_json_with_sha

DEFAULT_MAX_OUTPUT_TOKENS = 32768


def _require_clean_exact_source(repository_root: Path, expected_sha: str) -> None:
    """Refuse to coordinate from a drifted or dirty source tree (CPU-only).

    Mirrors the accepted R5A/R5B ``require_clean_exact_source`` gate without
    importing the torch-carrying benchmarks module.
    """
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
        raise RuntimeError("external-Coordinator serving refuses a dirty source tree")


def _ensure_repo_on_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    for entry in (str(repo_root / "benchmarks"),):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    return repo_root


def _resolve(base: Path, value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else (base / candidate).resolve()


def build_controller(
    *,
    environment: Mapping[str, Any],
    r4_plan: Mapping[str, Any],
    local_plan: Mapping[str, Any],
    serving_records: Mapping[str, Any],
    repo_root: Path,
    node_agent_host: str,
    node_agent_port: int,
    scope_id: str,
    objective_metric: str,
) -> EpochServingController:
    """Assemble the accepted epoch controller with a remote realizer.

    Planning physically executes on this host; realization and execution are
    authorized out to the Node agent over the research wire.
    """
    _ensure_repo_on_path()
    from benchmarks.inferswarm_r5a.strategy import (
        evidence_catalog,
        objective as make_objective,
    )
    from benchmarks.inferswarm_r5b.strategy import (
        QwenTokenBoundaryStrategy,
        compile_candidate,
        operator_policy,
        planning_problem,
        policy_evaluator,
        snapshot_with_availability,
        transition_policy,
    )

    expected_sha = environment["implementation_commit"]
    environment = dict(environment)

    def compiler(evaluation):
        return compile_candidate(
            dict(evaluation), r4_plan=dict(r4_plan), local_plan=dict(local_plan)
        )

    realizer = make_remote_realizer(
        host=node_agent_host, port=node_agent_port, scope_id=scope_id
    )

    return EpochServingController(
        problem=planning_problem(expected_sha),
        initial_snapshot=snapshot_with_availability(environment, gpu_a1_available=False),
        policy=operator_policy(expected_sha),
        objective=make_objective(expected_sha, objective_metric),
        evidence_catalog=evidence_catalog(
            expected_sha, repo_root, serving_records=records_list(serving_records)
        ),
        compiler=compiler,
        realizer=realizer,
        transition_strategy=QwenTokenBoundaryStrategy(),
        transition_policy=policy_evaluator(transition_policy(expected_sha)),
    )


def records_list(serving_records: Mapping[str, Any]) -> list[dict[str, Any]]:
    value = serving_records.get("records", [])
    return list(value) if isinstance(value, list) else []


class _DecodeState:
    """Per-session incremental decode state (token count + sent prefix)."""

    def __init__(self) -> None:
        self.token_count = 0
        self.sent_prefix = ""


class CoordinatorRuntime:
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
        from benchmarks.inferswarm_r4.r4_plan import load_r4_plan

        r4_plan = load_r4_plan(Path(_resolve(base, str(config["participant_plan"]))))
        local_plan = json.loads(
            _resolve(base, str(config["local_participant_plan"])).read_text()
        )
        serving_records = load_json_with_sha(
            _resolve(base, str(config["serving_evidence"]))
        )
        repo_root = _ensure_repo_on_path()
        self.controller = build_controller(
            environment=environment,
            r4_plan=r4_plan,
            local_plan=local_plan,
            serving_records=serving_records,
            repo_root=repo_root,
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

    # ---- tokenizer (config/tokenizer metadata only; no weights) ----------

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
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **ctk,
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

    # ---- incremental detokenization (no freetoken imports) ---------------

    def _decode_incremental(
        self, session_id: int, token_ids: list[int], *, finished: bool
    ) -> str:
        state = self._decode_states.setdefault(session_id, _DecodeState())
        tokenizer = self._load_tokenizer()
        whole = printable_increment(tokenizer, token_ids, finished=finished)
        fresh = whole[len(state.sent_prefix) :] if whole.startswith(state.sent_prefix) else ""
        state.sent_prefix = whole
        return fresh

    # ---- ordinary serving path -------------------------------------------

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
            ids = prompt_ids + [
                event["token_id"] for event in events
            ] + [int(token_id)]
            text = self._decode_incremental(session_id, ids, finished=False)
            events.append(
                {
                    "step": step,
                    "token_id": int(token_id),
                    "epoch_id": commit.get("epoch_id"),
                    "plan_digest": commit.get("plan_digest"),
                    "position": commit.get("position"),
                    "committed_at_ns": commit.get("committed_at_ns"),
                    "text": text,
                }
            )

        inject_after_step = body.get("inferswarm_fencing_arm_after_step")
        injections: list[dict[str, Any]] = []

        def after_commit(step: int, ctrl: Any) -> None:
            # Controlled negative arm: after one real authorized operation
            # commits, route a stale/duplicate result carrying an already-used
            # position and a retired epoch id through the same acceptance path
            # and prove mechanical rejection without mutating the ledger.
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
                        position=ledger.committed_position - 1,  # duplicate
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
            "schema": "inferswarm.xc.coordinator-request/1",
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
                "kind": "cpu-only-external-coordinator",
                "scope_id": self.scope_id,
                "instance_id": self.instance_id,
                "node_agent": f"{self.node_agent_host}:{self.node_agent_port}",
                "requests": list(self.request_log),
            }
            write_json_with_sha(self.report_path, value)

    def close(self) -> None:
        self.controller.close()
        self.write_report()


def make_handler(runtime: CoordinatorRuntime):
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
            stream = bool(body.get("stream"))
            try:
                outcome = runtime.handle_chat(body)
            except Exception as exc:
                runtime.write_report()
                self._send_json(
                    500,
                    {
                        "error": (
                            f"external-coordinator request failed: "
                            f"{type(exc).__name__}: {exc}"
                        )
                    },
                )
                return
            if not stream:
                self._send_json(
                    200,
                    {
                        "id": f"chatcmpl-{runtime.instance_id}",
                        "object": "chat.completion",
                        "model": body.get("model"),
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": outcome["content"],
                                },
                                "finish_reason": "length",
                            }
                        ],
                        "usage": outcome["usage"],
                    },
                )
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            final_chunk_sent = False
            for event in outcome["record"]["token_events"]:
                if not event["text"]:
                    continue
                chunk = {
                    "id": f"chatcmpl-{runtime.instance_id}",
                    "object": "chat.completion.chunk",
                    "model": body.get("model"),
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": event["text"]},
                            "finish_reason": None,
                        }
                    ],
                }
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            final = {
                "id": f"chatcmpl-{runtime.instance_id}",
                "object": "chat.completion.chunk",
                "model": body.get("model"),
                "choices": [
                    {"index": 0, "delta": {}, "finish_reason": "length"}
                ],
                "usage": outcome["usage"],
            }
            self.wfile.write(f"data: {json.dumps(final)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            final_chunk_sent = True

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
    runtime = CoordinatorRuntime(config)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(runtime))

    import signal

    def _graceful_shutdown(signum, frame):  # noqa: ARG001
        # Retire/reclaim the remote epoch runtime and persist the final
        # report before the process exits; a hard kill would leave the
        # report claiming an ACTIVE epoch with no reclamation record.
        server.shutdown()

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
