"""Temporary ordinary-serving bridge for the static InferSwarm R5A gate."""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from typing import Any

from freetoken.message import AbortMsg, DetokenizeMsg, TokenizeMsg, UserReply
from freetoken.research.r3_planner import require_frozen
from freetoken.research.r5a_serving import StaticServingController
from freetoken.tokenizer.detokenize import DetokenizeManager


class R5AFrontendDispatcher:
    """Convert the existing TokenizeMsg waist to one frozen static controller."""

    def __init__(self, config_path: str, server_config: Any) -> None:
        from benchmarks.inferswarm_r4.r4_plan import load_r4_plan
        from benchmarks.inferswarm_r5a.runtime import (
            realize_network_plan,
            require_clean_exact_source,
        )
        from benchmarks.inferswarm_r5a.strategy import (
            compile_candidate,
            evidence_catalog,
            objective,
            operator_policy,
            planning_problem,
            resource_snapshot,
        )

        path = Path(config_path).resolve()
        raw = json.loads(path.read_text())
        self.config_path = path
        self.repository_root = Path(raw.get("repository_root") or path.parents[2]).resolve()
        environment_path = self._resolve(path, raw["environment"])
        r4_plan_path = self._resolve(path, raw["participant_plan"])
        self.report_path = self._resolve(path, raw["report_out"])
        environment = json.loads(environment_path.read_text())
        require_frozen(environment, "R5A frozen environment")
        expected_sha = environment["implementation_commit"]
        require_clean_exact_source(self.repository_root, expected_sha)
        r4_plan = load_r4_plan(r4_plan_path)
        if r4_plan.get("provenance", {}).get("r4", {}).get("producer_sha") != expected_sha:
            raise RuntimeError("participant plan producer differs from frozen R5A source")
        serving_records = []
        if raw.get("serving_evidence"):
            records_path = self._resolve(path, raw["serving_evidence"])
            records = json.loads(records_path.read_text())
            require_frozen(records, "R5A serving evidence catalog derivative")
            serving_records = records.get("records", [])
        problem = planning_problem(expected_sha)
        snapshot = resource_snapshot(environment)
        policy = operator_policy(expected_sha)
        declared_objective = objective(expected_sha, raw.get("objective", "ttft_ms"))
        evidence = evidence_catalog(
            expected_sha, self.repository_root, serving_records=serving_records
        )

        def compiler(evaluation):
            return compile_candidate(dict(evaluation), r4_plan=r4_plan)

        def realizer(execution_plan):
            if execution_plan["strategy_realization"]["path"] != "r4-persistent-boundary":
                raise RuntimeError(
                    "this bounded R5A bridge currently realizes only the selected legal "
                    "two-node backend-native candidate; collect/refresh evidence first"
                )
            return realize_network_plan(
                dict(execution_plan),
                r4_plan=r4_plan,
                model_path=server_config.model_path,
                peer_host=raw["peer_host"],
                peer_port=int(raw.get("peer_port", 18485)),
                diagnostic=bool(raw.get("diagnostic", False)),
            )

        self.controller = StaticServingController(
            problem=problem,
            snapshot=snapshot,
            policy=policy,
            objective=declared_objective,
            evidence_catalog=evidence,
            compiler=compiler,
            realizer=realizer,
            override_candidate_id=raw.get("override_candidate_id"),
        )
        self._detokenizer = None
        self._detokenizer_lock = threading.Lock()
        self._tasks: set[asyncio.Task] = set()
        self._write_report()

    @staticmethod
    def _resolve(config_path: Path, value: str) -> Path:
        path = Path(value)
        return path.resolve() if path.is_absolute() else (config_path.parent / path).resolve()

    def _ensure_detokenizer(self, state) -> DetokenizeManager:
        with self._detokenizer_lock:
            if self._detokenizer is None:
                tokenizer = state.frontend_tokenizer().tokenizer
                self._detokenizer = DetokenizeManager(tokenizer)
            return self._detokenizer

    async def submit(self, msg, state) -> None:
        if isinstance(msg, AbortMsg):
            return
        if not isinstance(msg, TokenizeMsg):
            raise RuntimeError("R5A static serving accepts generation requests only")
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

            def on_token(step, token_id, boundary):
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
                    # R4's report owns its coordinator-clock prefill wall. Do
                    # not mislabel that end-to-end interval as FreeToken's
                    # CUDA-event-only ``UserReply.prefill`` measurement.
                    prefill=None,
                )
                loop.call_soon_threadsafe(self._deliver, state, reply)

            self.controller.serve_tokens(
                session_id=msg.uid,
                prompt_token_ids=prompt_ids,
                max_new_tokens=maximum,
                on_token=on_token,
            )
            self._write_report()
        except Exception as exc:  # noqa: BLE001 - request must receive a terminal error
            reply = UserReply(
                uid=msg.uid,
                incremental_output="",
                finished=True,
                error=f"InferSwarm R5A request failed: {type(exc).__name__}: {exc}",
                error_code="inferswarm_r5a_failure",
            )
            loop.call_soon_threadsafe(self._deliver, state, reply)

    def _write_report(self) -> None:
        from freetoken.research.n0_model_block import write_json_with_sha

        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        value = self.controller.report()
        value["execution_plan"] = self.controller.execution_plan
        value["planner_decision"] = self.controller.decision
        write_json_with_sha(self.report_path, value)

    def close(self) -> None:
        self.controller.close()
        self._write_report()


__all__ = ["R5AFrontendDispatcher"]
