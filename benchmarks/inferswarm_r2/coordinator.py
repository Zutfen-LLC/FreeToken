"""Coordinator for the one frozen R2 local split plan."""

from __future__ import annotations

import json
import multiprocessing
import statistics
import time
from multiprocessing import shared_memory
from pathlib import Path

from .block_service import CONTROL_PROTOCOL, service_entry
from .correctness_support import diagnostic_shared_bytes


class LocalSplitCoordinator:
    def __init__(
        self,
        *,
        plan_path: str,
        model_path: str,
        diagnostic: bool,
        diagnostic_prefill_chunk: int | None = None,
        host_staging_policy: str = "release_after_final_residency",
    ):
        self.plan_path = plan_path
        self.plan = json.loads(Path(plan_path).read_text())
        self.model_path = model_path
        self.diagnostic = diagnostic
        self.context = multiprocessing.get_context("spawn")
        self.host_release_barrier = self.context.Barrier(2)
        self.shared_bytes = diagnostic_shared_bytes(
            self.plan, diagnostic_prefill_chunk if diagnostic else None
        )
        self.segment = shared_memory.SharedMemory(create=True, size=self.shared_bytes)
        self.connections = {}
        self.processes = {}
        self.ready = {}
        self.shutdown_records = {}
        units = {
            unit["id"]: unit
            for node in self.plan["resources"]["nodes"]
            for unit in node["compute_units"]
        }
        for role, unit_id in (("a", "gpu-a"), ("b", "gpu-b")):
            parent, child = self.context.Pipe()
            uuid = units[unit_id]["stable_device_id"]
            process = self.context.Process(
                target=service_entry,
                kwargs={
                    "role": role,
                    "stable_gpu_uuid": uuid,
                    "plan_path": plan_path,
                    "model_path": model_path,
                    "shared_name": self.segment.name,
                    "shared_bytes": self.shared_bytes,
                    "connection": child,
                    "diagnostic": diagnostic,
                    "host_staging_policy": host_staging_policy,
                    "host_release_barrier": self.host_release_barrier,
                },
            )
            process.start()
            self.connections[role] = parent
            self.processes[role] = process
        try:
            for role in ("a", "b"):
                ready = self.connections[role].recv()
                if ready.get("op") == "ERROR":
                    raise RuntimeError(f"Block {role} failed: {ready}")
                if ready.get("plan_digest") != self.plan["digest"]:
                    raise RuntimeError("participant plan digest mismatch")
                self.ready[role] = ready
        except Exception:
            self.shutdown()
            raise

    @staticmethod
    def _message(op: str, **values):
        return {"protocol": CONTROL_PROTOCOL, "op": op, **values}

    def _request(self, role: str, message: dict) -> dict:
        connection = self.connections[role]
        connection.send(message)
        response = connection.recv()
        if response.get("op") == "ERROR":
            raise RuntimeError(f"Block {role} error: {response}")
        return response

    def open(self, session_id: int) -> None:
        for role in ("a", "b"):
            self._request(role, self._message("OPEN_SESSION", session_id=session_id))

    def close_session(self, session_id: int) -> None:
        for role in ("a", "b"):
            self._request(role, self._message("CLOSE_SESSION", session_id=session_id))

    def boundary_step(
        self,
        *,
        session_id: int,
        operation: str,
        token_ids: list[int],
        position: int,
        capture_logits: bool,
        capture_state: bool = False,
    ) -> tuple[int, dict]:
        wall_started = time.perf_counter_ns()
        produced = self._request(
            "a",
            self._message(
                operation.upper(),
                session_id=session_id,
                token_ids=token_ids,
                position=position,
                capture_state=capture_state,
            ),
        )
        consumed = self._request(
            "b",
            self._message(
                "CONSUME_BOUNDARY",
                session_id=session_id,
                operation=operation,
                position=position,
                token_count=produced["token_count"],
                dtype=produced["dtype"],
                layout=produced["layout"],
                payload_bytes=produced["payload_bytes"],
                producer_sha256=produced["producer_sha256"],
                capture_logits=capture_logits,
                capture_state=capture_state,
            ),
        )
        wall_ended = time.perf_counter_ns()
        record = {
            **produced,
            "consumer_sha256": consumed["consumer_sha256"],
            "h2d_ns": consumed["h2d_ns"],
            "block_b_compute_ns": consumed["compute_ns"],
            "block_a_compute_ns": produced["compute_ns"],
            "boundary_transfer_ns": produced["d2h_ns"] + consumed["h2d_ns"],
            "step_wall_ns": wall_ended - wall_started,
        }
        if "logits" in consumed:
            record["logits"] = consumed["logits"]
        for key in (
            "producer_tensors",
            "producer_state",
            "consumer_tensors",
        ):
            if key in produced:
                record[key] = produced[key]
        for key in ("consumer_tensors", "consumer_state", "execution_diagnostic"):
            if key in consumed:
                record[key] = consumed[key]
        return consumed["token_id"], record

    def run_session(
        self,
        *,
        session_id: int,
        prompt_ids: list[int],
        max_new_tokens: int,
        prefill_chunk: int,
        capture_steps: set[int] | None = None,
        capture_prefill_state: bool = False,
        on_token=None,
    ) -> dict:
        capture_steps = capture_steps or set()
        self.open(session_id)
        request_started = time.perf_counter_ns()
        boundaries = []
        generated = []
        first_token_ns = None
        prefill_started = time.perf_counter_ns()
        for start in range(0, len(prompt_ids), prefill_chunk):
            chunk = prompt_ids[start : start + prefill_chunk]
            final = start + len(chunk) == len(prompt_ids)
            token, boundary = self.boundary_step(
                session_id=session_id,
                operation="prefill",
                token_ids=chunk,
                position=start,
                capture_logits=final and 0 in capture_steps,
                capture_state=capture_prefill_state,
            )
            boundaries.append(boundary)
            if final:
                boundary["generated_step"] = 0
                generated.append(token)
                first_token_ns = time.perf_counter_ns()
                if on_token is not None:
                    on_token(0, token, boundary)
        prefill_ended = first_token_ns
        decode_started = time.perf_counter_ns()
        while len(generated) < max_new_tokens:
            step = len(generated)
            position = len(prompt_ids) + step - 1
            token, boundary = self.boundary_step(
                session_id=session_id,
                operation="decode",
                token_ids=[generated[-1]],
                position=position,
                capture_logits=step in capture_steps,
            )
            generated.append(token)
            boundary["generated_step"] = step
            boundaries.append(boundary)
            if on_token is not None:
                on_token(step, token, boundary)
        decode_ended = time.perf_counter_ns()
        self.close_session(session_id)
        decode_latencies = [
            item["step_wall_ns"] for item in boundaries if item["operation"] == "decode"
        ]
        return {
            "session_id": session_id,
            "prompt_token_ids": prompt_ids,
            "generated_token_ids": generated,
            "boundaries": boundaries,
            "prompt_token_count": len(prompt_ids),
            "generated_token_count": len(generated),
            "prefill_wall_ns": prefill_ended - prefill_started,
            "ttft_ns": first_token_ns - request_started,
            "decode_wall_ns": decode_ended - decode_started,
            "total_request_wall_ns": decode_ended - request_started,
            "decode_tokens_per_second": (max_new_tokens - 1)
            / ((decode_ended - decode_started) / 1e9),
            "inter_token_latency_ns": decode_latencies,
            "inter_token_p50_ns": statistics.median(decode_latencies),
            "boundary_bytes": sum(item["payload_bytes"] for item in boundaries),
            "boundary_transfer_ns": sum(
                item["boundary_transfer_ns"] for item in boundaries
            ),
            "block_a_compute_ns": sum(
                item["block_a_compute_ns"] for item in boundaries
            ),
            "block_b_compute_ns": sum(
                item["block_b_compute_ns"] for item in boundaries
            ),
        }

    def reports(self) -> dict:
        return {
            role: self._request(role, self._message("REPORT")) for role in ("a", "b")
        }

    def shutdown(self) -> None:
        awaiting = []
        for role in ("a", "b"):
            process = self.processes.get(role)
            if process is not None and process.is_alive():
                try:
                    self.connections[role].send(self._message("SHUTDOWN"))
                    awaiting.append(role)
                except (EOFError, BrokenPipeError, OSError):
                    pass
        for role in awaiting:
            try:
                if self.connections[role].poll(5):
                    self.shutdown_records[role] = self.connections[role].recv()
            except (EOFError, BrokenPipeError, OSError):
                pass
        for process in self.processes.values():
            process.join(30)
            if process.is_alive():
                process.terminate()
                process.join(10)
        from freetoken.research.host_reclamation import snapshot_host_memory

        for role, process in self.processes.items():
            self.shutdown_records.setdefault(role, {})[
                "P6_worker_shutdown_complete"
            ] = snapshot_host_memory(process.pid)
        self.segment.close()
        self.segment.unlink()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.shutdown()


__all__ = ["LocalSplitCoordinator"]
