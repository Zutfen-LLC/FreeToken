from __future__ import annotations

import logging
import multiprocessing as mp
import os
import sys
from dataclasses import replace
from typing import TYPE_CHECKING

from freetoken.distributed import DistributedInfo
from freetoken.utils import init_logger

if TYPE_CHECKING:
    from .args import ServerArgs
    from .supervisor import BackendHandle


def _report_startup_error(ack_queue: mp.Queue, exc: BaseException) -> None:
    """Tell the parent WHY this worker is dying — push an ("error", reason) ack before it exits,
    so the supervisor reports the real cause (e.g. a config ValueError) instead of the generic
    "backend worker … exited during load". Best-effort and flushed (close + join_thread), since
    the process is about to terminate; a failure to report must never mask the original error."""
    try:
        ack_queue.put(("error", f"{type(exc).__name__}: {exc}"))
        ack_queue.close()
        ack_queue.join_thread()
    except Exception:  # noqa: BLE001 -- reporting is a nicety; never shadow the real exception
        pass


def _detach_process_group() -> None:
    """Shell mode only: move this worker out of the terminal's foreground process group.

    The shell binds ^C to "cancel this turn", but a terminal delivers SIGINT to the whole
    foreground group — which, with the engine running in this same process, includes the
    workers. They would take the same ^C and exit (``_run_scheduler`` below stops gracefully on
    KeyboardInterrupt), leaving the shell chatting with a dead engine. Nothing depends on the
    signal reaching them: the parent tears the workers down explicitly on every stop path
    (uvicorn's lifespan, plus the SIGTERM/SIGHUP handler and the reap backstop in api_server).

    ``ft serve`` keeps the default — uvicorn owns ^C there, and the group-wide delivery is part
    of how it stops."""
    try:
        os.setpgrp()
    except OSError:  # no job control (already a group leader / unusual environment)
        pass


def _run_tokenize_worker(detach: bool, **kwargs) -> None:
    """Module-level so it survives the spawn pickle; exists only to detach the group first."""
    if detach:
        _detach_process_group()
    from freetoken.tokenizer import tokenize_worker

    tokenize_worker(**kwargs)


def _run_scheduler(args: ServerArgs, ack_queue: mp.Queue[str]) -> None:
    if args.shell_mode:
        _detach_process_group()

    # published (not bound) here: the engine binds it after the allocator setup
    from freetoken.gpu_select import set_assigned_gpu

    # resolved UUIDs when we have them, the raw --gpu entries when NVML could not resolve them, else one CUDA ordinal per rank
    targets = (
        args.gpu_assigned or args.gpu or tuple(str(r) for r in range(args.tp_info.size))
    )
    set_assigned_gpu(targets[args.tp_info.rank])

    import torch
    from freetoken.scheduler import Scheduler

    if args.tp_info.is_primary():
        from freetoken.utils.progress import set_progress_sink

        set_progress_sink(
            lambda desc, done, total: ack_queue.put(("progress", desc, done, total))
        )

    with torch.inference_mode():
        try:
            scheduler = Scheduler(args)
            scheduler.sync_all_ranks()
        except Exception as exc:  # noqa: BLE001 -- surface the reason, then let it propagate
            # A startup failure (bad config, OOM, corrupt weights) would otherwise reach the
            # parent only as a dead process -> a generic "exited during load". Push the real
            # reason first so the supervisor (and the desktop failure modal) can surface it;
            # the traceback still prints and the process still exits non-zero.
            _report_startup_error(ack_queue, exc)
            raise

        if args.tp_info.is_primary():
            # Report the real per-unit cache VRAM costs (KV/expert/mamba), the device-wide free
            # VRAM, and the per-pool rebuild floors before the ready ack, so the supervisor has
            # them (and the desktop's slider bounds) by the time the gate flips. Optional +
            # best-effort: a failure here must never keep the model from serving, and older
            # consumers ignore ("meta", …).
            try:
                from freetoken.engine.runtime_report import build_runtime_report
                from freetoken.kvcache.cache_status import compute_cache_status_meta

                meta = compute_cache_status_meta(scheduler.engine)
                # the parent must not touch CUDA to learn this
                meta["gpus"] = scheduler.gpus
                # Resolved (not requested) engine configuration, for /v1/instrumentation.
                # build_runtime_report swallows its own failures, so it cannot cost readiness.
                meta["runtime_config"] = build_runtime_report(scheduler.engine)
                ack_queue.put(("meta", meta))
            except Exception:  # noqa: BLE001 -- metadata is a nicety; readiness is not
                pass
            ack_queue.put("Scheduler is ready")
            # The supervisor stops draining ack_queue once ready, so uninstall the sink:
            # runtime cache rebuilds re-run the graph capture (which emits progress) and
            # would otherwise push onto a queue nobody reads for the server's lifetime.
            set_progress_sink(None)

        if args.silent_output:
            logging.disable(logging.INFO)

        try:
            scheduler.run_forever()
        except KeyboardInterrupt:
            logger = init_logger(__name__)
            if args.tp_info.is_primary():
                print()  # for a clean newline after ^C
                logger.info("Scheduler exiting gracefully...")
            scheduler.shutdown()


def _resolve_server_gpu_args(server_args: ServerArgs) -> ServerArgs:
    """Resolve primary/secondary selectors before workers spawn, without touching CUDA."""
    from freetoken.gpu_select import resolve_gpu_uuids

    primary_assigned = server_args.gpu_assigned
    if server_args.gpu:
        primary_assigned = resolve_gpu_uuids(server_args.gpu)

    secondary_assigned = server_args.inferswarm_secondary_gpu_assigned
    secondary = server_args.inferswarm_secondary_gpu
    if secondary is not None:
        try:
            resolved = resolve_gpu_uuids([secondary])
        except ValueError as exc:
            message = str(exc).replace("--gpu", "--inferswarm-secondary-gpu", 1)
            raise ValueError(message) from exc
        secondary_assigned = resolved[0] if resolved else None

        # With an explicit primary, NVML can reject aliases of the same physical card before
        # CUDA/model initialization.  The worker repeats this check against CUDA identities,
        # which also covers an implicit primary and hosts where NVML is unavailable.
        if primary_assigned and secondary_assigned:
            primary_uuid = primary_assigned[0]
            if primary_uuid.upper() == secondary_assigned.upper():
                raise ValueError(
                    f"--inferswarm-secondary-gpu resolves to the same physical GPU as the primary ({primary_uuid})"
                )

    d3_assigned = []
    for label, selector in (("a", server_args.inferswarm_d3_worker_a_gpu), ("b", server_args.inferswarm_d3_worker_b_gpu)):
        if selector is None:
            d3_assigned.append(None)
            continue
        try:
            resolved = resolve_gpu_uuids([selector])
        except ValueError as exc:
            raise ValueError(str(exc).replace("--gpu", f"--inferswarm-d3-worker-{label}-gpu", 1)) from exc
        d3_assigned.append(resolved[0] if resolved else None)
    known = [
        value
        for value in (*tuple(primary_assigned or ()), secondary_assigned, *d3_assigned)
        if value is not None
    ]
    if len({value.upper() for value in known}) != len(known):
        raise ValueError("InferSwarm secondary/D3 worker selectors must name distinct physical GPUs")

    return replace(
        server_args,
        gpu_assigned=primary_assigned,
        inferswarm_secondary_gpu_assigned=secondary_assigned,
        inferswarm_d3_worker_a_gpu_assigned=d3_assigned[0],
        inferswarm_d3_worker_b_gpu_assigned=d3_assigned[1],
    )


def launch_server(
    run_shell: bool = False,
    argv: list[str] | None = None,
    prog: str | None = None,
) -> None:
    from .api_server import run_api_server
    from .args import parse_args

    server_args, run_shell = parse_args(
        sys.argv[1:] if argv is None else argv,
        run_shell,
        prog=prog,
    )
    logger = init_logger(__name__, "initializer")

    if server_args.gpu or server_args.inferswarm_secondary_gpu is not None:
        # Resolve here so a typo is one clear error before any worker spawns. CUDA visibility,
        # identity, and peer capability are verified again inside the bound engine worker.
        try:
            server_args = _resolve_server_gpu_args(server_args)
        except ValueError as exc:
            raise SystemExit(f"{prog or 'ft serve'}: error: {exc}") from exc
    if server_args.gpu:
        logger.info(
            f"--gpu {','.join(server_args.gpu)} -> {', '.join(server_args.gpu_assigned) if server_args.gpu_assigned else 'resolved at CUDA init (no NVML)'}"
        )
    if server_args.inferswarm_secondary_gpu is not None:
        logger.info(
            f"--inferswarm-secondary-gpu {server_args.inferswarm_secondary_gpu} -> {server_args.inferswarm_secondary_gpu_assigned or 'resolved at CUDA init (no NVML)'}"
        )
    if server_args.inferswarm_placement is not None:
        logger.info(
            "--inferswarm-placement configured (host-local path withheld from provenance)"
        )
    if server_args.inferswarm_remote_decode:
        logger.info(
            f"--inferswarm-remote-decode enabled (mode={server_args.inferswarm_remote_mode}, persistent-stream explicit host-staged transport)"
        )
    if server_args.inferswarm_experimental_d2_graph_remote:
        logger.info(
            "--inferswarm-experimental-d2-graph-remote enabled "
            "(post-NO-GO unified multi-device CUDA graph experiment)"
        )

    def start_subprocess() -> "BackendHandle":
        import multiprocessing as mp

        from .supervisor import BackendHandle

        mp.set_start_method("spawn", force=True)
        detach = server_args.shell_mode  # see _detach_process_group

        world_size = server_args.tp_info.size
        ack_queue: mp.Queue = mp.Queue()
        processes: list[mp.Process] = []

        for i in range(world_size):
            new_args = replace(server_args, tp_info=DistributedInfo(i, world_size))
            p = mp.Process(
                target=_run_scheduler,
                args=(new_args, ack_queue),
                daemon=False,
                name=f"freetoken-TP{i}-scheduler",
            )
            p.start()
            processes.append(p)

        num_tokenizers = server_args.num_tokenizer
        p = mp.Process(
            target=_run_tokenize_worker,
            kwargs={
                "detach": detach,
                "tokenizer_path": server_args.model_path,
                "addr": server_args.zmq_detokenizer_addr,
                "backend_addr": server_args.zmq_backend_addr,
                "frontend_addr": server_args.zmq_frontend_addr,
                "local_bs": 1,
                "create": server_args.tokenizer_create_addr,
                "tokenizer_id": num_tokenizers,
                "ack_queue": ack_queue,
            },
            daemon=False,
            name="freetoken-detokenizer-0",
        )
        p.start()
        processes.append(p)
        for i in range(num_tokenizers):
            p = mp.Process(
                target=_run_tokenize_worker,
                kwargs={
                    "detach": detach,
                    "tokenizer_path": server_args.model_path,
                    "addr": server_args.zmq_tokenizer_addr,
                    "backend_addr": server_args.zmq_backend_addr,
                    "frontend_addr": server_args.zmq_frontend_addr,
                    "local_bs": 1,
                    "create": server_args.tokenizer_create_addr,
                    "tokenizer_id": i,
                    "ack_queue": ack_queue,
                },
                daemon=False,
                name=f"freetoken-tokenizer-{i}",
            )
            p.start()
            processes.append(p)

        # Expected ready acks: 1 primary scheduler + num_tokenizers + 1 detokenizer.
        return BackendHandle(
            ack_queue=ack_queue,
            processes=processes,
            expected_acks=num_tokenizers + 2,
        )

    run_api_server(server_args, start_subprocess, run_shell=run_shell)


if __name__ == "__main__":
    launch_server()
