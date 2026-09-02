"""Reusable pinned host-bank primitives shared by the fast expert-load paths.

Two ideas the parallel read of the original checkpoint and FTW (read a repacked
contiguous cache) paths both rely on:

* **pin-after-fill** -- allocate the bank as a *lazy* anonymous ``mmap`` (no pages
  resident, instant), fill it with real data, and only THEN ``cudaHostRegister`` it.
  Registering already-resident pages just page-locks them; registering a lazy mmap first
  faults+zero-fills every page (~137 GiB -> ~47 s for DSV4) and that zero-fill is then
  immediately overwritten by the read. So pin-after-fill removes a whole redundant pass.
* **chunked multi-threaded O_DIRECT** -- DMA straight from disk into the (page-aligned)
  bank, bypassing the page cache, with many concurrent ``preadv`` on one fd (scales to the
  device's queue-depth ceiling even for a single file).

Serving banks are registered as explicit host materializations. Ordinary offload keeps
them attached for the cache lifetime; final-residency research paths may classify them
as optional reusable materialization or physically reclaim them.
"""

from __future__ import annotations

import contextlib
import ctypes
import itertools
import math
import mmap
import os
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from typing import Self

import torch

from freetoken.utils import init_logger

logger = init_logger(__name__)

_BLK = 4096  # O_DIRECT alignment (page size)


class HostResidency(str, Enum):
    """Residency class of a host bank layer.

    Only PINNED (cudaHostRegister'd) memory can feed the GPU movement paths; LOCKED (mlock'd, no device address) and PAGEABLE layers must decode on the CPU executor.
    The non-pinned classes exist for hosts that cap CUDA pin quota (WSL/WDDM: ~half of RAM).
    """

    PINNED = "pinned"
    LOCKED = "locked"
    PAGEABLE = "pageable"


_DEFAULT_CHUNK = 8 << 20


class HostMaterializationState(str, Enum):
    ATTACHED_REQUIRED = "attached_required"
    RETAINED_OPTIONAL = "retained_optional"
    RELEASED_PHYSICAL = "released_physical"


class HostMaterializationPolicy(str, Enum):
    RETAIN_REUSABLE_SOURCE = "retain_reusable_source"
    RELEASE_AFTER_FINAL_RESIDENCY = "release_after_final_residency"


_ALLOCATION_IDS = itertools.count(1)
_LIVE_MATERIALIZATIONS: dict[int, HostBank] = {}
_STORAGE_OWNERS: dict[int, int] = {}

def _env_born_pinned() -> bool | None:
    """``FREETOKEN_BANK_CUDA_ALLOC`` tri-state: unset -> ``None`` (default applies), else the parsed boolean."""
    v = os.environ.get("FREETOKEN_BANK_CUDA_ALLOC", "").strip().lower()
    if not v:
        return None
    return v in ("1", "true", "yes", "on")


def born_pinned_default() -> bool:
    """Whether PINNED serving banks use cudaHostAlloc instead of mmap + register-after-fill.

    Off by default: registered mmaps already read at the PCIe roofline and lazy mmaps commit pages only on fill. ``FREETOKEN_BANK_CUDA_ALLOC`` overrides."""
    env = _env_born_pinned()
    if env is not None:
        return env
    return False


class HostBank:
    """A page-aligned host buffer + its torch view, page-locked on demand: allocate -> fill -> ``pin()``/``lock()``.

    * ``"mmap"`` (default) -- lazy anonymous mmap; pages materialize on fill, then ``pin()`` registers or ``lock()`` OS-locks it.
    * ``"cuda"`` -- cudaHostAlloc, born pinned+mapped; ``pin()``/``lock()``/``release()`` are no-ops and it never takes LOCKED. See :func:`born_pinned_default`.

    The buffer is rounded up to the O_DIRECT block; ``tensor`` views exactly ``nbytes``. ``backing=None`` follows ``FREETOKEN_BANK_CUDA_ALLOC``."""

    __slots__ = (
        "_allocation_id",
        "_backing",
        "_buf",
        "_locked",
        "_pinned",
        "_release_invocations",
        "_state",
        "_storage_keys",
        "addr",
        "allocation_bytes",
        "nbytes",
        "tensor",
    )

    def __init__(self, shape: tuple[int, ...], dtype: torch.dtype,
                 *, backing: str | None = None):
        if backing is None:
            plan = _requested_residency
            # a plan with non-pinned labels vetoes born-pinned: cudaHostAlloc spends the pin quota the plan exists to save
            born = _env_born_pinned() and (plan is None or not plan.has_unpinned)
            backing = "cuda" if born else "mmap"
        assert backing in ("mmap", "cuda"), backing
        self._backing = backing
        self._state = HostMaterializationState.ATTACHED_REQUIRED
        self._release_invocations = 0
        self._allocation_id = next(_ALLOCATION_IDS)
        elsize = torch.empty((), dtype=dtype).element_size()
        self.nbytes = math.prod(shape) * elsize
        asize = ((self.nbytes + _BLK - 1) // _BLK) * _BLK
        self.allocation_bytes = asize + (_BLK if backing == "cuda" else 0)
        if backing == "cuda":
            from freetoken.kernel.pinned import alloc_pinned_tensor

            # direct-IO readers need page alignment, but cudaHostAlloc only guarantees ~512 in practice
            # over-allocate one block and carve the aligned window; the numpy slice keeps the pinned storage alive via .base
            raw = alloc_pinned_tensor(asize + _BLK, dtype=torch.uint8)  # cudaMallocHost
            raw.zero_()  # keep the anonymous-mmap guarantee: unwritten regions stay zero
            off = (-raw.data_ptr()) % _BLK
            self._buf = raw.numpy()[off:off + asize]
            self.addr = raw.data_ptr() + off
            assert self.addr % _BLK == 0
            self._pinned = True  # born pinned+mapped; pin() is a no-op
        else:
            self._buf = mmap.mmap(-1, asize)  # lazy: address space only, no resident pages yet
            self.addr = ctypes.addressof(ctypes.c_char.from_buffer(self._buf))
            self._pinned = False
        self.tensor = torch.frombuffer(self._buf, dtype=dtype, count=self.nbytes // elsize).view(*shape)
        self._locked = False
        self._storage_keys = {
            self.tensor.data_ptr(), self.tensor.untyped_storage().data_ptr()
        }
        _LIVE_MATERIALIZATIONS[self._allocation_id] = self
        for key in self._storage_keys:
            _STORAGE_OWNERS[key] = self._allocation_id

    @property
    def allocation_id(self) -> str:
        """Process-local diagnostic identity; deliberately not a raw address."""
        return f"host-materialization-{self._allocation_id}"

    @property
    def lifecycle_state(self) -> HostMaterializationState:
        return self._state

    @property
    def allocation_mechanism(self) -> str:
        return "cudaHostAlloc" if self._backing == "cuda" else "mmap+cudaHostRegister"

    def diagnostics(self) -> dict[str, object]:
        """Pointer-free ownership and lifecycle evidence."""
        return {
            "allocation_id": self.allocation_id,
            "requested_bytes": self.nbytes,
            "allocation_bytes": self.allocation_bytes,
            "allocation_mechanism": self.allocation_mechanism,
            "owning_python_class": type(self).__name__,
            "owner_tensor_object_present": self.tensor is not None,
            "storage_present": self._buf is not None,
            "registered_or_pinned": self._pinned,
            "locked": self._locked,
            "lifecycle_state": self._state.value,
            "release_invocation_count": self._release_invocations,
        }

    def mark_retained_optional(self) -> None:
        if self._state is HostMaterializationState.RELEASED_PHYSICAL:
            raise RuntimeError("cannot retain a physically released host materialization")
        self._state = HostMaterializationState.RETAINED_OPTIONAL

    def cede_tensor_ownership(self) -> None:
        """Leave Tensor lifetime to the loader/cache while retaining storage ownership."""
        self.tensor = None

    def release_physical(self) -> int:
        """Deregister and discard pages after every execution owner has detached."""
        self._release_invocations += 1
        if self._state is HostMaterializationState.RELEASED_PHYSICAL:
            raise RuntimeError("host materialization is already physically released")
        if self._backing == "mmap" and self._pinned:
            from freetoken.kernel.pinned import host_unregister

            host_unregister(self.addr)
            self._pinned = False
        if self._locked:
            _os_unlock(self.addr, self.allocation_bytes)
            self._locked = False
        for key in self._storage_keys:
            _STORAGE_OWNERS.pop(key, None)
        self._storage_keys = set()
        self.tensor = None
        if self._backing == "mmap":
            self._buf.madvise(mmap.MADV_DONTNEED)
            self._buf.close()
            self._buf = None
        else:
            # The NumPy view owns the from-blob Tensor chain whose deleter calls
            # cudaFreeHost. The cache permits this only after external source Tensor
            # objects are dead.
            self._buf = None
        self._state = HostMaterializationState.RELEASED_PHYSICAL
        return self.allocation_bytes

    @property
    def residency(self) -> HostResidency:
        if self._pinned:
            return HostResidency.PINNED
        if self._locked:
            return HostResidency.LOCKED
        return HostResidency.PAGEABLE

    def memoryview(self) -> memoryview:
        return memoryview(self._buf)

    def pin(self) -> None:
        """cudaHostRegister the (now-filled) buffer -- pin-after-fill.

        ``FREETOKEN_SKIP_BANK_PIN=1`` makes this a no-op for CPU-only tooling (the FTW converter); never set it when serving, the GPU paths need registered banks."""
        if self._pinned:
            return
        if os.environ.get("FREETOKEN_SKIP_BANK_PIN", "").strip().lower() in ("1", "true", "yes", "on"):
            return
        from freetoken.kernel.pinned import host_register

        try:
            host_register(self.addr, len(self._buf))
        except RuntimeError as exc:
            raise RuntimeError(
                f"cudaHostRegister failed for {len(self._buf) / 2**30:.1f} GiB"
            ) from exc
        self._pinned = True

    def release(self) -> None:
        """Drop the resident pages; the address space stays valid, the contents become undefined.

        For buffers that are done being read (the converter). No-op for born-pinned banks: registered pages cannot be dropped."""
        if self._pinned:
            return
        self._buf.madvise(mmap.MADV_DONTNEED)

    def lock(self) -> None:
        """mlock the (now-filled) buffer: resident without CUDA pin quota, but no device address -- only the CPU executor can serve a locked layer.

        Lock after fill, or the lazy mmap faults+zero-fills every page. A failed lock (RLIMIT_MEMLOCK) warns once and leaves the bank PAGEABLE, which every consumer treats the same."""
        if self._locked or self._pinned:  # cudaHostRegister already page-locks
            return
        global _os_lock_failed
        if _os_lock_failed:
            return  # the quota is exhausted for good; skip the syscall spam
        try:
            _os_lock(self.addr, len(self._buf))
        except (OSError, ImportError) as exc:
            _os_lock_failed = True
            logger.warning(f"bank lock failed; leaving this and later banks pageable: {exc}")
            return
        self._locked = True


_os_locked_total = 0  # bytes locked so far; the OS lock ceiling is a per-process quota
_os_lock_failed = False  # sticky: once over quota, later (bigger-total) locks fail too


def _os_lock(addr: int, nbytes: int) -> None:
    global _os_locked_total
    import resource

    # grow the soft RLIMIT_MEMLOCK (defaults to a few MiB); the hard limit needs privilege, past it mlock fails below
    want = _os_locked_total + nbytes + (256 << 20)
    soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    if soft != resource.RLIM_INFINITY and soft < want:
        new_soft = want if hard == resource.RLIM_INFINITY else min(want, hard)
        if new_soft > soft:
            try:
                resource.setrlimit(resource.RLIMIT_MEMLOCK, (new_soft, hard))
            except (OSError, ValueError):
                pass  # keep the old limit; mlock below reports the real ceiling
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.mlock(ctypes.c_void_p(addr), ctypes.c_size_t(nbytes)):
        err = ctypes.get_errno()
        raise OSError(
            err,
            f"mlock({nbytes / 2**30:.1f} GiB): {os.strerror(err)} "
            f"(RLIMIT_MEMLOCK / `ulimit -l` caps OS-locked bytes; raise it or "
            f"shrink --moe-cpu-layers)",
        )
    _os_locked_total += nbytes


def _os_unlock(addr: int, nbytes: int) -> None:
    global _os_locked_total
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.munlock(ctypes.c_void_p(addr), ctypes.c_size_t(nbytes)):
        err = ctypes.get_errno()
        raise OSError(err, f"munlock({nbytes}): {os.strerror(err)}")
    _os_locked_total = max(0, _os_locked_total - nbytes)


def materializations_for_tensors(
    tensors: dict[str, list[torch.Tensor]],
) -> list[HostBank]:
    """Resolve explicit HostBank storage owners without publishing raw addresses."""
    found: dict[int, HostBank] = {}
    for per_layer in tensors.values():
        for tensor in per_layer:
            owner = materialization_for_tensor(tensor)
            if owner is not None:
                found[owner._allocation_id] = owner
    return list(found.values())


def materialization_for_tensor(tensor: torch.Tensor) -> HostBank | None:
    keys = (tensor.untyped_storage().data_ptr(), tensor.data_ptr())
    allocation_id = next(
        (_STORAGE_OWNERS[key] for key in keys if key in _STORAGE_OWNERS), None
    )
    return (
        None if allocation_id is None else _LIVE_MATERIALIZATIONS[allocation_id]
    )


def alloc_banks(specs: dict[str, tuple[tuple[int, ...], torch.dtype]]) -> dict[str, HostBank]:
    """Allocate (lazy, unpinned) host banks from ``{name: (shape, dtype)}``."""
    return {name: HostBank(shape, dtype) for name, (shape, dtype) in specs.items()}


def alloc_layer_banks(
    specs: dict[str, tuple[tuple[int, ...], torch.dtype]], num_layers: int
) -> dict[str, list[HostBank]]:
    """Allocate per-layer host banks: ``{name: ([num_experts, ...] row shape, dtype)}``
    -> one independently allocated (page-aligned, independently pin/lock-able)
    ``HostBank`` per layer per name."""
    return {
        name: [HostBank(shape, dtype) for _ in range(num_layers)]
        for name, (shape, dtype) in specs.items()
    }


class _ResidencyPlan:
    """Per-layer ``HostResidency`` labels, ambiently visible to the bank settle points.

    Installed by ``load_expert_banks`` around the provider dispatch so every loader honors --moe-cpu-layers without a new parameter in each signature. ``applied`` flips once a settle point consults the plan."""

    __slots__ = ("actual", "applied", "has_unpinned", "labels")

    def __init__(self, labels: list[str]):
        self.labels = list(labels)
        self.applied = False
        self.has_unpinned = any(r != HostResidency.PINNED.value for r in labels)
        self.actual: dict[int, str] = {}

    def residency_for(self, layer_id: int) -> str:
        self.applied = True
        return self.labels[layer_id]

    def record(self, layer_id: int, achieved: str) -> None:
        """One pageable bank downgrades the whole layer (a failed lock settles PAGEABLE)."""
        if self.actual.get(layer_id) != HostResidency.PAGEABLE.value:
            self.actual[layer_id] = achieved


_requested_residency: _ResidencyPlan | None = None


@contextlib.contextmanager
def requested_residency(labels: list[str] | None):
    """Install the ambient per-layer residency plan for the enclosed bank load (``None`` = no plan, everything pins)."""
    global _requested_residency
    if labels is None:
        yield None
        return
    plan = _ResidencyPlan(labels)
    prev, _requested_residency = _requested_residency, plan
    try:
        yield plan
    finally:
        _requested_residency = prev


def _settle(bank: HostBank, residency: str) -> None:
    """Route a filled bank to its residency class (PAGEABLE = leave the plain mmap)."""
    if residency == HostResidency.PINNED.value:
        bank.pin()
    elif residency == HostResidency.LOCKED.value:
        bank.lock()


def pin_banks(banks: dict[str, HostBank | list[HostBank]]) -> None:
    """Settle every bank after it has been filled -- pin-after-fill by default.
    List-valued entries are per-layer and honor the ambient :func:`requested_residency` plan; scalar banks always pin."""
    plan = _requested_residency
    for bank in banks.values():
        if isinstance(bank, list):
            for layer_id, layer_bank in enumerate(bank):
                residency = (
                    HostResidency.PINNED.value if plan is None
                    else plan.residency_for(layer_id)
                )
                _settle(layer_bank, residency)
                if plan is not None and residency == HostResidency.LOCKED.value:
                    plan.record(layer_id, layer_bank.residency.value)
        else:
            bank.pin()


class PinPipeline:
    """Settle (pin or lock) filled banks while other banks are still being read.

    cudaHostRegister is driver-serialized, so one background thread drains a queue and submitters never block: load time ~= max(read, settle).
    LOCKED banks mlock on the same thread (the quota bookkeeping in ``_os_lock`` is not thread-safe).
    A clean context-manager exit drains the queue and re-raises the first settle failure.
    """

    def __init__(self) -> None:
        self._q: queue.SimpleQueue = queue.SimpleQueue()
        self._exc: BaseException | None = None
        # the current device is thread-local: a fresh thread sits on device 0 and cudaHostRegister would build its context there -- carry the creator's (bound) device into the worker
        self._device = torch.cuda.current_device() if torch.cuda.is_available() else None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        if self._device is not None:
            torch.cuda.set_device(self._device)
        while True:
            item = self._q.get()
            if item is None:
                return
            if self._exc is not None:
                continue  # drain without settling after a failure
            bank, residency, plan, layer_id = item
            try:
                _settle(bank, residency)
                if plan is not None and residency == HostResidency.LOCKED.value:
                    plan.record(layer_id, bank.residency.value)
            except BaseException as exc:  # noqa: BLE001 - surfaced by wait()/__exit__
                self._exc = exc

    def submit(self, bank: HostBank, residency: str = HostResidency.PINNED.value,
               plan=None, layer_id: int | None = None) -> None:
        self._q.put((bank, residency, plan, layer_id))

    def __call__(self, layer_id: int, banks: dict[str, HostBank]) -> None:
        """Layer-completion sink: queue every bank of the completed layer at its ambient :func:`requested_residency` label."""
        plan = _requested_residency
        residency = (
            HostResidency.PINNED.value if plan is None else plan.residency_for(layer_id)
        )
        for bank in banks.values():
            self.submit(bank, residency, plan, layer_id)

    def _join(self) -> None:
        self._q.put(None)
        self._thread.join()

    def wait(self) -> None:
        self._join()
        if self._exc is not None:
            raise self._exc

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            self._join()  # no thread leak; the in-flight exception wins
            return
        self.wait()


class LayerCompletionTracker:
    """Fire a sink once per layer, when all of that layer's writes have landed.

    ``note(layer_id)`` is called after each write; at ``expected_per_layer``
    notes the layer's banks are handed to ``on_layer(layer_id, {name: bank})``
    exactly once. Thread-safe (shard-driven loaders write layers from many
    threads in arbitrary order).
    """

    def __init__(
        self,
        expected_per_layer: int,
        banks: dict[str, list],
        on_layer,
    ) -> None:
        assert expected_per_layer > 0
        self._expected = expected_per_layer
        self._banks = banks
        self._on_layer = on_layer
        self._counts: dict[int, int] = {}
        self._lock = threading.Lock()

    def note(self, layer_id: int) -> None:
        with self._lock:
            n = self._counts.get(layer_id, 0) + 1
            self._counts[layer_id] = n
            fire = n == self._expected
        if fire:
            self._on_layer(layer_id, {name: per[layer_id] for name, per in self._banks.items()})


def read_file_into(buf: memoryview | mmap.mmap, path: str, *, workers: int = 8,
                   chunk: int = _DEFAULT_CHUNK, drop_cache: bool = True) -> int:
    """Chunked multi-threaded O_DIRECT read of the whole file ``path`` into ``buf``
    (page-aligned). Returns the file size. The buffer must be >= the rounded-up file size."""
    size = os.path.getsize(path)
    if drop_cache:
        try:
            fd0 = os.open(path, os.O_RDONLY)
            os.posix_fadvise(fd0, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd0)
        except OSError:
            pass
    mv = buf if isinstance(buf, memoryview) else memoryview(buf)
    fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    offs = list(range(0, size, chunk))

    def rd(o):
        want = min(chunk, len(mv) - o)
        want = min(want, ((size - o + _BLK - 1) // _BLK) * _BLK)
        os.preadv(fd, [mv[o:o + want]], o)

    try:
        if len(offs) <= 1:
            for o in offs:
                rd(o)
        else:
            with ThreadPoolExecutor(workers) as ex:
                list(ex.map(rd, offs))
    finally:
        os.close(fd)
    return size


def _preadv_all(fd: int, dst: memoryview, offset: int, need: int) -> None:
    """preadv into ``dst`` until ``need`` bytes have landed; O_DIRECT may return a short count."""
    done = 0
    while done < need:
        if done % _BLK:  # a continuation read has to stay block-aligned on both sides
            raise OSError(f"unaligned short O_DIRECT read: {done} of {need} bytes at {offset}")
        got = os.preadv(fd, [dst[done:]], offset + done)
        if got <= 0:
            raise OSError(f"short O_DIRECT read: {done} of {need} bytes at {offset}")
        done += got


def read_range_into(buf: memoryview | mmap.mmap, path: str, *, file_offset: int, nbytes: int,
                    dest_offset: int = 0, workers: int = 8, chunk: int = _DEFAULT_CHUNK,
                    drop_cache: bool = True) -> int:
    """Chunked multi-threaded O_DIRECT read of ``path[file_offset : file_offset + nbytes]`` into ``buf`` at ``dest_offset``. Returns ``nbytes``.

    Byte-range counterpart of :func:`read_file_into`, for one tensor inside a shard. O_DIRECT needs the file offset AND the destination address block-aligned at the same time, which only holds when the two share their offset mod 4096 -- a safetensors data offset practically never lines up with the tensor's slot in the bank. Chunks that do line up DMA straight into ``buf``; the rest DMA into a page-aligned bounce (source window rounded out to whole blocks) and are copied into place, which also covers the unaligned head and tail.
    """
    mv = (buf if isinstance(buf, memoryview) else memoryview(buf)).cast("B")
    if dest_offset + nbytes > len(mv):
        raise ValueError(f"destination holds {len(mv)} bytes, need {dest_offset + nbytes}")
    base = ctypes.addressof(ctypes.c_char.from_buffer(mv))
    if drop_cache:
        try:
            fd0 = os.open(path, os.O_RDONLY)
            os.posix_fadvise(fd0, file_offset, nbytes, os.POSIX_FADV_DONTNEED)
            os.close(fd0)
        except OSError:
            pass
    fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    scratch = threading.local()

    def rd(i: int) -> None:
        n = min(chunk, nbytes - i)
        src, dst = file_offset + i, dest_offset + i
        if src % _BLK == 0 and (base + dst) % _BLK == 0 and n % _BLK == 0:
            _preadv_all(fd, mv[dst:dst + n], src, n)
            return
        head = src % _BLK
        span = ((head + n + _BLK - 1) // _BLK) * _BLK
        bounce = getattr(scratch, "buf", None)
        if bounce is None or len(bounce) < span:
            bounce = scratch.buf = mmap.mmap(-1, span)  # anonymous mmaps are page-aligned
        bmv = memoryview(bounce)
        _preadv_all(fd, bmv[:span], src - head, head + n)
        mv[dst:dst + n] = bmv[head:head + n]

    try:
        offs = list(range(0, nbytes, chunk))
        if len(offs) <= 1:
            for o in offs:
                rd(o)
        else:
            with ThreadPoolExecutor(workers) as ex:
                list(ex.map(rd, offs))
    finally:
        os.close(fd)
    return nbytes


__all__ = [
    "HostBank",
    "HostMaterializationPolicy",
    "HostMaterializationState",
    "HostResidency",
    "LayerCompletionTracker",
    "PinPipeline",
    "alloc_banks",
    "alloc_layer_banks",
    "born_pinned_default",
    "materialization_for_tensor",
    "materializations_for_tensors",
    "pin_banks",
    "read_file_into",
    "read_range_into",
    "requested_residency",
]
