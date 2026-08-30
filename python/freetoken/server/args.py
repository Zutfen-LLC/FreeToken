from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List, Tuple

import torch

from freetoken.distributed import DistributedInfo
from freetoken.scheduler import SchedulerConfig
from freetoken.utils import init_logger


@dataclass(frozen=True)
class ServerArgs(SchedulerConfig):
    server_host: str = "127.0.0.1"
    server_port: int = 1919
    num_tokenizer: int = 0
    silent_output: bool = False
    # The terminal shell is attached to this server (ft shell --model / ft serve --shell-mode).
    # The workers read it to leave the shell's foreground process group, so the ^C that cancels
    # a turn cannot also kill the engine — see server/launch.py:_detach_process_group.
    shell_mode: bool = False
    served_model_name: str | None = None
    tool_call_parser: str = "llama3"
    # Reasoning parser that splits <think> reasoning from content for OpenAI
    # responses. None disables it (default for models without a reasoning protocol).
    reasoning_parser: str | None = None
    # "model": fill unspecified request sampling params from generation_config.json
    # (temperature/top_k/top_p), like sglang. "none": use framework defaults only.
    sampling_defaults: str = "model"
    # Default max output (decode) tokens for a request that omits one. None falls back to the
    # adapter's built-in default (32k).
    max_output_tokens: int | None = None
    # Report the prefix-cache hit in each response's usage block (OpenAI
    # prompt_tokens_details.cached_tokens, Anthropic cache_read_input_tokens, Responses
    # input_tokens_details.cached_tokens). Mirrors sglang's --enable-cache-report.
    enable_cache_report: bool = False
    # Comma-separated CORS allow-list for browser/webview clients (e.g. the desktop
    # app). Empty string disables CORS headers entirely; "*" allows any origin.
    cors_origins: str = "tauri://localhost,http://tauri.localhost,http://localhost:1420"
    # --gpu entries in TP-rank order, empty = not given
    gpu: tuple[str, ...] = ()
    # full UUIDs resolved from --gpu, entry i = TP rank i; None = NVML unavailable, each worker then resolves its raw entry against CUDA's own enumeration
    gpu_assigned: "tuple[str, ...] | None" = None
    # InferSwarm Phase-1 POC device, independent of the primary/TP rank list above.
    inferswarm_secondary_gpu: str | None = None
    # Full physical UUID resolved by the parent when NVML is available.  The worker still
    # verifies it against CUDA's visible enumeration before loading the model.
    inferswarm_secondary_gpu_assigned: str | None = None
    # InferSwarm Phase-1 P2 frozen placement artifact. It is meaningful only alongside the
    # explicitly resolved secondary above; the path is deliberately withheld from runtime
    # provenance because it is host-local.
    inferswarm_placement: str | None = None
    # InferSwarm Phase-1 P3 decode execution.  False preserves P1/P2 inert behavior.
    inferswarm_remote_decode: bool = False
    # Post-NO-GO D2 architecture-search path.  Separate opt-in keeps canonical P3/P4
    # behavior and its eager-only contract unchanged.
    inferswarm_experimental_d2_graph_remote: bool = False
    # D3 is intentionally a separate two-worker research interface.  It does not alter
    # canonical P3 or the D2 single-worker flag/placement contract.
    inferswarm_experimental_d3_graph_multiworker: bool = False
    inferswarm_experimental_d4_capability_weighted: bool = False
    # D5 startup-only loader experiment; independent of route execution semantics.
    inferswarm_experimental_d5_resident_loader: bool = False
    inferswarm_experimental_d5_compact_routes: bool = False
    inferswarm_d5_loader_cpu_workers: int = 4
    # Fixed startup-only D3 execution shape; never selected on a decode token.
    inferswarm_d3_active_workers: str = "ab"
    inferswarm_d3_placement: str | None = None
    inferswarm_d4_placement: str | None = None
    inferswarm_d3_worker_a_gpu: str | None = None
    inferswarm_d3_worker_b_gpu: str | None = None
    inferswarm_d3_worker_a_gpu_assigned: str | None = None
    inferswarm_d3_worker_b_gpu_assigned: str | None = None
    # P4 intended scheduling shape; serialized is a diagnostic P3-equivalent control.
    inferswarm_remote_mode: str = "overlap"
    # Detailed records are bounded by decode steps; cumulative gates continue past it.
    inferswarm_mechanism_max_steps: int = 256

    @property
    def share_tokenizer(self) -> bool:
        return self.num_tokenizer == 0

    @property
    def zmq_frontend_addr(self) -> str:
        return "ipc:///tmp/freetoken_3" + self._unique_suffix

    @property
    def zmq_tokenizer_addr(self) -> str:
        if self.share_tokenizer:
            return self.zmq_detokenizer_addr
        result = "ipc:///tmp/freetoken_4" + self._unique_suffix
        assert result != self.zmq_detokenizer_addr
        return result

    @property
    def tokenizer_create_addr(self) -> bool:
        return self.share_tokenizer

    @property
    def backend_create_detokenizer_link(self) -> bool:
        return not self.share_tokenizer

    @property
    def frontend_create_tokenizer_link(self) -> bool:
        return not self.share_tokenizer

    @property
    def distributed_addr(self) -> str:
        return f"tcp://127.0.0.1:{self.server_port + 1}"


def parse_args(
    args: List[str],
    run_shell: bool = False,
    prog: str | None = None,
) -> Tuple[ServerArgs, bool]:
    """
    Parse command line arguments and return an EngineConfig.

    Args:
        args: Command line arguments (e.g., sys.argv[1:])

    Returns:
        EngineConfig instance with parsed arguments
    """
    from freetoken.attention import validate_attn_backend
    from freetoken.kvcache import SUPPORTED_CACHE_MANAGER
    from freetoken.moe import SUPPORTED_MOE_BACKENDS

    def _parse_moe_cache_rate(value: str) -> float:
        try:
            rate = float(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("must be a number in [0, 1]") from exc
        if not 0 <= rate <= 1:
            raise argparse.ArgumentTypeError("must be in [0, 1]")
        return rate

    def _positive_int(value: str) -> int:
        try:
            n = int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("must be a positive integer") from exc
        if n < 1:
            raise argparse.ArgumentTypeError("must be >= 1")
        return n

    def _nonnegative_int(value: str) -> int:
        try:
            n = int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("must be a non-negative integer") from exc
        if n < 0:
            raise argparse.ArgumentTypeError("must be >= 0")
        return n

    def _lazy_gpu_arg(value: str) -> tuple[str, ...]:
        from freetoken.gpu_select import gpu_arg

        return gpu_arg(value)

    def _lazy_single_gpu_arg(value: str) -> str:
        from freetoken.gpu_select import single_gpu_arg

        return single_gpu_arg(value)

    def _infer_tool_call_parser(model_path: str) -> str:
        try:
            from freetoken.utils import cached_load_hf_config

            cfg = cached_load_hf_config(model_path).to_dict()
        except Exception:
            cfg = {}

        text_cfg = cfg.get("text_config") or {}
        candidates = [
            model_path,
            str(cfg.get("model_type", "")),
            str(text_cfg.get("model_type", "")),
            " ".join(str(v) for v in cfg.get("architectures", []) or []),
            " ".join(str(v) for v in text_cfg.get("architectures", []) or []),
        ]
        marker = " ".join(candidates).lower()
        if "gpt_oss" in marker or "gpt-oss" in marker or "gptoss" in marker:
            return "gpt_oss"
        # M3 first: its marker also contains the bare "minimax" substring, but the
        # namespaced tool grammar is a different protocol from M2's.
        if "minimax_m3" in marker or "minimax-m3" in marker or "minimaxm3" in marker:
            return "minimax_m3"
        if "minimax" in marker:
            return "minimax"
        if (
            "muse_glimmer" in marker
            or "muse-glimmer" in marker
            or "museglimmer" in marker
        ):
            return "muse_glimmer"
        if "gemma4" in marker:
            return "gemma4"
        if (
            "qwen3_5" in marker
            or "qwen3.5" in marker
            or ("qwen3" in marker and "coder" in marker)
        ):
            return "qwen3_coder"
        if "qwen" in marker:
            return "qwen25"
        if "deepseek" in marker and ("v4" in marker or "deepseek_v4" in marker):
            return "deepseekv32"
        if "deepseek" in marker and ("v3.2" in marker or "v32" in marker):
            return "deepseekv32"
        if "glm" in marker:
            return "glm47"
        if "mistral" in marker:
            return "mistral"
        return "llama3"

    def _infer_reasoning_parser(model_path: str) -> str | None:
        try:
            from freetoken.utils import cached_load_hf_config

            cfg = cached_load_hf_config(model_path).to_dict()
        except Exception:
            cfg = {}

        text_cfg = cfg.get("text_config") or {}
        candidates = [
            model_path,
            str(cfg.get("model_type", "")),
            str(text_cfg.get("model_type", "")),
            " ".join(str(v) for v in cfg.get("architectures", []) or []),
            " ".join(str(v) for v in text_cfg.get("architectures", []) or []),
        ]
        marker = " ".join(candidates).lower()
        if "gpt_oss" in marker or "gpt-oss" in marker or "gptoss" in marker:
            return "gpt_oss"
        if "deepseek" in marker and any(
            tag in marker for tag in ("v4", "deepseek_v4", "v3.2", "v32")
        ):
            return "deepseekv32"
        if "qwen3" in marker or "qwen3.5" in marker or "qwen3_5" in marker:
            return "qwen3"
        if "glm" in marker:
            return "glm"
        # M3 first ("minimax" is a substring): <mm:think> tags + 3 thinking gears,
        # not M2's always-on implicit <think>.
        if "minimax_m3" in marker or "minimax-m3" in marker or "minimaxm3" in marker:
            return "minimax_m3"
        if "minimax" in marker:
            return "minimax"
        if (
            "muse_glimmer" in marker
            or "muse-glimmer" in marker
            or "museglimmer" in marker
        ):
            return "muse_glimmer"
        if "gemma4" in marker:
            return "gemma4"
        return None

    parser = argparse.ArgumentParser(
        prog=prog, description="FreeToken Server Arguments"
    )

    parser.add_argument(
        "--model-path",
        "--model",
        type=str,
        required=True,
        help="The path of the model weights. This can be a local folder or a Hugging Face repo ID.",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Data type for model weights and activations. 'auto' will use FP16 for FP32/FP16 models and BF16 for BF16 models.",
    )

    parser.add_argument(
        "--tensor-parallel-size",
        "--tp-size",
        type=int,
        default=1,
        help="The tensor parallelism size.",
    )

    parser.add_argument(
        "--gpu",
        type=_lazy_gpu_arg,
        default=ServerArgs.gpu,
        help=(
            "GPU(s) to run on, comma-separated; entry i is TP rank i. Each entry is a GPU UUID (GPU-xxxx..., as nvidia-smi -L prints) or an nvidia-smi index"
        ),
    )

    parser.add_argument(
        "--inferswarm-secondary-gpu",
        type=_lazy_single_gpu_arg,
        default=ServerArgs.inferswarm_secondary_gpu,
        help=(
            "Experimental InferSwarm Phase-1 POC secondary GPU. Takes one GPU UUID "
            "(preferred) or one existing FreeToken GPU index; independent of tensor "
            "parallelism and never changes CUDA_VISIBLE_DEVICES"
        ),
    )

    parser.add_argument(
        "--inferswarm-placement",
        type=str,
        default=ServerArgs.inferswarm_placement,
        help=(
            "Experimental InferSwarm Phase-1 P2 frozen placement artifact. Requires --inferswarm-secondary-gpu and loads an inert resident expert bank at startup"
        ),
    )

    parser.add_argument(
        "--inferswarm-remote-decode",
        action="store_true",
        default=ServerArgs.inferswarm_remote_decode,
        help=(
            "Experimental InferSwarm Phase-1 P4 remote decode. Requires --inferswarm-secondary-gpu, --inferswarm-placement, and eager decode"
        ),
    )

    parser.add_argument(
        "--inferswarm-experimental-d2-graph-remote",
        action="store_true",
        default=ServerArgs.inferswarm_experimental_d2_graph_remote,
        help=(
            "EXPERIMENTAL post-NO-GO D2 batch-one remote decode captured into one "
            "multi-device CUDA graph; does not change --inferswarm-remote-decode"
        ),
    )
    parser.add_argument("--inferswarm-experimental-d3-graph-multiworker", action="store_true", default=ServerArgs.inferswarm_experimental_d3_graph_multiworker, help="EXPERIMENTAL D3 captured GPU0/A/B fan-out; separate from canonical and D2 paths")
    parser.add_argument("--inferswarm-experimental-d4-capability-weighted", action="store_true", default=ServerArgs.inferswarm_experimental_d4_capability_weighted, help="EXPERIMENTAL D4 placement on the unchanged D3 executor")
    parser.add_argument("--inferswarm-experimental-d5-resident-loader", action="store_true", default=ServerArgs.inferswarm_experimental_d5_resident_loader, help="EXPERIMENTAL D5 bulk pinned, concurrently verified resident loader")
    parser.add_argument("--inferswarm-experimental-d5-compact-routes", action="store_true", default=ServerArgs.inferswarm_experimental_d5_compact_routes, help="EXPERIMENTAL D5 stable compact physical route execution")
    parser.add_argument("--inferswarm-d5-loader-cpu-workers", type=int, choices=(1, 2, 4, 8), default=ServerArgs.inferswarm_d5_loader_cpu_workers, help="bounded CPU staging workers per D5 resident worker")
    parser.add_argument("--inferswarm-d3-active-workers", choices=("a", "b", "ab"), default=ServerArgs.inferswarm_d3_active_workers, help="EXPERIMENTAL D3 fixed captured worker shape (a, b, or ab)")
    parser.add_argument("--inferswarm-d3-placement", type=str, default=ServerArgs.inferswarm_d3_placement, help="SHA-pinned frozen D3 placement artifact")
    parser.add_argument("--inferswarm-d4-placement", type=str, default=ServerArgs.inferswarm_d4_placement, help="SHA-pinned frozen D4 placement artifact")
    parser.add_argument("--inferswarm-d3-worker-a-gpu", type=_lazy_single_gpu_arg, default=ServerArgs.inferswarm_d3_worker_a_gpu, help="D3 worker A GPU UUID or visible selector")
    parser.add_argument("--inferswarm-d3-worker-b-gpu", type=_lazy_single_gpu_arg, default=ServerArgs.inferswarm_d3_worker_b_gpu, help="D3 worker B GPU UUID or visible selector")

    parser.add_argument(
        "--inferswarm-remote-mode",
        choices=("overlap", "serialized"),
        default=ServerArgs.inferswarm_remote_mode,
        help=(
            "InferSwarm remote scheduling: overlap is the P4 candidate; serialized is a diagnostic P3-equivalent control"
        ),
    )

    parser.add_argument(
        "--inferswarm-mechanism-max-steps",
        type=_nonnegative_int,
        default=ServerArgs.inferswarm_mechanism_max_steps,
        help=(
            "Retain bounded per-step/per-layer InferSwarm mechanism records; cumulative F-gate counters continue after this capacity"
        ),
    )

    parser.add_argument(
        "--inferswarm-correctness-diagnostics",
        action="store_true",
        default=ServerArgs.inferswarm_correctness_diagnostics,
        help=(
            "Correctness-only InferSwarm C3 capture of exact generated token IDs and "
            "step-0 logits through the idle instrumentation snapshot. Disabled by default; "
            "must not be enabled for performance measurement"
        ),
    )

    parser.add_argument(
        "--max-running-requests",
        type=int,
        dest="max_running_req",
        default=ServerArgs.max_running_req,
        help="The maximum number of running requests.",
    )

    parser.add_argument(
        "--max-seq-len-override",
        type=int,
        default=ServerArgs.max_seq_len_override,
        help="The maximum sequence length override.",
    )

    parser.add_argument(
        "--max-output-tokens",
        type=_positive_int,
        default=ServerArgs.max_output_tokens,
        help="Default max output tokens for requests that omit one (default 32k).",
    )

    parser.add_argument(
        "--memory-ratio",
        type=float,
        default=ServerArgs.memory_ratio,
        help=(
            "Fraction of total GPU free memory the engine may use for weights + MoE cache + KV cache combined; the remainder is reserved runtime headroom."
        ),
    )

    assert ServerArgs.use_dummy_weight == False
    parser.add_argument(
        "--dummy-weight",
        action="store_true",
        dest="use_dummy_weight",
        help="Use dummy weights for testing.",
    )

    assert ServerArgs.use_pynccl == True
    parser.add_argument(
        "--disable-pynccl",
        action="store_false",
        dest="use_pynccl",
        help="Disable PyNCCL for tensor parallelism.",
    )

    parser.add_argument(
        "--host",
        type=str,
        dest="server_host",
        default=ServerArgs.server_host,
        help="The host address for the server.",
    )

    parser.add_argument(
        "--port",
        type=int,
        dest="server_port",
        default=ServerArgs.server_port,
        help="The port number for the server to listen on.",
    )

    parser.add_argument(
        "--cuda-graph-max-bs",
        "--graph",
        type=int,
        default=ServerArgs.cuda_graph_max_bs,
        help="The maximum batch size for CUDA graph capture. None means auto-tuning based on the GPU memory.",
    )

    parser.add_argument(
        "--num-tokenizer",
        "--tokenizer-count",
        type=int,
        default=ServerArgs.num_tokenizer,
        help="The number of tokenizer processes to launch. 0 means the tokenizer is shared with the detokenizer.",
    )

    parser.add_argument(
        "--max-prefill-length",
        "--max-extend-length",
        type=int,
        dest="max_extend_tokens",
        default=ServerArgs.max_extend_tokens,
        help="Chunk Prefill maximum chunk size in tokens.",
    )

    parser.add_argument(
        "--decode-log-interval",
        type=_positive_int,
        default=ServerArgs.decode_log_interval,
        help="Print one decode scheduler status line every N decode forwards.",
    )

    kv_capacity_group = parser.add_mutually_exclusive_group()
    kv_capacity_group.add_argument(
        "--num-pages",
        dest="num_page_override",
        type=int,
        default=ServerArgs.num_page_override,
        help="Set the maximum number of pages for KVCache.",
    )

    kv_capacity_group.add_argument(
        "--num-tokens",
        dest="num_token_override",
        type=int,
        default=ServerArgs.num_token_override,
        help=(
            "Total KV-cache capacity in tokens; must be a multiple of the resolved page size (DSV4: 128 window page, TRTLLM backend: 64). Mutually exclusive with --num-pages."
        ),
    )

    parser.add_argument(
        "--page-size",
        type=int,
        default=ServerArgs.page_size,
        help="Set the page size for system management.",
    )

    parser.add_argument(
        "--attention-backend",
        "--attn",
        type=validate_attn_backend,
        default=ServerArgs.attention_backend,
        help="The attention backend to use. If two backends are specified, the first one is used for prefill and the second one for decode.",
    )

    parser.add_argument(
        "--model-source",
        type=str,
        default="huggingface",
        choices=["huggingface", "modelscope"],
        help="The source to download model from. Either 'huggingface' or 'modelscope'.",
    )

    parser.add_argument(
        "--cache-type",
        type=str,
        default=ServerArgs.cache_type,
        choices=SUPPORTED_CACHE_MANAGER.supported_names(),
        help="KV cache strategy (naive | radix). For hybrid GDN models 'radix' is materialized as a GDN-aware radix (cross-request GDN-state prefix reuse); pass 'naive' to opt out.",
    )

    parser.add_argument(
        "--enable-cache-report",
        action="store_true",
        default=ServerArgs.enable_cache_report,
        help=(
            "Return the number of prefix-cached prompt tokens in each response's usage block "
            "(OpenAI usage.prompt_tokens_details.cached_tokens, Anthropic "
            "usage.cache_read_input_tokens, Responses usage.input_tokens_details.cached_tokens). "
            "On /v1/messages this also makes input_tokens EXCLUDE the cached prefix, matching "
            "Anthropic billing semantics."
        ),
    )

    parser.add_argument(
        "--sampling-defaults",
        type=str,
        default=ServerArgs.sampling_defaults,
        choices=["model", "none"],
        help=(
            "Source for unspecified request sampling params. 'model' fills "
            "temperature/top_k/top_p from the checkpoint's generation_config.json "
            "(recommended for reasoning models to avoid greedy repetition loops); "
            "'none' uses framework defaults only."
        ),
    )

    parser.add_argument(
        "--served-model-name",
        type=str,
        default=ServerArgs.served_model_name,
        help="Model id returned by /v1/models. Defaults to the basename of --model.",
    )

    parser.add_argument(
        "--tool-call-parser",
        type=str,
        default="auto",
        choices=[
            "auto",
            "llama3",
            "qwen",
            "qwen25",
            "qwen3_coder",
            "mistral",
            "deepseekv32",
            "gemma4",
            "glm47",
            "minimax",
            "minimax_m3",
            "muse_glimmer",
            "gpt_oss",
            "gpt-oss",
        ],
        help="Tool-call parser format for OpenAI-compatible tool responses.",
    )

    parser.add_argument(
        "--reasoning-parser",
        type=str,
        default="auto",
        choices=[
            "auto",
            "off",
            "deepseekv32",
            "gpt_oss",
            "qwen3",
            "glm",
            "minimax",
            "minimax_m3",
            "muse_glimmer",
            "gemma4",
        ],
        help=(
            "Reasoning parser that splits chain-of-thought into reasoning_content "
            "for OpenAI responses. 'auto' selects per model family (gpt-oss Harmony, "
            "<think> for qwen3/glm/minimax, <mm:think> for minimax-m3, ATEM to=self "
            "channels for muse-glimmer, gemma thought, dsv4); 'off' disables it."
        ),
    )

    parser.add_argument(
        "--moe-backend",
        default=ServerArgs.moe_backend,
        choices=["auto"] + SUPPORTED_MOE_BACKENDS.supported_names(),
        help=(
            "The MoE backend to use. 'auto' resolves a MoE model to the offload family "
            "(offload, or hybrid when a `ft bench bw` profile recommends it); resident "
            "'fused' experts must be requested explicitly."
        ),
    )

    parser.add_argument(
        "--nvfp4-backend",
        default=ServerArgs.nvfp4_backend,
        choices=["auto", "marlin", "flashinfer", "triton"],
        help=(
            "NVFP4 routed-expert GEMM backend (default: triton, the portable inline-dequant "
            "kernel). auto picks by GPU (marlin on sm80-99 + vLLM; flashinfer b12x on sm120+ "
            "& CUDA>=13; else triton). Force one to override; it fails loudly if it cannot run."
        ),
    )

    parser.add_argument(
        "--expert-load",
        default=ServerArgs.expert_load,
        choices=["auto", "serial", "parallel"],
        help=(
            "How MoE expert banks are read into host RAM. 'auto' (default) reads scattered "
            "experts in parallel (fast) but falls back to serial when free RAM can't cover "
            "the banks + the parallel reader's extra whole-shard buffer; 'serial' forces the "
            "low-memory reclaimable read (slower); 'parallel' forces the fast read."
        ),
    )

    moe_cache_group = parser.add_mutually_exclusive_group()
    moe_cache_group.add_argument(
        "--moe-cache-size",
        type=int,
        default=ServerArgs.moe_cache_size,
        help="The number of unified MoE expert slots on GPU.",
    )
    moe_cache_group.add_argument(
        "--moe-cache-rate",
        type=_parse_moe_cache_rate,
        default=ServerArgs.moe_cache_rate,
        help="The fraction of all MoE experts to keep in GPU cache.",
    )
    moe_cache_group.add_argument(
        "--moe-cache-auto",
        action="store_true",
        default=ServerArgs.moe_cache_auto,
        help=(
            "Auto-pick --moe-cache-size from free VRAM and expert size, MoE-priority (KV gets --kv-reserve-tokens as a floor). Not supported for owned-KV models."
        ),
    )

    parser.add_argument(
        "--kv-reserve-tokens",
        type=int,
        default=ServerArgs.kv_reserve_tokens,
        help="KV-cache token floor reserved before --moe-cache-auto fills experts.",
    )

    parser.add_argument(
        "--moe-cache-policy",
        default=ServerArgs.moe_cache_policy,
        choices=["lru"],
        help="The unified MoE cache eviction policy.",
    )

    parser.add_argument(
        "--moe-cpu-threads",
        type=int,
        default=ServerArgs.moe_cpu_threads,
        help=(
            "Number of CPU worker threads for --moe-backend cpu decode experts. 0 = auto (physical cores)."
        ),
    )

    parser.add_argument(
        "--moe-cpu-layers",
        type=str,
        default=ServerArgs.moe_cpu_layers,
        help=(
            "With --moe-backend offload/hybrid: which MoE layers compute on the "
            "CPU executor instead of the GPU offload/PCIe path (where CUDA pinning "
            "is quota-capped, e.g. WSL, their banks are OS-locked instead of pinned). Explicit id list ('3,7,11'), a count ('8' = 8 "
            "layers evenly strided), or a fraction ('0.5'). Unset = automatic where "
            "CUDA pinning is quota-capped, e.g. WSL (locks just enough head+tail "
            "layers when the banks exceed the pin budget, none otherwise); '0' "
            "forces all layers on GPU."
        ),
    )

    parser.add_argument(
        "--moe-hybrid-max-fetch",
        type=int,
        default=ServerArgs.moe_hybrid_max_fetch,
        help=(
            "For --moe-backend hybrid: max experts fetched over PCIe per (layer, decode "
            "step); the rest of that step's misses are computed on the CPU, overlapped. "
            "-1 (default) = auto: fetch the benched pcie/cpu bandwidth fraction of each "
            "step's misses (perfect overlap; needs an `ft bench bw` profile, else 1). "
            "0 = never fetch (all misses on CPU); large = behaves like plain offload."
        ),
    )

    parser.add_argument(
        "--disable-moe-prefill-overlap",
        action="store_false",
        dest="moe_prefill_overlap",
        default=ServerArgs.moe_prefill_overlap,
        help=(
            "Disable two-buffer overlap for prefill MoE expert copies. By default, prefill overlap is enabled and requires --moe-cache-size >= 2 * num_experts."
        ),
    )

    parser.add_argument(
        "--enable-special-token-ckpt",
        action="store_true",
        dest="special_token_ckpt",
        default=ServerArgs.special_token_ckpt,
        help=(
            "Checkpoint decode state at special tokens (currently the tool-call opener). "
            "When a GDN-hybrid or SWA model samples its tool-call opener token, the "
            "scheduler preserves a reuse point just after it (GDN: a state snapshot "
            "donated to the prefix cache; SWA: the trailing window is kept resumable), so "
            "a client that rewrites the echoed tool call only invalidates the call body, "
            "not the turn."
        ),
    )

    parser.add_argument(
        "--moe-prefill-hit-d2d",
        action="store_true",
        dest="moe_prefill_hit_d2d",
        default=ServerArgs.moe_prefill_hit_d2d,
        help=(
            "During prefill prefetch, copy cache-resident experts device-side into "
            "the double buffer and stream only the misses over PCIe "
            "(cudaMemcpyBatchAsync, CUDA >= 13.0). Effective with "
            "--moe-cache-size > 2 * num_experts."
        ),
    )

    parser.add_argument(
        "--moe-collect-stats",
        action="store_true",
        default=ServerArgs.moe_collect_stats,
        help=(
            "Opt in to device-side MoE decode hit/miss/fetch counters. Disabled by default; read/reset only through POST /v1/moe/instrumentation at an idle boundary."
        ),
    )

    parser.add_argument(
        "--moe-trace-max-steps",
        type=_nonnegative_int,
        default=ServerArgs.moe_trace_max_steps,
        help=(
            "Retain exact per-step/per-layer decode expert selections for at most this many "
            "steps (0 disables and allocates no trace VRAM). Exact tracing requires "
            "--cuda-graph-max-bs 0 so replay cannot return capture-time routes."
        ),
    )

    parser.add_argument(
        "--moe-layer-timing-max-steps",
        type=_nonnegative_int,
        default=ServerArgs.moe_layer_timing_max_steps,
        help=(
            "Retain complete routed-MoE layer timing for this many decode steps (0 disables; compatible with CUDA graph replay)"
        ),
    )

    parser.add_argument(
        "--moe-layer-timing-role",
        choices=("unspecified", "baseline", "candidate"),
        default=ServerArgs.moe_layer_timing_role,
        help="Optional diagnostic provenance for the common MoE timing schema",
    )

    parser.add_argument(
        "--shell-mode",
        action="store_true",
        help="Run the server in shell mode.",
    )

    parser.add_argument(
        "--cors-origins",
        type=str,
        default=ServerArgs.cors_origins,
        help=(
            "Comma-separated CORS allow-list for browser/webview clients (default: local Tauri/Vite dev origins). '' disables, '*' allows any."
        ),
    )

    # Parse arguments
    kwargs = parser.parse_args(args).__dict__.copy()

    # reject a too-long list here with a clear reason, not as a dead rank later
    if len(kwargs["gpu"]) not in (0, kwargs["tensor_parallel_size"]):
        if kwargs["tensor_parallel_size"] == 1 and len(kwargs["gpu"]) > 1:
            parser.error(
                "tensor parallelism is not supported yet: --gpu takes one entry"
            )
        parser.error(
            f"--gpu has {len(kwargs['gpu'])} entries but --tensor-parallel-size is {kwargs['tensor_parallel_size']}; give one entry per TP rank"
        )
    if (
        kwargs["inferswarm_secondary_gpu"] is not None
        and kwargs["tensor_parallel_size"] != 1
    ):
        parser.error(
            "--inferswarm-secondary-gpu is a two-device Phase-1 POC option and currently requires --tensor-parallel-size 1; it is not a TP rank"
        )
    if (
        kwargs["inferswarm_placement"] is not None
        and kwargs["inferswarm_secondary_gpu"] is None
    ):
        parser.error("--inferswarm-placement requires --inferswarm-secondary-gpu")
    if (
        kwargs["inferswarm_remote_decode"]
        and kwargs["inferswarm_secondary_gpu"] is None
    ):
        parser.error("--inferswarm-remote-decode requires --inferswarm-secondary-gpu")
    if kwargs["inferswarm_remote_decode"] and kwargs["inferswarm_placement"] is None:
        parser.error("--inferswarm-remote-decode requires --inferswarm-placement")
    d2 = kwargs["inferswarm_experimental_d2_graph_remote"]
    d3 = kwargs["inferswarm_experimental_d3_graph_multiworker"]
    d4 = kwargs["inferswarm_experimental_d4_capability_weighted"]
    d5_loader = kwargs["inferswarm_experimental_d5_resident_loader"]
    d5_compact = kwargs["inferswarm_experimental_d5_compact_routes"]
    multiworker = d3 or d5_compact
    if d2 and kwargs["inferswarm_remote_decode"]:
        parser.error(
            "--inferswarm-experimental-d2-graph-remote is mutually exclusive with "
            "--inferswarm-remote-decode"
        )
    if d2 and kwargs["inferswarm_secondary_gpu"] is None:
        parser.error(
            "--inferswarm-experimental-d2-graph-remote requires --inferswarm-secondary-gpu"
        )
    if d2 and kwargs["inferswarm_placement"] is None:
        parser.error(
            "--inferswarm-experimental-d2-graph-remote requires --inferswarm-placement"
        )
    if d3 and (d2 or kwargs["inferswarm_remote_decode"]):
        parser.error("--inferswarm-experimental-d3-graph-multiworker is mutually exclusive with canonical and D2 remote decode")
    d3_active = kwargs["inferswarm_d3_active_workers"]
    required_d3_workers = (("a", kwargs["inferswarm_d3_worker_a_gpu"]), ("b", kwargs["inferswarm_d3_worker_b_gpu"]))
    if d4 and not d3:
        parser.error("D4 capability weighting requires the unchanged D3 graph executor flag")
    if d5_loader and not multiworker:
        parser.error("D5 resident loader requires a D3/D4 worker configuration")
    if not d5_loader and kwargs["inferswarm_d5_loader_cpu_workers"] != 4:
        parser.error("--inferswarm-d5-loader-cpu-workers requires --inferswarm-experimental-d5-resident-loader")
    if d5_compact and (d3 or d4 or d2 or kwargs["inferswarm_remote_decode"]):
        parser.error("D5 compact routes is a separate executor and cannot be combined with historical D3/D4/D2/canonical remote flags")
    if d5_compact and not d5_loader:
        parser.error("D5 compact routes requires the frozen D5 resident loader")
    placement = kwargs["inferswarm_d4_placement"] if d4 else kwargs["inferswarm_d3_placement"]
    if multiworker and (placement is None or any(selector is None for label, selector in required_d3_workers if label in d3_active)):
        parser.error(f"D3/D4 graph multiworker shape {d3_active!r} requires its placement and active worker selector(s)")
    if d4 and kwargs["inferswarm_d3_placement"] is not None:
        parser.error("D4 must use --inferswarm-d4-placement")
    if not d4 and kwargs["inferswarm_d4_placement"] is not None:
        parser.error("--inferswarm-d4-placement requires the D4 experimental flag")
    if not multiworker and (kwargs["inferswarm_d3_active_workers"] != "ab" or any(kwargs[name] is not None for name in ("inferswarm_d3_placement", "inferswarm_d4_placement", "inferswarm_d3_worker_a_gpu", "inferswarm_d3_worker_b_gpu"))):
        parser.error("D3 placement and worker selectors require --inferswarm-experimental-d3-graph-multiworker")

    # resolve some arguments
    run_shell |= kwargs.pop("shell_mode")
    kwargs["shell_mode"] = run_shell
    if run_shell:
        kwargs["cuda_graph_max_bs"] = 1
        kwargs["max_running_req"] = 1
        kwargs["silent_output"] = True

    if kwargs["moe_trace_max_steps"] > 0 and kwargs["cuda_graph_max_bs"] != 0:
        parser.error(
            "--moe-trace-max-steps requires --cuda-graph-max-bs 0; exact routing cannot run under CUDA graph replay"
        )
    if kwargs["inferswarm_remote_decode"] and kwargs["cuda_graph_max_bs"] != 0:
        parser.error("--inferswarm-remote-decode requires --cuda-graph-max-bs 0")
    if d2 and kwargs["cuda_graph_max_bs"] != 1:
        parser.error(
            "--inferswarm-experimental-d2-graph-remote requires --cuda-graph-max-bs 1"
        )
    if d2 and kwargs["max_running_req"] != 1:
        parser.error(
            "--inferswarm-experimental-d2-graph-remote requires --max-running-requests 1"
        )
    if d3 and kwargs["cuda_graph_max_bs"] != 1:
        parser.error("D3 graph multiworker requires --cuda-graph-max-bs 1")
    if d3 and kwargs["max_running_req"] != 1:
        parser.error("D3 graph multiworker requires --max-running-requests 1")

    if kwargs["model_path"].startswith("~"):
        kwargs["model_path"] = os.path.expanduser(kwargs["model_path"])
    if kwargs["inferswarm_placement"] is not None:
        kwargs["inferswarm_placement"] = os.path.expanduser(
            kwargs["inferswarm_placement"]
        )

    if kwargs["served_model_name"] is None:
        kwargs["served_model_name"] = (
            os.path.basename(os.path.normpath(kwargs["model_path"]))
            or kwargs["model_path"]
        )

    if kwargs["tool_call_parser"] == "auto":
        kwargs["tool_call_parser"] = _infer_tool_call_parser(kwargs["model_path"])

    if kwargs["reasoning_parser"] == "auto":
        kwargs["reasoning_parser"] = _infer_reasoning_parser(kwargs["model_path"])
    elif kwargs["reasoning_parser"] == "off":
        kwargs["reasoning_parser"] = None

    # Offload-family backends (offload/cpu/hybrid) need a slot cache; if the user gave no
    # sizing flag at all, default to --moe-cache-auto so a bare `ft serve <FTW MoE>` works
    # out of the box (the scheduler resolves the size from free VRAM). Explicit
    # size/rate/auto is preserved.
    from freetoken.moe import is_offload_moe_backend

    _no_cache_flag = (
        kwargs["moe_cache_size"] == 0
        and not kwargs["moe_cache_auto"]
        and (kwargs["moe_cache_rate"] is None or kwargs["moe_cache_rate"] == 0)
    )
    if is_offload_moe_backend(kwargs["moe_backend"]) and _no_cache_flag:
        kwargs["moe_cache_auto"] = True

    if kwargs["model_source"] == "modelscope":
        model_path = kwargs["model_path"]
        if not os.path.isdir(model_path):
            from modelscope import snapshot_download

            ignore_patterns = []
            if kwargs["use_dummy_weight"]:
                ignore_patterns = ["*.bin", "*.safetensors", "*.pt", "*.ckpt"]
            model_path = snapshot_download(model_path, ignore_patterns=ignore_patterns)
            kwargs["model_path"] = model_path
    del kwargs["model_source"]

    # "auto" (or an unspecified dtype) resolves to the checkpoint's dtype. Multimodal /
    # hybrid configs (e.g. Qwen3.5-MoE) keep it under ``text_config`` and use the newer
    # ``dtype`` key rather than top-level ``torch_dtype``, so check both; default bf16.
    if (dtype_str := kwargs["dtype"]) in ("auto", None):
        from freetoken.utils import cached_load_hf_config

        cfg = cached_load_hf_config(kwargs["model_path"]).to_dict()
        text_cfg = cfg.get("text_config") or {}
        dtype_str = (
            cfg.get("torch_dtype")
            or cfg.get("dtype")
            or text_cfg.get("torch_dtype")
            or text_cfg.get("dtype")
            or "bfloat16"
        )

    DTYPE_MAP = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    kwargs["dtype"] = DTYPE_MAP[dtype_str] if isinstance(dtype_str, str) else dtype_str
    kwargs["tp_info"] = DistributedInfo(0, kwargs["tensor_parallel_size"])
    del kwargs["tensor_parallel_size"]

    result = ServerArgs(**kwargs)
    logger = init_logger(__name__)
    logger.info(f"Parsed arguments:\n{result}")
    return result, run_shell
