"""End-to-end campaign behaviour with the server, GPU and HTTP layer mocked out.

Nothing here needs CUDA, a checkpoint, or a network. What is exercised is the part that can
fail silently: whether every repetition reaches the artifacts, whether a run that lost
repetitions can still look successful, whether a run that satisfied every repetition but
broke a precommitted rule can still look *valid*, and whether a non-canonical run is
labelled as one.

The governing invariant, which most of these tests are one instance of:

    a canonical-looking Phase-0 artifact exists only when every precommitted prerequisite,
    held-constant rule, workload-shape rule, instrumentation requirement, provenance
    requirement and repetition requirement was actually satisfied.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from .fakes import (
    FAKE_UUID,
    SHA40,
    good_bench_bw_record,
    instrumentation_doc,
    runtime_config,
    write_manifest,
)

from inferswarm_phase0 import validity as V
from inferswarm_phase0.artifacts import STATUS_COMPLETE, STATUS_INCOMPLETE, block_stats
from inferswarm_phase0.baselines import BASELINE_ARMS_BY_ID, correctness_reference_arm
from inferswarm_phase0.manifest import load_manifest
from inferswarm_phase0.protocol import build_protocol
from inferswarm_phase0.runner import Campaign, ServeSettings
from inferswarm_phase0 import runner as runner_mod

ALL_CLASSES = ("W1", "W2", "W3", "W4")


def _manifest(tmp_path, classes=ALL_CLASSES, canonical=True, sampling=None):
    path = write_manifest(tmp_path, classes, canonical=canonical, sampling=sampling)
    return load_manifest(path, canonical=canonical)


def _campaign(
    tmp_path, *, arms=None, classes=ALL_CLASSES, canonical=True, manifest=None,
    dev_smoke=None, **protocol_kw
):
    """A campaign whose every gate would pass, so a test only has to break one thing.

    ``canonical`` is the *run's* intent and ``dev_smoke`` is the protocol override; they are
    separate because ``--allow-missing-provenance`` turns the first off while leaving the
    second alone, which is exactly the case a protocol-keyed banner would miss.
    """
    kw = dict(warmups=None, repetitions=None, session_id="session-1",
              reverse_order=False,
              dev_smoke=(not canonical) if dev_smoke is None else dev_smoke)
    kw.update(protocol_kw)
    return Campaign(
        arms=arms or [BASELINE_ARMS_BY_ID["B1"]],
        manifest=manifest or _manifest(tmp_path, classes, canonical=canonical),
        protocol=build_protocol(**kw),
        settings=ServeSettings(
            model_path=str(tmp_path / "model"),
            model_repository="nvidia/Qwen3.6-35B-A3B-NVFP4",
            model_revision=SHA40,
            gpu=FAKE_UUID,
            python_executable="python",
        ),
        out_root=tmp_path / "runs",
        short_name="unit",
        inferswarm_commit="b" * 40,
        canonical=canonical,
        echo_server_output=False,
    )


@pytest.fixture(autouse=True)
def complete_provenance(monkeypatch, resolved_gpu):
    """CI has no GPU, and a canonical run rightly refuses a document with holes in it.

    Supply the host-dependent blocks so the tests below exercise the campaign rather than
    re-testing the provenance gate (which has its own tests, including every refusal). The
    FreeToken checkout is reported clean for the same reason: a developer's working tree is
    usually dirty, and the dirty-tree refusal has its own dedicated test.
    """
    monkeypatch.setattr(
        runner_mod.prov, "git_commit",
        lambda repo_dir: {"value": "a" * 40, "dirty": False, "dirty_paths": []},
    )
    monkeypatch.setattr(
        runner_mod.prov, "gpu_provenance",
        lambda selector=None, resolved_uuid=None: {
            "gpus": [{"index": "0", "uuid": FAKE_UUID,
                      "name": "NVIDIA GeForce RTX 3060", "selected": True}],
            "topology": "GPU0\tX",
            "selected": {"requested": selector, "resolved_uuid": resolved_uuid},
        },
    )
    monkeypatch.setattr(
        runner_mod.prov, "host_provenance",
        lambda: {"os": "Linux 6.0", "cpu_model": "Test CPU", "ram_total_bytes": 64 << 30,
                 "environment": {}},
    )
    monkeypatch.setattr(runner_mod.prov, "_torch_versions", lambda: {"torch": "2.11.0", "cuda": "13.0"})


class _FakeHandle:
    def __init__(self):
        self.proc = None


@pytest.fixture
def mocked_server(monkeypatch):
    """Replace every process/HTTP boundary; record what the runner asked for.

    The defaults describe a run in which nothing went wrong, so any test that ends up
    INVALID did so because that test broke something specific.
    """
    calls = {"generations": [], "instrumentation": 0, "started": [], "stopped": 0,
             "order": [], "bench_bw": []}

    def fake_start_server(command, origin, log_path, **kwargs):
        calls["started"].append({"command": list(command), "env": kwargs.get("env_overrides")})
        calls["order"].append(("serve", _arm_of(command)))
        Path(log_path).write_text("fake server log\n")
        return _FakeHandle()

    def fake_fetch_instrumentation(origin, limit=8):
        calls["instrumentation"] += 1
        return instrumentation_doc()

    def fake_measure(origin, body, *, prefill_seq_floor, store_text, **kwargs):
        calls["generations"].append(dict(body))
        # A prompt length inside each class's frozen shape, so a shape violation in a test
        # below is that test's doing.
        class_id = body["messages"][0]["content"].rsplit(" ", 1)[-1]
        prompt_tokens = {"W1": 1800, "W2": 900, "W3": 16000, "W4": 128}.get(class_id, 900)
        return {
            "ttft_ms": 100.0 + len(calls["generations"]),
            "decode_tok_s": 20.0 + len(calls["generations"]),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": body["max_tokens"],
            "requested_max_tokens": body["max_tokens"],
            "decode_steps": body["max_tokens"] - 1,
            "inter_token_ms": [50.0, 51.0],
            "output_sha256": "deadbeef",
            "output_text": "hello" if store_text else None,
            "output_text_stored": bool(store_text),
            "vram_bytes": 11 << 30,
            "prefill": {"gpu_ms": 40.0, "new_tokens": prompt_tokens,
                        "prefill_tok_s": prompt_tokens / 0.04},
            "prefill_status": {"ok": True, "code": "ok", "reason": None, "attribution": "uid"},
        }

    def fake_bench_bw(*, python_executable, selection, dtype="nvfp4", **kwargs):
        calls["bench_bw"].append({"gpu": selection.resolved_uuid, "dtype": dtype})
        calls["order"].append(("bench_bw", selection.resolved_uuid))
        return runner_mod.bench_bw_mod.BenchBwResult(good_bench_bw_record(selection.resolved_uuid))

    monkeypatch.setattr(runner_mod.bench_bw_mod, "run_bench_bw", fake_bench_bw)
    monkeypatch.setattr(runner_mod, "start_server", fake_start_server)
    monkeypatch.setattr(runner_mod, "stop_server", lambda h: calls.__setitem__("stopped", calls["stopped"] + 1))
    monkeypatch.setattr(runner_mod, "fetch_instrumentation", fake_fetch_instrumentation)
    monkeypatch.setattr(runner_mod, "measure_generation", fake_measure)
    monkeypatch.setattr(runner_mod, "prefill_seq_floor", lambda origin: 0)
    monkeypatch.setattr(runner_mod, "_model_id", lambda origin: "qwen-test")
    return calls


def _arm_of(command):
    """B-arm identity of a serve command, read back from its --moe-backend/--nvfp4-backend."""
    flags = list(command)
    backend = flags[flags.index("--moe-backend") + 1]
    nvfp4 = flags[flags.index("--nvfp4-backend") + 1]
    return {
        ("offload", "auto"): "B1", ("hybrid", "triton"): "B2", ("auto", "auto"): "B3",
        ("offload", "triton"): "B4", ("cpu", "triton"): "B5",
    }.get((backend, nvfp4), f"{backend}/{nvfp4}")


# --- completeness -----------------------------------------------------------------------

def test_a_complete_run_preserves_every_repetition(tmp_path, mocked_server):
    doc = _campaign(tmp_path).execute()
    assert doc["execution_status"] == STATUS_COMPLETE
    assert doc["validity"] == V.VALIDITY_VALID
    assert doc["headline"] == "VALID CANONICAL CAMPAIGN"
    # 4 classes x 10 measured, per criteria section 9 rule 1 ("all four must be run")
    assert doc["measured_repetition_count"] == 40

    reps = [json.loads(l) for l in (Path(doc["run_directory"]) / "repetitions.jsonl").read_text().splitlines()]
    # warmups are recorded too -- tagged, never dropped
    assert len(reps) == 48
    assert [r["phase"] for r in reps[:12]] == ["warmup"] * 2 + ["measured"] * 10
    assert [r["execution_index"] for r in reps] == list(range(48))
    assert {r["class_id"] for r in reps} == set(ALL_CLASSES)
    # the raw per-token timings survive, so variance/CV/bootstrap remain computable later
    assert all(r["inter_token_ms"] for r in reps)
    assert all(r["seed"] is None and r["seed_unavailable"] for r in reps)
    assert all(r["batch_size"] == 1 for r in reps)


def test_every_generation_sends_the_frozen_prompt_settings(tmp_path, mocked_server):
    _campaign(tmp_path).execute()
    bodies = mocked_server["generations"]
    assert len(bodies) == 48
    expected_max_tokens = {"W1": 512, "W2": 512, "W3": 256, "W4": 128}
    for body in bodies:
        assert body["ignore_eos"] is True
        assert body["temperature"] == 0.0 and body["top_p"] == 1.0 and body["top_k"] == -1
        content = body["messages"][0]["content"]
        class_id = content.rsplit(" ", 1)[-1]
        assert body["max_tokens"] == expected_max_tokens[class_id]


def test_the_run_records_the_resolved_configuration_not_just_the_flags(tmp_path, mocked_server):
    doc = _campaign(tmp_path).execute()
    resolved = doc["resolved_configuration"]["B1"]["instrumentation"]["runtime_config"]
    assert resolved["moe"]["backend_resolved"] == "offload"
    assert resolved["nvfp4"]["resolved"] == "triton"
    assert "marlin_cache_cap" in resolved
    # criteria section 2.3 fields the earlier report did not carry at all
    assert resolved["runtime"]["max_prefill_length_resolved"] == 8192
    assert resolved["runtime"]["cache_type_resolved"] == "radix"


def test_a_failed_generation_cannot_produce_a_complete_run(tmp_path, mocked_server, monkeypatch):
    from inferswarm_phase0.client import GenerationError

    original = runner_mod.measure_generation
    state = {"n": 0}

    def flaky(*args, **kwargs):
        state["n"] += 1
        if state["n"] == 5:
            raise GenerationError("simulated stream drop")
        return original(*args, **kwargs)

    monkeypatch.setattr(runner_mod, "measure_generation", flaky)
    doc = _campaign(tmp_path).execute()

    assert doc["execution_status"] == STATUS_INCOMPLETE
    assert doc["label"] == "INCOMPLETE"
    assert doc["headline"] == "INCOMPLETE RUN"
    assert doc["failure_count"] == 1
    assert doc["measured_repetition_count"] == 39
    assert doc["incomplete_blocks"]
    assert V.GENERATION_FAILED in doc["campaign_invalidation_codes"]
    summary = (Path(doc["run_directory"]) / "SUMMARY.md").read_text()
    assert "INCOMPLETE RUN" in summary.splitlines()[0]
    assert "**NO**" in summary
    failures = (Path(doc["run_directory"]) / "failures.jsonl").read_text().strip().splitlines()
    assert len(failures) == 1 and "simulated stream drop" in failures[0]


def test_a_dead_server_fails_every_step_of_that_arm(tmp_path, mocked_server, monkeypatch):
    from inferswarm_phase0.client import ServerError

    def dead(*args, **kwargs):
        raise ServerError("server exited with code 1 during startup", "traceback tail")

    monkeypatch.setattr(runner_mod, "start_server", dead)
    doc = _campaign(tmp_path).execute()
    assert doc["execution_status"] == STATUS_INCOMPLETE
    assert doc["measured_repetition_count"] == 0
    assert doc["failure_count"] == 48  # nothing is silently skipped
    assert V.SERVER_FAILED in doc["campaign_invalidation_codes"]


# --- canonical labelling ------------------------------------------------------------------

def test_a_non_canonical_run_is_labelled_everywhere(tmp_path, mocked_server):
    doc = _campaign(tmp_path, canonical=False, warmups=1, repetitions=2).execute()
    # There is no bare `canonical: true/false` in the artifact at all: `validity` is the
    # verdict, and a boolean beside it is the field a reader would mistake for one.
    assert "canonical" not in doc
    assert doc["canonical_intent"] is False
    assert doc["validity"] == V.VALIDITY_NON_CANONICAL
    assert doc["headline"] == "NON-CANONICAL DEVELOPER RUN"
    assert doc["canonical_blockers"]
    summary = (Path(doc["run_directory"]) / "SUMMARY.md").read_text()
    assert summary.splitlines()[0] == "# NON-CANONICAL DEVELOPER RUN"
    assert "must not be published as a Phase-0 baseline" in summary


def test_the_summary_uses_overall_validity_not_only_the_protocol(tmp_path, mocked_server):
    """--allow-missing-provenance leaves the repetition protocol untouched, yet the campaign
    is non-canonical. Keying the banner off protocol["canonical"] alone would miss it."""
    campaign = _campaign(tmp_path, canonical=False, dev_smoke=False)
    assert campaign.protocol.canonical is True  # the protocol itself was NOT overridden
    doc = campaign.execute()
    assert doc["protocol"]["canonical"] is True
    assert doc["validity"] == V.VALIDITY_NON_CANONICAL
    summary = (Path(doc["run_directory"]) / "SUMMARY.md").read_text()
    assert summary.splitlines()[0] == "# NON-CANONICAL DEVELOPER RUN"


def test_a_canonical_run_refuses_missing_provenance(tmp_path, mocked_server):
    campaign = _campaign(tmp_path)
    campaign.inferswarm_commit = None
    with pytest.raises(ValueError, match="--inferswarm-commit is required"):
        campaign.execute()


def test_the_summary_never_computes_a_cross_configuration_ratio(tmp_path, mocked_server):
    doc = _campaign(
        tmp_path, arms=[BASELINE_ARMS_BY_ID["B1"], BASELINE_ARMS_BY_ID["B4"]]
    ).execute()
    summary = (Path(doc["run_directory"]) / "SUMMARY.md").read_text()
    assert "does **not** select `CANONICAL_PERFORMANCE_BASELINE`" in summary
    for banned in ("speedup", "faster than", "winner", "R_agg"):
        assert banned not in summary.lower() or banned == "winner" and "winner" not in summary


def test_correctness_reference_output_text_is_always_retained(tmp_path, mocked_server):
    arm = correctness_reference_arm("triton", 512)
    campaign = _campaign(tmp_path, arms=[arm])
    campaign.store_output_text = True
    doc = campaign.execute()
    reps = [json.loads(l) for l in (Path(doc["run_directory"]) / "repetitions.jsonl").read_text().splitlines()]
    assert all(r["output_text_stored"] for r in reps)
    assert all(r["arm_role"] == "correctness" for r in reps)


def test_prefill_instrumentation_is_requested_for_every_arm(tmp_path, mocked_server):
    _campaign(tmp_path, arms=[BASELINE_ARMS_BY_ID["B1"], BASELINE_ARMS_BY_ID["B5"]]).execute()
    assert [s["env"] for s in mocked_server["started"]] == [
        {"FREETOKEN_INSTRUMENT_PREFILL": "1"}
    ] * 2


def test_rerunning_into_the_same_directory_does_not_blend_campaigns(tmp_path, mocked_server):
    first = _campaign(tmp_path).execute()
    second = _campaign(tmp_path).execute()
    assert first["run_directory"] == second["run_directory"]
    reps = (Path(second["run_directory"]) / "repetitions.jsonl").read_text().strip().splitlines()
    assert len(reps) == 48


# --- statistics ---------------------------------------------------------------------------

def test_block_stats_reports_dispersion_not_just_a_mean():
    stats = block_stats([10.0, 11.0, 12.0, 40.0])
    assert stats["n"] == 4
    assert stats["min"] == 10.0 and stats["max"] == 40.0
    assert stats["median"] == 11.5
    assert stats["cv_percent"] > 0
    assert "iqr" in stats
    # the outlier is described, never removed
    assert stats["max"] == 40.0


def test_block_stats_on_no_data_says_so():
    assert block_stats([]) == {"n": 0}
    assert block_stats([None, None]) == {"n": 0}


# --- held-constant configuration ------------------------------------------------------------

def test_the_resolved_weight_format_is_backfilled_from_the_arms(tmp_path, mocked_server):
    """The expert format is only knowable once an engine has loaded the banks, so it must
    not stay "server not started yet" for the life of the artifact."""
    doc = _campaign(tmp_path, arms=[BASELINE_ARMS_BY_ID["B1"], BASELINE_ARMS_BY_ID["B4"]]).execute()
    quant = doc["model_expert_quant_resolved"]
    assert quant["value"] == "nvfp4"
    assert quant["consistent_across_arms"] is True
    assert quant["per_arm"] == {"B1": "nvfp4", "B4": "nvfp4"}


def test_arms_disagreeing_on_the_weight_format_invalidates(tmp_path, mocked_server, monkeypatch):
    """Criteria section 3 rule 4 holds the weight format constant across arms."""
    formats = iter(["nvfp4", "fp8_block"])
    monkeypatch.setattr(
        runner_mod, "fetch_instrumentation",
        lambda origin, limit=8: instrumentation_doc(
            runtime_config(model={"expert_quant": next(formats)})
        ),
    )
    doc = _campaign(tmp_path, arms=[BASELINE_ARMS_BY_ID["B1"], BASELINE_ARMS_BY_ID["B4"]]).execute()
    quant = doc["model_expert_quant_resolved"]
    assert quant["consistent_across_arms"] is False
    assert doc["validity"] == V.VALIDITY_INVALID
    assert V.EXPERT_QUANT_MISMATCH in doc["campaign_invalidation_codes"]
    # ... and it is complete: completeness and validity are separate answers
    assert doc["execution_status"] == STATUS_COMPLETE


@pytest.mark.parametrize(
    "block,field,value",
    [
        ("runtime", "memory_ratio", 0.8),
        ("runtime", "max_prefill_length_resolved", 4096),
        ("runtime", "cache_type_resolved", "naive"),
        ("cache", "kv_reserve_tokens", 16384),
        ("moe", "cpu_threads", 4),
    ],
)
def test_a_held_constant_value_differing_across_arms_invalidates(
    tmp_path, mocked_server, monkeypatch, block, field, value
):
    configs = iter([runtime_config(), runtime_config(**{block: {field: value}})])
    monkeypatch.setattr(
        runner_mod, "fetch_instrumentation",
        lambda origin, limit=8: instrumentation_doc(next(configs)),
    )
    doc = _campaign(tmp_path, arms=[BASELINE_ARMS_BY_ID["B1"], BASELINE_ARMS_BY_ID["B4"]]).execute()
    assert doc["validity"] == V.VALIDITY_INVALID
    assert V.HELD_CONSTANT_MISMATCH in doc["campaign_invalidation_codes"]
    disagreements = doc["cross_arm_checks"]["held_constant"]["disagreements"]
    assert [d["field"] for d in disagreements] == [f"{block}.{field}"]


def test_the_resolved_cache_size_may_legitimately_differ_across_arms(tmp_path, mocked_server, monkeypatch):
    """Section 3 rule 2 asks for the resolved KV/cache capacity to be REPORTED, not equalized:
    an auto-sized cache is expected to land differently per backend."""
    configs = iter([
        runtime_config(),
        runtime_config(cache={"resolved_slots": 640}, runtime={"num_pages": 30000}),
    ])
    monkeypatch.setattr(
        runner_mod, "fetch_instrumentation",
        lambda origin, limit=8: instrumentation_doc(next(configs)),
    )
    doc = _campaign(tmp_path, arms=[BASELINE_ARMS_BY_ID["B1"], BASELINE_ARMS_BY_ID["B4"]]).execute()
    assert doc["validity"] == V.VALIDITY_VALID


# --- required runtime configuration ---------------------------------------------------------

@pytest.mark.parametrize("path", list(runner_mod.REQUIRED_RUNTIME_FIELDS))
def test_a_missing_required_resolved_field_invalidates(tmp_path, mocked_server, monkeypatch, path):
    block, field = path.split(".")
    monkeypatch.setattr(
        runner_mod, "fetch_instrumentation",
        lambda origin, limit=8: instrumentation_doc(runtime_config(**{block: {field: None}})),
    )
    doc = _campaign(tmp_path).execute()
    assert doc["validity"] == V.VALIDITY_INVALID
    codes = doc["campaign_invalidation_codes"]
    assert V.RUNTIME_CONFIG_MISSING_FIELD in codes
    assert any(path in i["message"] for i in doc["campaign_invalidations"])


def test_a_hybrid_arm_without_a_resolved_fetch_fraction_invalidates(tmp_path, mocked_server, monkeypatch):
    """The fresh bandwidth profile exists to set this split; a hybrid arm that resolved
    without one did not consume the profile the campaign produced."""
    monkeypatch.setattr(
        runner_mod, "fetch_instrumentation",
        lambda origin, limit=8: instrumentation_doc(
            runtime_config(moe={"backend_resolved": "hybrid",
                                "hybrid_fetch_fraction_resolved": None})
        ),
    )
    doc = _campaign(tmp_path, arms=[BASELINE_ARMS_BY_ID["B2"]]).execute()
    assert doc["validity"] == V.VALIDITY_INVALID
    assert V.RUNTIME_CONFIG_MISSING_FIELD in doc["campaign_invalidation_codes"]


def test_missing_instrumentation_invalidates(tmp_path, mocked_server, monkeypatch):
    monkeypatch.setattr(
        runner_mod, "fetch_instrumentation",
        lambda origin, limit=8: {"unavailable": "HTTP 404 from /v1/instrumentation"},
    )
    doc = _campaign(tmp_path).execute()
    assert doc["validity"] == V.VALIDITY_INVALID
    assert V.INSTRUMENTATION_UNAVAILABLE in doc["campaign_invalidation_codes"]


def test_an_empty_runtime_config_invalidates(tmp_path, mocked_server, monkeypatch):
    monkeypatch.setattr(
        runner_mod, "fetch_instrumentation",
        lambda origin, limit=8: {
            "runtime_config": None,
            "runtime_config_unavailable": "readiness metadata has not been received yet",
            "prefill": {"enabled": True, "observed": 0, "records": []},
        },
    )
    doc = _campaign(tmp_path).execute()
    assert doc["validity"] == V.VALIDITY_INVALID
    assert V.RUNTIME_CONFIG_MISSING in doc["campaign_invalidation_codes"]


# --- B3's resolution ------------------------------------------------------------------------

def _sweep_configs(b3_backend, b3_nvfp4="triton"):
    """Per-arm instrumentation for a B1/B2/B3 sweep, with B3 resolving to ``b3_backend``."""
    per_arm = {
        "B1": runtime_config(),
        "B2": runtime_config(
            moe={"backend_requested": "hybrid", "backend_resolved": "hybrid"},
            nvfp4={"resolved": "not selected - native nvfp4 layout, Triton kernels",
                   "inert": True},
        ),
        "B3": runtime_config(
            moe={"backend_requested": "auto", "backend_resolved": b3_backend},
            nvfp4={"resolved": b3_nvfp4},
        ),
    }
    order = iter(["B1", "B2", "B3"])
    return lambda origin, limit=8: instrumentation_doc(per_arm[next(order)])


def test_b3_resolving_to_a_declared_path_is_valid(tmp_path, mocked_server, monkeypatch):
    monkeypatch.setattr(
        runner_mod, "fetch_instrumentation",
        _sweep_configs("hybrid", "not selected - native nvfp4 layout, Triton kernels"),
    )
    arms = [BASELINE_ARMS_BY_ID[i] for i in ("B1", "B2", "B3")]
    doc = _campaign(tmp_path, arms=arms).execute()
    check = doc["cross_arm_checks"]["b3_resolution"]
    assert check["coincides"] is True
    assert check["nvfp4_coincides"] is True
    assert doc["validity"] == V.VALIDITY_VALID


def test_b3_matching_the_backend_but_not_the_expert_path_invalidates(tmp_path, mocked_server, monkeypatch):
    """Coinciding on the MoE backend is not enough: B3 also resolves --nvfp4-backend auto,
    and a different resolution there is a different executing path."""
    monkeypatch.setattr(runner_mod, "fetch_instrumentation", _sweep_configs("offload", "marlin"))
    arms = [BASELINE_ARMS_BY_ID[i] for i in ("B1", "B2", "B3")]
    doc = _campaign(tmp_path, arms=arms).execute()
    check = doc["cross_arm_checks"]["b3_resolution"]
    assert check["coincides"] is True          # backend matches B1
    assert check["nvfp4_coincides"] is False   # ... but the expert path does not
    assert doc["validity"] == V.VALIDITY_INVALID
    assert V.B3_RESOLUTION_UNEXPECTED in doc["campaign_invalidation_codes"]


def test_b3_resolving_outside_b1_and_b2_invalidates(tmp_path, mocked_server, monkeypatch):
    """A new runtime policy is not silently accepted as part of the predeclared sweep."""
    monkeypatch.setattr(runner_mod, "fetch_instrumentation", _sweep_configs("fused"))
    arms = [BASELINE_ARMS_BY_ID[i] for i in ("B1", "B2", "B3")]
    doc = _campaign(tmp_path, arms=arms).execute()
    assert doc["validity"] == V.VALIDITY_INVALID
    assert V.B3_RESOLUTION_UNEXPECTED in doc["campaign_invalidation_codes"]
    # the data is preserved, not discarded
    assert doc["measured_repetition_count"] == 120
    check = doc["cross_arm_checks"]["b3_resolution"]
    assert check["b3_backend_resolved"] == "fused"
    assert check["expected"] == ["hybrid", "offload"]


# --- workload shape ---------------------------------------------------------------------------

def test_a_prompt_outside_its_frozen_class_shape_invalidates(tmp_path, mocked_server, monkeypatch):
    original = runner_mod.measure_generation

    def oversized(origin, body, **kwargs):
        record = original(origin, body, **kwargs)
        record["prompt_tokens"] = 5000  # W1's bound is 2000, W2's is 1000
        return record

    monkeypatch.setattr(runner_mod, "measure_generation", oversized)
    doc = _campaign(tmp_path, classes=("W1",), canonical=False).execute()
    assert V.PROMPT_SHAPE_VIOLATION in doc["campaign_invalidation_codes"]
    reps = [json.loads(l) for l in (Path(doc["run_directory"]) / "repetitions.jsonl").read_text().splitlines()]
    # the observation is preserved and the prompt was not rewritten
    assert all(r["prompt_tokens"] == 5000 for r in reps)
    assert all("exceeds the W1 bound" in r["prompt_token_deviation"] for r in reps)


def test_a_prompt_shape_violation_makes_a_canonical_campaign_invalid(tmp_path, mocked_server, monkeypatch):
    original = runner_mod.measure_generation

    def oversized(origin, body, **kwargs):
        record = original(origin, body, **kwargs)
        record["prompt_tokens"] = 5000
        return record

    monkeypatch.setattr(runner_mod, "measure_generation", oversized)
    doc = _campaign(tmp_path).execute()
    assert doc["execution_status"] == STATUS_COMPLETE
    assert doc["validity"] == V.VALIDITY_INVALID
    assert doc["headline"] == "INVALID CANONICAL ATTEMPT"


def test_w3_and_w4_target_bands_are_checked_not_only_the_upper_bound(tmp_path, mocked_server, monkeypatch):
    original = runner_mod.measure_generation

    def short(origin, body, **kwargs):
        record = original(origin, body, **kwargs)
        record["prompt_tokens"] = 900  # inside W3's 20000 ceiling, far below ~16000
        return record

    monkeypatch.setattr(runner_mod, "measure_generation", short)
    doc = _campaign(tmp_path, classes=("W3",), canonical=False).execute()
    assert V.PROMPT_SHAPE_VIOLATION in doc["campaign_invalidation_codes"]
    assert any("target band" in i["message"] for i in doc["campaign_invalidations"])


def test_a_completion_length_mismatch_invalidates(tmp_path, mocked_server, monkeypatch):
    """ignore_eos=true makes the output length exact; a short completion means the request
    did not run as declared, whatever the tokens it did produce."""
    original = runner_mod.measure_generation

    def truncated(origin, body, **kwargs):
        record = original(origin, body, **kwargs)
        record["completion_tokens"] = body["max_tokens"] - 3
        return record

    monkeypatch.setattr(runner_mod, "measure_generation", truncated)
    doc = _campaign(tmp_path).execute()
    assert doc["execution_status"] == STATUS_COMPLETE
    assert doc["validity"] == V.VALIDITY_INVALID
    assert V.COMPLETION_LENGTH_MISMATCH in doc["campaign_invalidation_codes"]
    reps = [json.loads(l) for l in (Path(doc["run_directory"]) / "repetitions.jsonl").read_text().splitlines()]
    assert all(r["completion_length_deviation"] for r in reps)


# --- prefill instrumentation ------------------------------------------------------------------

@pytest.mark.parametrize(
    "code,expected",
    [
        ("instrumentation_unavailable", V.PREFILL_UNAVAILABLE),
        ("instrumentation_disabled", V.PREFILL_DISABLED),
        ("no_fresh_record", V.PREFILL_MISSING),
        ("ambiguous_records", V.PREFILL_AMBIGUOUS),
        ("shared_batch", V.PREFILL_SHARED_BATCH),
        ("unusable_timing", V.PREFILL_UNUSABLE),
    ],
)
def test_every_unusable_prefill_state_invalidates(tmp_path, mocked_server, monkeypatch, code, expected):
    original = runner_mod.measure_generation

    def no_prefill(origin, body, **kwargs):
        record = original(origin, body, **kwargs)
        record["prefill"] = None
        record["prefill_status"] = {"ok": False, "code": code, "reason": f"simulated {code}"}
        return record

    monkeypatch.setattr(runner_mod, "measure_generation", no_prefill)
    doc = _campaign(tmp_path).execute()
    assert doc["validity"] == V.VALIDITY_INVALID
    assert expected in doc["campaign_invalidation_codes"]


def test_a_warmups_missing_prefill_does_not_invalidate(tmp_path, mocked_server, monkeypatch):
    """A warmup is discarded by construction (criteria section 10), so its prefill record is
    not part of the Phase-0 data -- unlike the fixture's shape and length, which belong to
    the block and are checked on every generation."""
    original = runner_mod.measure_generation
    seen = {"n": 0}

    def warmups_only(origin, body, **kwargs):
        record = original(origin, body, **kwargs)
        seen["n"] += 1
        if seen["n"] <= 2:  # the two warmups of the first block
            record["prefill"] = None
            record["prefill_status"] = {"ok": False, "code": "no_fresh_record",
                                        "reason": "simulated warmup race"}
        return record

    monkeypatch.setattr(runner_mod, "measure_generation", warmups_only)
    doc = _campaign(tmp_path).execute()
    assert doc["validity"] == V.VALIDITY_VALID


def test_a_measured_repetition_missing_prefill_does_invalidate(tmp_path, mocked_server, monkeypatch):
    original = runner_mod.measure_generation
    seen = {"n": 0}

    def third_only(origin, body, **kwargs):
        record = original(origin, body, **kwargs)
        seen["n"] += 1
        if seen["n"] == 3:  # the first MEASURED repetition
            record["prefill"] = None
            record["prefill_status"] = {"ok": False, "code": "no_fresh_record",
                                        "reason": "simulated"}
        return record

    monkeypatch.setattr(runner_mod, "measure_generation", third_only)
    doc = _campaign(tmp_path).execute()
    assert doc["validity"] == V.VALIDITY_INVALID
    assert V.PREFILL_MISSING in doc["campaign_invalidation_codes"]


def test_an_attributed_record_without_a_rate_still_invalidates(tmp_path, mocked_server, monkeypatch):
    original = runner_mod.measure_generation

    def zero_rate(origin, body, **kwargs):
        record = original(origin, body, **kwargs)
        record["prefill"] = {"gpu_ms": 0.0, "new_tokens": 900, "prefill_tok_s": None,
                             "prefill_tok_s_unavailable": "gpu_ms=0.0"}
        return record

    monkeypatch.setattr(runner_mod, "measure_generation", zero_rate)
    doc = _campaign(tmp_path).execute()
    assert doc["validity"] == V.VALIDITY_INVALID
    assert V.PREFILL_UNUSABLE in doc["campaign_invalidation_codes"]


def test_a_dev_smoke_run_records_a_null_prefill_without_claiming_validity(tmp_path, mocked_server, monkeypatch):
    original = runner_mod.measure_generation

    def no_prefill(origin, body, **kwargs):
        record = original(origin, body, **kwargs)
        record["prefill"] = None
        record["prefill_status"] = {"ok": False, "code": "instrumentation_disabled",
                                    "reason": "FREETOKEN_INSTRUMENT_PREFILL was not set"}
        return record

    monkeypatch.setattr(runner_mod, "measure_generation", no_prefill)
    doc = _campaign(tmp_path, canonical=False).execute()
    # explicit null + reason is acceptable for a smoke run, and the run stays NON_CANONICAL
    assert doc["validity"] == V.VALIDITY_NON_CANONICAL
    assert V.PREFILL_DISABLED in doc["campaign_invalidation_codes"]


# --- physical GPU ---------------------------------------------------------------------------

def test_the_resolved_uuid_is_what_every_child_process_is_given(tmp_path, mocked_server):
    campaign = _campaign(tmp_path)
    campaign.settings = campaign.settings.__class__(
        **{**campaign.settings.__dict__, "gpu": "0"}
    )
    campaign.gpu_selection = runner_mod.gpu_mod.resolve_gpu("0")
    campaign.execute()
    command = mocked_server["started"][0]["command"]
    assert command[command.index("--gpu") + 1] == FAKE_UUID


def test_a_runtime_gpu_mismatch_invalidates(tmp_path, mocked_server, monkeypatch):
    monkeypatch.setattr(
        runner_mod.gpu_mod, "engine_gpus",
        lambda origin: [{"index": 0, "uuid": "GPU-99999999-0000-0000-0000-000000000000"}],
    )
    doc = _campaign(tmp_path).execute()
    assert doc["validity"] == V.VALIDITY_INVALID
    assert V.GPU_MISMATCH in doc["campaign_invalidation_codes"]


def test_an_unprovable_runtime_gpu_invalidates(tmp_path, mocked_server, monkeypatch):
    monkeypatch.setattr(runner_mod.gpu_mod, "engine_gpus", lambda origin: [])
    doc = _campaign(tmp_path).execute()
    assert doc["validity"] == V.VALIDITY_INVALID
    assert V.GPU_UNPROVEN in doc["campaign_invalidation_codes"]


def test_a_canonical_run_refuses_a_gpu_it_cannot_resolve(tmp_path, mocked_server, monkeypatch):
    monkeypatch.setattr(runner_mod.gpu_mod, "_resolve_uuids", lambda selector: None)
    campaign = _campaign(tmp_path)
    with pytest.raises(ValueError, match="must resolve to a stable GPU UUID"):
        campaign.execute()
    assert mocked_server["started"] == []


# --- provenance refusals -----------------------------------------------------------------------

def test_a_dirty_freetoken_checkout_refuses_a_canonical_run(tmp_path, mocked_server, monkeypatch):
    monkeypatch.setattr(
        runner_mod.prov, "git_commit",
        lambda repo_dir: {"value": "a" * 40, "dirty": True,
                          "dirty_paths": [" M python/freetoken/engine/engine.py"]},
    )
    with pytest.raises(ValueError, match="working tree is dirty"):
        _campaign(tmp_path).execute()
    assert mocked_server["started"] == []


def test_a_dirty_checkout_names_the_modified_paths(tmp_path, mocked_server, monkeypatch):
    monkeypatch.setattr(
        runner_mod.prov, "git_commit",
        lambda repo_dir: {"value": "a" * 40, "dirty": True,
                          "dirty_paths": [" M python/freetoken/moe/offload_cache.py"]},
    )
    with pytest.raises(ValueError, match="offload_cache.py"):
        _campaign(tmp_path).execute()


def test_a_dev_smoke_run_proceeds_on_a_dirty_checkout_as_non_canonical(tmp_path, mocked_server, monkeypatch):
    monkeypatch.setattr(
        runner_mod.prov, "git_commit",
        lambda repo_dir: {"value": "a" * 40, "dirty": True, "dirty_paths": [" M x.py"]},
    )
    doc = _campaign(tmp_path, canonical=False).execute()
    assert doc["validity"] == V.VALIDITY_NON_CANONICAL
    assert doc["software"]["freetoken_commit"]["dirty"] is True


def test_a_short_inferswarm_commit_refuses_a_canonical_run(tmp_path, mocked_server):
    campaign = _campaign(tmp_path)
    campaign.inferswarm_commit = "abc1234"
    with pytest.raises(ValueError, match="not a 40-hex commit SHA"):
        campaign.execute()


def test_a_snapshot_revision_mismatch_refuses_a_canonical_run(tmp_path, mocked_server):
    snapshot = tmp_path / "hub" / "models--nvidia--Qwen3.6-35B-A3B-NVFP4" / "snapshots" / ("c" * 40)
    snapshot.mkdir(parents=True)
    campaign = _campaign(tmp_path)
    campaign.settings = campaign.settings.__class__(
        **{**campaign.settings.__dict__, "model_path": str(snapshot)}
    )
    with pytest.raises(ValueError, match="disagrees with --model-revision"):
        campaign.execute()
    assert mocked_server["started"] == []


def test_a_matching_snapshot_revision_is_accepted(tmp_path, mocked_server):
    snapshot = tmp_path / "hub" / "models--nvidia--Qwen3.6-35B-A3B-NVFP4" / "snapshots" / SHA40
    snapshot.mkdir(parents=True)
    campaign = _campaign(tmp_path)
    campaign.settings = campaign.settings.__class__(
        **{**campaign.settings.__dict__, "model_path": str(snapshot)}
    )
    doc = campaign.execute()
    assert doc["validity"] == V.VALIDITY_VALID
    identity = doc["model"]["snapshot_identity"]
    assert identity["value"] == SHA40
    assert identity["repository"] == "nvidia/Qwen3.6-35B-A3B-NVFP4"
