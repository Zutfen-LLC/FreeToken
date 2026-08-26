"""Frozen workload manifest: the four Phase-0/1 workload classes, hash-pinned.

Criteria section 9 fixes the classes and the selection rules; **InferSwarm issue #3 supplies
the realistic fixtures**. This module therefore defines the *container and its validation*,
not the content -- inventing W1/W3/W4 prompts here to make the harness look finished would
be exactly the cherry-picking section 9 rule 7 prohibits.

    W1  coding / agentic            prompt <=  2,000 tokens   output 512
    W2  open-ended reasoning        prompt <=  1,000 tokens   output 512
    W3  long context                prompt ~= 16,000 tokens   output 256
    W4  short interactive           prompt ~=    128 tokens   output 128

A manifest is a JSON file, version-controlled next to its fixtures, that pins for every
class: the fixture path (or inline content), the sha256 of the exact prompt bytes, the
output-token count, the sampling parameters, ``ignore_eos``, the chat-template settings,
and the seed (which must be null -- see ``_validate_seed``).

Validation runs *before* a server is started, so a manifest error costs seconds, not a
model load.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping

SCHEMA = "inferswarm.phase0.workload-manifest/1"

# Criteria section 9, table. ``prompt_max_tokens`` is a bound ("<=") for W1/W2 and a target
# ("~=") for W3/W4; ``prompt_tolerance`` expresses the difference honestly instead of
# pretending "approximately 16,000" is an equality.
@dataclass(frozen=True)
class ClassSpec:
    class_id: str
    character: str
    prompt_max_tokens: int
    prompt_target_tokens: int | None  # None => only the bound applies
    prompt_tolerance: float           # fractional tolerance around the target
    output_tokens: int


CLASS_SPECS: Dict[str, ClassSpec] = {
    "W1": ClassSpec("W1", "coding / agentic", 2000, None, 0.0, 512),
    "W2": ClassSpec("W2", "open-ended reasoning / conversation", 1000, None, 0.0, 512),
    "W3": ClassSpec("W3", "long context", 20000, 16000, 0.15, 256),
    "W4": ClassSpec("W4", "short interactive", 200, 128, 0.25, 128),
}
REQUIRED_CLASSES = ("W1", "W2", "W3", "W4")

# The criteria state W1/W2 as bounds ("<= 2,000", "<= 1,000") and W3/W4 as targets
# ("~ 16,000", "~ 128"). "~" is not a rule a machine can check, so the harness FREEZES the
# tolerance here -- before any measurement, in version control, recorded verbatim in every
# run artifact -- rather than choosing one after seeing a fixture's token count. Widening
# these numbers is a deliberate, reviewable change to a frozen contract (criteria section 9
# rule 3), never a per-run accommodation.
CLASS_SHAPE_RULE = {
    "source": "InferSwarm criteria section 9, table; '~' frozen by this harness",
    "W1": "prompt_tokens <= 2000 (criteria bound)",
    "W2": "prompt_tokens <= 1000 (criteria bound)",
    "W3": "prompt_tokens within +/-15% of 16000, and <= 20000 (criteria '~16,000')",
    "W4": "prompt_tokens within +/-25% of 128, and <= 200 (criteria '~128')",
    "output": (
        "completion_tokens must EQUAL the class's frozen output_tokens: canonical runs set "
        "ignore_eos=true (criteria section 3 rule 5), so the length is exact by construction "
        "and any deviation means the request did not run as declared"
    ),
}

# criteria section 5.3: FreeToken exposes no seed, so greedy decoding is the only
# reproducible fixture available. These are the exact request-level values that mean greedy
# in FreeToken's SamplingParams (python/freetoken/core.py).
CANONICAL_GREEDY_SAMPLING: Dict[str, Any] = {"temperature": 0.0, "top_p": 1.0, "top_k": -1}


class ManifestError(ValueError):
    """A manifest that cannot be trusted to reproduce a run. Always fatal."""


@dataclass(frozen=True)
class Workload:
    class_id: str
    prompt: str
    content_sha256: str
    output_tokens: int
    sampling: Dict[str, Any]
    ignore_eos: bool
    greedy: bool
    chat_template_kwargs: Dict[str, Any]
    role: str
    fixture_path: str | None
    description: str = ""

    def request_body(
        self, model_id: str, *, sampling_override: Mapping[str, Any] | None = None
    ) -> Dict[str, Any]:
        """The exact chat-completions body for this workload. Every generation field is
        stated explicitly: nothing is left for the server's own defaults to fill in.

        ``sampling_override`` replaces the manifest's frozen sampling for this request. It
        exists for exactly one caller: ``CORRECTNESS_REFERENCE``, which must be greedy
        (criteria section 5.3) even when the frozen performance sampling deliberately is
        not. ``--sampling-defaults none`` is not enough on its own -- the body below states
        temperature/top_p/top_k explicitly, and a request-level value always wins over a
        server default, so a sampled performance manifest would otherwise make the
        correctness reference sampled too. The performance sweep never passes this.
        """
        return {
            "model": model_id,
            "messages": [{"role": self.role, "content": self.prompt}],
            "max_tokens": self.output_tokens,
            "ignore_eos": self.ignore_eos,
            "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": dict(self.chat_template_kwargs),
            **(dict(sampling_override) if sampling_override is not None else self.sampling),
        }

    def greedy_reference_body(self, model_id: str) -> Dict[str, Any]:
        """The CORRECTNESS_REFERENCE request: same frozen prompt/content, greedy sampling."""
        return self.request_body(model_id, sampling_override=CANONICAL_GREEDY_SAMPLING)


@dataclass(frozen=True)
class Manifest:
    schema: str
    manifest_id: str
    canonical: bool
    workloads: List[Workload]
    source_path: str
    frozen_at: str | None = None
    notes: str = ""
    manifest_sha256: str = ""
    provenance: Dict[str, Any] = field(default_factory=dict)

    def by_class(self) -> Dict[str, Workload]:
        return {w.class_id: w for w in self.workloads}

    def missing_classes(self) -> List[str]:
        present = {w.class_id for w in self.workloads}
        return [c for c in REQUIRED_CLASSES if c not in present]

    def record(self) -> Dict[str, Any]:
        """The manifest as it goes into the run artifact: identity and hashes, not content.

        Prompt text is deliberately not copied here -- the sha256 is the pin, and a fixture
        may be large (W3 is ~16k tokens)."""
        return {
            "schema": self.schema,
            "manifest_id": self.manifest_id,
            "canonical": self.canonical,
            "frozen_at": self.frozen_at,
            "source_path": self.source_path,
            "manifest_sha256": self.manifest_sha256,
            "notes": self.notes,
            "provenance": dict(self.provenance),
            "workloads": [
                {
                    "class_id": w.class_id,
                    "fixture_path": w.fixture_path,
                    "content_sha256": w.content_sha256,
                    "output_tokens": w.output_tokens,
                    "sampling": dict(w.sampling),
                    "ignore_eos": w.ignore_eos,
                    "greedy": w.greedy,
                    "chat_template_kwargs": dict(w.chat_template_kwargs),
                    "role": w.role,
                    "prompt_chars": len(w.prompt),
                }
                for w in self.workloads
            ],
            "missing_required_classes": self.missing_classes(),
            # Frozen before any measurement and reproduced in every artifact, so a reader
            # can check the shape rule the campaign was judged against without reading the
            # harness source.
            "class_shape_rule": dict(CLASS_SHAPE_RULE),
        }


def sha256_text(text: str) -> str:
    """Hash of the exact prompt bytes (UTF-8), which is what the server receives."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _require(cond: bool, message: str) -> None:
    if not cond:
        raise ManifestError(message)


def _validate_seed(entry: Mapping[str, Any], class_id: str) -> None:
    """FreeToken exposes no seed anywhere.

    ``SamplingParams`` (``python/freetoken/core.py``) carries temperature / top_k / top_p /
    ignore_eos / max_tokens / stop_strs and nothing else, and neither the OpenAI nor the
    Anthropic request model adds one. A manifest that pins a seed would therefore describe a
    determinism the runtime cannot deliver, so a non-null seed is rejected rather than
    silently ignored. Determinism comes from greedy decoding instead (criteria section 5.3).
    """
    seed = entry.get("seed", None)
    _require(
        seed is None,
        f"{class_id}: seed={seed!r} but FreeToken's SamplingParams exposes no seed "
        "parameter (python/freetoken/core.py); use greedy sampling for a reproducible "
        "fixture and leave seed null",
    )


def _validate_sampling(entry: Mapping[str, Any], class_id: str) -> tuple[Dict[str, Any], bool]:
    sampling = entry.get("sampling")
    _require(isinstance(sampling, dict), f"{class_id}: 'sampling' must be an object")
    missing = [k for k in ("temperature", "top_p", "top_k") if k not in sampling]
    _require(
        not missing,
        f"{class_id}: sampling is missing {missing}; every sampling field must be stated "
        "explicitly so the server's own defaults can never change a measurement",
    )
    unknown = set(sampling) - {"temperature", "top_p", "top_k"}
    _require(not unknown, f"{class_id}: unknown sampling field(s) {sorted(unknown)}")
    temperature = float(sampling["temperature"])
    top_p = float(sampling["top_p"])
    top_k = int(sampling["top_k"])
    # Mirrors SamplingParams.is_greedy (python/freetoken/core.py).
    greedy = (temperature <= 0.0 or top_k == 1) and top_p == 1.0
    return {"temperature": temperature, "top_p": top_p, "top_k": top_k}, greedy


def _resolve_prompt(entry: Mapping[str, Any], base_dir: Path, class_id: str) -> tuple[str, str | None]:
    has_path = "fixture_path" in entry and entry["fixture_path"] is not None
    has_content = "content" in entry and entry["content"] is not None
    _require(
        has_path != has_content,
        f"{class_id}: give exactly one of 'fixture_path' or 'content'",
    )
    if has_content:
        content = entry["content"]
        _require(isinstance(content, str), f"{class_id}: 'content' must be a string")
        return content, None
    rel = str(entry["fixture_path"])
    _require(not Path(rel).is_absolute(), f"{class_id}: fixture_path must be relative to the manifest")
    path = (base_dir / rel).resolve()
    _require(path.is_file(), f"{class_id}: fixture {rel} not found next to the manifest")
    # Read as bytes and decode explicitly: the sha256 pins the bytes that are sent.
    return path.read_bytes().decode("utf-8"), rel


def load_manifest(path: str | Path, *, canonical: bool) -> Manifest:
    """Parse and fully validate a workload manifest.

    ``canonical=True`` applies the criteria's frozen rules (all four classes present, the
    declared output-token counts, ``ignore_eos``). ``canonical=False`` is the developer
    smoke-test mode: the same structural validation, but the class-completeness and
    protocol rules are relaxed and the manifest is flagged non-canonical everywhere it is
    recorded.
    """
    path = Path(path)
    _require(path.is_file(), f"workload manifest not found: {path}")
    raw_bytes = path.read_bytes()
    try:
        doc = json.loads(raw_bytes.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ManifestError(f"{path}: not valid UTF-8 JSON ({e})") from e
    _require(isinstance(doc, dict), f"{path}: top level must be an object")
    _require(
        doc.get("schema") == SCHEMA,
        f"{path}: schema must be {SCHEMA!r}, got {doc.get('schema')!r}",
    )
    manifest_id = doc.get("manifest_id")
    _require(isinstance(manifest_id, str) and manifest_id, f"{path}: 'manifest_id' is required")
    declared_canonical = bool(doc.get("canonical", False))
    _require(
        not (canonical and not declared_canonical),
        f"{path}: manifest declares canonical=false but the run asked for the canonical "
        "protocol; a smoke-test manifest may not produce a canonical baseline",
    )

    entries = doc.get("workloads")
    _require(isinstance(entries, list) and entries, f"{path}: 'workloads' must be a non-empty list")

    workloads: List[Workload] = []
    seen: set[str] = set()
    for entry in entries:
        _require(isinstance(entry, dict), f"{path}: each workload must be an object")
        class_id = entry.get("class_id")
        _require(
            class_id in CLASS_SPECS,
            f"{path}: class_id {class_id!r} is not one of {sorted(CLASS_SPECS)}",
        )
        _require(class_id not in seen, f"{path}: duplicate class_id {class_id}")
        seen.add(class_id)
        spec = CLASS_SPECS[class_id]

        prompt, fixture_path = _resolve_prompt(entry, path.parent, class_id)
        declared_hash = entry.get("content_sha256")
        _require(
            isinstance(declared_hash, str) and declared_hash,
            f"{class_id}: 'content_sha256' is required (it is what freezes the prompt)",
        )
        actual = sha256_text(prompt)
        _require(
            actual == declared_hash.lower(),
            f"{class_id}: content_sha256 mismatch -- manifest says {declared_hash}, "
            f"the fixture hashes to {actual}. The prompt changed; freeze it again "
            "deliberately (criteria section 9 rule 3) rather than updating the hash to match.",
        )

        output_tokens = entry.get("output_tokens")
        _require(isinstance(output_tokens, int) and output_tokens > 0,
                 f"{class_id}: 'output_tokens' must be a positive integer")
        if canonical:
            _require(
                output_tokens == spec.output_tokens,
                f"{class_id}: canonical output_tokens is {spec.output_tokens} "
                f"(criteria section 9), manifest says {output_tokens}",
            )

        ignore_eos = entry.get("ignore_eos", True)
        _require(isinstance(ignore_eos, bool), f"{class_id}: 'ignore_eos' must be a boolean")
        if canonical:
            _require(
                ignore_eos,
                f"{class_id}: canonical runs need ignore_eos=true so the output length is "
                "exact and identical across arms (criteria section 3 rule 5)",
            )

        _validate_seed(entry, class_id)
        sampling, greedy = _validate_sampling(entry, class_id)

        ctk = entry.get("chat_template_kwargs", {})
        _require(isinstance(ctk, dict), f"{class_id}: 'chat_template_kwargs' must be an object")
        role = entry.get("role", "user")
        _require(role in ("user", "system"), f"{class_id}: 'role' must be 'user' or 'system'")

        workloads.append(
            Workload(
                class_id=class_id,
                prompt=prompt,
                content_sha256=actual,
                output_tokens=output_tokens,
                sampling=sampling,
                ignore_eos=ignore_eos,
                greedy=greedy,
                chat_template_kwargs=ctk,
                role=role,
                fixture_path=fixture_path,
                description=str(entry.get("description", "")),
            )
        )

    manifest = Manifest(
        schema=SCHEMA,
        manifest_id=manifest_id,
        canonical=declared_canonical,
        workloads=workloads,
        source_path=str(path),
        frozen_at=doc.get("frozen_at"),
        notes=str(doc.get("notes", "")),
        manifest_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        provenance=dict(doc.get("provenance", {})),
    )
    if canonical:
        missing = manifest.missing_classes()
        _require(
            not missing,
            f"{path}: canonical runs need all four workload classes; missing {missing}. "
            "InferSwarm issue #3 supplies the W1/W3/W4 fixtures -- do not substitute "
            "invented content to fill the gap.",
        )
    return manifest


def check_prompt_tokens(class_id: str, prompt_tokens: int) -> str | None:
    """Compare a server-reported prompt length against the class's frozen shape.

    Returns None when it fits, else a human-readable deviation string.

    The observation is always preserved in the repetition record, and the prompt is **never**
    rewritten or truncated to make it fit -- the tokenizer decides the real count, and
    silently reshaping a frozen fixture would break the very thing section 9 rule 3 freezes.
    But for a *canonical* run the shape is part of the experimental contract, so a deviation
    is not merely recorded: the runner turns it into a campaign invalidation
    (``validity.PROMPT_SHAPE_VIOLATION``). See ``CLASS_SHAPE_RULE`` for the exact rule.
    """
    spec = CLASS_SPECS.get(class_id)
    if spec is None:
        return f"unknown workload class {class_id}"
    if prompt_tokens > spec.prompt_max_tokens:
        return (
            f"prompt_tokens={prompt_tokens} exceeds the {class_id} bound "
            f"{spec.prompt_max_tokens} (criteria section 9)"
        )
    if spec.prompt_target_tokens is not None:
        lo = spec.prompt_target_tokens * (1 - spec.prompt_tolerance)
        hi = spec.prompt_target_tokens * (1 + spec.prompt_tolerance)
        if not lo <= prompt_tokens <= hi:
            return (
                f"prompt_tokens={prompt_tokens} is outside the {class_id} target band "
                f"[{lo:.0f}, {hi:.0f}] around {spec.prompt_target_tokens} "
                f"(criteria section 9 '~{spec.prompt_target_tokens}', tolerance frozen by "
                f"this harness at +/-{spec.prompt_tolerance:.0%})"
            )
    return None


def check_completion_tokens(
    class_id: str, completion_tokens: Any, requested_max_tokens: Any
) -> str | None:
    """Whether the generation produced exactly the frozen output length.

    Canonical runs pass ``ignore_eos=true`` (criteria section 3 rule 5) precisely so the
    output length is exact and identical across arms. A completion that is shorter or longer
    than the request therefore did not run as declared -- a truncation, a server-side cap, or
    an ignored ``ignore_eos`` -- and the block is not a valid observation of the frozen
    workload even though the tokens it produced were really produced.
    """
    if completion_tokens is None or requested_max_tokens is None:
        return (
            f"{class_id}: completion_tokens={completion_tokens!r} / "
            f"requested max_tokens={requested_max_tokens!r}; the output length cannot be checked"
        )
    if int(completion_tokens) != int(requested_max_tokens):
        return (
            f"{class_id}: completion_tokens={completion_tokens} != requested max_tokens="
            f"{requested_max_tokens}; canonical runs use ignore_eos=true so the length is "
            "exact (criteria section 3 rule 5)"
        )
    return None
