# InferSwarm Pre-R6 integration evidence

Disposition: `PRE_R6_INTEGRATION_REQUALIFICATION_PASS`

This directory records the bounded refresh required by
Zutfen-LLC/inferswarm#64. It is an integration requalification, not a new R5B
performance campaign and not the beginning of R6 implementation.

## Frozen lineage

- accepted R5B research head:
  `00ccd01fede8d2ad21ee83104f3b998c89ff9d1f`;
- accepted R5B physical producer:
  `7dd945a67c04198ec2d9afe782a39c90e8141f5e`;
- accepted R5B evidence head:
  `b6f674b5bf0f76b9b40bd2f79e36cfc18cb6f7e6`;
- accepted R5B disposition: `R5B_PLAN_EPOCH_RECOVERY_PASS`;
- frozen FreeToken `main` target:
  `6eca2d7d2b8576c7ad0ba62853df9f618cba929f`;
- merge base: `a05c26543f2d9a8cc2168fe789cdd4c92273378e`;
- explicit two-parent integration merge and final qualification producer:
  `0f44f86f91db3c7a82f6d380c074a3731535f3a9`.

The integration merge has parents `00ccd01...` and `6eca2d7...`; the accepted
research lineage was neither rebased nor squashed.

## Qualification result

Fresh merged-tree native builds produced `_pinned_tensor`, `_cpu_moe`, and the
new `_ple_store` extension on both physical participants. Research tests passed
199/199, InferSwarm benchmark tests passed 563/563, and server tests passed
583/583. The broad applicable upstream run passed 1,214, skipped 31,
deselected 58, and reported six failures. Five failures reproduced exactly on a
clean `main@6eca2d7...` build; the sixth (`swiglu_clamp`) passed alone on both
trees and was collateral from the preceding pinned-memory CUDA failure. No
integration regression remained.

The fresh two-node preflight passed with environment digest
`sha256:13f6956ad9ec948c4b74094958f18082c41e788d74e7a47febe26432ed59f24c`
and participant-plan digest
`sha256:75e01df4f78ac0ee92cb9e520a947253abbe4d69706b491d21cc59434e2c02ce`.
Both nodes used the exact clean qualification producer and the same frozen
Qwen3.6 NVFP4 checkpoint identity.

An ordinary `/v1/chat/completions` request traversed the accepted OpenAI
adapter, `GenSpec`, `TokenizeMsg`, epoch-aware controller, frozen plan,
realization, and backend-native resident execution. It emitted the expected
two-token prefix `[9764, 393]` (`"Let $"`) with exact comparator agreement,
zero fallbacks, zero unauthorized host expert/source fetches, zero unexplained
persistent host-mirror bytes, zero steady-state model-state movement, matched
realization, no unplanned persistent materialization, and released transient
staging with zero current staging bytes at the lifecycle boundary.

The focused two-epoch smoke used the accepted controller's existing
`after_commit` research seam. It transitioned from the two-node resident plan
to the same-node resident plan, preserved immutable generations, replayed the
trusted committed range `[0, 1]`, changed mutable authority unambiguously,
rejected one injected result from the retired epoch as
`RETIRED_OR_SUPERSEDED_EPOCH`, emitted `[9764, 393]` exactly, and reclaimed
both epochs. Postflight found no GPU execution processes on either node.

The earlier accepted R0-R5B and Pre-R5 evidence directories were not modified
or regenerated. Their performance evidence remains historical and
context-specific.

See `SUMMARY.json` for the machine-readable gate result, `CONFLICTS.md` for
semantic resolutions, `TESTS.md` for regression classification,
`NATIVE_REBUILD.md` for build provenance, and `MANIFEST.sha256` for retained
artifact hashes.
