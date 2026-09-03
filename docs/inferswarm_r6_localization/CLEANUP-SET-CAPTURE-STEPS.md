# Post-#71 harness cleanup

This cleanup is intentionally separate from the accepted #71 localization evidence.

The merged localization harness carried an unused diagnostic operation, `SET_CAPTURE_STEPS`, whose handler assigned a list payload to the scalar runtime `_capture_step` field. The physical #71 campaign never invoked this operation; all retained captures used the per-request integer `capture_step` field on `PREFILL` requests.

The cleanup removes only that dead malformed handler. It does not change any accepted localization result, capture format, physical producer, historical R6 evidence, comparator, or runtime semantics used by the campaign.
