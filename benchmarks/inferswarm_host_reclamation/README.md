# Pre-R3 host reclamation evidence

This research namespace distinguishes attached required source state, optional retained
materialization, and physically reclaimed capacity. It does not define the R3 planner or
a public resource schema.

`allocation_primitives` runs each bounded primitive in an isolated process. The full
runner uses the frozen R2 `[0,19) / [19,40)` split and accepted v2 reference for W2/W4.
Neither runner drops global page cache, uses swap as reclamation, or kills a worker to
claim RELEASE success.
