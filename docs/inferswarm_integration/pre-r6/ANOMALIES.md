# Development-run anomalies

Several exploratory external-event timing attempts delivered the resource
event before token serving began. They produced valid zero-token cutovers but
did not exercise the requested committed-boundary transition and are not used
as passing evidence.

The retained passing epoch evidence instead uses the existing, explicit
`after_commit` controller research seam to make the boundary deterministic.
No accepted historical evidence was altered by these attempts.

