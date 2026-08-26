# Example manifests — NOT Phase-0 workloads

Everything in this directory is a **developer smoke-test fixture**. The prompts are short,
synthetic, and deliberately meaningless: they exist to exercise the harness's plumbing
without a model, and they are declared `"canonical": false` so a canonical run refuses
them.

The real W1/W3/W4 fixtures are supplied by
[InferSwarm issue #3](https://github.com/Zutfen-LLC/inferswarm/issues/3) (routing-trace
experiment) and W2 by the criteria's own reasoning workload. Substituting invented content
here to make the set look complete is exactly the cherry-picking
[criteria](https://github.com/Zutfen-LLC/inferswarm/blob/main/docs/phase1-poc-success-criteria.md)
section 9 rule 7 prohibits.

To freeze a real fixture into a manifest:

```bash
python benchmarks/phase0_baseline.py hash path/to/fixture.txt
```

and paste the digest into `content_sha256`.
