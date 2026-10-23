# Distributed attention campaign

Read the [contract](../../../docs/benchmarks/decode-attention.md). Both staged
candidate workspaces are uncompiled proposals; no >5x result exists.

`incremental.json` requires 1.05x and `target-5x.json` requires 5.01x against
the same frozen vectorized baseline. Both use six-case numerical checks, fresh
paired confirmation and <=10% memory growth. Screens cannot qualify. Source
changes require new baselines and state files. Report only measured cells;
model TPS, hardware performance and energy remain unmeasured.
