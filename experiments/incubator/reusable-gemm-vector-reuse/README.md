# reusable-gemm-vector-reuse

Unmeasured incubator candidate; not registered for promotion.

Preserve canonical W0 across complete P-step cycles and omit buffers overwritten before first read. Also use contiguous column-vector accumulation with a final ABI transpose.

Reject any output, per-frame working-weight or canonical-input bit failure, less than 1.05x cycle speedup, more than 1.10x memory growth or failed fresh paired confirmation.

Protected sequence, snapshots, exits and repetition functions are unchanged.
Source is derived from the sibling reusable-gemm baseline; see its attribution
and Apache license. Canonical W0 restoration after each P-step cycle remains
a hypothesis to be checked by actual SDK state snapshots.
