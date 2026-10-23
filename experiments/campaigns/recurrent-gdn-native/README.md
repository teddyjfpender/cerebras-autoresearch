# Native-width recurrent head-group challenge

The registered T4/48PE and T8/96PE profiles use native K128/V128 dimensions for
one key-head group and three value heads. All native-width states and outputs
are reconstructed, every physical immutable/prefix replica is checked, and all
normalization/decay/state operations and full snapshots remain timed.

The reference already uses the freshly confirmed FP32 vector-state algorithm
from the scaled operator. Its native-width/rectangular adaptation was frozen
before device observations. Qualify T4 first, then T8; registration and upstream
CPU correspondence alone do not qualify device programs.

Initial paired seed20261038; fresh confirmation seed20261039. Every terminal V2
attempt and decision is automatically journaled in results/results.tsv. Preserve
source/profile/harness/SDK identities and full raw evidence. This is a model-shaped
operator after QKV/convolution and precomputed g/beta, before gated normalization
and output projection, with one of the model's key-head groups. No model TPS or
production-runtime comparison follows. Each profile fixes its own PE budget.
