# Full native recurrent head configuration

Independent FP32 vector reference for16 key heads,48 value heads, K128/V128,
a fixed two-token decode pair and768 PEs (32x24). Each PE owns all128 K entries
and8 value columns of one value head. Complete canonical copies, normalized
prefix replicas, every full state/output and final live state are checked.

The source is the native one-group vector reference specialized to T2, with new
profile/layout/oracle contracts. All normalization/decay/state work and all full
snapshots remain timed. No device qualification or production-runtime result yet.
Inputs are synthetic projected/conv Q/K/V and precomputed g/beta. Surrounding
projections, convolution, gated normalization and output projection are excluded.
