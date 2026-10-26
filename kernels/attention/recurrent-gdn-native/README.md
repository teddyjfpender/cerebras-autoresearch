# Native-width vector recurrent head group

Independent FP32 vector reference for one key head and three value heads, native
K128/V128, column-sharded across a fixed48 or96 PE grid. Every full state and
intermediate is stored inside the timer. BF16 Q/K/V/output and FP32 recurrent
state/normalization/arithmetic match the separate native operator contract.
No device qualification yet. Source has the corrected bounded-domain FP32 exp;
its fixed decay bound is unchanged from the scaled oracle.
