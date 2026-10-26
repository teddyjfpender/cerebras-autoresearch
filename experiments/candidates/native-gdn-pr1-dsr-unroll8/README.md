# Persistent DSR walk unroll8

The three K-row loops execute exactly128 FP32 FMAs apiece, in unchanged
ascending order. Dedicated DSRs advance rows automatically; fixed accumulator
and delta addresses reset after every vector instruction. Only loop overhead
is reduced. All full states, timing, precision and original bounds remain.
